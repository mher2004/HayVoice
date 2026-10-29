"""
Speaker counting on VoxConverse: (optional Demucs) -> Silero VAD -> ECAPA embeddings
-> agglomerative clustering -> count. Compares against "always 3" and (optionally) pyannote.

Colab setup:
    !pip install -q datasets soundfile torchaudio speechbrain silero-vad scikit-learn pandas demucs pyannote.audio
    import os; os.environ["HF_TOKEN"] = "hf_..."   # only needed for --pyannote

Usage:
    python count_speakers.py --split dev --n 5                  # quick test
    python count_speakers.py --split dev --n 30 --pyannote      # full baseline run
    python count_speakers.py --split dev --n 30 --demucs        # with music/effects removal

Embeddings are cached, so re-running with different --thresholds is fast.
"""
import argparse, io, os, subprocess, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_dataset
from sklearn.cluster import AgglomerativeClustering

SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REQUIRED = {"audio", "timestamps_start", "timestamps_end", "speakers"}


# ---------- audio helpers ----------
def to_16k_mono(wav: np.ndarray, sr: int) -> torch.Tensor:
    wav = torch.from_numpy(wav).float()
    if wav.ndim == 2:                      # soundfile returns (frames, channels)
        wav = wav.mean(dim=1)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav


def run_demucs(wav: torch.Tensor) -> torch.Tensor:
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.wav"
        sf.write(inp, wav.numpy(), SR)
        subprocess.run(["demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", tmp, str(inp)],
                       check=True, capture_output=True)
        v, sr = sf.read(Path(tmp) / "htdemucs" / "in" / "vocals.wav")
    return to_16k_mono(v, sr)


# ---------- ground truth ----------
def true_count(row, min_time):
    dur = {}
    for s, e, spk in zip(row["timestamps_start"], row["timestamps_end"], row["speakers"]):
        dur[spk] = dur.get(spk, 0.0) + (e - s)
    return sum(d >= min_time for d in dur.values()), len(dur)


# ---------- pipeline stages ----------
def speech_windows(wav, vad, get_ts, win, step):
    speech = get_ts(wav, vad, sampling_rate=SR, return_seconds=True)
    spans = []
    for seg in speech:                     # segments shorter than `win` are skipped
        t = seg["start"]
        while t + win <= seg["end"]:
            spans.append(t)
            t += step
    return np.array(spans)


@torch.no_grad()
def embed(enc, wav, starts, win, bs=64):
    n = int(win * SR)
    out = []
    for i in range(0, len(starts), bs):
        batch = torch.stack([wav[int(s * SR): int(s * SR) + n] for s in starts[i:i + bs]]).to(DEVICE)
        emb = enc.encode_batch(batch).squeeze(1)
        out.append(torch.nn.functional.normalize(emb, dim=1).cpu())
    return torch.cat(out).numpy()


def count_from_embeddings(embs, thr, step, min_time):
    if len(embs) == 0:
        return 0
    if len(embs) == 1:
        return 1 if step >= min_time else 0
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=thr,
                                     metric="cosine", linkage="average").fit_predict(embs)
    times = np.bincount(labels) * step     # each window "owns" `step` seconds
    return int((times >= min_time).sum())


def pyannote_count(pipe, wav, min_time):
    d = pipe({"waveform": wav.unsqueeze(0), "sample_rate": SR})
    d = getattr(d, "speaker_diarization", d)   # newer pyannote wraps the output
    return sum(d.label_duration(l) >= min_time for l in d.labels())


# ---------- evaluation ----------
def summarize(df, cols):
    buckets = {"1-3": (1, 3), "4-6": (4, 6), "7+": (7, 999)}
    rows = []
    for c in cols:
        r = {"method": c,
             "exact_acc": (df[c] == df["true"]).mean(),
             "MAE": (df[c] - df["true"]).abs().mean()}
        for name, (lo, hi) in buckets.items():
            sub = df[(df["true"] >= lo) & (df["true"] <= hi)]
            r[f"MAE_{name}"] = (sub[c] - sub["true"]).abs().mean() if len(sub) else np.nan
        rows.append(r)
    return pd.DataFrame(rows).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="diarizers-community/voxconverse")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=20, help="number of files to process")
    ap.add_argument("--min_time", type=float, default=5.0, help="min total speaking seconds to count a speaker")
    ap.add_argument("--win", type=float, default=1.5)
    ap.add_argument("--step", type=float, default=0.75)
    ap.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--demucs", action="store_true")
    ap.add_argument("--pyannote", action="store_true")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    cache_dir = Path(args.cache) / f"{args.split}_{'demucs' if args.demucs else 'raw'}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out).mkdir(exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))   # decode ourselves: works across `datasets` versions
    first = next(iter(ds))
    if not REQUIRED.issubset(first.keys()):
        raise SystemExit(f"Unexpected columns: {list(first.keys())}. Edit REQUIRED / true_count() to match.")

    from silero_vad import get_speech_timestamps, load_silero_vad
    from speechbrain.inference.speaker import EncoderClassifier
    vad = load_silero_vad()
    enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                         run_opts={"device": DEVICE})
    pipe = None
    if args.pyannote:
        from pyannote.audio import Pipeline
        tok = os.environ.get("HF_TOKEN")
        try:
            pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=tok)
        except TypeError:
            pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=tok)
        pipe.to(torch.device(DEVICE))

    records = []
    for idx, row in enumerate(ds.take(args.n)):
        fid = f"{idx:04d}_{Path(row['audio'].get('path') or 'file').stem}"
        cfile = cache_dir / f"{fid}.npz"
        t_count, t_all = true_count(row, args.min_time)

        d = np.load(cfile) if cfile.exists() else None
        need_pya = args.pyannote and (d is None or int(d["pya"]) < 0)

        if d is None or need_pya:
            wav_np, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
            wav = to_16k_mono(wav_np, sr)
            pya = pyannote_count(pipe, wav, args.min_time) if args.pyannote else -1
            if d is None:
                clean = run_demucs(wav) if args.demucs else wav
                starts = speech_windows(clean, vad, get_speech_timestamps, args.win, args.step)
                embs = embed(enc, clean, starts, args.win) if len(starts) else np.zeros((0, 192))
            else:
                embs, pya = d["embs"], pya
            np.savez(cfile, embs=embs, pya=pya)
        else:
            embs, pya = d["embs"], int(d["pya"])

        rec = {"file": fid, "true": t_count, "true_all_speakers": t_all, "always_3": 3}
        if args.pyannote:
            rec["pyannote"] = pya
        for thr in thresholds:
            rec[f"ecapa_thr{thr}"] = count_from_embeddings(embs, thr, args.step, args.min_time)
        records.append(rec)
        print(f"[{idx + 1}/{args.n}] {fid}: true={t_count}  " +
              "  ".join(f"{k}={v}" for k, v in rec.items() if k.startswith(("ecapa", "pyannote"))))

    df = pd.DataFrame(records)
    df.to_csv(Path(args.out) / f"per_file_{args.split}.csv", index=False)
    methods = [c for c in df.columns if c.startswith(("always", "ecapa", "pyannote"))]
    summary = summarize(df, methods)
    summary.to_csv(Path(args.out) / f"summary_{args.split}.csv", index=False)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
