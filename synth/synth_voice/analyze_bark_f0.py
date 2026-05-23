"""F0 analysis for Bark smoke-test outputs vs real child references.

Decision rule:
  - Child 14-18mo F0 ~250-450 Hz mean voiced (literature: Kent & Murray 1982; Lee et al. 1999)
  - Adult F0: ~120 Hz male, ~220 Hz female mean voiced
  - Bark output mean F0 < 250 Hz on most prompts → label-corruption confirmed; cloning needed
  - Bark output mean F0 > 250 Hz → cheap path possible, scale up
"""
import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
import librosa


def f0_stats(wav_path: str) -> dict:
    """Mean voiced F0 + voicing fraction via librosa.pyin."""
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=70.0, fmax=600.0, sr=sr, frame_length=2048, hop_length=512
    )
    voiced_f0 = f0[voiced_flag & np.isfinite(f0)]
    if len(voiced_f0) < 5:
        return {"path": wav_path, "voiced_frames": int(len(voiced_f0)),
                "f0_mean": np.nan, "f0_median": np.nan, "f0_std": np.nan,
                "voicing_frac": float(voiced_flag.mean()), "duration_sec": float(len(y) / sr)}
    return {
        "path": wav_path,
        "voiced_frames": int(len(voiced_f0)),
        "f0_mean": float(np.mean(voiced_f0)),
        "f0_median": float(np.median(voiced_f0)),
        "f0_std": float(np.std(voiced_f0)),
        "voicing_frac": float(voiced_flag.mean()),
        "duration_sec": float(len(y) / sr),
    }


def summarize(rows: list, label: str) -> None:
    valid = [r for r in rows if not np.isnan(r["f0_mean"])]
    if not valid:
        print(f"\n=== {label} === (no voiced frames anywhere)")
        return
    means = [r["f0_mean"] for r in valid]
    medians = [r["f0_median"] for r in valid]
    voicing = [r["voicing_frac"] for r in valid]
    print(f"\n=== {label}  (n={len(valid)}/{len(rows)}) ===")
    print(f"  F0 mean:    avg {np.mean(means):.0f}   range [{min(means):.0f}, {max(means):.0f}] Hz")
    print(f"  F0 median:  avg {np.mean(medians):.0f}   range [{min(medians):.0f}, {max(medians):.0f}] Hz")
    print(f"  voicing fr: avg {np.mean(voicing):.2f}   range [{min(voicing):.2f}, {max(voicing):.2f}]")
    print(f"  per-file f0_mean:")
    for r in sorted(valid, key=lambda x: -x["f0_mean"]):
        name = os.path.basename(r["path"])
        print(f"    {name:38s}  f0={r['f0_mean']:5.0f}Hz  voic={r['voicing_frac']:.2f}  {r['duration_sec']:4.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bark-dir", default="synth/synth_voice/spec019_bark_smoke")
    p.add_argument("--ref-csv", default="synth_results/manifests/segment_manifest_v2.csv")
    p.add_argument("--n-refs", type=int, default=10)
    args = p.parse_args()

    print("Bark smoke test analysis")
    print("=" * 60)

    for sub in ("unconditioned", "preset_speaker9"):
        wavs = sorted(glob.glob(os.path.join(args.bark_dir, sub, "*.wav")))
        rows = [f0_stats(w) for w in wavs]
        summarize(rows, f"BARK {sub}")

    import pandas as pd
    df = pd.read_csv(args.ref_csv, low_memory=False)
    refs = df[
        (df["speaker_role"] == "target_child")
        & (df["age_band"] == "14_18_months")
        & (df["quality_score"] == 1.0)
        & (df["duration_sec"] >= 4.0)
        & (df["duration_sec"] <= 8.0)
        & (df["split"] == "train")
        & (df["source_dataset"] == "tinyvox")
    ].sample(args.n_refs, random_state=42)

    print(f"\nLoading {len(refs)} TinyVox 14-18mo reference clips for F0 grounding...")
    rows = []
    for _, row in refs.iterrows():
        # crop to start_time:end_time from raw audio
        y, sr = sf.read(row["audio_path"])
        if y.ndim > 1:
            y = y.mean(axis=1)
        s = int(row["start_time_sec"] * sr)
        e = int(row["end_time_sec"] * sr)
        y = y[s:e]
        # write tmp for f0_stats
        tmp = "/tmp/_ref_seg.wav"
        sf.write(tmp, y, sr)
        rows.append(f0_stats(tmp))
    summarize(rows, "REAL TinyVox child 14-18mo")


if __name__ == "__main__":
    main()
