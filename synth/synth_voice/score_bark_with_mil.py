"""Score Bark smoke-test clips with the trained Whisper-MIL hardneg model.

Goal: confirm that the model classifier treats Bark output as "child positive".
F0 sanity passed (Bark mean ~241-282 Hz vs real refs ~195 Hz). The real test
is whether the Whisper backbone + MIL head treats these clips as positives —
the only signal that determines whether augmentation will help.

Decision rule:
  - >= 60% of Bark clips scored above 0.5 (the tuned threshold) → class-relevant,
    proceed to scaled generation + MIL augmentation experiment.
  - < 60% → augmentation likely useless; pivot or stop.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd
import torch
import torchaudio

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from mil.mil_model import build_mil_model


def load_audio_windows(path: str, window_sec: float, stride_sec: float, sr: int = 16000):
    wav, native_sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if native_sr != sr:
        wav = torchaudio.functional.resample(wav, native_sr, sr)
    win = int(window_sec * sr)
    stride = int(stride_sec * sr)
    if wav.shape[1] < win:
        pad = win - wav.shape[1]
        wav = torch.nn.functional.pad(wav, (0, pad))
        return [wav]
    wins = []
    s = 0
    while s + win <= wav.shape[1]:
        wins.append(wav[:, s:s + win])
        s += stride
    return wins


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bark-dir", default="synth/synth_voice/spec019_bark_smoke")
    p.add_argument("--checkpoint", default="mil/mil_results/whisper_mil_hardneg_synth/best_checkpoint.pt")
    p.add_argument("--config", default="mil/mil_results/whisper_mil_hardneg_synth/config.json")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    rows = []
    for sub in ("unconditioned", "preset_speaker9"):
        for w in sorted(glob.glob(os.path.join(args.bark_dir, sub, "*.wav"))):
            rows.append({
                "audio_path": os.path.abspath(w),
                "subset": sub,
                "prompt": os.path.basename(w).rsplit("_", 1)[0],
            })
    df = pd.DataFrame(rows)
    print(f"Scoring {len(df)} Bark clips...", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_mil_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    scores = []
    for _, row in df.iterrows():
        wins = load_audio_windows(row["audio_path"],
                                  cfg["window_sec"], cfg["stride_sec"])
        score, _ = model.predict_bag(wins)
        scores.append(score)
    df["score"] = scores
    df["pred"] = (df["score"] >= args.threshold).astype(int)

    print()
    print(f"=== Whisper-MIL hardneg on Bark clips (threshold={args.threshold}) ===")
    for sub, sub_df in df.groupby("subset"):
        n_pos = int(sub_df["pred"].sum())
        print(f"\n  {sub}:  predicted child = {n_pos}/{len(sub_df)}  "
              f"(mean score={sub_df['score'].mean():.3f}, "
              f"median={sub_df['score'].median():.3f})")
        for _, r in sub_df.sort_values("score", ascending=False).iterrows():
            tag = "POS" if r["pred"] == 1 else "neg"
            print(f"    [{tag}]  {r['prompt']:30s}  score={r['score']:.3f}")

    out = os.path.join(args.bark_dir, "mil_scores.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    overall_pos = int(df["pred"].sum())
    print(f"\nOVERALL: {overall_pos}/{len(df)} ({100*overall_pos/len(df):.0f}%) Bark clips classified as child by Whisper-MIL.")
    print(f"Mean score: {df['score'].mean():.3f} (real test-set positive mean ~0.7-0.9)")


if __name__ == "__main__":
    main()
