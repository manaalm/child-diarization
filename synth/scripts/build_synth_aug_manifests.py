"""Build synth-derived training manifests for downstream augmentation experiments.

Outputs (under synth_results/manifests/):
  synthetic_hardneg.csv         — C3: hard negatives for MIL hardneg pipeline
  synthetic_cross_child_aug.csv — C4: full synth pool reformatted for MIL cross-child
  synthetic_audio_llm_shots.csv — C6: 1 positive + 1 negative for Audio LLM 2-shot
  synthetic_train_aug.csv       — generic seen-child augmentation (used by C2/C5 too)

All CSVs match the schemas the existing pipelines already accept; no code changes
to consumers required.
"""
import argparse
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(REPO, "synth_results/manifests/synthetic_manifest.csv"))
    ap.add_argument("--output-dir", default=os.path.join(REPO, "synth_results/manifests"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"Loaded {len(df)} synth scenes from {args.manifest}", file=sys.stderr)
    print("Scene type distribution:", file=sys.stderr)
    print(df["scene_type"].value_counts().to_string(), file=sys.stderr)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- C3: hard negatives ----------
    hardneg_mask = df["scene_type"].isin(["adult_only_negative", "background_speech_negative"])
    hardneg = df[hardneg_mask].copy()
    hardneg_out = pd.DataFrame({
        "audio_path": hardneg["audio_path"],
        "start_sec": 0.0,
        "end_sec": 30.0,
        "label": 0,
        "child_id": "synth_hardn_" + hardneg["age_band"].astype(str),
        "timepoint_norm": "synthetic_" + hardneg["age_band"].astype(str),
        "source": "synthetic_" + hardneg["scene_type"].astype(str),
    })
    hardneg_path = os.path.join(args.output_dir, "synthetic_hardneg.csv")
    hardneg_out.to_csv(hardneg_path, index=False)
    print(f"Wrote {len(hardneg_out)} hard negatives → {hardneg_path}", file=sys.stderr)

    # ---------- C4: cross-child augmentation pool ----------
    cross_aug = pd.DataFrame({
        "audio_path": df["audio_path"],
        "child_id": "synth_" + df["age_band"].astype(str) + "_" + df["scene_type"].astype(str),
        "timepoint_norm": df["age_band"].astype(str),
        "label": df["target_child_vocalized"].astype(int),
        "split": "train",
        "scene_type": df["scene_type"],
        "source": "synthetic",
    })
    cross_path = os.path.join(args.output_dir, "synthetic_cross_child_aug.csv")
    cross_aug.to_csv(cross_path, index=False)
    print(f"Wrote {len(cross_aug)} cross-child aug rows → {cross_path}", file=sys.stderr)

    # ---------- C6: audio-LLM 2-shot demos ----------
    rng = pd.Series(range(len(df))).sample(n=len(df), random_state=args.seed).reset_index(drop=True)
    df_shuf = df.iloc[rng].reset_index(drop=True)
    pos = df_shuf[df_shuf["scene_type"] == "positive"].head(1)
    neg = df_shuf[df_shuf["scene_type"] == "adult_only_negative"].head(1)
    shots = pd.concat([pos, neg], ignore_index=True)
    # audio_llm_baseline.py extracts child_id from path via regex sub-([A-Za-z0-9]+).
    # Fake it by giving each shot a child_id that matches the regex shape from any test path.
    shots_out = pd.DataFrame({
        "child_id": "synthshot",  # singleton — selection always picks both
        "audio_path": shots["audio_path"],
        "label": shots["target_child_vocalized"].astype(int),
        "timepoint_norm": shots["age_band"].astype(str),
    })
    shots_path = os.path.join(args.output_dir, "synthetic_audio_llm_shots.csv")
    shots_out.to_csv(shots_path, index=False)
    print(f"Wrote {len(shots_out)} audio-LLM shot demos → {shots_path}", file=sys.stderr)
    print(shots_out.to_string(index=False), file=sys.stderr)

    # ---------- Generic seen-child train augmentation (C2/C5) ----------
    train_aug = pd.DataFrame({
        "audio_path": df["audio_path"],
        "child_id": "synth_" + df["age_band"].astype(str) + "_" + df["scene_type"].astype(str),
        "timepoint_norm": df["age_band"].astype(str),
        "label": df["target_child_vocalized"].astype(int),
        "split": "train",
        "scene_type": df["scene_type"],
        "source": "synthetic",
        "rttm_path": df["rttm_path"],
    })
    train_aug_path = os.path.join(args.output_dir, "synthetic_train_aug.csv")
    train_aug.to_csv(train_aug_path, index=False)
    print(f"Wrote {len(train_aug)} generic train-aug rows → {train_aug_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
