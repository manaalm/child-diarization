"""Post-v4-corpus orchestrator: build all v4 augmentation manifests.

Run AFTER ``synth/slurm/run_v4_pipeline.sh`` finishes. Mirrors
``build_v3_aug_manifests.py`` but for the v4 corpus
(``synth_results/synthetic_scenes_v4/``) which folds in WORLD/CLEESE/
cross-lingual VC sources via ``segment_manifest_v4.csv`` plus empirical
turn-taking.

Expects:
  - synth_results/synthetic_scenes_v4/wav/*.wav (5000 v4 scenes)
  - synth_results/manifests/synthetic_manifest.csv  (overwritten by v4 gen,
                                                     contains v4 rows)
  - synth_results/manifests/synthetic_manifest_v3.csv (v3 backup, preserved)

Steps:
  1. Snapshot the just-written canonical manifest as synthetic_manifest_v4.csv.
  2. Run build_synth_aug_manifests.py with --suffix _v4 to produce
     synthetic_{hardneg,cross_child_aug,train_aug,audio_llm_shots}_v4.csv.
  3. Build a v4 cross-child split dir (baselines/splits_synth_aug_v4/) by
     cloning baselines/splits/{train,val,test}.csv and appending synthetic
     v4 cross-child rows to train.csv only.
"""
import os
import shutil
import subprocess
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFESTS = os.path.join(REPO, "synth_results/manifests")
CANONICAL = os.path.join(MANIFESTS, "synthetic_manifest.csv")
V4_SNAPSHOT = os.path.join(MANIFESTS, "synthetic_manifest_v4.csv")


def _is_v4(manifest_path: str) -> bool:
    df = pd.read_csv(manifest_path, nrows=10)
    return df["audio_path"].astype(str).str.contains("synthetic_scenes_v4").any()


def step1_snapshot_v4() -> None:
    if not os.path.isfile(CANONICAL):
        sys.exit(f"ERROR: canonical manifest missing: {CANONICAL}")
    if not _is_v4(CANONICAL):
        sys.exit(
            f"ERROR: {CANONICAL} does not look like v4 (no synthetic_scenes_v4 path). "
            "Did the v4 scene-gen job finish? Check logs/adult/v4_pipeline_*.out."
        )
    shutil.copy(CANONICAL, V4_SNAPSHOT)
    print(f"[1] Snapshotted v4 manifest -> {V4_SNAPSHOT}", flush=True)


def step2_run_aug_builder() -> None:
    cmd = [
        sys.executable,
        os.path.join(REPO, "synth/scripts/build_synth_aug_manifests.py"),
        "--manifest", V4_SNAPSHOT,
        "--output-dir", MANIFESTS,
        "--suffix", "_v4",
    ]
    print(f"[2] Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def step3_build_cross_child_split_v4() -> None:
    src = os.path.join(REPO, "baselines/splits")
    dst = os.path.join(REPO, "baselines/splits_synth_aug_v4")
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(src, "val.csv"), os.path.join(dst, "val.csv"))
    shutil.copy(os.path.join(src, "test.csv"), os.path.join(dst, "test.csv"))

    real_train = pd.read_csv(os.path.join(src, "train.csv"))
    synth = pd.read_csv(os.path.join(MANIFESTS, "synthetic_cross_child_aug_v4.csv"))

    required = ["audio_path", "child_id", "timepoint_norm", "label", "split"]
    synth_norm = pd.DataFrame({
        "audio_path": synth["audio_path"],
        "child_id": synth["child_id"],
        "timepoint_norm": synth["timepoint_norm"],
        "label": synth["label"].astype(int),
        "split": "train",
    })
    real_norm = real_train[[c for c in required if c in real_train.columns]].copy()
    combined = pd.concat([real_norm, synth_norm], ignore_index=True)
    combined.to_csv(os.path.join(dst, "train.csv"), index=False)

    n_pos = int((synth_norm["label"] == 1).sum())
    n_neg = int((synth_norm["label"] == 0).sum())
    print(
        f"[3] Wrote {dst}/train.csv ({len(combined)} rows; "
        f"real={len(real_norm)} + synth={len(synth_norm)} pos={n_pos} neg={n_neg})",
        flush=True,
    )


def main() -> None:
    step1_snapshot_v4()
    step2_run_aug_builder()
    step3_build_cross_child_split_v4()
    print("\nv4 manifests ready. Next: submit MIL with whisper_mil_{hardneg,cross_child}_synth_v4.yaml", flush=True)


if __name__ == "__main__":
    main()
