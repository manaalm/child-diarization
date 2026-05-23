"""Post-v3-corpus orchestrator: build all v3 augmentation manifests.

Run AFTER `synth/slurm/run_scene_generation_v3.sh` finishes. It expects:
  - synth_results/synthetic_scenes_v3_perturb/wav/*.wav (5000 v3 scenes)
  - synth_results/manifests/synthetic_manifest.csv  (just-overwritten by v3 gen,
                                                     contains v3 rows)
  - synth_results/manifests/synthetic_manifest_v2.csv (v2 backup, preserved)

Steps:
  1. Snapshot the just-written canonical manifest as synthetic_manifest_v3.csv
     (so it survives any future v4 run that overwrites the canonical name).
  2. Run build_synth_aug_manifests.py with --suffix _v3 reading from the v3
     manifest, producing synthetic_{hardneg,cross_child_aug,train_aug,
     audio_llm_shots}_v3.csv.
  3. Build a v3 cross-child split dir (baselines/splits_synth_aug_v3/) by
     cloning baselines/splits/{train,val,test}.csv and appending synthetic
     v3 cross-child rows to train.csv only.

Idempotent: re-runnable; later runs overwrite their own outputs.
"""
import os
import shutil
import subprocess
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFESTS = os.path.join(REPO, "synth_results/manifests")
CANONICAL = os.path.join(MANIFESTS, "synthetic_manifest.csv")
V3_SNAPSHOT = os.path.join(MANIFESTS, "synthetic_manifest_v3.csv")
V2_SNAPSHOT = os.path.join(MANIFESTS, "synthetic_manifest_v2.csv")


def _is_v3(manifest_path: str) -> bool:
    """Confirm the canonical manifest looks like v3 (audio_path under v3 dir)."""
    df = pd.read_csv(manifest_path, nrows=10)
    return df["audio_path"].astype(str).str.contains("synthetic_scenes_v3_perturb").any()


def step1_snapshot_v3() -> None:
    if not os.path.isfile(CANONICAL):
        sys.exit(f"ERROR: canonical manifest missing: {CANONICAL}")
    if not _is_v3(CANONICAL):
        sys.exit(
            f"ERROR: {CANONICAL} does not look like v3 (no synthetic_scenes_v3_perturb path). "
            "Did the v3 scene-gen job finish? Check logs/synth/scene_gen_v3_*.out."
        )
    shutil.copy(CANONICAL, V3_SNAPSHOT)
    print(f"[1] Snapshotted v3 manifest → {V3_SNAPSHOT}", flush=True)


def step2_run_aug_builder() -> None:
    cmd = [
        sys.executable,
        os.path.join(REPO, "synth/scripts/build_synth_aug_manifests.py"),
        "--manifest", V3_SNAPSHOT,
        "--output-dir", MANIFESTS,
        "--suffix", "_v3",
    ]
    print(f"[2] Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def step3_build_cross_child_split_v3() -> None:
    src = os.path.join(REPO, "baselines/splits")
    dst = os.path.join(REPO, "baselines/splits_synth_aug_v3")
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(src, "val.csv"), os.path.join(dst, "val.csv"))
    shutil.copy(os.path.join(src, "test.csv"), os.path.join(dst, "test.csv"))

    real_train = pd.read_csv(os.path.join(src, "train.csv"))
    synth = pd.read_csv(os.path.join(MANIFESTS, "synthetic_cross_child_aug_v3.csv"))

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
    step1_snapshot_v3()
    step2_run_aug_builder()
    step3_build_cross_child_split_v3()
    print("\nv3 manifests ready. Next: submit MIL sweep with the *_v3 configs.", flush=True)


if __name__ == "__main__":
    main()
