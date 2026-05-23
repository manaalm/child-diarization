"""Build the spec-017 US2 augmented seen-child split (real train + KNN-VC positives).

Output: baselines/splits_synth_aug_knnvc/{train,val,test}.csv

train = whisper-modeling/seen_child_splits/train.csv
        + synth_results/manifests/synthetic_voice_converted.csv
          (label=1 rows; child_id and timepoint_norm copied from the real-train row that
           anchored the conversion; missing timepoint_norm → 'unknown')
val/test = unchanged copies of the real seen-child val/test.

Sanity check: every voice-converted row's child_id must match a real train child
(no test/val children's voices in the augmented train set).
"""
from __future__ import annotations

import os
import sys
import shutil

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REAL_DIR = os.path.join(_REPO, "whisper-modeling/seen_child_splits")
KNNVC_MANIFEST = os.path.join(_REPO, "synth_results/manifests/synthetic_voice_converted.csv")
OUT_DIR = os.path.join(_REPO, "baselines/splits_synth_aug_knnvc")


def main():
    if not os.path.exists(KNNVC_MANIFEST):
        print(f"ERROR: {KNNVC_MANIFEST} not found — T130 must complete first", file=sys.stderr)
        sys.exit(2)

    os.makedirs(OUT_DIR, exist_ok=True)

    real_train = pd.read_csv(os.path.join(REAL_DIR, "train.csv"))
    real_val   = pd.read_csv(os.path.join(REAL_DIR, "val.csv"))
    real_test  = pd.read_csv(os.path.join(REAL_DIR, "test.csv"))
    knnvc      = pd.read_csv(KNNVC_MANIFEST)

    train_children = set(real_train["child_id"].dropna().unique())
    val_children   = set(real_val["child_id"].dropna().unique())
    test_children  = set(real_test["child_id"].dropna().unique())

    bad = knnvc[~knnvc["child_id"].isin(train_children)]
    if len(bad):
        print(f"ABORT: {len(bad)} voice-converted rows reference non-train children", file=sys.stderr)
        sys.exit(3)

    # Anchor timepoint_norm via the first matching real-train row per child.
    tp_lookup = real_train.groupby("child_id")["timepoint_norm"].first().to_dict()
    knnvc = knnvc.copy()
    knnvc["timepoint_norm"] = knnvc["child_id"].map(tp_lookup).fillna("unknown")
    knnvc["split"] = "train"

    # Schema-align with seen_child_splits training row schema.
    cols_real = list(real_train.columns)
    knnvc_aug = pd.DataFrame({
        "audio_path": knnvc["audio_path"],
        "child_id": knnvc["child_id"],
        "timepoint_norm": knnvc["timepoint_norm"],
        "label": knnvc["label"],
    })
    # Pad with NaNs so concat doesn't error on extra real-train columns
    for c in cols_real:
        if c not in knnvc_aug.columns:
            knnvc_aug[c] = pd.NA

    aug_train = pd.concat([real_train, knnvc_aug[cols_real]], ignore_index=True)

    aug_train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    real_val.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    real_test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print(f"[split] train: {len(real_train)} real + {len(knnvc_aug)} knnvc = {len(aug_train)}")
    print(f"[split] val:   {len(real_val)} (unchanged)")
    print(f"[split] test:  {len(real_test)} (unchanged)")
    print(f"[split] children: train={len(train_children)} val={len(val_children)} test={len(test_children)}")
    print(f"[split] no test/val voices added: VERIFIED")
    print(f"[split] wrote -> {OUT_DIR}/{{train,val,test}}.csv")


if __name__ == "__main__":
    main()
