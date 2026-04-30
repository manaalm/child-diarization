"""Build a cross-child split directory with synth scenes appended to train.

Mirrors baselines/splits/{train,val,test}.csv but train.csv has 5000 synth rows
appended. val.csv and test.csv are copied unchanged so cross-child evaluation
on real children is unchanged.

Output: baselines/splits_synth_aug/{train,val,test}.csv
"""
import os
import shutil

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "baselines/splits")
DST = os.path.join(REPO, "baselines/splits_synth_aug")
SYNTH = os.path.join(REPO, "synth_results/manifests/synthetic_cross_child_aug.csv")

os.makedirs(DST, exist_ok=True)
shutil.copy(os.path.join(SRC, "val.csv"), os.path.join(DST, "val.csv"))
shutil.copy(os.path.join(SRC, "test.csv"), os.path.join(DST, "test.csv"))

real_train = pd.read_csv(os.path.join(SRC, "train.csv"))
synth = pd.read_csv(SYNTH)

# Keep only the columns mil_train.load_split actually consumes; pad missing with NaN.
required = ["audio_path", "child_id", "timepoint_norm", "label", "split"]
synth_norm = pd.DataFrame({
    "audio_path": synth["audio_path"],
    "child_id": synth["child_id"],
    "timepoint_norm": synth["timepoint_norm"],
    "label": synth["label"].astype(int),
    "split": "train",
})
# Add the same columns to real_train to harmonize, then concat.
real_norm = real_train[[c for c in required if c in real_train.columns]].copy()
combined = pd.concat([real_norm, synth_norm], ignore_index=True)
combined.to_csv(os.path.join(DST, "train.csv"), index=False)
print(f"Real train: {len(real_norm)}  Synth: {len(synth_norm)}  Combined: {len(combined)}")
print(f"Wrote {DST}/train.csv ({len(combined)} rows)")
