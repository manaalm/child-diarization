"""Build a seen-child split directory with synth scenes appended to train,
mirroring whisper-modeling/seen_child_splits/{train,val,test}.csv.

Output: whisper-modeling/seen_child_splits_synth_aug/{train,val,test}.csv
"""
import os
import shutil

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "whisper-modeling/seen_child_splits")
DST = os.path.join(REPO, "whisper-modeling/seen_child_splits_synth_aug")
SYNTH = os.path.join(REPO, "synth_results/manifests/synthetic_train_aug.csv")

os.makedirs(DST, exist_ok=True)
shutil.copy(os.path.join(SRC, "val.csv"), os.path.join(DST, "val.csv"))
shutil.copy(os.path.join(SRC, "test.csv"), os.path.join(DST, "test.csv"))

real_train = pd.read_csv(os.path.join(SRC, "train.csv"))
synth = pd.read_csv(SYNTH)

required = ["audio_path", "child_id", "timepoint_norm", "label", "split"]
real_norm = real_train[[c for c in required if c in real_train.columns]].copy()
synth_norm = pd.DataFrame({
    "audio_path": synth["audio_path"],
    "child_id": synth["child_id"],
    "timepoint_norm": synth["timepoint_norm"],
    "label": synth["label"].astype(int),
    "split": "train",
})
combined = pd.concat([real_norm, synth_norm], ignore_index=True)
combined.to_csv(os.path.join(DST, "train.csv"), index=False)
print(f"Real train: {len(real_norm)}  Synth: {len(synth_norm)}  Combined: {len(combined)}")
print(f"Wrote {DST}/train.csv ({len(combined)} rows)")
