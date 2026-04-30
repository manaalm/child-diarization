"""Build a combined RTTM cache for seg-MIL synth-augmentation experiment.

Copies (symlinks) all USC-SAIL RTTM cache entries (real seen-child clips) plus
synth ground-truth RTTMs (renamed to {stem}__{md5(audio_path)}.rttm) into a
single directory that seg_dataset.py can resolve uniformly.

Output: mil/seg_mil_combined_cache/
"""
import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_USC_SAIL = os.path.join(REPO, "whisper-modeling/usc_sail_rttm_cache")
SYNTH_MANIFEST = os.path.join(REPO, "synth_results/manifests/synthetic_manifest.csv")
DST = os.path.join(REPO, "mil/seg_mil_combined_cache")

os.makedirs(DST, exist_ok=True)

# 1) Copy USC-SAIL real RTTMs (symlink for speed/space)
n_real = 0
for fname in os.listdir(SRC_USC_SAIL):
    if not fname.endswith(".rttm"):
        continue
    src = os.path.abspath(os.path.join(SRC_USC_SAIL, fname))
    dst = os.path.join(DST, fname)
    if not os.path.exists(dst):
        os.symlink(src, dst)
    n_real += 1

# 2) Symlink synth RTTMs with the seg_dataset naming convention
df = pd.read_csv(SYNTH_MANIFEST)
n_synth = 0
n_missing = 0
for row in df.itertuples(index=False):
    audio_path = row.audio_path
    rttm_src = row.rttm_path
    if not os.path.exists(rttm_src):
        n_missing += 1
        continue
    stem = Path(audio_path).stem
    cid = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
    dst = os.path.join(DST, f"{stem}__{cid}.rttm")
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(rttm_src), dst)
    n_synth += 1

print(f"Real RTTMs symlinked: {n_real}")
print(f"Synth RTTMs symlinked: {n_synth}  (missing: {n_missing})")
print(f"Combined cache → {DST}")
