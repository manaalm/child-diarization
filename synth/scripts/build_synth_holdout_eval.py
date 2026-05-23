"""Build a held-out synth localization eval set.

Picks 200 scenes from the v2 corpus (`synth_results/synthetic_scenes_v2/`) —
balanced 100 positive (TARGET_CHILD speaks) + 100 negative (no TARGET_CHILD)
— with a deterministic seed (43, distinct from the v2 generation seed of 42)
so the selection is reproducible.

Why a separate holdout: every v2 scene appears in at least one spec-016
training manifest, so all 5000 are "training-tainted" for the synth-trained
candidates (C1 USC-SAIL synth, C2 pseudo-frame synth, C2_v2, C1-distill).
The 9 non-synth-trained systems (USC-SAIL real, Pyannote, BabAR, VTC,
VTC-KCHI, VBx, EEND-EDA, Sortformer, Pseudo-frame baseline) never see synth
during training, so for them this 200-scene set is genuinely held out.

Outputs:
  synth_results/synthetic_scenes_v2/holdout_eval_200/wav/   (symlinks)
  synth_results/synthetic_scenes_v2/holdout_eval_200/rttm/  (symlinks)
  synth_results/manifests/synth_holdout_eval.csv            (selected scene_ids + meta)
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
V2_DIR = os.path.join(REPO, "synth_results", "synthetic_scenes_v2")
WAV_DIR = os.path.join(V2_DIR, "wav")
RTTM_DIR = os.path.join(V2_DIR, "rttm")
HOLDOUT_DIR = os.path.join(V2_DIR, "holdout_eval_200")
HOLDOUT_WAV = os.path.join(HOLDOUT_DIR, "wav")
HOLDOUT_RTTM = os.path.join(HOLDOUT_DIR, "rttm")
MANIFEST = os.path.join(REPO, "synth_results", "manifests", "synth_holdout_eval.csv")

N_POS = 100
N_NEG = 100
SEED = 43


def has_target_child(rttm_path: str) -> bool:
    with open(rttm_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 8 and parts[7] == "TARGET_CHILD":
                return True
    return False


def main():
    os.makedirs(HOLDOUT_WAV, exist_ok=True)
    os.makedirs(HOLDOUT_RTTM, exist_ok=True)

    pos = []
    neg = []
    for rttm_fname in sorted(os.listdir(RTTM_DIR)):
        if not rttm_fname.endswith(".rttm"):
            continue
        scene_id = rttm_fname[:-len(".rttm")]
        wav_path = os.path.join(WAV_DIR, f"{scene_id}.wav")
        rttm_path = os.path.join(RTTM_DIR, rttm_fname)
        if not os.path.isfile(wav_path):
            continue
        (pos if has_target_child(rttm_path) else neg).append(scene_id)

    print(f"v2 corpus: {len(pos)} positive (TARGET_CHILD speaks), "
          f"{len(neg)} negative")

    rng = random.Random(SEED)
    sel_pos = sorted(rng.sample(pos, N_POS))
    sel_neg = sorted(rng.sample(neg, N_NEG))
    selected = sel_pos + sel_neg

    # Symlink wav + rttm into the holdout dir
    n_skipped = 0
    for scene_id in selected:
        src_wav = os.path.join(WAV_DIR, f"{scene_id}.wav")
        src_rttm = os.path.join(RTTM_DIR, f"{scene_id}.rttm")
        dst_wav = os.path.join(HOLDOUT_WAV, f"{scene_id}.wav")
        dst_rttm = os.path.join(HOLDOUT_RTTM, f"{scene_id}.rttm")
        for src, dst in ((src_wav, dst_wav), (src_rttm, dst_rttm)):
            if os.path.islink(dst) or os.path.isfile(dst):
                n_skipped += 1
                continue
            os.symlink(src, dst)
    print(f"Symlinked {len(selected)*2 - n_skipped} files; "
          f"{n_skipped} already existed.")

    # Manifest
    with open(MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "label", "audio_path", "rttm_path"])
        for sid in sel_pos:
            w.writerow([sid, 1,
                        os.path.join(HOLDOUT_WAV, f"{sid}.wav"),
                        os.path.join(HOLDOUT_RTTM, f"{sid}.rttm")])
        for sid in sel_neg:
            w.writerow([sid, 0,
                        os.path.join(HOLDOUT_WAV, f"{sid}.wav"),
                        os.path.join(HOLDOUT_RTTM, f"{sid}.rttm")])
    print(f"Wrote manifest: {MANIFEST}  "
          f"({N_POS} pos, {N_NEG} neg, seed={SEED})")
    print(f"Holdout audio dir: {HOLDOUT_WAV}")
    print(f"Holdout rttm dir:  {HOLDOUT_RTTM}")


if __name__ == "__main__":
    main()
