"""Build a 2000-scene subsample of the v3 cross-child synth split for spec-021 US2 (T040).

Mirrors synth/scripts/build_v3_aug_manifests.py:step3 but caps the synth append
at exactly 2000 rows (stratified label-balanced, seed=42 per spec FR-014/SC-012).

Output: baselines/splits_synth_aug_v3_2k/{train,val,test}.csv.
val.csv and test.csv are copied unchanged from baselines/splits/, so the v3-vs-
v3-2k cross-child comparison is apples-to-apples on the *evaluation* side.
"""
import os
import shutil

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "baselines/splits")
SYNTH_V3 = os.path.join(REPO, "synth_results/manifests/synthetic_cross_child_aug_v3.csv")
DST = os.path.join(REPO, "baselines/splits_synth_aug_v3_2k")

CAP = 2000
SEED = 42


def stratified_cap(synth: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    pos = synth[synth["label"] == 1]
    neg = synth[synth["label"] == 0]
    pos_frac = len(pos) / len(synth)
    n_pos = round(cap * pos_frac)
    n_neg = cap - n_pos
    n_pos = min(n_pos, len(pos))
    n_neg = min(n_neg, len(neg))
    sampled = pd.concat([
        pos.sample(n=n_pos, random_state=seed),
        neg.sample(n=n_neg, random_state=seed),
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


def main() -> None:
    os.makedirs(DST, exist_ok=True)
    shutil.copy(os.path.join(SRC, "val.csv"), os.path.join(DST, "val.csv"))
    shutil.copy(os.path.join(SRC, "test.csv"), os.path.join(DST, "test.csv"))

    real_train = pd.read_csv(os.path.join(SRC, "train.csv"))
    synth_full = pd.read_csv(SYNTH_V3)
    synth = stratified_cap(synth_full, CAP, SEED)

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
    combined.to_csv(os.path.join(DST, "train.csv"), index=False)

    n_pos_synth = int((synth_norm["label"] == 1).sum())
    n_neg_synth = int((synth_norm["label"] == 0).sum())
    print(f"Real train: {len(real_norm)}  Synth: {len(synth_norm)} "
          f"({n_pos_synth} pos / {n_neg_synth} neg)  Combined: {len(combined)}")
    print(f"Wrote {DST}/train.csv ({len(combined)} rows)")
    print(f"v3-2k synth stratification: pos_frac = {len(synth.query('label==1'))/len(synth):.4f}")


if __name__ == "__main__":
    main()
