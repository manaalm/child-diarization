import os
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Config:
    annotations_csv: str = "/orcd/scratch/bcs/001/sensein/sails/BIDS_data/anotated_processed.csv"
    out_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling"
    seed: int = 42

    # within-child split
    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2

    min_clips_per_child: int = 5
    require_timepoint: bool = True


CFG = Config()


def bidsprocessed_to_audio_path(bids_processed_path: str) -> str:
    if pd.isna(bids_processed_path):
        return ""
    s = str(bids_processed_path).strip()
    suffix = "_desc-processed_beh.mp4"
    if not s.endswith(suffix):
        return ""
    return s[:-len(suffix)] + "_audio.wav"


def normalize_timepoint(tp: str) -> Optional[str]:
    if pd.isna(tp):
        return None
    tp = str(tp).strip()
    if tp in {"14_month", "36_month"}:
        return tp
    return None


def vocalizations_to_label(v) -> Optional[int]:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s == "yes":
        return 1
    if s == "no":
        return 0
    try:
        x = float(s)
        if x == 1:
            return 1
        if x == 0:
            return 0
    except Exception:
        pass
    return None


def build_master_dataframe(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.annotations_csv)

    out = df.copy()
    out["audio_path"] = out["BidsProcessed"].apply(bidsprocessed_to_audio_path)
    out["child_id"] = out["ID"].astype(str).str.strip()
    out["timepoint_norm"] = out["timepoint"].apply(normalize_timepoint)
    out["label"] = out["Vocalizations"].apply(vocalizations_to_label)

    out = out[out["audio_path"].astype(str) != ""].copy()
    out = out[out["child_id"].astype(str) != ""].copy()
    out = out[out["label"].notna()].copy()
    out["label"] = out["label"].astype(int)

    if cfg.require_timepoint:
        out = out[out["timepoint_norm"].notna()].copy()

    out["audio_exists"] = out["audio_path"].apply(os.path.exists)
    out = out[out["audio_exists"]].copy()

    out = out.reset_index(drop=True)
    return out


def split_one_group(df_group: pd.DataFrame, rng: np.random.RandomState, train_frac: float, val_frac: float):
    idx = np.arange(len(df_group))
    rng.shuffle(idx)

    n = len(idx)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))

    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    n_test = n - n_train - n_val
    if n_test < 1:
        if n_train > 1:
            n_train -= 1
        else:
            n_val -= 1
        n_test = 1

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    split = np.array([""] * n, dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    return split


def make_seen_child_split(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    df = build_master_dataframe(cfg)

    rng = np.random.RandomState(cfg.seed)
    split_parts = []

    dropped_children = []

    # split within each child and timepoint together, so both ages can exist separately if needed
    group_cols = ["child_id", "timepoint_norm"]

    for group_key, sub in df.groupby(group_cols, dropna=False):
        if len(sub) < cfg.min_clips_per_child:
            dropped_children.append(
                {"group": str(group_key), "n_rows": int(len(sub))}
            )
            continue

        sub = sub.copy()
        sub["split"] = split_one_group(sub, rng, cfg.train_frac, cfg.val_frac)
        split_parts.append(sub)

    if not split_parts:
        raise RuntimeError("No groups survived splitting. Lower min_clips_per_child.")

    full = pd.concat(split_parts, axis=0).reset_index(drop=True)

    train_df = full[full["split"] == "train"].copy()
    val_df = full[full["split"] == "val"].copy()
    test_df = full[full["split"] == "test"].copy()

    full.to_csv(os.path.join(cfg.out_dir, "master_with_split.csv"), index=False)
    train_df.to_csv(os.path.join(cfg.out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(cfg.out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(cfg.out_dir, "test.csv"), index=False)

    summary = {
        "seed": cfg.seed,
        "n_total": int(len(full)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_children_total": int(full["child_id"].nunique()),
        "n_children_train": int(train_df["child_id"].nunique()),
        "n_children_val": int(val_df["child_id"].nunique()),
        "n_children_test": int(test_df["child_id"].nunique()),
        "timepoints_train": train_df["timepoint_norm"].value_counts().to_dict(),
        "timepoints_val": val_df["timepoint_norm"].value_counts().to_dict(),
        "timepoints_test": test_df["timepoint_norm"].value_counts().to_dict(),
        "labels_train": train_df["label"].value_counts().to_dict(),
        "labels_val": val_df["label"].value_counts().to_dict(),
        "labels_test": test_df["label"].value_counts().to_dict(),
        "dropped_groups": dropped_children,
    }

    with open(os.path.join(cfg.out_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    make_seen_child_split(CFG)