import argparse
import json
import os
import shutil
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from bids_timepoint import derive_timepoint_with_provenance


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

    # spec 022 US1: BIDS-derived timepoint (instead of spreadsheet's `timepoint` column)
    use_bids_timepoint: bool = True
    # spec 022 US3: also emit all_children_splits/test_all.csv (zero-shot eval only)
    build_all_children_split: bool = False


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
    out["spreadsheet_timepoint"] = out["timepoint"].apply(normalize_timepoint)
    out["label"] = out["Vocalizations"].apply(vocalizations_to_label)

    if cfg.use_bids_timepoint:
        provenance = out.apply(
            lambda r: derive_timepoint_with_provenance(
                r["audio_path"], r["spreadsheet_timepoint"]
            ),
            axis=1,
        )
        prov_df = pd.DataFrame(list(provenance))
        for col in ["bids_session_id", "bids_timepoint", "agree", "rationale_if_disagree", "decision"]:
            out[col] = prov_df[col].values
        # canonical timepoint column = BIDS when decision says keep-bids; else spreadsheet
        out["timepoint_norm"] = np.where(
            out["decision"].isin(["keep-bids"]),
            out["bids_timepoint"],
            out["spreadsheet_timepoint"],
        )
        out["timepoint_norm"] = out["timepoint_norm"].replace({"unknown": None})
    else:
        out["timepoint_norm"] = out["spreadsheet_timepoint"]

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
    splits_dir = os.path.join(cfg.out_dir, "seen_child_splits")
    os.makedirs(splits_dir, exist_ok=True)

    # spec 022 US1: back up prior splits before overwriting (Constitution VI)
    for fname in ("master_with_split.csv", "train.csv", "val.csv", "test.csv", "split_summary.json"):
        src = os.path.join(splits_dir, fname)
        if os.path.exists(src):
            backup = src + ".legacy_pre_bids_022"
            if not os.path.exists(backup):
                shutil.copyfile(src, backup)

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

    full.to_csv(os.path.join(splits_dir, "master_with_split.csv"), index=False)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

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

    with open(os.path.join(splits_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # spec 022 US1: dump per-row BIDS provenance to bids_correction_provenance.json
    if cfg.use_bids_timepoint and "bids_session_id" in full.columns:
        prov_cols = [
            "child_id", "audio_path", "bids_session_id", "bids_timepoint",
            "spreadsheet_timepoint", "agree", "rationale_if_disagree", "decision",
        ]
        prov_records = full[prov_cols].to_dict(orient="records")
        with open(os.path.join(splits_dir, "bids_correction_provenance.json"), "w") as f:
            json.dump(prov_records, f, indent=2)

    print(json.dumps(summary, indent=2))


def make_all_children_split(cfg: Config) -> None:
    """spec 022 US3 / FR-014: emit a universal-coverage eval split with no
    timepoint-balance filter, for zero-shot baselines only. No train/val/test
    partitioning; consumers reuse the seen-child val threshold."""
    relaxed = Config(**{**cfg.__dict__, "require_timepoint": False, "min_clips_per_child": 1})
    df = build_master_dataframe(relaxed)

    seen_child_master = os.path.join(cfg.out_dir, "seen_child_splits", "master_with_split.csv")
    excluded_clip_ids = set()
    if os.path.exists(seen_child_master):
        seen_df = pd.read_csv(seen_child_master)
        # Heuristic: identify rows by (child_id, audio_path) tuples
        seen_keys = set(zip(seen_df["child_id"], seen_df["audio_path"]))
        df["excluded_from_seen_child_split"] = df.apply(
            lambda r: (r["child_id"], r["audio_path"]) not in seen_keys, axis=1
        )
    else:
        df["excluded_from_seen_child_split"] = True

    # Per-child clip count
    counts = df.groupby("child_id").size().rename("n_clips_for_this_child").reset_index()
    df = df.merge(counts, on="child_id", how="left")

    # Exclusion reason
    def _reason(r):
        if not r["excluded_from_seen_child_split"]:
            return "none"
        if r["timepoint_norm"] is None or pd.isna(r["timepoint_norm"]):
            return "timepoint-missing"
        if r["n_clips_for_this_child"] < CFG.min_clips_per_child:
            return "min-clips-per-child"
        return "none"

    df["exclusion_reason"] = df.apply(_reason, axis=1)

    out_dir = os.path.join(cfg.out_dir, "all_children_splits")
    os.makedirs(out_dir, exist_ok=True)
    keep = [
        "child_id", "audio_path", "timepoint_norm", "label",
        "n_clips_for_this_child", "excluded_from_seen_child_split", "exclusion_reason",
    ]
    # add clip_id if present in spreadsheet (use SourceFile or row index)
    if "FileName" in df.columns:
        df["clip_id"] = df["FileName"].astype(str)
        keep = ["clip_id"] + keep
    df[keep].to_csv(os.path.join(out_dir, "test_all.csv"), index=False)

    summary = {
        "n_rows": int(len(df)),
        "n_children": int(df["child_id"].nunique()),
        "labels": df["label"].value_counts().to_dict(),
        "exclusion_reasons": df["exclusion_reason"].value_counts().to_dict(),
        "n_excluded_from_seen_child": int(df["excluded_from_seen_child_split"].sum()),
    }
    with open(os.path.join(out_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nall-children split summary:")
    print(json.dumps(summary, indent=2))


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--use-bids-timepoint", dest="use_bids_timepoint", action="store_true",
                   default=CFG.use_bids_timepoint,
                   help="derive timepoint from BIDS ses-NN rather than spreadsheet (default true)")
    p.add_argument("--no-bids-timepoint", dest="use_bids_timepoint", action="store_false",
                   help="legacy mode: read timepoint_norm from spreadsheet only")
    p.add_argument("--build-all-children-split", action="store_true", default=False,
                   help="also emit whisper-modeling/all_children_splits/test_all.csv (spec 022 US3)")
    p.add_argument("--annotations-csv", default=CFG.annotations_csv)
    p.add_argument("--out-dir", default=CFG.out_dir)
    p.add_argument("--seed", type=int, default=CFG.seed)
    args = p.parse_args()
    return Config(
        annotations_csv=args.annotations_csv,
        out_dir=args.out_dir,
        seed=args.seed,
        use_bids_timepoint=args.use_bids_timepoint,
        build_all_children_split=args.build_all_children_split,
    )


if __name__ == "__main__":
    cfg = _parse_args() if len(os.sys.argv) > 1 else CFG
    make_seen_child_split(cfg)
    if cfg.build_all_children_split:
        make_all_children_split(cfg)