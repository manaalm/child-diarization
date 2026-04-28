"""Age-stratified MIL evaluation for one backbone variant and one age cohort.

Usage:
    python mil/mil_age_stratified.py \\
        --checkpoint mil/mil_results/wavlm_mil/best_checkpoint.pt \\
        --config     mil/mil_results/wavlm_mil/config.json \\
        --age-group  12_16m \\
        --manifest   playlogue/manifest.csv
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from mil.mil_dataset import MILBagDataset, mil_collate_fn
from mil.mil_model import build_mil_model
from mil.mil_utils import compute_metrics, per_timepoint_metrics, save_csv, save_json
from mil.mil_train import load_split


def run_age_stratified(checkpoint_path: str, config_path: str,
                       age_group: str, manifest_path: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)

    result_dir = os.path.dirname(config_path)
    out_dir = os.path.join(result_dir, "age_stratified", age_group)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_mil_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_metrics_path = os.path.join(result_dir, "val_metrics_tuned.json")
    with open(val_metrics_path) as f:
        threshold = float(json.load(f)["threshold"])
    print(f"Threshold: {threshold:.4f} | Age group: {age_group}", flush=True)

    # Load test split and filter to age_group via manifest
    test_df = load_split(cfg["split_dir"], "test")
    manifest_df = pd.read_csv(os.path.join(_REPO, manifest_path))
    if "age_group" not in manifest_df.columns:
        raise ValueError(f"Manifest {manifest_path} must have an 'age_group' column")

    # playlogue/manifest.csv uses "path"; seen_child_splits uses "audio_path"
    if "audio_path" not in manifest_df.columns and "path" in manifest_df.columns:
        manifest_df = manifest_df.rename(columns={"path": "audio_path"})
    merged = test_df.merge(
        manifest_df[["audio_path", "age_group"]].drop_duplicates("audio_path"),
        on="audio_path",
        how="inner",
    )
    cohort_df = merged[merged["age_group"] == age_group].reset_index(drop=True)
    print(f"Cohort clips: {len(cohort_df)}", flush=True)

    if len(cohort_df) == 0:
        print(f"WARNING: No test clips found for age_group={age_group}. "
              f"Check manifest path and column values.", flush=True)
        return

    w_sec = cfg.get("window_sec", 2.0)
    s_sec = cfg.get("stride_sec", 1.0)
    ds = MILBagDataset(cohort_df, window_sec=w_sec, stride_sec=s_sec)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=mil_collate_fn, num_workers=2)

    scores, labels, meta = [], [], []
    with torch.no_grad():
        for batch in loader:
            for i, windows in enumerate(batch["windows"]):
                logit, _ = model(windows)
                scores.append(float(torch.sigmoid(logit).item()))
                labels.append(int(batch["labels"][i].item()))
                meta.append({
                    "audio_path": batch["audio_paths"][i],
                    "child_id": batch["child_ids"][i],
                    "timepoint_norm": batch["timepoint_norms"][i],
                })

    metrics = compute_metrics(labels, scores, threshold=threshold)
    metrics["threshold"] = threshold

    preds_df = pd.DataFrame([
        {**m, "label": lbl, "score": sc, "prediction": int(sc >= threshold)}
        for m, lbl, sc in zip(meta, labels, scores)
    ])
    tp_df = per_timepoint_metrics(preds_df)

    save_json(metrics, os.path.join(out_dir, "test_metrics_tuned.json"))
    save_csv(preds_df, os.path.join(out_dir, "test_predictions.csv"))
    save_csv(tp_df, os.path.join(out_dir, "test_metrics_by_timepoint.csv"))

    print(f"  F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}  "
          f"AUPRC={metrics['auprc']:.4f}", flush=True)
    print(f"Results written to: {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Age-stratified MIL evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--age-group", required=True, choices=["12_16m", "34_38m"])
    parser.add_argument("--manifest", required=True,
                        help="Age-annotated manifest CSV (from prepare_age_manifests.py)")
    args = parser.parse_args()
    run_age_stratified(args.checkpoint, args.config, args.age_group, args.manifest)


if __name__ == "__main__":
    main()
