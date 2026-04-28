"""Segment-instance MIL sweep training entry point.

Purpose: Train and evaluate all 16 (frontend × aggregator) configurations,
         writing per-config results and an all_configs.json summary.
Inputs:  seg_mil_sweep.yaml config file.
Outputs: mil/mil_results/seg_mil/{frontend}_{aggregator}/ per-config directories,
         mil/mil_results/seg_mil/all_configs.json summary.
Side effects: Writes segment embedding cache to mil/seg_embedding_cache/.
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import WavLMModel

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from mil.seg_dataset import SegmentBagDataset, precompute_embeddings
from mil.seg_embedding_cache import SegmentEmbeddingCache
from mil.seg_model import build_aggregator
from mil.mil_utils import compute_metrics, per_timepoint_metrics, save_csv, save_json, tune_threshold


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(split_dir: str, split: str) -> pd.DataFrame:
    path = os.path.join(_REPO, split_dir, f"{split}.csv")
    df = pd.read_csv(path)
    if "audio_exists" in df.columns:
        df = df[df["audio_exists"] == True]
    if "timepoint_norm" not in df.columns and "timepoint" in df.columns:
        df = df.rename(columns={"timepoint": "timepoint_norm"})
    return df[["audio_path", "child_id", "timepoint_norm", "label"]].reset_index(drop=True)


def _abs_path(rel: str) -> str:
    return os.path.join(_REPO, rel)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _run_inference(
    model: nn.Module,
    ds: SegmentBagDataset,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[List[float], List[int], pd.DataFrame]:
    """Run model over full dataset; return (scores, labels, pred_df)."""
    model.eval()
    rows = []
    scores_out, labels_out = [], []

    with torch.no_grad():
        for idx in range(len(ds)):
            bag, mask, label, meta = ds[idx]
            bag = bag.to(device)
            mask = mask.to(device)
            logit, weights = model(bag, mask)
            prob = float(torch.sigmoid(logit).item())
            pred = int(prob >= threshold)
            scores_out.append(prob)
            labels_out.append(int(label))

            n_inst = meta["n_instances"]
            segs = ds._bags[idx]  # direct index access

            top_start, top_end, top_weight = None, None, None
            if weights is not None and n_inst > 0:
                w_np = weights.cpu().numpy()
                valid_w = w_np[:n_inst]
                # T017: assert weights sum to ~1
                assert abs(valid_w.sum() - 1.0) < 1e-4, (
                    f"Weights sum={valid_w.sum():.6f} for {meta['audio_path']}"
                )
                best_i = int(np.argmax(valid_w))
                top_weight = float(valid_w[best_i])
                if best_i < len(segs):
                    top_start = segs[best_i]["start"]
                    top_end = segs[best_i]["end"]

            rows.append({
                "audio_path": meta["audio_path"],
                "child_id": meta["child_id"],
                "timepoint_norm": meta["timepoint_norm"],
                "label": int(label),
                "prob": prob,
                "pred": pred,
                "n_instances": n_inst,
                "top_seg_start": top_start,
                "top_seg_end": top_end,
                "top_seg_weight": top_weight,
            })

    return scores_out, labels_out, pd.DataFrame(rows)


def _run_segment_weights(
    model: nn.Module,
    ds: SegmentBagDataset,
    device: torch.device,
) -> pd.DataFrame:
    """Per-(clip, segment) attention weight DataFrame for attention configs."""
    model.eval()
    rows = []
    with torch.no_grad():
        for idx in range(len(ds)):
            bag, mask, label, meta = ds[idx]
            bag = bag.to(device)
            mask = mask.to(device)
            _, weights = model(bag, mask)
            if weights is None:
                continue
            n_inst = meta["n_instances"]
            w_np = weights.cpu().numpy()[:n_inst]
            for i, seg in enumerate(ds._bags[idx][:n_inst]):
                rows.append({
                    "audio_path": meta["audio_path"],
                    "child_id": meta["child_id"],
                    "seg_start": seg["start"],
                    "seg_end": seg["end"],
                    "attention_weight": float(w_np[i]),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T009: train_one_config
# ---------------------------------------------------------------------------

def train_one_config(
    cfg: dict,
    train_ds: SegmentBagDataset,
    val_ds: SegmentBagDataset,
    test_ds: SegmentBagDataset,
    device: torch.device,
) -> Tuple[nn.Module, dict, dict, pd.DataFrame, pd.DataFrame]:
    """Train one (frontend, aggregator) configuration.

    Returns (model, val_metrics, test_metrics, val_pred_df, test_pred_df).
    """
    assert cfg["seed"] == 42, "seed must be 42"
    setup_seed(cfg["seed"])

    embed_dim = train_ds._embed_dim
    agg_name = cfg["aggregator"]
    model = build_aggregator(
        agg_name, embed_dim,
        attn_dim=cfg.get("attn_dim", 256),
        k=cfg.get("top_k", 3),
        transformer_config=cfg.get("transformer_config"),
    ).to(device)

    weight_decay = cfg.get("weight_decay", 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = -1.0
    best_state: Optional[dict] = None
    patience_counter = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        indices = list(range(len(train_ds)))
        random.shuffle(indices)
        batch_size = cfg["batch_size"]

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start: batch_start + batch_size]
            optimizer.zero_grad()
            logits = []
            labels_batch = []
            for i in batch_idx:
                bag, mask, label, _ = train_ds[i]
                logit, _ = model(bag.to(device), mask.to(device))
                logits.append(logit)
                labels_batch.append(float(label))
            logits_t = torch.stack(logits)
            labels_t = torch.tensor(labels_batch, device=device)
            loss = criterion(logits_t, labels_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validate
        val_scores, val_labels, _ = _run_inference(model, val_ds, device)
        val_m = compute_metrics(val_labels, val_scores)
        val_auroc = val_m["auroc"] if not np.isnan(val_m.get("auroc", float("nan"))) else 0.0
        print(
            f"  epoch {epoch:2d}  loss={total_loss / max(n_batches, 1):.4f}  "
            f"val_auroc={val_auroc:.4f}",
            flush=True,
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch}", flush=True)
                break

    # Reload best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Tune threshold on val, evaluate on test
    val_scores, val_labels, _ = _run_inference(model, val_ds, device)
    threshold = tune_threshold(val_labels, val_scores)
    val_metrics = compute_metrics(val_labels, val_scores, threshold)
    val_metrics["threshold"] = threshold

    test_scores, test_labels, _ = _run_inference(model, test_ds, device)
    test_metrics = compute_metrics(test_labels, test_scores, threshold)
    test_metrics["threshold"] = threshold

    # Build final prediction DataFrames with attention columns
    _, _, val_pred_df = _run_inference(model, val_ds, device, threshold)
    _, _, test_pred_df = _run_inference(model, test_ds, device, threshold)

    return model, val_metrics, test_metrics, val_pred_df, test_pred_df


# ---------------------------------------------------------------------------
# T012 + T015 + T016: results writing
# ---------------------------------------------------------------------------

def write_config_results(
    out_dir: str,
    cfg: dict,
    val_metrics: dict,
    test_metrics: dict,
    val_pred_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    model: nn.Module,
    val_ds: SegmentBagDataset,
    test_ds: SegmentBagDataset,
    device: torch.device,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    agg_name = cfg["aggregator"]

    save_json(cfg, os.path.join(out_dir, "config.json"))
    save_csv(val_pred_df, os.path.join(out_dir, "val_predictions.csv"))
    save_csv(test_pred_df, os.path.join(out_dir, "test_predictions.csv"))
    save_json(val_metrics, os.path.join(out_dir, "val_metrics.json"))
    save_json(test_metrics, os.path.join(out_dir, "test_metrics.json"))

    # Per-timepoint breakdown (JSON + CSV)
    for split_name, pred_df in [("val", val_pred_df), ("test", test_pred_df)]:
        if "timepoint_norm" in pred_df.columns and "prob" in pred_df.columns:
            tp_df = pred_df.rename(columns={"prob": "score", "pred": "prediction"})
            tp_metrics = per_timepoint_metrics(tp_df)
            save_json(
                tp_metrics.to_dict(orient="records"),
                os.path.join(out_dir, f"{split_name}_metrics_by_timepoint.json"),
            )
            save_csv(tp_metrics, os.path.join(out_dir, f"{split_name}_metrics_by_timepoint.csv"))

    # Per-segment attention weight CSVs (attention-variant configs)
    if agg_name in ("attention", "gated_attention", "transformer"):
        val_sw = _run_segment_weights(model, val_ds, device)
        test_sw = _run_segment_weights(model, test_ds, device)
        save_csv(val_sw, os.path.join(out_dir, "val_segment_weights.csv"))
        save_csv(test_sw, os.path.join(out_dir, "test_segment_weights.csv"))


# ---------------------------------------------------------------------------
# T013: summary writer
# ---------------------------------------------------------------------------

def write_all_configs_summary(output_dir: str) -> None:
    """Read all per-config results and write all_configs.json."""
    entries = []
    for subdir in sorted(os.listdir(output_dir)):
        cfg_path = os.path.join(output_dir, subdir, "config.json")
        val_path = os.path.join(output_dir, subdir, "val_metrics.json")
        test_path = os.path.join(output_dir, subdir, "test_metrics.json")
        if not (os.path.exists(cfg_path) and os.path.exists(val_path) and os.path.exists(test_path)):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        with open(val_path) as f:
            val_m = json.load(f)
        with open(test_path) as f:
            test_m = json.load(f)
        # Per-age-band AUROC from test_metrics_by_timepoint.json (if present)
        tp_path = os.path.join(output_dir, subdir, "test_metrics_by_timepoint.json")
        auroc_14m, auroc_36m = None, None
        f1_14m, f1_36m = None, None
        if os.path.exists(tp_path):
            with open(tp_path) as f:
                tp_records = json.load(f)
            for rec in tp_records:
                tp = rec.get("timepoint", "")
                if "14" in tp:
                    auroc_14m = rec.get("auroc")
                    f1_14m = rec.get("f1")
                elif "36" in tp:
                    auroc_36m = rec.get("auroc")
                    f1_36m = rec.get("f1")

        entries.append({
            "frontend": cfg.get("frontend"),
            "aggregator": cfg.get("aggregator"),
            "val_f1": val_m.get("f1"),
            "val_auroc": val_m.get("auroc"),
            "val_auprc": val_m.get("auprc"),
            "test_f1": test_m.get("f1"),
            "test_precision": test_m.get("precision"),
            "test_recall": test_m.get("recall"),
            "test_auroc": test_m.get("auroc"),
            "test_auprc": test_m.get("auprc"),
            "test_auroc_14month": auroc_14m,
            "test_auroc_36month": auroc_36m,
            "test_f1_14month": f1_14m,
            "test_f1_36month": f1_36m,
            "threshold": test_m.get("threshold"),
            "n_train_bags": cfg.get("n_train_bags"),
            "n_empty_bags_train": cfg.get("n_empty_bags_train"),
            "config_path": os.path.join("mil/mil_results/seg_mil", subdir, "config.json"),
        })

    out_path = os.path.join(output_dir, "all_configs.json")
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Summary written: {out_path} ({len(entries)} entries)", flush=True)


# ---------------------------------------------------------------------------
# T010 + T011: main sweep loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Segment-instance MIL sweep")
    parser.add_argument("--config", required=True, help="Path to seg_mil_sweep.yaml")
    parser.add_argument(
        "--precompute-only",
        action="store_true",
        help="Pre-fill embedding cache for all frontends and exit without training",
    )
    args = parser.parse_args()

    with open(os.path.join(_REPO, args.config)) as f:
        sweep_cfg = yaml.safe_load(f)

    assert sweep_cfg["seed"] == 42, "seed must be 42"
    setup_seed(sweep_cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    output_dir = _abs_path(sweep_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Load splits once
    train_df = load_split(sweep_cfg["split_dir"], "train")
    val_df = load_split(sweep_cfg["split_dir"], "val")
    test_df = load_split(sweep_cfg["split_dir"], "test")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}", flush=True)

    # Load encoder once — shared across all frontends
    print(f"Loading encoder: {sweep_cfg['encoder']}", flush=True)
    encoder = WavLMModel.from_pretrained(sweep_cfg["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder = encoder.to(device)
    embed_dim = encoder.config.hidden_size
    encoder_layer = sweep_cfg.get("encoder_layer", -1)
    print(f"Encoder embed_dim={embed_dim}", flush=True)

    min_seg_dur = sweep_cfg.get("min_seg_dur_sec", 0.4)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    for frontend_cfg in sweep_cfg["frontends"]:
        frontend_name = frontend_cfg["name"]
        rttm_cache_dir = _abs_path(frontend_cfg["rttm_cache_dir"])
        embed_cache_dir = os.path.join(_abs_path(sweep_cfg["embedding_cache_dir"]), frontend_name)
        embed_cache = SegmentEmbeddingCache(embed_cache_dir)

        print(f"\n=== Frontend: {frontend_name} ===", flush=True)

        # Precompute embeddings once per frontend (shared across all 4 aggregators)
        precompute_embeddings(
            frontend_name, rttm_cache_dir, all_df, embed_cache,
            encoder, device, min_seg_dur, encoder_layer,
        )

        if args.precompute_only:
            continue

        # Build datasets once per frontend (model=None → use cache only)
        train_ds = SegmentBagDataset(
            frontend_name, rttm_cache_dir, train_df, embed_cache,
            None, device, min_seg_dur, encoder_layer,
        )
        val_ds = SegmentBagDataset(
            frontend_name, rttm_cache_dir, val_df, embed_cache,
            None, device, min_seg_dur, encoder_layer,
        )
        test_ds = SegmentBagDataset(
            frontend_name, rttm_cache_dir, test_df, embed_cache,
            None, device, min_seg_dur, encoder_layer,
        )

        n_empty_train = sum(1 for b in train_ds._bags if len(b) == 0)
        print(f"  Train bags: {len(train_ds)}, empty: {n_empty_train}", flush=True)

        for agg_name in sweep_cfg["aggregators"]:
            config_key = f"{frontend_name}_{agg_name}"
            config_out_dir = os.path.join(output_dir, config_key)

            # Resume support
            if os.path.exists(os.path.join(config_out_dir, "test_metrics.json")):
                print(f"  Skipping {config_key} (already done)", flush=True)
                continue

            print(f"\n  --- {config_key} ---", flush=True)

            # Aggregate weight_decay: use transformer config if applicable
            tc = sweep_cfg.get("transformer_config", {})
            weight_decay = tc.get("weight_decay", 0.0) if agg_name == "transformer" else 0.0

            run_cfg = {
                "frontend": frontend_name,
                "aggregator": agg_name,
                "encoder": sweep_cfg["encoder"],
                "encoder_layer": encoder_layer,
                "seed": sweep_cfg["seed"],
                "lr": sweep_cfg["lr"],
                "epochs": sweep_cfg["epochs"],
                "patience": sweep_cfg["patience"],
                "batch_size": sweep_cfg["batch_size"],
                "attn_dim": sweep_cfg.get("attn_dim", 256),
                "top_k": sweep_cfg.get("top_k", 3),
                "weight_decay": weight_decay,
                "min_seg_dur_sec": min_seg_dur,
                "n_train_bags": len(train_ds),
                "n_empty_bags_train": n_empty_train,
            }
            # Log transformer HPs in config for reproducibility
            if agg_name == "transformer" and tc:
                run_cfg["transformer_config"] = tc
                run_cfg["transformer_num_layers"] = tc.get("num_layers", 2)
                run_cfg["transformer_num_heads"] = tc.get("num_heads", 4)
                run_cfg["transformer_ffn_dim"] = tc.get("ffn_dim", 1536)
                run_cfg["transformer_dropout"] = tc.get("dropout", 0.3)
                run_cfg["transformer_weight_decay"] = tc.get("weight_decay", 0.01)

            trained_model, val_metrics, test_metrics, val_pred_df, test_pred_df = train_one_config(
                run_cfg, train_ds, val_ds, test_ds, device,
            )

            write_config_results(
                config_out_dir, run_cfg,
                val_metrics, test_metrics,
                val_pred_df, test_pred_df,
                trained_model, val_ds, test_ds, device,
            )

            print(
                f"  {config_key}: val_auroc={val_metrics.get('auroc', float('nan')):.4f}  "
                f"test_f1={test_metrics.get('f1', float('nan')):.4f}  "
                f"test_auroc={test_metrics.get('auroc', float('nan')):.4f}",
                flush=True,
            )

            # Update summary incrementally after each completed config
            write_all_configs_summary(output_dir)

    if not args.precompute_only:
        write_all_configs_summary(output_dir)
        print("\nSweep complete.", flush=True)
    else:
        print("\nPrecompute-only done.", flush=True)


if __name__ == "__main__":
    main()
