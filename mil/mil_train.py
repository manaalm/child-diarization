"""MIL training entry point.

Usage:
    python mil/mil_train.py --config mil/configs/wavlm_mil.yaml

Supports three head types via cfg["head"]:
  - gated_abmil (default; legacy)
  - acmil       (spec-014 US3)
  - tsmil       (spec-014 US4 — needs cfg["prototype_cache"])

Plus optional learnable weighted-layer-sum backbone (cfg["layer_aggregation"] = "weighted_sum",
spec-014 US1). All flags are backward-compatible — unset → existing behavior.

In legacy "last"-layer mode, the precompute caches per-window mean-pooled embeddings
(N_windows, D). In weighted_sum mode it caches per-layer per-window means (N_windows,
L, D); the head input is recomputed each step as softmax(layer_weights) @ stacked.
"""

import argparse
import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# Resolve repo root so imports work regardless of CWD
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from mil.mil_dataset import MILBagDataset
from mil.mil_model import ACMILHead, MILModel, TSMILHead, build_mil_model
from mil.mil_utils import (
    compute_metrics,
    per_timepoint_metrics,
    save_csv,
    save_json,
    tune_threshold,
)


def parse_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    required = ["variant_name", "backbone", "split_dir", "seed"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Config missing required key: {k}")
    return cfg


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
    return df.reset_index(drop=True)


def _load_prototype_cache(path: str) -> dict:
    """Load prototype .npz produced by mil/scripts/build_prototype_cache.py.

    Returns a dict mapping {child_id}__{timepoint_norm} -> torch.FloatTensor (D_proto,).
    """
    full = path if os.path.isabs(path) else os.path.join(_REPO, path)
    if not os.path.isfile(full):
        raise FileNotFoundError(
            f"Prototype cache not found at {full}. Run "
            f"`python mil/scripts/build_prototype_cache.py` first."
        )
    data = np.load(full, allow_pickle=False)
    out = {}
    for key in data.files:
        out[key] = torch.from_numpy(data[key]).float()
    return out


def _attach_prototypes(records: List[dict], proto_cache: dict, log_prefix: str) -> Tuple[List[dict], dict]:
    """Drop records whose (child_id, timepoint_norm) prototype is missing.

    Returns (filtered_records, missing_info_dict).
    """
    filtered = []
    missing_keys = set()
    n_missing = 0
    for rec in records:
        key = f"{rec['child_id']}__{rec['timepoint_norm']}"
        if key not in proto_cache:
            missing_keys.add(key)
            n_missing += 1
            continue
        rec["prototype"] = proto_cache[key]
        filtered.append(rec)
    if missing_keys:
        print(
            f"  [{log_prefix}] WARNING: dropped {n_missing} clips with missing prototypes "
            f"(across {len(missing_keys)} unique (child, timepoint) keys)",
            flush=True,
        )
    return filtered, {"missing_count": n_missing, "missing_keys": sorted(missing_keys)}


def _precompute_embeddings_last_layer(model: MILModel, dataset: MILBagDataset, device) -> dict:
    """Cache per-window mean-pooled embeddings (N_windows, D), indexed by audio_path.

    Used for layer_aggregation="last" mode (legacy precompute path).
    """
    cache = {}
    model.backbone.eval()
    n = len(dataset)
    with torch.no_grad():
        for idx in range(n):
            item = dataset[idx]
            path = item["audio_path"]
            if path in cache:
                continue
            embs = []
            for w in item["windows"]:
                w_t = w.unsqueeze(0).to(device)        # (1, 1, T)
                frames = model.backbone(w_t)            # (1, T_frames, D)
                emb = frames.mean(dim=1).squeeze(0)    # (D,)
                embs.append(emb.cpu())
            cache[path] = torch.stack(embs, dim=0)    # (N_windows, D) on CPU
            if (idx + 1) % 100 == 0 or (idx + 1) == n:
                print(f"  {idx + 1}/{n} clips embedded", flush=True)
    return cache


def _precompute_embeddings_per_layer(model: MILModel, dataset: MILBagDataset, device) -> dict:
    """Cache per-window per-layer mean-pooled embeddings (N_windows, L, D).

    Used for layer_aggregation="weighted_sum" mode. Dramatically more memory than the
    last-layer cache (×L), but avoids re-running the backbone every training step.
    """
    cache = {}
    model.backbone.eval()
    n = len(dataset)
    skip_first = model.backbone.layer_aggregation_skip_first
    n_layers = model.backbone._n_layers_combined
    is_whisper = model.backbone._is_whisper
    with torch.no_grad():
        for idx in range(n):
            item = dataset[idx]
            path = item["audio_path"]
            if path in cache:
                continue
            window_means = []  # list of (L, D) tensors, one per window
            for w in item["windows"]:
                w_t = w.unsqueeze(0).to(device)        # (1, 1, T)
                w_squeezed = w_t.squeeze(1)             # (1, T)
                if is_whisper:
                    waveform_np = w_squeezed.cpu().float().numpy()
                    inputs = model.backbone.processor(
                        waveform_np,
                        sampling_rate=model.backbone.sample_rate,
                        return_tensors="pt",
                    )
                    input_features = inputs["input_features"].to(device)
                    out = model.backbone.model.encoder(input_features, output_hidden_states=True)
                else:
                    out = model.backbone.model(w_squeezed, output_hidden_states=True)
                hidden_states = out.hidden_states
                if skip_first and len(hidden_states) > n_layers:
                    layers = hidden_states[1:]
                else:
                    layers = hidden_states
                stack = torch.stack(list(layers), dim=0)   # (L, 1, T_frames, D)
                stack = stack.squeeze(1)                     # (L, T_frames, D)
                mean = stack.mean(dim=1)                     # (L, D)
                window_means.append(mean.cpu())
            cache[path] = torch.stack(window_means, dim=0)  # (N_windows, L, D)
            if (idx + 1) % 100 == 0 or (idx + 1) == n:
                print(f"  {idx + 1}/{n} clips embedded (per-layer)", flush=True)
    return cache


def _make_instance_embeddings(
    rec: dict,
    use_weighted_sum: bool,
    layer_weights: Optional[torch.Tensor],
    device,
) -> torch.Tensor:
    """Convert a record's cached embedding tensor into the (N_windows, D) instance tensor.

    For "last" mode this is a no-op (cached as (N, D) already).
    For "weighted_sum" mode, mix per-layer caches via softmax(layer_weights).
    """
    if not use_weighted_sum:
        return rec["emb"].to(device)
    cached = rec["emb"].to(device)            # (N_windows, L, D)
    w = torch.softmax(layer_weights, dim=0)    # (L,)
    return torch.einsum("nld,l->nd", cached, w)


def _head_forward(
    model: MILModel,
    emb: torch.Tensor,
    prototype: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call mil_head with the right signature for each head type.

    Returns (logit, attn, div_loss). div_loss is 0 for non-ACMIL heads.
    """
    if isinstance(model.mil_head, ACMILHead):
        logit, attn, _branch_attn, div_loss = model.mil_head(emb)
        return logit, attn, div_loss
    if isinstance(model.mil_head, TSMILHead):
        if prototype is None:
            raise ValueError("TSMILHead requires a prototype tensor in the record.")
        logit, attn = model.mil_head(emb, prototype)
        return logit, attn, torch.zeros((), device=emb.device, dtype=emb.dtype)
    logit, attn = model.mil_head(emb)
    return logit, attn, torch.zeros((), device=emb.device, dtype=emb.dtype)


def train(cfg: dict) -> None:
    setup_seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    variant = cfg["variant_name"]
    result_dir = os.path.join(_REPO, "mil", "mil_results", variant)
    os.makedirs(result_dir, exist_ok=True)

    # ── Pre-flight: child-adapted backbone checkpoint guard (US2 FR-007) ──
    backbone_path = cfg.get("backbone_path")
    if backbone_path:
        full_bp = backbone_path if os.path.isabs(backbone_path) else os.path.join(_REPO, backbone_path)
        if not (os.path.isdir(full_bp) and os.path.isfile(os.path.join(full_bp, "config.json"))):
            print(
                f"ERROR: backbone_path={backbone_path} missing or incomplete. "
                f"For child-adapted runs, submit synth/slurm/run_wavlm_pretrain.sh first.",
                file=sys.stderr, flush=True,
            )
            sys.exit(2)

    # ── Data ───────────────────────────────────────────────────────────────
    train_df = load_split(cfg["split_dir"], "train")
    val_df = load_split(cfg["split_dir"], "val")

    extra_neg_csv = cfg.get("extra_negatives_csv")
    if extra_neg_csv:
        extra_path = os.path.join(_REPO, extra_neg_csv)
        extra_df = pd.read_csv(extra_path)
        cap = cfg.get("extra_negatives_cap", len(train_df[train_df["label"] == 0]))
        if len(extra_df) > cap:
            extra_df = extra_df.sample(n=cap, random_state=cfg["seed"])
        train_df = pd.concat([train_df, extra_df], ignore_index=True)
        print(f"Extra negatives: {len(extra_df)} rows from {extra_neg_csv}", flush=True)

    print(f"Train clips: {len(train_df)}  |  Val clips: {len(val_df)}", flush=True)

    w_sec = cfg.get("window_sec", 2.0)
    s_sec = cfg.get("stride_sec", 1.0)
    pad_to_sec = cfg.get("pad_to_sec", None)
    train_ds = MILBagDataset(train_df, window_sec=w_sec, stride_sec=s_sec, pad_to_sec=pad_to_sec)
    val_ds = MILBagDataset(val_df, window_sec=w_sec, stride_sec=s_sec, pad_to_sec=pad_to_sec)

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_mil_model(cfg).to(device)
    use_weighted_sum = cfg.get("layer_aggregation", "last") == "weighted_sum"

    trainable_params = list(model.mil_head.parameters())
    if model.backbone.layer_weights is not None:
        trainable_params.append(model.backbone.layer_weights)
    n_trainable = sum(p.numel() for p in trainable_params)
    print(
        f"Backbone: {cfg['backbone']} | head={cfg.get('head', 'gated_abmil')} | "
        f"layer_aggregation={cfg.get('layer_aggregation', 'last')} | "
        f"trainable params: {n_trainable:,}",
        flush=True,
    )

    # ── Pre-compute backbone embeddings ────────────────────────────────────
    if use_weighted_sum:
        print("Pre-computing per-layer train embeddings ...", flush=True)
        train_emb_cache = _precompute_embeddings_per_layer(model, train_ds, device)
        print("Pre-computing per-layer val embeddings ...", flush=True)
        val_emb_cache = _precompute_embeddings_per_layer(model, val_ds, device)
    else:
        print("Pre-computing train embeddings ...", flush=True)
        train_emb_cache = _precompute_embeddings_last_layer(model, train_ds, device)
        print("Pre-computing val embeddings ...", flush=True)
        val_emb_cache = _precompute_embeddings_last_layer(model, val_ds, device)

    train_records = []
    for _, row in train_df.iterrows():
        rec = {
            "label": float(row["label"]),
            "audio_path": str(row["audio_path"]),
            "child_id": str(row["child_id"]),
            "timepoint_norm": str(row["timepoint_norm"]),
            "emb": train_emb_cache[str(row["audio_path"])],
        }
        train_records.append(rec)
    val_records = []
    for _, row in val_df.iterrows():
        rec = {
            "label": float(row["label"]),
            "audio_path": str(row["audio_path"]),
            "child_id": str(row["child_id"]),
            "timepoint_norm": str(row["timepoint_norm"]),
            "emb": val_emb_cache[str(row["audio_path"])],
        }
        val_records.append(rec)

    # ── Optional: prototype cache for TS-MIL (US4) ─────────────────────────
    proto_cache_path = cfg.get("prototype_cache")
    if proto_cache_path:
        proto_cache = _load_prototype_cache(proto_cache_path)
        train_records, train_missing = _attach_prototypes(train_records, proto_cache, "train")
        val_records, val_missing = _attach_prototypes(val_records, proto_cache, "val")
        save_json(
            {"train": train_missing, "val": val_missing},
            os.path.join(result_dir, "missing_prototypes.json"),
        )

    optimizer = torch.optim.Adam(trainable_params, lr=cfg.get("lr", 1e-3))

    pos_weight = cfg.get("pos_weight")
    if pos_weight is not None:
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_f1 = -1.0
    best_ckpt_path = os.path.join(result_dir, "best_checkpoint.pt")
    patience = cfg.get("patience", 5)
    no_improve = 0
    history = []
    batch_size = cfg.get("batch_size", 8)

    for epoch in range(1, cfg.get("epochs", 20) + 1):
        model.mil_head.train()
        if model.backbone.layer_weights is not None:
            model.backbone.eval()  # transformer modules stay in eval; layer_weights param is independent
        indices = torch.randperm(len(train_records)).tolist()
        train_bce_losses = []
        train_div_losses = []

        for start in range(0, len(indices), batch_size):
            batch_recs = [train_records[i] for i in indices[start:start + batch_size]]
            batch_labels = torch.tensor(
                [r["label"] for r in batch_recs], dtype=torch.float32, device=device
            )
            batch_logits = []
            batch_div = []
            for rec in batch_recs:
                emb = _make_instance_embeddings(
                    rec, use_weighted_sum, model.backbone.layer_weights, device
                )
                proto = rec.get("prototype")
                if proto is not None:
                    proto = proto.to(device)
                logit, _attn, div_loss = _head_forward(model, emb, proto)
                batch_logits.append(logit)
                batch_div.append(div_loss)

            logits = torch.stack(batch_logits)
            bce = criterion(logits, batch_labels)
            div = torch.stack(batch_div).mean()
            loss = bce + div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_bce_losses.append(bce.item())
            train_div_losses.append(float(div.item()))

        # ── Val pass ──
        model.mil_head.eval()
        val_scores, val_labels_list, val_losses = [], [], []
        with torch.no_grad():
            for rec in val_records:
                emb = _make_instance_embeddings(
                    rec, use_weighted_sum, model.backbone.layer_weights, device
                )
                proto = rec.get("prototype")
                if proto is not None:
                    proto = proto.to(device)
                logit, _attn, _div = _head_forward(model, emb, proto)
                score = float(torch.sigmoid(logit).item())
                val_scores.append(score)
                val_labels_list.append(int(rec["label"]))
                val_losses.append(
                    criterion(logit, torch.tensor(rec["label"], device=device)).item()
                )

        val_metrics = compute_metrics(val_labels_list, val_scores, threshold=0.5)
        mean_train_bce = float(np.mean(train_bce_losses))
        mean_train_div = float(np.mean(train_div_losses))
        mean_val_loss = float(np.mean(val_losses))

        print(
            f"Epoch {epoch:3d} | train_bce={mean_train_bce:.4f} train_div={mean_train_div:.4f} "
            f"val_loss={mean_val_loss:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f}",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": mean_train_bce + mean_train_div,
            "loss_bce": mean_train_bce,
            "loss_div": mean_train_div,
            "val_loss": mean_val_loss,
            "val_f1": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({"model_state": model.state_dict(), "cfg": cfg}, best_ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})", flush=True)
                break

    # ── Post-training: threshold tuning + save outputs ─────────────────────
    print("Loading best checkpoint for val predictions …", flush=True)
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.mil_head.eval()

    val_scores, val_labels_list, val_meta = [], [], []
    with torch.no_grad():
        for rec in val_records:
            emb = _make_instance_embeddings(
                rec, use_weighted_sum, model.backbone.layer_weights, device
            )
            proto = rec.get("prototype")
            if proto is not None:
                proto = proto.to(device)
            logit, _attn, _div = _head_forward(model, emb, proto)
            val_scores.append(float(torch.sigmoid(logit).item()))
            val_labels_list.append(int(rec["label"]))
            val_meta.append({
                "audio_path": rec["audio_path"],
                "child_id": rec["child_id"],
                "timepoint_norm": rec["timepoint_norm"],
            })

    threshold = tune_threshold(val_labels_list, val_scores)
    val_metrics_tuned = compute_metrics(val_labels_list, val_scores, threshold=threshold)
    val_metrics_tuned["threshold"] = threshold

    val_preds_df = pd.DataFrame([
        {**meta, "label": lbl, "score": sc, "prediction": int(sc >= threshold)}
        for meta, lbl, sc in zip(val_meta, val_labels_list, val_scores)
    ])

    # Persist final layer-weights softmax for inspection (US1 FR-004)
    layer_weights_softmax = model.backbone.layer_weights_softmax()
    if layer_weights_softmax is not None:
        save_json(
            {str(i): w for i, w in enumerate(layer_weights_softmax)},
            os.path.join(result_dir, "layer_weights.json"),
        )

    save_json(cfg, os.path.join(result_dir, "config.json"))
    save_json(val_metrics_tuned, os.path.join(result_dir, "val_metrics_tuned.json"))
    save_csv(pd.DataFrame(history), os.path.join(result_dir, "training_history.csv"))
    save_csv(val_preds_df, os.path.join(result_dir, "val_predictions.csv"))
    tp_df = per_timepoint_metrics(val_preds_df)
    save_csv(tp_df, os.path.join(result_dir, "val_metrics_by_timepoint.csv"))

    print("\n=== Training complete ===", flush=True)
    print(f"  Val F1 (tuned, threshold={threshold:.2f}): {val_metrics_tuned['f1']:.4f}", flush=True)
    print(f"  Val AUROC: {val_metrics_tuned['auroc']:.4f}", flush=True)
    print(f"  Results in: {result_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MIL child presence model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = parse_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
