"""
Conditioned Segment MIL (US6, spec-013) — Tier 4 architectural baseline.

Each ECAPA segment embedding is conditioned on the child's ECAPA prototype:
  instance_feat = [seg_emb (192), proto (192), seg_emb − proto (192)]  → 576-dim
A GatedABMIL head pools conditioned instances → clip-level child-presence score.

This forces the model to learn "is this segment from THIS specific child" rather than
"is this segment child-like in general." Contrasts with unconditioned seg_mil (US3/4).

Protocol:
  - Enrollment: BabAR KCHI training segments → duration-weighted ECAPA prototype per child
  - Instances: ALL BabAR RTTM segments (KCHI + ADT) for each clip
  - Embedding: ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb), 192-dim
  - MIL head: GatedABMIL over 576-dim conditioned embeddings
  - Evaluation: seen-child split only (requires per-child prototypes)

Usage:
    python mil/seg_conditioned_train.py --split seen_child
    # Dry run (no training, 10 clips):
    python mil/seg_conditioned_train.py --dry-run --max-clips 10
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from mil.mil_utils import compute_metrics, tune_threshold, save_json, save_csv
from mil.seg_model import GatedAttnAgg  # reuse existing aggregator

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
BABAR_RTTM_DIR = _REPO / "babar/babar_output/rttm"
SPLITS_DIR = _REPO / "whisper-modeling/seen_child_splits"
RESULTS_DIR = _REPO / "mil/mil_results/seg_conditioned_mil"
CACHE_FILE = _REPO / "mil/mil_results/seg_conditioned_mil/ecapa_seg_cache.npz"

SR = 16000
ECAPA_DIM = 192
COND_DIM = ECAPA_DIM * 3  # [seg, proto, seg-proto]
MIN_SEG_DUR = 0.5   # seconds — minimum segment length
K_MAX = 64          # max segments per clip


# ---------------------------------------------------------------------------
# ECAPA helpers
# ---------------------------------------------------------------------------

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def load_ecapa(device: str):
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(
        source=ECAPA_SOURCE,
        run_opts={"device": device},
    )
    ec.eval()
    return ec


def load_audio_mono(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.squeeze(0)  # (T,)


def parse_rttm_all_segments(audio_path: str) -> List[Dict]:
    """Return ALL segments (KCHI + ADT) from BabAR RTTM."""
    stem = Path(audio_path).stem
    rttm_path = BABAR_RTTM_DIR / f"{stem}.rttm"
    if not rttm_path.exists():
        return []
    segs = []
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            start, dur, label = float(parts[3]), float(parts[4]), parts[7]
            if dur >= MIN_SEG_DUR:
                segs.append({"start": start, "end": start + dur, "dur": dur, "label": label})
    return segs


def parse_kchi_segments(audio_path: str) -> List[Dict]:
    """Return KCHI-only segments from BabAR RTTM."""
    return [s for s in parse_rttm_all_segments(audio_path) if s["label"] == "KCHI"]


def embed_segment(wav: torch.Tensor, start: float, end: float, ecapa, device: str) -> Optional[np.ndarray]:
    s = int(start * SR)
    e = int(end * SR)
    chunk = wav[s:e]
    if len(chunk) < int(MIN_SEG_DUR * SR):
        return None
    try:
        emb = ecapa.encode_batch(chunk.unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()
        return l2_normalize(emb)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prototype building
# ---------------------------------------------------------------------------

def build_prototypes(train_df: pd.DataFrame, ecapa, device: str) -> Dict[str, np.ndarray]:
    """Build per-(child_id, timepoint_norm) ECAPA prototypes from KCHI training segments."""
    prototypes: Dict[str, np.ndarray] = {}
    pos = train_df[train_df["label"] == 1].copy()
    groups = list(pos.groupby(["child_id", "timepoint_norm"]))
    print(f"Building prototypes for {len(groups)} (child, timepoint) pairs "
          f"from {len(pos)} positive training clips ...", flush=True)
    n_missing = 0

    for idx, ((cid, tp), sub) in enumerate(groups):
        key = f"{cid}__{tp}"
        all_embs, all_durs = [], []
        for row in sub.itertuples():
            segs = parse_kchi_segments(row.audio_path)
            if not segs:
                n_missing += 1
                continue
            try:
                wav = load_audio_mono(row.audio_path)
            except Exception:
                continue
            for seg in segs:
                emb = embed_segment(wav, seg["start"], seg["end"], ecapa, device)
                if emb is not None:
                    all_embs.append(emb)
                    all_durs.append(seg["dur"])
        if not all_embs:
            continue
        embs = np.stack(all_embs)
        weights = np.array(all_durs)
        proto = np.average(embs, axis=0, weights=weights)
        prototypes[key] = l2_normalize(proto)
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(groups)}] {len(prototypes)} prototypes built", flush=True)

    print(f"Prototypes built: {len(prototypes)}/{len(groups)} "
          f"({n_missing} clips had no BabAR RTTM)", flush=True)
    return prototypes


# ---------------------------------------------------------------------------
# Bag builder (conditioned embeddings)
# ---------------------------------------------------------------------------

def build_conditioned_bag(
    audio_path: str,
    proto: np.ndarray,
    ecapa,
    device: str,
) -> Optional[np.ndarray]:
    """Return [K x COND_DIM] conditioned bag, or None if no segments."""
    segs = parse_rttm_all_segments(audio_path)
    if not segs:
        return None
    try:
        wav = load_audio_mono(audio_path)
    except Exception:
        return None

    rows = []
    for seg in segs[:K_MAX]:
        emb = embed_segment(wav, seg["start"], seg["end"], ecapa, device)
        if emb is None:
            continue
        cond = np.concatenate([emb, proto, emb - proto])  # 576-dim
        rows.append(cond)

    if not rows:
        return None
    return np.stack(rows)  # (K, 576)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CondBagDataset(torch.utils.data.Dataset):
    """In-memory dataset of conditioned bags (after pre-computation)."""

    def __init__(
        self,
        bags: List[Optional[np.ndarray]],
        labels: List[int],
        meta: List[Dict],
    ):
        self.bags = bags
        self.labels = labels
        self.meta = meta

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int):
        bag_np = self.bags[idx]
        label = self.labels[idx]
        if bag_np is None or len(bag_np) == 0:
            bag_t = torch.zeros(K_MAX, COND_DIM)
            mask_t = torch.zeros(K_MAX, dtype=torch.bool)
        else:
            k = min(len(bag_np), K_MAX)
            bag_t = torch.zeros(K_MAX, COND_DIM)
            bag_t[:k] = torch.from_numpy(bag_np[:k].astype(np.float32))
            mask_t = torch.zeros(K_MAX, dtype=torch.bool)
            mask_t[:k] = True
        return bag_t, mask_t, torch.tensor(float(label)), self.meta[idx]


def precompute_bags(
    df: pd.DataFrame,
    prototypes: Dict[str, np.ndarray],
    ecapa,
    device: str,
    split_name: str,
    max_clips: Optional[int] = None,
) -> Tuple[List[Optional[np.ndarray]], List[int], List[Dict]]:
    """Pre-compute conditioned bags for all clips in df."""
    bags, labels, metas = [], [], []
    total = min(len(df), max_clips) if max_clips else len(df)
    n_missing_proto, n_missing_rttm = 0, 0

    for i, row in enumerate(df.itertuples()):
        if max_clips and i >= max_clips:
            break
        key = f"{row.child_id}__{row.timepoint_norm}"
        proto = prototypes.get(key)
        if proto is None:
            n_missing_proto += 1
            bags.append(None)
        else:
            bag = build_conditioned_bag(row.audio_path, proto, ecapa, device)
            if bag is None:
                n_missing_rttm += 1
            bags.append(bag)
        labels.append(int(row.label))
        metas.append({
            "audio_path": row.audio_path,
            "child_id": row.child_id,
            "timepoint_norm": row.timepoint_norm,
        })
        if (i + 1) % 100 == 0:
            print(f"  [{split_name}] {i+1}/{total} clips embedded", flush=True)

    if n_missing_proto:
        print(f"  WARNING: {n_missing_proto} clips had no prototype", flush=True)
    if n_missing_rttm:
        print(f"  WARNING: {n_missing_rttm} clips had no BabAR RTTM", flush=True)
    return bags, labels, metas


# ---------------------------------------------------------------------------
# MIL model (GatedABMIL over 576-dim conditioned embeddings)
# ---------------------------------------------------------------------------

class CondMILModel(nn.Module):
    """Gated attention MIL over conditioned ECAPA embeddings."""

    def __init__(self, cond_dim: int = COND_DIM, attn_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn_V = nn.Linear(512, attn_dim)
        self.attn_U = nn.Linear(512, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.head = nn.Linear(512, 1)

    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # bag: (K, COND_DIM), mask: (K,) bool
        k = mask.sum().item()
        if k == 0:
            return torch.tensor(0.0, device=bag.device), None
        h = self.proj(bag[:int(k)])  # (k, 512)
        v = torch.tanh(self.attn_V(h))
        u = torch.sigmoid(self.attn_U(h))
        e = self.attn_w(v * u).squeeze(-1)  # (k,)
        a = torch.softmax(e, dim=0)  # (k,)
        z = (a.unsqueeze(-1) * h).sum(0)  # (512,)
        logit = self.head(z).squeeze()
        return logit, a


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    train_ds: CondBagDataset,
    val_ds: CondBagDataset,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 20,
    patience: int = 5,
    batch_size: int = 8,
    dropout: float = 0.25,
) -> CondMILModel:
    model = CondMILModel(cond_dim=COND_DIM, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        indices = list(range(len(train_ds)))
        random.shuffle(indices)
        total_loss, n_batches = 0.0, 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            optimizer.zero_grad()
            logits, batch_labels = [], []
            for i in batch_idx:
                bag, mask, label, _ = train_ds[i]
                logit, _ = model(bag.to(device), mask.to(device))
                logits.append(logit)
                batch_labels.append(float(label))
            logits_t = torch.stack(logits)
            labels_t = torch.tensor(batch_labels, device=device)
            loss = criterion(logits_t, labels_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_scores, val_labels = run_inference(model, val_ds, device)
        val_m = compute_metrics(val_labels, val_scores)
        val_auroc = val_m.get("auroc", 0.0) or 0.0
        print(f"  epoch {epoch:2d}  loss={total_loss / max(n_batches, 1):.4f}  "
              f"val_auroc={val_auroc:.4f}", flush=True)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def run_inference(
    model: CondMILModel,
    ds: CondBagDataset,
    device: torch.device,
) -> Tuple[List[float], List[int]]:
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for idx in range(len(ds)):
            bag, mask, label, _ = ds[idx]
            logit, _ = model(bag.to(device), mask.to(device))
            prob = float(torch.sigmoid(logit).item())
            scores.append(prob)
            labels.append(int(label.item()))
    return scores, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--output-dir", default=str(RESULTS_DIR))
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device if not args.dry_run else "cpu"
    device = torch.device(device_str)

    # Load data
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test.csv")
    for df in [train_df, val_df, test_df]:
        if "audio_exists" in df.columns:
            df.drop(df[~df["audio_exists"].astype(bool)].index, inplace=True)

    if args.dry_run:
        print("DRY RUN: truncating to 10 clips per split", flush=True)
        train_df = train_df.head(10)
        val_df   = val_df.head(10)
        test_df  = test_df.head(10)

    print(f"Device: {device_str} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Load ECAPA
    print("Loading ECAPA ...", flush=True)
    ecapa = load_ecapa(device_str)
    print("ECAPA loaded.", flush=True)

    # Build prototypes
    prototypes = build_prototypes(train_df, ecapa, device_str)

    # Pre-compute bags
    print("Pre-computing train bags ...", flush=True)
    tr_bags, tr_labels, tr_meta = precompute_bags(
        train_df, prototypes, ecapa, device_str, "train", args.max_clips)
    print("Pre-computing val bags ...", flush=True)
    va_bags, va_labels, va_meta = precompute_bags(
        val_df, prototypes, ecapa, device_str, "val", args.max_clips)
    print("Pre-computing test bags ...", flush=True)
    te_bags, te_labels, te_meta = precompute_bags(
        test_df, prototypes, ecapa, device_str, "test", args.max_clips)

    train_ds = CondBagDataset(tr_bags, tr_labels, tr_meta)
    val_ds   = CondBagDataset(va_bags, va_labels, va_meta)
    test_ds  = CondBagDataset(te_bags, te_labels, te_meta)

    if args.dry_run:
        print("DRY RUN complete — no training.", flush=True)
        return

    # Train
    print("Training CondMIL ...", flush=True)
    model = train_model(
        train_ds, val_ds, device,
        lr=args.lr, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, dropout=args.dropout,
    )

    # Evaluate
    val_scores, val_labels = run_inference(model, val_ds, device)
    threshold = tune_threshold(val_labels, val_scores)
    val_metrics = compute_metrics(val_labels, val_scores, threshold)
    val_metrics["threshold"] = threshold

    test_scores, test_labels = run_inference(model, test_ds, device)
    test_metrics = compute_metrics(test_labels, test_scores, threshold)
    test_metrics["threshold"] = threshold

    print(f"\nVal  F1={val_metrics['f1']:.4f}  AUROC={val_metrics['auroc']:.4f}", flush=True)
    print(f"Test F1={test_metrics['f1']:.4f}  AUROC={test_metrics['auroc']:.4f}  "
          f"AUPRC={test_metrics['auprc']:.4f}", flush=True)

    # Save
    torch.save(model.state_dict(), str(out_dir / "best_checkpoint.pt"))
    save_json(val_metrics, str(out_dir / "val_metrics_tuned.json"))
    save_json(test_metrics, str(out_dir / "test_metrics_tuned.json"))

    # Prediction CSVs
    def make_pred_df(df, scores, labels, thr):
        rows = []
        for i, row in enumerate(df.itertuples()):
            rows.append({
                "audio_path": row.audio_path,
                "child_id": row.child_id,
                "timepoint_norm": row.timepoint_norm,
                "label": labels[i],
                "prob": scores[i],
                "pred": int(scores[i] >= thr),
            })
        return pd.DataFrame(rows)

    save_csv(make_pred_df(val_df, val_scores, val_labels, threshold),
             str(out_dir / "val_predictions.csv"))
    save_csv(make_pred_df(test_df, test_scores, test_labels, threshold),
             str(out_dir / "test_predictions.csv"))

    # Per-timepoint
    def per_tp(df_src, scores, labels, thr):
        rows_tp = []
        for tp, grp in pd.DataFrame({
            "tp": [r.timepoint_norm for r in df_src.itertuples()],
            "label": labels, "prob": scores
        }).groupby("tp"):
            m = compute_metrics(grp["label"].values, grp["prob"].values, thr)
            rows_tp.append({"timepoint_norm": tp, **m, "n": len(grp)})
        return pd.DataFrame(rows_tp)

    save_csv(per_tp(test_df, test_scores, test_labels, threshold),
             str(out_dir / "test_metrics_by_timepoint.csv"))

    save_json({
        "variant_name": "seg_conditioned_mil",
        "ecapa_source": ECAPA_SOURCE,
        "babar_rttm_dir": str(BABAR_RTTM_DIR),
        "cond_dim": COND_DIM,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "seed": args.seed,
        "threshold": float(threshold),
        "split_dir": str(SPLITS_DIR),
    }, str(out_dir / "config.json"))

    print(f"Done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
