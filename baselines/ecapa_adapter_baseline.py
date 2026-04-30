"""
ECAPA Adapter Triplet Baseline (US7, spec-013) — Tier 4.

Fine-tunes a lightweight 2-layer adapter (192 → 64 → 192) on top of frozen
ECAPA-TDNN using triplet loss on BabAR KCHI child speech segments.

Hypothesis: General adult-speaker ECAPA embeddings may not optimally separate
child identities. Triplet fine-tuning with child-specific pairs could tighten
within-child clusters and push apart between-child embeddings.

Training:
  - Anchors: KCHI segments for each child from positive training clips
  - Positives: other segments from the SAME child
  - Negatives: segments from a DIFFERENT child (random negative sampling)
  - Loss: BatchHard triplet margin loss (margin=0.3)

Evaluation:
  - Rebuild ECAPA prototypes using adapter-transformed embeddings
  - Score ALL test segments with adapted embeddings (BabAR RTTM segments)
  - Compare enrollment AUROC to unadapted BabAR enrollment (0.820)

Usage:
    python baselines/ecapa_adapter_baseline.py --split val
    python baselines/ecapa_adapter_baseline.py --split test  # requires val first
    python baselines/ecapa_adapter_baseline.py --split val --dry-run
"""

import argparse
import hashlib
import json
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

SPLITS_DIR = _REPO / "whisper-modeling/seen_child_splits"
RESULTS_BASE = _REPO / "baselines/ecapa_adapter_baseline_runs"
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
BABAR_RTTM_DIR = _REPO / "babar/babar_output/rttm"
SR = 16000
ECAPA_DIM = 192
MIN_SEG_DUR = 0.5


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def load_audio_mono(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.squeeze(0)


def parse_kchi_segments(audio_path: str) -> List[dict]:
    stem = Path(audio_path).stem
    rttm = BABAR_RTTM_DIR / f"{stem}.rttm"
    if not rttm.exists():
        return []
    segs = []
    with open(rttm) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER" and parts[7] == "KCHI":
                start, dur = float(parts[3]), float(parts[4])
                if dur >= MIN_SEG_DUR:
                    segs.append({"start": start, "end": start + dur, "dur": dur})
    return segs


def parse_all_segments(audio_path: str) -> List[dict]:
    stem = Path(audio_path).stem
    rttm = BABAR_RTTM_DIR / f"{stem}.rttm"
    if not rttm.exists():
        return []
    segs = []
    with open(rttm) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start, dur = float(parts[3]), float(parts[4])
                if dur >= MIN_SEG_DUR:
                    segs.append({"start": start, "end": start + dur, "dur": dur,
                                 "label": parts[7]})
    return segs


# ---------------------------------------------------------------------------
# Adapter model
# ---------------------------------------------------------------------------

class ECAPAAdapter(nn.Module):
    """Lightweight post-ECAPA adapter: 192 → 64 → 192 with residual."""

    def __init__(self, dim: int = ECAPA_DIM, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)  # init to identity (residual = 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 192) — returns L2-normalized adapted embeddings."""
        adapted = x + self.up(torch.relu(self.down(x)))
        return nn.functional.normalize(adapted, dim=-1)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_segment_embs(
    audio_path: str,
    segments: List[dict],
    ecapa,
    adapter: Optional[ECAPAAdapter],
    device: str,
) -> List[np.ndarray]:
    """Extract (adapted) ECAPA embeddings for a list of segments."""
    if not segments:
        return []
    try:
        wav = load_audio_mono(audio_path)
    except Exception:
        return []
    embs = []
    for seg in segments:
        s, e = int(seg["start"] * SR), int(seg["end"] * SR)
        chunk = wav[s:e]
        if len(chunk) < int(MIN_SEG_DUR * SR):
            continue
        try:
            chunk_t = chunk.unsqueeze(0).to(device)
            with torch.no_grad():
                raw_emb = ecapa.encode_batch(chunk_t).squeeze()  # (192,)
                if adapter is not None:
                    raw_emb = adapter(raw_emb.unsqueeze(0)).squeeze()
            embs.append(raw_emb.cpu().numpy())
        except Exception:
            continue
    return embs


# ---------------------------------------------------------------------------
# Triplet dataset builder
# ---------------------------------------------------------------------------

def build_triplet_pool(
    train_df: pd.DataFrame,
    ecapa,
    device: str,
) -> Dict[str, List[np.ndarray]]:
    """Build {proto_key: [ECAPA emb, ...]} from KCHI training segments."""
    pool: Dict[str, List[np.ndarray]] = {}
    pos = train_df[train_df["label"] == 1]

    for row in pos.itertuples():
        key = f"{row.child_id}__{row.timepoint_norm}"
        segs = parse_kchi_segments(row.audio_path)
        embs = extract_segment_embs(row.audio_path, segs, ecapa, None, device)
        if embs:
            pool.setdefault(key, []).extend(embs)

    for key in list(pool.keys()):
        if len(pool[key]) < 2:
            del pool[key]

    keys = sorted(pool.keys())
    print(f"Triplet pool: {len(keys)} children, "
          f"median {int(np.median([len(pool[k]) for k in keys]))} segs/child", flush=True)
    return pool


def sample_triplets(
    pool: Dict[str, List[np.ndarray]],
    n_triplets: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (anchor, positive, negative) triplets from pool."""
    keys = list(pool.keys())
    anchors, positives, negatives = [], [], []
    for _ in range(n_triplets):
        anchor_key = rng.choice(keys)
        neg_key = rng.choice([k for k in keys if k != anchor_key])
        if len(pool[anchor_key]) < 2:
            continue
        a, p = rng.sample(pool[anchor_key], 2)
        n = rng.choice(pool[neg_key])
        anchors.append(a)
        positives.append(p)
        negatives.append(n)
    return (np.stack(anchors).astype(np.float32),
            np.stack(positives).astype(np.float32),
            np.stack(negatives).astype(np.float32))


# ---------------------------------------------------------------------------
# Triplet training
# ---------------------------------------------------------------------------

def train_adapter(
    pool: Dict[str, List[np.ndarray]],
    device: torch.device,
    margin: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 30,
    n_triplets_per_epoch: int = 1024,
    seed: int = 42,
) -> ECAPAAdapter:
    rng = random.Random(seed)
    torch.manual_seed(seed)

    adapter = ECAPAAdapter().to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    for epoch in range(1, epochs + 1):
        a_np, p_np, n_np = sample_triplets(pool, n_triplets_per_epoch, rng)
        a_t = torch.from_numpy(a_np).to(device)
        p_t = torch.from_numpy(p_np).to(device)
        n_t = torch.from_numpy(n_np).to(device)

        adapter.train()
        optimizer.zero_grad()
        a_out = adapter(a_t)
        p_out = adapter(p_t)
        n_out = adapter(n_t)
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            adapter.eval()
            with torch.no_grad():
                a_out = adapter(a_t)
                p_out = adapter(p_t)
                n_out = adapter(n_t)
                # Fraction of triplets where d(a,p) < d(a,n)
                dp = torch.norm(a_out - p_out, dim=1)
                dn = torch.norm(a_out - n_out, dim=1)
                acc = (dp < dn).float().mean().item()
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}  triplet_acc={acc:.3f}", flush=True)

    return adapter


# ---------------------------------------------------------------------------
# Prototype building with adapter
# ---------------------------------------------------------------------------

def build_adapted_prototypes(
    train_df: pd.DataFrame,
    ecapa,
    adapter: ECAPAAdapter,
    device: str,
) -> Dict[str, np.ndarray]:
    prototypes: Dict[str, np.ndarray] = {}
    pos = train_df[train_df["label"] == 1]
    groups = list(pos.groupby(["child_id", "timepoint_norm"]))

    for (cid, tp), sub in groups:
        key = f"{cid}__{tp}"
        all_embs, all_durs = [], []
        for row in sub.itertuples():
            segs = parse_kchi_segments(row.audio_path)
            embs = extract_segment_embs(row.audio_path, segs, ecapa, adapter, device)
            for emb, seg in zip(embs, segs):
                all_embs.append(emb)
                all_durs.append(seg["dur"])
        if not all_embs:
            continue
        proto = np.average(np.stack(all_embs), axis=0,
                           weights=np.array(all_durs))
        prototypes[key] = l2_normalize(proto)

    print(f"Adapted prototypes: {len(prototypes)}/{len(groups)}", flush=True)
    return prototypes


# ---------------------------------------------------------------------------
# Clip scoring (enrollment)
# ---------------------------------------------------------------------------

def score_clip(
    audio_path: str,
    proto: np.ndarray,
    ecapa,
    adapter: Optional[ECAPAAdapter],
    device: str,
) -> float:
    segs = parse_all_segments(audio_path)
    if not segs:
        return 0.0
    embs = extract_segment_embs(audio_path, segs, ecapa, adapter, device)
    if not embs:
        return 0.0
    sims = [float(np.dot(e, proto)) for e in embs]
    durs = [s["dur"] for s in segs[:len(embs)]]
    return float(sum(s * d for s, d in zip(sims, durs)) / max(sum(durs), 1e-8))


def run_split(
    df: pd.DataFrame,
    prototypes: Dict[str, np.ndarray],
    ecapa,
    adapter: Optional[ECAPAAdapter],
    device: str,
    dry_run: bool = False,
    max_clips: Optional[int] = None,
) -> pd.DataFrame:
    records = []
    total = min(len(df), max_clips) if max_clips else len(df)
    n_missing = 0

    for i, row in enumerate(df.itertuples()):
        if max_clips and i >= max_clips:
            break
        key = f"{row.child_id}__{row.timepoint_norm}"
        proto = prototypes.get(key)
        if dry_run:
            prob = 0.5
        elif proto is None:
            prob = 0.0
            n_missing += 1
        else:
            prob = score_clip(row.audio_path, proto, ecapa, adapter, device)
        records.append({
            "audio_path": row.audio_path, "label": int(row.label),
            "child_id": row.child_id, "timepoint_norm": row.timepoint_norm,
            "prob": prob,
        })
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}]", flush=True)

    if n_missing:
        print(f"  WARNING: {n_missing} clips had no prototype (scored 0.0)", flush=True)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-triplets", type=int, default=1024)
    p.add_argument("--output-dir", default=str(RESULTS_BASE))
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if not args.dry_run else "cpu"

    print(f"Split={args.split}  device={device}", flush=True)

    # Load ECAPA
    from speechbrain.inference.speaker import EncoderClassifier
    print("Loading ECAPA ...", flush=True)
    ecapa = EncoderClassifier.from_hparams(
        source=ECAPA_SOURCE, run_opts={"device": device})
    ecapa.eval()
    print("ECAPA loaded.", flush=True)

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    if "audio_exists" in train_df.columns:
        train_df = train_df[train_df["audio_exists"].astype(bool)]

    if args.dry_run:
        train_df = train_df.head(20)

    # Build triplet pool and train adapter
    print("Building triplet pool ...", flush=True)
    pool = build_triplet_pool(train_df, ecapa, device)

    print(f"Training adapter ({args.epochs} epochs, {args.n_triplets} triplets/epoch) ...", flush=True)
    adapter = train_adapter(
        pool, torch.device(device),
        margin=args.margin, lr=args.lr,
        epochs=args.epochs, n_triplets_per_epoch=args.n_triplets,
        seed=args.seed,
    )

    # Build adapted prototypes
    print("Building adapted prototypes ...", flush=True)
    prototypes = build_adapted_prototypes(train_df, ecapa, adapter, device)

    if args.split == "test":
        val_path = out_dir / "val_metrics_tuned.json"
        if not val_path.exists():
            print(f"ERROR: {val_path} not found. Run --split val first.", file=sys.stderr)
            sys.exit(2)
        with open(val_path) as f:
            threshold = float(json.load(f)["threshold"])
        print(f"Loaded threshold={threshold:.4f} from val", flush=True)

        meta_df = pd.read_csv(SPLITS_DIR / "test.csv")
        if "audio_exists" in meta_df.columns:
            meta_df = meta_df[meta_df["audio_exists"].astype(bool)]
        print(f"Test clips: {len(meta_df)}", flush=True)

        preds = run_split(meta_df, prototypes, ecapa, adapter, device,
                          args.dry_run, args.max_clips)
        metrics = compute_metrics(preds["label"].values, preds["prob"].values, threshold)
        metrics["threshold"] = threshold
        save_json(metrics, str(out_dir / "test_metrics_tuned.json"))
        save_csv(preds, str(out_dir / "test_predictions.csv"))

    else:
        meta_df = pd.read_csv(SPLITS_DIR / "val.csv")
        if "audio_exists" in meta_df.columns:
            meta_df = meta_df[meta_df["audio_exists"].astype(bool)]
        print(f"Val clips: {len(meta_df)}", flush=True)

        preds = run_split(meta_df, prototypes, ecapa, adapter, device,
                          args.dry_run, args.max_clips)
        threshold = tune_threshold(preds["label"].values, preds["prob"].values)
        metrics = compute_metrics(preds["label"].values, preds["prob"].values, threshold)
        metrics["threshold"] = threshold
        save_json(metrics, str(out_dir / "val_metrics_tuned.json"))
        save_csv(preds, str(out_dir / "val_predictions.csv"))

        # Per-timepoint
        rows = []
        for tp, grp in preds.groupby("timepoint_norm"):
            m = compute_metrics(grp["label"].values, grp["prob"].values, threshold)
            rows.append({"timepoint_norm": tp, **m, "n": len(grp)})
        save_csv(pd.DataFrame(rows), str(out_dir / "val_metrics_by_timepoint.csv"))

    print(f"\n{args.split.capitalize()} metrics (threshold={threshold:.4f}):", flush=True)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}", flush=True)

    # Save checkpoint and config
    torch.save(adapter.state_dict(), str(out_dir / "adapter_checkpoint.pt"))
    save_json({
        "ecapa_source": ECAPA_SOURCE,
        "margin": args.margin, "lr": args.lr, "epochs": args.epochs,
        "n_triplets_per_epoch": args.n_triplets, "seed": args.seed,
        "threshold": float(threshold),
    }, str(out_dir / "config.json"))
    print(f"Done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
