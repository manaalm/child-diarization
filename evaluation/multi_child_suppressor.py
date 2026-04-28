"""Multi-child FP suppressor (spec-012 US3).

Trains a LR classifier on n_children>=2 train clips using frozen
WavLM-Base+ mean-pool embeddings. At test time merges with best_audio_mil
approximation for n_children>=2 clips only; single-child clips pass through
unchanged.

Usage:
  python evaluation/multi_child_suppressor.py [--dry-run]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mil.mil_model import BackboneExtractor
from evaluation.metadata_router import (
    BASELINE_AUROC,
    BASELINE_F1,
    MASTER_CSV,
    SEED,
    compute_metrics,
    load_metadata,
    load_split,
    load_system_scores,
    save_results,
    tune_threshold,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_REPO, "mil/mil_results/multi_child_suppressor")
BACKBONE = "microsoft/wavlm-base-plus"
SR = 16000
MAX_SAMPLES = SR * 30  # 30-second clip cap
EMB_DIM = 768


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def _best_audio_mil(df: pd.DataFrame) -> np.ndarray:
    cols = ["babar_prob", "vtc_prob", "wavlm_mil_prob", "whisper_mil_prob"]
    avail = [c for c in cols if c in df.columns]
    return df[avail].mean(axis=1).to_numpy(dtype=float)


def _load_train_meta() -> pd.DataFrame:
    """Load and parse train-split metadata rows only."""
    df = pd.read_csv(MASTER_CSV)

    def _to_int(v, default):
        try:
            return int(str(v).strip().split("+")[0])
        except Exception:
            return default

    df["n_children_int"] = df["#_children"].apply(lambda v: _to_int(v, 1))
    keep = ["audio_path", "split", "label", "timepoint_norm", "n_children_int"]
    return df[df["split"] == "train"][keep].reset_index(drop=True)


# ── Embedding cache ──────────────────────────────────────────────────────────

def embed_clip(audio_path: str, backbone, device: torch.device) -> np.ndarray:
    """Load clip, run through frozen WavLM backbone, mean-pool → (768,)."""
    import torchaudio

    wav, sr = torchaudio.load(audio_path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav[:, :MAX_SAMPLES].to(device)
    frame_embs = backbone(wav.unsqueeze(0))  # (1, T_frames, D)
    return frame_embs.squeeze(0).mean(0).cpu().numpy()


def build_embedding_cache(df: pd.DataFrame, backbone, device: torch.device,
                          cache_path: str) -> dict:
    """Cache embeddings keyed by md5(audio_path). Returns dict hash→emb."""
    existing = {}
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        existing = {k: data[k] for k in data.files}

    paths = df["audio_path"].tolist()
    missing = [p for p in paths if _hash(p) not in existing]
    print(f"  Embedding {len(missing)} clips ({len(existing)} cached) ...", flush=True)

    backbone.eval()
    for i, path in enumerate(missing):
        try:
            emb = embed_clip(path, backbone, device)
        except Exception as exc:
            print(f"  WARN [{i}]: {os.path.basename(path)}: {exc}", flush=True)
            emb = np.zeros(EMB_DIM, dtype=np.float32)
        existing[_hash(path)] = emb.astype(np.float32)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(missing)}", flush=True)
            np.savez(cache_path, **existing)

    np.savez(cache_path, **existing)
    return existing


def _get_embs(df: pd.DataFrame, cache: dict) -> np.ndarray:
    return np.stack([cache.get(_hash(p), np.zeros(EMB_DIM, dtype=np.float32))
                     for p in df["audio_path"]])


# ── Training ─────────────────────────────────────────────────────────────────

def train_suppressor(train_df: pd.DataFrame, backbone, device: torch.device,
                     seed: int = SEED):
    from sklearn.linear_model import LogisticRegression

    mc_df = train_df[train_df["n_children_int"] >= 2].copy()
    print(f"  n_children>=2 train clips: {len(mc_df)} "
          f"(pos={mc_df['label'].sum()}, neg={len(mc_df)-mc_df['label'].sum()})", flush=True)

    cache = build_embedding_cache(mc_df, backbone, device,
                                  os.path.join(OUT_DIR, "emb_cache.npz"))
    X = _get_embs(mc_df, cache)
    y = mc_df["label"].to_numpy(dtype=int)

    clf = LogisticRegression(C=0.1, max_iter=500, random_state=seed)
    clf.fit(X, y)
    print(f"  Suppressor trained. Val classes={clf.classes_}", flush=True)
    return clf


def tune_alpha(val_df: pd.DataFrame, clf, backbone, device: torch.device) -> float:
    """Tune merge weight on n_children>=2 val clips."""
    mc_df = val_df[val_df["n_children_int"] >= 2].copy()
    print(f"  n_children>=2 val clips: {len(mc_df)}", flush=True)
    if len(mc_df) == 0:
        print("  WARNING: no val multi-child clips; defaulting alpha=0.5", flush=True)
        return 0.5

    cache = build_embedding_cache(mc_df, backbone, device,
                                  os.path.join(OUT_DIR, "emb_cache.npz"))
    X = _get_embs(mc_df, cache)
    supp = clf.predict_proba(X)[:, 1]
    main = _best_audio_mil(mc_df)
    y = mc_df["label"].to_numpy(dtype=int)

    best_a, best_f1 = 0.5, -1.0
    for a in np.linspace(0.0, 1.0, 21):
        merged = a * main + (1 - a) * supp
        t = tune_threshold(y, merged)
        f1 = compute_metrics(y, merged, t)["f1"]
        if f1 > best_f1:
            best_f1, best_a = f1, float(a)

    print(f"  Best alpha={best_a:.2f}  (val multi-child F1={best_f1:.4f})", flush=True)
    return best_a


def apply_suppressor(test_df: pd.DataFrame, clf, alpha: float,
                     backbone, device: torch.device):
    """Apply suppressor to test set; return (final_scores, mc_mask, main_scores)."""
    mc_mask = (test_df["n_children_int"] >= 2).to_numpy()
    mc_df = test_df[mc_mask].copy()
    sc_df = test_df[~mc_mask].copy()

    main_all = _best_audio_mil(test_df)
    final = main_all.copy()

    if len(mc_df) > 0:
        cache = build_embedding_cache(mc_df, backbone, device,
                                      os.path.join(OUT_DIR, "emb_cache.npz"))
        X = _get_embs(mc_df, cache)
        supp = clf.predict_proba(X)[:, 1]
        mc_main = _best_audio_mil(mc_df)
        final[mc_mask] = alpha * mc_main + (1 - alpha) * supp

    return final, mc_mask, main_all


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-child FP suppressor")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stratum sizes and exit before training")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("logs/evaluation", exist_ok=True)

    train_df = _load_train_meta()
    mc_train = (train_df["n_children_int"] >= 2).sum()
    print(f"Train: {len(train_df)} clips | n_children>=2: {mc_train}", flush=True)

    if args.dry_run:
        print("--dry-run: exiting before training.", flush=True)
        return

    # Load val/test system scores + metadata
    print("Loading val/test system scores ...", flush=True)
    val_scores = load_system_scores("val")
    test_scores = load_system_scores("test")
    meta = load_metadata()
    val_df = load_split(val_scores, meta, "val")
    test_df = load_split(test_scores, meta, "test")
    print(f"Val: {len(val_df)} | Test: {len(test_df)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    backbone = BackboneExtractor(BACKBONE).to(device)

    clf = train_suppressor(train_df, backbone, device, seed=args.seed)
    alpha = tune_alpha(val_df, clf, backbone, device)

    final_test, mc_mask, main_test = apply_suppressor(test_df, clf, alpha, backbone, device)
    y_test = test_df["label"].to_numpy(dtype=int)

    # Tune threshold on overall val merged scores
    mc_val = (val_df["n_children_int"] >= 2).to_numpy()
    val_main = _best_audio_mil(val_df)
    val_final = val_main.copy()
    if mc_val.sum() > 0:
        mc_val_df = val_df[mc_val].copy()
        cache = np.load(os.path.join(OUT_DIR, "emb_cache.npz"), allow_pickle=True)
        cache_dict = {k: cache[k] for k in cache.files}
        X_v = _get_embs(mc_val_df, cache_dict)
        supp_v = clf.predict_proba(X_v)[:, 1]
        val_final[mc_val] = alpha * _best_audio_mil(mc_val_df) + (1 - alpha) * supp_v

    t = tune_threshold(val_df["label"].to_numpy(), val_final)
    val_m = compute_metrics(val_df["label"].to_numpy(), val_final, t)
    val_m["threshold"] = t
    test_m = compute_metrics(y_test, final_test, t)
    test_m["threshold"] = t

    # Per-stratum metrics
    def _stratum_metrics(mask, tag):
        if mask.sum() == 0:
            print(f"  {tag}: 0 clips, skipping", flush=True)
            return {}
        m_before = compute_metrics(y_test[mask], main_test[mask], t)
        m_after = compute_metrics(y_test[mask], final_test[mask], t)
        out = {**m_after, "before_f1": m_before["f1"], "after_f1": m_after["f1"],
               "n": int(mask.sum())}
        path = os.path.join(OUT_DIR, f"test_metrics_{tag}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  {tag}: before_F1={m_before['f1']:.4f} after_F1={m_after['f1']:.4f}",
              flush=True)
        return out

    _stratum_metrics(mc_mask, "multi_child_only")
    _stratum_metrics(~mc_mask, "single_child_only")

    preds = test_df[["audio_path", "label", "timepoint_norm"]].copy()
    preds["score"] = final_test
    preds["main_score"] = main_test
    preds["prediction"] = (final_test >= t).astype(int)
    preds["n_children_int"] = test_df["n_children_int"].to_numpy()
    preds["suppressor_applied"] = mc_mask.astype(int)

    cfg = {
        "sub_feature": "C",
        "backbone": BACKBONE,
        "alpha": alpha,
        "val_threshold": t,
        "seed": args.seed,
        "n_train_mc": int(mc_train),
        "created": "2026-04-28",
    }
    save_results(OUT_DIR, val_m, test_m, preds, cfg)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
