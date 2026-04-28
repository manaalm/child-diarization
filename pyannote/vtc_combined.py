"""
Combined feature model for VTC diarizer: logistic regression / GBM over
diarizer features + ECAPA embedding features.

VTC does not produce phoneme transcripts, so the phoneme feature group is
absent. Feature sets: role_only, diarizer_all, embedding_all,
diarizer_plus_embedding.

Expects:
  - VTC RTTM cache at pyannote/vtc_rttm_cache/{stem}__{md5}.rttm
  - Seen-child splits at whisper-modeling/seen_child_splits/

Usage:
    python vtc_combined.py --results-dir vtc_combined_runs/
    python vtc_combined.py --results-dir vtc_combined_runs/ --skip-extraction
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from speechbrain.inference.speaker import EncoderClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPLIT_DIR = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/seen_child_splits"
VTC_RTTM_CACHE = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/vtc_rttm_cache"
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000
MIN_SEG_DUR = 0.4
MAX_ENROLL_SEGS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def audio_to_cache_id(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode("utf-8")).hexdigest()


def l2_normalize(x: np.ndarray, eps=1e-8):
    return x / max(np.linalg.norm(x), eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    m = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:
        m["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        m["auroc"] = float("nan")
    try:
        m["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        m["auprc"] = float("nan")
    return m


def per_timepoint_metrics(pred_df, threshold):
    rows = []
    for tp, sub in pred_df.groupby("timepoint_norm"):
        m = compute_metrics(sub["label"].values, sub["prob"].values, threshold)
        m["timepoint"] = tp
        m["n"] = len(sub)
        rows.append(m)
    return pd.DataFrame(rows)


def tune_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        f = float(f1_score(y_true, (y_prob >= t).astype(int), zero_division=0))
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Audio / ECAPA
# ---------------------------------------------------------------------------

def load_audio_mono(path, sr=SAMPLE_RATE):
    wav, orig_sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav.squeeze(0)


def crop_segment(wav, sr, start, end):
    s = max(0, int(round(start * sr)))
    e = min(wav.numel(), int(round(end * sr)))
    return wav[s:e] if e > s else torch.zeros(1, dtype=wav.dtype)


class ECAPAEmbedder:
    def __init__(self, source, device):
        self.model = EncoderClassifier.from_hparams(
            source=source, run_opts={"device": device}
        )
        self.device = device

    @torch.no_grad()
    def embed_waveform(self, wav_1d):
        wav = wav_1d.unsqueeze(0).to(self.device)
        emb = self.model.encode_batch(wav)
        return emb.squeeze().detach().cpu().numpy()


# ---------------------------------------------------------------------------
# VTC RTTM parsing
# ---------------------------------------------------------------------------

def find_vtc_rttm(audio_path: str) -> Optional[str]:
    stem = Path(audio_path).stem
    md5 = audio_to_cache_id(audio_path)
    candidates = [
        os.path.join(VTC_RTTM_CACHE, f"{stem}__{md5}.rttm"),
        os.path.join(VTC_RTTM_CACHE, f"{stem}.rttm"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def parse_rttm_kchi(rttm_path: Optional[str]) -> List[Dict]:
    segs = []
    if not rttm_path or not os.path.exists(rttm_path):
        return segs
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                start, dur = float(parts[3]), float(parts[4])
            except ValueError:
                continue
            label = parts[7]
            if label == "KCHI" and dur >= MIN_SEG_DUR:
                segs.append({"start": start, "end": start + dur, "dur": dur})
    return segs


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def get_clip_duration(audio_path: str) -> float:
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception:
        return 0.0


def extract_diarizer_features(segs: List[Dict], clip_duration_sec: float) -> Dict:
    if not segs:
        return {
            "kchi_total_dur": 0.0,
            "kchi_n_segments": 0,
            "kchi_mean_seg_dur": 0.0,
            "kchi_max_seg_dur": 0.0,
            "kchi_proportion": 0.0,
        }
    durs = [s["dur"] for s in segs]
    total = sum(durs)
    return {
        "kchi_total_dur": total,
        "kchi_n_segments": len(segs),
        "kchi_mean_seg_dur": total / len(segs),
        "kchi_max_seg_dur": max(durs),
        "kchi_proportion": total / max(clip_duration_sec, 0.01),
    }


def extract_embedding_features(
    audio_path: str,
    segs: List[Dict],
    prototype: Optional[np.ndarray],
    embedder: ECAPAEmbedder,
) -> Dict:
    defaults = {"sim_weighted_mean": 0.0, "sim_max": 0.0, "sim_top3_mean": 0.0}
    if prototype is None or not segs:
        return defaults

    wav = load_audio_mono(audio_path)
    scored = []
    for seg in segs:
        clip = crop_segment(wav, SAMPLE_RATE, seg["start"], seg["end"])
        if clip.numel() < int(MIN_SEG_DUR * SAMPLE_RATE):
            continue
        try:
            emb = embedder.embed_waveform(clip)
            scored.append((cosine_similarity(emb, prototype), seg["dur"]))
        except Exception:
            continue

    if not scored:
        return defaults

    sims, durs = zip(*scored)
    weighted_mean = sum(s * d for s, d in scored) / sum(durs)
    top_k = sorted(sims, reverse=True)[:3]
    return {
        "sim_weighted_mean": weighted_mean,
        "sim_max": max(sims),
        "sim_top3_mean": sum(top_k) / len(top_k),
    }


def extract_all_features(
    df: pd.DataFrame,
    prototypes: Dict[str, np.ndarray],
    embedder: ECAPAEmbedder,
) -> pd.DataFrame:
    rows = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Extracting features: {i+1}/{total}")
        ap = row["audio_path"]
        clip_dur = get_clip_duration(ap)
        rttm = find_vtc_rttm(ap)
        segs = parse_rttm_kchi(rttm)
        proto = prototypes.get(f"{row['child_id']}__{row['timepoint_norm']}")
        rows.append({
            "audio_path": ap,
            "child_id": row["child_id"],
            "timepoint_norm": row["timepoint_norm"],
            "label": int(row["label"]),
            **extract_diarizer_features(segs, clip_dur),
            **extract_embedding_features(ap, segs, proto, embedder),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prototype building
# ---------------------------------------------------------------------------

def build_child_prototypes(
    train_df: pd.DataFrame,
    embedder: ECAPAEmbedder,
) -> Dict[str, np.ndarray]:
    prototypes = {}
    for (child_id, timepoint), sub in train_df[train_df["label"] == 1].groupby(["child_id", "timepoint_norm"]):
        proto_key = f"{child_id}__{timepoint}"
        all_pairs = []
        for _, row in sub.iterrows():
            ap = row["audio_path"]
            segs = parse_rttm_kchi(find_vtc_rttm(ap))
            if not segs:
                continue
            wav = load_audio_mono(ap)
            for seg in segs:
                clip = crop_segment(wav, SAMPLE_RATE, seg["start"], seg["end"])
                if clip.numel() < int(MIN_SEG_DUR * SAMPLE_RATE):
                    continue
                try:
                    all_pairs.append((embedder.embed_waveform(clip), seg["dur"]))
                except Exception:
                    continue
                if len(all_pairs) >= MAX_ENROLL_SEGS:
                    break
            if len(all_pairs) >= MAX_ENROLL_SEGS:
                break
        if all_pairs:
            embs = np.stack([e for e, _ in all_pairs])
            weights = np.array([d for _, d in all_pairs])
            prototypes[proto_key] = l2_normalize(np.average(embs, axis=0, weights=weights))
    return prototypes


# ---------------------------------------------------------------------------
# Feature sets (no phoneme group — VTC has no phoneme output)
# ---------------------------------------------------------------------------

FEATURE_SETS = {
    "role_only": ["kchi_total_dur"],
    "enrollment_only": ["sim_weighted_mean"],
    "diarizer_all": [
        "kchi_total_dur", "kchi_n_segments", "kchi_mean_seg_dur",
        "kchi_max_seg_dur", "kchi_proportion",
    ],
    "embedding_all": ["sim_weighted_mean", "sim_max", "sim_top3_mean"],
    "diarizer_plus_embedding": [
        "kchi_total_dur", "kchi_n_segments", "kchi_mean_seg_dur",
        "kchi_max_seg_dur", "kchi_proportion",
        "sim_weighted_mean", "sim_max", "sim_top3_mean",
    ],
}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _build_clf(model_type):
    if model_type == "gbm":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )
    return LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs")


def train_and_evaluate(train_f, val_f, test_f, feature_cols, model_name, model_type="logistic"):
    def prep(df):
        return np.nan_to_num(df[feature_cols].values.astype(float))

    X_tr, y_tr = prep(train_f), train_f["label"].values
    X_va, y_va = prep(val_f), val_f["label"].values
    X_te, y_te = prep(test_f), test_f["label"].values

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    clf = _build_clf(model_type)
    clf.fit(X_tr, y_tr)

    val_probs = clf.predict_proba(X_va)[:, 1]
    test_probs = clf.predict_proba(X_te)[:, 1]
    threshold = tune_threshold(y_va, val_probs)

    test_pred_df = test_f[["label", "timepoint_norm"]].copy()
    test_pred_df["prob"] = test_probs
    tp_metrics = per_timepoint_metrics(test_pred_df, threshold)

    importances = (
        dict(zip(feature_cols, clf.feature_importances_.tolist()))
        if model_type == "gbm"
        else dict(zip(feature_cols, clf.coef_[0].tolist()))
    )

    return {
        "model_name": model_name,
        "model_type": model_type,
        "features": feature_cols,
        "threshold": threshold,
        "val_metrics": compute_metrics(y_va, val_probs, threshold),
        "test_metrics": compute_metrics(y_te, test_probs, threshold),
        "test_by_timepoint": tp_metrics.to_dict(orient="records"),
        "importances": importances,
        "test_probs": test_probs,
        "val_probs": val_probs,
    }


def train_and_evaluate_per_timepoint(train_f, val_f, test_f, feature_cols, model_name, model_type="logistic"):
    timepoints = sorted(test_f["timepoint_norm"].unique())
    all_test_probs = np.zeros(len(test_f))
    all_val_probs = np.zeros(len(val_f))
    tp_results = []

    for tp in timepoints:
        tr = train_f[train_f["timepoint_norm"] == tp]
        va = val_f[val_f["timepoint_norm"] == tp]
        te = test_f[test_f["timepoint_norm"] == tp]
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            continue

        def prep(df):
            return np.nan_to_num(df[feature_cols].values.astype(float))

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(prep(tr))
        X_va = scaler.transform(prep(va))
        X_te = scaler.transform(prep(te))

        clf = _build_clf(model_type)
        clf.fit(X_tr, tr["label"].values)

        va_p = clf.predict_proba(X_va)[:, 1]
        te_p = clf.predict_proba(X_te)[:, 1]
        t = tune_threshold(va["label"].values, va_p)

        m = compute_metrics(te["label"].values, te_p, t)
        m.update({"timepoint": tp, "n": len(te), "threshold": t})
        tp_results.append(m)

        for i, idx in enumerate(te.index):
            pos = test_f.index.get_loc(idx)
            all_test_probs[pos] = te_p[i]
        for i, idx in enumerate(va.index):
            pos = val_f.index.get_loc(idx)
            all_val_probs[pos] = va_p[i]

    overall = compute_metrics(test_f["label"].values, all_test_probs, threshold=0.5)
    return {
        "model_name": model_name,
        "model_type": model_type,
        "features": feature_cols,
        "test_metrics": overall,
        "test_by_timepoint": tp_results,
        "test_probs": all_test_probs,
        "val_probs": all_val_probs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="vtc_combined_runs")
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    tr_path = os.path.join(args.results_dir, "train_features.csv")
    va_path = os.path.join(args.results_dir, "val_features.csv")
    te_path = os.path.join(args.results_dir, "test_features.csv")

    if args.skip_extraction and all(os.path.exists(p) for p in [tr_path, va_path, te_path]):
        print("Loading cached features...")
        train_feats = pd.read_csv(tr_path)
        val_feats   = pd.read_csv(va_path)
        test_feats  = pd.read_csv(te_path)
    else:
        print(f"Loading ECAPA on {DEVICE}...")
        embedder = ECAPAEmbedder(ECAPA_SOURCE, DEVICE)

        print("Building prototypes from VTC KCHI segments...")
        prototypes = build_child_prototypes(train_df, embedder)
        print(f"Built {len(prototypes)} prototypes.")

        print("\nExtracting train features...")
        train_feats = extract_all_features(train_df, prototypes, embedder)
        print("\nExtracting val features...")
        val_feats   = extract_all_features(val_df,   prototypes, embedder)
        print("\nExtracting test features...")
        test_feats  = extract_all_features(test_df,  prototypes, embedder)

        train_feats.to_csv(tr_path, index=False)
        val_feats.to_csv(va_path,   index=False)
        test_feats.to_csv(te_path,  index=False)
        print("Features cached.")

    all_results = {}

    for name, feat_cols in FEATURE_SETS.items():
        for mt in ("logistic", "gbm"):
            label = f"{mt}_{name}"
            print(f"\n--- {label} ({len(feat_cols)} features) ---")
            result = train_and_evaluate(train_feats, val_feats, test_feats, feat_cols, label, mt)
            all_results[label] = {
                "features": result["features"],
                "threshold": result["threshold"],
                "val_metrics": result["val_metrics"],
                "test_metrics": result["test_metrics"],
                "test_by_timepoint": result["test_by_timepoint"],
                "importances": result["importances"],
            }
            tm = result["test_metrics"]
            print(f"  Test — F1: {tm['f1']:.3f}  AUROC: {tm['auroc']:.3f}  AUPRC: {tm['auprc']:.3f}")

            # Save predictions
            pred_df = test_feats[["audio_path", "child_id", "timepoint_norm", "label"]].copy()
            pred_df["prob"] = result["test_probs"]
            pred_df["pred_label"] = (pred_df["prob"] >= result["threshold"]).astype(int)
            pred_df.to_csv(os.path.join(args.results_dir, f"{label}_test_predictions.csv"), index=False)

            val_pred_df = val_feats[["audio_path", "child_id", "timepoint_norm", "label"]].copy()
            val_pred_df["prob"] = result["val_probs"]
            val_pred_df["pred_label"] = (val_pred_df["prob"] >= result["threshold"]).astype(int)
            val_pred_df.to_csv(os.path.join(args.results_dir, f"{label}_val_predictions.csv"), index=False)

    # Per-timepoint variants for best feature set
    for mt in ("logistic", "gbm"):
        name = "diarizer_plus_embedding"
        feat_cols = FEATURE_SETS[name]
        label = f"pertp_{mt}_{name}"
        print(f"\n--- {label} (per-timepoint) ---")
        result = train_and_evaluate_per_timepoint(train_feats, val_feats, test_feats, feat_cols, label, mt)
        all_results[label] = {
            "features": result["features"],
            "test_metrics": result["test_metrics"],
            "test_by_timepoint": result["test_by_timepoint"],
        }
        tm = result["test_metrics"]
        print(f"  Test — F1: {tm['f1']:.3f}  AUROC: {tm['auroc']:.3f}  AUPRC: {tm['auprc']:.3f}")

        pred_df = test_feats[["audio_path", "child_id", "timepoint_norm", "label"]].copy()
        pred_df["prob"] = result["test_probs"]
        pred_df["pred_label"] = (pred_df["prob"] >= 0.5).astype(int)
        pred_df.to_csv(os.path.join(args.results_dir, f"{label}_test_predictions.csv"), index=False)

        val_pred_df = val_feats[["audio_path", "child_id", "timepoint_norm", "label"]].copy()
        val_pred_df["prob"] = result["val_probs"]
        val_pred_df["pred_label"] = (val_pred_df["prob"] >= 0.5).astype(int)
        val_pred_df.to_csv(os.path.join(args.results_dir, f"{label}_val_predictions.csv"), index=False)

    save_json(all_results, os.path.join(args.results_dir, "all_model_results.json"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<45} {'F1':>6} {'AUROC':>7} {'AUPRC':>7}")
    print("-" * 70)
    for name in sorted(all_results.keys()):
        tm = all_results[name]["test_metrics"]
        print(f"{name:<45} {tm['f1']:>6.3f} {tm['auroc']:>7.3f} {tm['auprc']:>7.3f}")

    print(f"\nResults saved to {args.results_dir}")


if __name__ == "__main__":
    main()
