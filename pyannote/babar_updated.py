"""
Combined feature model: logistic regression over diarizer + embedding + phoneme features.

Expects BabAR to have already been run (RTTM + phoneme CSVs exist).
Expects ECAPA enrollment to have already been run (prototypes exist).

Usage:
    python combined_features.py \
        --babar-output /home/manaal/orcd/scratch/child-adult-diarization/babar/babar_output \
        --results-dir /home/manaal/orcd/scratch/child-adult-diarization/babar_combined_runs
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
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


# =============================================================
# Config
# =============================================================

SPLIT_DIR = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/seen_child_splits"
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000
MIN_SEG_DUR = 0.4
MAX_ENROLL_SEGS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================
# Utilities
# =============================================================

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


# =============================================================
# Audio / ECAPA
# =============================================================

def load_audio_mono(path, sr=16000):
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


# =============================================================
# BabAR RTTM + phoneme parsing
# =============================================================

def find_babar_rttm(audio_path: str, babar_output_dir: str) -> Optional[str]:
    """
    BabAR names output files after the symlink stem used during prepare().
    Try both: stem__md5.rttm and stem.rttm
    """
    stem = Path(audio_path).stem
    cid = audio_to_cache_id(audio_path)
    rttm_dir = os.path.join(babar_output_dir, "rttm")

    candidates = [
        os.path.join(rttm_dir, f"{stem}__{cid}.rttm"),
        os.path.join(rttm_dir, f"{stem}.rttm"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def find_babar_phonemes(audio_path: str, babar_output_dir: str) -> Optional[str]:
    stem = Path(audio_path).stem
    cid = audio_to_cache_id(audio_path)
    phon_dir = os.path.join(babar_output_dir, "phonemes")

    candidates = [
        os.path.join(phon_dir, f"{stem}__{cid}.csv"),
        os.path.join(phon_dir, f"{stem}.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def parse_rttm_kchi(rttm_path: str) -> List[Dict[str, float]]:
    """Parse RTTM, return only KCHI segments."""
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


def parse_phoneme_csv(phon_path: str) -> List[Dict]:
    """
    Parse BabAR phoneme CSV.
    Columns: filename, onset, offset, speaker, phonemes
    """
    rows = []
    if not phon_path or not os.path.exists(phon_path):
        return rows
    try:
        df = pd.read_csv(phon_path)
    except Exception:
        return rows
    for _, row in df.iterrows():
        phon_str = str(row.get("phonemes", "")).strip()
        phonemes = phon_str.split() if phon_str and phon_str != "nan" else []
        rows.append({
            "onset": float(row.get("onset", 0)),
            "offset": float(row.get("offset", 0)),
            "phonemes": phonemes,
        })
    return rows


# =============================================================
# IPA phoneme classification
# =============================================================

CONSONANTS = set("p b t d k ɡ g m n ŋ f v s z ʃ ʒ θ ð h l ɹ r w j ʔ tʃ dʒ ɾ".split())
VOWELS = set("i ɪ e ɛ æ ɑ ɒ ɔ o ʊ u ə ʌ aɪ aʊ ɔɪ eɪ oʊ iː uː ɜː ɑː ɔː a".split())


def classify_phoneme(ph):
    """Classify a phoneme as consonant, vowel, or other."""
    if ph in CONSONANTS:
        return "C"
    elif ph in VOWELS:
        return "V"
    return "O"


# =============================================================
# Feature extraction
# =============================================================

def extract_diarizer_features(segs: List[Dict], clip_duration_sec: float) -> Dict[str, float]:
    """Features from KCHI segments alone (no embedding model needed)."""
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


def extract_phoneme_features(phoneme_rows: List[Dict]) -> Dict[str, float]:
    """Features derived from BabAR's phoneme transcriptions."""
    if not phoneme_rows:
        return {
            "phon_n_utterances": 0,
            "phon_n_total": 0,
            "phon_n_unique": 0,
            "phon_n_consonants": 0,
            "phon_n_vowels": 0,
            "phon_cv_ratio": 0.0,
            "phon_mean_per_utt": 0.0,
            "phon_max_per_utt": 0,
            "phon_unique_ratio": 0.0,
        }

    all_phonemes = []
    utt_lengths = []
    n_consonants = 0
    n_vowels = 0

    for row in phoneme_rows:
        phs = row["phonemes"]
        utt_lengths.append(len(phs))
        for ph in phs:
            all_phonemes.append(ph)
            cat = classify_phoneme(ph)
            if cat == "C":
                n_consonants += 1
            elif cat == "V":
                n_vowels += 1

    n_total = len(all_phonemes)
    n_unique = len(set(all_phonemes))
    n_utt_with_phonemes = sum(1 for l in utt_lengths if l > 0)

    return {
        "phon_n_utterances": len(phoneme_rows),
        "phon_n_total": n_total,
        "phon_n_unique": n_unique,
        "phon_n_consonants": n_consonants,
        "phon_n_vowels": n_vowels,
        "phon_cv_ratio": n_consonants / max(n_vowels, 1),
        "phon_mean_per_utt": n_total / max(len(phoneme_rows), 1),
        "phon_max_per_utt": max(utt_lengths) if utt_lengths else 0,
        "phon_unique_ratio": n_unique / max(n_total, 1),
    }


def extract_embedding_features(
    audio_path: str,
    segs: List[Dict],
    prototype: Optional[np.ndarray],
    embedder: ECAPAEmbedder,
) -> Dict[str, float]:
    """
    Multiple pooling strategies for cosine similarity scores.
    Returns weighted mean, max, and top-3 mean.
    """
    defaults = {
        "sim_weighted_mean": 0.0,
        "sim_max": 0.0,
        "sim_top3_mean": 0.0,
    }

    if prototype is None or not segs:
        return defaults

    wav = load_audio_mono(audio_path, SAMPLE_RATE)
    scored = []

    for seg in segs:
        clip = crop_segment(wav, SAMPLE_RATE, seg["start"], seg["end"])
        if clip.numel() < int(MIN_SEG_DUR * SAMPLE_RATE):
            continue
        try:
            emb = embedder.embed_waveform(clip)
            sim = cosine_similarity(emb, prototype)
            scored.append((sim, seg["dur"]))
        except Exception:
            continue

    if not scored:
        return defaults

    sims = [s for s, _ in scored]
    durs = [d for _, d in scored]
    total_dur = sum(durs)

    # Duration-weighted mean
    weighted_mean = sum(s * d for s, d in scored) / total_dur

    # Max
    sim_max = max(sims)

    # Top-3 mean (or fewer if less than 3 segments)
    top_k = sorted(sims, reverse=True)[:3]
    top3_mean = sum(top_k) / len(top_k)

    return {
        "sim_weighted_mean": weighted_mean,
        "sim_max": sim_max,
        "sim_top3_mean": top3_mean,
    }


def get_clip_duration(audio_path: str) -> float:
    """Get audio duration in seconds without loading the full waveform."""
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception:
        return 0.0


# =============================================================
# Prototype building (same as unified pipeline)
# =============================================================

def build_child_prototypes(
    train_df: pd.DataFrame,
    babar_output_dir: str,
    embedder: ECAPAEmbedder,
) -> Dict[str, np.ndarray]:
    prototypes = {}
    pos_train = train_df[train_df["label"] == 1]

    for (child_id, timepoint), sub in pos_train.groupby(["child_id", "timepoint_norm"]):
        proto_key = f"{child_id}__{timepoint}"
        all_pairs = []

        for _, row in sub.iterrows():
            ap = row["audio_path"]
            rttm = find_babar_rttm(ap, babar_output_dir)
            segs = parse_rttm_kchi(rttm)
            if not segs:
                continue

            wav = load_audio_mono(ap, SAMPLE_RATE)
            for seg in segs:
                clip = crop_segment(wav, SAMPLE_RATE, seg["start"], seg["end"])
                if clip.numel() < int(MIN_SEG_DUR * SAMPLE_RATE):
                    continue
                try:
                    emb = embedder.embed_waveform(clip)
                    all_pairs.append((emb, seg["dur"]))
                except Exception:
                    continue

                if len(all_pairs) >= MAX_ENROLL_SEGS:
                    break
            if len(all_pairs) >= MAX_ENROLL_SEGS:
                break

        if not all_pairs:
            continue

        embs = np.stack([e for e, _ in all_pairs])
        weights = np.array([d for _, d in all_pairs])
        proto = np.average(embs, axis=0, weights=weights)
        prototypes[proto_key] = l2_normalize(proto)

    return prototypes


# =============================================================
# Full feature extraction for a dataset split
# =============================================================

def extract_all_features(
    df: pd.DataFrame,
    babar_output_dir: str,
    prototypes: Dict[str, np.ndarray],
    embedder: ECAPAEmbedder,
) -> pd.DataFrame:
    rows = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Extracting features: {i+1}/{total}")

        ap = row["audio_path"]
        child_id = row["child_id"]

        # Get clip duration
        clip_dur = get_clip_duration(ap)

        # Parse BabAR outputs
        rttm = find_babar_rttm(ap, babar_output_dir)
        segs = parse_rttm_kchi(rttm)

        phon_path = find_babar_phonemes(ap, babar_output_dir)
        phon_rows = parse_phoneme_csv(phon_path)

        # Get prototype for this child at this timepoint
        proto_key = f"{child_id}__{row['timepoint_norm']}"
        proto = prototypes.get(proto_key, None)

        # Extract feature groups
        diar_feats = extract_diarizer_features(segs, clip_dur)
        phon_feats = extract_phoneme_features(phon_rows)
        emb_feats = extract_embedding_features(ap, segs, proto, embedder)

        # Combine
        feat_row = {
            "audio_path": ap,
            "child_id": child_id,
            "timepoint_norm": row["timepoint_norm"],
            "label": int(row["label"]),
            **diar_feats,
            **phon_feats,
            **emb_feats,
        }
        rows.append(feat_row)

    return pd.DataFrame(rows)


# =============================================================
# Model training and evaluation
# =============================================================

FEATURE_SETS = {
    "role_only": ["kchi_total_dur"],
    "enrollment_only": ["sim_weighted_mean"],
    "diarizer_all": [
        "kchi_total_dur", "kchi_n_segments", "kchi_mean_seg_dur",
        "kchi_max_seg_dur", "kchi_proportion",
    ],
    "embedding_all": [
        "sim_weighted_mean", "sim_max", "sim_top3_mean",
    ],
    "phoneme_all": [
        "phon_n_utterances", "phon_n_total", "phon_n_unique",
        "phon_n_consonants", "phon_n_vowels", "phon_cv_ratio",
        "phon_mean_per_utt", "phon_max_per_utt", "phon_unique_ratio",
    ],
    "diarizer_plus_embedding": [
        "kchi_total_dur", "kchi_n_segments", "kchi_mean_seg_dur",
        "kchi_max_seg_dur", "kchi_proportion",
        "sim_weighted_mean", "sim_max", "sim_top3_mean",
    ],
    "diarizer_plus_phoneme": [
        "kchi_total_dur", "kchi_n_segments", "kchi_mean_seg_dur",
        "kchi_max_seg_dur", "kchi_proportion",
        "phon_n_utterances", "phon_n_total", "phon_n_unique",
        "phon_n_consonants", "phon_n_vowels", "phon_cv_ratio",
        "phon_mean_per_utt", "phon_max_per_utt", "phon_unique_ratio",
    ],
    "all_features": [
        "kchi_total_dur", "kchi_n_segments", "kchi_mean_seg_dur",
        "kchi_max_seg_dur", "kchi_proportion",
        "sim_weighted_mean", "sim_max", "sim_top3_mean",
        "phon_n_utterances", "phon_n_total", "phon_n_unique",
        "phon_n_consonants", "phon_n_vowels", "phon_cv_ratio",
        "phon_mean_per_utt", "phon_max_per_utt", "phon_unique_ratio",
    ],
}


def train_and_evaluate(
    train_feats: pd.DataFrame,
    val_feats: pd.DataFrame,
    test_feats: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
) -> Dict:
    """
    Train logistic regression on train, tune threshold on val, evaluate on test.
    """
    X_train = train_feats[feature_cols].values.astype(float)
    y_train = train_feats["label"].values

    X_val = val_feats[feature_cols].values.astype(float)
    y_val = val_feats["label"].values

    X_test = test_feats[feature_cols].values.astype(float)
    y_test = test_feats["label"].values

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    # Get probabilities
    val_probs = clf.predict_proba(X_val)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]

    # Tune threshold on val
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        f = float(f1_score(y_val, (val_probs >= t).astype(int), zero_division=0))
        if f > best_f1:
            best_f1, best_t = f, float(t)

    val_metrics = compute_metrics(y_val, val_probs, threshold=best_t)
    test_metrics = compute_metrics(y_test, test_probs, threshold=best_t)

    # Per-timepoint test metrics
    test_pred_df = test_feats[["label", "timepoint_norm"]].copy()
    test_pred_df["prob"] = test_probs
    tp_metrics = per_timepoint_metrics(test_pred_df, best_t)

    # Feature importances (logistic regression coefficients)
    coefs = dict(zip(feature_cols, clf.coef_[0].tolist()))

    return {
        "model_name": model_name,
        "features": feature_cols,
        "threshold": best_t,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_by_timepoint": tp_metrics.to_dict(orient="records"),
        "coefficients": coefs,
        "test_probs": test_probs,
    }


# =============================================================
# Main
# =============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--babar-output", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip feature extraction, load from cached CSVs.",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load splits
    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    train_feat_path = os.path.join(args.results_dir, "train_features.csv")
    val_feat_path = os.path.join(args.results_dir, "val_features.csv")
    test_feat_path = os.path.join(args.results_dir, "test_features.csv")

    if args.skip_extraction and all(
        os.path.exists(p) for p in [train_feat_path, val_feat_path, test_feat_path]
    ):
        print("Loading cached features...")
        train_feats = pd.read_csv(train_feat_path)
        val_feats = pd.read_csv(val_feat_path)
        test_feats = pd.read_csv(test_feat_path)
    else:
        # Load ECAPA
        print("Loading ECAPA...")
        embedder = ECAPAEmbedder(ECAPA_SOURCE, DEVICE)

        # Build prototypes
        print("Building prototypes...")
        prototypes = build_child_prototypes(train_df, args.babar_output, embedder)
        print(f"Built prototypes for {len(prototypes)} (child, timepoint) pairs.")

        seen_combos = set(zip(train_df["child_id"], train_df["timepoint_norm"]))
        missing = {f"{c}__{t}" for c, t in seen_combos} - set(prototypes.keys())
        if missing:
            print(f"WARNING: {len(missing)} (child, timepoint) pairs missing prototypes: {missing}")

        # Extract features
        print("\nExtracting train features...")
        train_feats = extract_all_features(train_df, args.babar_output, prototypes, embedder)

        print("\nExtracting val features...")
        val_feats = extract_all_features(val_df, args.babar_output, prototypes, embedder)

        print("\nExtracting test features...")
        test_feats = extract_all_features(test_df, args.babar_output, prototypes, embedder)

        # Cache features
        train_feats.to_csv(train_feat_path, index=False)
        val_feats.to_csv(val_feat_path, index=False)
        test_feats.to_csv(test_feat_path, index=False)
        print("Features cached.")

    # Train and evaluate all feature set combinations
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    all_results = {}

    for name, feat_cols in FEATURE_SETS.items():
        print(f"\n--- {name} ({len(feat_cols)} features) ---")
        result = train_and_evaluate(train_feats, val_feats, test_feats, feat_cols, name)

        all_results[name] = {
            "features": result["features"],
            "threshold": result["threshold"],
            "val_metrics": result["val_metrics"],
            "test_metrics": result["test_metrics"],
            "test_by_timepoint": result["test_by_timepoint"],
            "coefficients": result["coefficients"],
        }

        tm = result["test_metrics"]
        print(f"  Test — F1: {tm['f1']:.3f}  AUROC: {tm['auroc']:.3f}  AUPRC: {tm['auprc']:.3f}")
        for tp_row in result["test_by_timepoint"]:
            print(f"    {tp_row['timepoint']}: AUROC={tp_row['auroc']:.3f}  AUPRC={tp_row['auprc']:.3f}")

        # Save per-model predictions
        test_pred = test_feats[["audio_path", "child_id", "timepoint_norm", "label"]].copy()
        test_pred["prob"] = result["test_probs"]
        test_pred["pred_label"] = (test_pred["prob"] >= result["threshold"]).astype(int)
        test_pred.to_csv(
            os.path.join(args.results_dir, f"{name}_test_predictions.csv"),
            index=False,
        )

    # Save summary
    save_json(all_results, os.path.join(args.results_dir, "all_model_results.json"))

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'F1':>6} {'AUROC':>7} {'AUPRC':>7}")
    print("-" * 55)
    for name in FEATURE_SETS:
        tm = all_results[name]["test_metrics"]
        print(f"{name:<30} {tm['f1']:>6.3f} {tm['auroc']:>7.3f} {tm['auprc']:>7.3f}")

    print(f"\nResults saved to {args.results_dir}")


if __name__ == "__main__":
    main()
