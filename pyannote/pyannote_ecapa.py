import os
import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from speechbrain.inference.speaker import EncoderClassifier
from pyannote.audio import Pipeline


# =========================================================
# Config
# =========================================================

@dataclass
class Config:
    split_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/seen_child_splits"
    results_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/pyannote_enrollment_runs"

    sample_rate: int = 16000
    min_seg_dur_sec: float = 0.4
    max_enrollment_segments_per_child: int = 200

    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # PyAnnote
    pyannote_model: str = "pyannote/speaker-diarization-community-1"
    hf_token: str = os.environ.get("HF_TOKEN", "")
    pyannote_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/pyannote_rttm_cache"

    similarity_threshold_min: float = 0.1
    similarity_threshold_max: float = 0.95
    similarity_threshold_steps: int = 171


CFG = Config()


# =========================================================
# Utilities
# =========================================================

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auroc"] = float("nan")
    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["auprc"] = float("nan")
    return metrics


def add_pred_labels(pred_df: pd.DataFrame, threshold: float):
    out = pred_df.copy()
    out["pred_label"] = (out["prob"] >= threshold).astype(int)
    return out


def per_timepoint_metrics(pred_df: pd.DataFrame, threshold: float):
    rows = []
    for tp, sub in pred_df.groupby("timepoint_norm"):
        y_true = sub["label"].to_numpy()
        y_prob = sub["prob"].to_numpy()
        m = compute_metrics(y_true, y_prob, threshold=threshold)
        m["timepoint"] = tp
        m["n"] = int(len(sub))
        rows.append(m)
    return pd.DataFrame(rows)


def load_split(split_dir: str):
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"))
    return train_df, val_df, test_df


def l2_normalize(x: np.ndarray, eps: float = 1e-8):
    n = np.linalg.norm(x)
    return x / max(n, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def audio_to_cache_id(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode("utf-8")).hexdigest()


def rttm_cache_path_for_audio(audio_path: str, cache_dir: str) -> str:
    cache_id = audio_to_cache_id(audio_path)
    stem = Path(audio_path).stem
    return os.path.join(cache_dir, f"{stem}__{cache_id}.rttm")


# =========================================================
# Audio / ECAPA
# =========================================================

def load_audio_mono(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


def crop_segment(wav: torch.Tensor, sr: int, start: float, end: float) -> torch.Tensor:
    s = max(0, int(round(start * sr)))
    e = min(wav.numel(), int(round(end * sr)))
    if e <= s:
        return torch.zeros(1, dtype=wav.dtype)
    return wav[s:e]


class ECAPAEmbedder:
    def __init__(self, source: str, device: str):
        self.model = EncoderClassifier.from_hparams(
            source=source,
            run_opts={"device": device},
        )
        self.device = device

    @torch.no_grad()
    def embed_waveform(self, wav_1d: torch.Tensor) -> np.ndarray:
        wav = wav_1d.unsqueeze(0).to(self.device)
        emb = self.model.encode_batch(wav)
        return emb.squeeze().detach().cpu().numpy()


# =========================================================
# PyAnnote front-end
# =========================================================

def load_pyannote_pipeline(cfg: Config):
    if not cfg.hf_token:
        raise ValueError("Set HF_TOKEN in environment before running.")
    pipeline = Pipeline.from_pretrained(cfg.pyannote_model, token=cfg.hf_token)
    if "cuda" in cfg.device and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline


def run_pyannote_inference(audio_path: str, pipeline, cfg: Config) -> str:
    os.makedirs(cfg.pyannote_cache_dir, exist_ok=True)
    target_rttm = rttm_cache_path_for_audio(audio_path, cfg.pyannote_cache_dir)
    if os.path.exists(target_rttm):
        return target_rttm

    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    out = pipeline({
        "waveform": wav,
        "sample_rate": sr,
    })

    ann = getattr(out, "speaker_diarization", out)

    with open(target_rttm, "w") as f:
        ann.write_rttm(f)

    return target_rttm


def parse_pyannote_rttm(rttm_path: str) -> List[Dict]:
    segments = []
    if not os.path.exists(rttm_path):
        return segments

    with open(rttm_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            if dur <= 0:
                continue
            segments.append({
                "start": start,
                "end": start + dur,
                "dur": dur,
                "speaker": speaker,
            })
    return segments


def get_pyannote_segments(audio_path: str, pipeline, cfg: Config) -> List[Dict]:
    rttm_path = run_pyannote_inference(audio_path, pipeline, cfg)
    segs = parse_pyannote_rttm(rttm_path)
    segs = [s for s in segs if s["dur"] >= cfg.min_seg_dur_sec]
    return segs


# =========================================================
# Segment embedding extraction
# =========================================================

def extract_segment_embeddings_from_segments(
    audio_path: str,
    segments: List[Dict],
    embedder: ECAPAEmbedder,
    cfg: Config,
    wav: Optional[torch.Tensor] = None,
) -> List[Tuple[np.ndarray, float]]:
    """
    Returns list of (embedding, duration) pairs for valid segments.
    If *wav* is provided the audio is not reloaded from disk.
    """
    if wav is None:
        wav = load_audio_mono(audio_path, cfg.sample_rate)

    emb_dur_pairs: List[Tuple[np.ndarray, float]] = []
    for seg in segments:
        clip = crop_segment(wav, cfg.sample_rate, seg["start"], seg["end"])
        if clip.numel() < int(cfg.min_seg_dur_sec * cfg.sample_rate):
            continue
        try:
            emb = embedder.embed_waveform(clip)
            emb_dur_pairs.append((emb, seg["dur"]))
        except Exception:
            continue
    return emb_dur_pairs


# =========================================================
# Prototype building
# =========================================================

def build_child_prototypes_from_pyannote(
    train_df: pd.DataFrame,
    pyannote_pipeline,
    embedder: ECAPAEmbedder,
    cfg: Config,
):
    prototypes = {}
    stats = []

    pos_train = train_df[train_df["label"] == 1].copy()

    for child_id, sub in pos_train.groupby("child_id"):
        all_pairs: List[Tuple[np.ndarray, float]] = []

        for _, row in sub.iterrows():
            audio_path = row["audio_path"]
            segs = get_pyannote_segments(audio_path, pyannote_pipeline, cfg)
            pairs = extract_segment_embeddings_from_segments(
                audio_path, segs, embedder, cfg
            )
            all_pairs.extend(pairs)

            if len(all_pairs) >= cfg.max_enrollment_segments_per_child:
                all_pairs = all_pairs[:cfg.max_enrollment_segments_per_child]
                break

        if len(all_pairs) == 0:
            stats.append({
                "child_id": child_id,
                "n_segments": 0,
                "status": "no_valid_segments",
            })
            continue

        # Duration-weighted prototype
        embs = np.stack([e for e, _ in all_pairs], axis=0)
        weights = np.array([d for _, d in all_pairs])
        proto = np.average(embs, axis=0, weights=weights)
        prototypes[child_id] = l2_normalize(proto)

        stats.append({
            "child_id": child_id,
            "n_segments": int(len(all_pairs)),
            "status": "ok",
        })

    return prototypes, pd.DataFrame(stats)


# =========================================================
# Scoring
# =========================================================

def score_clip_with_pyannote_enrollment(
    audio_path: str,
    target_child_id: str,
    prototypes: Dict[str, np.ndarray],
    pyannote_pipeline,
    embedder: ECAPAEmbedder,
    cfg: Config,
) -> float:
    """
    Score a clip using duration-weighted mean cosine similarity to the
    enrolled prototype.  Returns 0.0 if the child has no prototype
    (unseen split) or if no valid segments are found.
    """
    if target_child_id not in prototypes:
        return 0.0

    segments = get_pyannote_segments(audio_path, pyannote_pipeline, cfg)
    if len(segments) == 0:
        return 0.0

    # Load audio once and pass through
    wav = load_audio_mono(audio_path, cfg.sample_rate)
    proto = prototypes[target_child_id]

    scored: List[Tuple[float, float]] = []
    for seg in segments:
        clip = crop_segment(wav, cfg.sample_rate, seg["start"], seg["end"])
        if clip.numel() < int(cfg.min_seg_dur_sec * cfg.sample_rate):
            continue
        try:
            emb = embedder.embed_waveform(clip)
            sim = cosine_similarity(emb, proto)
            scored.append((sim, seg["dur"]))
        except Exception:
            continue

    if len(scored) == 0:
        return 0.0

    # Duration-weighted mean similarity
    total_dur = sum(d for _, d in scored)
    return float(sum(s * d for s, d in scored) / total_dur)


def run_enrollment(df: pd.DataFrame, prototypes, pyannote_pipeline, embedder, cfg: Config):
    rows = []
    for _, row in df.iterrows():
        score = score_clip_with_pyannote_enrollment(
            audio_path=row["audio_path"],
            target_child_id=row["child_id"],
            prototypes=prototypes,
            pyannote_pipeline=pyannote_pipeline,
            embedder=embedder,
            cfg=cfg,
        )
        rows.append({
            "audio_path": row["audio_path"],
            "child_id": row["child_id"],
            "timepoint_norm": row["timepoint_norm"],
            "label": int(row["label"]),
            "prob": float(score),
        })
    return pd.DataFrame(rows)


def tune_similarity_threshold(val_pred_df: pd.DataFrame, cfg: Config):
    y_true = val_pred_df["label"].to_numpy()
    y_prob = val_pred_df["prob"].to_numpy()

    thresholds = np.linspace(
        cfg.similarity_threshold_min,
        cfg.similarity_threshold_max,
        cfg.similarity_threshold_steps,
    )

    best_t = 0.5
    best_metrics = compute_metrics(y_true, y_prob, threshold=best_t)
    best_f1 = best_metrics["f1"]

    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
            best_metrics = m

    return best_t, best_metrics


# =========================================================
# Main
# =========================================================

def main():
    os.makedirs(CFG.results_dir, exist_ok=True)
    os.makedirs(CFG.pyannote_cache_dir, exist_ok=True)

    save_json(asdict(CFG), os.path.join(CFG.results_dir, "config.json"))

    train_df, val_df, test_df = load_split(CFG.split_dir)

    print(f"Train: {len(train_df)} rows")
    print(f"Val:   {len(val_df)} rows")
    print(f"Test:  {len(test_df)} rows")

    print("Loading PyAnnote...")
    pyannote_pipeline = load_pyannote_pipeline(CFG)

    print("Loading ECAPA...")
    embedder = ECAPAEmbedder(CFG.ecapa_source, CFG.device)

    print("Building child prototypes from positive train clips...")
    prototypes, child_stats_df = build_child_prototypes_from_pyannote(
        train_df, pyannote_pipeline, embedder, CFG
    )
    child_stats_df.to_csv(os.path.join(CFG.results_dir, "child_prototype_stats.csv"), index=False)
    print(f"Built prototypes for {len(prototypes)} children.")

    # Verify no seen children are missing prototypes
    seen_children = set(train_df["child_id"].unique())
    missing = seen_children - set(prototypes.keys())
    if missing:
        print(f"WARNING: {len(missing)} seen children have no prototype "
              f"(diarizer found no segments): {missing}")

    print("Running PyAnnote + enrollment on val/test...")
    val_df_pred = run_enrollment(val_df, prototypes, pyannote_pipeline, embedder, CFG)
    test_df_pred = run_enrollment(test_df, prototypes, pyannote_pipeline, embedder, CFG)

    sim_t, val_metrics = tune_similarity_threshold(val_df_pred, CFG)

    val_df_pred = add_pred_labels(val_df_pred, sim_t)
    test_df_pred = add_pred_labels(test_df_pred, sim_t)

    val_df_pred.to_csv(os.path.join(CFG.results_dir, "val_predictions.csv"), index=False)
    test_df_pred.to_csv(os.path.join(CFG.results_dir, "test_predictions.csv"), index=False)

    save_json(
        {"threshold": sim_t, **val_metrics},
        os.path.join(CFG.results_dir, "val_metrics.json"),
    )

    test_metrics = compute_metrics(
        test_df_pred["label"].to_numpy(),
        test_df_pred["prob"].to_numpy(),
        threshold=sim_t,
    )
    save_json(
        {"threshold": sim_t, **test_metrics},
        os.path.join(CFG.results_dir, "test_metrics.json"),
    )

    per_timepoint_metrics(val_df_pred, sim_t).to_csv(
        os.path.join(CFG.results_dir, "val_metrics_by_timepoint.csv"),
        index=False,
    )
    per_timepoint_metrics(test_df_pred, sim_t).to_csv(
        os.path.join(CFG.results_dir, "test_metrics_by_timepoint.csv"),
        index=False,
    )

    print("Done.")
    print(f"Best threshold: {sim_t:.3f}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
