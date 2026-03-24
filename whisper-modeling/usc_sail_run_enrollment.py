import os
import json
import hashlib
import subprocess
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


# =========================================================
# Config
# =========================================================

@dataclass
class Config:
    # seen-child split directory
    split_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/seen_child_splits"

    # output directory for this experiment
    results_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/usc_sail_enrollment_runs"

    # audio / segment settings
    sample_rate: int = 16000
    min_seg_dur_sec: float = 0.4
    max_enrollment_segments_per_child: int = 200

    # ECAPA
    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # role-only threshold tuning
    duration_threshold_grid: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)

    # enrollment threshold tuning
    similarity_threshold_min: float = 0.1
    similarity_threshold_max: float = 0.95
    similarity_threshold_steps: int = 171

    # USC-SAIL paths
    usc_sail_repo_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling"
    usc_sail_script: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/scripts/infer_long_wav_files.py"
    usc_sail_model_path: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/whisper-base_rank8_pretrained_50k.pt"
    usc_sail_python: str = "python"

    usc_window_size: float = 10.0
    usc_stride: float = 5.0

    # caches
    segment_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/usc_sail_segment_cache"
    rttm_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/usc_sail_rttm_cache"


CFG = Config()


# =========================================================
# General utilities
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


# =========================================================
# USC-SAIL wrappers
# =========================================================

def audio_to_cache_id(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode("utf-8")).hexdigest()


def cache_path_for_audio(audio_path: str, cache_dir: str) -> str:
    cache_id = audio_to_cache_id(audio_path)
    return os.path.join(cache_dir, f"{cache_id}.json")


def rttm_cache_path_for_audio(audio_path: str, cache_dir: str) -> str:
    """
    Must match the patched infer_long_wav_files.py naming scheme.
    """
    cache_id = audio_to_cache_id(audio_path)
    stem = Path(audio_path).stem
    return os.path.join(cache_dir, f"{stem}__{cache_id}.rttm")


def parse_rttm_for_child_segments(rttm_path: str) -> List[Dict[str, float]]:
    """
    Parses RTTM lines like:
    SPEAKER rec 1 start dur <NA> <NA> CHI <NA> <NA>
    and returns only CHI segments.
    """
    child_segments = []

    if not os.path.exists(rttm_path):
        return child_segments

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
            label = parts[7]

            if label == "CHI":
                child_segments.append({
                    "start": start,
                    "end": start + dur,
                })

    return child_segments


def run_usc_sail_inference(audio_path: str, cfg: Config) -> str:
    """
    Runs USC-SAIL inference if RTTM is not already cached.
    Returns path to cached RTTM.
    """
    os.makedirs(cfg.rttm_cache_dir, exist_ok=True)

    target_rttm = rttm_cache_path_for_audio(audio_path, cfg.rttm_cache_dir)
    if os.path.exists(target_rttm):
        return target_rttm

    cmd = [
        cfg.usc_sail_python,
        cfg.usc_sail_script,
        "--wav_file", audio_path,
        "--out_dir", cfg.rttm_cache_dir,
        "--model_path", cfg.usc_sail_model_path,
        "--device", "cuda" if "cuda" in cfg.device else "cpu",
        "--window_size", str(cfg.usc_window_size),
        "--stride", str(cfg.usc_stride),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = cfg.usc_sail_repo_dir

    subprocess.run(
        cmd,
        cwd=cfg.usc_sail_repo_dir,
        env=env,
        check=True,
    )

    if not os.path.exists(target_rttm):
        raise FileNotFoundError(
            f"USC-SAIL finished but expected RTTM not found: {target_rttm}"
        )

    return target_rttm


def get_child_segments_usc_sail(audio_path: str, cfg: Config) -> List[Dict[str, float]]:
    rttm_path = run_usc_sail_inference(audio_path, cfg)
    return parse_rttm_for_child_segments(rttm_path)


def get_child_segments_cached(audio_path: str, cfg: Config) -> List[Dict[str, float]]:
    os.makedirs(cfg.segment_cache_dir, exist_ok=True)
    cp = cache_path_for_audio(audio_path, cfg.segment_cache_dir)

    if os.path.exists(cp):
        with open(cp, "r") as f:
            return json.load(f)

    segs = get_child_segments_usc_sail(audio_path, cfg)
    with open(cp, "w") as f:
        json.dump(segs, f)

    return segs


# =========================================================
# Audio / embedding helpers
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


def l2_normalize(x: np.ndarray, eps: float = 1e-8):
    n = np.linalg.norm(x)
    return x / max(n, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


class ECAPAEmbedder:
    def __init__(self, source: str, device: str):
        run_opts = {"device": device}
        self.model = EncoderClassifier.from_hparams(
            source=source,
            run_opts=run_opts,
        )
        self.device = device

    @torch.no_grad()
    def embed_waveform(self, wav_1d: torch.Tensor) -> np.ndarray:
        wav = wav_1d.unsqueeze(0).to(self.device)  # [1, T]
        emb = self.model.encode_batch(wav)
        return emb.squeeze().detach().cpu().numpy()


# =========================================================
# Segment extraction / features
# =========================================================

def get_valid_child_segments(audio_path: str, cfg: Config) -> List[Dict[str, float]]:
    segs = get_child_segments_cached(audio_path, cfg)
    out = []
    for seg in segs:
        start = float(seg["start"])
        end = float(seg["end"])
        dur = end - start
        if dur >= cfg.min_seg_dur_sec:
            out.append({"start": start, "end": end, "dur": dur})
    return out


def total_child_duration(audio_path: str, cfg: Config) -> float:
    segs = get_valid_child_segments(audio_path, cfg)
    return float(sum(seg["dur"] for seg in segs))


def extract_segment_embeddings(audio_path: str, embedder: ECAPAEmbedder, cfg: Config) -> List[np.ndarray]:
    wav = load_audio_mono(audio_path, cfg.sample_rate)
    segs = get_valid_child_segments(audio_path, cfg)

    embs = []
    for seg in segs:
        clip = crop_segment(wav, cfg.sample_rate, seg["start"], seg["end"])
        if clip.numel() < int(cfg.min_seg_dur_sec * cfg.sample_rate):
            continue
        try:
            embs.append(embedder.embed_waveform(clip))
        except Exception:
            continue
    return embs


# =========================================================
# Enrollment building
# =========================================================

def build_child_prototypes(train_df: pd.DataFrame, embedder: ECAPAEmbedder, cfg: Config):
    prototypes = {}
    stats = []

    pos_train = train_df[train_df["label"] == 1].copy()

    for child_id, sub in pos_train.groupby("child_id"):
        all_embs = []

        for _, row in sub.iterrows():
            audio_path = row["audio_path"]
            embs = extract_segment_embeddings(audio_path, embedder, cfg)
            all_embs.extend(embs)

            if len(all_embs) >= cfg.max_enrollment_segments_per_child:
                all_embs = all_embs[:cfg.max_enrollment_segments_per_child]
                break

        if len(all_embs) == 0:
            stats.append({
                "child_id": child_id,
                "n_segments": 0,
                "status": "no_valid_segments",
            })
            continue

        proto = np.mean(np.stack(all_embs, axis=0), axis=0)
        prototypes[child_id] = l2_normalize(proto)

        stats.append({
            "child_id": child_id,
            "n_segments": int(len(all_embs)),
            "status": "ok",
        })

    return prototypes, pd.DataFrame(stats)


# =========================================================
# Role-only baseline
# =========================================================

def run_role_only(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        dur = total_child_duration(row["audio_path"], cfg)
        rows.append({
            "audio_path": row["audio_path"],
            "child_id": row["child_id"],
            "timepoint_norm": row["timepoint_norm"],
            "label": int(row["label"]),
            "score_duration_sec": float(dur),
        })
    return pd.DataFrame(rows)


def tune_role_only_threshold(val_role_df: pd.DataFrame, cfg: Config):
    y_true = val_role_df["label"].to_numpy()

    best_t = cfg.duration_threshold_grid[0]
    best_prob = (val_role_df["score_duration_sec"] >= best_t).astype(float).to_numpy()
    best_metrics = compute_metrics(y_true, best_prob, threshold=0.5)
    best_f1 = best_metrics["f1"]

    for t in cfg.duration_threshold_grid:
        prob = (val_role_df["score_duration_sec"] >= t).astype(float).to_numpy()
        m = compute_metrics(y_true, prob, threshold=0.5)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t
            best_metrics = m

    return float(best_t), best_metrics


def role_df_to_pred_df(role_df: pd.DataFrame, threshold_sec: float):
    out = role_df.copy()
    out["prob"] = (out["score_duration_sec"] >= threshold_sec).astype(float)
    out["pred_label"] = out["prob"].astype(int)
    return out


# =========================================================
# Enrollment inference
# =========================================================

def score_clip_with_enrollment(
    audio_path: str,
    target_child_id: str,
    prototypes: Dict[str, np.ndarray],
    embedder: ECAPAEmbedder,
    cfg: Config,
) -> float:
    if target_child_id not in prototypes:
        return 0.0

    seg_embs = extract_segment_embeddings(audio_path, embedder, cfg)
    if len(seg_embs) == 0:
        return 0.0

    proto = prototypes[target_child_id]
    sims = [cosine_similarity(emb, proto) for emb in seg_embs]
    return float(max(sims))


def run_enrollment(df: pd.DataFrame, prototypes: Dict[str, np.ndarray], embedder: ECAPAEmbedder, cfg: Config):
    rows = []
    for _, row in df.iterrows():
        score = score_clip_with_enrollment(
            audio_path=row["audio_path"],
            target_child_id=row["child_id"],
            prototypes=prototypes,
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
    os.makedirs(CFG.segment_cache_dir, exist_ok=True)
    os.makedirs(CFG.rttm_cache_dir, exist_ok=True)

    save_json(asdict(CFG), os.path.join(CFG.results_dir, "config.json"))

    train_df, val_df, test_df = load_split(CFG.split_dir)

    print(f"Train: {len(train_df)} rows")
    print(f"Val:   {len(val_df)} rows")
    print(f"Test:  {len(test_df)} rows")
    print(f"Train children: {train_df['child_id'].nunique()}")
    print(f"Val children:   {val_df['child_id'].nunique()}")
    print(f"Test children:  {test_df['child_id'].nunique()}")

    # -----------------------------------------------------
    # Role-only baseline
    # -----------------------------------------------------
    print("\nRunning role-only baseline...")
    val_role_df = run_role_only(val_df, CFG)
    test_role_df = run_role_only(test_df, CFG)

    role_t, role_val_metrics = tune_role_only_threshold(val_role_df, CFG)

    val_role_pred = role_df_to_pred_df(val_role_df, role_t)
    test_role_pred = role_df_to_pred_df(test_role_df, role_t)

    val_role_pred.to_csv(os.path.join(CFG.results_dir, "role_only_val_predictions.csv"), index=False)
    test_role_pred.to_csv(os.path.join(CFG.results_dir, "role_only_test_predictions.csv"), index=False)

    save_json(
        {"threshold_sec": role_t, **role_val_metrics},
        os.path.join(CFG.results_dir, "role_only_val_metrics.json"),
    )

    role_test_metrics = compute_metrics(
        test_role_pred["label"].to_numpy(),
        test_role_pred["prob"].to_numpy(),
        threshold=0.5,
    )
    save_json(
        {"threshold_sec": role_t, **role_test_metrics},
        os.path.join(CFG.results_dir, "role_only_test_metrics.json"),
    )

    per_timepoint_metrics(val_role_pred, 0.5).to_csv(
        os.path.join(CFG.results_dir, "role_only_val_metrics_by_timepoint.csv"),
        index=False,
    )
    per_timepoint_metrics(test_role_pred, 0.5).to_csv(
        os.path.join(CFG.results_dir, "role_only_test_metrics_by_timepoint.csv"),
        index=False,
    )

    # -----------------------------------------------------
    # Enrollment
    # -----------------------------------------------------
    print("\nLoading ECAPA...")
    embedder = ECAPAEmbedder(CFG.ecapa_source, CFG.device)

    print("Building child prototypes from positive train clips...")
    prototypes, child_stats_df = build_child_prototypes(train_df, embedder, CFG)
    child_stats_df.to_csv(os.path.join(CFG.results_dir, "child_prototype_stats.csv"), index=False)
    print(f"Built prototypes for {len(prototypes)} children.")

    print("Running enrollment on val/test...")
    val_enroll_df = run_enrollment(val_df, prototypes, embedder, CFG)
    test_enroll_df = run_enrollment(test_df, prototypes, embedder, CFG)

    sim_t, val_sim_metrics = tune_similarity_threshold(val_enroll_df, CFG)

    val_enroll_df = add_pred_labels(val_enroll_df, sim_t)
    test_enroll_df = add_pred_labels(test_enroll_df, sim_t)

    val_enroll_df.to_csv(os.path.join(CFG.results_dir, "enroll_val_predictions.csv"), index=False)
    test_enroll_df.to_csv(os.path.join(CFG.results_dir, "enroll_test_predictions.csv"), index=False)

    save_json(
        {"threshold": sim_t, **val_sim_metrics},
        os.path.join(CFG.results_dir, "enroll_val_metrics.json"),
    )

    test_sim_metrics = compute_metrics(
        test_enroll_df["label"].to_numpy(),
        test_enroll_df["prob"].to_numpy(),
        threshold=sim_t,
    )
    save_json(
        {"threshold": sim_t, **test_sim_metrics},
        os.path.join(CFG.results_dir, "enroll_test_metrics.json"),
    )

    per_timepoint_metrics(val_enroll_df, sim_t).to_csv(
        os.path.join(CFG.results_dir, "enroll_val_metrics_by_timepoint.csv"),
        index=False,
    )
    per_timepoint_metrics(test_enroll_df, sim_t).to_csv(
        os.path.join(CFG.results_dir, "enroll_test_metrics_by_timepoint.csv"),
        index=False,
    )

    print("\nDone.")
    print(f"Role-only threshold (sec): {role_t}")
    print(f"Enrollment threshold: {sim_t:.3f}")
    print(f"Enrollment test metrics: {test_sim_metrics}")


if __name__ == "__main__":
    main()