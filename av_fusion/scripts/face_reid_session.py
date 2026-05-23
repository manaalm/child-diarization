"""Per-session face re-identification via DBSCAN clustering (spec-021 US4 T077).

Pipeline:
  1. For each (child_id, timepoint_norm) session, gather all clip-level MP4s.
  2. Sample N frames per clip (default 8), detect faces via YuNet.
  3. Crop each face, run InsightFace ArcFace embedding (512-d).
  4. Cluster embeddings *within session* via DBSCAN(eps=0.4, min_samples=3) on
     cosine distance → cluster_id per (clip, track).
  5. Identify the *target-child cluster* per session: smallest mean-area cluster
     (children's faces are smaller). Pick by median bbox area across all detections.

Output: av_fusion/face_reid/session_clusters.csv
    columns: audio_path, clip_id, child_id, timepoint_norm, session,
             n_face_detections, target_child_cluster, target_cluster_size

CLI:
    python av_fusion/scripts/face_reid_session.py \
        --bids /orcd/scratch/bcs/001/sensein/sails/BIDS_data/final_bids-dataset \
        --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
        --out av_fusion/face_reid/session_clusters.csv

This is intentionally CPU-only (or single-GPU for InsightFace). Estimated runtime
on full SAILS BIDS: ~6-10 hours, depending on number of frames sampled.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "av_fusion" / "scripts"))
from face_utils import YuNetDetector  # noqa: E402


def load_arcface(device: str = "cpu"):
    """Lazy-load InsightFace ArcFace model."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(allowed_modules=["recognition"], providers=(
        ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]))
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    return app


def sample_frames(video_path: str, n_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, max(0, total - 1), num=min(n_frames, total)).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def crop_face(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    x, y, w, h = bbox[:4]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def embed_face(arcface, face_bgr: np.ndarray) -> np.ndarray | None:
    """ArcFace expects 112x112 BGR aligned face. We pass the raw crop and let
    InsightFace's recognition module handle internal preprocessing."""
    if face_bgr is None or face_bgr.size == 0:
        return None
    try:
        face_resized = cv2.resize(face_bgr, (112, 112))
        # Synthesise a minimal Face object: insightface.app.FaceAnalysis.get
        # normally wraps detection + recognition; here we drive the recognition
        # model directly via the underlying onnx session.
        rec = arcface.models.get("recognition")
        if rec is None:
            return None
        # InsightFace recognition models accept (1, 3, 112, 112) BGR-mean-norm
        # via their .get_feat() helper.
        emb = rec.get_feat(face_resized)
        emb = np.asarray(emb).flatten().astype("float32")
        return emb / (np.linalg.norm(emb) + 1e-9)
    except Exception:
        return None


def collect_session_face_records(
    metadata: pd.DataFrame,
    bids_root: Path,
    detector: YuNetDetector,
    arcface,
    n_frames: int,
    max_clips: int | None,
) -> list[dict]:
    """For every clip in metadata, sample frames, detect faces, embed each face."""
    records = []
    n_processed = 0
    for _, row in metadata.iterrows():
        if max_clips is not None and n_processed >= max_clips:
            break
        clip_id = row.get("clip_id", row.get("audio_path"))
        child_id = row["child_id"]
        timepoint = row["timepoint_norm"]
        video_path = row.get("video_path") or row.get("audio_path", "").replace(".wav", ".mp4")
        if not video_path or not os.path.isfile(video_path):
            continue

        frames = sample_frames(video_path, n_frames)
        if not frames:
            continue
        for fi, frame in enumerate(frames):
            faces = detector.detect(frame)
            for face in faces:
                bbox = face[:4]
                conf = face[4]
                if conf < 0.6:
                    continue
                crop = crop_face(frame, bbox)
                emb = embed_face(arcface, crop)
                if emb is None:
                    continue
                area = float(bbox[2] * bbox[3])
                records.append({
                    "audio_path": row.get("audio_path"),
                    "clip_id": clip_id,
                    "child_id": child_id,
                    "timepoint_norm": timepoint,
                    "session": f"{child_id}__{timepoint}",
                    "frame_idx": int(fi),
                    "bbox_area": area,
                    "embedding": emb,
                })
        n_processed += 1
        if n_processed % 25 == 0:
            print(f"  processed {n_processed} clips, {len(records)} face records")
    return records


def cluster_session(records: list[dict], eps: float, min_samples: int) -> list[dict]:
    """Per-session DBSCAN clustering. Returns records with `cluster_id` added.
    Cluster IDs are session-scoped (not unique across sessions)."""
    if not records:
        return []
    df = pd.DataFrame(records)
    out = []
    for session, sess_df in df.groupby("session"):
        embs = np.stack([np.asarray(e) for e in sess_df["embedding"]])
        if len(embs) < min_samples:
            sess_df = sess_df.copy()
            sess_df["cluster_id"] = -1
            out.append(sess_df)
            continue
        # cosine distance: 1 - cos(a,b); embeddings are L2-normalized, so
        # cos = a·b → distance = 1 - a·b.
        dist = 1.0 - embs @ embs.T
        dist = np.clip(dist, 0, 2)
        clust = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(dist)
        sess_df = sess_df.copy()
        sess_df["cluster_id"] = clust.labels_
        out.append(sess_df)
    return pd.concat(out, ignore_index=True)


def pick_target_child_cluster(clustered: pd.DataFrame) -> pd.DataFrame:
    """Per session, pick the cluster with the smallest median bbox area as the
    target-child cluster (heuristic: child faces are smaller than adult faces).
    Returns a DataFrame keyed on (audio_path) with the target cluster id."""
    out = []
    for session, sess_df in clustered.groupby("session"):
        valid = sess_df[sess_df["cluster_id"] >= 0]
        if valid.empty:
            target_cluster = -1
            cluster_size = 0
        else:
            cluster_stats = valid.groupby("cluster_id").agg(
                median_area=("bbox_area", "median"),
                n=("bbox_area", "count"),
            ).reset_index()
            target_row = cluster_stats.loc[cluster_stats["median_area"].idxmin()]
            target_cluster = int(target_row["cluster_id"])
            cluster_size = int(target_row["n"])
        for clip_id, clip_df in sess_df.groupby("clip_id"):
            audio_path = clip_df["audio_path"].iloc[0]
            child_id = clip_df["child_id"].iloc[0]
            timepoint = clip_df["timepoint_norm"].iloc[0]
            out.append({
                "audio_path": audio_path,
                "clip_id": clip_id,
                "child_id": child_id,
                "timepoint_norm": timepoint,
                "session": session,
                "n_face_detections": int(len(clip_df)),
                "target_child_cluster": target_cluster,
                "target_cluster_size": cluster_size,
            })
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids", type=Path, required=True)
    ap.add_argument("--metadata-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.4)
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata_csv)
    needed = ["child_id", "timepoint_norm", "audio_path"]
    for c in needed:
        if c not in metadata.columns:
            raise SystemExit(f"metadata-csv missing required column '{c}'")

    print(f"Loading detectors (device={args.device})...")
    detector = YuNetDetector()
    arcface = load_arcface(device=args.device)
    print(f"  arcface modules: {list(arcface.models.keys())}")

    print(f"Collecting face records over {len(metadata)} clips "
          f"(max_clips={args.max_clips})...")
    records = collect_session_face_records(
        metadata, args.bids, detector, arcface,
        args.n_frames, args.max_clips,
    )
    print(f"  {len(records)} face records collected")

    if not records:
        print("No face records — writing empty CSV and exiting.")
        pd.DataFrame(columns=[
            "audio_path", "clip_id", "child_id", "timepoint_norm",
            "session", "n_face_detections", "target_child_cluster",
            "target_cluster_size",
        ]).to_csv(args.out, index=False)
        return

    print(f"Clustering with DBSCAN(eps={args.eps}, min_samples={args.min_samples})...")
    clustered = cluster_session(records, args.eps, args.min_samples)
    target = pick_target_child_cluster(clustered)
    target.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(target)} rows over {target['session'].nunique()} sessions")
    n_no_target = int((target["target_child_cluster"] == -1).sum())
    print(f"  sessions with no detected target-child cluster: {n_no_target}")


if __name__ == "__main__":
    main()
