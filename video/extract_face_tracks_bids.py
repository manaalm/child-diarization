"""Batch S3FD face-track extraction for BIDS clips missing the per-frame cache.

Reads `whisper-modeling/seen_child_splits/master_with_split.csv`, filters to
rows whose `av_fusion/face_track_cache/<md5(BidsProcessed)>.json` is missing,
and runs the same S3FD + IoU tracking pipeline as `video/run_asd.py`
(`detect_faces_in_video`) over them. The detector is loaded once and reused
across all clips. Each clip's cache is written incrementally so the job
resumes cleanly after a preempt/requeue.

Usage:
  video/.venv/bin/python video/extract_face_tracks_bids.py
  video/.venv/bin/python video/extract_face_tracks_bids.py --split test
  video/.venv/bin/python video/extract_face_tracks_bids.py --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_HERE = Path(__file__).parent.resolve()
_REPO = _HERE.parent
_TALKNET_DIR = _HERE / "TalkNet-ASD"

if str(_TALKNET_DIR) not in sys.path:
    sys.path.insert(0, str(_TALKNET_DIR))

CACHE_DIR = _REPO / "av_fusion/face_track_cache"
SPLIT_CSV = _REPO / "whisper-modeling/seen_child_splits/master_with_split.csv"


def cache_key(path: str) -> str:
    return hashlib.md5(path.encode("utf-8")).hexdigest()


def _compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


def _mean_bbox_area(frames):
    if not frames:
        return 0.0
    s = 0.0
    n = 0
    for f in frames:
        x1, y1, x2, y2 = f["bbox"]
        s += max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        n += 1
    return float(s / n) if n else 0.0


def load_s3fd(device: str):
    os.chdir(str(_TALKNET_DIR))
    from model.faceDetector.s3fd import S3FD  # noqa: E402
    return S3FD(device=device)


def detect_one(video_path: str, detector, conf_threshold: float = 0.9):
    """Stream a video through S3FD + IoU tracker. Returns finished tracks."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    active_tracks: list[dict] = []
    finished_tracks: list[dict] = []
    track_id_counter = 0
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                dets = detector.detect_faces(rgb, conf_th=conf_threshold, scales=[0.25])
            except Exception:
                dets = np.zeros((0, 5), dtype=np.float32)
            dets = np.array(dets) if len(dets) > 0 else np.zeros((0, 5), dtype=np.float32)

            matched_det_idxs = set()
            new_active: list[dict] = []

            for track in active_tracks:
                last_box = track["frames"][-1]["bbox"]
                best_iou, best_det_idx = 0.0, -1
                for di, det in enumerate(dets):
                    if di in matched_det_idxs:
                        continue
                    iou = _compute_iou(last_box, det[:4].tolist())
                    if iou > best_iou:
                        best_iou, best_det_idx = iou, di
                if best_iou >= 0.5 and best_det_idx >= 0:
                    matched_det_idxs.add(best_det_idx)
                    det = dets[best_det_idx]
                    track["frames"].append({
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                        "bbox": det[:4].tolist(),
                        "score": float(det[4]),
                    })
                    new_active.append(track)
                else:
                    if len(track["frames"]) >= 10:
                        track["mean_area"] = _mean_bbox_area(track["frames"])
                        finished_tracks.append(track)

            for di, det in enumerate(dets):
                if di not in matched_det_idxs:
                    new_active.append({
                        "track_id": track_id_counter,
                        "frames": [{
                            "frame_idx": frame_idx,
                            "timestamp": timestamp,
                            "bbox": det[:4].tolist(),
                            "score": float(det[4]),
                        }],
                        "mean_area": 0.0,
                    })
                    track_id_counter += 1

            active_tracks = new_active
            frame_idx += 1
    finally:
        cap.release()

    for track in active_tracks:
        if len(track["frames"]) >= 10:
            track["mean_area"] = _mean_bbox_area(track["frames"])
            finished_tracks.append(track)

    for i, t in enumerate(finished_tracks):
        t["track_id"] = i

    return finished_tracks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N missing clips (for smoke tests).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List missing clips and exit; do not load S3FD.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SPLIT_CSV)
    if args.split != "all":
        df = df[df["split"] == args.split]
    df = df[df["audio_exists"] == True].reset_index(drop=True)

    df["cache_path"] = df["BidsProcessed"].astype(str).apply(
        lambda p: CACHE_DIR / f"{cache_key(p)}.json")
    df["has_cache"] = df["cache_path"].apply(lambda p: p.exists())
    df["video_exists"] = df["BidsProcessed"].astype(str).apply(lambda p: Path(p).exists())

    needs = df[(~df["has_cache"]) & df["video_exists"]].reset_index(drop=True)
    print(f"Scope: split={args.split}", flush=True)
    print(f"  rows audio_exists:    {len(df)}", flush=True)
    print(f"  rows with cache:      {df['has_cache'].sum()}", flush=True)
    print(f"  rows missing cache:   {len(needs)}", flush=True)
    print(f"  rows missing video:   {(~df['video_exists']).sum()}", flush=True)

    if args.limit is not None:
        needs = needs.head(args.limit).reset_index(drop=True)
        print(f"  --limit applied:      {len(needs)}", flush=True)

    if args.dry_run or len(needs) == 0:
        print("[dry-run / nothing to do]", flush=True)
        return

    print(f"Loading S3FD on device={args.device} ...", flush=True)
    t0 = time.time()
    detector = load_s3fd(args.device)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    t_start = time.time()
    n_total = len(needs)
    n_done = 0
    for i, row in enumerate(needs.itertuples(index=False)):
        vp = str(row.BidsProcessed)
        cp = Path(row.cache_path)
        if cp.exists():
            n_done += 1
            continue
        t_clip = time.time()
        try:
            tracks = detect_one(vp, detector)
        except Exception as e:
            print(f"  [{i+1}/{n_total}] FAIL {os.path.basename(vp)}: {e}", flush=True)
            continue
        with open(cp, "w") as f:
            json.dump(tracks, f)
        n_done += 1
        dt = time.time() - t_clip
        if (i + 1) % 5 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_start
            eta_min = elapsed / max(1, n_done) * (n_total - n_done) / 60.0
            print(f"  [{i+1}/{n_total}]  {os.path.basename(vp)}  tracks={len(tracks)}  "
                  f"clip={dt:.1f}s  elapsed={elapsed/60:.1f}min  eta={eta_min:.1f}min",
                  flush=True)

    print(f"Done. Wrote face tracks for {n_done}/{n_total} clips into {CACHE_DIR}",
          flush=True)


if __name__ == "__main__":
    main()
