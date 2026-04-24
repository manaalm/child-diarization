"""Face detection, tracking, and visual quality utilities.

Provides:
- YuNetDetector: wraps cv2.FaceDetectorYN for frame-level face detection.
- IouCentroidTracker: simple IoU-based tracker for short clips (≤60 s, ≤4 faces).
- visual_quality_score: Laplacian-variance + brightness estimate.
- child_candidate_score: smallest-face-track heuristic for child identification.
- compute_visual_eligibility: weighted combination of sub-scores.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

class YuNetDetector:
    """Wrapper around cv2.FaceDetectorYN (YuNet); requires OpenCV 4.8+."""

    def __init__(
        self,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5,
    ) -> None:
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._last_size: Tuple[int, int] = (-1, -1)

    def _ensure_detector(self, h: int, w: int) -> None:
        if self._detector is None or (h, w) != self._last_size:
            self._detector = cv2.FaceDetectorYN.create(
                model="",
                config="",
                input_size=(w, h),
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k,
            )
            self._last_size = (h, w)

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Detect faces in a BGR frame.

        Returns list of (x, y, w, h, confidence) in pixel coordinates.
        """
        h, w = frame.shape[:2]
        try:
            self._ensure_detector(h, w)
            _, faces = self._detector.detect(frame)
        except cv2.error:
            return []

        if faces is None:
            return []

        results = []
        for face in faces:
            x, y, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
            conf = float(face[-1])
            results.append((x, y, fw, fh, conf))
        return results


class MediaPipeDetector:
    """Fallback face detector using MediaPipe Face Detection."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        import mediapipe as mp  # optional dependency
        self._mp_face = mp.solutions.face_detection
        self._detector = self._mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=min_detection_confidence
        )

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Detect faces in a BGR frame; returns (x, y, w, h, confidence)."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        if not results.detections:
            return []
        out = []
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x = max(0.0, bb.xmin) * w
            y = max(0.0, bb.ymin) * h
            fw = bb.width * w
            fh = bb.height * h
            conf = float(det.score[0])
            out.append((x, y, fw, fh, conf))
        return out


def make_detector(name: str = "yunet") -> YuNetDetector | MediaPipeDetector:
    if name == "mediapipe":
        return MediaPipeDetector()
    return YuNetDetector()


# ---------------------------------------------------------------------------
# Face tracking
# ---------------------------------------------------------------------------

class IouCentroidTracker:
    """Simple IoU-based centroid tracker for short clips with few faces.

    Assigns detections to existing tracks by IoU overlap. Creates new tracks
    for unmatched detections. Does not delete tracks within a clip (they may
    be temporarily occluded).
    """

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
        self._tracks: Dict[int, List[Tuple[int, Tuple[float, float, float, float, float]]]] = {}
        self._next_id = 0

    def reset(self) -> None:
        self._tracks = {}
        self._next_id = 0

    def update(
        self, frame_idx: int, detections: List[Tuple[float, float, float, float, float]]
    ) -> Dict[int, Tuple[float, float, float, float, float]]:
        """Update tracks with current frame detections.

        Returns dict mapping track_id → (x, y, w, h, conf) for this frame.
        """
        if not detections:
            return {}

        if not self._tracks:
            assignments = {}
            for det in detections:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = [(frame_idx, det)]
                assignments[tid] = det
            return assignments

        # Compute IoU between last known bbox of each track and each detection
        track_ids = list(self._tracks.keys())
        track_last = [self._tracks[tid][-1][1] for tid in track_ids]

        assigned_det_indices = set()
        assignments = {}

        for i, tid in enumerate(track_ids):
            best_iou, best_j = 0.0, -1
            for j, det in enumerate(detections):
                if j in assigned_det_indices:
                    continue
                iou = _bbox_iou(track_last[i], det)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= self.iou_threshold and best_j >= 0:
                self._tracks[tid].append((frame_idx, detections[best_j]))
                assignments[tid] = detections[best_j]
                assigned_det_indices.add(best_j)

        # Unmatched detections → new tracks
        for j, det in enumerate(detections):
            if j not in assigned_det_indices:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = [(frame_idx, det)]
                assignments[tid] = det

        return assignments

    def get_tracks(self) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float, float]]]]:
        """Return all tracks accumulated so far: {track_id: [(frame_idx, (x,y,w,h,conf)), ...]}."""
        return dict(self._tracks)


def _bbox_iou(
    a: Tuple[float, float, float, float, float],
    b: Tuple[float, float, float, float, float],
) -> float:
    """Compute IoU between two bounding boxes given as (x, y, w, h, conf)."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    inter_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_y = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_x * inter_y
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Visual quality
# ---------------------------------------------------------------------------

def visual_quality_score(frames: List[np.ndarray]) -> float:
    """Estimate visual quality as mean Laplacian variance (blur proxy) + brightness.

    Returns float in [0, 1] — 0 = very blurry/dark, 1 = sharp and well-lit.
    """
    if not frames:
        return 0.0

    lap_vars, brightnesses = [], []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        lap_vars.append(lap_var)
        brightnesses.append(float(gray.mean()))

    # Normalise Laplacian variance: clip at 300 → maps to 1.0
    blur_score = min(1.0, float(np.mean(lap_vars)) / 300.0)
    # Normalise brightness: [0, 255] → [0, 1]; prefer 60–200 range
    brightness = float(np.mean(brightnesses))
    brightness_score = min(1.0, brightness / 128.0) if brightness < 128 else min(1.0, (255 - brightness) / 127.0 + 0.5)
    brightness_score = max(0.0, min(1.0, brightness_score))

    return 0.7 * blur_score + 0.3 * brightness_score


# ---------------------------------------------------------------------------
# Child candidate and eligibility
# ---------------------------------------------------------------------------

def child_candidate_score(
    tracks: Dict[int, List],
    frame_count: int,
    frame_area: float,
) -> Tuple[float, float, float]:
    """Compute child visibility score using smallest-face-track heuristic.

    The target child in naturalistic home video is typically the smallest
    person visible. The smallest median bounding box track is the child candidate.

    Returns:
        (child_visible_score, off_camera_likely_score, max_track_fraction)
        All in [0, 1].
    """
    if not tracks or frame_count == 0:
        return 0.0, 1.0, 0.0

    # Per-track: median box area, duration fraction
    track_stats = {}
    for tid, frames_list in tracks.items():
        areas = [(fw * fh) / frame_area for (_, (_, _, fw, fh, _)) in frames_list]
        track_stats[tid] = {
            "median_area": float(np.median(areas)),
            "fraction": len(frames_list) / frame_count,
        }

    # Smallest-area track = child candidate
    candidate_tid = min(track_stats, key=lambda t: track_stats[t]["median_area"])
    candidate = track_stats[candidate_tid]
    max_track_fraction = max(v["fraction"] for v in track_stats.values())

    # Child visible score: weight duration × inverse area (small face = child)
    # Area fraction for child-sized face is typically 0.01–0.10 of frame
    candidate_area = candidate["median_area"]
    area_score = min(1.0, 0.05 / (candidate_area + 1e-6)) if candidate_area < 0.2 else 0.1
    child_visible = min(1.0, candidate["fraction"] * area_score * 5.0)

    # Off-camera likely: no face track covers most of the clip
    off_camera = max(0.0, 1.0 - max_track_fraction)

    return float(child_visible), float(off_camera), float(max_track_fraction)


def compute_visual_eligibility(
    child_visible: float,
    track_fraction: float,
    quality: float,
    detection_confidence: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Weighted combination of visual sub-scores → eligibility score in [0, 1].

    Default weights from research.md §5:
        child_visible: 0.40, track_fraction: 0.25, quality: 0.20, confidence: 0.15
    """
    if weights is None:
        weights = {"child_visible": 0.40, "track_fraction": 0.25, "quality": 0.20, "detection_confidence": 0.15}
    score = (
        weights["child_visible"] * child_visible
        + weights["track_fraction"] * track_fraction
        + weights["quality"] * quality
        + weights["detection_confidence"] * detection_confidence
    )
    return float(np.clip(score, 0.0, 1.0))
