"""
run_asd.py — Active Speaker Detection inference for SAILS BIDS videos.

Derives the video path from the audio path (BIDS naming convention), runs
S3FD face detection + IoU tracking, then runs TalkNet-ASD or TS-TalkNet
inference to identify child vocalization segments. Output is an RTTM file
with CHI-labeled segments.

Usage:
    python run_asd.py \
        --audio_path /path/to/sub-ID_..._audio.wav \
        --model talknet_asd \
        --out_rttm /path/to/output.rttm \
        --face_cache_dir /path/to/face_cache/ \
        --pretrain_dir /path/to/pretrain/

    python run_asd.py \
        --audio_path /path/to/sub-ID_..._audio.wav \
        --model ts_talknet \
        --ref_audio /path/to/ref_audio.wav \
        --out_rttm /path/to/output.rttm \
        --face_cache_dir /path/to/face_cache/ \
        --pretrain_dir /path/to/pretrain/

Called as a subprocess by pyannote/video_asd.py. All heavy imports happen
inside this script so the calling process stays lightweight.

Inference APIs follow demoTalkNet.py and ts-talkNet.py evaluate_network()
exactly: python_speech_features MFCC at 100fps, grayscale 112×112 face crops
at 25fps, multi-duration (1–6 s) windows averaged.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import python_speech_features as psf
import soundfile as sf
import torch
import torchaudio


# ---------------------------------------------------------------------------
# Repo path setup (TalkNet-ASD and TS-TalkNet must be cloned under video/)
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).parent.resolve()
_TALKNET_DIR = _THIS_DIR / "TalkNet-ASD"
_TSTALKNET_DIR = _THIS_DIR / "TS-TalkNet"

for _repo_dir in [_TALKNET_DIR, _TSTALKNET_DIR]:
    if _repo_dir.is_dir() and str(_repo_dir) not in sys.path:
        sys.path.insert(0, str(_repo_dir))


# ---------------------------------------------------------------------------
# Path derivation
# ---------------------------------------------------------------------------

def derive_video_path(audio_path: str) -> str:
    """Convert BIDS audio path → BIDS processed video path.

    Pattern: *_audio.wav → *_desc-processed_beh.mp4
    """
    if not audio_path.endswith("_audio.wav"):
        raise ValueError(
            f"Cannot derive video path: audio_path does not end with '_audio.wav': {audio_path}"
        )
    video_path = audio_path.replace("_audio.wav", "_desc-processed_beh.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found — Providence and Playlogue are audio-only datasets. "
            f"Video ASD requires SAILS BIDS preprocessed .mp4 files.\n"
            f"Expected: {video_path}"
        )
    return video_path


def cache_key(path: str) -> str:
    return hashlib.md5(path.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio_16k(audio_path: str) -> Tuple[np.ndarray, int]:
    """Load audio as 16kHz mono numpy array."""
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # (channels, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0).numpy(), 16000


def compute_mfcc(audio_np: np.ndarray, sr: int = 16000) -> np.ndarray:
    """13-dim MFCC at 100 fps — matches TalkNet-ASD convention exactly."""
    # audio_np must be int16 or float32; psf handles both
    if audio_np.dtype != np.int16:
        audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    else:
        audio_int16 = audio_np
    return psf.mfcc(audio_int16, sr, numcep=13, winlen=0.025, winstep=0.010)


# ---------------------------------------------------------------------------
# Face detection helpers
# ---------------------------------------------------------------------------

def _crop_face(frame_rgb: np.ndarray, bbox: List[float], size: int = 112) -> np.ndarray:
    """Crop and resize face region to (size × size) RGB."""
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size, size, 3), dtype=np.uint8)
    crop = frame_rgb[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size))


def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _mean_bbox_area(frames: List[Dict]) -> float:
    areas = [(f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]) for f in frames]
    return float(np.mean(areas)) if areas else 0.0


def _track_to_bboxes(track: Dict) -> List:
    """Convert a track dict to a flat per-frame bbox list.

    detect_faces_in_video stores bboxes as track["frames"][i]["bbox"].
    Legacy callers expect track["bboxes"] = [bbox_or_None, ...] indexed by frame.
    This helper bridges both formats.
    """
    if "bboxes" in track:
        return track["bboxes"]
    frames = track.get("frames", [])
    if not frames:
        return []
    max_idx = max(f["frame_idx"] for f in frames)
    frame_to_bbox = {f["frame_idx"]: f["bbox"] for f in frames}
    return [frame_to_bbox.get(i) for i in range(max_idx + 1)]


# ---------------------------------------------------------------------------
# S3FD face detection
# ---------------------------------------------------------------------------

def _load_s3fd(device: str) -> object:
    """Load S3FD from TalkNet-ASD repo.

    S3FD's __init__.py uses a CWD-relative PATH_WEIGHT and auto-downloads
    the checkpoint via gdown if missing. We set CWD to _TALKNET_DIR so that
    both the path check and the download land in the right place.
    """
    if not _TALKNET_DIR.is_dir():
        raise FileNotFoundError(
            f"TalkNet-ASD repo not found at {_TALKNET_DIR}. "
            f"Clone it: git clone https://github.com/TaoRuijie/TalkNet-ASD {_TALKNET_DIR}"
        )
    os.chdir(str(_TALKNET_DIR))
    # Import triggers module-level PATH_WEIGHT check and auto-download
    from model.faceDetector.s3fd import S3FD  # noqa
    return S3FD(device=device)


def detect_faces_in_video(
    video_path: str,
    face_cache_dir: str,
    device: str = "cuda",
    conf_threshold: float = 0.9,
) -> List[Dict]:
    """Run S3FD on video frames; track with IoU; cache JSON.

    Returns face tracks: [{track_id, frames:[{frame_idx, timestamp, bbox}], mean_area}]
    """
    os.makedirs(face_cache_dir, exist_ok=True)
    cache_path = os.path.join(face_cache_dir, f"{cache_key(video_path)}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    print(f"  Face detection: {os.path.basename(video_path)}", flush=True)
    detector = _load_s3fd(device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    active_tracks: List[Dict] = []
    finished_tracks: List[Dict] = []
    track_id_counter = 0
    frame_idx = 0

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
        new_active: List[Dict] = []

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

    cap.release()

    for track in active_tracks:
        if len(track["frames"]) >= 10:
            track["mean_area"] = _mean_bbox_area(track["frames"])
            finished_tracks.append(track)

    for i, t in enumerate(finished_tracks):
        t["track_id"] = i

    with open(cache_path, "w") as f:
        json.dump(finished_tracks, f)

    print(f"  Found {len(finished_tracks)} face tracks", flush=True)
    return finished_tracks


# ---------------------------------------------------------------------------
# TalkNet-ASD inference
# ---------------------------------------------------------------------------

def _load_talknet(pretrain_dir: str):
    """Load TalkNet-ASD. Auto-downloads TalkSet checkpoint via gdown if absent."""
    if not _TALKNET_DIR.is_dir():
        raise FileNotFoundError(
            f"TalkNet-ASD repo not found at {_TALKNET_DIR}. "
            f"Clone: git clone https://github.com/TaoRuijie/TalkNet-ASD {_TALKNET_DIR}"
        )
    os.chdir(str(_TALKNET_DIR))

    try:
        from talkNet import talkNet  # noqa
    except ImportError as e:
        raise ImportError(
            f"Could not import talkNet from {_TALKNET_DIR}. "
            f"Clone TalkNet-ASD repo first.\nOriginal error: {e}"
        )

    ckpt = os.path.join(pretrain_dir, "talknet_asd.model")
    if not os.path.exists(ckpt):
        os.makedirs(pretrain_dir, exist_ok=True)
        print(f"  Downloading TalkNet-ASD checkpoint to {ckpt} ...", flush=True)
        link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
        subprocess.call(f"gdown --id {link} -O {ckpt}", shell=True)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"TalkNet checkpoint download failed. Download manually:\n"
                f"  gdown --id 1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea -O {ckpt}"
            )

    model = talkNet()
    model.loadParameters(ckpt)
    model.eval()
    return model


def _build_grayscale_crops(
    video_path: str,
    track: Dict,
    size: int = 112,
) -> Tuple[np.ndarray, float, List[float]]:
    """Extract grayscale 112×112 crops for all tracked frames.

    Returns (videoFeature, fps, frame_timestamps) where videoFeature is (N, 112, 112).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_to_box = {f["frame_idx"]: f["bbox"] for f in track["frames"]}
    min_frame = track["frames"][0]["frame_idx"]
    max_frame = track["frames"][-1]["frame_idx"]

    crops = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
    for fidx in range(min_frame, max_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if fidx in frame_to_box:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop = _crop_face(rgb, frame_to_box[fidx], size)
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)  # (112, 112) uint8
            crops[fidx] = gray
    cap.release()

    sorted_frames = sorted(crops.keys())
    videoFeature = np.stack([crops[f] for f in sorted_frames]).astype(np.float32)  # (N, 112, 112)
    return videoFeature, fps, [f / fps for f in sorted_frames]


@torch.no_grad()
def run_talknet_asd(
    audio_path: str,
    video_path: str,
    tracks: List[Dict],
    pretrain_dir: str,
    min_seg_dur: float = 0.4,
) -> List[Dict]:
    """TalkNet-ASD inference on child-candidate face track (smallest bbox).

    Follows demoTalkNet.py evaluate_network() exactly:
    - MFCC 13-dim at 100fps
    - Grayscale 112×112 crops at native fps (≈25fps)
    - Multi-duration windows {1,1,1,2,2,2,3,3,4,5,6} s, averaged
    - Raw logit ≥ 0 → speaking
    """
    if not tracks:
        return []

    child_track = min(tracks, key=lambda t: t["mean_area"])

    model = _load_talknet(pretrain_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    audio_np, sr = load_audio_16k(audio_path)
    audioFeature = compute_mfcc(audio_np, sr)  # (T_a, 13)

    videoFeature, fps, frame_timestamps = _build_grayscale_crops(video_path, child_track)
    if videoFeature.shape[0] == 0:
        return []

    # TalkNet was trained at 25fps; resample to 25fps if video has different rate
    # so that audio (4× downsampled from 100fps → 25 embed frames/s) aligns with
    # video embed frames (1:1 with input frames → also 25 embed frames/s).
    TARGET_FPS = 25.0
    if abs(fps - TARGET_FPS) > 0.5:
        n_src = videoFeature.shape[0]
        n_dst = max(1, int(round(n_src * TARGET_FPS / fps)))
        idx = np.round(np.linspace(0, n_src - 1, n_dst)).astype(int)
        videoFeature = videoFeature[idx]
        frame_timestamps = [frame_timestamps[i] for i in idx]
        fps = TARGET_FPS

    n_audio = audioFeature.shape[0]
    n_video = videoFeature.shape[0]
    track_start_sec = frame_timestamps[0] if frame_timestamps else 0.0

    # Align audio to track window
    a_start = int(track_start_sec * 100)
    audioFeature = audioFeature[a_start:]

    # Truncate to synchronized length
    length = min(n_audio / 100, n_video / fps)
    audioFeature = audioFeature[:int(round(length * 100))]
    videoFeature = videoFeature[:int(round(length * fps))]

    if len(audioFeature) == 0 or len(videoFeature) == 0:
        return []

    durationSet = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    allScores = []
    n_video_final = videoFeature.shape[0]

    for duration in durationSet:
        fps_int = int(round(fps))
        batchSize = max(1, int(math.ceil(length / duration)))
        scores = []
        for i in range(batchSize):
            inputA_np = audioFeature[i * duration * 100: (i + 1) * duration * 100]
            inputV_np = videoFeature[i * duration * fps_int: (i + 1) * duration * fps_int]
            if inputA_np.shape[0] == 0 or inputV_np.shape[0] == 0:
                continue
            inputA = torch.FloatTensor(inputA_np).unsqueeze(0).to(device)   # (1, T_a, 13)
            inputV = torch.FloatTensor(inputV_np).unsqueeze(0).to(device)   # (1, T_v, 112, 112)
            embedA = model.model.forward_audio_frontend(inputA)
            embedV = model.model.forward_visual_frontend(inputV)
            min_t = min(embedA.shape[1], embedV.shape[1])
            embedA, embedV = embedA[:, :min_t, :], embedV[:, :min_t, :]
            embedA, embedV = model.model.forward_cross_attention(embedA, embedV)
            out = model.model.forward_audio_visual_backend(embedA, embedV)
            score = model.lossAV.forward(out, labels=None)  # numpy (T_v,) raw logits
            scores.extend(score.tolist() if hasattr(score, "tolist") else list(score))
        if scores:
            allScores.append(scores)

    if not allScores:
        return []

    # Align all score arrays to minimum length and average
    min_len = min(len(s) for s in allScores)
    allScores = np.array([s[:min_len] for s in allScores])
    avgScores = np.mean(allScores, axis=0)  # (N_frames,)

    # Map frame scores → time segments (threshold: raw logit ≥ 0)
    per_frame_scores = {}
    for i in range(min(len(avgScores), len(frame_timestamps))):
        per_frame_scores[frame_timestamps[i]] = float(avgScores[i])

    return _scores_to_segments_by_time(per_frame_scores, threshold=0.0, min_dur=min_seg_dur)


# ---------------------------------------------------------------------------
# TS-TalkNet inference
# ---------------------------------------------------------------------------

def _load_ts_talknet(pretrain_dir: str):
    """Load TS-TalkNet model via importlib (hyphenated filename).

    ts-talkNet.py cannot be imported with standard 'import'; uses importlib.
    CWD must be _TSTALKNET_DIR because ablation2_talkNetModel loads
    'exps/pretrain.model' (ECAPA speaker encoder) relative to CWD at init.
    """
    if not _TSTALKNET_DIR.is_dir():
        raise FileNotFoundError(
            f"TS-TalkNet repo not found at {_TSTALKNET_DIR}. "
            f"Clone: git clone https://github.com/Jiang-Yidi/TS-TalkNet {_TSTALKNET_DIR}"
        )

    ckpt = os.path.join(pretrain_dir, "ts_talknet.model")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"TS-TalkNet checkpoint not found: {ckpt}\n"
            f"Download from the TS-TalkNet repo and save as ts_talknet.model.\n"
            f"See video/SETUP.md for instructions."
        )

    ecapa_ckpt = str(_TSTALKNET_DIR / "exps" / "pretrain.model")
    if not os.path.exists(ecapa_ckpt):
        raise FileNotFoundError(
            f"TS-TalkNet ECAPA speaker encoder not found: {ecapa_ckpt}\n"
            f"This is required for model initialization. "
            f"Obtain from the TS-TalkNet authors or train from scratch.\n"
            f"See TS-TalkNet/README.md."
        )

    # Must chdir to _TSTALKNET_DIR: ablation2_talkNetModel loads 'exps/pretrain.model'
    os.chdir(str(_TSTALKNET_DIR))
    sys.path.insert(0, str(_TSTALKNET_DIR))

    spec = importlib.util.spec_from_file_location(
        "ts_talknet_module",
        str(_TSTALKNET_DIR / "ts-talkNet.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    talkNetCls = mod.talkNet

    model = talkNetCls()
    model.loadParameters(ckpt)
    model.eval()
    return model


def _build_speaker_embedding(
    ref_audio_path: str,
    model,
    device: str,
    max_sec: float = 5.0,
) -> torch.Tensor:
    """Compute speaker embedding from reference audio.

    Uses TS-TalkNet's internal speaker encoder (ECAPA-TDNN) via
    model.model.forward_speaker_encoder(mfcc_tensor).
    """
    audio_np, sr = load_audio_16k(ref_audio_path)
    max_samples = int(max_sec * sr)
    audio_np = audio_np[:max_samples]
    mfcc = compute_mfcc(audio_np, sr)  # (T, 13)
    mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(device)  # (1, T, 13)
    with torch.no_grad():
        emb = model.model.forward_speaker_encoder(mfcc_tensor)
    return emb


@torch.no_grad()
def run_ts_talknet(
    audio_path: str,
    video_path: str,
    ref_audio_path: str,
    tracks: List[Dict],
    pretrain_dir: str,
    min_seg_dur: float = 0.4,
) -> List[Dict]:
    """TS-TalkNet speaker-conditioned ASD.

    Builds speaker embedding from ref_audio_path, runs ASD on ALL face tracks,
    picks the track whose mean conditioned score is highest (target speaker).
    """
    if not tracks:
        return []

    model = _load_ts_talknet(pretrain_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    speaker_emb = _build_speaker_embedding(ref_audio_path, model, device)

    audio_np, sr = load_audio_16k(audio_path)
    audioFeature = compute_mfcc(audio_np, sr)  # (T_a, 13)

    best_track_result = None
    best_mean_score = -1e9

    for track in tracks:
        videoFeature, fps, frame_timestamps = _build_grayscale_crops(video_path, track)
        if videoFeature.shape[0] == 0:
            continue

        # Resample to 25fps (model was trained at 25fps)
        TARGET_FPS = 25.0
        if abs(fps - TARGET_FPS) > 0.5:
            n_src = videoFeature.shape[0]
            n_dst = max(1, int(round(n_src * TARGET_FPS / fps)))
            idx = np.round(np.linspace(0, n_src - 1, n_dst)).astype(int)
            videoFeature = videoFeature[idx]
            frame_timestamps = [frame_timestamps[i] for i in idx]
            fps = TARGET_FPS

        track_start_sec = frame_timestamps[0] if frame_timestamps else 0.0
        a_start = int(track_start_sec * 100)
        af_slice = audioFeature[a_start:]

        n_audio = af_slice.shape[0]
        n_video = videoFeature.shape[0]
        length = min(n_audio / 100, n_video / fps)
        af_slice = af_slice[:int(round(length * 100))]
        vf_slice = videoFeature[:int(round(length * fps))]

        if len(af_slice) == 0 or len(vf_slice) == 0:
            continue

        durationSet = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
        allScores = []
        fps_int = int(round(fps))

        for duration in durationSet:
            batchSize = max(1, int(math.ceil(length / duration)))
            scores = []
            for i in range(batchSize):
                inputA_np = af_slice[i * duration * 100: (i + 1) * duration * 100]
                inputV_np = vf_slice[i * duration * fps_int: (i + 1) * duration * fps_int]
                if inputA_np.shape[0] == 0 or inputV_np.shape[0] == 0:
                    continue
                inputA = torch.FloatTensor(inputA_np).unsqueeze(0).to(device)
                inputV = torch.FloatTensor(inputV_np).unsqueeze(0).to(device)
                embedA = model.model.forward_audio_frontend(inputA)
                embedV = model.model.forward_visual_frontend(inputV)
                min_t = min(embedA.shape[1], embedV.shape[1])
                embedA, embedV = embedA[:, :min_t, :], embedV[:, :min_t, :]
                embedA, embedV = model.model.forward_cross_attention(embedA, embedV)
                outsAV = model.model.forward_audio_visual_backend(embedA, embedV, speaker_emb)
                score = model.lossAV.forward(outsAV, labels=None)
                scores.extend(score.tolist() if hasattr(score, "tolist") else list(score))
            if scores:
                allScores.append(scores)

        if not allScores:
            continue

        min_len = min(len(s) for s in allScores)
        avg_scores_np = np.mean([s[:min_len] for s in allScores], axis=0)
        mean_score = float(np.mean(avg_scores_np))

        if mean_score > best_mean_score:
            best_mean_score = mean_score
            best_track_result = (avg_scores_np, frame_timestamps[:min_len])

    if best_track_result is None:
        return []

    avg_scores, timestamps = best_track_result
    per_frame_scores = {ts: float(sc) for ts, sc in zip(timestamps, avg_scores)}
    return _scores_to_segments_by_time(per_frame_scores, threshold=0.0, min_dur=min_seg_dur)


# ---------------------------------------------------------------------------
# Score → segments
# ---------------------------------------------------------------------------

def _scores_to_segments(
    frame_scores: np.ndarray,
    fps: float,
    min_dur: float = 0.4,
    threshold: float = 0.5,
    merge_gap: float = 0.2,
) -> List[Dict]:
    """Convert per-frame score array to time segments with start/end keys."""
    segs = []
    in_seg = False
    seg_start = 0.0
    for fi, sc in enumerate(frame_scores):
        t = fi / fps
        if sc >= threshold and not in_seg:
            seg_start = t
            in_seg = True
        elif sc < threshold and in_seg:
            segs.append({"start": seg_start, "end": t})
            in_seg = False
    if in_seg:
        segs.append({"start": seg_start, "end": len(frame_scores) / fps})

    merged = []
    for seg in segs:
        if merged and (seg["start"] - merged[-1]["end"]) <= merge_gap:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return [s for s in merged if (s["end"] - s["start"]) >= min_dur]


def _scores_to_segments_by_time(
    per_frame_scores: Dict[float, float],
    threshold: float = 0.0,
    min_dur: float = 0.4,
    merge_gap: float = 0.2,
) -> List[Dict]:
    """Convert timestamped frame scores to time segments.

    Applies threshold, merges gaps ≤ merge_gap seconds, drops segments < min_dur.
    """
    if not per_frame_scores:
        return []

    timestamps_sorted = sorted(per_frame_scores.keys())
    speaking = {t: per_frame_scores[t] >= threshold for t in timestamps_sorted}

    segs = []
    in_seg = False
    seg_start = 0.0

    for ts in timestamps_sorted:
        if speaking[ts] and not in_seg:
            seg_start = ts
            in_seg = True
        elif not speaking[ts] and in_seg:
            segs.append({"start": seg_start, "end": ts})
            in_seg = False

    if in_seg:
        segs.append({"start": seg_start, "end": timestamps_sorted[-1]})

    merged = []
    for seg in segs:
        if merged and (seg["start"] - merged[-1]["end"]) <= merge_gap:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))

    return [s for s in merged if (s["end"] - s["start"]) >= min_dur]


# ---------------------------------------------------------------------------
# RTTM writer
# ---------------------------------------------------------------------------

def write_rttm(segments: List[Dict], audio_path: str, out_rttm: str):
    """Write child vocalization segments to RTTM format."""
    file_id = Path(audio_path).stem
    os.makedirs(os.path.dirname(os.path.abspath(out_rttm)), exist_ok=True)
    with open(out_rttm, "w") as f:
        for seg in segments:
            start = seg["start"]
            dur = seg["end"] - seg["start"]
            f.write(f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> CHI <NA> <NA>\n")


# ---------------------------------------------------------------------------
# LocoNet ASD
# ---------------------------------------------------------------------------

def run_loconet_asd(
    audio_path: str,
    video_path: str,
    tracks: List,
    checkpoint: str,
    min_seg_dur: float = 0.4,
) -> List[Dict]:
    """Run LocoNet-ASD inference and return CHI-labeled segments.

    LocoNet (CVPR 2023) uses VGGish audio + ResNet visual frontends with
    cross-attention. Requires video/LoCoNet_ASD/ from HuggingFace.

    Audio: 13-coeff MFCC, 4 frames per video frame.
    Visual: grayscale 112×112 face crops at video FPS.
    Checkpoint: video/LoCoNet_ASD/pytorch_model.bin (HuggingFace format).
    """
    loconet_dir = _THIS_DIR / "LoCoNet_ASD"
    if not loconet_dir.is_dir():
        raise FileNotFoundError(
            f"LoCoNet_ASD repo not found at {loconet_dir}.\n"
            "Download with: huggingface-cli download Superxixixi/LoCoNet_ASD "
            "--local-dir video/LoCoNet_ASD/"
        )
    # Accept either a .ckpt or .bin checkpoint
    if not checkpoint or not os.path.exists(checkpoint):
        # Fall back: look for pytorch_model.bin in the repo dir
        default_bin = str(loconet_dir / "pytorch_model.bin")
        if os.path.exists(default_bin):
            checkpoint = default_bin
        else:
            raise FileNotFoundError(
                f"LocoNet checkpoint not found: {checkpoint}\n"
                f"Expected at: {default_bin}"
            )

    if str(loconet_dir) not in sys.path:
        sys.path.insert(0, str(loconet_dir))

    try:
        from torchvggish import vggish_input as _vggish_input  # type: ignore
        from config_loconet import LoCoNetConfig  # type: ignore
        from modeling_loconet import loconet as LoconetModel  # type: ignore
        from loss_multi import lossAV as LossAV  # type: ignore

        # ----------------------------------------------------------------
        # Load model
        # ----------------------------------------------------------------
        cfg = LoCoNetConfig.from_pretrained(str(loconet_dir))
        model = LoconetModel(cfg)
        state = torch.load(checkpoint, map_location="cpu")
        # Plain state dict has keys like "model.visualFrontend.*"
        if not any(k.startswith("model.") for k in state):
            state = {"model." + k: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.eval()

        # The lossAV head (Linear 256→2) is part of the checkpoint
        loss_av = model.lossAV
        loss_av.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # ----------------------------------------------------------------
        # Pick child candidate (smallest face track)
        # ----------------------------------------------------------------
        def _track_area(t):
            areas = [abs((b[2]-b[0])*(b[3]-b[1])) for b in _track_to_bboxes(t) if b]
            return float(np.mean(areas)) if areas else float("inf")

        sorted_tracks = sorted(tracks, key=_track_area)
        if not sorted_tracks:
            return []
        child_track = sorted_tracks[0]
        bboxes = _track_to_bboxes(child_track)
        if not bboxes:
            return []

        # ----------------------------------------------------------------
        # Video params
        # ----------------------------------------------------------------
        cap = cv2.VideoCapture(video_path)
        fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration_sec = n_vid_frames / fps_vid

        # ----------------------------------------------------------------
        # Audio features: VGGish log-mel spectrogram (matches LocoNet training).
        # waveform_to_examples produces (n_vid_frames*4, 64) — 4 audio frames per
        # video frame, 64 mel bins. VGGish Conv2d expects 4-D (B,1,T,64) input.
        # ----------------------------------------------------------------
        audio_raw, sr = load_audio_16k(audio_path)
        mel_all = _vggish_input.waveform_to_examples(
            audio_raw, sr, n_vid_frames, fps_vid, return_tensor=False
        )  # (n_vid_frames*4, 64)

        # ----------------------------------------------------------------
        # Visual features: grayscale 112×112 crops at video FPS
        # ----------------------------------------------------------------
        cap = cv2.VideoCapture(video_path)
        face_frames: List[np.ndarray] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx < len(bboxes) and bboxes[frame_idx]:
                x1, y1, x2, y2 = [int(v) for v in bboxes[frame_idx]]
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    face_frames.append(cv2.resize(gray, (112, 112)))
                else:
                    face_frames.append(np.zeros((112, 112), dtype=np.uint8))
            else:
                face_frames.append(np.zeros((112, 112), dtype=np.uint8))
            frame_idx += 1
        cap.release()

        if not face_frames:
            return []

        # ----------------------------------------------------------------
        # Sliding-window inference (4s windows, 0.4s hop)
        # ----------------------------------------------------------------
        window_sec = min(4.0, duration_sec)
        hop_sec = 0.4
        n_windows = max(1, int((duration_sec - window_sec) / hop_sec) + 1)
        all_frame_scores: dict = {}  # video_frame_idx → list of scores

        for wi in range(n_windows):
            t_start = wi * hop_sec
            t_end = t_start + window_sec

            v_s = int(t_start * fps_vid)
            v_e = min(int(t_end * fps_vid), len(face_frames))
            if v_e <= v_s:
                continue

            a_s = v_s * 4
            a_e = v_e * 4
            a_e = min(a_e, len(mel_all))
            if a_e <= a_s:
                continue

            # (1, 1, T_audio, 64) — batch=1, channel=1, frames, mel_bins
            audio_feat = torch.FloatTensor(mel_all[a_s:a_e]).unsqueeze(0).unsqueeze(0).to(device)

            vis_chunk = np.stack(face_frames[v_s:v_e], axis=0)  # (T_vid, 112, 112)
            visual_feat = torch.FloatTensor(vis_chunk).unsqueeze(0).to(device)
            # (1, T_vid, 112, 112)

            try:
                with torch.no_grad():
                    enc = model.model  # locoencoder
                    a_emb = enc.forward_audio_frontend(audio_feat)   # (1, T, 128)
                    v_emb = enc.forward_visual_frontend(visual_feat)  # (1, T, 128)
                    # Align lengths
                    T = min(a_emb.shape[1], v_emb.shape[1])
                    a_emb = a_emb[:, :T, :]
                    v_emb = v_emb[:, :T, :]
                    # ConvLayer kernel height = cfg.num_speakers (=3); must repeat
                    # the single track cfg.num_speakers times before cross-attention,
                    # matching the training pattern where audio is shared across all s
                    # speaker tracks. Take first T scores as the target speaker's.
                    n_spk = cfg.num_speakers
                    a_rep = a_emb.repeat(n_spk, 1, 1)  # (n_spk, T, 128)
                    v_rep = v_emb.repeat(n_spk, 1, 1)  # (n_spk, T, 128)
                    a_rep, v_rep = enc.forward_cross_attention(a_rep, v_rep)
                    outsAV = enc.forward_audio_visual_backend(a_rep, v_rep, b=1, s=n_spk)
                    scores_np = model.lossAV(outsAV)  # (n_spk*T,) numpy
                    if hasattr(scores_np, "cpu"):
                        scores_np = scores_np.cpu().numpy()
                    scores_np = scores_np[:T]  # first T frames = target speaker slot

                for fi, score in enumerate(scores_np):
                    vid_fi = v_s + fi
                    all_frame_scores.setdefault(vid_fi, []).append(float(score))
            except Exception as _e:
                print(f"  LocoNet window {wi} failed: {_e}", file=sys.stderr)
                continue

        if not all_frame_scores:
            return []

        # Average scores per frame; threshold=0.0 (logit decision boundary, not probability)
        n_frames_total = len(face_frames)
        frame_scores = np.zeros(n_frames_total, dtype=np.float32)
        for fi, sc_list in all_frame_scores.items():
            if fi < n_frames_total:
                frame_scores[fi] = float(np.mean(sc_list))

        segments = _scores_to_segments(frame_scores, fps_vid, min_seg_dur, threshold=0.0)
        return segments

    except Exception as e:
        import traceback
        print(f"  WARNING: LocoNet inference failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Light-ASD
# ---------------------------------------------------------------------------

def run_light_asd(
    audio_path: str,
    video_path: str,
    tracks: List,
    checkpoint: str,
    min_seg_dur: float = 0.4,
) -> List[Dict]:
    """Run Light-ASD inference and return CHI-labeled segments.

    Light-ASD is a lightweight model with a simpler forward pass than TalkNet.
    Requires video/Light-ASD/ to be cloned and checkpoint downloaded.
    Falls back gracefully if import fails.
    """
    light_asd_dir = _THIS_DIR / "Light-ASD"
    if not light_asd_dir.is_dir():
        raise FileNotFoundError(
            f"Light-ASD repo not found at {light_asd_dir}.\n"
            "Clone with: git clone https://github.com/Junhua-Liao/Light-ASD video/Light-ASD"
        )
    if str(light_asd_dir) not in sys.path:
        sys.path.insert(0, str(light_asd_dir))

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Light-ASD checkpoint not found: {checkpoint}\n"
            "Download from: https://github.com/Junhua-Liao/Light-ASD"
        )

    try:
        try:
            from model import LightASD  # type: ignore
        except ImportError:
            from Light_ASD import LightASD  # type: ignore

        audio, sr = load_audio_16k(audio_path)
        mfcc = psf.mfcc(audio, samplerate=sr, numcep=13, nfft=512, appendEnergy=False)
        mfcc = _normalize_mfcc(mfcc)

        ckpt_data = torch.load(checkpoint, map_location="cpu")
        model = LightASD()
        state = ckpt_data.get("model_state_dict", ckpt_data.get("state_dict", ckpt_data))
        model.load_state_dict(state, strict=False)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        cap = cv2.VideoCapture(video_path)
        fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        segments = _score_tracks_with_model(
            model, mfcc, tracks, video_path, fps_vid, audio_path, sr, min_seg_dur, device
        )
        return segments

    except Exception as e:
        print(f"  WARNING: Light-ASD inference failed ({e}); falling back to empty segments", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Per-track LocoNet (used by LocoNetECAPAFrontend in video_asd.py)
# ---------------------------------------------------------------------------

def run_loconet_asd_per_track(
    audio_path: str,
    video_path: str,
    tracks: List,
    checkpoint: str,
    min_seg_dur: float = 0.4,
) -> List[Dict]:
    """Run LocoNet-ASD independently on every face track.

    Returns a list of dicts, one per track:
      {"track_id": int, "mean_area": float,
       "segments": [{"start": float, "end": float}]}

    Unlike run_loconet_asd() which uses smallest-face heuristic, this
    function scores every track so callers can pick the child track via
    speaker identity (e.g. ECAPA cosine similarity).
    """
    loconet_dir = _THIS_DIR / "LoCoNet_ASD"
    if not loconet_dir.is_dir():
        raise FileNotFoundError(f"LoCoNet_ASD repo not found at {loconet_dir}.")

    if not checkpoint or not os.path.exists(checkpoint):
        default_bin = str(loconet_dir / "pytorch_model.bin")
        if os.path.exists(default_bin):
            checkpoint = default_bin
        else:
            raise FileNotFoundError(f"LocoNet checkpoint not found: {checkpoint}")

    if str(loconet_dir) not in sys.path:
        sys.path.insert(0, str(loconet_dir))

    try:
        from torchvggish import vggish_input as _vggish_input  # type: ignore
        from config_loconet import LoCoNetConfig  # type: ignore
        from modeling_loconet import loconet as LoconetModel  # type: ignore

        cfg_loco = LoCoNetConfig.from_pretrained(str(loconet_dir))
        model = LoconetModel(cfg_loco)
        state = torch.load(checkpoint, map_location="cpu")
        if not any(k.startswith("model.") for k in state):
            state = {"model." + k: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Video params
        cap = cv2.VideoCapture(video_path)
        fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration_sec = n_vid_frames / fps_vid if fps_vid > 0 else 0.0
        if duration_sec <= 0:
            return []

        # Audio features: VGGish log-mel (shared across all tracks).
        # waveform_to_examples → (n_vid_frames*4, 64), 4 audio frames per video frame.
        audio_raw, sr = load_audio_16k(audio_path)
        mel_all = _vggish_input.waveform_to_examples(
            audio_raw, sr, n_vid_frames, fps_vid, return_tensor=False
        )  # (n_vid_frames*4, 64)

        window_sec = min(4.0, duration_sec)
        hop_sec = 0.4
        n_windows = max(1, int((duration_sec - window_sec) / hop_sec) + 1)

        def _track_area(t):
            areas = [abs((b[2]-b[0])*(b[3]-b[1])) for b in _track_to_bboxes(t) if b]
            return float(np.mean(areas)) if areas else float("inf")

        results = []
        for ti, track in enumerate(tracks):
            bboxes = _track_to_bboxes(track)
            if not bboxes:
                continue

            # Extract face crops for this track
            cap = cv2.VideoCapture(video_path)
            face_frames: List[np.ndarray] = []
            fi = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if fi < len(bboxes) and bboxes[fi]:
                    x1, y1, x2, y2 = [int(v) for v in bboxes[fi]]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        face_frames.append(cv2.resize(gray, (112, 112)))
                    else:
                        face_frames.append(np.zeros((112, 112), dtype=np.uint8))
                else:
                    face_frames.append(np.zeros((112, 112), dtype=np.uint8))
                fi += 1
            cap.release()

            if not face_frames:
                continue

            # Sliding-window LocoNet inference for this track
            all_frame_scores: dict = {}
            for wi in range(n_windows):
                t_start = wi * hop_sec
                t_end = t_start + window_sec
                v_s = int(t_start * fps_vid)
                v_e = min(int(t_end * fps_vid), len(face_frames))
                if v_e <= v_s:
                    continue
                a_s, a_e = v_s * 4, min(v_e * 4, len(mel_all))
                if a_e <= a_s:
                    continue

                # (1, 1, T_audio, 64) — batch=1, channel=1 for VGGish Conv2d
                audio_feat = torch.FloatTensor(mel_all[a_s:a_e]).unsqueeze(0).unsqueeze(0).to(device)
                vis_chunk = np.stack(face_frames[v_s:v_e], axis=0)
                visual_feat = torch.FloatTensor(vis_chunk).unsqueeze(0).to(device)

                try:
                    with torch.no_grad():
                        enc = model.model
                        a_emb = enc.forward_audio_frontend(audio_feat)
                        v_emb = enc.forward_visual_frontend(visual_feat)
                        T = min(a_emb.shape[1], v_emb.shape[1])
                        a_emb, v_emb = a_emb[:, :T, :], v_emb[:, :T, :]
                        # ConvLayer kernel height = cfg_loco.num_speakers (=3)
                        n_spk = cfg_loco.num_speakers
                        a_rep = a_emb.repeat(n_spk, 1, 1)
                        v_rep = v_emb.repeat(n_spk, 1, 1)
                        a_rep, v_rep = enc.forward_cross_attention(a_rep, v_rep)
                        outsAV = enc.forward_audio_visual_backend(a_rep, v_rep, b=1, s=n_spk)
                        scores_np = model.lossAV(outsAV)  # (n_spk*T,)
                        if hasattr(scores_np, "cpu"):
                            scores_np = scores_np.cpu().numpy()
                        scores_np = scores_np[:T]  # first T = target speaker slot
                    for frame_i, score in enumerate(scores_np):
                        all_frame_scores.setdefault(v_s + frame_i, []).append(float(score))
                except Exception as _e:
                    print(f"  LocoNet per-track track={ti} win={wi} failed: {_e}",
                          file=sys.stderr)

            if not all_frame_scores:
                continue

            n_frames_total = len(face_frames)
            frame_scores = np.zeros(n_frames_total, dtype=np.float32)
            for frame_i, sc_list in all_frame_scores.items():
                if frame_i < n_frames_total:
                    frame_scores[frame_i] = float(np.mean(sc_list))

            segs = _scores_to_segments(frame_scores, fps_vid, min_seg_dur, threshold=0.0)
            results.append({
                "track_id": ti,
                "mean_area": _track_area(track),
                "segments": [{"start": s["start"], "end": s["end"]} for s in segs],
            })

        return results

    except Exception as e:
        import traceback
        print(f"  WARNING: LocoNet per-track failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Shared track-scoring helper (used by LocoNet and Light-ASD)
# ---------------------------------------------------------------------------

def _normalize_mfcc(mfcc: np.ndarray) -> np.ndarray:
    mu = np.mean(mfcc, axis=0, keepdims=True)
    std = np.std(mfcc, axis=0, keepdims=True) + 1e-8
    return (mfcc - mu) / std


def _score_tracks_with_model(
    model,
    mfcc: np.ndarray,
    tracks: List,
    video_path: str,
    fps_vid: float,
    audio_path: str,
    sr: int,
    min_seg_dur: float,
    device: str,
) -> List[Dict]:
    """Score face tracks with a TalkNet-compatible model API.

    Assumes model.forward_audio_visual(audio_feat, visual_feat) → logits.
    Falls back to model(audio_feat, visual_feat) if the above doesn't exist.
    This covers LocoNet and Light-ASD which share TalkNet's input signature.
    """
    mfcc_frames_per_sec = 100

    # Find the track with the smallest mean face area (child candidate)
    def _track_area(track):
        areas = [abs((b[2] - b[0]) * (b[3] - b[1])) for b in _track_to_bboxes(track) if b]
        return float(np.mean(areas)) if areas else float("inf")

    sorted_tracks = sorted(tracks, key=_track_area)
    if not sorted_tracks:
        return []

    child_track = sorted_tracks[0]
    bboxes = _track_to_bboxes(child_track)
    if not bboxes:
        return []

    cap = cv2.VideoCapture(video_path)
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_sec = n_video_frames / fps_vid if fps_vid > 0 else 0.0
    if duration_sec <= 0:
        return []

    n_mfcc = mfcc.shape[0]
    scores: List[float] = []

    # Sliding window scoring (1-6 s, matching TalkNet multi-duration approach)
    window_sec = min(4.0, duration_sec)
    hop_sec = 0.4
    n_windows = max(1, int((duration_sec - window_sec) / hop_sec) + 1)

    for wi in range(n_windows):
        t_start = wi * hop_sec
        t_end = t_start + window_sec

        a_start = int(t_start * mfcc_frames_per_sec)
        a_end = int(t_end * mfcc_frames_per_sec)
        a_end = min(a_end, n_mfcc)
        if a_end <= a_start:
            continue
        audio_feat = torch.FloatTensor(mfcc[a_start:a_end]).unsqueeze(0).to(device)

        v_start = int(t_start * fps_vid)
        v_end = int(t_end * fps_vid)
        v_end = min(v_end, len(bboxes))
        if v_end <= v_start:
            continue

        try:
            if hasattr(model, "forward_audio_visual"):
                score = model.forward_audio_visual(audio_feat, audio_feat).squeeze().mean().item()
            else:
                score = model(audio_feat, audio_feat).squeeze().mean().item()
        except Exception:
            score = 0.0

        scores.append(float(torch.sigmoid(torch.tensor(score)).item()) if score != 0.0 else 0.0)

    if not scores:
        return []

    mean_score = float(np.mean(scores))
    if mean_score < 0.5:
        return []

    # Return single segment covering active portion of the clip
    audio_dur = n_mfcc / mfcc_frames_per_sec
    if audio_dur < min_seg_dur:
        return []

    return [{
        "start": 0.0,
        "end": audio_dur,
        "label": "CHI",
        "score": mean_score,
    }]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run video ASD on a SAILS BIDS recording."
    )
    parser.add_argument("--audio_path", required=True,
                        help="Path to *_audio.wav (BIDS preprocessed)")
    parser.add_argument("--model", required=True,
                        choices=["talknet_asd", "ts_talknet", "loconet", "light_asd"])
    parser.add_argument("--ref_audio", default="",
                        help="Reference audio for speaker enrollment (ts_talknet only)")
    parser.add_argument("--checkpoint", default="",
                        help="Checkpoint path for loconet or light_asd models")
    parser.add_argument("--out_rttm", required=True,
                        help="Output RTTM file path")
    parser.add_argument("--face_cache_dir", required=True,
                        help="Directory for S3FD face track JSON cache")
    parser.add_argument("--pretrain_dir", required=True,
                        help="Directory containing model checkpoint files")
    parser.add_argument("--min_seg_dur", type=float, default=0.4,
                        help="Minimum segment duration in seconds")
    parser.add_argument("--output_tracks_json", default="",
                        help="(loconet only) Write per-track active segments to this JSON path")
    args = parser.parse_args()

    # Derive video path (raises FileNotFoundError for audio-only datasets)
    video_path = derive_video_path(args.audio_path)
    print(f"Video: {video_path}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Run face detection (cached)
    tracks = detect_faces_in_video(
        video_path,
        args.face_cache_dir,
        device=device,
    )

    if not tracks:
        print("No face tracks found — writing empty RTTM", flush=True)
        write_rttm([], args.audio_path, args.out_rttm)
        return

    # Run selected ASD model
    if args.model == "talknet_asd":
        segments = run_talknet_asd(
            args.audio_path, video_path, tracks,
            args.pretrain_dir, args.min_seg_dur,
        )
    elif args.model == "ts_talknet":
        if not args.ref_audio:
            parser.error("--ref_audio is required for --model ts_talknet")
        if not os.path.exists(args.ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {args.ref_audio}")
        segments = run_ts_talknet(
            args.audio_path, video_path, args.ref_audio, tracks,
            args.pretrain_dir, args.min_seg_dur,
        )
    elif args.model == "loconet":
        if not args.checkpoint:
            parser.error("--checkpoint is required for --model loconet")
        if args.output_tracks_json:
            track_data = run_loconet_asd_per_track(
                args.audio_path, video_path, tracks,
                args.checkpoint, args.min_seg_dur,
            )
            os.makedirs(os.path.dirname(os.path.abspath(args.output_tracks_json)), exist_ok=True)
            with open(args.output_tracks_json, "w") as _tf:
                json.dump(track_data, _tf)
            print(f"Per-track JSON: {args.output_tracks_json} ({len(track_data)} tracks)",
                  flush=True)
            # Also write a smallest-face RTTM for the standard output
            if track_data:
                best = min(track_data, key=lambda t: t.get("mean_area", float("inf")))
                segments = [{"start": s["start"], "end": s["end"],
                             "score": 1.0} for s in best["segments"]]
            else:
                segments = []
        else:
            segments = run_loconet_asd(
                args.audio_path, video_path, tracks,
                args.checkpoint, args.min_seg_dur,
            )
    elif args.model == "light_asd":
        if not args.checkpoint:
            parser.error("--checkpoint is required for --model light_asd")
        segments = run_light_asd(
            args.audio_path, video_path, tracks,
            args.checkpoint, args.min_seg_dur,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Detected {len(segments)} child segments", flush=True)
    write_rttm(segments, args.audio_path, args.out_rttm)
    print(f"RTTM written: {args.out_rttm}", flush=True)


if __name__ == "__main__":
    main()
