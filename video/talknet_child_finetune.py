#!/usr/bin/env python3
"""
Fine-tune TalkNet-ASD for clip-level child vocalization detection.

Replaces TS-TalkNet (checkpoint blocked by author).  Uses the auto-downloadable
pretrained TalkNet-ASD checkpoint as backbone, adds a clip-level pooling head,
and fine-tunes on the SAILS BIDS seen-child split.

Architecture:
  TalkNet backbone (audio+visual encoders + cross-attention) [frozen in phase 1]
  + mean-pool over time
  + Linear(256→1) for AV clips  /  Linear(128→1) for audio-only clips
  + BCEWithLogitsLoss (pos_weight for class imbalance)

Two-phase training:
  Phase 1 — freeze backbone, train head only (lr_head=1e-4, 5 epochs)
  Phase 2 — unfreeze backbone, fine-tune everything (lr_backbone=1e-5, lr_head=1e-4, 15 epochs)

Data:
  Face crops precomputed from pyannote/video_face_cache/ → video/talknet_child_finetuned/crops/
  Audio MFCC computed on-the-fly (fast, ~50 ms per clip)
  Fallback to audio-only for clips with no detected faces

Output:
  video_finetuned_talknet_runs/best_checkpoint.pt
  video_finetuned_talknet_runs/val_metrics_tuned.json
  video_finetuned_talknet_runs/test_metrics_tuned.json
  video_finetuned_talknet_runs/test_predictions.csv
  video_finetuned_talknet_runs/config.json
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import python_speech_features as psf
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repo path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _THIS_DIR.parent
_TALKNET_DIR = _THIS_DIR / "TalkNet-ASD"

for _d in [_TALKNET_DIR]:
    if _d.is_dir() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio_16k(audio_path: str, max_sec: float = 12.0) -> np.ndarray:
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.squeeze(0).numpy()
    return wav[: int(max_sec * 16000)]


def compute_mfcc(audio_np: np.ndarray, sr: int = 16000) -> np.ndarray:
    if audio_np.dtype != np.int16:
        audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    else:
        audio_int16 = audio_np
    return psf.mfcc(audio_int16, sr, numcep=13, winlen=0.025, winstep=0.010)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def derive_video_path(audio_path: str) -> str:
    return audio_path.replace("_audio.wav", "_desc-processed_beh.mp4")


def cache_key(path: str) -> str:
    return hashlib.md5(path.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Face-crop precomputation and loading
# ---------------------------------------------------------------------------

def _precompute_crops_for_clip(
    video_path: str,
    track: dict,
    size: int = 112,
    max_sec: float = 12.0,
) -> np.ndarray:
    """Extract grayscale 112×112 crops for a face track.  Returns (N, H, W) uint8."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frame = int(max_sec * fps)

    frames_meta = track.get("frames", [])
    if not frames_meta:
        cap.release()
        return np.zeros((0, size, size), dtype=np.uint8)

    frame_to_box = {f["frame_idx"]: f["bbox"] for f in frames_meta}
    min_frame = frames_meta[0]["frame_idx"]
    max_track_frame = min(frames_meta[-1]["frame_idx"], max_frame)

    crops = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
    for fidx in range(min_frame, max_track_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if fidx in frame_to_box:
            bbox = frame_to_box[fidx]
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
            x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2GRAY)
                crops[fidx] = gray
    cap.release()

    if not crops:
        return np.zeros((0, size, size), dtype=np.uint8)
    sorted_keys = sorted(crops.keys())
    return np.stack([crops[k] for k in sorted_keys]).astype(np.uint8)


def precompute_all_crops(df: pd.DataFrame, face_cache_dir: str, crops_dir: str) -> None:
    """Precompute and save face crops for all clips that have face caches."""
    os.makedirs(crops_dir, exist_ok=True)
    n_saved, n_skip, n_no_face = 0, 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing crops"):
        audio_path = row["audio_path"]
        video_path = derive_video_path(audio_path)
        key = cache_key(video_path)
        crops_file = os.path.join(crops_dir, f"{key}.npz")

        if os.path.exists(crops_file):
            n_skip += 1
            continue

        face_cache = os.path.join(face_cache_dir, f"{key}.json")
        if not os.path.exists(face_cache):
            n_no_face += 1
            continue

        tracks = json.load(open(face_cache))
        if not tracks:
            # Save empty marker so we don't re-check
            np.savez_compressed(crops_file, crops=np.zeros((0, 112, 112), dtype=np.uint8))
            n_no_face += 1
            continue

        if not os.path.exists(video_path):
            n_no_face += 1
            continue

        child_track = min(tracks, key=lambda t: t.get("mean_area", float("inf")))
        crops = _precompute_crops_for_clip(video_path, child_track)
        np.savez_compressed(crops_file, crops=crops)
        n_saved += 1

    print(f"Crops precomputed: {n_saved} new, {n_skip} cached, {n_no_face} no-face")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChildVocDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, crops_dir: str, face_cache_dir: str, max_sec: float = 10.0):
        self.df = pd.read_csv(csv_path)
        self.crops_dir = crops_dir
        self.face_cache_dir = face_cache_dir
        self.max_sec = max_sec

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_path = str(row["audio_path"])
        label = float(row["label"])
        clip_id = str(row.get("clip_id", idx))

        # Audio MFCC
        audio_np = load_audio_16k(audio_path, max_sec=self.max_sec)
        mfcc = compute_mfcc(audio_np)  # (T_a, 13)
        mfcc_t = torch.FloatTensor(mfcc)  # (T_a, 13)

        # Face crops
        video_path = derive_video_path(audio_path)
        key = cache_key(video_path)
        crops_file = os.path.join(self.crops_dir, f"{key}.npz")
        video_t = None
        if os.path.exists(crops_file):
            data = np.load(crops_file)["crops"]  # (N, 112, 112) uint8
            if data.shape[0] >= 10:
                # Resample to at most max_sec * 25 fps frames
                max_frames = int(self.max_sec * 25)
                if data.shape[0] > max_frames:
                    idx_v = np.round(np.linspace(0, data.shape[0] - 1, max_frames)).astype(int)
                    data = data[idx_v]
                video_t = torch.FloatTensor(data.astype(np.float32))  # (N, 112, 112)

        return mfcc_t, video_t, label, clip_id


def collate_fn(batch):
    """Variable-length collate: keep batch_size=1, unsqueeze to match TalkNet input."""
    mfcc_t, video_t, label, clip_id = batch[0]
    return (
        mfcc_t.unsqueeze(0),   # (1, T_a, 13)
        video_t.unsqueeze(0) if video_t is not None else None,  # (1, N, 112, 112) or None
        torch.tensor(label, dtype=torch.float32),
        clip_id,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TalkNetChildDetector(nn.Module):
    """TalkNet backbone + clip-level pooling head for child vocalization."""

    def __init__(self, pretrain_path: str, freeze_backbone: bool = True):
        super().__init__()
        if not _TALKNET_DIR.is_dir():
            raise FileNotFoundError(
                f"TalkNet-ASD repo not found at {_TALKNET_DIR}. "
                f"Clone: git clone https://github.com/TaoRuijie/TalkNet-ASD {_TALKNET_DIR}"
            )
        os.chdir(str(_TALKNET_DIR))
        from talkNet import talkNet  # noqa: F401

        self.talknet = talkNet()
        self.talknet.loadParameters(pretrain_path)

        # Classification heads
        self.av_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 1))
        self.a_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.talknet.model.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.talknet.model.parameters():
            p.requires_grad = True

    def forward(self, mfcc: torch.Tensor, video: torch.Tensor | None) -> torch.Tensor:
        """
        mfcc:  (1, T_a, 13)
        video: (1, N, 112, 112) float32, or None
        Returns: scalar logit
        """
        embedA = self.talknet.model.forward_audio_frontend(mfcc)  # (1, T_ae, 128)

        if video is not None:
            embedV = self.talknet.model.forward_visual_frontend(video)  # (1, T_ve, 128)
            T = min(embedA.shape[1], embedV.shape[1])
            embedA_c, embedV_c = self.talknet.model.forward_cross_attention(
                embedA[:, :T, :], embedV[:, :T, :]
            )
            # Pool over time → (256,)
            pooled = torch.cat([embedA_c.mean(1), embedV_c.mean(1)], dim=-1).squeeze(0)
            return self.av_head(pooled).squeeze(-1)
        else:
            pooled = embedA.mean(1).squeeze(0)  # (128,)
            return self.a_head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_score, threshold: float) -> dict:
    y_pred = (np.array(y_score) >= threshold).astype(int)
    try:
        return {
            "auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": float(average_precision_score(y_true, y_score)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "threshold": threshold,
        }
    except Exception:
        return {"auroc": float("nan"), "auprc": float("nan"), "f1": 0.0,
                "precision": 0.0, "recall": 0.0, "threshold": threshold}


def tune_threshold_f1(y_true, y_score, n: int = 100) -> tuple[float, float]:
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, n):
        f = f1_score(y_true, (np.array(y_score) >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train: bool) -> tuple[float, list, list]:
    model.train(train)
    losses, labels_out, scores_out = [], [], []

    # Positive weight for class imbalance
    pos_weight = torch.tensor([2.0], device=device)

    for mfcc, video, label, _ in tqdm(loader, leave=False, desc="train" if train else "eval"):
        mfcc = mfcc.to(device)
        video = video.to(device) if video is not None else None
        label = label.to(device)

        with torch.set_grad_enabled(train):
            logit = model(mfcc, video)
            loss = F.binary_cross_entropy_with_logits(
                logit.unsqueeze(0), label.unsqueeze(0), pos_weight=pos_weight
            )

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(loss.item())
        labels_out.append(int(label.item()))
        scores_out.append(torch.sigmoid(logit).item())

    return float(np.mean(losses)), labels_out, scores_out


def evaluate_split(model, csv_path: str, crops_dir: str, face_cache_dir: str,
                   device: str, threshold: float | None = None) -> tuple[dict, pd.DataFrame]:
    dataset = ChildVocDataset(csv_path, crops_dir, face_cache_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                          num_workers=2, collate_fn=collate_fn)
    model.eval()
    labels_out, scores_out, clip_ids = [], [], []
    with torch.no_grad():
        for mfcc, video, label, clip_id in tqdm(loader, leave=False, desc="eval"):
            mfcc = mfcc.to(device)
            video = video.to(device) if video is not None else None
            logit = model(mfcc, video)
            labels_out.append(int(label.item()))
            scores_out.append(torch.sigmoid(logit).item())
            clip_ids.append(clip_id)

    if threshold is None:
        threshold, _ = tune_threshold_f1(labels_out, scores_out)

    metrics = compute_metrics(labels_out, scores_out, threshold)
    preds_df = pd.DataFrame({"clip_id": clip_ids, "label": labels_out, "prob": scores_out})
    return metrics, preds_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune TalkNet for child vocalization")
    parser.add_argument("--pretrain-path", default="video/pretrain/talknet_asd.model")
    parser.add_argument("--train-csv", default="whisper-modeling/seen_child_splits/train.csv")
    parser.add_argument("--val-csv", default="whisper-modeling/seen_child_splits/val.csv")
    parser.add_argument("--test-csv", default="whisper-modeling/seen_child_splits/test.csv")
    parser.add_argument("--face-cache-dir", default="pyannote/video_face_cache")
    parser.add_argument("--crops-dir", default="video/talknet_child_finetuned/crops")
    parser.add_argument("--output-dir", default="video_finetuned_talknet_runs")
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-precompute", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # All default paths are relative to repo root; chdir there first so resolves are correct.
    os.chdir(str(_REPO_ROOT))
    # Resolve ALL paths to absolute before TalkNetChildDetector.__init__ calls os.chdir(_TALKNET_DIR).
    args.output_dir = os.path.abspath(args.output_dir)
    args.crops_dir = os.path.abspath(args.crops_dir)
    args.face_cache_dir = os.path.abspath(args.face_cache_dir)
    args.train_csv = os.path.abspath(args.train_csv)
    args.val_csv = os.path.abspath(args.val_csv)
    args.test_csv = os.path.abspath(args.test_csv)
    pretrain_path = os.path.abspath(args.pretrain_path)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.crops_dir, exist_ok=True)

    # --- Auto-download TalkNet checkpoint if missing ---
    if not os.path.exists(pretrain_path):
        os.makedirs(os.path.dirname(pretrain_path), exist_ok=True)
        print(f"Downloading TalkNet-ASD checkpoint to {pretrain_path} ...")
        import subprocess
        subprocess.call(
            f"gdown --id 1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea -O {pretrain_path}", shell=True
        )

    # --- Precompute face crops ---
    if not args.skip_precompute:
        print("=== Precomputing face crops ===")
        all_df = pd.concat([
            pd.read_csv(args.train_csv),
            pd.read_csv(args.val_csv),
            pd.read_csv(args.test_csv),
        ], ignore_index=True)
        precompute_all_crops(all_df, args.face_cache_dir, args.crops_dir)

    # --- Build datasets ---
    train_dataset = ChildVocDataset(args.train_csv, args.crops_dir, args.face_cache_dir)
    val_dataset = ChildVocDataset(args.val_csv, args.crops_dir, args.face_cache_dir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    # --- Model ---
    device = args.device
    model = TalkNetChildDetector(pretrain_path, freeze_backbone=True).to(device)

    n_backbone = sum(p.numel() for p in model.talknet.model.parameters())
    n_head = sum(p.numel() for p in list(model.av_head.parameters()) +
                 list(model.a_head.parameters()))
    print(f"Backbone params: {n_backbone/1e6:.2f}M  |  Head params: {n_head}")

    # --- Phase 1: frozen backbone ---
    print(f"\n=== Phase 1: head-only ({args.phase1_epochs} epochs) ===")
    opt1 = torch.optim.Adam(
        list(model.av_head.parameters()) + list(model.a_head.parameters()),
        lr=args.lr_head,
    )
    sched1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=1, gamma=0.95)

    best_val_auroc, best_epoch = 0.0, 0
    best_ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pt")

    for epoch in range(1, args.phase1_epochs + 1):
        t0 = time.time()
        loss, _, _ = run_epoch(model, train_loader, opt1, device, train=True)
        sched1.step()
        _, val_labels, val_scores = run_epoch(model, val_loader, opt1, device, train=False)
        val_auroc = float(roc_auc_score(val_labels, val_scores)) if len(set(val_labels)) > 1 else 0.5
        elapsed = time.time() - t0
        print(f"  [P1 ep{epoch:02d}] loss={loss:.4f}  val_auroc={val_auroc:.4f}  ({elapsed:.0f}s)")
        if val_auroc > best_val_auroc:
            best_val_auroc, best_epoch = val_auroc, epoch
            torch.save(model.state_dict(), best_ckpt_path)

    # --- Phase 2: full fine-tune ---
    print(f"\n=== Phase 2: full fine-tune ({args.phase2_epochs} epochs) ===")
    model.unfreeze_backbone()
    opt2 = torch.optim.Adam([
        {"params": model.talknet.model.parameters(), "lr": args.lr_backbone},
        {"params": list(model.av_head.parameters()) + list(model.a_head.parameters()),
         "lr": args.lr_head},
    ])
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.phase2_epochs)

    for epoch in range(1, args.phase2_epochs + 1):
        t0 = time.time()
        loss, _, _ = run_epoch(model, train_loader, opt2, device, train=True)
        sched2.step()
        _, val_labels, val_scores = run_epoch(model, val_loader, opt2, device, train=False)
        val_auroc = float(roc_auc_score(val_labels, val_scores)) if len(set(val_labels)) > 1 else 0.5
        elapsed = time.time() - t0
        print(f"  [P2 ep{epoch:02d}] loss={loss:.4f}  val_auroc={val_auroc:.4f}  ({elapsed:.0f}s)")
        if val_auroc > best_val_auroc:
            best_val_auroc, best_epoch = val_auroc, args.phase1_epochs + epoch
            torch.save(model.state_dict(), best_ckpt_path)

    print(f"\nBest val AUROC={best_val_auroc:.4f} at epoch {best_epoch}. Loading best checkpoint.")

    # --- Final evaluation ---
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    val_metrics, val_preds = evaluate_split(
        model, args.val_csv, args.crops_dir, args.face_cache_dir, device
    )
    print(f"Val:  F1={val_metrics['f1']:.3f}  AUROC={val_metrics['auroc']:.3f}  "
          f"AUPRC={val_metrics['auprc']:.3f}  threshold={val_metrics['threshold']:.3f}")

    test_metrics, test_preds = evaluate_split(
        model, args.test_csv, args.crops_dir, args.face_cache_dir, device,
        threshold=val_metrics["threshold"],
    )
    print(f"Test: F1={test_metrics['f1']:.3f}  AUROC={test_metrics['auroc']:.3f}  "
          f"AUPRC={test_metrics['auprc']:.3f}")

    # --- Save outputs ---
    def _save_json(obj, path):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    _save_json(val_metrics, os.path.join(args.output_dir, "val_metrics_tuned.json"))
    _save_json(test_metrics, os.path.join(args.output_dir, "test_metrics_tuned.json"))
    val_preds.to_csv(os.path.join(args.output_dir, "val_predictions.csv"), index=False)
    test_preds.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

    config = {
        "pretrain_path": pretrain_path,
        "phase1_epochs": args.phase1_epochs,
        "phase2_epochs": args.phase2_epochs,
        "lr_head": args.lr_head,
        "lr_backbone": args.lr_backbone,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
    }
    _save_json(config, os.path.join(args.output_dir, "config.json"))

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
