"""Short-vocalization specialized head (spec-012 US4).

Identifies clips with CHI segments < 0.5s from USC-SAIL RTTM cache,
trains a 1D-CNN head (500ms windows, 250ms hop) over frozen WavLM-Base+
features, and merges with the main pipeline via a val-tuned beta weight.

Usage:
  python evaluation/short_voc_head.py [--dry-run]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mil.mil_model import BackboneExtractor
from evaluation.metadata_router import (
    BASELINE_AUROC,
    BASELINE_F1,
    SEED,
    compute_metrics,
    load_metadata,
    load_split,
    load_system_scores,
    save_results,
    tune_threshold,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_REPO, "mil/mil_results/short_voc_head")
BACKBONE = "microsoft/wavlm-base-plus"
USC_SAIL_RTTM_CACHE = os.path.join(_REPO, "whisper-modeling/usc_sail_rttm_cache")
VTC_RTTM_CACHE = os.path.join(_REPO, "pyannote/vtc_rttm_cache")
HARD_NEG_CSV = os.path.join(_REPO, "synth_results/manifests/hard_negatives_manifest.csv")
SR = 16000
MAX_SAMPLES = SR * 30  # 30-second cap
EMB_DIM = 768
# 1D-CNN parameters (20ms WavLM frames):
CNN_KERNEL = 25   # 500ms window
CNN_STRIDE = 12   # ~240ms hop


# ── RTTM utilities ───────────────────────────────────────────────────────────

def _rttm_path(audio_path: str, cache_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    cid = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{stem}__{cid}.rttm")


def _parse_chi_durations(rttm_path: str) -> list:
    durations = []
    if not os.path.exists(rttm_path):
        return durations
    try:
        with open(rttm_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[7] == "CHI":
                    durations.append(float(parts[4]))
    except Exception:
        pass
    return durations


def identify_short_voc_clips(split_df: pd.DataFrame,
                              threshold_sec: float = 0.5) -> np.ndarray:
    """Return boolean mask where any CHI segment < threshold_sec."""
    mask = np.zeros(len(split_df), dtype=bool)
    found_rttm, short_voc = 0, 0
    for i, (_, row) in enumerate(split_df.iterrows()):
        path = row["audio_path"]
        durations = _parse_chi_durations(_rttm_path(path, USC_SAIL_RTTM_CACHE))
        if not durations:
            durations = _parse_chi_durations(_rttm_path(path, VTC_RTTM_CACHE))
        if durations:
            found_rttm += 1
        if any(d < threshold_sec for d in durations):
            mask[i] = True
            short_voc += 1
    pos = int(split_df["label"].sum())
    print(f"  RTTM found: {found_rttm}/{len(split_df)} | "
          f"short-voc clips: {short_voc} | "
          f"short-voc among positives: "
          f"{int((mask & (split_df['label']==1).to_numpy()).sum())}/{pos}", flush=True)
    return mask


# ── Model ────────────────────────────────────────────────────────────────────

class ShortVocHead(nn.Module):
    """Frozen WavLM-Base+ + 1D-CNN (500ms/250ms) + max-pool + linear."""

    def __init__(self, embed_dim: int = EMB_DIM,
                 kernel_size: int = CNN_KERNEL, stride: int = CNN_STRIDE) -> None:
        super().__init__()
        self.backbone = BackboneExtractor(BACKBONE)
        self.cnn = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size,
                             stride=stride, padding=0)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, T)  →  logit: (B,)"""
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        with torch.no_grad():
            frame_embs = self.backbone(waveform)     # (B, T_frames, D)
        x = frame_embs.transpose(1, 2)               # (B, D, T_frames)
        x = F.relu(self.cnn(x))                      # (B, D, T_short)
        x = x.max(dim=2).values                      # (B, D)  global max-pool
        return self.head(x).squeeze(-1)              # (B,)


# ── Dataset ──────────────────────────────────────────────────────────────────

class AudioClipDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        import torchaudio
        row = self.df.iloc[idx]
        try:
            wav, sr = torchaudio.load(row["audio_path"])
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            # Slice if start_sec/end_sec present (hard negatives)
            if "start_sec" in row and pd.notna(row.get("start_sec")):
                s = int(row["start_sec"] * SR)
                e = int(row["end_sec"] * SR) if pd.notna(row.get("end_sec")) else wav.shape[-1]
                wav = wav[:, s:e]
            wav = wav[:, :MAX_SAMPLES]
            # Pad to MAX_SAMPLES
            if wav.shape[-1] < MAX_SAMPLES:
                wav = F.pad(wav, (0, MAX_SAMPLES - wav.shape[-1]))
        except Exception:
            wav = torch.zeros(1, MAX_SAMPLES)
        return wav.squeeze(0), int(row["label"])


def _collate(batch):
    wavs, labels = zip(*batch)
    return torch.stack(wavs), torch.tensor(labels, dtype=torch.float32)


# ── Training ─────────────────────────────────────────────────────────────────

def _build_train_df(split_df: pd.DataFrame) -> pd.DataFrame:
    """Build training set: all train positives + hard negatives (cap=344)."""
    pos_df = split_df[split_df["label"] == 1][["audio_path", "label",
                                               "timepoint_norm"]].copy()
    hard_neg = pd.read_csv(HARD_NEG_CSV)
    hard_neg = hard_neg[["audio_path", "label", "timepoint_norm",
                          "start_sec", "end_sec"]].copy()
    hard_neg = hard_neg.head(344)  # cap to match original negative count
    return pd.concat([pos_df, hard_neg], ignore_index=True)


def train_short_head(train_split_df: pd.DataFrame, val_df: pd.DataFrame,
                     device: torch.device, seed: int = SEED,
                     epochs: int = 15, patience: int = 5) -> ShortVocHead:
    torch.manual_seed(seed)
    train_df = _build_train_df(train_split_df)
    print(f"  Short-head train set: {len(train_df)} clips "
          f"(pos={int(train_df['label'].sum())}, neg={len(train_df)-int(train_df['label'].sum())})",
          flush=True)

    model = ShortVocHead().to(device)
    # Freeze backbone; only CNN + head are trainable
    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(
        list(model.cnn.parameters()) + list(model.head.parameters()), lr=1e-3
    )
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(AudioClipDataset(train_df), batch_size=8,
                              shuffle=True, collate_fn=_collate, num_workers=2)
    val_loader = DataLoader(AudioClipDataset(val_df), batch_size=16,
                            shuffle=False, collate_fn=_collate, num_workers=2)

    best_val_f1, best_state, wait = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for wavs, labels in train_loader:
            wavs, labels = wavs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(wavs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate on val short-voc clips
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for wavs, labels in val_loader:
                wavs = wavs.to(device)
                probs = torch.sigmoid(model(wavs)).cpu().numpy()
                val_probs.extend(probs.tolist())
                val_labels.extend(labels.numpy().tolist())

        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels, dtype=int)
        t = tune_threshold(val_labels, val_probs)
        val_f1 = compute_metrics(val_labels, val_probs, t)["f1"]
        print(f"  Epoch {epoch:2d}: train_loss={train_loss/len(train_loader):.4f} "
              f"val_F1={val_f1:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = os.path.join(OUT_DIR, "best_checkpoint.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"  Best val F1={best_val_f1:.4f} | checkpoint: {ckpt}", flush=True)
    return model


# ── Inference & evaluation ───────────────────────────────────────────────────

def _run_inference(df: pd.DataFrame, model: ShortVocHead,
                   device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(AudioClipDataset(df), batch_size=16, shuffle=False,
                        collate_fn=_collate, num_workers=2)
    probs = []
    with torch.no_grad():
        for wavs, _ in loader:
            p = torch.sigmoid(model(wavs.to(device))).cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


def _best_audio_mil(df: pd.DataFrame) -> np.ndarray:
    cols = ["babar_prob", "vtc_prob", "wavlm_mil_prob", "whisper_mil_prob"]
    avail = [c for c in cols if c in df.columns]
    return df[avail].mean(axis=1).to_numpy(dtype=float)


def merge_and_evaluate(test_df: pd.DataFrame, val_df: pd.DataFrame,
                       model: ShortVocHead, short_voc_mask_test: np.ndarray,
                       device: torch.device) -> None:
    print("\nRunning inference on val and test ...", flush=True)
    val_head = _run_inference(val_df, model, device)
    test_head = _run_inference(test_df, model, device)

    val_main = _best_audio_mil(val_df)
    test_main = _best_audio_mil(test_df)
    y_val = val_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    # Tune beta on val (sweep 0.0–1.0 in 0.05 steps)
    best_beta, best_f1 = 0.5, -1.0
    for beta in np.arange(0.0, 1.05, 0.05):
        merged_val = beta * val_main + (1 - beta) * val_head
        t_tmp = tune_threshold(y_val, merged_val)
        f1 = compute_metrics(y_val, merged_val, t_tmp)["f1"]
        if f1 > best_f1:
            best_f1, best_beta = f1, float(beta)
    print(f"  Best beta={best_beta:.2f} (val F1={best_f1:.4f})", flush=True)

    # Apply to test
    final_val = best_beta * val_main + (1 - best_beta) * val_head
    final_test = best_beta * test_main + (1 - best_beta) * test_head
    t = tune_threshold(y_val, final_val)

    val_m = compute_metrics(y_val, final_val, t)
    val_m["threshold"] = t
    test_m = compute_metrics(y_test, final_test, t)
    test_m["threshold"] = t

    # Per-stratum: short-voc vs. non-short-voc test clips
    def _stratum(mask, tag):
        if mask.sum() == 0:
            return
        before = (test_main[mask] >= t).astype(int)
        after = (final_test[mask] >= t).astype(int)
        gt = y_test[mask]
        was_wrong = before != gt
        now_right = after == gt
        n_recovered = int((was_wrong & now_right).sum())
        was_right = before == gt
        now_wrong = after != gt
        n_hurt = int((was_right & now_wrong).sum())
        m = compute_metrics(gt, final_test[mask], t)
        m.update({
            "before_f1": float(compute_metrics(gt, test_main[mask], t)["f1"]),
            "after_f1": float(m["f1"]),
            "n_recovered": n_recovered,
            "n_hurt": n_hurt,
            "n": int(mask.sum()),
        })
        path = os.path.join(OUT_DIR, f"test_metrics_{tag}.json")
        with open(path, "w") as f:
            json.dump(m, f, indent=2)
        print(f"  {tag}: before_F1={m['before_f1']:.4f} after_F1={m['after_f1']:.4f} "
              f"n_recovered={n_recovered} n_hurt={n_hurt}", flush=True)

    _stratum(short_voc_mask_test, "short_voc_clips")
    _stratum(~short_voc_mask_test, "non_short_voc_clips")

    preds = test_df[["audio_path", "label", "timepoint_norm"]].copy()
    preds["score"] = final_test
    preds["main_score"] = test_main
    preds["head_score"] = test_head
    preds["prediction"] = (final_test >= t).astype(int)
    preds["short_voc"] = short_voc_mask_test.astype(int)

    cfg = {
        "sub_feature": "D",
        "backbone": BACKBONE,
        "beta": best_beta,
        "val_threshold": t,
        "cnn_kernel_ms": CNN_KERNEL * 20,
        "cnn_stride_ms": CNN_STRIDE * 20,
        "seed": SEED,
        "created": "2026-04-28",
    }
    save_results(OUT_DIR, val_m, test_m, preds, cfg)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Short-vocalization specialized head")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print short-voc clip counts and exit before training")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("logs/evaluation", exist_ok=True)

    print("Loading val/test system scores ...", flush=True)
    val_scores = load_system_scores("val")
    test_scores = load_system_scores("test")
    meta = load_metadata()
    train_split_df = meta[meta["split"] == "train"].copy()
    val_df = load_split(val_scores, meta, "val")
    test_df = load_split(test_scores, meta, "test")

    print("\nIdentifying short-voc clips in train split:", flush=True)
    short_train = identify_short_voc_clips(train_split_df)
    print("\nIdentifying short-voc clips in val split:", flush=True)
    short_val = identify_short_voc_clips(val_df)
    print("\nIdentifying short-voc clips in test split:", flush=True)
    short_test = identify_short_voc_clips(test_df)

    if args.dry_run:
        print("\n--dry-run: exiting before training.", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}", flush=True)

    print("\nTraining short-voc head ...", flush=True)
    model = train_short_head(train_split_df, val_df, device,
                             seed=args.seed, epochs=args.epochs, patience=args.patience)

    merge_and_evaluate(test_df, val_df, model, short_test, device)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
