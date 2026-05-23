"""ECAPA-TDNN fine-tune on TinyVox + Providence child speech (spec-021 US4 T071).

Loads the pre-trained `speechbrain/spkrec-ecapa-voxceleb` checkpoint, adds an
AAM-Softmax classification head over the speakers in
`models/ecapa_child_finetune/speaker_pair_manifest.csv`, and fine-tunes the
embedding network for ~10 epochs with class-balanced sampling.

This is *adaptation* training, not a from-scratch speaker-id task: the goal is
to nudge the ECAPA embedding space toward child acoustics so downstream cosine
enrollment (BabAR / VTC pipeline) discriminates target-child vs sibling better.

Output: models/ecapa_child_finetune/{best.pt, config.yaml, eer_log.json,
training_history.csv}.

CLI:
    python fit_ecapa_child.py --pairs models/ecapa_child_finetune/speaker_pair_manifest.csv \
        --epochs 10 --lr 1e-4 --batch 64 --out models/ecapa_child_finetune/
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.lobes.features import Fbank
from torch.utils.data import DataLoader, Dataset


SAMPLE_RATE = 16000
CROP_SECONDS = 3.0
CROP_SAMPLES = int(SAMPLE_RATE * CROP_SECONDS)
HELDOUT_FRAC = 0.05  # Per R4.1: hold out 5% of speakers for EER eval.


class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax (Deng 2019, ArcFace)."""

    def __init__(self, embedding_dim: int, n_classes: int,
                 margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_normal_(self.W)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb_n = F.normalize(embeddings, dim=-1)
        W_n = F.normalize(self.W, dim=-1)
        cos_theta = torch.clamp(emb_n @ W_n.t(), -1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos_theta)
        target_cos = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=cos_theta.size(1)).bool()
        logits = torch.where(one_hot, target_cos, cos_theta) * self.scale
        return F.cross_entropy(logits, labels)


class ChildSpeakerDataset(Dataset):
    """Random-crop dataset over the speaker manifest."""

    def __init__(self, df: pd.DataFrame, speaker_to_idx: dict[str, int],
                 crop_samples: int = CROP_SAMPLES):
        self.df = df.reset_index(drop=True)
        self.speaker_to_idx = speaker_to_idx
        self.crop = crop_samples

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int] | None:
        r = self.df.iloc[i]
        path = r["audio_path"]
        try:
            info = sf.info(path)
            sr_native = info.samplerate
            n_total = info.frames
            target_native = int(round(self.crop * sr_native / SAMPLE_RATE))
            if n_total <= target_native:
                audio, sr = sf.read(path, dtype="float32")
            else:
                start = random.randint(0, n_total - target_native)
                audio, sr = sf.read(path, frames=target_native, start=start,
                                    dtype="float32")
        except Exception:
            return None
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != SAMPLE_RATE:
            import scipy.signal as sps
            audio = sps.resample(audio, int(len(audio) * SAMPLE_RATE / sr)).astype("float32")
        n = len(audio)
        if n < self.crop:
            audio = np.pad(audio, (0, self.crop - n), mode="constant")
        elif n > self.crop:
            audio = audio[:self.crop]
        return torch.from_numpy(audio), self.speaker_to_idx[r["speaker_id"]]


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    audios = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return audios, labels


class ClassBalancedSampler(torch.utils.data.Sampler):
    """Sample (n_speakers_per_batch * utts_per_speaker) per call: pick speakers
    uniformly without replacement, then pick utts within each speaker."""

    def __init__(self, df: pd.DataFrame, speaker_to_idx: dict[str, int],
                 batch_size: int, n_batches_per_epoch: int, seed: int = 42):
        self.df = df.reset_index(drop=True)
        self.s2i = speaker_to_idx
        self.batch_size = batch_size
        self.n_batches = n_batches_per_epoch
        self.utts_per_speaker = 2  # contrast pairs per speaker in batch
        self.spkrs_per_batch = batch_size // self.utts_per_speaker
        self.spk_to_idx_list = (
            self.df.groupby("speaker_id").indices  # dict[speaker_id] -> np.ndarray[int]
        )
        self.spk_keys = [k for k in self.spk_to_idx_list.keys() if k in self.s2i]
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.n_batches):
            speakers = self.rng.choice(
                self.spk_keys, size=self.spkrs_per_batch,
                replace=len(self.spk_keys) < self.spkrs_per_batch,
            )
            indices = []
            for s in speakers:
                pool = self.spk_to_idx_list[s]
                pick = self.rng.choice(pool, size=self.utts_per_speaker,
                                       replace=len(pool) < self.utts_per_speaker)
                indices.extend(pick.tolist())
            for i in indices:
                yield int(i)

    def __len__(self):
        return self.n_batches * self.batch_size


def split_speakers(df: pd.DataFrame, heldout_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out `heldout_frac` of speakers entirely (unseen-speaker EER eval)."""
    rng = np.random.default_rng(seed)
    speakers = sorted(df["speaker_id"].unique())
    n_held = max(1, int(round(len(speakers) * heldout_frac)))
    held = set(rng.choice(speakers, size=n_held, replace=False).tolist())
    train = df[~df["speaker_id"].isin(held)].reset_index(drop=True)
    heldout = df[df["speaker_id"].isin(held)].reset_index(drop=True)
    return train, heldout


def encode(model, fbank, audio: torch.Tensor) -> torch.Tensor:
    """Run audio (B, T) through ECAPA encoder → embeddings (B, E)."""
    feats = fbank(audio)  # (B, T_frames, n_mel)
    feats = (feats - feats.mean(dim=1, keepdim=True)) / (feats.std(dim=1, keepdim=True) + 1e-5)
    emb = model.encode_batch(audio).squeeze(1)  # (B, E)
    return emb


def evaluate_eer(model, heldout_df: pd.DataFrame, device: torch.device,
                 max_pairs: int = 2000, seed: int = 42) -> dict:
    """Sample positive (same-speaker) and negative (different-speaker) pairs,
    score each with cosine similarity, compute EER."""
    if heldout_df.empty:
        return {"eer": None, "n_pos": 0, "n_neg": 0, "note": "no heldout speakers"}
    rng = np.random.default_rng(seed)
    spk_groups = heldout_df.groupby("speaker_id").indices
    spk_keys = list(spk_groups.keys())
    pairs, labels = [], []
    target_pos = max_pairs // 2
    n_attempt = 0
    while sum(labels) < target_pos and n_attempt < target_pos * 5:
        s = rng.choice(spk_keys)
        pool = spk_groups[s]
        if len(pool) < 2:
            n_attempt += 1
            continue
        i, j = rng.choice(pool, size=2, replace=False)
        pairs.append((heldout_df.iloc[i]["audio_path"], heldout_df.iloc[j]["audio_path"]))
        labels.append(1)
        n_attempt += 1
    while len(labels) - sum(labels) < (max_pairs - target_pos):
        s1, s2 = rng.choice(spk_keys, size=2, replace=False)
        i = rng.choice(spk_groups[s1])
        j = rng.choice(spk_groups[s2])
        pairs.append((heldout_df.iloc[i]["audio_path"], heldout_df.iloc[j]["audio_path"]))
        labels.append(0)

    print(f"  EER eval: {len(labels)} pairs ({sum(labels)} pos / {len(labels)-sum(labels)} neg)")
    model.eval()
    cache = {}

    def emb_for(p: str) -> np.ndarray | None:
        if p in cache:
            return cache[p]
        try:
            info = sf.info(p)
            sr_native = info.samplerate
            target_native = int(round(CROP_SAMPLES * sr_native / SAMPLE_RATE))
            if info.frames <= target_native:
                audio, sr = sf.read(p, dtype="float32")
            else:
                audio, sr = sf.read(p, frames=target_native, dtype="float32")
        except Exception:
            return None
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != SAMPLE_RATE:
            import scipy.signal as sps
            audio = sps.resample(audio, int(len(audio) * SAMPLE_RATE / sr)).astype("float32")
        if len(audio) < CROP_SAMPLES:
            audio = np.pad(audio, (0, CROP_SAMPLES - len(audio)))
        else:
            audio = audio[:CROP_SAMPLES]
        with torch.no_grad():
            t = torch.from_numpy(audio).unsqueeze(0).to(device)
            emb = model.encode_batch(t).squeeze(0).squeeze(0).cpu().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        cache[p] = emb
        return emb

    scores = []
    for (a, b) in pairs:
        ea, eb = emb_for(a), emb_for(b)
        if ea is None or eb is None:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(ea, eb)))
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    return {"eer": eer, "n_pos": int(sum(labels)), "n_neg": int(len(labels) - sum(labels))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-batches", type=int, default=200,
                    help="batches per epoch (class-balanced sampler is infinite)")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source", default="speechbrain/spkrec-ecapa-voxceleb")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[startup] argparse done; out={args.out}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"[startup] seeds set; checking CUDA...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[startup] device={device}", flush=True)
    print(f"Loading ECAPA from {args.source} on {device}", flush=True)
    encoder = EncoderClassifier.from_hparams(
        source=args.source,
        savedir=str(args.out / "ecapa_pretrained"),
        run_opts={"device": str(device)},
    )

    print(f"Reading manifest {args.pairs}")
    df = pd.read_csv(args.pairs)
    df = df.dropna(subset=["audio_path", "speaker_id"]).reset_index(drop=True)
    print(f"  rows={len(df)}, speakers={df['speaker_id'].nunique()}")

    train_df, held_df = split_speakers(df, HELDOUT_FRAC, args.seed)
    print(f"  train: {len(train_df)} rows, {train_df['speaker_id'].nunique()} speakers")
    print(f"  heldout: {len(held_df)} rows, {held_df['speaker_id'].nunique()} speakers")

    speakers = sorted(train_df["speaker_id"].unique())
    s2i = {s: i for i, s in enumerate(speakers)}
    n_classes = len(speakers)

    embedding_dim = 192  # ECAPA-VoxCeleb default
    head = AAMSoftmax(embedding_dim, n_classes, margin=args.margin, scale=args.scale).to(device)
    encoder.mods.embedding_model.train()

    params = list(encoder.mods.embedding_model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    ds = ChildSpeakerDataset(train_df, s2i)
    sampler = ClassBalancedSampler(train_df, s2i, args.batch, args.n_batches, args.seed)
    loader = DataLoader(ds, batch_size=args.batch, sampler=sampler,
                        collate_fn=collate_skip_none, num_workers=0,
                        drop_last=True)

    history, best_eer, best_path = [], None, args.out / "best.pt"
    pre_eer = evaluate_eer(encoder, held_df, device, seed=args.seed)
    print(f"Pre-fine-tune EER: {pre_eer}")

    for epoch in range(1, args.epochs + 1):
        encoder.mods.embedding_model.train()
        head.train()
        t0 = time.time()
        loss_sum, n = 0.0, 0
        for batch in loader:
            if batch is None:
                continue
            audio, labels = batch
            audio = audio.to(device)
            labels = labels.to(device)
            emb = encoder.encode_batch(audio).squeeze(1)
            loss = head(emb, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach())
            n += 1
            if n % 25 == 0:
                print(f"  epoch {epoch} batch {n}/{args.n_batches}  "
                      f"running_loss={loss_sum/n:.4f}  "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
        avg_loss = loss_sum / max(1, n)
        eer_info = evaluate_eer(encoder, held_df, device, seed=args.seed + epoch)
        eer = eer_info.get("eer")
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "loss": avg_loss, "eer": eer,
                        "elapsed_sec": round(elapsed, 1)})
        print(f"epoch {epoch}: loss={avg_loss:.4f} eer={eer} time={elapsed:.1f}s")
        if eer is not None and (best_eer is None or eer < best_eer):
            best_eer = eer
            torch.save({
                "embedding_state_dict": encoder.mods.embedding_model.state_dict(),
                "head_state_dict": head.state_dict(),
                "speakers": speakers,
                "embedding_dim": embedding_dim,
                "epoch": epoch,
                "eer": eer,
            }, best_path)
            print(f"  saved best.pt (eer={eer:.4f})")

    pd.DataFrame(history).to_csv(args.out / "training_history.csv", index=False)
    eer_log = {
        "pre_finetune_eer": pre_eer,
        "post_finetune_best_eer": best_eer,
        "history": history,
        "n_speakers_train": int(n_classes),
        "n_heldout_speakers": int(held_df["speaker_id"].nunique()),
    }
    (args.out / "eer_log.json").write_text(json.dumps(eer_log, indent=2))
    (args.out / "config.yaml").write_text(json.dumps({
        "source": args.source,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "n_batches": args.n_batches,
        "margin": args.margin,
        "scale": args.scale,
        "seed": args.seed,
        "n_classes": n_classes,
    }, indent=2))
    print(f"Wrote eer_log.json: {eer_log}")
    print(f"Best EER: {best_eer}")


if __name__ == "__main__":
    main()
