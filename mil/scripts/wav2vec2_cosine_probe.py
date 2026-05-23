"""Cosine-prototype probe for spec-021 US2 (T033-T035).

Per R2.2: load the converted wav2vec2 LL_4300 backbone via Wav2Vec2Model,
take last-hidden-state mean-pooled embeddings of 30 random val clips, build
per-target-child prototypes from train positives (positive clips of that
child), and rank val clips by cosine similarity to their target child's
prototype. Emit val AUROC + verdict.

Verdict bar (SC-010 + R2.2): val AUROC >= 0.55 == POSITIVE (any signal),
< 0.55 == NEGATIVE.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import roc_auc_score
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != target_sr:
        # Lazy resample to avoid librosa dependency; only fires if needed.
        import scipy.signal as sps
        new_len = int(len(audio) * target_sr / sr)
        audio = sps.resample(audio, new_len).astype("float32")
    return audio


def embed_clips(model: Wav2Vec2Model, fe: Wav2Vec2FeatureExtractor,
                paths: list[str], device: torch.device,
                max_seconds: float = 10.0) -> dict[str, np.ndarray]:
    """Returns {path: 768-d mean-pooled embedding}."""
    out = {}
    for p in paths:
        if p in out:
            continue
        try:
            audio = load_wav(p)
        except Exception as e:
            print(f"  SKIP {p}: load failed: {e}")
            continue
        if len(audio) < 1600:
            print(f"  SKIP {p}: too short ({len(audio)} samples)")
            continue
        # Cap to max_seconds to keep CPU runtime bounded.
        max_samp = int(max_seconds * 16000)
        if len(audio) > max_samp:
            audio = audio[:max_samp]
        inp = fe(audio, sampling_rate=16000, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            h = model(**inp).last_hidden_state  # (1, T, 768)
        emb = h.mean(dim=1).squeeze(0).cpu().numpy()
        out[p] = emb / (np.linalg.norm(emb) + 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="models/wav2vec2_naturalistic_LL_4300_hf")
    ap.add_argument("--train-csv", default="whisper-modeling/seen_child_splits/train.csv")
    ap.add_argument("--val-csv", default="whisper-modeling/seen_child_splits/val.csv")
    ap.add_argument("--n-val-sample", type=int, default=30,
                    help="Number of val clips to score (per spec FR-011)")
    ap.add_argument("--max-protos-per-child", type=int, default=8,
                    help="Cap train-positive clips used for each prototype")
    ap.add_argument("--max-seconds", type=float, default=10.0,
                    help="Cap audio length to bound CPU runtime")
    ap.add_argument("--output-dir", default="mil/mil_results/wav2vec2_naturalistic_probe", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading backbone from {args.backbone} on {device}...")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.backbone)
    model = Wav2Vec2Model.from_pretrained(args.backbone).to(device).eval()

    train = pd.read_csv(args.train_csv)
    val = pd.read_csv(args.val_csv)

    # 1) Sample 30 random val clips (any label, any child).
    val_sample = val.sample(n=min(args.n_val_sample, len(val)), random_state=args.seed)
    print(f"Val sample: {len(val_sample)} clips, "
          f"label_counts={val_sample['label'].value_counts().to_dict()}, "
          f"distinct_children={val_sample['child_id'].nunique()}")

    # 2) Build a prototype per target child for which train has >=1 positive.
    target_children = sorted(val_sample["child_id"].unique())
    proto_paths_per_child = {}
    for c in target_children:
        train_pos = train[(train["child_id"] == c) & (train["label"] == 1)]
        if len(train_pos) == 0:
            print(f"  WARN no train positives for child {c}; skipping clips of this child")
            continue
        subset = train_pos.sample(n=min(args.max_protos_per_child, len(train_pos)),
                                  random_state=args.seed)
        proto_paths_per_child[c] = subset["audio_path"].tolist()

    proto_path_set = set()
    for ps in proto_paths_per_child.values():
        proto_path_set.update(ps)
    val_path_set = set(val_sample["audio_path"].tolist())
    print(f"Embedding {len(proto_path_set)} prototype clips + {len(val_path_set)} val clips...")

    t0 = time.time()
    proto_embs = embed_clips(model, fe, sorted(proto_path_set), device,
                             max_seconds=args.max_seconds)
    val_embs = embed_clips(model, fe, sorted(val_path_set), device,
                           max_seconds=args.max_seconds)
    elapsed = time.time() - t0
    print(f"Embedding done in {elapsed:.1f}s")

    # 3) Mean-pool prototypes per child.
    child_protos = {}
    for c, ps in proto_paths_per_child.items():
        embs = [proto_embs[p] for p in ps if p in proto_embs]
        if not embs:
            continue
        proto = np.mean(np.stack(embs, axis=0), axis=0)
        child_protos[c] = proto / (np.linalg.norm(proto) + 1e-9)
    print(f"Built prototypes for {len(child_protos)} target children")

    # 4) Score val sample.
    rows = []
    for _, r in val_sample.iterrows():
        c = r["child_id"]
        path = r["audio_path"]
        proto = child_protos.get(c)
        emb = val_embs.get(path)
        if proto is None or emb is None:
            continue
        cos = float(np.dot(emb, proto))
        rows.append({
            "audio_path": path,
            "child_id": c,
            "timepoint_norm": r["timepoint_norm"],
            "label": int(r["label"]),
            "score": cos,
            "n_proto_clips": len(proto_paths_per_child[c]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "cosine_prototype_val.csv", index=False)
    print(f"Wrote cosine_prototype_val.csv ({len(df)} rows)")

    # 5) AUROC + verdict.
    if df["label"].nunique() < 2:
        verdict = "NEGATIVE"
        rationale = f"val sample has only one class ({df['label'].unique().tolist()})"
        auroc = None
    else:
        auroc = float(roc_auc_score(df["label"], df["score"]))
        verdict = "POSITIVE" if auroc >= 0.55 else "NEGATIVE"
        rationale = f"val AUROC = {auroc:.4f} {'>=' if auroc >= 0.55 else '<'} 0.55 minimum-signal bar"

    summary = {
        "backbone": args.backbone,
        "n_val_clips_scored": int(len(df)),
        "n_target_children": int(df["child_id"].nunique()),
        "auroc_val": auroc,
        "verdict": verdict,
        "rationale": rationale,
        "elapsed_seconds": round(elapsed, 1),
        "device": str(device),
    }
    (args.output_dir / "config.yaml").write_text(json.dumps({
        "backbone": args.backbone,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "n_val_sample": args.n_val_sample,
        "max_protos_per_child": args.max_protos_per_child,
        "max_seconds": args.max_seconds,
        "seed": args.seed,
    }, indent=2))
    (args.output_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
