"""
Proxy quality metrics on unlabeled core dataset recordings.

For each session WAV in --core-dir:
  1. Run BabAR and USC-SAIL diarizers to get child segments
  2. Embed child segments with ECAPA and compute cosine similarity to age-group prototype
  3. Compute inter-frontend agreement (child-present per 10ms frame)

Outputs:
  {output_dir}/
    config.json
    per_session_scores.csv        # cosine similarity per session per frontend
    inter_frontend_agreement.csv  # frame-level agreement BabAR vs USC-SAIL
    detection_rate_stats.csv      # detection rate per frontend summary

Usage:
    python pyannote/proxy_analysis.py \\
        --core-dir core/audio/ \\
        --prototype-dir pyannote/age_group_prototypes/ \\
        --output-dir pyannote/core_proxy_analysis/
"""

import argparse
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
from unified import (
    BaseConfig,
    BabARFrontend,
    ECAPAEmbedder,
    VBxFrontend,
    VTCFrontend,
    build_frontend,
    cosine_similarity,
    extract_segment_embeddings,
    l2_normalize,
    save_json,
)

FRAME_STEP_SEC = 0.01  # 10ms


def _get_audio_duration(audio_path: str) -> float:
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate


def _load_prototype(prototype_dir: str, age_group: str) -> np.ndarray | None:
    for name in (f"{age_group}.pt", f"{age_group}_prototype.pt",
                 f"prototype_{age_group}.pt"):
        p = os.path.join(prototype_dir, name)
        if os.path.exists(p):
            t = torch.load(p, map_location="cpu")
            if isinstance(t, dict):
                t = t.get("embedding", next(iter(t.values())))
            arr = t.squeeze().float().numpy()
            return l2_normalize(arr)
    return None


def _age_group_from_manifest(session_id: str, manifest_path: str) -> str:
    if not manifest_path or not os.path.exists(manifest_path):
        return "unknown"
    df = pd.read_csv(manifest_path)
    if "session_id" in df.columns and "age_group" in df.columns:
        row = df[df["session_id"] == session_id]
        if len(row):
            return str(row.iloc[0]["age_group"])
    return "unknown"


def _build_child_frame_mask(segments: list, duration_sec: float) -> np.ndarray:
    """Convert child segments to binary 10ms frame mask."""
    n_frames = max(1, math.ceil(duration_sec / FRAME_STEP_SEC))
    mask = np.zeros(n_frames, dtype=bool)
    for seg in segments:
        start_f = int(seg["start"] / FRAME_STEP_SEC)
        end_f = min(n_frames, math.ceil(seg["end"] / FRAME_STEP_SEC))
        mask[start_f:end_f] = True
    return mask


def _compute_session_scores(
    audio_path: str,
    frontend,
    embedder: ECAPAEmbedder,
    cfg: BaseConfig,
    prototype: np.ndarray | None,
) -> tuple[dict, np.ndarray]:
    """Return (scores_dict, frame_mask) for one session/frontend."""
    try:
        segments = frontend.get_segments(audio_path, cfg)
    except Exception as e:
        duration = _get_audio_duration(audio_path)
        empty_mask = np.zeros(max(1, math.ceil(duration / FRAME_STEP_SEC)), dtype=bool)
        return {"detection_rate": float("nan"), "mean_similarity": float("nan"),
                "n_segments": 0, "error": str(e)}, empty_mask

    duration = _get_audio_duration(audio_path)
    mask = _build_child_frame_mask(segments, duration)
    n_segments = len(segments)

    if n_segments == 0:
        return {"detection_rate": 0.0, "mean_similarity": float("nan"),
                "n_segments": 0}, mask

    child_dur = sum(s["end"] - s["start"] for s in segments)
    detection_rate = child_dur / duration if duration > 0 else float("nan")

    mean_sim = float("nan")
    if prototype is not None:
        embs = extract_segment_embeddings(
            audio_path, segments, embedder, cfg, max_segments=30
        )
        if len(embs) > 0:
            sims = [cosine_similarity(e, prototype) for e in embs]
            mean_sim = float(np.mean(sims))

    return {
        "detection_rate": detection_rate,
        "mean_similarity": mean_sim,
        "n_segments": n_segments,
    }, mask


def _compute_agreement(mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    n = min(len(mask_a), len(mask_b))
    if n == 0:
        return {"agreement_rate": float("nan"), "n_frames": 0,
                "both_child": 0, "neither": 0, "only_a": 0, "only_b": 0}
    a, b = mask_a[:n].astype(bool), mask_b[:n].astype(bool)
    both = int((a & b).sum())
    neither = int((~a & ~b).sum())
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    return {
        "agreement_rate": float((both + neither) / n),
        "n_frames": n,
        "both_child": both,
        "neither": neither,
        "only_a": only_a,
        "only_b": only_b,
    }


def main():
    parser = argparse.ArgumentParser(description="Proxy analysis on core dataset.")
    parser.add_argument("--core-dir", required=True,
                        help="Directory of core dataset WAV files.")
    parser.add_argument("--prototype-dir", default="",
                        help="Directory of age-group ECAPA prototype .pt files.")
    parser.add_argument("--output-dir", default="",
                        help="Output dir (default: pyannote/core_proxy_analysis/).")
    parser.add_argument("--manifest", default="",
                        help="Optional CSV with session_id, age_group columns.")
    parser.add_argument("--frontends", default="babar,usc_sail",
                        help="Comma-separated diarizer names (default: babar,usc_sail).")
    parser.add_argument("--babar-dir", default="")
    parser.add_argument("--vtc-dir", default="")
    parser.add_argument("--vbx-dir", default="")
    parser.add_argument("--splits-dir", default="")
    args = parser.parse_args()

    if not os.path.isdir(args.core_dir):
        print(f"ERROR: --core-dir not found: {args.core_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = BaseConfig()
    if args.splits_dir:
        cfg.split_dir = args.splits_dir
    if args.babar_dir:
        cfg.babar_dir = args.babar_dir
    if args.vtc_dir:
        cfg.vtc_dir = args.vtc_dir
    if args.vbx_dir:
        cfg.vbx_dir = args.vbx_dir

    here = Path(__file__).parent
    out_dir = Path(args.output_dir or str(here / "core_proxy_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)

    frontend_names = [f.strip() for f in args.frontends.split(",") if f.strip()]
    if not frontend_names:
        print("ERROR: --frontends must specify at least one diarizer.", file=sys.stderr)
        sys.exit(1)

    wav_files = sorted(Path(args.core_dir).glob("**/*.wav")) + \
                sorted(Path(args.core_dir).glob("**/*.flac"))
    if not wav_files:
        print(f"ERROR: No WAV/FLAC files found in {args.core_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(wav_files)} audio files")

    audio_paths = [str(p) for p in wav_files]
    frontends: dict = {}
    for name in frontend_names:
        try:
            fe = build_frontend(name, cfg)
            if isinstance(fe, (BabARFrontend, VTCFrontend, VBxFrontend)):
                fe.prepare(audio_paths)
            frontends[name] = fe
            print(f"Initialized frontend: {name}")
        except Exception as e:
            print(f"WARNING: Could not build frontend '{name}': {e}", file=sys.stderr)

    if not frontends:
        print("ERROR: No frontends could be initialized.", file=sys.stderr)
        sys.exit(1)

    embedder = ECAPAEmbedder(cfg.ecapa_source, cfg.device)

    prototypes: dict = {}
    if args.prototype_dir and os.path.isdir(args.prototype_dir):
        for ag in ("12_16m", "34_38m"):
            prototypes[ag] = _load_prototype(args.prototype_dir, ag)
            status = "loaded" if prototypes[ag] is not None else "NOT FOUND"
            print(f"Prototype {ag}: {status}")

    session_rows = []
    agreement_rows = []

    for wav_path in wav_files:
        session_id = wav_path.stem
        age_group = _age_group_from_manifest(session_id, args.manifest)
        proto = prototypes.get(age_group) or prototypes.get("12_16m")
        masks_this_session: dict[str, np.ndarray] = {}

        for fe_name, fe in frontends.items():
            print(f"  [{session_id}] {fe_name} ...", end=" ", flush=True)
            scores, mask = _compute_session_scores(str(wav_path), fe, embedder, cfg, proto)
            masks_this_session[fe_name] = mask
            session_rows.append({
                "session_id": session_id,
                "age_group": age_group,
                "frontend": fe_name,
                "detection_rate": scores["detection_rate"],
                "mean_similarity": scores["mean_similarity"],
                "n_segments": scores["n_segments"],
            })
            sim_str = f"{scores['mean_similarity']:.3f}" if not isinstance(
                scores["mean_similarity"], float) or not np.isnan(
                scores["mean_similarity"]) else "nan"
            print(f"dr={scores['detection_rate']:.3f} sim={sim_str}")

        fe_list = list(frontends.keys())
        for i in range(len(fe_list)):
            for j in range(i + 1, len(fe_list)):
                a_name, b_name = fe_list[i], fe_list[j]
                agr = _compute_agreement(
                    masks_this_session.get(a_name, np.array([])),
                    masks_this_session.get(b_name, np.array([])),
                )
                agreement_rows.append({
                    "session_id": session_id,
                    "age_group": age_group,
                    "frontend_a": a_name,
                    "frontend_b": b_name,
                    **agr,
                })

    sessions_df = pd.DataFrame(session_rows)
    sessions_df.to_csv(out_dir / "per_session_scores.csv", index=False)

    agreement_df = pd.DataFrame(agreement_rows) if agreement_rows else pd.DataFrame(
        columns=["session_id", "age_group", "frontend_a", "frontend_b",
                 "agreement_rate", "n_frames", "both_child", "neither", "only_a", "only_b"]
    )
    agreement_df.to_csv(out_dir / "inter_frontend_agreement.csv", index=False)

    stats_rows = []
    for fe_name in frontends:
        sub = sessions_df[sessions_df["frontend"] == fe_name]
        dr = sub["detection_rate"].dropna()
        sim = sub["mean_similarity"].dropna()
        stats_rows.append({
            "frontend": fe_name,
            "n_sessions": len(sub),
            "detection_rate_mean": float(dr.mean()) if len(dr) else float("nan"),
            "detection_rate_std": float(dr.std()) if len(dr) else float("nan"),
            "detection_rate_median": float(dr.median()) if len(dr) else float("nan"),
            "similarity_mean": float(sim.mean()) if len(sim) else float("nan"),
            "similarity_std": float(sim.std()) if len(sim) else float("nan"),
        })
    pd.DataFrame(stats_rows).to_csv(out_dir / "detection_rate_stats.csv", index=False)

    if len(agreement_df) > 0:
        pairs = agreement_df.groupby(["frontend_a", "frontend_b"])["agreement_rate"].mean()
        for (a, b), rate in pairs.items():
            print(f"Mean frame agreement {a} vs {b}: {rate:.3f}")

    save_json(
        {**asdict(cfg), "core_dir": args.core_dir, "prototype_dir": args.prototype_dir,
         "frontends": frontend_names, "n_sessions": len(wav_files)},
        str(out_dir / "config.json"),
    )
    print(f"\nProxy analysis done → {out_dir}")
    print(f"  per_session_scores.csv:       {len(sessions_df)} rows")
    print(f"  inter_frontend_agreement.csv: {len(agreement_df)} rows")
    print(f"  detection_rate_stats.csv:     {len(stats_rows)} rows")


if __name__ == "__main__":
    main()
