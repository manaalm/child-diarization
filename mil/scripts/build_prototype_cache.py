"""Build per-(child_id, timepoint_norm) ECAPA prototype cache for TS-MIL training.

Reuses the duration-weighted prototype construction from
`pyannote/unified.py:build_child_prototypes` (line ~559) but writes the prototype
dict to disk as a `.npz` so the MIL training loop can read it without re-running
ECAPA inference per run.

Usage (from repo root):
    python mil/scripts/build_prototype_cache.py \\
        --frontend  babar_vtc \\
        --train-csv whisper-modeling/seen_child_splits/train.csv \\
        --output    mil/prototypes/babar_vtc.npz

The output `.npz` has one key per (child, timepoint) pair, formatted as
`{child_id}__{timepoint_norm}`, mapping to a 192-d float32 L2-normalized
ECAPA embedding (identical to the in-memory prototypes used by enrollment).

A companion `_stats.csv` file is also written, mirroring `child_prototype_stats.csv`
(child_id, timepoint_norm, n_segments, status).
"""

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "pyannote"))

# pyannote/unified.py contains the canonical prototype construction
from pyannote.unified import (  # type: ignore  # noqa: E402
    BaseConfig,
    ECAPAEmbedder,
    build_child_prototypes,
    build_frontend,
)


# Note: `babar_vtc` is the seg-MIL alias for VTC standalone (KCHI+OCH); the
# diarizer-name passed to build_frontend is `vtc`. Same RTTM cache.
_FRONTEND_TO_DIARIZER_NAME = {
    "usc_sail": "usc_sail",
    "pyannote": "pyannote",
    "babar_vtc": "vtc",
    "babar": "babar",
    "vbx": "vbx",
    "vtc": "vtc",
    "vtc_kchi": "vtc_kchi",
}

_FRONTEND_RTTM_DIRS = {
    "usc_sail": "whisper-modeling/usc_sail_rttm_cache",
    "pyannote": "pyannote/pyannote_rttm_cache",
    "babar_vtc": "pyannote/vtc_rttm_cache",
    "babar": "pyannote/babar_rttm_cache",
    "vbx": "pyannote/vbx_rttm_cache",
    "vtc": "pyannote/vtc_rttm_cache",
    "vtc_kchi": "pyannote/vtc_rttm_cache",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TS-MIL prototype cache")
    parser.add_argument("--frontend", default="babar_vtc",
                        choices=sorted(_FRONTEND_RTTM_DIRS.keys()),
                        help="Diarizer frontend whose RTTMs are used to extract speaker segments")
    parser.add_argument("--train-csv", default="whisper-modeling/seen_child_splits/train.csv",
                        help="Path to training split CSV (one row per labelled clip)")
    parser.add_argument("--output", required=True,
                        help="Output .npz path (e.g., mil/prototypes/babar_vtc.npz)")
    parser.add_argument("--ecapa-source", default="speechbrain/spkrec-ecapa-voxceleb",
                        help="HF/SpeechBrain source for the ECAPA embedder")
    parser.add_argument("--device", default="cuda",
                        help="Device for ECAPA inference (cuda or cpu)")
    parser.add_argument("--max-segments-per-child", type=int, default=200,
                        help="Cap segments aggregated into one prototype")
    parser.add_argument("--skip-uncached", action="store_true",
                        help="Drop training rows whose RTTM is not already in the cache "
                             "(avoids subprocess fallback for envs that cannot run inference)")
    args = parser.parse_args()

    train_csv = args.train_csv if os.path.isabs(args.train_csv) else os.path.join(_REPO, args.train_csv)
    if not os.path.isfile(train_csv):
        print(f"ERROR: --train-csv not found: {train_csv}", file=sys.stderr)
        sys.exit(2)

    train_df = pd.read_csv(train_csv)
    if "audio_exists" in train_df.columns:
        train_df = train_df[train_df["audio_exists"] == True]
    if "timepoint_norm" not in train_df.columns and "timepoint" in train_df.columns:
        train_df = train_df.rename(columns={"timepoint": "timepoint_norm"})

    cfg = BaseConfig()
    cfg.ecapa_source = args.ecapa_source
    cfg.device = args.device
    cfg.max_enrollment_segments_per_child = args.max_segments_per_child
    cfg.diarizer = args.frontend
    cfg.results_dir = os.path.dirname(os.path.abspath(args.output))
    rttm_cache_dir = _FRONTEND_RTTM_DIRS[args.frontend]
    cfg.rttm_cache_dir = os.path.join(_REPO, rttm_cache_dir)

    print(f"Frontend: {args.frontend}  |  RTTM cache: {rttm_cache_dir}", flush=True)
    print(f"Train clips: {len(train_df)}  |  Positive clips: {(train_df['label']==1).sum()}", flush=True)

    if args.skip_uncached:
        # Filter to only files whose RTTM is already cached. The cache filename
        # convention is `{stem}__{md5(audio_path)}.rttm` (audio_to_cache_id).
        import hashlib
        from pathlib import Path
        cache_dir = cfg.rttm_cache_dir
        def _cached(audio_path: str) -> bool:
            cid = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
            stem = Path(audio_path).stem
            return os.path.isfile(os.path.join(cache_dir, f"{stem}__{cid}.rttm"))
        before = len(train_df)
        train_df = train_df[train_df["audio_path"].apply(_cached)].reset_index(drop=True)
        skipped = before - len(train_df)
        print(f"  --skip-uncached: dropped {skipped} of {before} rows missing from RTTM cache "
              f"({len(train_df)} retained, {(train_df['label']==1).sum()} positives)", flush=True)
        if len(train_df) == 0:
            print("ERROR: no rows remain after --skip-uncached filter", file=sys.stderr)
            sys.exit(2)

    diarizer_name = _FRONTEND_TO_DIARIZER_NAME[args.frontend]
    print(f"Loading frontend ({diarizer_name}) ...", flush=True)
    frontend = build_frontend(diarizer_name, cfg)
    print("Loading ECAPA embedder ...", flush=True)
    embedder = ECAPAEmbedder(cfg.ecapa_source, cfg.device)

    print("Building prototypes ...", flush=True)
    prototypes, stats_df = build_child_prototypes(train_df, frontend, embedder, cfg)
    print(f"Built {len(prototypes)} prototypes.", flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {
        key: vec.astype(np.float32) for key, vec in prototypes.items()
    }
    np.savez(args.output, **arrays)
    stats_csv = args.output.replace(".npz", "_stats.csv")
    stats_df.to_csv(stats_csv, index=False)

    print(f"  Saved prototypes → {args.output}", flush=True)
    print(f"  Saved stats      → {stats_csv}", flush=True)

    if len(prototypes) > 0:
        any_vec = next(iter(prototypes.values()))
        print(f"  Embedding dim: {any_vec.shape[0]}  dtype: {any_vec.dtype}", flush=True)


if __name__ == "__main__":
    main()
