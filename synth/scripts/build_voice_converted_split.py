"""Build a training-append manifest from kNN-VC voice-converted synth scenes.

Source: synth_results/manifests/synthetic_voice_converted.csv (1090 rows, all
label=1, produced by spec-017 US2 / synth/scripts/voice_convert_knnvc.py).
Each row was converted to sound like a specific target child.

Two output modes:

  --paradigm seen_child  (default for Phase A):
      No filtering. seen-child splits already share all 109 children across
      train/val/test (60/20/20 of each child's clips), so VC scenes targeting
      any of those children are valid training augmentation.
      Output: synth_results/manifests/synthetic_voice_converted_train.csv

  --paradigm cross_child (for cross-child sweeps):
      Drop rows whose target child_id is in baselines/splits/{val,test}.csv —
      otherwise the model trains on VC scenes shaped toward held-out children.
      Output: synth_results/manifests/synthetic_voice_converted_cross_child.csv
"""
import argparse
import os
import sys

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VC_MANIFEST = os.path.join(_REPO, "synth_results/manifests/synthetic_voice_converted.csv")
SEEN_SPLIT_DIR = os.path.join(_REPO, "whisper-modeling/seen_child_splits")
CROSS_SPLIT_DIR = os.path.join(_REPO, "baselines/splits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paradigm", choices=["seen_child", "cross_child"], default="seen_child")
    args = parser.parse_args()

    vc = pd.read_csv(VC_MANIFEST)
    print(f"VC manifest: {len(vc)} rows, {vc['child_id'].nunique()} target children")

    if args.paradigm == "cross_child":
        val = pd.read_csv(os.path.join(CROSS_SPLIT_DIR, "val.csv"))
        test = pd.read_csv(os.path.join(CROSS_SPLIT_DIR, "test.csv"))
        held_out = set(val["child_id"]).union(set(test["child_id"]))
        print(f"Cross-child held-out children: {len(held_out)}")
        leaked = vc[vc["child_id"].isin(held_out)]
        if len(leaked) > 0:
            print(f"  Dropping {len(leaked)} rows ({leaked['child_id'].nunique()} children) — would shape training toward held-out voices")
        kept = vc[~vc["child_id"].isin(held_out)].copy()
        out_path = os.path.join(_REPO, "synth_results/manifests/synthetic_voice_converted_cross_child.csv")
    else:
        kept = vc.copy()
        out_path = os.path.join(_REPO, "synth_results/manifests/synthetic_voice_converted_train.csv")

    missing = [p for p in kept["audio_path"] if not os.path.isfile(p)]
    if missing:
        print(f"WARNING: {len(missing)} audio_path entries not on disk; dropping", file=sys.stderr)
        kept = kept[kept["audio_path"].apply(os.path.isfile)].copy()

    out = pd.DataFrame({
        "audio_path": kept["audio_path"].astype(str),
        "child_id": kept["child_id"].astype(str),
        "timepoint_norm": "14_month",
        "label": kept["label"].astype(int),
        "audio_exists": True,
        "split": "train",
    })

    assert (out["label"] == 1).all(), "VC manifest expected to be all positives"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(out)} rows (all label=1), {out['child_id'].nunique()} children")


if __name__ == "__main__":
    main()
