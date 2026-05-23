"""Generate cross-child kfold MIL configs."""

import os
import yaml

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
BASE = os.path.join(REPO, "mil/configs/whisper_mil.yaml")
OUT_DIR = os.path.join(REPO, "mil/configs/cross_child_kfold")
os.makedirs(OUT_DIR, exist_ok=True)
K = 3


def main():
    with open(BASE) as f:
        base = yaml.safe_load(f)
    for fold in range(K):
        cfg = dict(base)
        cfg["variant_name"] = f"whisper_mil_cross_child_kfold{K}_f{fold}"
        cfg["split_dir"] = f"baselines/splits_kfold/fold_{fold}"
        out = os.path.join(OUT_DIR, f"whisper_mil_cross_child_f{fold}.yaml")
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
