"""Generate Whisper-MIL HP-sensitivity configs and a SLURM array driver.

Sweep:
  lr   ∈ {1e-3 (baseline), 3e-4, 1e-4}
  seed ∈ {42 (baseline), 1, 2}

9 configs total (1 baseline + 8 new). The baseline `whisper_mil.yaml` is
unchanged; new configs are written to `mil/configs/hp_sensitivity/`.
"""

import os
import yaml

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
BASE_CFG = os.path.join(REPO, "mil/configs/whisper_mil.yaml")
OUT_DIR = os.path.join(REPO, "mil/configs/hp_sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

LRS = [1e-3, 3e-4, 1e-4]
SEEDS = [42, 1, 2]


def main():
    with open(BASE_CFG) as f:
        base = yaml.safe_load(f)

    configs_made = []
    for lr in LRS:
        for seed in SEEDS:
            if lr == 1e-3 and seed == 42:
                continue  # baseline already exists
            cfg = dict(base)
            cfg["lr"] = lr
            cfg["seed"] = seed
            cfg["variant_name"] = f"whisper_mil_lr{lr:.0e}_seed{seed}"
            cfg["split_dir"] = base["split_dir"]
            out = os.path.join(OUT_DIR, f"{cfg['variant_name']}.yaml")
            with open(out, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            configs_made.append(out)

    listfile = os.path.join(OUT_DIR, "configs.txt")
    with open(listfile, "w") as f:
        for c in configs_made:
            f.write(c + "\n")

    print(f"Wrote {len(configs_made)} configs to {OUT_DIR}/")
    print(f"List: {listfile}")


if __name__ == "__main__":
    main()
