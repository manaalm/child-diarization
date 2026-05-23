"""Generate Whisper-MIL backbone-size variants.

Whisper-tiny (39M), Whisper-base (74M), Whisper-medium (769M).
Whisper-small (244M, baseline) already exists as whisper_mil.yaml.
"""

import os
import yaml

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
BASE = os.path.join(REPO, "mil/configs/whisper_mil.yaml")
OUT_DIR = os.path.join(REPO, "mil/configs/backbone_size")
os.makedirs(OUT_DIR, exist_ok=True)


VARIANTS = {
    "whisper_tiny":   "openai/whisper-tiny",
    "whisper_base":   "openai/whisper-base",
    "whisper_medium": "openai/whisper-medium",
}


def main():
    with open(BASE) as f:
        base = yaml.safe_load(f)
    made = []
    for tag, hf_id in VARIANTS.items():
        cfg = dict(base)
        cfg["variant_name"] = f"{tag}_mil"
        cfg["backbone"] = hf_id
        if tag == "whisper_medium":
            cfg["batch_size"] = 4  # larger backbone needs smaller batch
        out = os.path.join(OUT_DIR, f"{tag}_mil.yaml")
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        made.append(out)
    listfile = os.path.join(OUT_DIR, "configs.txt")
    with open(listfile, "w") as f:
        for c in made:
            f.write(c + "\n")
    print(f"Wrote {len(made)} configs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
