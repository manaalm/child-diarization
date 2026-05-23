"""Energy and carbon cost estimate for the thesis SLURM workload.

Sums documented training/inference GPU-hours from CLAUDE.md and the megadoc,
multiplies by A100 thermal design power (TDP) and the MIT/ISO-NE grid carbon
intensity for 2025-2026.

Outputs:
  evaluation/energy_cost.csv   (per-run breakdown)
  evaluation/energy_cost.md    (summary)

Numbers are estimates, not measurements. For a measured number, codecarbon
needs to be wired into the SLURM submission scripts (future work).

Sources:
  A100 SXM TDP = 400 W (NVIDIA datasheet)
  ISO-NE grid carbon intensity 2024-2025 ≈ 240 gCO2eq/kWh (ISO-NE 2024 report,
    rounded; New England's mix is ~50% gas, 30% nuclear, 12% renewables, 8% other).
  PUE: assume 1.5 for university DC (typical for non-hyperscale).
"""

from __future__ import annotations

import os
import pandas as pd

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
OUT_CSV = os.path.join(REPO, "evaluation", "energy_cost.csv")
OUT_MD = os.path.join(REPO, "evaluation", "energy_cost.md")

A100_TDP_W = 400.0
PUE = 1.5
GRID_GCO2_PER_KWH = 240.0  # ISO-NE 2024 average


# (run_name, gpu_hours, gpus_concurrent)
# Sourced from CLAUDE.md "Recent Changes" and §23.5 Active SLURM jobs in megadoc.
RUNS = [
    ("USC-SAIL Whisper LoRA pretrain (50k steps, anfengxu)", 60.0, 1),  # documented baseline
    ("USC-SAIL fine-tune on BIDS",                               6.0,  1),
    ("BabAR / VTC / VTC-KCHI inference (RTTM gen, 2.1k clips)", 2.0,  1),
    ("Pyannote inference (2.1k clips)",                          1.5,  1),
    ("VBx inference (2.1k clips)",                               2.0,  1),
    ("Sortformer inference (2.1k clips)",                        1.0,  1),
    ("EEND-EDA inference (2.1k clips)",                          1.0,  1),
    ("ECAPA enrollment runs × 8 diarizers",                      0.5,  1),  # cached embeddings
    ("Whisper-MIL frame-window training",                        3.0,  1),
    ("WavLM-MIL frame-window training",                          3.0,  1),
    ("Whisper-MIL TS-MIL concat (spec-014 US4)",                 3.0,  1),
    ("HuBERT-large MIL layersum (spec-014 US1)",                 6.0,  1),
    ("Whisper-MIL ACMIL variants (spec-014 US3 ext, 6 configs)", 18.0, 1),
    ("Child-adapted WavLM SSL pretrain (50k steps, spec-009 US3)",48.0,1),
    ("WavLM-MIL TinyVox aug (spec-009)",                         3.0,  1),
    ("Hard-negative MIL × 2 backbones (spec-013)",               6.0,  1),
    ("Segment-instance MIL sweep (28 configs, spec-004)",        24.0, 1),
    ("Pseudo-frame WavLM training (spec-013 US6)",               0.3,  1),
    ("Pseudo-frame synth-aug (spec-016 C2)",                     0.5,  1),
    ("Pseudo-frame C1-distill (spec-016 follow-up #8)",          0.5,  1),
    ("Audio LLM Qwen2-Audio-7B inference (v1, val + test, base)",   8.0,  1),
    ("Audio LLM Qwen2-Audio-7B inference (v1, 2-shot, val + test)", 8.0,  1),
    ("Audio LLM Qwen2-Audio-7B cross-child variant (v1)",           4.0,  1),
    ("Audio LLM Qwen2.5-Omni-7B inference (v2, val + test, base)",  8.0,  1),
    ("Synth scene generation v1 (5000 scenes, CPU)",             0.0,  0),  # CPU only
    ("Synth scene generation v2 (5000 scenes, CPU)",             0.0,  0),  # CPU only
    ("Synth augmentation training × 6 ratios × 2 backbones",     36.0, 1),
    ("Spec-014 MIL extensions full sweep (11 jobs)",             24.0, 1),
    ("Spec-016 candidate trainings (9 candidates × 2 versions)", 36.0, 1),
    ("AV fusion training (manual_only + agebandfix)",            0.5,  1),
    ("Voice transfer experiment (spec-016 follow-up #1)",        0.3,  1),
    ("K-fold seen-child enrollment (7 diarizers × 3 folds)",     1.5,  1),  # cached caches
    ("K-fold seen-child trained models (4 systems × 3 folds)",   36.0, 1),
    ("Spec-012 metadata stacker / router (CPU sklearn)",         0.0,  0),
    ("Spec-012 multi-child suppressor (US3)",                    0.5,  1),
    ("Spec-012 short-voc head (US4)",                            4.0,  1),
    ("Cross-child enrollment runs (5 of 8 completed)",           0.5,  1),
]


def main():
    rows = []
    total_gpu_h = 0.0
    for name, gpu_h, _ in RUNS:
        kwh_compute = gpu_h * A100_TDP_W / 1000.0
        kwh_total = kwh_compute * PUE
        co2_kg = kwh_total * GRID_GCO2_PER_KWH / 1000.0
        rows.append(dict(
            run=name, gpu_hours=gpu_h,
            kwh_compute=round(kwh_compute, 3),
            kwh_with_pue=round(kwh_total, 3),
            kg_co2eq=round(co2_kg, 3),
        ))
        total_gpu_h += gpu_h

    total_kwh = total_gpu_h * A100_TDP_W / 1000.0 * PUE
    total_co2 = total_kwh * GRID_GCO2_PER_KWH / 1000.0

    df = pd.DataFrame(rows).sort_values("kg_co2eq", ascending=False)
    df.to_csv(OUT_CSV, index=False)

    md = []
    md.append("# Energy and Carbon Cost Estimate\n")
    md.append("Estimates based on documented SLURM GPU-hours from CLAUDE.md and "
              "THESIS_MEGADOC.md §23.5. **These are estimates, not measurements** — "
              "for a measured number, `codecarbon` needs to be wired into SLURM "
              "submission scripts.\n")
    md.append("## Assumptions\n")
    md.append(f"- A100 SXM TDP: **{A100_TDP_W} W** (NVIDIA datasheet)")
    md.append(f"- Data-center PUE: **{PUE}** (typical university DC)")
    md.append(f"- ISO-NE grid carbon intensity: **{GRID_GCO2_PER_KWH} gCO2eq/kWh** "
              "(ISO-NE 2024 average; New England mix is ~50% gas / 30% nuclear / "
              "12% renewables / 8% other)\n")
    md.append("## Headline numbers\n")
    md.append(f"- **Total GPU-hours (sum of estimates)**: ~{total_gpu_h:.0f}")
    md.append(f"- **Total electricity (with PUE)**: ~{total_kwh:.1f} kWh")
    md.append(f"- **Total CO2-equivalent**: ~{total_co2:.1f} kg")
    md.append(f"- For reference: ~{total_co2/0.404:.0f} miles driven by an average US "
              "passenger vehicle (EPA 0.404 kgCO2/mile)\n")
    md.append("## Largest single contributors\n")
    md.append(df.head(10).to_markdown(index=False))
    md.append("\n## Caveats\n")
    md.append("- Numbers reflect **succesful runs only**; failed/restarted jobs add ~10-20%.")
    md.append("- Inference cost on shared dev nodes is not tracked.")
    md.append("- ECAPA enrollment runs use cached embeddings; the *first* enrollment "
              "computation for each diarizer dominated cost (already counted in the "
              "diarizer inference rows).")
    md.append("- Synth-scene generation (10,000 scenes total across v1+v2) ran on "
              "CPU nodes; not in this table.")
    md.append("- All numbers are point estimates; uncertainty is at least ±30%.")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"Total GPU-hours: ~{total_gpu_h:.0f}")
    print(f"Total kWh: ~{total_kwh:.1f}")
    print(f"Total kgCO2eq: ~{total_co2:.1f}")
    print(f"Wrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
