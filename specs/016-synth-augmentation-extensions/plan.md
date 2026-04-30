# Implementation Plan: Synthetic Data Augmentation Extensions (C1–C6)

**Branch**: `013-missing-baselines` (work performed here; spec dir is `016-synth-augmentation-extensions`) | **Date**: 2026-04-29 | **Spec**: `specs/016-synth-augmentation-extensions/spec.md`
**Input**: Post-hoc documentation of work performed in a single auto-mode session.

## Summary

Six independent training-recipe variants route the existing 5000-scene synth corpus (`synth_results/synthetic_scenes/{wav,rttm,json}/`, generated 2026-04-27 via job 12770080) into pipelines where the labels are load-bearing. All six were implemented and submitted in a single session; results are still in flight at submission time of this plan.

The shared design is: build a synth-derived CSV/cache in the format the target pipeline already accepts, then point an existing config at it — minimal new training code.

## Technical Context

**Language/Version**: Python 3.10 (`child-vocalizations` conda env)
**Primary Dependencies**: torch 2.8+cu128, transformers 4.57+ (WavLM/Whisper backbones), torchaudio, numpy, pandas, scikit-learn, pytorch-lightning (USC-SAIL only), `pyannote.unified_rttm` (synth → frame mask)
**Storage**: results under canonical per-pipeline folders; new RTTM/audio caches gitignored
**Testing**: smoke runs on `--limit 5` (synth pseudo-label builder); SLURM job IDs logged for reproducibility
**Target Platform**: SLURM cluster (`ou_bcs_normal,pi_satra`), 1 GPU per job
**Project Type**: ML experiment pipeline
**Performance Goals**: each candidate ≤6h GPU; C6 (Qwen2-Audio inference) ≤15min on H100
**Constraints**: seed=42 everywhere; val-only threshold tuning; cross-child split must exclude test-child speakers from synth (already enforced via spec-009 `build_segment_manifest.py --exclude-speakers-csv`); no test-set leakage
**Scale/Scope**: 5000 synth scenes (90/10 train/val for C1, full pool for C2/C4/C5, scene-type-filtered for C3/C6); seen-child eval n=441 test, cross-child eval n=496 test

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| I. Reproducibility | ✅ PASS | seed=42; SLURM job IDs in spec014_jobs.json; config.json written per result dir |
| II. Split discipline | ✅ PASS | Synth excludes test speakers; val/test CSVs untouched in `splits_synth_aug/` and `seen_child_splits_synth_aug/` |
| III. Baseline-first | ✅ PASS | Each candidate has a corresponding non-synth baseline already in CLAUDE.md results |
| IV. Metrics | ✅ PASS | F1, precision, recall, AUROC, AUPRC + per-timepoint required for each candidate |
| V. Ablations | ✅ PASS | C3 ablates synth-vs-real hardneg sources; C6 ablates synth-vs-real demos |
| VI. Thesis sync | ✅ PASS | Results paths follow project convention; spec014_jobs.json + CLAUDE.md updated when results land |
| VII. Documentation | ✅ PASS | This spec + tasks.md document the work |
| File deletion | ✅ PASS | No real files deleted; .specify/feature.json was rewritten (own creation); 012/plan.md template overwrite reverted via git checkout |

No violations.

## Project Structure

### Documentation (this feature)

```text
specs/016-synth-augmentation-extensions/
├── spec.md                # User stories C1–C6
├── plan.md                # This file
└── tasks.md               # Per-candidate task list with status, paths, SLURM IDs
```

### Source Code (created in this session)

```text
synth/scripts/
├── build_synth_aug_manifests.py        # All-in-one: builds C3/C4/C6/generic CSVs
├── build_cross_child_synth_split.py    # C4: baselines/splits_synth_aug/
├── build_pseudo_synth_split.py         # C2: whisper-modeling/seen_child_splits_synth_aug/
├── build_seg_mil_synth_cache.py        # C5: mil/seg_mil_combined_cache/
└── build_usc_sail_synth_data.py        # C1: synth_results/usc_sail_data/{audios,labels}/

pseudo_frame/
└── build_synth_pseudo_labels.py        # C2: appends 5000 synth GT rows to pseudo_labels/index.csv

mil/configs/
├── wavlm_mil_hardneg_synth.yaml        # C3
├── whisper_mil_hardneg_synth.yaml      # C3
├── wavlm_mil_cross_child_synth.yaml    # C4
├── whisper_mil_cross_child_synth.yaml  # C4
└── seg_mil_synth.yaml                  # C5

pseudo_frame/configs/
└── wavlm_pseudo_synth.yaml             # C2

whisper-modeling/configs/
└── config_synth.yaml                    # C1

mil/slurm/
└── seg_mil_synth.sh                     # C5 single-frontend wrapper

baselines/slurm/
└── run_audio_llm_synth_shots.sh         # C6

whisper-modeling/
└── run_train_synth.sh                   # C1

baselines/
└── audio_llm_baseline.py                # C6: added --universal-shots flag (1 patched function, 1 new arg)
```

### Generated artifacts

```text
synth_results/manifests/
├── synthetic_hardneg.csv                # 2004 rows (C3)
├── synthetic_cross_child_aug.csv        # 5000 rows (C4)
├── synthetic_audio_llm_shots.csv        # 2 rows: 1 pos + 1 neg (C6)
└── synthetic_train_aug.csv              # 5000 rows (C2/C5)

baselines/splits_synth_aug/{train,val,test}.csv         # C4: 6469 train (1469+5000)
whisper-modeling/seen_child_splits_synth_aug/*.csv      # C2/C5: 6311 train (1311+5000)
pseudo_frame/pseudo_labels/{md5}.npy + index.csv        # +5000 synth GT pseudo-frames (C2)
mil/seg_mil_combined_cache/                              # 7112 RTTMs (C5)
synth_results/usc_sail_data/{audios,labels}/{train,val}/ # 4500/500 split (C1)
```

## Phase 0: Research (consolidated findings)

Three parallel `Explore` sub-agents mapped the existing pipelines for C1, C2+C5, and C3+C4+C6. Findings:

- **C1 USC-SAIL**: original training data path (`/data/anfengxu/...`) does not exist on this cluster → only viable path is *training from scratch on synth*, not fine-tuning. RTTM→CSV converter required (5-ms grid for overlap detection: TARGET_CHILD only → `c`, ADULT_0 only → `a`, both → `o`, gaps auto-filled as `si`).
- **C2 pseudo-frame**: existing `build_pseudo_labels.py` averages noisy VTC + USC-SAIL masks. For synth, parse GT RTTM directly with TARGET_CHILD spans → exact 0/1 labels at 50 Hz. Sentinel `n_sources=2` so train loop never down-weights synth labels.
- **C3 hardneg MIL**: existing `wavlm_mil_hardneg.yaml` already supports `extra_negatives_csv` with full slice-loading via `start_sec`/`end_sec`. Drop-in replacement.
- **C4 cross-child**: cross-child split lives at `baselines/splits/`; cleanest path is to copy val/test unchanged and append synth rows to train.csv in a sibling `splits_synth_aug/` dir.
- **C5 seg-MIL**: per-frontend RTTM cache resolved by `{stem}__{md5(audio_path)}.rttm`. Combine USC-SAIL real RTTMs + synth GT RTTMs (renamed) into one cache; frontend lookup is uniform.
- **C6 audio-LLM**: existing few-shot path filters train_csv by query child_id. Add `--universal-shots` flag that bypasses filter so all queries see the same 1-pos + 1-neg synth demos.

## Phase 1: Design contracts

Each candidate is one config + one SLURM job; no shared design surface beyond:

1. **Manifest contract** (`synth_results/manifests/synthetic_*.csv`): all derived CSVs use the column schema the target pipeline already consumes (no new schemas introduced).
2. **Split contract** (`*_synth_aug/`): val.csv + test.csv are bit-identical copies of the real splits; only train.csv is augmented.
3. **Cache contract** (`mil/seg_mil_combined_cache/`): synth RTTMs are *renamed* to `{stem}__{md5}.rttm` so seg_dataset.py needs no code change.

Re-evaluating constitution post-design: still PASS. No principle violations introduced.

## Outputs (in flight at write time)

SLURM jobs submitted 2026-04-29 ~14:30–14:50 EDT:
- C1: 12845895 (USC-SAIL synth-only, ~3–6h)
- C2: 12845617 (pseudo-frame synth-aug, ~1–2h)
- C3: 12845253 (wavlm), 12845254 (whisper) (~30–45m)
- C4: 12845381 (wavlm), 12845382 (whisper) (~30–45m)
- C5: 12845699 (seg-MIL synth combined, ~30–60m)
- C6: 12845414 val, 12845610 test (DONE, F1=0.863, AUROC=0.713 — clean negative)

Final result tracking: `mil/spec014_jobs.json` style entries appended; CLAUDE.md results table updated when jobs land.
