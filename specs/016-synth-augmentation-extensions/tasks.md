# Tasks: Synthetic Data Augmentation Extensions (C1–C6)

**Spec**: `specs/016-synth-augmentation-extensions/spec.md`
**Plan**: `specs/016-synth-augmentation-extensions/plan.md`
**Session**: single auto-mode session, 2026-04-29

This file records what was actually done, in execution order, with paths and SLURM job IDs for reproducibility. Status reflects state at the time of the SLURM submission round; final results land asynchronously.

---

## Phase 0 — Research (parallel exploration)

- [x] **T000** Map USC-SAIL training pipeline (Explore agent) — config keys, CSV schema, RTTM converter requirements, padding-shortcut risk.
- [x] **T001** Map pseudo-frame + seg-MIL pipelines (Explore agent) — input formats, child_id assumptions, minimum integration cost.
- [x] **T002** Map MIL hard-neg + cross-child + audio-LLM pipelines (Explore agent) — existing config keys, schema columns, landmines.

## Phase 1 — Synth-derived manifests (data-only, no code mods)

- [x] **T010** Build C3/C4/C6/generic-train manifests in one pass.
  - File: `synth/scripts/build_synth_aug_manifests.py`
  - Outputs: `synth_results/manifests/synthetic_{hardneg,cross_child_aug,audio_llm_shots,train_aug}.csv`
  - Synth scene distribution: 2509 positive, 1276 adult_only_negative, 728 background_speech_negative, 487 noise_only_negative.

- [x] **T011** Build C4 cross-child synth-augmented split dir.
  - File: `synth/scripts/build_cross_child_synth_split.py`
  - Output: `baselines/splits_synth_aug/{train,val,test}.csv` (train=6469 rows, val/test copied as-is)

- [x] **T012** Build C2/C5 seen-child synth-augmented split dir.
  - File: `synth/scripts/build_pseudo_synth_split.py`
  - Output: `whisper-modeling/seen_child_splits_synth_aug/{train,val,test}.csv` (train=6311)

- [x] **T013** Build C5 combined RTTM cache (real USC-SAIL + synth GT).
  - File: `synth/scripts/build_seg_mil_synth_cache.py`
  - Output: `mil/seg_mil_combined_cache/` — 2112 USC-SAIL symlinks + 5000 synth GT symlinks = 7112 entries.

- [x] **T014** Build C1 USC-SAIL training data layout (RTTM→CSV converter + 90/10 split).
  - File: `synth/scripts/build_usc_sail_synth_data.py`
  - Output: `synth_results/usc_sail_data/{audios,labels}/{train,val}/` (4500/500 split)
  - Speaker mapping with overlap detection at 5 ms grid: TARGET_CHILD only → `c`, ADULT_0 only → `a`, both → `o`, gaps → `si`.

## Phase 2 — Synth pseudo-frame label generation

- [x] **T020** Smoke-test synth pseudo-label builder on 5 scenes.
  - File: `pseudo_frame/build_synth_pseudo_labels.py`
  - Verified `pyannote.unified_rttm.parse_rttm` + `segments_to_frame_mask` work on synth RTTM format (TARGET_CHILD speaker label).

- [x] **T021** Run full synth pseudo-label build (5000 scenes).
  - Output: 5000 new `pseudo_frame/pseudo_labels/{md5}.npy` files.
  - Index: `pseudo_frame/pseudo_labels/index.csv` grew from 2183 → 7183 rows. Synth rows tagged `sources="synth_gt"`, `n_sources=2` (sentinel).

## Phase 3 — Configs (one per candidate)

- [x] **T030** C3 configs: `mil/configs/{wavlm,whisper}_mil_hardneg_synth.yaml`
  - Diff vs `wavlm_mil_hardneg.yaml`: `extra_negatives_csv: synth_results/manifests/synthetic_hardneg.csv` (was Playlogue+Providence-mined). `extra_negatives_cap=623` matched to real-hardneg variant for direct comparison.

- [x] **T031** C4 configs: `mil/configs/{wavlm,whisper}_mil_cross_child_synth.yaml`
  - Diff vs `*_cross_child.yaml`: `split_dir: baselines/splits_synth_aug` (was `baselines/splits`).

- [x] **T032** C5 config: `mil/configs/seg_mil_synth.yaml`
  - Single-frontend (`usc_sail_synth_combined` → `mil/seg_mil_combined_cache`) × two aggregators (`gated_attention`, `transformer`); reuses encoder + HPs from `seg_mil_sweep.yaml`.

- [x] **T033** C2 config: `pseudo_frame/configs/wavlm_pseudo_synth.yaml`
  - Diff vs `wavlm_pseudo.yaml`: `split_dir: whisper-modeling/seen_child_splits_synth_aug` (was `seen_child_splits`).

- [x] **T034** C1 config: `whisper-modeling/configs/config_synth.yaml`
  - New config (separate `name: whisper_base_synth` so checkpoints don't collide with the existing pretrained 50k checkpoint). Audio + annotation paths point at `synth_results/usc_sail_data/`. `batch_size=32` (vs 64) for memory headroom; `max_epochs=20`.

## Phase 4 — SLURM wrappers (where existing scripts didn't fit)

- [x] **T040** C5 wrapper: `mil/slurm/seg_mil_synth.sh` — single-frontend variant of `seg_mil_sweep.sh` (drops the 4-frontend preflight checks).

- [x] **T041** C6 wrapper: `baselines/slurm/run_audio_llm_synth_shots.sh` — passes `--universal-shots --train-csv synth_results/manifests/synthetic_audio_llm_shots.csv --n-shot 2`; sets `model_slug=qwen2_audio_7b_synth_2shot` for cache isolation.

- [x] **T042** C1 wrapper: `whisper-modeling/run_train_synth.sh` — calls `scripts/main.py --debug f --config configs/config_synth.yaml`; logs to `logs/whisper_modeling/`.

## Phase 5 — Code patches

- [x] **T050** C6 patch on `baselines/audio_llm_baseline.py`:
  - Added `--universal-shots` CLI flag.
  - Patched `_find_few_shot_examples()` to bypass per-query child filter when universal=True.
  - Total diff: 1 new arg, 1 new branch in selection function (~15 lines).

## Phase 6 — Submit jobs

- [x] **T060** C3 submit: `sbatch mil/slurm/train_eval_spec014.sh mil/configs/wavlm_mil_hardneg_synth.yaml` → job 12845253; whisper variant → 12845254.

- [x] **T061** C4 submit: same script with cross_child_synth configs → 12845381 (wavlm), 12845382 (whisper).

- [x] **T062** C6 submit val: `sbatch baselines/slurm/run_audio_llm_synth_shots.sh val` → 12845414 (COMPLETED 1:15).

- [x] **T063** C2 submit: `sbatch pseudo_frame/slurm/train_pseudo.sh pseudo_frame/configs/wavlm_pseudo_synth.yaml` → 12845617.

- [x] **T064** C5 submit: `sbatch mil/slurm/seg_mil_synth.sh` → 12845699.

- [x] **T065** C6 submit test (after val finished): `sbatch baselines/slurm/run_audio_llm_synth_shots.sh test` → 12845610 (COMPLETED 1:18).

- [x] **T066** C1 submit: `sbatch whisper-modeling/run_train_synth.sh` → 12845895 FAILED (PYTHONPATH); resubmit 12847606 FAILED (transformers mel-3000 enforcement); resubmit 12848196 with `window_size: 30` + `PYTHONPATH=.` running.

## Phase 7 — Result aggregation (in progress)

- [x] **T070** C6 results recorded:
  - Test: F1=0.8633, AUROC=0.7127, AUPRC=0.8611, threshold=0.75
  - Val: F1=0.8607, AUROC=0.7581, AUPRC=0.8693
  - Compare to zero-shot (CLAUDE.md): F1=0.871, AUROC=0.725, AUPRC=0.853
  - Compare to real-2shot (`qwen2_audio_7b_2shot/test_metrics_tuned.json`): F1=0.871, AUROC=0.725, AUPRC=0.853
  - **Conclusion**: synth demos roughly neutral / slightly worse than zero-shot. Few-shot mechanism is a low-leverage axis on Qwen2-Audio for this task regardless of demo source.

- [x] **T071a** C2 pseudo-frame results recorded (job 12845617, 14:20 elapsed):
  - Test: F1=0.876, AUROC=0.763, AUPRC=0.910, threshold=0.45
  - Frame localization: Pearson_mean=0.468, Spearman_mean=0.454, frame-AUROC=0.812, n_pos_clips=335
  - Baseline (CLAUDE.md, real-only training): F1=0.869, AUROC=0.831, AUPRC=0.937; frame-Pearson 0.566
  - Delta: F1 +0.007, AUROC −0.068, AUPRC −0.027, frame-Pearson −0.098 → **NEGATIVE**
  - Verified files: `pseudo_frame/results/wavlm_pseudo_frame_synth/{test_metrics_tuned,frame_localization,config}.json`, `test_predictions.csv`, `frame_localization_per_clip.csv`, `best_checkpoint.pt`

- [x] **T071b** C3 wavlm hardneg synth results recorded (job 12845253, 20:44 elapsed):
  - Test: F1=0.8634, AUROC=0.6568, AUPRC=0.8509, threshold=0.05
  - Real-hardneg baseline (`mil/mil_results/wavlm_mil_hardneg/test_metrics_tuned.json`): F1=0.8634, AUROC=0.6421, AUPRC=0.8436
  - Delta: F1 0.000, AUROC +0.015, AUPRC +0.007 → **tiny POSITIVE**
  - Whisper side (12845254) still RUNNING — full C3 acceptance pending whisper finish.

- [x] **T071c** C5 seg-MIL synth combined cache results recorded (job 12845699, 20:45 elapsed):
  - gated_attention: test F1=0.8648, AUROC=0.6361, AUPRC=0.8336
  - transformer:    test F1=0.8714, AUROC=0.6373, AUPRC=0.8294
  - Real seg-MIL/usc_sail baseline: gated_attention AUROC=0.601, transformer AUROC=0.518
  - Delta gated_attention: +0.035 AUROC; Delta transformer: **+0.119 AUROC** → **POSITIVE (transformer especially strong)**
  - Verified files: `mil/mil_results/seg_mil_synth/usc_sail_synth_combined_{gated_attention,transformer}/`, `all_configs.json`

- [x] **T071d-partial** C4 wavlm cross-child synth recorded (job 12845381, 35:17 elapsed):
  - Test: F1=0.8637, AUROC=0.6199, AUPRC=0.8353, threshold=0.3
  - Real cross-child wavlm baseline: F1=0.8627, AUROC=0.6902, AUPRC=0.8445
  - Delta: F1 +0.001, AUROC **−0.070**, AUPRC −0.010 → **NEGATIVE**
  - Note: same train-pool swamping as C2 (5000 synth ≫ 1469 real cross-child train); cross-child was the expected best case for synth but loss mirrors C2.

- [x] **T071d-c1** C1 USC-SAIL synth-only training COMPLETED on 5th attempt (12849231, 24:51):
  - Best checkpoint: `whisper-modeling/checkpoints/whisper_base_synth/epoch=17-val_loss=0.235.ckpt`
  - Test frame-level accuracy: **0.9223** (test_loss=0.235); spec acceptance ">50% val accuracy" met
  - 4 sequential transformers-API-drift fixes were needed:
    1. attempt 1 (12845895): `ModuleNotFoundError: lightning_modules` → fixed via `PYTHONPATH=.`
    2. attempt 2 (12847606): `ValueError: mel input length 3000` → fixed via `window_size: 10→30, batch_size: 32→16`
    3. attempt 3 (12848196): `WhisperAttention(config=None)` → fixed by passing `config=config` to `WhisperAttention.__init__` (whisper-modeling/models/whisper.py:35)
    4. attempt 4 (12848844): tuple unpack `result[0],result[1]` instead of 3-tuple (whisper-modeling/models/whisper.py:73)
  - Frame-level metric not directly comparable to the enrollment AUROC table; to integrate into the diarizer comparison would need to wire this checkpoint into `pyannote/unified.py --diarizer usc_sail` and rerun enrollment vs the existing pretrained `whisper-base_rank8_pretrained_50k.pt`. Out of C1 scope but documented as the obvious follow-on.

- [x] **T071e-c3b** C3 whisper hardneg synth recorded (job 12845254, 56:25 elapsed):
  - Test: F1=0.8769, AUROC=0.8224, AUPRC=0.9314, threshold=0.5
  - Real-hardneg baseline (`mil/mil_results/whisper_mil_hardneg/`): F1=0.8688, AUROC=0.8182, AUPRC=0.9292
  - Delta: F1 +0.008, AUROC +0.004, AUPRC +0.002 → **tiny POSITIVE**
  - Mirrors C3 wavlm; both backbones confirm synth-mined hardnegs are a viable drop-in replacement for RTTM-mined ones.

- [x] **T071e-c4b** C4 whisper cross-child synth COMPLETED (job 12845382, 02:07:23 elapsed):
  - Test: F1=0.8594, AUROC=0.5893, AUPRC=0.7804, threshold=0.35
  - Real cross-child whisper baseline: F1=0.8951, AUROC=0.8761, AUPRC=0.9538
  - Delta: F1 −0.036, AUROC **−0.287**, AUPRC −0.174 → **STRONG NEGATIVE**
  - Note: Much bigger drop than C4 wavlm (−0.070 AUROC). Whisper-small more sensitive to synth swamping in cross-child. Both backbones confirm: 5000 synth ≫ 1469 real-train → model overfits to synth distribution; real cross-child baseline outperforms despite being smaller.

- [x] **T072** Final aggregation complete:
  - All 10 spec-016 result rows recorded in `mil/spec016_jobs.json` (10 jobs tracked, 0 pending)
  - CLAUDE.md results table updated with all 6 candidates (C1 frame-level row + C2/C3/C4/C5/C6 clip-level rows)
  - CLAUDE.md "Recent Changes" + "Important Gotchas" sections updated for spec-016 + the 4 USC-SAIL fixes
  - results_summary.md granite/canary/cohere section updated to reflect 2026-04-29 findings
  - 4 derived synth-aug split dirs + 1 RTTM cache + 5000 new pseudo-label .npy files committed
  - All builder scripts under `synth/scripts/build_*` + `pseudo_frame/build_synth_pseudo_labels.py`
  - 7 new YAML configs across `mil/configs/`, `pseudo_frame/configs/`, `whisper-modeling/configs/`
  - 3 new SLURM wrappers + 1 audio_llm CLI flag (`--universal-shots`)

- [X] **T072** Final aggregation: spec-016 final summary written to CLAUDE.md results table + Recent Changes (10 rows), `mil/spec016_jobs.json` (10/10 completed with verdicts/notes), and `results_summary.md` §16 (full pattern analysis + C1 patch story).

## Side work in same session (not part of C1–C6)

- [x] **S00** No-retrain ACMIL branch_selection eval for both wavlm + whisper ACMIL checkpoints. Login-node CPU run killed at 19 min; resubmitted as GPU SLURM jobs 12844150 (wavlm, COMPLETED) and 12844151 (whisper, COMPLETED). Outputs at `mil/mil_results/{wavlm,whisper}_mil_acmil/branch_selection.{csv,json}`. Wavlm best single branch (branch_3): test F1=0.876, AUROC=0.742 (+0.016 over mean baseline).
- [x] **S01** Helper script `mil/slurm/run_acmil_branch_selection.sh` for future runs.
- [x] **S02** spec014_jobs.json updated with 6 ACMIL retrain entries (max/gated/topk × wavlm/whisper) including test metrics + notes.
- [x] **S03** CLAUDE.md updated: 4 new ACMIL retrain rows in the results table (whisper_mil_acmil_max marked as best new variant: F1=0.891, AUROC=0.842, AUPRC=0.936); Recent Changes entry for branch-aggregation extension; Active Technologies entry for new result dirs.
- [x] **S04** Restored `specs/012-metadata-routing-ensemble/plan.md` from git after the speckit setup-plan script overwrote it with the empty template (template was triggered by stale `.specify/feature.json` pointer; subsequently repointed to this spec dir).
