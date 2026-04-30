# Feature Specification: Synthetic Data Augmentation Extensions (C1–C6)

**Feature Branch**: `016-synth-augmentation-extensions` (work performed on `013-missing-baselines`)
**Created**: 2026-04-29
**Status**: Documentation of completed exploration

## Overview

Six follow-on experiments that route the existing 5000-scene synthetic-audio generator (`synth/`, spec-009) into pipelines where the synth labels are not just permitted but *load-bearing* — frame-level supervised classifiers, controlled few-shot demos, hard-negative pools, and clean segment proposals. Each experiment is independently testable; together they map where synthetic mixing actually buys signal vs. where the prototype-based or LLM-based scoring is invariant to it.

The motivating prior is that the spec-009 ratio sweep produced bit-identical metrics across 0×–10× synth ratios on the enrollment classifier — synth has no effect on prototype-based scoring by design. The TinyVox MIL augmentation produced a clean negative (−0.10 AUROC, padding-shortcut artifact). This spec covers the remaining surface area where augmentation might still help.

---

## User Scenarios & Testing

### User Story C1 — USC-SAIL Whisper LoRA on synth scenes (Priority: P1)

A researcher trains the USC-SAIL Whisper-base + LoRA frame classifier from scratch on 5000 synthetic scenes (90/10 train/val) using ground-truth RTTMs converted to per-frame label CSVs. Original training data (5k anfengxu simulated conversations) does not exist on this cluster, so synth is the only viable training corpus.

**Why this priority**: Frame-level supervision is what synth labels are made for; this is the single experiment most likely to materially improve the USC-SAIL diarizer used by every enrollment frontend. Cleanest test of "do synth scenes produce a working frame classifier?"

**Independent test**: Train USC-SAIL on synth from scratch, evaluate the trained checkpoint via the existing enrollment pipeline (`pyannote/unified.py --diarizer usc_sail`).

**Acceptance**: Training completes; loss decreases over 20 epochs; resulting checkpoint produces a non-trivial frame-level prediction (>50% accuracy on val).

---

### User Story C2 — Pseudo-frame classifier with synth ground-truth labels (Priority: P1)

A researcher extends the pseudo-frame WavLM classifier (`pseudo_frame/`, currently F1=0.869, AUROC=0.831) by appending 5000 synth scenes with *exact* per-frame target-child labels to the training pool, replacing the noisy mean-of-VTC+USC-SAIL pseudo-labels for that subset. Synth labels carry confidence=1.0; real-clip labels keep their soft pseudo-target.

**Why this priority**: Pseudo-frame already wins on frame localization (Pearson 0.566 vs MIL 0.084). Clean labels on 5k extra clips should compound that advantage.

**Independent test**: Augmented index.csv (7183 rows = 2183 real + 5000 synth) feeds the existing pseudo_train.py loop with `split_dir=whisper-modeling/seen_child_splits_synth_aug/`. Evaluate on real seen-child test.

**Acceptance**: Test F1 / AUROC / AUPRC reported alongside baseline; per-clip Pearson on held-out real test pseudo-labels also reported.

---

### User Story C3 — Hard-negative MIL pool from synth (Priority: P2)

A researcher reuses the spec-009 hard-negative MIL pipeline but swaps the Playlogue/Providence-mined hard negatives (612 windows) for synth-derived negatives — 1276 `adult_only_negative` + 728 `background_speech_negative` scenes (n=2004). Same training recipe, same cap (extra_negatives_cap=623 → 1:1 pos:neg).

**Why this priority**: Tests whether synth can replace mined hard negatives, which are expensive (require RTTM mining) and bounded (612 max). Synth pool is 3.3× larger and pre-labeled.

**Independent test**: `extra_negatives_csv: synth_results/manifests/synthetic_hardneg.csv`; same MIL pipeline as `wavlm_mil_hardneg`/`whisper_mil_hardneg`. Compare against real-hardneg variant on test.

**Acceptance**: Test metrics for both wavlm and whisper backbones; delta vs real-hardneg variant documented.

---

### User Story C4 — Cross-child MIL with synth augmentation (Priority: P2)

A researcher augments the cross-child MIL train split (97 unseen-children, 1469 clips) with 5000 synth scenes (which already exclude test speakers per `--exclude-speakers-csv`). Cross-child val/test stay real-only. Most data-hungry split benefits most from extra labeled audio.

**Why this priority**: Cross-child (97 train children) is half the size of seen-child (109 train children) and has the worst headroom. Synth was generated specifically with test-speaker exclusion, so this is leakage-safe.

**Independent test**: New split dir `baselines/splits_synth_aug/` (val.csv + test.csv unchanged from `baselines/splits/`; train.csv = real 1469 + synth 5000 = 6469 rows). Train wavlm + whisper MIL.

**Acceptance**: Test metrics reported; delta vs real-only cross-child MIL (CLAUDE.md: AUROC ~0.5–0.7 range) documented.

---

### User Story C5 — Seg-MIL with combined real + synth RTTM cache (Priority: P2)

A researcher builds a unified RTTM cache (`mil/seg_mil_combined_cache/` with 2112 USC-SAIL RTTMs for real clips + 5000 synth GT RTTMs renamed to the `{stem}__{md5(audio_path)}.rttm` convention) and trains seg-MIL on the synth-augmented seen-child split using two aggregators (`gated_attention`, `transformer`).

**Why this priority**: Segment-MIL underperforms frame-window MIL — hypothesis is that diarizer-segment noise is the bottleneck. Synth scenes provide *clean* (GT) segments mixed with noisy real ones, isolating the noise contribution.

**Independent test**: New seg_mil_synth.yaml; `split_dir=whisper-modeling/seen_child_splits_synth_aug/`; eval on real test.

**Acceptance**: Test metrics for both aggregators; comparison vs `seg_mil/usc_sail_*` baseline.

---

### User Story C6 — Audio LLM 2-shot with synthetic demos (Priority: P3)

A researcher replaces the per-query same-child in-context demos (existing `qwen2_audio_7b_2shot` variant) with universal synthetic demos — one positive synth scene + one adult-only-negative synth scene used identically for every test query. Tests whether controlled synthetic demos generalize better than variable real-child clips.

**Why this priority**: Audio LLM is one frontend among many; few-shot demos are known to be a low-leverage axis on Qwen2-Audio. Quick to run; documents the negative result rather than leaving it implicit.

**Independent test**: New `--universal-shots` flag on `audio_llm_baseline.py` bypasses per-query child filtering. Run val + test with `--train-csv synth_results/manifests/synthetic_audio_llm_shots.csv`.

**Acceptance**: Val + test metrics under model_slug `qwen2_audio_7b_synth_2shot`; delta vs zero-shot and real-2shot variants documented.

---

## Scope and Non-Goals

**In scope**: training-recipe variants that re-route synth labels into existing classifiers. All experiments reuse seed=42 and existing test splits; no new audio generation.

**Out of scope**: regenerating synthetic scenes; training on TinyVox (already shown NEGATIVE in spec-009); fine-tuning third-party diarizers (Pyannote, BabAR, VTC, VBx, EEND-EDA, Sortformer — not retrainable here); video-side augmentation (no synthetic video generator).

**Constraints**: every result directory writes `config.json` per the project constitution; every job uses seed=42; val-only threshold tuning preserved.

---

## Outcome Reporting

Each user story records its result in `mil/spec014_jobs.json` style — variant_name, job_id, last_state, test_f1/auroc/auprc, optional note. CLAUDE.md results table extended with new rows under spec-016.
