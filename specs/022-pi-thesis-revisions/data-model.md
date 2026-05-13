# Phase 1 Data Model — Spec 022 PI Thesis Revisions

Formalises the entities introduced or modified by this spec. Each entity lists fields, source-of-truth, validation rules, and any state transitions.

---

## 1. BIDS session

- **Source of truth**: `/orcd/scratch/bcs/001/sensein/sails/BIDS_data/final_bids-dataset/sub-<ID>/ses-<NN>/` and `participants.tsv`.
- **Fields**:
  - `sub_id` (string, e.g., `A1H3H9Y3T1`) — primary child identifier in BIDS.
  - `ses_id` (string, one of `ses-01`, `ses-02`, or rare non-standard) — session label.
  - `age_months` (int, derived from `participants.tsv`) — used to validate the `ses → timepoint` mapping.
- **Validation rules**:
  - Every `(sub_id, ses_id)` directory MUST be parseable from filesystem walk.
  - `participants.tsv` MUST contain an `age` (or equivalent) column; if absent, the implementation falls back to `ses_id` parsing and logs the gap.
- **Transitions**: none (read-only source).

## 2. Timepoint mapping

- **Source of truth**: BIDS session ID via `whisper-modeling/bids_timepoint.py:bids_session_to_timepoint(audio_path)`.
- **Fields** (per (child_id, clip_id) row):
  - `bids_timepoint` (enum {`14_month`, `36_month`, `unknown`}) — derived from BIDS.
  - `spreadsheet_timepoint` (enum {`14_month`, `36_month`, `unknown`}) — read from `anotated_processed.csv` `timepoint` column.
  - `agree` (bool) — `bids_timepoint == spreadsheet_timepoint` and neither is `unknown`.
  - `rationale_if_disagree` (string, nullable) — short note (`non-standard-session-id`, `spreadsheet-missing`, `spreadsheet-stale`).
- **Validation rules**:
  - `bids_timepoint` MUST be deterministic given `audio_path`; reordering rows MUST yield identical output.
  - For every row, at least one of `bids_timepoint` or `spreadsheet_timepoint` MUST be non-`unknown` — if both are unknown, the row is dropped from the seen-child split (matches existing `require_timepoint=True` behaviour).
- **Transitions**:
  - `pre-correction` → `post-correction`: `timepoint_norm` column in splits CSVs is overwritten with `bids_timepoint` value; pre-correction state preserved in `bids_correction_provenance.json`.

## 3. Split (existing + new variants)

- **Source of truth**: `whisper-modeling/seen_child_splits/{master_with_split,train,val,test}.csv` (existing); `whisper-modeling/all_children_splits/test_all.csv` (new, US3).
- **Fields** (per row):
  - `child_id` (string) — child identifier.
  - `clip_id` (string) — clip identifier.
  - `audio_path` (string, absolute) — path to the 16kHz mono WAV.
  - `timepoint_norm` (enum) — BIDS-derived (post-US1).
  - `label` (int, 0 or 1) — vocalisation present.
  - `split` (enum {`train`, `val`, `test`}) — only in seen-child splits; absent in all-children-coverage split (zero-shot eval only).
- **Validation rules**:
  - Seen-child split: ≥5 clips per (`child_id`, `timepoint_norm`) group; seed=42 stratification preserved across regeneration.
  - All-children-coverage split: no minimum-clip filter; includes every clip with `label != NaN` and `audio_path` resolvable.
  - Split membership MUST be mutually exclusive within seen-child split (no clip in two splits).
  - The all-children-coverage split MUST NOT be used for training (zero-shot eval only); enforced by absence of `split` column.
- **Transitions**:
  - `pre-US1` → `post-US1`: `timepoint_norm` regenerated with BIDS-derived values; row count may shift by ±N for children whose BIDS-derived timepoint differs from spreadsheet (some rows now meet the ≥5/timepoint guard, others fall out).

## 4. K-fold scheme

- **Existing**: within-child 3-fold via `mil/scripts/build_kfold_splits.py` (or equivalent); same children appear in train and test of different folds.
- **New (US2)**: group-stratified 5-fold via `sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; children disjoint per fold; positive-rate stratified.
- **Fields** (per fold):
  - `fold_id` (int, 0..k-1).
  - `train_children` (list[str]).
  - `test_children` (list[str]).
  - `train_label_dist` (dict {`0`: count, `1`: count}).
  - `test_label_dist` (dict {`0`: count, `1`: count}).
- **Validation rules**:
  - `train_children ∩ test_children == ∅` for every fold (group-disjoint guarantee).
  - `|test_positive_rate - global_positive_rate| ≤ 0.05` for every fold (stratification target; bootstrap-noise tolerance).
  - Sum of test sets across folds == full set (every child held out exactly once).
- **Transitions**:
  - Per-system retraining produces `mil/mil_results/<system>_groupstrat5_f{0..4}/` artefacts; the within-child `*_kfold3_f{0,1,2}/` dirs are preserved as legacy.

## 5. Metric set (existing + extended)

- **Existing fields** (in `mil/mil_utils.py:compute_metrics()` return dict): `{f1, precision, recall, auroc, auprc}`.
- **New fields** (US2 FR-007): `{f1_macro, f1_weighted, balanced_accuracy}`.
- **Validation rules**:
  - `f1` (existing) is `f1_score(y_true, y_pred, average='binary')` — equivalent to positive-class F1. Preserved unchanged.
  - `f1_macro` = `f1_score(..., average='macro')` (unweighted per-class mean).
  - `f1_weighted` = `f1_score(..., average='weighted')` (support-weighted per-class mean) — the imbalance-aware F1 the PI requested.
  - `balanced_accuracy` = `balanced_accuracy_score(y_true, y_pred)` = mean of recall on each class.
  - For trivial-floor reporting, the same metric set is computed for the constant predictor `y_pred = argmax(class_priors)`; emitted alongside the row.
- **Transitions**:
  - Every existing `test_metrics_tuned.json` regains a `_v2` sibling with the extended metric set, generated by `evaluation/balanced_metrics.py`. The original JSONs are not modified (file-deletion discipline). Headline reporting moves to `evaluation/balanced_metrics_summary.csv` (one row per system, extended metric set).

## 6. Baseline system (extended)

- **Existing entries** (per `CLAUDE.md` headline table): ~30 systems each with `(system_id, predictions_csv, metrics_json, source_dir)`.
- **New entries** (US3): `qwen35_omni_7b`, `yamnet`, `ast`. Each has:
  - `system_id` (string): the registry key.
  - `result_dir` (path): canonical results directory.
  - `splits_evaluated` (list[enum {`seen_child`, `all_children_coverage`}]): which splits the system was run on.
  - `predictions_path` (per split): CSV with columns `{clip_id, child_id, label, score, prediction, timepoint_norm}` plus auxiliary columns per system.
  - `metrics_path` (per split): JSON with extended metric set + threshold metadata.
- **Validation rules**:
  - Every new entry MUST have predictions on the seen-child split (for cross-system comparison).
  - Zero-shot entries (Qwen 3.5-Omni, YAMNet, AST) MUST additionally have predictions on the all-children-coverage split.
  - Each entry's `config.json` MUST capture every parameter that affects scoring (prompt template, AudioSet class mapping, threshold-tuning split).

## 7. Encoder pipeline

- **Components**: input waveform (16kHz mono) → frozen encoder (Whisper-base/small/medium/large-v3 or WavLM-Base+) → pooling (mean or attention) → linear classifier (single FC layer).
- **Fused variant**: parallel encoder streams (Whisper × WavLM) → concat embeddings → pooling → linear classifier.
- **Validation rules**:
  - The pipeline figure (US4 FR-018) MUST show every component with shapes labelled (e.g., `(T, 768)` for WavLM-base hidden state).
  - The fusion-of-encoders prose elaboration (US4 FR-019) MUST answer: which encoders, fusion stage, weighting, head parameterisation.
- **Transitions**:
  - `baselines/baseline_encoders.py` → `encoders/baseline_encoders.py` via `git mv`; shim left at old path for one release cycle.

## 8. Per-model training-data registry

- **Source**: `docs/per_model_training_data.csv`, regenerated by `docs/per_model_training_data.py` which introspects every `config.json` under canonical result roots (`mil/mil_results/`, `pseudo_frame/results/`, `baselines/audio_llm_baseline_runs/`, `baselines/scene_analysis_runs/`, `whisper-modeling/usc_sail_enrollment_runs/`, etc.).
- **Fields**:
  - `system_name` (string).
  - `train_split` (enum {`seen_child`, `cross_child`, `synth_*`, `zero_shot`, `frozen`}).
  - `train_children` (int — count, not list).
  - `train_clip_count` (int).
  - `includes_synthetic` (bool).
  - `synth_corpus_version` (string, nullable, one of {`v1`, `v2`, `v3_perturb`, `v4`, `v4_hardneg`, etc.}).
  - `eval_split` (enum {`seen_child`, `cross_child`, `all_children_coverage`, `synth_holdout`}).
- **Validation rules**:
  - Every row's `train_children + train_clip_count + includes_synthetic` MUST be derivable from a committed `config.json`.
  - Zero-shot rows have `train_clip_count = 0` and `train_split = "zero_shot"` (or `"frozen"` for frozen-backbone baselines).
- **Transitions**: regenerated on demand; latest version always reflects the live result dirs.

## 9. Posthoc analysis section

- **Source**: thesis chapter LaTeX (not in repo); maintained in the megadoc and propagated to the chapter.
- **Fields**:
  - `system_name` (string).
  - `combined_auroc` (float).
  - `14m_auroc` (float).
  - `36m_auroc` (float).
  - `delta_36m_minus_14m` (float).
  - `flagged_large_delta` (bool, `|delta| > 0.05`).
- **Validation rules**:
  - Every system in the headline table MUST appear in the posthoc table.
  - `flagged_large_delta` rows MUST have a short prose interpretation.
- **Transitions**: regenerated whenever per-system per-timepoint metric files change (i.e., after US1 correction lands).

---

## Entity relationships

```text
BIDS session ──┬── feeds ──→ Timepoint mapping ──┬── overrides ──→ Split.timepoint_norm
               └── via filesystem walk           └── records diff in bids_vs_spreadsheet_diff.csv

Split ──┬── seen_child (train/val/test) ──→ K-fold scheme (within-child 3-fold legacy + group-stratified 5-fold new)
        └── all_children_coverage (test only) ──→ zero-shot Baseline system evaluation

Baseline system ──┬── existing (~30) ──→ predictions_csv ──→ Metric set (extended) ──→ balanced_metrics_summary.csv
                  └── new (qwen35_omni_7b, yamnet, ast) ──→ same pipeline

Encoder pipeline ──→ Figure (US4) + fusion prose (US4) + per-model training-data CSV (US4)

Per-timepoint metrics ──→ Posthoc analysis section (US5)
```

## Cross-cutting invariants

- No row in any split CSV may have a NaN `label` (existing constraint; preserved).
- Every metric value in `balanced_metrics_summary.csv` is reproducible by piping the corresponding `test_predictions.csv` through `evaluation/balanced_metrics.py`.
- Every `config.json` next to a `test_metrics_tuned.json` exactly describes the run that produced those metrics (Constitution VI).
