# Schema: `whisper-modeling/all_children_splits/test_all.csv`

Universal-coverage zero-shot evaluation split (US3 FR-014). No `split` column because the file is used as test only; no `train`/`val`/`test` partitioning.

## Columns

| Column | Type | Description |
|---|---|---|
| `child_id` | str | SAILS child identifier. |
| `clip_id` | str | Clip identifier. |
| `audio_path` | str | Absolute path to the 16kHz mono WAV. |
| `timepoint_norm` | str | BIDS-derived timepoint (`14_month`, `36_month`, or `unknown`). |
| `label` | int | 0 or 1 — child vocalisation present. |
| `n_clips_for_this_child` | int | Total clips this child has in the universal-coverage split. |
| `excluded_from_seen_child_split` | bool | True if this row was excluded from `whisper-modeling/seen_child_splits/master_with_split.csv` (e.g., because the child has < 5 clips at one timepoint). |
| `exclusion_reason` | str | One of `min-clips-per-child`, `timepoint-missing`, `none`. |

## Validation

- Every row MUST have `label ∈ {0, 1}` (drop NaN; matches existing seen-child convention).
- Every row MUST have `audio_path` resolvable on disk.
- No `train`/`val`/`test` partitioning — the file is consumed by zero-shot baselines only (Constitution II: no model selection or threshold tuning may use this file's labels for training).
- Threshold tuning for zero-shot baselines on the universal-coverage split MUST reuse the threshold tuned on the seen-child VAL split — the universal-coverage split has no val partition and threshold tuning on it would constitute test leakage.

## Generation

`whisper-modeling/make_seen_child_split.py --build-all-children-split` (new flag) produces `test_all.csv` alongside the seen-child split CSVs. The all-children split is derived from `build_master_dataframe(cfg)` with `cfg.require_timepoint=False` and `cfg.min_clips_per_child=1` overrides.

## Relationship to seen-child split

| Property | seen-child | all-children-coverage |
|---|---|---|
| Use case | Train + val + test for all systems | Zero-shot eval only |
| Min clips per (child, timepoint) | 5 | 1 |
| Timepoint required | Yes | No |
| Children count | ~109 | Higher (~120-150 expected) |
| Clip count | ~2183 | ~3000-4000 expected |
| Threshold tuning | On seen-child val | Reuse seen-child val threshold |
| Train allowed? | Yes | No |
