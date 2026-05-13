# Schema: `evaluation/balanced_metrics_summary.csv`

Single canonical reporting artefact for US2. Replaces ad-hoc CLAUDE.md headline-table copy-paste.

## Columns

| Column | Type | Description |
|---|---|---|
| `system_name` | str | System slug, e.g., `whisper_mil`, `whisper_pseudo_frame`, `babar_combined`, `qwen35_omni_7b`, `yamnet`, `ast`. |
| `split` | enum | One of `seen_child_test`, `all_children_coverage`. |
| `n_clips` | int | Number of clips in the evaluation. |
| `pos_rate` | float | Fraction of clips with `label == 1`. |
| `threshold_source` | enum | `val` (default per Constitution IV) or `test` (rare; flagged in plan). |
| `tuned_threshold` | float | The threshold value applied to compute F1/balanced-accuracy. |
| `f1` | float | Binary F1 (positive class). Preserved from existing `compute_metrics()`. |
| `f1_macro` | float | Macro-averaged F1 (unweighted per-class mean). |
| `f1_weighted` | float | Support-weighted F1 (imbalance-aware). |
| `balanced_accuracy` | float | Mean of per-class recall. |
| `precision` | float | Positive-class precision. |
| `recall` | float | Positive-class recall. |
| `auroc` | float | ROC AUC. NaN if a single class is present. |
| `auprc` | float | Average precision. NaN if a single class is present. |
| `trivial_f1` | float | F1 of constant `predict-all-1` predictor. |
| `trivial_f1_macro` | float | Macro F1 of constant predictor. |
| `trivial_balanced_accuracy` | float | Balanced accuracy of constant predictor (always 0.5 for binary). |
| `predictions_path` | str | Absolute path to the source `*_predictions.csv`. |
| `metrics_json_path` | str | Absolute path to the system's existing `*_metrics_tuned.json` (for cross-check). |
| `regenerated_at` | str | ISO8601 timestamp when the row was produced. |

## Validation

- Every numeric column is in `[0, 1]` (or NaN for auroc/auprc edge case).
- `f1` MUST match the value in `metrics_json_path` within 1e-6 (regression guard).
- `trivial_balanced_accuracy` MUST equal 0.5 for binary (sanity guard).
- `regenerated_at` MUST be later than the source predictions CSV's mtime.

## Example row

```csv
system_name,split,n_clips,pos_rate,threshold_source,tuned_threshold,f1,f1_macro,f1_weighted,balanced_accuracy,precision,recall,auroc,auprc,trivial_f1,trivial_f1_macro,trivial_balanced_accuracy,predictions_path,metrics_json_path,regenerated_at
whisper_pseudo_frame,seen_child_test,872,0.760,val,0.40,0.880,0.770,0.851,0.730,0.890,0.870,0.881,0.928,0.864,0.464,0.500,/orcd/.../pseudo_frame/results/whisper_pseudo_frame/test_predictions.csv,/orcd/.../pseudo_frame/results/whisper_pseudo_frame/test_metrics_tuned.json,2026-05-15T10:30:22Z
```
