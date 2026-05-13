# Schema: `evaluation/group_stratified_kfold_summary.csv`

Reports group-stratified k-fold performance (children disjoint per fold) for the top-band systems. Sibling to `evaluation/balanced_metrics_summary.csv`.

## Columns

| Column | Type | Description |
|---|---|---|
| `system_name` | str | System slug. |
| `k` | int | Number of folds (default 5; 3 if positive-rate guard violated). |
| `seed` | int | Random seed used by `StratifiedGroupKFold` (default 42). |
| `n_children_total` | int | Total number of children in the cross-validated population. |
| `n_clips_total` | int | Total number of clips across all folds. |
| `pos_rate_overall` | float | Fraction of positives across all folds. |
| `pos_rate_min_fold` | float | Minimum per-fold positive rate (stratification audit). |
| `pos_rate_max_fold` | float | Maximum per-fold positive rate. |
| `auroc_mean` | float | Mean AUROC across folds. |
| `auroc_std` | float | Std of AUROC across folds. |
| `auroc_per_fold` | str | JSON list of per-fold AUROC values, e.g., `[0.881, 0.862, 0.902, 0.870, 0.879]`. |
| `balanced_accuracy_mean` | float | Mean balanced accuracy across folds. |
| `balanced_accuracy_std` | float | Std of balanced accuracy across folds. |
| `balanced_accuracy_per_fold` | str | JSON list. |
| `f1_weighted_mean` | float | Mean of class-weighted F1 across folds. |
| `f1_weighted_per_fold` | str | JSON list. |
| `compared_to_within_child_3fold` | str | Optional. AUROC delta vs the legacy within-child k-fold for the same system, formatted `+0.043` or `-0.010` or `n/a`. |
| `result_dirs` | str | JSON list of per-fold result-dir absolute paths. |
| `regenerated_at` | str | ISO8601. |

## Validation

- `pos_rate_max_fold - pos_rate_min_fold ≤ 0.10` (stratification audit; reject and rerun with k=3 if violated).
- `len(auroc_per_fold) == k`.
- `n_children_total == sum of per-fold test_children` (every child held out exactly once).

## Example row

```csv
system_name,k,seed,n_children_total,n_clips_total,pos_rate_overall,pos_rate_min_fold,pos_rate_max_fold,auroc_mean,auroc_std,auroc_per_fold,balanced_accuracy_mean,balanced_accuracy_std,balanced_accuracy_per_fold,f1_weighted_mean,f1_weighted_per_fold,compared_to_within_child_3fold,result_dirs,regenerated_at
whisper_pseudo_frame,5,42,109,2183,0.762,0.731,0.793,0.879,0.014,"[0.881,0.862,0.902,0.870,0.879]",0.728,0.018,"[0.730,0.715,0.748,0.722,0.725]",0.852,"[0.851,0.842,0.870,0.846,0.852]",-0.005,"[/orcd/.../whisper_pseudo_frame_groupstrat5_f0/, ...]",2026-05-15T15:00:00Z
```
