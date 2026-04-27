# Contract: Training Manifest CSV

**Files**: `synth_results/manifests/train_{ratio}x_manifest.csv`
**Produced by**: `synth/scripts/generate_training_sets.py`
**Consumed by**: `synth/scripts/train_with_synthetic.py`, `synth/scripts/evaluate_synthetic_augmentation.py`

---

## Schema

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `audio_path` | string | yes | absolute path to WAV file (real or synthetic) |
| `rttm_path` | string | no | RTTM file path; null for real clips that lack per-clip RTTMs |
| `label` | int | yes | 0 or 1; 1 if target child vocalized |
| `child_id` | string | yes | real child ID or `synthetic_{scene_id}` |
| `timepoint_norm` | string | yes | `14_month` or `36_month` (matches existing seen-child split convention) |
| `split` | string | yes | always `train` in training manifests |
| `is_synthetic` | bool | yes | `true` for synthetic rows, `false` for real rows |
| `source_config` | string | no | config name for synthetic rows; empty for real rows |
| `age_band` | string | yes | `14_18_months` or `34_38_months` |

## Integrity Constraints

- `split` = `train` for every row (training manifests never include val or test rows).
- `is_synthetic = true` rows have `child_id` starting with `synthetic_`.
- The set of real rows is identical across all ratio manifests (same clips, same order).
- Ratio `n`x means `round(n * len(real_train_rows))` synthetic rows appended.
- For `0x` ratio: synthetic rows are absent; manifest contains only real training clips.
- No test-set `child_id` values appear in any row.

## Ratios Generated

| Filename | Synthetic rows | Real rows |
|----------|---------------|-----------|
| `train_0x_manifest.csv` | 0 | ~1311 |
| `train_0.5x_manifest.csv` | ~655 | ~1311 |
| `train_1x_manifest.csv` | ~1311 | ~1311 |
| `train_2x_manifest.csv` | ~2622 | ~1311 |
| `train_5x_manifest.csv` | ~6555 | ~1311 |
| `train_10x_manifest.csv` | ~13110 | ~1311 |

## Integration with Existing Pipeline

These manifests are designed to replace `whisper-modeling/seen_child_splits/train.csv` as input to enrollment and classifier training scripts. The `child_id` and `timepoint_norm` columns match the existing CSV schema, enabling drop-in substitution.
