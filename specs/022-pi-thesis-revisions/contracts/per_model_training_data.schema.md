# Schema: `docs/per_model_training_data.csv`

One row per evaluated system, documenting exactly what data each variant was trained on. Reproducible from `docs/per_model_training_data.py` (US4 FR-020).

## Columns

| Column | Type | Description |
|---|---|---|
| `system_name` | str | System slug (matches `evaluation/balanced_metrics_summary.csv` rows). |
| `system_family` | enum | One of `mil_frame_window`, `mil_segment_instance`, `pseudo_frame`, `usc_sail_whisper`, `pyannote_family`, `audio_llm`, `audio_scene_analysis`, `encoder_baseline`, `ensemble`, `metadata_stacker`. |
| `train_split` | enum | One of `seen_child_train`, `cross_child_train`, `synth_train_<version>`, `zero_shot`, `frozen_backbone_only`. |
| `train_children` | int | Number of unique children in train (0 for zero-shot). |
| `train_clip_count` | int | Number of training clips (0 for zero-shot). |
| `includes_synthetic` | bool | True if synthetic data was mixed into training. |
| `synth_corpus_version` | str | Nullable. One of `v1`, `v2`, `v3_perturb`, `v4`, `v4_hardneg`, `v4_cross_child`. |
| `synth_clip_count` | int | Number of synthetic clips contributed to training (0 if `includes_synthetic == false`). |
| `pretrained_backbone` | str | E.g., `openai/whisper-small`, `microsoft/wavlm-base-plus`, `MIT/ast-finetuned-audioset-10-10-0.4593`, `frozen-ecapa`. |
| `backbone_frozen` | bool | True if the backbone was frozen during training. |
| `eval_splits` | str | JSON list of splits the system was evaluated on, e.g., `["seen_child_test", "all_children_coverage"]`. |
| `config_path` | str | Absolute path to the `config.json` introspected to populate this row. |
| `result_dir` | str | Absolute path to the system's canonical result dir. |

## Validation

- Every row MUST have `config_path` resolvable on disk.
- Zero-shot rows MUST have `train_children == 0`, `train_clip_count == 0`, `train_split == "zero_shot"`, `backbone_frozen == true`.
- `includes_synthetic == true` MUST imply non-null `synth_corpus_version` and positive `synth_clip_count`.

## Example rows

```csv
system_name,system_family,train_split,train_children,train_clip_count,includes_synthetic,synth_corpus_version,synth_clip_count,pretrained_backbone,backbone_frozen,eval_splits,config_path,result_dir
whisper_mil,mil_frame_window,seen_child_train,65,1318,false,,0,openai/whisper-small,true,"[\"seen_child_test\"]",/orcd/.../mil/mil_results/whisper_mil/config.json,/orcd/.../mil/mil_results/whisper_mil/
whisper_mil_hardneg_synth_v4,mil_frame_window,seen_child_train,65,1318,true,v4_hardneg,2000,openai/whisper-small,true,"[\"seen_child_test\"]",/orcd/.../mil/mil_results/whisper_mil_hardneg_synth_v4/config.json,/orcd/.../mil/mil_results/whisper_mil_hardneg_synth_v4/
qwen35_omni_7b,audio_llm,zero_shot,0,0,false,,0,Qwen/Qwen3.5-Omni-7B,true,"[\"seen_child_test\",\"all_children_coverage\"]",/orcd/.../baselines/audio_llm_baseline_runs/qwen35_omni_7b/config.json,/orcd/.../baselines/audio_llm_baseline_runs/qwen35_omni_7b/
yamnet,audio_scene_analysis,zero_shot,0,0,false,,0,google/yamnet-tfhub,true,"[\"seen_child_test\",\"all_children_coverage\"]",/orcd/.../baselines/scene_analysis_runs/yamnet/config.json,/orcd/.../baselines/scene_analysis_runs/yamnet/
ast,audio_scene_analysis,zero_shot,0,0,false,,0,MIT/ast-finetuned-audioset-10-10-0.4593,true,"[\"seen_child_test\",\"all_children_coverage\"]",/orcd/.../baselines/scene_analysis_runs/ast/config.json,/orcd/.../baselines/scene_analysis_runs/ast/
```
