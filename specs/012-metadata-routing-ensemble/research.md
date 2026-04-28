# Research: Metadata-Conditioned Routing and Ensemble Extensions

## System Predictions Inventory

All system predictions are on the seen-child test split (441 clips). Score column name varies:

| System | Path | Score col | Notes |
|---|---|---|---|
| BabAR | `babar_ecapa_enrollment_runs/enroll_test_predictions.csv` | `prob` | |
| VTC | `vtc_ecapa_enrollment_runs/enroll_test_predictions.csv` | `prob` | |
| VTC-KCHI | `vtc_kchi_ecapa_enrollment_runs/enroll_test_predictions.csv` | `prob` | |
| VBx | `vbx_ecapa_enrollment_runs/enroll_test_predictions.csv` | `prob` | |
| USC-SAIL | `whisper-modeling/usc_sail_enrollment_runs/enroll_test_predictions.csv` | `prob` | |
| Pyannote | `pyannote/pyannote_enrollment_runs/test_predictions.csv` | `prob` | different folder/naming |
| EEND-EDA | `eend_eda_ecapa_enrollment_runs/enroll_test_predictions.csv` | `prob` | |
| Sortformer | `sortformer_ecapa_enrollment_runs/enroll_test_predictions.csv` | `prob` | |
| WavLM-MIL | `mil/mil_results/wavlm_mil/test_predictions.csv` | `score` | sigmoid score, join on audio_path |
| Whisper-MIL | `mil/mil_results/whisper_mil/test_predictions.csv` | `score` | sigmoid score |
| Audio-LLM | `baselines/audio_llm_baseline_runs/qwen2_audio_7b/test_predictions.csv` | `score` | check col name |

**Note**: MIL predictions use `score` not `prob`; treat as equivalent (already sigmoid-transformed). All 10/11 systems present for 441 clips. Pyannote path differs: use `test_predictions.csv` at `pyannote/pyannote_enrollment_runs/`.

## Metadata Column Mapping

Metadata lives in `whisper-modeling/seen_child_splits/master_with_split.csv`. Join key: `audio_path`.

| Analysis name | Actual column | Type | Notes |
|---|---|---|---|
| n_adults | `#_adults` | string ("0","1","2+"...) | Cast to int; group ≥2 |
| n_children | `#_children` | string ("1","2",...) | Cast to int; group ≥2 |
| task / context | `Context` | 9-category string | "general social...", "unknown", etc. |
| interaction | `Interaction_with_child` | yes/no string | Binary encode |
| location | `Location` | 6-category string | inside private, inside public, etc. |
| age band | `timepoint_norm` | 14_month / 36_month | Already in prediction CSVs |
| face visibility | `Video_Quality_Child_Face_Visibility` | 1–10 float | optional for stacker |
| gestures | `Gestures` | yes/no | optional |

## Stratified F1 Findings (from `evaluation/stratified_analysis/`)

### n_adults strata
| n_adults | BabAR | WavLM-MIL | Whisper-MIL | Sortformer | EEND-EDA |
|---|---|---|---|---|---|
| 0 | 0.886 | 0.887 | 0.895 | 0.857 | 0.849 |
| 1 | 0.855 | 0.877 | 0.873 | 0.811 | 0.842 |
| 2+ | **0.545** | 0.778 | **0.750** | 0.750 | **0.778** |

**Router rule A1**: n_adults ≥ 2 → use EEND-EDA or WavLM-MIL (tie; use mean of both)

### n_children strata
| n_children | BabAR | VTC | Whisper-MIL | Pyannote |
|---|---|---|---|---|
| 1 | 0.887 | **0.905** | 0.896 | 0.862 |
| 2+ | 0.825 | 0.822 | **0.847** | **0.838** |

**Router rule A2**: n_children = 1 → use VTC (best); n_children ≥ 2 → use Whisper-MIL

### Context/task strata (key finding)
| Context | BabAR | Sortformer | Pyannote | Whisper-MIL | WavLM-MIL |
|---|---|---|---|---|---|
| unknown | **0.000** | **0.947** | 0.842 | 0.824 | 0.842 |
| socialroutine | 0.952 | 0.850 | 0.955 | 0.930 | 0.930 |
| dailyroutine | 0.981 | 0.963 | 0.926 | 0.962 | 0.982 |
| generalcommunication | 0.912 | 0.891 | 0.915 | 0.924 | 0.929 |

**Router rule A3**: Context = "unknown" → use Sortformer (F1=0.947 vs BabAR 0.000)

## Rule-Based Router Decision Tree (Sub-feature A)

Priority order (first matching rule wins):
1. `Context contains "unknown"` → Sortformer score
2. `#_adults ≥ 2` → mean(WavLM-MIL, EEND-EDA)
3. `#_children = 1` → VTC score
4. `#_children ≥ 2` → Whisper-MIL score
5. default → best_audio_mil mean (BabAR + VTC + WavLM-MIL-gated + VBx-max)

## Sub-feature B: Stacker Feature Set

**System score features** (10 columns, one per system): `babar_prob`, `vtc_prob`, `vtc_kchi_prob`, `vbx_prob`, `usc_sail_prob`, `pyannote_prob`, `eend_eda_prob`, `sortformer_prob`, `wavlm_mil_score`, `whisper_mil_score`
- Note: audio_llm has only 425/441 clips (cached subset); use 0.5 imputation for missing

**Metadata features** (7 columns):
- `n_adults_int` (int), `n_children_int` (int), `n_adults_ge2` (binary), `n_children_ge2` (binary)
- `context_unknown` (binary), `has_interaction` (binary), `timepoint_is_36m` (binary)
- Optional: `face_visibility` (float 1–10)

**Models**: LR (C=1.0, max_iter=500) and HistGradientBoostingClassifier (max_iter=200, learning_rate=0.1). Train on val, evaluate on test. Tune threshold on val.

## Sub-feature C: Multi-Child Suppressor

**Training set**: Train-split clips where `#_children ≥ 2` (from `whisper-modeling/seen_child_splits/train.csv`). Label 0 = target child silent (FP from main pipeline), label 1 = target child vocalizing.

**Features**: Reuse WavLM-MIL embeddings from `mil/mil_results/wavlm_mil/` embedding cache (if available) OR re-embed on-the-fly with frozen WavLM backbone + mean pooling over windows. Single 256-d → linear head.

**Application**: At inference on test, apply suppressor only to clips where `#_children ≥ 2`. Merge: `final_score = alpha * main_score + (1-alpha) * suppressor_score`, where alpha tuned on val.

**Stratum sizes** (estimated from seen-child split): ~30% of clips have n_children ≥ 2 (~132 test clips).

## Sub-feature D: Short-Vocalization Head

**Short-voc clip identification**: From ground-truth RTTMs (e.g., BabAR/VTC RTTM cache or USC-SAIL cache), identify clips where at least one CHI segment is <0.5s. These are the hard false-negative cases.

**Architecture**: Frozen WavLM-Base+ backbone + 1D-CNN or attentive pooling over 500ms/250ms-hop windows instead of the 2s/1s baseline. Single binary output.

**Training set**: Positive clips that have ≥1 short CHI segment (from train split RTTMs); negative clips from train split hard negatives (from `synth_results/manifests/hard_negatives_manifest.csv`).

**Merge**: `final_score = beta * main_score + (1-beta) * short_head_score`, beta tuned on val. Apply to all clips (not just short-voc identified ones, since at test time we don't have ground-truth RTTM).

## Decision: Implementation Approach

- **A/B**: Single script `evaluation/metadata_router.py`; runs CPU-only in ~1 min
- **C**: `evaluation/multi_child_suppressor.py`; short GPU SLURM job (~30 min)  
- **D**: `evaluation/short_voc_head.py`; GPU SLURM job (~1–2h)
- All output to separate result directories, no overwriting of existing results
- Constitution II: all threshold tuning gated behind `split == "val"` assertion
