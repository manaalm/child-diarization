# AST baseline (spec 022 US3 / FR-013)

**Backbone**: [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) — Audio Spectrogram Transformer, fine-tuned on AudioSet 527-class multi-label classification, 86M params.

**Runtime**: HuggingFace `transformers` in the project's `child-vocalizations` conda env. In-process (no subprocess bridge). Driver: `baselines/scene_analysis_baseline.py --model ast`.

## Class-to-score mapping (FR-016)

Same as YAMNet — the AudioSet ontology and aggregation rule are model-agnostic:

```
p_child_voc = max(P[Child speech], P[Babbling], P[Baby cry], P[Children shouting])
```

| Display name | Ontology ID |
|---|---|
| Child speech, kid speaking | `/m/02zsn` |
| Babbling | `/m/0463cq4` |
| Baby cry, infant cry | `/t/dd00002` |
| Children shouting | `/m/02p0sh1` |

**Per-class probability** is `sigmoid(logits)` since AudioSet is multi-label. The AST model card recommends this aggregation directly.

## Differences vs YAMNet baseline

| | YAMNet | AST |
|---|---|---|
| Architecture | MobileNetV1 (CNN) | Transformer (12 layers) |
| Params | ~3.7M | ~86M |
| Sample rate | 16 kHz | 16 kHz (resampled by feature extractor) |
| Per-clip aggregation | Mean over ~975 ms frames | Single forward pass over whole clip |
| Env | TF 2.16 (sibling env) | PyTorch (project env) |
| Strength | Lightweight, fast | Higher AudioSet mAP (~0.485 vs YAMNet 0.314) |
| Weakness | Simpler classifier may miss nuance | Heavier; more memory |

## Threshold tuning

Tuned on seen-child val per Constitution IV. `test` and `test_all` reuse the val-tuned threshold.

## Caveats / methodological flags

Same as YAMNet (AudioSet's child-vocalisation classes are under-represented and age-agnostic).

Two additional flags specific to AST:

1. AST's feature extractor pads or truncates input to a fixed length (~10.24 s). SAILS clips are ~2-30 s; long clips are truncated. Posthoc analysis on truncation impact deferred to a follow-up; clip durations in `whisper-modeling/seen_child_splits/master_with_split.csv:Vid_duration`.
2. AST uses ImageNet-style normalisation on log-mel-spectrograms. This is baked into the HF feature extractor and is not a tunable parameter here.

## Reproduction

```bash
python baselines/scene_analysis_baseline.py --model ast --split val
python baselines/scene_analysis_baseline.py --model ast --split test
python baselines/scene_analysis_baseline.py --model ast --split test_all
```
