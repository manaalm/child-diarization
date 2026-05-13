# YAMNet baseline (spec 022 US3 / FR-013)

**Backbone**: [YAMNet](https://tfhub.dev/google/yamnet/1) — Google's AudioSet-pretrained MobileNetV1 classifier, 521 classes.

**Runtime**: TensorFlow 2.16 + tensorflow-hub 0.16 in a sibling env at `yamnet-eval/.venv/` (separate from the project's PyTorch 2.8 env to avoid TF↔PyTorch ABI conflicts). Driven by `baselines/scene_analysis_baseline.py` via subprocess bridge to `encoders/yamnet_worker.py`.

## Class-to-score mapping (FR-016)

Per-clip child-vocalisation probability:

```
p_child_voc = max(P[Child speech], P[Babbling], P[Baby cry], P[Children shouting])
```

AudioSet ontology IDs (canonical reference: <https://research.google.com/audioset/ontology>):

| Display name | Ontology ID | Notes |
|---|---|---|
| Child speech, kid speaking | `/m/02zsn` | Primary child-vocalising class. |
| Babbling | `/m/0463cq4` | Infant/toddler pre-linguistic vocalisation. |
| Baby cry, infant cry | `/t/dd00002` | Distress vocalisation. |
| Children shouting | `/m/02p0sh1` | Loud non-speech child vocalisation. |

**Why `max` over `sum`**: AudioSet labels are multi-label per clip and the four target labels are semantically overlapping (a babbling clip can also score on Child-speech). `max` answers "is there any kind of child vocalisation here?" without inflating clips that happen to hit multiple labels.

**Why these four labels**: SAILS positives include any vocalisation by the target child — speech, babbling, crying, shouting, laughter. The four selected labels cover the first four cases; laughter is omitted because AudioSet's "Laughter" class is age-agnostic and would introduce false positives on adult laughter. "Children playing" is also omitted as it captures ambient sound rather than the target child's own vocalisation.

## Clip-level aggregation

YAMNet emits per-frame (~975 ms hop) scores; per-clip score = `tf.reduce_mean(scores, axis=0)` (uniform-weight frame average). This matches the YAMNet model card's recommended aggregation for clip-level classification.

## Threshold tuning

Threshold tuned on the seen-child val split per Constitution IV. Tuned threshold + val metrics in `val_metrics_tuned.json`. `test` and `test_all` evaluations reuse the val-tuned threshold — no test-set tuning.

## Caveats / methodological flags

1. YAMNet was trained on AudioSet, which has a known annotator skew toward adult-recorded YouTube content. Child-vocalisation classes are present but under-represented (~1% of clips). False-negative rate on toddler vocalisations is therefore expected to be high; this is a feature of the baseline, not a bug.
2. AudioSet's "Child speech" label is age-agnostic — it covers toddlers through teenagers. The SAILS target population is 14-month and 36-month visits; YAMNet's class is broader than the SAILS positive definition.
3. We do NOT fine-tune YAMNet — this is a strict zero-shot baseline. A tuned head on top of YAMNet embeddings is out of scope for spec 022 (would land as a separate baseline if pursued).

## Reproduction

```bash
# One-time env setup
uv venv yamnet-eval/.venv --python 3.10
source yamnet-eval/.venv/bin/activate
uv pip install tensorflow==2.16 tensorflow-hub==0.16 soundfile==0.12 scipy
deactivate

# Run val + test + test_all
python baselines/scene_analysis_baseline.py --model yamnet --split val
python baselines/scene_analysis_baseline.py --model yamnet --split test
python baselines/scene_analysis_baseline.py --model yamnet --split test_all
```
