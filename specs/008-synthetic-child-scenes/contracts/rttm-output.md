# Contract: RTTM Output Format

**Files**: `synth_results/synthetic_scenes/rttm/{scene_id}.rttm`
**Produced by**: `synth/labels.py` → `synth/scripts/generate_scenes.py`
**Consumed by**: `synth/scripts/evaluate_synthetic_augmentation.py`, `pyannote/unified_rttm.py` (with label mapping)

---

## Format

Standard RTTM format (NIST), one segment per line:

```
SPEAKER <file_id> 1 <start_sec> <dur_sec> <NA> <NA> <label> <NA> <NA>
```

## Speaker Labels

| Label | Meaning |
|-------|---------|
| `TARGET_CHILD` | The target child whose vocalization is the binary classification target |
| `ADULT_0` | Primary adult speaker (caregiver) |
| `ADULT_1` | Secondary adult speaker (if present) |
| `OTHER_CHILD_0` | Non-target child (sibling or peer) |
| `BACKGROUND_SPEECH` | Far-field or TV/radio speech treated as non-target speaker |

**Note**: Non-speech noise is NOT assigned an RTTM label. Only speech-like sources get RTTM entries.

## Overlap Representation

Overlapping speech is represented by two concurrent RTTM lines with different speaker labels. Standard RTTM parsers handle this correctly.

## Constraints

- `start_sec ≥ 0`
- `dur_sec > 0`
- `start_sec + dur_sec ≤ scene_duration_sec`
- All timestamps are in seconds, 3 decimal places

## Label Mapping for Downstream Evaluation

When evaluating synthetic RTTMs with existing scripts that expect CHI/KCHI labels:

```python
SYNTHETIC_TO_CHI = {
    "TARGET_CHILD": "CHI",
    "ADULT_0": "ADT",
    "ADULT_1": "ADT",
    "OTHER_CHILD_0": "OCH",
    "BACKGROUND_SPEECH": "ADT",
}
```

## Example

```
SPEAKER default_14_18mo_42_000001 1 2.340 0.620 <NA> <NA> TARGET_CHILD <NA> <NA>
SPEAKER default_14_18mo_42_000001 1 5.100 3.850 <NA> <NA> ADULT_0 <NA> <NA>
SPEAKER default_14_18mo_42_000001 1 9.400 0.480 <NA> <NA> TARGET_CHILD <NA> <NA>
SPEAKER default_14_18mo_42_000001 1 9.200 2.100 <NA> <NA> ADULT_0 <NA> <NA>
```
*(Lines 3 and 4 above overlap: TARGET_CHILD vocalizes during ADULT_0 speech, 9.2–9.4 s.)*
