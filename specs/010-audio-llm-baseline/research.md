# Research: Audio LLM Zero-Shot Baseline

**Feature**: 010-audio-llm-baseline
**Date**: 2026-04-27

---

## D1: Primary Model Choice

**Decision**: Qwen2-Audio-7B-Instruct (`Qwen/Qwen2-Audio-7B-Instruct` on HuggingFace)

**Rationale**:
- Pure audio LLM (no video/image processing) — clean apples-to-apples comparison against other audio-only baselines (BabAR, USC-SAIL, WavLM-MIL, Whisper-MIL)
- 7B parameters fit in 40GB A100 at bfloat16 full precision (~16–18GB)
- Supports variable-length audio up to ~30 seconds natively; all SAILS clips are ≤ 30s
- Actively maintained HuggingFace release with well-documented chat template
- Falls within the same "audio foundation model" category as the encoder baselines in `baselines/`, making it easy to slot into the existing comparison table

**Alternatives considered**:
- *Qwen2.5-Omni-7B*: More capable (audio+video+image), but adds complexity and uses a different model class; audio-only clips would not exercise its full modality stack; saves for an optional second experiment
- *SALMONN-7B*: Good performance on AudioCaps-style tasks but harder to install (requires custom repo, BEATs + Whisper fusion); higher setup cost than Qwen2-Audio
- *audio-LLaMA / LTU-AS*: Less maintained; fewer downstream audio QA benchmarks; not recommended

**Fallback order if primary unavailable on cluster**:
1. `Qwen/Qwen2.5-Omni-7B` (same HuggingFace API, adds visual modality — ignore it)
2. `SALMONN/SALMONN` (requires separate env and repo clone)

---

## D2: Confidence Score Extraction

**Decision**: Extract logit for "yes" vs. "no" token from the first generated position; normalize to [0,1] via softmax.

**Rationale**:
- `model.generate(..., output_scores=True, return_dict_in_generate=True)` returns a tuple of per-step logit tensors; `scores[0][0]` is the logit vector over the full vocabulary at the first generation step
- Tokenizer encodes "yes" and "no" to single tokens; we take `processor.tokenizer.encode("yes")[-1]` and similarly for "no"
- `softmax([logit_yes, logit_no])` gives a calibrated binary probability without temperature scaling; this is the `prob` column in the predictions CSV
- Threshold is tuned on the val split to maximize F1 (same as all other diarizers)

**Fallback**: If log-probs are unavailable (e.g., model wrapped in a serving API), parse the text response: "yes" → prob=1.0, "no" → prob=0.0, anything else → prob=0.5 and flag as `parse_status=fallback`.

---

## D3: Prompt Template

**Decision**:
```
System: You are a child speech detection assistant.
User: Listen to this audio clip. Is there a child vocalizing (speaking, babbling, or crying)? Answer only: yes or no.
<audio clip embedded here>
```

**Rationale**:
- Instructs the model to answer in one token, enabling clean logit extraction
- "vocalizing" covers child speech, babbling, crying, and vocalization — matching the RTTM CHI label semantics
- System prompt primes the model without including task-specific examples (zero-shot)
- Kept as a single string constant in config so prompt variants can be A/B tested

**Few-shot template addition** (US2):
```
User: Here is a clip WITH a child vocalizing:
<audio example_positive>
Answer: yes

Here is a clip WITHOUT a child vocalizing:
<audio example_negative>
Answer: no

Now, is there a child vocalizing in this clip?
<audio query_clip>
Answer only: yes or no.
```

---

## D4: Result Folder Location

**Decision**: `baselines/audio_llm_baseline_runs/{model_slug}/` — co-located with existing encoder baselines.

**Rationale**:
- Audio LLM inference is a pure audio classification method (no diarization pipeline), making it structurally equivalent to the `baselines/baseline_results/{variant}/` encoder results
- Avoids cluttering the project root with another top-level result folder
- `model_slug` = `qwen2_audio_7b` for the primary experiment; `qwen25_omni_7b` if a second run is done

**Folder contents** (matching enrollment result conventions per Constitution §VI):
```
baselines/audio_llm_baseline_runs/qwen2_audio_7b/
├── config.json
├── val_predictions.csv
├── val_metrics_tuned.json
├── test_predictions.csv
├── test_metrics_tuned.json
└── test_metrics_by_timepoint.csv
```

---

## D5: Caching Strategy

**Decision**: Per-clip JSON cache at `baselines/audio_llm_cache/{model_slug}/{stem}__{md5}.json`, storing `prob`, `response_raw`, `parse_status`, `model_name`, `timestamp`. Script skips clips with existing cache entries before inference.

**Rationale**:
- Mirrors the RTTM cache approach used by all diarizer frontends
- MD5 of audio_path as key guarantees uniqueness even if file renames occur
- JSON format is human-inspectable for debugging degenerate outputs
- Cache is gitignored (like all RTTM/embedding caches)

---

## D6: Degenerate Output Detection

**Decision**: Compute `prediction_variance = np.var(all_probs)` across the test set. If variance < 0.01 (nearly all same), set `degenerate_flag=true` in `config.json` and print a prominent warning. Separately, count `frac_yes` and `frac_no` and include in metrics JSON.

**Rationale**:
- Child babble is far out of distribution for audio LLMs trained on captioned adult audio; the model may default to always-yes or always-no
- Documenting this cleanly supports the thesis claim ("foundation model degradation at the LLM level")

---

## D7: Environment & Installation

**Decision**: Install into the existing `child-vocalizations` conda environment; add `transformers>=4.45`, `accelerate`, `bitsandbytes` (optional, for 4-bit quantization), `soundfile`, `torchaudio`.

**Rationale**:
- Avoids creating another isolated env — the inference script only requires transformers + torchaudio, both of which are already available or easy to add
- `accelerate` enables `device_map="auto"` for multi-GPU support on large-memory nodes
- `bitsandbytes` is optional; bfloat16 full precision is preferred for cleaner probability extraction (quantization can shift logits)

**Constitution §I compliance**: HuggingFace model weights download on first run and are cached at `~/.cache/huggingface/`; this cache is outside the repo and does not need to be committed.
