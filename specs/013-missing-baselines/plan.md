# Implementation Plan: Missing Baselines — Tiers 1–4

**Branch**: `013-missing-baselines` | **Date**: 2026-04-29 | **Spec**: `specs/013-missing-baselines/spec.md`

## Summary

Seven new baselines across four tiers fill gaps in the current evaluation. Tier 1 adds an agnostic speech-presence reference (VAD/energy). Tier 2 adds a diarizer-free ECAPA ablation. Tier 3 adds AudioSet (PANNS), CLAP zero-shot, and two larger SSL encoders. Tier 4 adds child-conditioned MIL and ECAPA triplet fine-tuning — the only architectural novelty beyond existing baselines.

US1 and US2 are highest priority (P1): they directly answer thesis questions about what diarization is actually contributing. US3–US5 are P2 (comparative reference points). US6–US7 are P3 (novel architecture, highest compute cost, highest risk).

## Technical Context

**Language/Version**: Python 3.10 (`child-vocalizations` conda env)
**Primary Dependencies**: torch 2.8+cu128, torchaudio, transformers 4.57+, numpy, pandas, scikit-learn, tqdm; `panns_inference` (PANNS), `laion_clap` (CLAP) — both pip-installable into existing env
**Storage**: Results under canonical folders per constitution; caches under `baselines/*_cache/`
**Testing**: Manual dry-run (`--max-clips 5 --dry-run`) before SLURM submission
**Target Platform**: SLURM cluster (NVIDIA GPU nodes, `ou_bcs_normal,pi_satra`)
**Project Type**: ML experiment pipeline
**Performance Goals**: Each US1–US5 job ≤12h GPU; US6–US7 ≤48h GPU
**Constraints**: seed=42 everywhere; val-only threshold tuning; no test leakage; config.json committed with every result
**Scale/Scope**: 2183 clips (seen-child) + 908 clips (cross-child val+test) per job

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Reproducibility | ✅ PASS | seed=42 in all configs; SLURM scripts log job ID; config.json committed alongside results |
| II. Split discipline | ✅ PASS | US1/US3/US4/US5 run on both splits (labeled); US2/US6/US7 seen-child only (labeled); no cross-contamination |
| III. Baseline-first | ✅ PASS | All USs are baselines or ablations of existing methods |
| IV. Metrics | ✅ PASS | F1, precision, recall, AUROC, AUPRC + per-timepoint breakdown required for each US |
| V. Ablations | ✅ PASS | US2 explicitly ablates diarizer contribution; US6 ablates joint vs. late-fusion conditioning |
| VI. Thesis sync | ✅ PASS | Results committed to canonical paths; config.json required |
| VII. Documentation | ✅ PASS | Each script requires docstring; CLAUDE.md to be updated after results |
| File deletion | ✅ PASS | No deletions planned; only new files created |

No violations. No complexity tracking required.

## Project Structure

### Documentation (this feature)

```text
specs/013-missing-baselines/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code

```text
baselines/
├── vad_baseline.py                          # US1: Silero VAD + energy scoring
├── raw_ecapa_baseline.py                    # US2: raw-clip ECAPA (no diarizer)
├── clap_baseline.py                         # US3: CLAP zero-shot
├── panns_baseline.py                        # US5: CNN14/AudioSet features + linear head
├── vad_baseline_runs/{silero,energy}/       # US1 results
├── raw_ecapa_baseline_runs/{mean,max,top3}/ # US2 results
├── clap_baseline_runs/clap_htsat_fused/     # US3 results
└── panns_baseline_runs/cnn14/              # US5 results

mil/
├── configs/
│   ├── hubert_large_mil.yaml               # US4
│   ├── wav2vec2_large_mil.yaml             # US4
│   └── conditioned_wavlm_mil.yaml          # US6
├── mil_model.py                             # US6: add ConditionedGatedABMIL
├── ecapa_adapter_finetune.py               # US7
└── mil_results/
    ├── hubert_large_mil/                   # US4
    ├── wav2vec2_large_mil/                 # US4
    ├── conditioned_wavlm_mil/              # US6
    └── ecapa_adapter/                      # US7

baselines/slurm/
├── run_vad_baseline.sh                     # US1
├── run_raw_ecapa_baseline.sh               # US2
├── run_clap_baseline.sh                    # US3
├── run_panns_baseline.sh                   # US5
└── run_hubert_wav2vec2_mil.sh             # US4

mil/slurm/
├── run_conditioned_mil.sh                  # US6
└── run_ecapa_adapter.sh                    # US7
```

---

## Phase 0: Research

*(See `research.md`)*

---

## Phase 1: Implementation Notes

### US1 — VAD Coverage Baseline

**Silero mode**: Use `torch.hub.load("snakers4/silero-vad", "silero_vad")` → per-frame binary VAD → `speech_fraction = detected_frames / total_frames` → threshold-tune on val.

**Energy mode**: Compute RMS energy per 20ms frame → threshold at 40dB to get binary voiced frames → `speech_fraction`.

Score direction: higher speech_fraction = more speech = possibly child present. But unlike Parakeet gap_ratio, VAD is direction-agnostic (it detects all speech, not just ASR-transcribable speech). AUROC expectation: ~0.50–0.60 (task not trivially solvable by speech presence alone, since adult-only clips also have high VAD coverage).

**Why this matters**: If AUROC ≈ 0.50, confirms the task requires speaker identity, not just speech presence. If AUROC ≈ 0.65+, suggests some child-specific acoustic property (energy, prosody) is accessible without speaker modeling.

### US2 — Raw-Clip ECAPA Baseline

Uses the existing `ecapa_tdnn` prototype-building infrastructure from `unified.py`. For each test clip:
1. Segment clip into 1.5s windows (50% overlap)
2. Embed each window with frozen ECAPA-TDNN (same model as enrollment)
3. Compute cosine similarity to child prototype
4. Score = `mean` / `max` / `mean-of-top-3` over windows

This is the "enrollment without diarization" ablation. Expected AUROC: somewhere between random (0.50) and BabAR enrollment (0.820). If close to 0.820, the diarizer frontends contribute little. If much lower (e.g., 0.65), the diarizer's KCHI selection is doing most of the work.

**Implementation**: Reuse `ECAPAEncoder` from `unified.py`, load prototypes from `babar_ecapa_enrollment_runs/child_prototype_stats.csv`. New script; no new model training.

### US3 — CLAP Zero-Shot Baseline

`laion/clap-htsat-fused` via HuggingFace. Process:
1. Embed each clip with CLAP audio encoder
2. Embed prompts with CLAP text encoder:
   - Positive: `"a young child vocalizing"`, `"a baby or toddler making sounds"`
   - Negative: `"an adult speaking"`, `"silence"`
3. Score = `cos_sim(audio_emb, mean(positive_text_embs)) − cos_sim(audio_emb, mean(negative_text_embs))`
4. Map to [0,1] via sigmoid; threshold-tune on val

**Why this matters**: CLAP is contrastive (trained on audio-text pairs), unlike Qwen2-Audio (autoregressive). Different architecture family, different failure modes. If CLAP AUROC ≈ Qwen2's 0.725, confirms the zero-shot acoustic discrimination is architecture-agnostic.

### US4 — HuBERT-large / wav2vec2-large MIL

Copy `mil/configs/wavlm_mil.yaml` → replace `backbone` with:
- `facebook/hubert-large-ls960-ft` (HuBERT-large, 316M params, ~1024-dim)
- `facebook/wav2vec2-large-960h` (wav2vec2-large, 315M params, ~1024-dim)

Both are SSL speech encoders pretrained on LibriSpeech 960h. The `BackboneExtractor` in `mil_model.py` already handles arbitrary HuggingFace speech encoders via `AutoModel`. The 1024-dim embedding requires updating `mil_hidden_dim` (256→512) and potentially `batch_size` (8→4) due to memory.

**Key question**: Does a larger SSL encoder with more representational capacity improve child-voice discrimination? Hypothesis: marginal gain (the task is more about speaker identity than phonetic representation, which Whisper-small already captures well).

### US5 — PANNS AudioSet Features

`panns_inference` provides `Cnn14` with 527-class AudioSet probability outputs (+ 2048-dim embedding). For child vocalization detection:
1. Extract 2048-dim embedding per clip via `CNN14.forward(audio)` (full clip, no windowing)
2. Train linear head (LR or small MLP) on seen-child train split
3. Cross-child: same linear head, evaluate directly (no personalization)

AudioSet classes include `Speech`, `Child speech, kid speaking`, `Baby cry, infant cry`, `Babbling` — directly relevant. PANNS embeddings may carry complementary signal to speech-pretrained encoders.

**Installation**: `pip install panns_inference` into `child-vocalizations` env; model downloads automatically on first call (~120MB checkpoint).

### US6 — Child-Conditioned MIL

Extend `GatedABMILHead` in `mil_model.py` with prototype conditioning:

```python
class ConditionedGatedABMILHead(nn.Module):
    def __init__(self, emb_dim, proto_dim, hidden_dim, dropout):
        # instance_input = [seg_emb | proto_emb | seg_emb - proto_emb]
        # input_dim = emb_dim + proto_dim + emb_dim
        super().__init__()
        input_dim = 2 * emb_dim + proto_dim
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(emb_dim, 1)  # pools over seg_embs
```

At each clip, load the child's ECAPA prototype (from the enrollment prototype cache). The prototype embedding is concatenated with each segment embedding to compute attention weights. The final classifier pools original segment embeddings.

**Training change**: `MILBagDataset` must return the child prototype alongside the bag. Prototype lookup by `child_id + timepoint_norm` key.

**Why this matters**: This is the only experiment that directly encodes the "who is the target child" signal into the MIL aggregation rather than treating it as post-hoc ensemble fusion. High ceiling if child-specificity is the main discriminating factor; similar performance if acoustic category (child-like sounds) is sufficient.

### US7 — ECAPA Adapter Triplet Fine-Tuning

Fine-tune a 2-layer MLP adapter `g: R^192 → R^192` on top of frozen ECAPA-TDNN using triplet loss:

```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

where:
- Anchor: positive segment from target child (from training RTTM)
- Positive: another positive segment from the same child
- Negative: positive segment from a different child

After fine-tuning, rebuild prototypes using `g(ecapa_emb)` and re-evaluate enrollment.

**Scale**: ~967 positive training clips × avg 3 KCHI segments each = ~2900 anchors. Small dataset — likely needs careful regularization. Use cosine triplet loss (more stable than Euclidean for this scale).

**Expected outcome**: Marginal improvement (5–10%) on AUROC if adult-trained ECAPA has significant domain mismatch on child speech. Could be null result if ECAPA generalizes well (evidence: BabAR AUROC 0.820 is already strong).

---

## Priority & Ordering

| US | Tier | Priority | Compute | Expected AUROC | Thesis value |
|----|------|----------|---------|----------------|--------------|
| US1 (VAD) | 1 | P1 | <1h CPU/GPU | 0.50–0.60 | High: validates speaker-specificity requirement |
| US2 (raw ECAPA) | 2 | P1 | 2–4h GPU | 0.65–0.80 | High: ablates diarizer contribution |
| US3 (CLAP) | 3 | P2 | 2–4h GPU | 0.60–0.75 | Medium: second zero-shot reference |
| US5 (PANNS) | 3 | P2 | 4–8h GPU | 0.70–0.85 | Medium: new feature family |
| US4 (HuBERT/w2v2) | 3 | P2 | 24–48h GPU | similar to WavLM | Low: scaling check |
| US6 (conditioned MIL) | 4 | P3 | 24–48h GPU | 0.85–0.92 | High if it works; hard to justify if null |
| US7 (ECAPA adapter) | 4 | P3 | 12–24h GPU | 0.82–0.87 | Medium: well-motivated but uncertain |

**Recommended execution order**: US1 → US2 → US3 → US5 → US4 → US6 → US7

Start with US1+US2 immediately (quick, high thesis value). Submit US3+US5 in parallel. US4 last among P2 (lower value). US6+US7 only if thesis timeline permits.
