# Feature Specification: Missing Baselines — Tier 1–4

**Feature Branch**: `013-missing-baselines`
**Created**: 2026-04-29
**Status**: Draft

## User Scenarios & Testing

### US1 (P1): VAD Coverage Baseline — Tier 1

A VAD-based agnostic baseline scoring clips by speech-presence fraction (Silero VAD or energy). Provides reference point: "how much does speaker-aware scoring add over pure speech detection?"

**Acceptance criteria:**
- Script `baselines/vad_baseline.py`; modes: `silero`, `energy`
- Output: `baselines/vad_baseline_runs/{silero,energy}/{val,test}_metrics_tuned.json` on both splits
- AUROC reported; expected ~0.50–0.60 (task not trivially solvable by speech presence)

### US2 (P1): Raw-Clip ECAPA Baseline — Tier 2

Score clips by cosine similarity between per-child prototype and mean/max/top-3 ECAPA window embeddings — no diarizer frontend. Isolates diarizer contribution vs. raw embedding matching.

**Acceptance criteria:**
- Script `baselines/raw_ecapa_baseline.py`; modes: `mean`, `max`, `top3`
- Output: `baselines/raw_ecapa_baseline_runs/{mean,max,top3}/test_metrics_tuned.json`
- Seen-child split only (requires enrollment prototypes)

### US3 (P2): CLAP Zero-Shot Baseline — Tier 3

Score clips by CLAP audio-text cosine similarity against "a young child vocalizing" vs "an adult speaking." Zero-shot, no training.

**Acceptance criteria:**
- Script `baselines/clap_baseline.py`; model: `laion/clap-htsat-fused`
- Output: `baselines/clap_baseline_runs/clap_htsat_fused/test_metrics_tuned.json` on both splits

### US4 (P2): HuBERT-large / wav2vec2-large MIL — Tier 3

Frame-window MIL with HuBERT-large and wav2vec2-large backbones, same pipeline as `wavlm_mil`.

**Acceptance criteria:**
- New configs `mil/configs/{hubert_large_mil,wav2vec2_large_mil}.yaml`
- Output: `mil/mil_results/{hubert_large_mil,wav2vec2_large_mil}/test_metrics_tuned.json`

### US5 (P2): PANNS AudioSet Features — Tier 3

Frozen CNN14 (PANNS, AudioSet-pretrained) feature extractor + linear head.

**Acceptance criteria:**
- Script `baselines/panns_baseline.py`; model: CNN14 from `panns_inference`
- Output: `baselines/panns_baseline_runs/cnn14/test_metrics_tuned.json` on both splits

### US6 (P3): Child-Conditioned MIL — Tier 4

Architectural: each segment score conditioned on child ECAPA prototype — `f([seg, proto, seg−proto])`. Learns "is this segment from this specific child" not "is this child-like."

**Acceptance criteria:**
- New class `ConditionedGatedABMIL` in `mil/mil_model.py`
- Config `mil/configs/conditioned_wavlm_mil.yaml`
- Output: `mil/mil_results/conditioned_wavlm_mil/test_metrics_tuned.json`
- Seen-child split only

### US7 (P3): ECAPA Adapter Triplet Fine-Tuning — Tier 4

Fine-tune a 2-layer MLP adapter on ECAPA with triplet loss on child segments to improve prototype discriminativeness.

**Acceptance criteria:**
- Script `mil/ecapa_adapter_finetune.py`
- Output: `mil/mil_results/ecapa_adapter/test_metrics_tuned.json`
- Seen-child split only
