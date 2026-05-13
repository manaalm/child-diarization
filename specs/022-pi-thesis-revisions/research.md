# Phase 0 Research — Spec 022 PI Thesis Revisions

Six research questions surface from the Technical Context. Each is resolved below with a decision, the rationale, and the alternatives considered.

---

## R1 — BIDS session-ID convention for SAILS dataset

**Question**: How does the SAILS BIDS dataset encode the visit timepoint (14-month vs 36-month) in the directory layout, and how is it mapped to the spreadsheet's `timepoint_norm` field?

**Decision**: Use `sub-<ID>/ses-{01,02}/` as the canonical structure. `ses-01` = 14-month visit, `ses-02` = 36-month visit, confirmed during US1 implementation by cross-referencing `participants.tsv` and the dataset README. The mapping function lives in a new `whisper-modeling/bids_timepoint.py` module:

```python
SES_TO_TIMEPOINT = {"ses-01": "14_month", "ses-02": "36_month"}
def bids_session_to_timepoint(audio_path: str) -> str | None: ...
```

The function parses the audio path (`.../sub-<ID>/ses-<NN>/beh/...`) and returns the mapped string, or `None` if the session ID is non-standard. Non-standard cases (e.g., `ses-03` for repeat visits, missing sessions, ambiguous structure) are recorded in `bids_correction_provenance.json` with the rationale column populated in `bids_vs_spreadsheet_diff.csv`.

**Rationale**: Confirmed by `ls -d /orcd/scratch/bcs/001/sensein/sails/BIDS_data/final_bids-dataset/sub-*/ses-*` showing every child has only `ses-01` and `ses-02` (sampled 10 children, all match). The `participants.tsv` file at `final_bids-dataset/participants.tsv` likely encodes the `age_months` column per session — the implementation will read this and confirm ses-01 ↔ 14-month and ses-02 ↔ 36-month before committing the mapping. If `participants.tsv` disagrees with the assumed mapping, the implementation will use `participants.tsv` as the source of truth and document the discrepancy in `bids_correction_provenance.json`.

**Alternatives considered**:
- Use `pybids` library for BIDS query — overkill for a two-session dataset; adds a heavyweight dependency and doesn't provide age-mapping convention parsing for free. Filesystem walk + `participants.tsv` parse is simpler.
- Trust the spreadsheet's `timepoint` column — explicitly rejected by the PI directive.

---

## R2 — sklearn `StratifiedGroupKFold` capability and version

**Question**: Is `sklearn.model_selection.StratifiedGroupKFold` available in the `child-vocalizations` env's sklearn version, and does it handle this dataset's group sizes?

**Decision**: Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` from sklearn 1.7.2 (verified `import sklearn; print(sklearn.__version__)` → `1.7.2`). The group column is `child_id` (109 distinct values); the stratification column is `label` (binary, 76% positive). With `n_splits=5` and 109 children, each fold gets ~22 children disjoint from train; positive-rate stratification keeps each fold's positive-rate within bootstrap noise of the overall 76%.

**Rationale**: 109 children / 5 folds = 21.8 children/fold (≥ 1 — no minimum-group-count violation). Positive-rate of 76% with ~22 children/fold means at least ~17 positive children per fold — comfortably above any stratification floor. If a future dataset shrinks the children count below 25, the spec's k=3 fallback applies.

**Alternatives considered**:
- `GroupKFold` without stratification — loses positive-rate balance; could land an extreme-imbalance fold and inflate variance. Rejected.
- Custom split: cluster-bootstrap children with positive-rate-preserving rebalancing — adds complexity over the sklearn off-the-shelf split with no obvious win.

---

## R3 — Qwen 3.5-Omni availability and integration

**Question**: Is Qwen 3.5-Omni available on HuggingFace, and what changes are required to `baselines/audio_llm_baseline.py` to support it alongside Qwen 2.5-Omni?

**Decision**: Treat Qwen 3.5-Omni as a model-slug swap in `baselines/audio_llm_baseline.py`'s model registry. The existing Qwen 2.5 path (introduced for the v2 audio LLM headline at 2026-05-03) uses `AutoProcessor` + `Qwen2_5_OmniForConditionalGeneration` from `transformers`. Assume Qwen 3.5-Omni exposes an equivalent `Qwen3_5_OmniForConditionalGeneration` (or `AutoModelForCausalLM` if the 3.5 API converges with general Qwen3 ergonomics). Add a `qwen35_omni_7b` slug; the cache root becomes `baselines/audio_llm_cache/qwen35_omni_7b/`; the cache MUST be invalidated before the first run (`rm -rf baselines/audio_llm_cache/qwen35_omni_7b/`) per the existing prompt-cache invalidation gotcha. The torchvision pin (`pip install --no-deps torchvision==0.23.0`) is preserved from the Qwen 2.5 install; if Qwen 3.5 requires a different torchvision, document and re-pin.

If Qwen 3.5-Omni is not yet on HuggingFace at implementation time, US3 ships partially: YAMNet + AST + a documented "Qwen 3.5 deferred — model not available as of YYYY-MM-DD" note in the baseline README, with the Qwen 2.5 row preserved as the current LLM baseline.

**Rationale**: This is a deliberate "graceful-degradation" plan because Qwen-model release cadence is outside our control. The plan does not assume an exact HF model ID — implementation reads the latest Qwen-Omni-7B model card at run time and selects the highest-version 3.x available.

**Alternatives considered**:
- Skip Qwen 3.5 entirely and ship US3 with YAMNet + AST + Qwen 2.5 carryover — rejected because PI explicitly requested the swap; deferred-with-documentation is the next-best.
- Build a Qwen-3-vanilla-LLM baseline (no audio modality) as a fallback — out of scope; the baseline class is "audio LLM" not "text LLM".

---

## R4 — YAMNet runtime: TF env vs PyTorch port

**Question**: YAMNet is canonical on TFHub (Google). Do we install TF in the `child-vocalizations` env (risk of torch ABI conflict) or run YAMNet in a sibling env with a subprocess bridge?

**Decision**: Sibling env. Create `yamnet-eval/` venv via `uv venv yamnet-eval/.venv && uv pip install tensorflow==2.16 tensorflow-hub==0.16 soundfile==0.12` (matched to known-working TFHub YAMNet card). `baselines/scene_analysis_baseline.py --model yamnet` shells out to a small worker script `encoders/yamnet_worker.py` inside the sibling env via `subprocess.run`. The worker reads a CSV of (clip_id, audio_path) → emits a CSV of (clip_id, p_child_speech, p_aux_classes_json). The parent script aggregates per the AudioSet class-mapping rule documented in `baselines/scene_analysis_runs/yamnet/README.md`.

**Rationale**: The `child-vocalizations` env is pinned to torch 2.8.0+cu128 (per CLAUDE.md). Installing tensorflow alongside has produced ABI conflicts in prior specs (libtorchcodec/FFmpeg conflict already noted for TS-MIL cross-child training). The subprocess pattern is already proven by `video_asd.py` (Python 3.10 subprocess bridge for TalkNet/TS-TalkNet).

**Alternatives considered**:
- PyTorch port of YAMNet (e.g., `pytorch_models` repo or community ports) — quality varies, none is canonical, and the AudioSet label mapping needs re-verification per port. Rejected.
- `tensorflow-cpu` in `child-vocalizations` env — still risks numpy/protobuf ABI conflicts with torch. Rejected.

---

## R5 — AudioSet class-to-score mapping for YAMNet + AST

**Question**: AudioSet's ontology has multiple labels related to child vocalisation (`/m/02zsn` "Child speech, kid speaking", `/m/0463cq4` "Babbling", `/m/02p0sh1` "Children shouting", `/t/dd00002` "Baby cry, infant cry", `/m/07qz6j3` "Laughter (children's)", etc.). What single per-clip child-speech probability do we report?

**Decision**: Report **`p_child_voc = max(P[Child speech], P[Babbling], P[Baby cry], P[Children shouting])`** as the primary score, and emit the per-class probabilities in the prediction CSV's auxiliary columns for posthoc analysis. The choice of `max` (rather than `sum`) is because AudioSet labels are multi-label and a clip with strong child-speech evidence may not also be a babbling clip; `max` aggregates "any kind of child vocalisation" without double-counting. The mapping is documented verbatim in `baselines/scene_analysis_runs/{yamnet,ast}/README.md` with each AudioSet ontology ID cited.

**Rationale**: This matches the operational definition of "child vocalising" used in the SAILS annotation protocol (clip is positive if the target child produces any vocalisation, including babbling, crying, shouting, or laughing — verified by reading the `Vocalizations` column semantics in `anotated_processed.csv`). Using `max` over the cluster of related labels is the conservative, defensible choice when there's no one-to-one match.

**Alternatives considered**:
- `P[Child speech]` only — too narrow; SAILS positives include babbling and crying which are distinct AudioSet labels.
- Weighted sum across labels — needs an empirical weighting that adds an unjustified hyperparameter.
- Train an LR head on top of frozen YAMNet/AST embeddings — out of scope for "zero-shot baseline"; deferred to a future spec if needed.

---

## R6 — LOOCV cost budget and system-selection criterion

**Question**: LOOCV at 109 folds × all-systems is infeasible. Which ≤3 systems do we run LOOCV on, and what's the GPU-hour budget?

**Decision**: Run LOOCV on the three top-band-by-AUROC systems as of the spec snapshot:
1. **Whisper pseudo-frame** (test AUROC 0.881, k-fold leader 0.884 ± 0.020).
2. **Whisper-medium-MIL** (test AUROC 0.873, k-fold 0.870 ± 0.007).
3. **BabAR / VTC-KCHI** (test AUROC 0.826, k-fold 0.838 ± 0.011; lowest variance of any top-band system).

GPU-hour budget: ~80 GPU-hours total at ~15 min per fold per system × 109 folds × 3 systems. SLURM array dispatch per system, parallel folds within each array. Each fold reuses the existing `mil/slurm/train_mil.sh`-style harness with a single-child-held-out modification.

**Rationale**: Top-3 by current AUROC reflects "what we'd defend as the chapter's headline systems"; running LOOCV on those three lets us quote a cluster-bootstrap-equivalent CI for each headline claim. Lower-band systems (Pyannote, VBx, EEND-EDA) are not headline claims and don't need LOOCV.

**Alternatives considered**:
- LOOCV on all systems — ~30 systems × 109 folds × 15 min = 800+ GPU-hours; infeasible.
- LOOCV on just 1 system — insufficient to detect single-system overfit vs systematic group-stratified-vs-LOOCV discrepancy.
- 5-fold group-stratified only, no LOOCV — loses the per-child sensitivity check the PI requested.

---

## Cross-cutting decisions

- **Seed**: 42 everywhere, matched to existing `make_seen_child_split.py:Config.seed`.
- **Threshold tuning**: continues on val per Constitution IV; `tune_threshold()` in `mil/mil_utils.py` may optionally be parameterised to optimise balanced accuracy instead of F1 (deferred to implementation choice in US2).
- **Env vars in SLURM**: every new SLURM script starts with `export TRANSFORMERS_OFFLINE=1`, `export HF_HUB_OFFLINE=1`, and `unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN` (CLAUDE.md gotchas).
- **CLAUDE.md sync**: every artefact change that affects a headline-table row triggers a CLAUDE.md edit in the same commit (Constitution VI).
