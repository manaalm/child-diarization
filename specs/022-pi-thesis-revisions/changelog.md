# Spec 022 US1 — Change Log

**Date**: 2026-05-12
**Scope**: MVP (Phase 1 + 2 + US1) of spec 022. Phase 1 setup, Phase 2 `compute_metrics()` extension, US1 BIDS-derived timepoint correction.

## Headline finding

The SAILS spreadsheet (`anotated_processed.csv`) had **855 rows where `timepoint` was missing** that the BIDS directory structure can resolve unambiguously. Only **3 rows have a value disagreement** between BIDS and the spreadsheet (all 3: BIDS=36_month, spreadsheet=14_month — spreadsheet was stale). Net effect of the BIDS correction: the seen-child split grows from **2183 rows / 109 children → 3145 rows / 130 children** (+962 / +21).

Per-timepoint metrics on the existing test set (441 rows, of which 3 changed timepoint) are nearly identical to legacy values to 3 decimal places; the imbalance-aware extensions (`f1_weighted`, `balanced_accuracy`) reveal a much wider per-system spread than F1 alone suggested.

## Artefacts produced

- `whisper-modeling/bids_timepoint.py` — new module exposing `bids_session_to_timepoint(audio_path)` and `derive_timepoint_with_provenance(...)`.
- `whisper-modeling/make_seen_child_split.py` — modified: adds `--use-bids-timepoint` (default true), `--build-all-children-split`, writes to `seen_child_splits/` subdir (was writing one level up — discovered & fixed during US1), backs up prior splits to `*.legacy_pre_bids_022`.
- `whisper-modeling/seen_child_splits/{master_with_split,train,val,test}.csv` — REGENERATED with BIDS timepoints.
- `whisper-modeling/seen_child_splits/{master_with_split,train,val,test,split_summary}.csv.legacy_pre_bids_022` — backups of pre-BIDS versions.
- `whisper-modeling/seen_child_splits/bids_correction_provenance.json` — per-row diff (3145 rows × {bids_session_id, bids_timepoint, spreadsheet_timepoint, agree, rationale, decision}).
- `mil/mil_utils.py:compute_metrics()` — EXTENDED: now also returns `f1_macro`, `f1_weighted`, `balanced_accuracy`. Existing keys (`f1`, `precision`, `recall`, `auroc`, `auprc`) preserved verbatim.
- `evaluation/regenerate_per_timepoint_tables.py` — new utility; walks repo for `test_predictions.csv` + `enroll_test_predictions.csv`, joins on `audio_path` against the BIDS-corrected master, recomputes `test_metrics_by_timepoint.csv` (now with imbalance-aware columns), backs up legacy.
- **298 of 316 `*_metrics_by_timepoint.csv` files regenerated** (with `.legacy_pre_bids_022` backups). 18 files skipped — see Deferred section below.
- `specs/022-pi-thesis-revisions/regenerate_per_timepoint_summary.json` — full run log.

## Split-summary delta (legacy spreadsheet vs new BIDS)

| Metric | Legacy | New | Δ |
|---|---|---|---|
| n_total | 2183 | 3145 | +962 |
| n_train | 1311 | 1884 | +573 |
| n_val | 431 | 626 | +195 |
| n_test | 441 | 635 | +194 |
| n_children_total | 109 | 130 | +21 |
| dropped_groups (children with <5 clips/timepoint) | 76 | 59 | -17 |

## Disagreement breakdown (3145 rows)

| Rationale | Rows |
|---|---|
| (agree — spreadsheet matches BIDS) | 2287 |
| `spreadsheet-missing` (BIDS recovers a timepoint the spreadsheet omitted) | 855 |
| `spreadsheet-stale` (BIDS says one timepoint, spreadsheet says another) | 3 |
| `non-standard-session-id` | 0 |
| `bids-missing` | 0 |

All 3145 decisions are `keep-bids`.

## Per-timepoint deltas on the legacy 441-row test set

Three rows changed timepoint (spreadsheet-stale → BIDS):
- 14_month group: 234 → 233 rows
- 36_month group: 207 → 208 rows

Resulting metric changes per system are all sub-0.005 absolute on f1/auroc/auprc. The far more visible change is the new `f1_weighted` and `balanced_accuracy` columns:

| System | timepoint | f1 (legacy) | f1 (new) | f1_weighted (NEW) | balanced_accuracy (NEW) |
|---|---|---|---|---|---|
| whisper_mil | 14_month | 0.853 | 0.852 | 0.786 | 0.738 |
| whisper_mil | 36_month | 0.917 | 0.918 | 0.854 | 0.711 |
| whisper_pseudo_frame | 14_month | 0.832 | 0.831 | 0.638 | 0.550 |
| whisper_pseudo_frame | 36_month | 0.914 | 0.915 | 0.795 | 0.554 |
| wavlm_mil | 14_month | 0.853 | 0.852 | 0.721 | 0.628 |
| wavlm_mil | 36_month | 0.913 | 0.913 | 0.815 | 0.600 |

`whisper_pseudo_frame`'s balanced accuracy of 0.55 (vs F1 of 0.83) reveals it is predicting positive on nearly every clip (recall≈0.99, precision≈0.72). The PI's instinct to report balanced accuracy was right — the F1 headline understated the imbalance problem.

## Post-MVP polish: split paradigm BIDS audit (2026-05-12)

User flagged that not every split paradigm had been BIDS-corrected by the MVP commit. Audit confirmed gap on the **cross-child split** (`baselines/splits/`) and **within-child k-fold** (`seen_child_splits_kfold_3fold/`); the per-timepoint metric tables on cross-child systems were already BIDS-rejoined via the audio_path-keyed regenerator, but the split CSVs themselves still embedded spreadsheet timepoints.

Patches:
1. `whisper-modeling/seen_child_splits_kfold_3fold_bids/` — new within-child 3-fold, sourced from the BIDS-corrected master via `make_kfold_seen_child_split.py --out-dir seen_child_splits_kfold_3fold_bids` (3145 rows / 130 children, same modulo-K within-cell partitioning as the legacy splitter; legacy `seen_child_splits_kfold_3fold/` preserved).
2. `baselines/splits/` — new cross-child split via `baselines/make_cross_child_split_bids.py`. Sources from `make_seen_child_split.build_master_dataframe` with `use_bids_timepoint=True, require_timepoint=True, min_clips_per_child=1` (relaxed filter matching the legacy cross-child population intent). 3314 rows / 151 children, 105/23/23 train/val/test children disjoint. Legacy `baselines/splits/*` backed up to `.legacy_pre_bids_022`.
3. `baselines/splits_kfold/` — rebuilt cross-child 3-fold against the new cross-child master via `make_cross_child_kfold_split.py --k 3 --seed 42`. 51/50/50 test children per fold. Legacy `splits_kfold/` backed up to `.legacy_pre_bids_022/` directory.

Verification: 6 split-paradigm dirs are BIDS-correct end-to-end (`seen_child_splits/`, `all_children_splits/`, `seen_child_splits_groupstrat_3fold/`, `seen_child_splits_kfold_3fold_bids/`, `baselines/splits/`, `baselines/splits_kfold/`). 1 legacy preserved at `seen_child_splits_kfold_3fold/` (pre-BIDS, 2183/109 — left for reproducibility per Constitution VI).

## US2 GPU dispatch + ensemble BA-retune + thesis_v2 updates (2026-05-12 evening)

Reversal of the earlier "skip US2 GPU work" decision; user requested
that the deferred GPU tasks and thesis chapter handoffs be done.

**Group-stratified 3-fold k-fold dispatch (T019/T021).** Extended
`evaluation/generate_kfold_configs.py` to accept `--variant groupstrat`,
which writes configs that point at `seen_child_splits_groupstrat_3fold/`
and tag result dirs `*_groupstrat3_f<fold>`. 21 SLURM array jobs
submitted (jobs 13863907–13863913) covering 7 systems × 3 folds:
wavlm_mil, whisper_mil, whisper_mil_tsmil_concat, whisper_medium_mil,
whisper_mil_acmil_max, wavlm_pseudo_frame, whisper_pseudo_frame.
Dispatcher scripts: `mil/slurm/train_mil_groupstrat.sh` and
`pseudo_frame/slurm/train_pseudo_groupstrat.sh`. Estimated runtime
~30 GPU-hours total (varies with queue contention).

**Ensemble candidate retune by balanced accuracy (T037 polish).**
Patched `pyannote/ensemble_combined.py` to also dump
`val_predictions.csv` (24 candidate score columns mirroring the
test-side file). New utility `evaluation/retune_ensemble_candidates_by_ba.py`
sweeps per-candidate thresholds on val for max balanced accuracy,
applies to test, and writes 22 BA-tuned candidate rows to
`evaluation/balanced_metrics_summary.csv` (total now 343 rows).
Top candidate: `with_sortformer_mean` BA=0.827 at threshold 0.40
(F1=0.874, AUROC=0.870). This is the highest balanced-accuracy
quantity in the catalog, exceeding both `metadata_stack` (BA=0.812)
and `best_audio_mil_mean` (BA=0.806 after BA-tuning).

**Thesis_v2 updates.** Chapter 04 systems §4.2.4 (T047): added prose
elaboration on the fused Whisper+WavLM concat fusion stage (per-frame
concat along channel axis to (T, 1536), attention pool over time,
single linear FC head; both backbones frozen) + module-relocation
note pointing at `encoders/` and `docs/per_model_training_data.csv`.
Chapter 05 results: (1) headline table extended with a Balanced
Accuracy column + new rows for YAMNet, AST, Qwen3-Omni, encoder
attn-pool variants, with_sortformer_mean ensemble; the previously
implicit "Whisper pseudo-frame is the AUROC leader" claim now
travels with its BA=0.552 footnote. (2) New §5.13 spec-022 section
documenting BIDS-derived timepoint correction, imbalance-aware
metric defaults, the zero-shot scene-analysis baselines, the
group-stratified 3-fold rebuild, and the BA-tuned ensemble retune
(5 subsections). Appendix C: new §spec-022 section with two
detailed BA tables — one full per-system imbalance-aware view
(headline systems × {F1, F1_weighted, BA, precision, recall, AUROC})
and one coverage-split comparison showing all three zero-shot
baselines improve on the broader all-children-coverage population.

**LOOCV (T023/T024).** Deferred even after the reversal. Rationale:
LOOCV at 130 children × 3 top-band systems = 390 SLURM jobs and
~100 GPU-h. With the 21 group-stratified jobs already queued, adding
390 more would saturate the user's queue allocation, congest the
scratch filesystem (260+ trained checkpoints at ~1GB each, ~260GB
disk pressure plus the 1M-inode quota concern), and provide
diminishing methodological return — the group-stratified 3-fold
already gives a defensible cross-child generalisation estimate.
LOOCV is preserved as a future polish item; the SLURM dispatcher
pattern is documented at `specs/022-pi-thesis-revisions/quickstart.md`
and is a single config-generator + dispatcher away from being
runnable.

## BA-retune everywhere (2026-05-12 late evening)

User asked for thesis tables to use balanced-accuracy threshold tuning
instead of F1 threshold tuning, including appendices. Implemented as:

1. **Bug fix in `evaluation/balanced_metrics.py`**: the loader was
   missing the enrollment-system threshold filename (`enroll_val_metrics.json`,
   without `_tuned` suffix). Enrollment systems were defaulting to
   threshold=0.5, producing wrong F1-tuned values (BabAR F1=0.367 instead
   of the correct 0.872 at val-tuned threshold 0.145). Patch adds
   `enroll_val_metrics.json` to the threshold-loader fallback chain.
   Re-ran balanced_metrics.py: F1-tuned summary now 324 rows with
   corrected enrollment thresholds.

2. **New utility `evaluation/retune_all_by_ba.py`** walks every (val,
   test) prediction pair under canonical result roots and:
   - sweeps thresholds 0.05–0.95 step 0.05 on val to maximise
     balanced accuracy
   - applies tuned threshold to test once (Constitution IV)
   - records both BA-tuned metrics and the legacy F1-tuned metrics
     for delta-audit
   - emits `evaluation/balanced_metrics_ba_tuned_summary.csv`
     (280 systems, 1 schema-fail on the top-level multi-column
     ensemble file)

3. **Thesis_v2 chapter 5 headline table updated** to BA-tuned values
   (`thesis_v2/chapters/05_results.tex` Tab.~tab:headline). Most
   non-AV non-ensemble rows now report metrics at the BA-tuned
   threshold instead of the F1-tuned threshold. AUROC and AUPRC are
   threshold-independent and unchanged. Footnote explains the
   pseudo-frame BA recovery (0.55 → 0.77 for Whisper, 0.54 → 0.74
   for WavLM).

4. **Thesis_v2 appendix C corrected and extended**: (a) the
   imbalance-aware headline table (`tab:appC-spec022-headline`) now
   shows the corrected F1-tuned values for enrollment systems with
   their actual val-tuned thresholds; (b) new
   `tab:appC-spec022-ba-retune` table provides per-system F1→BA
   threshold-retune audit with ΔF1 and ΔBA columns. Mean catalog
   trade-off: −0.029 F1 for +0.054 BA. Max retune gain: Whisper
   pseudo-frame +0.219 BA; max loss: −0.113 F1 (Qwen2.5-Omni at
   threshold sweep from 0.45 → 0.85).

5. **CLAUDE.md** new headline-finding paragraph documents the
   retune at the level of "what is the new everywhere default
   operating point".

Not retuned (val_predictions.csv missing in their dir):
- `ensemble_runs/metadata_stack`, `ensemble_runs/metadata_router_learned`,
  similar — the ensemble-stacker variants store predictions in their
  own dir but the val side is only available via the full ensemble
  pipeline rerun. Their F1-tuned values stand in the headline table.

## Out-of-scope for MVP (carried forward)

1. **194 new test rows have no predictions** for any existing system. They're in the new BIDS-derived test set but the systems were trained on the legacy 441-row set. Filling these in requires GPU reruns (US2 group-stratified k-fold or US3 universal-coverage eval).
2. **18 prediction CSVs skipped** by the regenerator because they lack a binarised prediction column (only raw scores). They are: `baselines/parakeet_baseline_runs/*`, `baselines/panns_baseline_runs/*`, `baselines/ecapa_adapter_baseline_runs/*`, `baselines/clap_baseline_runs/*`, `baselines/raw_ecapa_baseline_runs/*`, plus 1 outlier missing `audio_path`. None are headline systems. Follow-up: extend `regenerate_per_timepoint_tables.py` to look up the val-tuned threshold from each system's `val_metrics_tuned.json` and re-binarise.
3. **BabAR per-timepoint** lives in `babar_combined_runs/all_model_results.json` (a single JSON keyed by LR/GBM × 8 feature combos), not in a `test_metrics_by_timepoint.csv`. The regenerator didn't touch it. The published values (BabAR 14m F1=0.864 / 36m F1=0.897) remain in CLAUDE.md as-is; a separate JSON-aware regenerator is needed.
4. **US2** (group-stratified k-fold + LOOCV + canonical balanced_metrics_summary.csv), **US3** (Qwen 3.5-Omni + YAMNet + AST + all-children-coverage split), **US4** (encoder refactor + figure + per-model training-data CSV), **US5** (per-timepoint posthoc restructure) — not started.

## Constitution-gate notes

- **Reproducibility**: all new artefacts come with config commitments — `make_seen_child_split.py` now records `cfg.use_bids_timepoint` in `split_summary.json` (via the script's existing summary block); the regenerator stores its summary at `specs/022-pi-thesis-revisions/regenerate_per_timepoint_summary.json`.
- **Data Integrity & Split Discipline**: prior splits backed up to `*.legacy_pre_bids_022`, NOT deleted. Same for per-timepoint CSVs. No model selection or threshold tuning on test data.
- **File deletion discipline**: zero deletions. Every overwrite has a `.legacy_pre_bids_022` sibling.
- **Thesis Synchronization**: CLAUDE.md updated in the same commit window (see CLAUDE.md additions).
