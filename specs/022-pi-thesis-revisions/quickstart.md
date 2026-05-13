# Quickstart — Spec 022 PI Thesis Revisions

End-to-end verification recipe for each user story. Once these commands return clean, the spec is done. Designed to be runnable in any order modulo the cross-US dependencies flagged in `spec.md`.

---

## US1 — BIDS-derived timepoint correction

```bash
cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

# 1. Commit the existing seen-child split (so the regen diff is auditable)
git status whisper-modeling/seen_child_splits/
# (commit if any local changes)

# 2. Run the BIDS-aware split generator
cd whisper-modeling && PYTHONPATH=. python make_seen_child_split.py --use-bids-timepoint

# 3. Inspect the BIDS-vs-spreadsheet diff
head specs/022-pi-thesis-revisions/bids_vs_spreadsheet_diff.csv
cat  specs/022-pi-thesis-revisions/bids_vs_spreadsheet_diff_summary.json

# 4. Regenerate per-system per-timepoint tables from cached predictions (no model rerun)
python evaluation/regenerate_per_timepoint_tables.py  # NEW utility — emits new test_metrics_by_timepoint.csv per system

# 5. Update CLAUDE.md per-timepoint blocks; record diff in specs/022-pi-thesis-revisions/changelog.md
```

**Success signals**:
- `bids_vs_spreadsheet_diff.csv` exists with `agree`/`disagree` rows.
- `master_with_split.csv` has BIDS-corrected `timepoint_norm`.
- Per-system `test_metrics_by_timepoint.csv` files have new values (or unchanged if no disagreement for that system's clips).
- `CLAUDE.md` headline tables updated; `changelog.md` records the diff.

---

## US2 — Imbalance-aware metrics + group-stratified k-fold

```bash
# 1. Recompute extended metric set from all cached predictions
python evaluation/balanced_metrics.py
head evaluation/balanced_metrics_summary.csv

# 2. Audit current k-fold mechanics
python evaluation/audit_kfold.py  # writes evaluation/kfold_audit.md
less evaluation/kfold_audit.md

# 3. Smoke-test group-stratified k-fold splitter (no training)
python evaluation/group_stratified_kfold.py --system whisper_mil --split-only
cat mil/mil_results/whisper_mil_groupstrat5_membership.json

# 4. Dispatch group-stratified k-fold retraining on top-band systems
for sys in whisper_pseudo_frame whisper_medium_mil whisper_mil whisper_mil_ts_mil_concat babar_combined usc_sail; do
  sbatch evaluation/slurm/run_group_stratified_kfold.sh $sys
done
# (~30 GPU-hours total, depending on system)

# 5. After completion, summarise
python evaluation/group_stratified_kfold.py --aggregate-summary
head evaluation/group_stratified_kfold_summary.csv

# 6. Dispatch LOOCV on top-3 systems
for sys in whisper_pseudo_frame whisper_medium_mil babar_combined; do
  sbatch evaluation/slurm/run_loocv_subset.sh $sys
done

# 7. Update CLAUDE.md within-child k-fold block: keep legacy numbers, add new group-stratified rows
```

**Success signals**:
- `evaluation/balanced_metrics_summary.csv` has ~30 rows × 2 splits (seen-child + all-children-coverage where applicable).
- `evaluation/kfold_audit.md` cites the relevant code paths and explicitly states whether current 3-fold is within-child.
- `evaluation/group_stratified_kfold_summary.csv` has 6 rows (one per top-band system) with `auroc_mean ± auroc_std`.
- `evaluation/loocv_subset_summary.csv` has ~327 rows (3 systems × 109 children).
- `compared_to_within_child_3fold` column shows the delta — if any system's group-stratified mean differs from within-child mean by > 0.05, flag for chapter discussion.

---

## US3 — Audio-scene-analysis baseline expansion

```bash
# 1. Build the universal-coverage split
cd whisper-modeling && PYTHONPATH=. python make_seen_child_split.py --build-all-children-split
head all_children_splits/test_all.csv
cd ..

# 2. YAMNet sibling env (one-time setup)
uv venv yamnet-eval/.venv --python 3.10
source yamnet-eval/.venv/bin/activate
uv pip install tensorflow==2.16 tensorflow-hub==0.16 soundfile==0.12
deactivate

# 3. Run YAMNet on seen-child + all-children splits
sbatch baselines/slurm/run_scene_analysis_baseline.sh yamnet val
sbatch baselines/slurm/run_scene_analysis_baseline.sh yamnet test
sbatch baselines/slurm/run_scene_analysis_baseline.sh yamnet test_all

# 4. Run AST on seen-child + all-children splits
sbatch baselines/slurm/run_scene_analysis_baseline.sh ast val
sbatch baselines/slurm/run_scene_analysis_baseline.sh ast test
sbatch baselines/slurm/run_scene_analysis_baseline.sh ast test_all

# 5. Invalidate Qwen 3.5 cache (will fail loudly otherwise per cache-stale guard)
rm -rf baselines/audio_llm_cache/qwen35_omni_7b/
sbatch baselines/slurm/run_audio_llm_baseline.sh val qwen35_omni_7b
sbatch baselines/slurm/run_audio_llm_baseline.sh test qwen35_omni_7b
sbatch baselines/slurm/run_audio_llm_baseline.sh test_all qwen35_omni_7b

# 6. Re-summarise (the balanced-metrics rerun from US2 step 1 picks up the new dirs automatically)
python evaluation/balanced_metrics.py
grep -E "(yamnet|ast|qwen35)" evaluation/balanced_metrics_summary.csv
```

**Success signals**:
- `whisper-modeling/all_children_splits/test_all.csv` has more rows than `seen_child_splits/test.csv`.
- `baselines/scene_analysis_runs/{yamnet,ast}/test_metrics_tuned.json` exist.
- `baselines/scene_analysis_runs/{yamnet,ast}/README.md` documents AudioSet class-to-score mapping with ontology IDs.
- `baselines/audio_llm_baseline_runs/qwen35_omni_7b/test_metrics_tuned.json` exists; if Qwen 3.5-Omni unavailable, README documents the deferral.
- `evaluation/balanced_metrics_summary.csv` has new rows: `yamnet × {seen_child_test, all_children_coverage}`, `ast × {seen_child_test, all_children_coverage}`, `qwen35_omni_7b × {seen_child_test, all_children_coverage}`.

---

## US4 — Encoder section restructure

```bash
# 1. Relocate encoder code with history preservation
git mv baselines/baseline_encoders.py encoders/baseline_encoders.py
git mv baselines/run_fused_attn_unfreeze2_backbone_swap.py encoders/run_fused_attn_unfreeze2_backbone_swap.py
git mv baselines/run_fused_attn_unfreeze2_kfold.py encoders/run_fused_attn_unfreeze2_kfold.py
touch encoders/__init__.py

# 2. Add backward-compat shims (one-cycle deprecation)
cat > baselines/baseline_encoders.py <<'EOF'
"""Shim — moved to encoders/baseline_encoders.py. Imports preserved for one release cycle."""
from encoders.baseline_encoders import *  # noqa: F401, F403
EOF
# (similar for the two run_fused_attn shims)

# 3. Generate the encoder pipeline figure
python docs/figures/build_encoder_pipeline_figure.py  # NEW script — produces docs/figures/encoder_pipeline.{png,pdf}

# 4. Author the fusion-of-encoders prose (manual edit of thesis chapter / megadoc)

# 5. Generate per-model training-data CSV
python docs/per_model_training_data.py
head docs/per_model_training_data.csv

# 6. Smoke test: re-run encoder evaluation from old import path; should still work via shim
python -c "from baselines.baseline_encoders import EncoderBaseline; print('shim OK')"
```

**Success signals**:
- `encoders/baseline_encoders.py` exists; `git log --follow encoders/baseline_encoders.py` shows full pre-move history.
- `baselines/baseline_encoders.py` shim re-exports; old import paths still work.
- `docs/figures/encoder_pipeline.{png,pdf}` rendered with all four canonical steps + fused panel.
- `docs/per_model_training_data.csv` lists ~30 rows.
- Thesis chapter / megadoc has the fusion-of-encoders prose elaboration.

---

## US5 — Per-timepoint posthoc analysis

```bash
# 1. Build the consolidated posthoc table (after US1 corrections land)
python evaluation/build_posthoc_per_timepoint_table.py
less evaluation/posthoc_per_timepoint_table.md

# 2. Restructure thesis chapter (manual)
#    - Headline tables: drop per-timepoint columns; show only combined.
#    - New subsection: "Posthoc: per-timepoint stratification" — paste the consolidated table.
#    - Flag systems with |14m - 36m AUROC| > 0.05.

# 3. Update CLAUDE.md headline tables to mirror the chapter structure (combined-only headline)
```

**Success signals**:
- `evaluation/posthoc_per_timepoint_table.md` has one row per system with combined / 14m / 36m / delta / flagged columns.
- Thesis chapter headline tables show combined-timepoint metrics only.
- Per-timepoint breakdown appears as a single consolidated posthoc subsection.
- `CLAUDE.md` headline tables mirror the chapter (combined-only).

---

## Full verification (run after all US slices complete)

```bash
# Cross-check: every system in the headline table has a row in both balanced_metrics_summary.csv and per_model_training_data.csv
python evaluation/spec022_completeness_check.py
# Expects: 0 mismatches; reports per-system status.

# Check Constitution gates: every new result dir has config.json; no within-child k-fold dir was deleted; encoders/ git history preserved
python evaluation/spec022_constitution_check.py
```
