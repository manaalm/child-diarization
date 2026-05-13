"""spec 022 constitution-compliance check (Polish T054).

Verifies the four constitution requirements that spec 022 specifically depended on:

  I.  Reproducibility — every new result dir under spec-022 has a `config.json`.
  II. Data Integrity — `all_children_splits/test_all.csv` has no split column
      (zero-shot eval only; not used for training).
  VI. Thesis Sync — `evaluation/balanced_metrics_summary.csv` exists; `CLAUDE.md`
      mentions the spec 022 work.
  Dev Std: File-deletion discipline — no within-child legacy k-fold dirs were deleted;
      encoder relocation preserved git history for the tracked source file.

Exits non-zero on any violation.
"""

import json
import os
import subprocess
import sys

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def main():
    violations = []
    notes = []

    # --- Principle I: every new spec-022 result dir has a config.json
    target_roots = [
        "baselines/scene_analysis_runs/yamnet",
        "baselines/scene_analysis_runs/ast",
    ]
    for root in target_roots:
        full = os.path.join(REPO_ROOT, root)
        if not os.path.isdir(full):
            notes.append(f"[I] result dir not yet built: {root}")
            continue
        if not os.path.exists(os.path.join(full, "config.json")):
            violations.append(f"[I] missing config.json: {root}")

    # --- Principle II: all_children_splits/test_all.csv must not have a 'split' column
    test_all = os.path.join(REPO_ROOT, "whisper-modeling", "all_children_splits", "test_all.csv")
    if os.path.exists(test_all):
        import pandas as pd
        df_head = pd.read_csv(test_all, nrows=1)
        if "split" in df_head.columns:
            violations.append("[II] all_children_splits/test_all.csv has `split` column "
                              "(should be zero-shot-eval-only with no train/val/test partitioning)")
        else:
            notes.append("[II] all_children_splits/test_all.csv correctly omits `split` column")

    # --- Principle VI: balanced_metrics_summary.csv exists, CLAUDE.md mentions spec 022
    bm = os.path.join(REPO_ROOT, "evaluation", "balanced_metrics_summary.csv")
    if not os.path.exists(bm):
        violations.append("[VI] evaluation/balanced_metrics_summary.csv missing")

    claude_md = os.path.join(REPO_ROOT, "CLAUDE.md")
    if os.path.exists(claude_md):
        with open(claude_md) as f:
            text = f.read()
        if "spec-022" not in text and "spec 022" not in text and "spec022" not in text:
            violations.append("[VI] CLAUDE.md does not mention spec 022")
        else:
            n_mentions = text.count("spec-022") + text.count("spec 022") + text.count("spec022")
            notes.append(f"[VI] CLAUDE.md mentions spec 022 in {n_mentions} location(s)")

    # --- File-deletion discipline: no legacy within-child k-fold dirs deleted
    expected_legacy_dirs = [
        "mil/mil_results/whisper_mil_kfold3_f0",
        "mil/mil_results/whisper_mil_kfold3_f1",
        "mil/mil_results/whisper_mil_kfold3_f2",
        "pseudo_frame/results/whisper_pseudo_frame_kfold3_f0",
    ]
    for d in expected_legacy_dirs:
        if not os.path.isdir(os.path.join(REPO_ROOT, d)):
            violations.append(f"[Dev Std] legacy within-child k-fold dir missing (deletion violation): {d}")

    # --- File-deletion discipline: encoder relocation preserved git history.
    # Check 1: --follow only works after the rename has been COMMITTED.
    # If the rename is currently staged-but-not-committed, fall back to verifying
    # the staged rename is detected as a rename by git diff --cached.
    code, stdout, stderr = _run(["git", "log", "--follow", "--oneline", "encoders/baseline_encoders.py"])
    if code != 0:
        violations.append(f"[Dev Std] git log --follow encoders/baseline_encoders.py failed: {stderr}")
    elif stdout:
        n_history_lines = len(stdout.splitlines())
        notes.append(f"[Dev Std] encoders/baseline_encoders.py has {n_history_lines} commits in --follow history")
    else:
        # Empty -- check whether the rename is staged
        code2, staged, _ = _run(["git", "diff", "--cached", "--diff-filter=R", "--name-status"])
        if "baseline_encoders.py" in staged and "encoders/baseline_encoders.py" in staged:
            notes.append("[Dev Std] encoders/baseline_encoders.py rename is staged-but-not-committed; "
                         "--follow history will appear after the commit lands. OK.")
        else:
            violations.append("[Dev Std] encoders/baseline_encoders.py has no git history and no staged rename "
                              "(should have history from baselines/baseline_encoders.py via git mv)")

    # --- Report
    print("=== spec 022 constitution check ===")
    print(f"\nNotes ({len(notes)}):")
    for n in notes:
        print(f"  {n}")
    print(f"\nViolations ({len(violations)}):")
    if violations:
        for v in violations:
            print(f"  {v}")
        print(json.dumps({"violations": len(violations), "notes": len(notes), "status": "FAIL"}, indent=2))
        sys.exit(1)
    else:
        print("  (none)")
        print(json.dumps({"violations": 0, "notes": len(notes), "status": "PASS"}, indent=2))
        sys.exit(0)


if __name__ == "__main__":
    main()
