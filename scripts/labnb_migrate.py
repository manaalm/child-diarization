"""
Migrate all historical child-adult-diarization experiments into the labnb global notebook.

Usage:
    python scripts/labnb_migrate.py [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LAB_ROOT = Path.home() / ".local/state/lab-notebook"
SKILL_SCRIPTS = Path.home() / ".claude/skills/labnb/scripts"
PROJECT_SLUG = "child-adult-diarization"
PROJECT_ROOT = str(REPO)

# Wall-time estimates for historical jobs (budget fields are required by the tool)
ENROLL_BUDGET = "8h"
BASELINE_BUDGET = "4h"
MIL_BUDGET = "48h"
AV_BUDGET = "4h"
COMBINED_BUDGET = "2h"


def register(slug, objective, metric_name, direction, status, budget, metrics_dict,
             notes_extra="", source_ids=None, dry_run=False):
    """Call register_experiment.py and append a results row."""
    cmd = [
        sys.executable, str(SKILL_SCRIPTS / "register_experiment.py"),
        "--lab-root", str(LAB_ROOT),
        "--project-root", PROJECT_ROOT,
        "--project-slug", PROJECT_SLUG,
        "--experiment-slug", slug,
        "--objective", objective,
        "--entry-kind", "experiment",
        "--status", status,
        "--metric-name", metric_name,
        "--direction", direction,
        "--overall-budget", budget,
        "--loop-budget", budget,
    ]
    if source_ids:
        for sid in source_ids:
            cmd += ["--source-id", sid]

    if dry_run:
        print(f"  [DRY-RUN] {slug}")
        print(f"    objective: {objective}")
        if metrics_dict:
            print(f"    metrics:   {metrics_dict}")
        return None

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR registering {slug}:\n{result.stderr.strip()}", file=sys.stderr)
        return None

    exp_dir = Path(result.stdout.strip())
    print(f"  registered → {exp_dir.name}")

    # Write metrics into results.tsv
    if metrics_dict:
        tsv_path = exp_dir / "results.tsv"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        notes = json.dumps(metrics_dict)
        if notes_extra:
            notes = notes_extra + " | " + notes
        with open(tsv_path, "a", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            f1 = metrics_dict.get("f1", metrics_dict.get("test_f1", ""))
            w.writerow([1, ts, "completed", f1, 0, 0, "historical", notes])

    return exp_dir


def load_json(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ── helpers ───────────────────────────────────────────────────────────────────

def enroll_metrics(folder):
    """Load enroll_test_metrics.json or test_metrics.json from a folder."""
    for name in ("enroll_test_metrics.json", "test_metrics.json"):
        d = load_json(Path(folder) / name)
        if d:
            return d
    return {}


def baseline_metrics(folder):
    return load_json(Path(folder) / "test_metrics_tuned.json") or {}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dry = args.dry_run

    LAB_ROOT.mkdir(parents=True, exist_ok=True)

    print("=== Group 1: Enrollment diarizers ===")

    enrollment_runs = [
        (
            "usc-sail-enrollment",
            "USC-SAIL Whisper+LoRA frame classifier with ECAPA enrollment: "
            "classify target-child vocalization in seen-child clips",
            REPO / "whisper-modeling/usc_sail_enrollment_runs",
        ),
        (
            "pyannote-enrollment",
            "Pyannote speaker-diarization-community-1 with ECAPA enrollment: "
            "diarize then match target child",
            REPO / "pyannote/pyannote_enrollment_runs",
        ),
        (
            "babar-enrollment",
            "BabAR VTC 2.0 full pipeline (VTC + phoneme step) with ECAPA enrollment: "
            "extract KCHI segments for seen-child classification",
            REPO / "babar_ecapa_enrollment_runs",
        ),
        (
            "vtc-enrollment",
            "VTC 2.0 standalone (KCHI+OCH) with ECAPA enrollment: "
            "voice type classifier without phoneme step",
            REPO / "vtc_ecapa_enrollment_runs",
        ),
        (
            "vtc-kchi-enrollment",
            "VTC 2.0 standalone KCHI-only with ECAPA enrollment: "
            "voice type classifier, key-child hypothesis only",
            REPO / "vtc_kchi_ecapa_enrollment_runs",
        ),
        (
            "vbx-enrollment",
            "VBx Variational Bayes HMM diarization (pyannote VAD + ECAPA x-vectors) "
            "with cosine similarity target-child identification",
            REPO / "vbx_ecapa_enrollment_runs",
        ),
        (
            "talknet-asd-enrollment",
            "TalkNet-ASD video active speaker detection with ECAPA enrollment: "
            "identify child as smallest face track in SAILS BIDS video",
            REPO / "video_asd_ecapa_enrollment_runs/talknet_asd",
        ),
        (
            "eend-eda-enrollment",
            "EEND-EDA (ESPnet2) end-to-end neural diarization with encoder-decoder attractors "
            "and ECAPA enrollment for target-child identification",
            REPO / "eend_eda_ecapa_enrollment_runs",
        ),
        (
            "sortformer-enrollment",
            "Sortformer (NeMo) sort-based transformer diarization with ECAPA enrollment: "
            "target-child identification via cosine similarity",
            REPO / "sortformer_ecapa_enrollment_runs",
        ),
    ]

    enroll_ids = {}
    for slug, objective, folder in enrollment_runs:
        metrics = enroll_metrics(folder)
        exp = register(
            slug=slug,
            objective=objective,
            metric_name="F1",
            direction="higher",
            status="completed",
            budget=ENROLL_BUDGET,
            metrics_dict=metrics,
            dry_run=dry,
        )
        if exp:
            enroll_ids[slug] = exp.name

    print()
    print("=== Group 2: Baseline encoders (cross-child split) ===")

    baseline_dir = REPO / "baselines/baseline_results"
    baseline_runs = []
    for subdir in sorted(baseline_dir.iterdir()):
        cfg = load_json(subdir / "config.json")
        if cfg is None:
            continue
        metrics = baseline_metrics(subdir)
        if not metrics:
            continue
        model_type = cfg.get("model_type", "?")
        pooling = cfg.get("pooling", "?")
        frozen = cfg.get("freeze_backbone", True)
        unfreeze = cfg.get("unfreeze_last_n_layers", 0)
        lw = cfg.get("use_layer_weights", False)
        aug = "aug" in subdir.name
        ptt = cfg.get("per_timepoint_threshold", False) or "ptt" in subdir.name

        slug = f"baseline-{subdir.name.replace('_', '-')}"
        objective = (
            f"Baseline encoder ({model_type}, pooling={pooling}, "
            f"frozen={frozen}, unfreeze={unfreeze}, layer_weights={lw}, "
            f"aug={aug}, per_timepoint_threshold={ptt}) on cross-child split: "
            "clip-level target-child presence classification"
        )
        baseline_runs.append((slug, objective, metrics))

    for slug, objective, metrics in baseline_runs:
        register(
            slug=slug,
            objective=objective,
            metric_name="F1",
            direction="higher",
            status="completed",
            budget=BASELINE_BUDGET,
            metrics_dict=metrics,
            dry_run=dry,
        )

    print()
    print("=== Group 3: BabAR combined feature models ===")

    combined_path = REPO / "babar_combined_runs/all_model_results.json"
    combined_data = load_json(combined_path) or {}

    babar_enroll_id = enroll_ids.get("babar-enrollment", "")

    for model_key, model_data in combined_data.items():
        features = model_data.get("features", [])
        val_m = model_data.get("val_metrics", {})
        test_m = model_data.get("test_metrics", {})
        combined_metrics = {
            "features": features,
            "val_f1": val_m.get("f1"),
            "val_auroc": val_m.get("auroc"),
            "test_f1": test_m.get("f1"),
            "test_auroc": test_m.get("auroc"),
            "test_auprc": test_m.get("auprc"),
            "test_precision": test_m.get("precision"),
            "test_recall": test_m.get("recall"),
        }
        slug = f"babar-combined-{model_key.replace('_', '-')}"
        objective = (
            f"BabAR combined feature model '{model_key}' — "
            f"features: {', '.join(features) if features else 'see notes'}; "
            "logistic regression or GBM over diarizer/phoneme/embedding feature sets "
            "for seen-child target-child vocalization classification"
        )
        source = [babar_enroll_id] if babar_enroll_id else None
        register(
            slug=slug,
            objective=objective,
            metric_name="F1",
            direction="higher",
            status="completed",
            budget=COMBINED_BUDGET,
            metrics_dict=combined_metrics,
            notes_extra=model_key,
            source_ids=source,
            dry_run=dry,
        )

    print()
    print("=== Group 4: Segment-instance MIL sweep ===")

    mil_all = load_json(REPO / "mil/mil_results/seg_mil/all_configs.json") or []

    # Map frontend → enrollment source id
    frontend_source_map = {
        "babar_vtc": enroll_ids.get("babar-enrollment", ""),
        "pyannote": enroll_ids.get("pyannote-enrollment", ""),
        "usc_sail": enroll_ids.get("usc-sail-enrollment", ""),
        "vbx": enroll_ids.get("vbx-enrollment", ""),
    }

    for entry in mil_all:
        frontend = entry.get("frontend", "?")
        aggregator = entry.get("aggregator", "?")
        slug = f"mil-seg-{frontend.replace('_', '-')}-{aggregator.replace('_', '-')}"
        objective = (
            f"Segment-instance MIL: {frontend} diarizer frontend + "
            f"{aggregator} aggregator over WavLM-base+ segment embeddings; "
            "seen-child split, variable-length bag of diarizer-proposed speech segments"
        )
        source_id = frontend_source_map.get(frontend, "")
        metrics = {k: v for k, v in entry.items() if k not in ("frontend", "aggregator", "config_path")}
        register(
            slug=slug,
            objective=objective,
            metric_name="test_f1",
            direction="higher",
            status="completed",
            budget=MIL_BUDGET,
            metrics_dict=metrics,
            source_ids=[source_id] if source_id else None,
            dry_run=dry,
        )

    print()
    print("=== Group 5: AV fusion pipeline (manual-only) ===")

    av_metrics = load_json(REPO / "av_fusion/av_results/manual_only/metrics_overall.json") or {}
    babar_id = enroll_ids.get("babar-enrollment", "")

    av_models = {
        "audio-only": (
            "Audio-only baseline in AV fusion pipeline: threshold-tuned BabAR enrollment score",
            av_metrics.get("audio_only", {}),
        ),
        "video-only": (
            "Video-only baseline in AV fusion pipeline: XGBoost on manual BIDS visual annotations "
            "(face visibility, lighting, child count)",
            av_metrics.get("video_only", {}),
        ),
        "always-fuse-av": (
            "Always-fuse AV model: late fusion alpha * audio_prob + (1-alpha) * visual_prob "
            "for all clips regardless of visual eligibility",
            av_metrics.get("always_fuse", {}),
        ),
        "gated-av": (
            "Gated AV fusion model: late fusion only for visually eligible clips "
            "(visual_eligibility_score threshold tuned on val), audio-only fallback",
            av_metrics.get("gated_av", {}),
        ),
        "cascaded-av": (
            "3-stage cascaded AV pipeline: VAD gate → child-ID gate → AV fusion; "
            "thresholds grid-searched on val F1",
            av_metrics.get("cascaded_av", {}),
        ),
    }

    for model_slug, (objective, metrics) in av_models.items():
        register(
            slug=f"av-fusion-{model_slug}",
            objective=objective,
            metric_name="F1",
            direction="higher",
            status="completed",
            budget=AV_BUDGET,
            metrics_dict=metrics,
            source_ids=[babar_id] if babar_id else None,
            dry_run=dry,
        )

    print()
    print("=== Done ===")
    if not dry:
        # Rebuild index summary
        result = subprocess.run(
            [sys.executable, str(SKILL_SCRIPTS / "summarize_index.py"),
             "--lab-root", str(LAB_ROOT),
             "--project-slug", PROJECT_SLUG],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(result.stdout)
        # Show count
        exp_dirs = list((LAB_ROOT / "experiments").glob(f"*--{PROJECT_SLUG}--*"))
        print(f"Total registered experiments for {PROJECT_SLUG}: {len(exp_dirs)}")


if __name__ == "__main__":
    main()
