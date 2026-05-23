"""Background updater for tab:headline-cross. Polls every 60s for each of the
4 remaining gap-fill targets; when a target lands, computes the metric, edits
the LaTeX file in place, and emits one stdout line that Monitor turns into a
notification. Exits once all 4 are done."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
TEX = REPO / "thesis v3/chapters/results_bids.tex"

GAP_FILL = {
    "fused_medium": {
        "label": "Whisper+WavLM fused (medium, PU last 2)",
        "ckpt_dir": REPO / "baselines/baseline_results_cross_child_bids/fused_attn_unfreeze2_whisper_medium",
        "tex_match": re.compile(
            r"(Whisper\+WavLM fused \(medium, PU last 2\)\s*&\s*)---(\s*&\s*)---(\s*&\s*)---(\s*&\s*\$0\.852)",
        ),
        "kind": "single_split",
        "already_filled_marker": "Whisper+WavLM fused (medium, PU last 2)",
    },
    "fused_large": {
        "label": "Whisper+WavLM fused (large-v3, PU last 2)",
        "ckpt_dir": REPO / "baselines/baseline_results_cross_child_bids/fused_attn_unfreeze2_whisper_large",
        "tex_match": re.compile(
            r"(Whisper\+WavLM fused \(large-v3, PU last 2\)\s*&\s*)---(\s*&\s*)---(\s*&\s*)---(\s*&\s*\$0\.879)",
        ),
        "kind": "single_split",
        "already_filled_marker": "Whisper+WavLM fused (large-v3, PU last 2)",
    },
    "whisper_mean_gs": {
        "label": "whisper_mean group-strat 3-fold",
        "fold_dirs": [REPO / f"baseline_results_seen_child/whisper_mean_groupstrat3_f{i}" for i in range(3)],
        "tex_match": re.compile(
            r"(Whisper-small mean \(frozen\)\s*&\s*0\.816 & 0\.810 & 0\.865\s*&\s*)---(\s*\\\\)",
        ),
        "kind": "groupstrat3",
        "already_filled_marker": "Whisper-small mean (frozen)",
    },
    "wavlm_attn_gs": {
        "label": "wavlm_attn group-strat 3-fold",
        "fold_dirs": [REPO / f"baseline_results_seen_child/wavlm_attn_groupstrat3_f{i}" for i in range(3)],
        "tex_match": re.compile(
            r"(WavLM-Base\$\+\$ attention \(frozen\)\s*&\s*0\.721 & 0\.737 & 0\.818\s*&\s*)---(\s*\\\\)",
        ),
        "kind": "groupstrat3",
        "already_filled_marker": "WavLM-Base$+$ attention (frozen)",
    },
}


def compute_single_split(ckpt_dir: Path) -> dict:
    """Read test_metrics_tuned.json + predictions; compute Wgt F1 + BA + AUROC."""
    j = json.load(open(ckpt_dir / "test_metrics_tuned.json"))
    thr = j.get("threshold", 0.5)
    df = pd.read_csv(ckpt_dir / "test_predictions.csv")
    y = df.label.astype(int).values
    if "prob" in df.columns:
        p = df.prob.astype(float).values
    else:
        p = df.score.astype(float).values
    preds = (p >= thr).astype(int)
    tp = ((preds == 1) & (y == 1)).sum()
    tn = ((preds == 0) & (y == 0)).sum()
    fp = ((preds == 1) & (y == 0)).sum()
    fn = ((preds == 0) & (y == 1)).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    prec_n = tn / max(tn + fn, 1); rec_n = tn / max(tn + fp, 1)
    f1_n = 2 * prec_n * rec_n / max(prec_n + rec_n, 1e-9)
    n = len(y); npos = int(y.sum()); nneg = n - npos
    fw = (npos / n) * f1 + (nneg / n) * f1_n
    ba = j.get("balanced_accuracy", 0.5 * (rec + (tn / max(tn + fp, 1))))
    auroc = j["auroc"]
    return {"f1_weighted": fw, "ba": ba, "auroc": auroc}


def compute_groupstrat3(fold_dirs: list[Path]) -> dict:
    """Read 3 fold JSONs, compute mean ± std AUROC."""
    aurocs = []
    for d in fold_dirs:
        if not (d / "test_metrics_tuned.json").exists():
            raise FileNotFoundError(d)
        aurocs.append(json.load(open(d / "test_metrics_tuned.json"))["auroc"])
    return {"auroc_mean": float(np.mean(aurocs)), "auroc_std": float(np.std(aurocs, ddof=0))}


def fmt_single_split_cells(m: dict) -> str:
    return f"{m['f1_weighted']:.3f} & {m['ba']:.3f} & {m['auroc']:.3f}"


def fmt_groupstrat3_cell(m: dict) -> str:
    return f"${m['auroc_mean']:.3f} \\pm {m['auroc_std']:.3f}$"


def apply_update(name: str, info: dict) -> bool:
    """Detect landing + apply LaTeX edit. Returns True when this target is done
    (either because we just filled it, or because the placeholder no longer
    matches — i.e. it was filled in a previous run / by hand)."""
    text = TEX.read_text()
    # Cheap early exit: if the underlying result files don't yet exist, it's
    # not landed.
    if info["kind"] == "single_split":
        if not (info["ckpt_dir"] / "test_metrics_tuned.json").exists():
            return False
        m = compute_single_split(info["ckpt_dir"])
        cells = fmt_single_split_cells(m)
        repl = lambda mm: mm.group(1) + cells.split(" & ")[0] + mm.group(2) + \
                          cells.split(" & ")[1] + mm.group(3) + \
                          cells.split(" & ")[2] + mm.group(4)
        new = info["tex_match"].sub(repl, text, count=1)
        if new == text:
            # Placeholder no longer present — assume already filled.
            print(f"[ALREADY-FILLED] {info['label']}: no placeholder to replace; marking done", flush=True)
            return True
        TEX.write_text(new)
        print(f"[LANDED] {info['label']}: weighted F1={m['f1_weighted']:.4f}  BA={m['ba']:.4f}  AUROC={m['auroc']:.4f}", flush=True)
        return True
    elif info["kind"] == "groupstrat3":
        if not all((d / "test_metrics_tuned.json").exists() for d in info["fold_dirs"]):
            return False
        m = compute_groupstrat3(info["fold_dirs"])
        cell = fmt_groupstrat3_cell(m)
        new = info["tex_match"].sub(lambda mm: mm.group(1) + cell + mm.group(2), text, count=1)
        if new == text:
            print(f"[ALREADY-FILLED] {info['label']}: AUROC={m['auroc_mean']:.4f} ± {m['auroc_std']:.4f} (placeholder absent; marking done)", flush=True)
            return True
        TEX.write_text(new)
        print(f"[LANDED] {info['label']}: AUROC={m['auroc_mean']:.4f} ± {m['auroc_std']:.4f}", flush=True)
        return True
    return False


def main():
    done = set()
    iter_n = 0
    while len(done) < len(GAP_FILL):
        iter_n += 1
        for name, info in GAP_FILL.items():
            if name in done:
                continue
            if apply_update(name, info):
                done.add(name)
        if iter_n % 10 == 0 and len(done) < len(GAP_FILL):
            pending = sorted(set(GAP_FILL) - done)
            print(f"[iter={iter_n}] pending: {pending}", flush=True)
        if len(done) == len(GAP_FILL):
            break
        time.sleep(60)
    print(f"[DONE] all 4 gap-fill targets landed and tab:headline-cross updated", flush=True)


if __name__ == "__main__":
    main()
