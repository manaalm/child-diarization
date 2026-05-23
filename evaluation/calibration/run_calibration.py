"""Driver for spec-021 US7 (T133-T138).

Reads each top-band system's val/test predictions, fits the calibrator family
on val, picks lowest val-Brier, applies to test, computes pre/post Brier+ECE,
runs split-conformal at alpha=0.10, runs decision-curve at four miss-cost
ratios, and writes:
    evaluation/calibration/per_system_pre_post.csv
    evaluation/calibration/conformal_intervals.csv
    evaluation/calibration/decision_curves.csv
    evaluation/calibration/calibrated_test_predictions/{system}.csv
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from calibrators import calibrate_system, brier, ece
from conformal import split_conformal
from decision_curve import RATIOS, threshold_from_ratio, net_benefit, treat_all_net_benefit


SYSTEMS = [
    ("whisper_mil",          "mil/mil_results/whisper_mil/val_predictions.csv",
                             "mil/mil_results/whisper_mil/test_predictions.csv"),
    ("whisper_medium_mil",   "mil/mil_results/whisper_medium_mil/val_predictions.csv",
                             "mil/mil_results/whisper_medium_mil/test_predictions.csv"),
    ("whisper_large_mil",    "mil/mil_results/whisper_large_mil/val_predictions.csv",
                             "mil/mil_results/whisper_large_mil/test_predictions.csv"),
    ("whisper_pseudo_frame", "pseudo_frame/results/whisper_pseudo_frame/val_predictions.csv",
                             "pseudo_frame/results/whisper_pseudo_frame/test_predictions.csv"),
    ("fused_whisper_medium", "baseline_results_seen_child/fused_attn_unfreeze2_whisper_medium/val_predictions.csv",
                             "baseline_results_seen_child/fused_attn_unfreeze2_whisper_medium/test_predictions.csv"),
    ("fused_whisper_large",  "baseline_results_seen_child/fused_attn_unfreeze2_whisper_large/val_predictions.csv",
                             "baseline_results_seen_child/fused_attn_unfreeze2_whisper_large/test_predictions.csv"),
    ("metadata_stack",       "ensemble_runs/metadata_stack/val_predictions.csv",
                             "ensemble_runs/metadata_stack/test_predictions.csv"),
    ("babar_ecapa",          "babar_ecapa_enrollment_runs/enroll_val_predictions.csv",
                             "babar_ecapa_enrollment_runs/enroll_test_predictions.csv"),
    ("vtc_kchi_ecapa",       "vtc_kchi_ecapa_enrollment_runs/enroll_val_predictions.csv",
                             "vtc_kchi_ecapa_enrollment_runs/enroll_test_predictions.csv"),
]


def main() -> None:
    out_dir = Path("evaluation/calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_dir = out_dir / "calibrated_test_predictions"
    cal_dir.mkdir(exist_ok=True)

    pre_post_rows = []
    conformal_rows = []
    decision_rows = []
    calibrated_probs_by_system: dict[str, np.ndarray] = {}
    test_labels_by_system: dict[str, np.ndarray] = {}

    for name, val_path, test_path in SYSTEMS:
        val_p = Path(val_path)
        test_p = Path(test_path)
        if not val_p.exists() or not test_p.exists():
            print(f"SKIP {name}: missing val or test predictions ({val_path}, {test_path})")
            continue
        try:
            res = calibrate_system(val_p, test_p, name)
        except Exception as e:
            print(f"SKIP {name}: {e}")
            continue

        # Save calibrated test predictions
        cal_csv = cal_dir / f"{name}.csv"
        cal_df = pd.DataFrame({
            "child_id": res["test_child_ids"],
            "timepoint": res["test_cohorts"],
            "label": res["test_labels"],
            "calibrated_prob": res["test_calibrated_probs"],
        })
        cal_df.to_csv(cal_csv, index=False)

        pre_post_rows.append({
            "system": name,
            "selected_calibrator": res["selected_calibrator"],
            "n_val": res["n_val"],
            "n_test": res["n_test"],
            "pre_brier": round(res["pre_brier"], 4),
            "post_brier": round(res["post_brier"], 4),
            "pre_ece": round(res["pre_ece"], 4),
            "post_ece": round(res["post_ece"], 4),
            "delta_brier": round(res["post_brier"] - res["pre_brier"], 4),
            "delta_ece": round(res["post_ece"] - res["pre_ece"], 4),
        })

        # Conformal: use the selected calibrator's val probabilities as conformity calibration set
        val_df = pd.read_csv(val_p)
        score_col = res["score_col"]
        val_y = val_df["label"].to_numpy()
        # Apply the same calibrator to val predictions
        from calibrators import (
            GlobalPlatt, GlobalIsotonic, GlobalTemperature,
            PerCohortPlatt, PerChildPlatt
        )
        val_s = val_df[score_col].to_numpy()
        val_c = val_df[
            "timepoint_norm" if "timepoint_norm" in val_df.columns else "timepoint"
        ].to_numpy()
        val_id = val_df["child_id"].to_numpy()
        sc = res["selected_calibrator"]
        if sc == "global_platt":
            cal = GlobalPlatt().fit(val_s, val_y).transform(val_s)
        elif sc == "global_isotonic":
            cal = GlobalIsotonic().fit(val_s, val_y).transform(val_s)
        elif sc == "global_temperature":
            cal = GlobalTemperature().fit(val_s, val_y).transform(val_s)
        elif sc == "per_cohort_platt":
            cal = PerCohortPlatt().fit(val_s, val_y, val_c).transform(val_s, val_c)
        elif sc == "per_child_platt":
            cal = PerChildPlatt().fit(val_s, val_y, val_id, val_c).transform(val_s, val_id, val_c)
        else:
            cal = val_s

        test_post = np.array(res["test_calibrated_probs"])
        test_y = np.array(res["test_labels"]).astype(int)

        cr = split_conformal(cal, val_y.astype(int), test_post, test_y, alpha=0.10)
        conformal_rows.append({
            "system": name,
            "alpha": cr.alpha,
            "q": round(cr.q, 4),
            "coverage_test": round(cr.coverage_test, 4),
            "coverage_holdout_mean": round(cr.coverage_holdout_mean, 4),
            "coverage_holdout_std": round(cr.coverage_holdout_std, 4),
            "set_size_mean": round(cr.set_size_mean, 4),
            "n_calib": cr.n_calib,
            "n_test": cr.n_test,
            "abs_dev_from_nominal": round(abs(cr.coverage_holdout_mean - 0.90), 4),
            "passes_sc061": "YES" if abs(cr.coverage_holdout_mean - 0.90) <= 0.02 else "NO",
        })

        # Decision-curve at four ratios
        calibrated_probs_by_system[name] = test_post
        test_labels_by_system[name] = test_y

        for label, ratio in RATIOS.items():
            pt = threshold_from_ratio(ratio)
            nb = net_benefit(test_post, test_y, pt)
            nb_treat_all = treat_all_net_benefit(test_y, pt)
            decision_rows.append({
                "system": name,
                "miss_cost_ratio": label,
                "p_t": round(pt, 4),
                "net_benefit": round(nb, 4),
                "net_benefit_treat_all": round(nb_treat_all, 4),
                "advantage_over_treat_all": round(nb - nb_treat_all, 4),
            })

        print(f"OK {name:25s} pre_ece={res['pre_ece']:.4f} -> post_ece={res['post_ece']:.4f} "
              f"calibrator={res['selected_calibrator']} conformal_cov={cr.coverage_holdout_mean:.4f}")

    # ---- Write outputs ----
    pd.DataFrame(pre_post_rows).to_csv(out_dir / "per_system_pre_post.csv", index=False)
    pd.DataFrame(conformal_rows).to_csv(out_dir / "conformal_intervals.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(out_dir / "decision_curves.csv", index=False)

    # SC-062 check: stacker net-benefit dominates Whisper-MIL on at least one ratio
    if "metadata_stack" in calibrated_probs_by_system and "whisper_mil" in calibrated_probs_by_system:
        dominant = []
        for label, ratio in RATIOS.items():
            pt = threshold_from_ratio(ratio)
            stacker_nb = net_benefit(calibrated_probs_by_system["metadata_stack"], test_labels_by_system["metadata_stack"], pt)
            whisper_nb = net_benefit(calibrated_probs_by_system["whisper_mil"], test_labels_by_system["whisper_mil"], pt)
            if stacker_nb > whisper_nb:
                dominant.append({"ratio": label, "stacker_nb": round(stacker_nb, 4), "whisper_nb": round(whisper_nb, 4)})
        sc062 = {
            "passes_sc062": len(dominant) > 0,
            "dominant_ratios": dominant,
        }
    else:
        sc062 = {"passes_sc062": None, "note": "stacker or whisper_mil missing"}

    (out_dir / "sc062_check.json").write_text(json.dumps(sc062, indent=2))
    print(f"\n[SC-062] {sc062}")

    print(f"\nWrote {out_dir}/per_system_pre_post.csv ({len(pre_post_rows)} rows)")
    print(f"Wrote {out_dir}/conformal_intervals.csv ({len(conformal_rows)} rows)")
    print(f"Wrote {out_dir}/decision_curves.csv ({len(decision_rows)} rows)")


if __name__ == "__main__":
    main()
