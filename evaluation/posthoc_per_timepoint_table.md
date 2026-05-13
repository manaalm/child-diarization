# Posthoc: per-timepoint stratification (spec 022 US5)
Generated from `evaluation/balanced_metrics_summary.csv` + each system's `test_metrics_by_timepoint.csv` (BIDS-corrected per US1).
**Rows**: 299 systems with per-timepoint breakdowns available.
**Flag threshold**: |Δ AUROC 36m−14m| > 0.05 → flagged.
**Flagged systems**: 85.

## Headline table — combined-timepoint metrics (primary)

| System | F1 | Balanced Acc | AUROC |
|---|---|---|---|
| `baseline_results_seen_child/fused_attn_unfreeze2_whisper_large` | 0.896 | 0.759 | 0.906 |
| `ensemble_runs/metadata_stack_av` | 0.898 | 0.816 | 0.905 |
| `ensemble_runs/metadata_stack` | 0.905 | 0.812 | 0.904 |
| `ensemble_runs/metadata_stack_av_11sys_backup` | 0.897 | 0.819 | 0.904 |
| `ensemble_runs/advanced_av/av_per_track_added` | 0.898 | 0.816 | 0.902 |
| `ensemble_runs/advanced_av/av_pure_visual` | 0.894 | 0.813 | 0.902 |
| `pseudo_frame/results/whisper_pseudo_frame_kfold3_f2` | 0.895 | 0.657 | 0.902 |
| `ensemble_runs/advanced_av/av_eligibility_only` | 0.904 | 0.811 | 0.900 |
| `ensemble_runs/advanced_av/audio_per_child` | 0.904 | 0.766 | 0.900 |
| `ensemble_runs/advanced/per_child_offset` | 0.904 | 0.766 | 0.900 |
| `ensemble_runs/metadata_stack_11sys_backup` | 0.901 | 0.795 | 0.900 |
| `baseline_results_seen_child/fused_attn_unfreeze2_whisper_medium_kfold3_f1` | 0.903 | 0.815 | 0.900 |
| `ensemble_runs/advanced/fp_focused` | 0.901 | 0.801 | 0.899 |
| `ensemble_runs/advanced/cv_stacked` | 0.901 | 0.795 | 0.899 |
| `ensemble_runs/advanced/pure` | 0.901 | 0.798 | 0.899 |
| `ensemble_runs/advanced_av/audio_pure` | 0.901 | 0.798 | 0.899 |
| `ensemble_runs/advanced/bagged_stacker` | 0.902 | 0.806 | 0.899 |
| `ensemble_runs/advanced_av/av_full` | 0.907 | 0.766 | 0.898 |
| `ensemble_runs/advanced_av/av_visual_per_child` | 0.906 | 0.764 | 0.898 |
| `ensemble_runs/advanced/pair_disagreement` | 0.902 | 0.824 | 0.898 |
| `ensemble_runs/no_metadata_stack` | 0.909 | 0.806 | 0.897 |
| `ensemble_runs/advanced/topk_systems` | 0.899 | 0.806 | 0.897 |
| `ensemble_runs/advanced/rank_stacker` | 0.896 | 0.773 | 0.896 |
| `ensemble_runs/advanced/isotonic_weighted` | 0.904 | 0.752 | 0.894 |
| `ensemble_runs/advanced/blend_topk` | 0.904 | 0.752 | 0.894 |
| `ensemble_runs/advanced/per_timepoint` | 0.901 | 0.777 | 0.894 |
| `ensemble_runs/advanced/cv_stacked_then_offset` | 0.909 | 0.791 | 0.893 |
| `ensemble_runs/no_metadata_stack_audio` | 0.896 | 0.776 | 0.893 |
| `ensemble_runs/advanced/confidence_weighted` | 0.909 | 0.764 | 0.893 |
| `baseline_results_seen_child/fused_attn_unfreeze2_whisper_large_kfold3_f0` | 0.904 | 0.770 | 0.893 |

## Per-timepoint posthoc breakdown (full sort by combined AUROC)

| System | 14m AUROC | 14m BA | 14m n | 36m AUROC | 36m BA | 36m n | Δ AUROC | flagged |
|---|---|---|---|---|---|---|---|---|
| `baseline_results_seen_child/fused_attn_unfreeze2_whisper_large` | 0.899 | 0.774 | 233.000 | 0.915 | 0.742 | 208 | 0.016 |  |
| `ensemble_runs/metadata_stack_av` | 0.897 | 0.819 | 233.000 | 0.902 | 0.780 | 208 | 0.005 |  |
| `ensemble_runs/metadata_stack` | 0.898 | 0.807 | 233.000 | 0.900 | 0.791 | 208 | 0.002 |  |
| `ensemble_runs/metadata_stack_av_11sys_backup` | 0.895 | 0.823 | 233.000 | 0.902 | 0.780 | 208 | 0.007 |  |
| `ensemble_runs/advanced_av/av_per_track_added` | 0.896 | 0.812 | 233.000 | 0.895 | 0.794 | 208 | -0.001 |  |
| `ensemble_runs/advanced_av/av_pure_visual` | 0.897 | 0.816 | 233.000 | 0.894 | 0.777 | 208 | -0.003 |  |
| `pseudo_frame/results/whisper_pseudo_frame_kfold3_f2` | 0.880 | 0.661 | 356.000 | 0.929 | 0.646 | 307 | 0.049 |  |
| `ensemble_runs/advanced_av/av_eligibility_only` | 0.897 | 0.817 | 233.000 | 0.889 | 0.771 | 208 | -0.008 |  |
| `ensemble_runs/advanced_av/audio_per_child` | 0.897 | 0.771 | 233.000 | 0.889 | 0.737 | 208 | -0.008 |  |
| `ensemble_runs/advanced/per_child_offset` | 0.897 | 0.771 | 233.000 | 0.889 | 0.737 | 208 | -0.008 |  |
| `ensemble_runs/metadata_stack_11sys_backup` | 0.893 | 0.785 | 233.000 | 0.894 | 0.788 | 208 | 0.002 |  |
| `baseline_results_seen_child/fused_attn_unfreeze2_whisper_medium_kfold3_f1` | 0.893 | 0.815 | 384.000 | 0.912 | 0.820 | 341 | 0.018 |  |
| `ensemble_runs/advanced/fp_focused` | 0.897 | 0.804 | 233.000 | 0.889 | 0.763 | 208 | -0.007 |  |
| `ensemble_runs/advanced/cv_stacked` | 0.895 | 0.785 | 233.000 | 0.891 | 0.788 | 208 | -0.004 |  |
| `ensemble_runs/advanced/pure` | 0.895 | 0.800 | 233.000 | 0.888 | 0.763 | 208 | -0.008 |  |
| `ensemble_runs/advanced_av/audio_pure` | 0.895 | 0.800 | 233.000 | 0.888 | 0.763 | 208 | -0.008 |  |
| `ensemble_runs/advanced/bagged_stacker` | 0.895 | 0.811 | 233.000 | 0.889 | 0.763 | 208 | -0.006 |  |
| `ensemble_runs/advanced_av/av_full` | 0.896 | 0.778 | 233.000 | 0.886 | 0.726 | 208 | -0.009 |  |
| `ensemble_runs/advanced_av/av_visual_per_child` | 0.896 | 0.775 | 233.000 | 0.885 | 0.726 | 208 | -0.012 |  |
| `ensemble_runs/advanced/pair_disagreement` | 0.887 | 0.816 | 233.000 | 0.897 | 0.803 | 208 | 0.011 |  |
| `ensemble_runs/no_metadata_stack` | 0.894 | 0.815 | 233.000 | 0.887 | 0.763 | 208 | -0.007 |  |
| `ensemble_runs/advanced/topk_systems` | 0.886 | 0.798 | 233.000 | 0.896 | 0.788 | 208 | 0.009 |  |
| `ensemble_runs/advanced/rank_stacker` | 0.888 | 0.764 | 233.000 | 0.893 | 0.763 | 208 | 0.005 |  |
| `ensemble_runs/advanced/isotonic_weighted` | 0.897 | 0.756 | 233.000 | 0.882 | 0.734 | 208 | -0.015 |  |
| `ensemble_runs/advanced/blend_topk` | 0.897 | 0.756 | 233.000 | 0.882 | 0.734 | 208 | -0.015 |  |
| `ensemble_runs/advanced/per_timepoint` | 0.892 | 0.777 | 233.000 | 0.879 | 0.751 | 208 | -0.013 |  |
| `ensemble_runs/advanced/cv_stacked_then_offset` | 0.891 | 0.793 | 233.000 | 0.881 | 0.765 | 208 | -0.010 |  |
| `ensemble_runs/no_metadata_stack_audio` | 0.891 | 0.778 | 233.000 | 0.880 | 0.745 | 208 | -0.012 |  |
| `ensemble_runs/advanced/confidence_weighted` | 0.896 | 0.766 | 233.000 | 0.879 | 0.751 | 208 | -0.016 |  |
| `baseline_results_seen_child/fused_attn_unfreeze2_whisper_large_kfold3_f0` | 0.893 | 0.753 | 421.000 | 0.892 | 0.770 | 373 | -0.001 |  |
| `baseline_results_seen_child/fused_attn_unfreeze2_whisper_medium` | 0.893 | 0.810 | 233.000 | 0.888 | 0.757 | 208 | -0.005 |  |
| `mil/mil_results/short_voc_head` | 0.888 | 0.765 | 233.000 | 0.887 | 0.765 | 208 | -0.001 |  |
| `ensemble_runs/advanced_av/av_pure_visual_motion` | 0.886 | 0.792 | 233.000 | 0.887 | 0.762 | 208 | 0.001 |  |
| `mil/mil_results/multi_child_suppressor` | 0.888 | 0.765 | 233.000 | 0.887 | 0.765 | 208 | -0.001 |  |
| `evaluation/cross_child_vtc_kchi_role_only` | 0.878 | 0.795 | 274.000 | 0.903 | 0.854 | 205 | 0.026 |  |
| `pseudo_frame/results/whisper_pseudo_frame_kfold3_f0` | 0.881 | 0.526 | 421.000 | 0.892 | 0.507 | 373 | 0.011 |  |
| `ensemble_runs/advanced/calibrated_mean` | 0.898 | 0.753 | 233.000 | 0.859 | 0.740 | 208 | -0.039 |  |
| `baseline_results_seen_child/fused_attn_unfreeze2` | 0.901 | 0.775 | 233.000 | 0.862 | 0.740 | 208 | -0.039 |  |
| `ensemble_runs/advanced/mean` | 0.900 | 0.814 | 233.000 | 0.848 | 0.748 | 208 | -0.052 | **FLAG** |
| `baseline_results_seen_child/fused_attn_unfreeze2_kfold3_f1` | 0.884 | 0.789 | 384.000 | 0.878 | 0.730 | 341 | -0.006 |  |
| `mil/mil_results/whisper_mil_tsmil_concat_kfold3_f2` | 0.855 | 0.730 | 351.000 | 0.903 | 0.751 | 306 | 0.048 |  |
| `pseudo_frame/results/whisper_pseudo_frame` | 0.870 | 0.550 | 233.000 | 0.879 | 0.554 | 208 | 0.008 |  |
| `ensemble_runs/cross_child_best_audio_mil_with_clap` | 0.846 | 0.705 | 274.000 | 0.923 | 0.699 | 205 | 0.077 | **FLAG** |
| `baseline_results_seen_child/fused_attn_unfreeze2_whisper_large_kfold3_f2` | 0.877 | 0.764 | 356.000 | 0.888 | 0.704 | 307 | 0.011 |  |
| `ensemble_runs/cross_child_best_audio_mil` | 0.840 | 0.650 | 274.000 | 0.926 | 0.641 | 205 | 0.086 | **FLAG** |
| `mil/mil_results/whisper_mil_cross_child` | 0.844 | 0.665 | 274.000 | 0.914 | 0.663 | 205 | 0.070 | **FLAG** |
| `mil/mil_results/whisper_mil/age_stratified/34_38m` | — | — | — | 0.875 | 0.711 | 207 | — |  |
| `mil/mil_results/whisper_mil_cross_child_synth_cap0p5x` | 0.854 | 0.680 | 274.000 | 0.896 | 0.686 | 205 | 0.043 |  |
| `mil/mil_results/whisper_medium_mil_kfold3_f2` | 0.839 | 0.699 | 356.000 | 0.916 | 0.730 | 307 | 0.076 | **FLAG** |
| `mil/mil_results/whisper_medium_mil_kfold3_f0` | 0.859 | 0.766 | 421.000 | 0.894 | 0.766 | 373 | 0.035 |  |
| `mil/mil_results/whisper_medium_mil` | 0.866 | 0.771 | 233.000 | 0.872 | 0.768 | 208 | 0.006 |  |
| `pseudo_frame/results/speaker_informed_asd` | 0.864 | 0.719 | 233.000 | 0.880 | 0.740 | 208 | 0.016 |  |
| `baseline_results_seen_child/fused_attn` | 0.882 | 0.731 | 233.000 | 0.838 | 0.696 | 208 | -0.044 |  |
| `mil/mil_results/whisper_mil_tsmil_concat` | 0.861 | 0.733 | 229.000 | 0.863 | 0.647 | 207 | 0.002 |  |
| `mil/mil_results/whisper_large_mil` | 0.860 | 0.675 | 233.000 | 0.861 | 0.620 | 208 | 0.001 |  |
| `mil/mil_results/whisper_mil_cross_child_kfold3_f2` | 0.822 | 0.729 | 370.000 | 0.913 | 0.826 | 315 | 0.091 | **FLAG** |
| `baseline_results_seen_child/whisper_attn_lw` | 0.859 | 0.728 | 233.000 | 0.848 | 0.680 | 208 | -0.011 |  |
| `baseline_results_seen_child/whisper_attn` | 0.868 | 0.747 | 233.000 | 0.834 | 0.688 | 208 | -0.034 |  |
| `baseline_results_seen_child/whisper_attn_ptt` | 0.871 | 0.718 | 233.000 | 0.830 | 0.713 | 208 | -0.041 |  |
| `mil/mil_results/whisper_mil_kfold3_f2` | 0.827 | 0.704 | 356.000 | 0.904 | 0.717 | 307 | 0.076 | **FLAG** |
| `mil/mil_results/whisper_medium_mil_kfold3_f1` | 0.853 | 0.747 | 384.000 | 0.873 | 0.762 | 341 | 0.020 |  |
| `pseudo_frame/results/whisper_pseudo_frame_kfold3_f1` | 0.852 | 0.500 | 384.000 | 0.871 | 0.500 | 341 | 0.019 |  |
| `mil/mil_results/whisper_mil_kfold3_f1` | 0.855 | 0.743 | 384.000 | 0.866 | 0.736 | 341 | 0.011 |  |
| `baseline_results_seen_child/whisper_mean` | 0.867 | 0.773 | 233.000 | 0.830 | 0.679 | 208 | -0.038 |  |
| `baseline_results_seen_child/whisper_attn_unfreeze2` | 0.863 | 0.705 | 233.000 | 0.852 | 0.688 | 208 | -0.011 |  |
| `baseline_results_seen_child/whisper_attn_aug` | 0.862 | 0.732 | 233.000 | 0.831 | 0.654 | 208 | -0.031 |  |
| `evaluation/cross_child_babar_role_only` | 0.850 | 0.775 | 274.000 | 0.900 | 0.850 | 205 | 0.049 |  |
| `mil/mil_results/whisper_mil_lr1e-03_seed1` | 0.833 | 0.709 | 233.000 | 0.872 | 0.637 | 208 | 0.039 |  |
| `baselines/baseline_results/fused_attn_lw` | 0.841 | 0.683 | 274.000 | 0.889 | 0.657 | 205 | 0.048 |  |
| `mil/mil_results/seg_conditioned_mil` | 0.836 | 0.753 | 233.000 | 0.874 | 0.719 | 208 | 0.038 |  |
| `baselines/baseline_results/whisper_mean` | 0.844 | 0.746 | 274.000 | 0.894 | 0.699 | 205 | 0.050 | **FLAG** |
| `baseline_results_seen_child/whisper_attn_aug_ptt` | 0.858 | 0.715 | 233.000 | 0.832 | 0.688 | 208 | -0.026 |  |
| `baselines/baseline_results/whisper_attn_ptt` | 0.837 | 0.746 | 274.000 | 0.899 | 0.605 | 205 | 0.062 | **FLAG** |
| `mil/mil_results/whisper_mil_lr1e-03_seed2` | 0.835 | 0.729 | 233.000 | 0.866 | 0.722 | 208 | 0.030 |  |
| `mil/mil_results/whisper_mil_hardneg_synth` | 0.851 | 0.767 | 233.000 | 0.842 | 0.691 | 208 | -0.009 |  |
| `mil/mil_results/whisper_mil` | 0.828 | 0.738 | 233.000 | 0.876 | 0.711 | 208 | 0.048 |  |
| `baselines/baseline_results/whisper_attn_aug` | 0.841 | 0.711 | 274.000 | 0.888 | 0.719 | 205 | 0.047 |  |
| `mil/mil_results/whisper_mil_cross_child_kfold3_f1` | 0.841 | 0.681 | 306.000 | 0.851 | 0.637 | 387 | 0.010 |  |
| `mil/mil_results/whisper_mil_kfold3_f0` | 0.823 | 0.586 | 421.000 | 0.878 | 0.579 | 373 | 0.054 | **FLAG** |
| `baselines/baseline_results/whisper_attn` | 0.830 | 0.723 | 274.000 | 0.896 | 0.729 | 205 | 0.067 | **FLAG** |
| `mil/mil_results/whisper_mil_tsmil_concat_kfold3_f1` | 0.855 | 0.741 | 378.000 | 0.840 | 0.628 | 339 | -0.015 |  |
| `baseline_results_seen_child/fused_attn_unfreeze2_whisper_medium_kfold3_f0` | 0.841 | 0.785 | 421.000 | 0.855 | 0.777 | 373 | 0.014 |  |
| `babar_ecapa_enrollment_runs_kfold3_f0` | 0.838 | 0.814 | 421.000 | 0.863 | 0.812 | 373 | 0.025 |  |
| `vtc_kchi_ecapa_enrollment_runs_kfold3_f0` | 0.838 | 0.814 | 421.000 | 0.863 | 0.812 | 373 | 0.025 |  |
| `usc_sail_ecapa_enrollment_runs_kfold3_f0` | 0.833 | 0.841 | 421.000 | 0.877 | 0.816 | 373 | 0.044 |  |
| `baselines/baseline_results/whisper_attn_aug_ptt` | 0.844 | 0.739 | 274.000 | 0.874 | 0.562 | 205 | 0.030 |  |
| `mil/mil_results/whisper_mil_tsmil_concat_kfold3_f0` | 0.831 | 0.681 | 415.000 | 0.858 | 0.546 | 371 | 0.028 |  |
| `mil/mil_results/whisper_mil_hardneg_synth_v3` | 0.838 | 0.726 | 233.000 | 0.838 | 0.668 | 208 | 0.001 |  |
| `mil/mil_results/whisper_mil_cross_child_kfold3_f0` | 0.825 | 0.706 | 529.000 | 0.848 | 0.677 | 383 | 0.023 |  |
| `mil/mil_results/whisper_mil_acmil_max` | 0.813 | 0.754 | 233.000 | 0.875 | 0.685 | 208 | 0.062 | **FLAG** |
| `babar_ecapa_enrollment_runs_kfold3_f2` | 0.822 | 0.792 | 356.000 | 0.863 | 0.845 | 307 | 0.041 |  |
| `vtc_kchi_ecapa_enrollment_runs_kfold3_f2` | 0.822 | 0.792 | 356.000 | 0.863 | 0.845 | 307 | 0.041 |  |
| `usc_sail_ecapa_enrollment_runs_kfold3_f1` | 0.821 | 0.804 | 384.000 | 0.859 | 0.819 | 341 | 0.039 |  |
| `mil/mil_results/whisper_mil_hardneg_synth_v4` | 0.838 | 0.711 | 233.000 | 0.829 | 0.625 | 208 | -0.009 |  |
| `pseudo_frame/results/wavlm_pseudo_frame_kfold3_f0` | 0.814 | 0.529 | 421.000 | 0.863 | 0.533 | 373 | 0.049 |  |
| `baseline_results_seen_child/fused_attn_unfreeze2_whisper_medium_kfold3_f2` | 0.820 | 0.754 | 356.000 | 0.852 | 0.813 | 307 | 0.032 |  |
| `pseudo_frame/results/wavlm_pseudo_frame` | 0.802 | 0.536 | 233.000 | 0.863 | 0.540 | 208 | 0.061 | **FLAG** |
| `baselines/baseline_results/fused_attn` | 0.831 | 0.686 | 274.000 | 0.844 | 0.654 | 205 | 0.012 |  |
| `mil/mil_results/whisper_mil/age_stratified/12_16m` | 0.828 | 0.738 | 233.000 | — | 1.000 | 1 | — |  |
| `baselines/baseline_results/whisper_attn_unfreeze2` | 0.803 | 0.696 | 274.000 | 0.875 | 0.673 | 205 | 0.072 | **FLAG** |
| `pyannote/babar_age_stratified/34_38m/34_38m` | — | — | — | 0.827 | 0.778 | 207 | — |  |
| `pyannote/vtc_kchi_age_stratified/34_38m/34_38m` | — | — | — | 0.827 | 0.778 | 207 | — |  |
| `vtc_kchi_ecapa_enrollment_runs_kfold3_f1` | 0.823 | 0.783 | 384.000 | 0.836 | 0.800 | 341 | 0.013 |  |
| `babar_ecapa_enrollment_runs_kfold3_f1` | 0.823 | 0.783 | 384.000 | 0.836 | 0.800 | 341 | 0.013 |  |
| `babar_ecapa_enrollment_runs` | 0.825 | 0.810 | 233.000 | 0.828 | 0.779 | 208 | 0.004 |  |
| `synth_results/augmentation_experiments/default_14_18mo/ratio_0x` | 0.825 | 0.810 | 233.000 | 0.828 | 0.779 | 208 | 0.004 |  |
| `vtc_kchi_ecapa_enrollment_runs` | 0.825 | 0.810 | 233.000 | 0.828 | 0.779 | 208 | 0.004 |  |
| `synth_results/augmentation_experiments/default_14_18mo/ratio_2x` | 0.825 | 0.810 | 233.000 | 0.828 | 0.779 | 208 | 0.004 |  |
| `mil/mil_results/whisper_mil_voiceconv_synth_hardneg` | 0.819 | 0.682 | 233.000 | 0.828 | 0.660 | 208 | 0.009 |  |
| `synth_results/augmentation_experiments/default_14_18mo/ratio_10x` | 0.825 | 0.810 | 233.000 | 0.828 | 0.779 | 208 | 0.004 |  |
| `synth_results/augmentation_experiments/default_14_18mo/ratio_1x` | 0.825 | 0.810 | 233.000 | 0.828 | 0.779 | 208 | 0.004 |  |
| `synth_results/augmentation_experiments/default_14_18mo/ratio_5x` | 0.825 | 0.810 | 233.000 | 0.828 | 0.779 | 208 | 0.004 |  |
| `synth_results/augmentation_experiments/default_14_18mo/ratio_0.5x` | 0.825 | 0.810 | 233.000 | 0.828 | 0.779 | 208 | 0.004 |  |
| `pyannote/vtc_kchi_age_stratified/12_16m/12_16m` | 0.825 | 0.796 | 233.000 | — | 1.000 | 1 | — |  |
| `pyannote/babar_age_stratified/12_16m/12_16m` | 0.825 | 0.796 | 233.000 | — | 1.000 | 1 | — |  |
| `usc_sail_ecapa_enrollment_runs_kfold3_f2` | 0.810 | 0.766 | 356.000 | 0.836 | 0.844 | 307 | 0.027 |  |
| `vtc_ecapa_enrollment_runs_kfold3_f0` | 0.815 | 0.792 | 421.000 | 0.823 | 0.741 | 373 | 0.007 |  |
| `mil/mil_results/whisper_mil_hardneg_synth_v1` | 0.818 | 0.656 | 233.000 | 0.814 | 0.594 | 208 | -0.003 |  |
| `mil/mil_results/whisper_mil_lr3e-04_seed2` | 0.797 | 0.713 | 233.000 | 0.845 | 0.660 | 208 | 0.048 |  |
| `baselines/audio_llm_baseline_runs/qwen25_omni_7b_cross_child` | 0.786 | 0.613 | 274.000 | 0.869 | 0.592 | 205 | 0.083 | **FLAG** |
| `evaluation/cross_child_vtc_role_only` | 0.779 | 0.746 | 274.000 | 0.876 | 0.827 | 205 | 0.097 | **FLAG** |
| `mil/mil_results/whisper_mil_voiceconv_synth_full` | 0.784 | 0.661 | 233.000 | 0.855 | 0.640 | 208 | 0.070 | **FLAG** |
| `baselines/baseline_results/whisper_stats_lw` | 0.784 | 0.613 | 274.000 | 0.877 | 0.585 | 205 | 0.093 | **FLAG** |
| `mil/mil_results/whisper_mil_hardneg` | 0.814 | 0.714 | 233.000 | 0.802 | 0.642 | 208 | -0.012 |  |
| `baselines/baseline_results/wavlm_attn_lw` | 0.808 | 0.638 | 274.000 | 0.850 | 0.578 | 205 | 0.042 |  |
| `babar_ecapa_child_enrollment_runs` | 0.817 | 0.800 | 233.000 | 0.821 | 0.756 | 208 | 0.004 |  |
| `mil/mil_results/whisper_mil_voiceconv_synth_half` | 0.779 | 0.686 | 233.000 | 0.860 | 0.694 | 208 | 0.081 | **FLAG** |
| `mil/mil_results/whisper_mil_acmil_topk` | 0.800 | 0.727 | 233.000 | 0.826 | 0.634 | 208 | 0.026 |  |
| `mil/mil_results/seg_mil/babar_vtc_exp_softmax_pool` | 0.826 | 0.684 | 233.000 | 0.806 | 0.597 | 208 | -0.020 |  |
| `mil/mil_results/seg_mil/babar_vtc_dsmil` | 0.830 | 0.500 | 233.000 | 0.789 | 0.500 | 208 | -0.041 |  |
| `mil/mil_results/hubert_large_mil_layersum` | 0.792 | 0.620 | 233.000 | 0.826 | 0.569 | 208 | 0.034 |  |
| `mil/mil_results/seg_mil/babar_vtc_auto_pool` | 0.821 | 0.680 | 233.000 | 0.805 | 0.611 | 208 | -0.015 |  |
| `pyannote/babar_augmented/12_16m_ratio1.0` | 0.812 | 0.795 | 233.000 | — | 1.000 | 1 | — |  |
| `vtc_ecapa_enrollment_runs` | 0.805 | 0.738 | 233.000 | 0.797 | 0.745 | 208 | -0.008 |  |
| `mil/mil_results/whisper_mil_lr3e-04_seed42` | 0.786 | 0.672 | 233.000 | 0.836 | 0.577 | 208 | 0.050 |  |
| `baseline_results_seen_child/whisper_stats_lw` | 0.813 | 0.712 | 233.000 | 0.770 | 0.631 | 208 | -0.042 |  |
| `mil/mil_results/whisper_mil_cross_child_synth_cap1x` | 0.767 | 0.660 | 274.000 | 0.872 | 0.699 | 205 | 0.106 | **FLAG** |
| `mil/mil_results/seg_mil/babar_vtc_gated_attention` | 0.832 | 0.680 | 233.000 | 0.767 | 0.580 | 208 | -0.064 | **FLAG** |
| `baselines/baseline_results/wavlm_attn` | 0.798 | 0.676 | 274.000 | 0.841 | 0.660 | 205 | 0.043 |  |
| `vtc_ecapa_enrollment_runs_kfold3_f1` | 0.790 | 0.761 | 384.000 | 0.825 | 0.768 | 341 | 0.035 |  |
| `pseudo_frame/results/wavlm_pseudo_frame_synth` | 0.788 | 0.553 | 233.000 | 0.823 | 0.517 | 208 | 0.036 |  |
| `pyannote/vtc_age_stratified/12_16m/12_16m` | 0.805 | 0.738 | 233.000 | — | 1.000 | 1 | — |  |
| `baseline_results_seen_child/fused_attn_unfreeze2_whisper_large_kfold3_f1` | 0.813 | 0.788 | 384.000 | 0.787 | 0.744 | 341 | -0.026 |  |
| `pseudo_frame/results/speaker_informed_asd_per_track` | 0.793 | 0.600 | 233.000 | 0.823 | 0.674 | 208 | 0.029 |  |
| `baseline_results_seen_child/wavlm_attn_lw` | 0.792 | 0.686 | 233.000 | 0.793 | 0.603 | 208 | 0.000 |  |
| `mil/mil_results/seg_mil/vbx_max` | 0.796 | 0.655 | 233.000 | 0.810 | 0.565 | 208 | 0.014 |  |
| `vtc_ecapa_enrollment_runs_kfold3_f2` | 0.776 | 0.737 | 356.000 | 0.819 | 0.782 | 307 | 0.043 |  |
| `baselines/baseline_results/fused_attn_unfreeze2` | 0.784 | 0.763 | 274.000 | 0.824 | 0.778 | 205 | 0.040 |  |
| `baseline_results_seen_child/fused_attn_unfreeze2_kfold3_f2` | 0.797 | 0.754 | 356.000 | 0.796 | 0.766 | 307 | -0.001 |  |
| `baselines/baseline_results/wavlm_stats_lw` | 0.766 | 0.508 | 274.000 | 0.852 | 0.539 | 205 | 0.086 | **FLAG** |
| `pyannote/vtc_age_stratified/34_38m/34_38m` | — | — | — | 0.796 | 0.745 | 207 | — |  |
| `baselines/baseline_results/wavlm_mean` | 0.782 | 0.605 | 274.000 | 0.842 | 0.575 | 205 | 0.060 | **FLAG** |
| `mil/mil_results/seg_mil/babar_vtc_gmap` | 0.813 | 0.652 | 233.000 | 0.760 | 0.574 | 208 | -0.053 | **FLAG** |
| `mil/mil_results/seg_mil/babar_vtc_mean` | 0.805 | 0.680 | 233.000 | 0.777 | 0.597 | 208 | -0.028 |  |
| `mil/mil_results/seg_mil/vbx_dsmil` | 0.782 | 0.653 | 233.000 | 0.798 | 0.565 | 208 | 0.016 |  |
| `baseline_results_seen_child/wavlm_attn` | 0.784 | 0.688 | 233.000 | 0.790 | 0.548 | 208 | 0.005 |  |
| `mil/mil_results/seg_mil/vbx_attention` | 0.781 | 0.660 | 233.000 | 0.799 | 0.591 | 208 | 0.018 |  |
| `mil/mil_results/seg_mil/babar_vtc_attention` | 0.813 | 0.648 | 233.000 | 0.748 | 0.574 | 208 | -0.064 | **FLAG** |
| `mil/mil_results/seg_mil/vbx_gated_attention` | 0.781 | 0.653 | 233.000 | 0.794 | 0.580 | 208 | 0.014 |  |
| `baseline_results_seen_child/fused_attn_unfreeze2_kfold3_f0` | 0.784 | 0.724 | 421.000 | 0.791 | 0.723 | 373 | 0.008 |  |
| `mil/mil_results/whisper_mil_acmil_max_kfold3_f2` | 0.737 | 0.617 | 356.000 | 0.852 | 0.646 | 307 | 0.115 | **FLAG** |
| `mil/mil_results/seg_mil/babar_vtc_top_k` | 0.802 | 0.651 | 233.000 | 0.747 | 0.574 | 208 | -0.055 | **FLAG** |
| `baseline_results_seen_child/wavlm_stats_lw` | 0.800 | 0.637 | 233.000 | 0.727 | 0.520 | 208 | -0.073 | **FLAG** |
| `pseudo_frame/results/wavlm_pseudo_frame_kfold3_f2` | 0.757 | 0.510 | 356.000 | 0.812 | 0.535 | 307 | 0.055 | **FLAG** |
| `mil/mil_results/seg_mil/vbx_exp_softmax_pool` | 0.780 | 0.689 | 233.000 | 0.778 | 0.577 | 208 | -0.002 |  |
| `mil/mil_results/seg_mil/vbx_gmap` | 0.777 | 0.624 | 233.000 | 0.792 | 0.580 | 208 | 0.015 |  |
| `mil/mil_results/whisper_mil_cross_child_synth_v4` | 0.722 | 0.615 | 274.000 | 0.857 | 0.582 | 205 | 0.135 | **FLAG** |
| `baselines/audio_llm_baseline_runs/qwen2_audio_7b_cross_child` | 0.772 | 0.550 | 274.000 | 0.786 | 0.565 | 205 | 0.015 |  |
| `baselines/audio_llm_baseline_runs/qwen25_omni_7b_synth_2shot` | 0.762 | 0.652 | 233.000 | 0.790 | 0.577 | 208 | 0.028 |  |
| `baseline_results_seen_child/wavlm_mean` | 0.768 | 0.630 | 233.000 | 0.770 | 0.563 | 208 | 0.002 |  |
| `mil/mil_results/wavlm_mil/age_stratified/12_16m` | 0.776 | 0.628 | 233.000 | — | 1.000 | 1 | — |  |
| `mil/mil_results/whisper_tiny_mil` | 0.788 | 0.676 | 233.000 | 0.744 | 0.563 | 208 | -0.044 |  |
| `mil/mil_results/seg_mil/vbx_auto_pool` | 0.769 | 0.628 | 233.000 | 0.773 | 0.565 | 208 | 0.004 |  |
| `mil/mil_results/wavlm_mil_acmil_topk` | 0.767 | 0.657 | 233.000 | 0.774 | 0.594 | 208 | 0.007 |  |
| `mil/mil_results/seg_mil/babar_vtc_max` | 0.802 | 0.536 | 233.000 | 0.707 | 0.517 | 208 | -0.095 | **FLAG** |
| `mil/mil_results/wavlm_mil` | 0.776 | 0.628 | 233.000 | 0.759 | 0.600 | 208 | -0.017 |  |
| `mil/mil_results/seg_mil/vbx_top_k` | 0.764 | 0.613 | 233.000 | 0.771 | 0.551 | 208 | 0.007 |  |
| `baselines/audio_llm_baseline_runs/qwen25_omni_7b` | 0.739 | 0.668 | 233.000 | 0.793 | 0.591 | 208 | 0.054 | **FLAG** |
| `baselines/audio_llm_baseline_runs/qwen25_omni_7b_5shot` | 0.739 | 0.668 | 233.000 | 0.793 | 0.591 | 208 | 0.054 | **FLAG** |
| `baselines/audio_llm_baseline_runs/qwen25_omni_7b_2shot` | 0.739 | 0.665 | 233.000 | 0.795 | 0.591 | 208 | 0.057 | **FLAG** |
| `pseudo_frame/results/wavlm_pseudo_frame_kfold3_f1` | 0.760 | 0.508 | 384.000 | 0.778 | 0.507 | 341 | 0.018 |  |
| `mil/mil_results/seg_mil/vbx_mean` | 0.757 | 0.621 | 233.000 | 0.766 | 0.565 | 208 | 0.008 |  |
| `pseudo_frame/results/wavlm_pseudo_frame_synth_v1` | 0.734 | 0.578 | 233.000 | 0.797 | 0.540 | 208 | 0.063 | **FLAG** |
| `mil/mil_results/wavlm_mil_acmil_max` | 0.759 | 0.639 | 233.000 | 0.752 | 0.625 | 208 | -0.007 |  |
| `mil/mil_results/wavlm_mil/age_stratified/34_38m` | — | — | — | 0.758 | 0.600 | 207 | — |  |
| `mil/mil_results/whisper_mil_acmil_max_kfold3_f1` | 0.718 | 0.608 | 384.000 | 0.781 | 0.592 | 341 | 0.063 | **FLAG** |
| `mil/mil_results/hubert_large_mil` | 0.766 | 0.614 | 233.000 | 0.703 | 0.560 | 208 | -0.062 | **FLAG** |
| `mil/mil_results/seg_mil/babar_vtc_transformer` | 0.761 | 0.682 | 233.000 | 0.691 | 0.671 | 208 | -0.070 | **FLAG** |
| `pyannote/babar_augmented/34_38m_ratio1.0` | — | — | — | 0.745 | 0.766 | 207 | — |  |
| `pseudo_frame/results/audio2video_distilled` | 0.730 | 0.557 | 233.000 | 0.748 | 0.534 | 208 | 0.018 |  |
| `mil/mil_results/whisper_mil_cross_child_synth_v3` | 0.731 | 0.538 | 274.000 | 0.752 | 0.575 | 205 | 0.021 |  |
| `mil/mil_results/wav2vec2_large_mil` | 0.741 | 0.638 | 233.000 | 0.716 | 0.586 | 208 | -0.025 |  |
| `mil/mil_results/wavlm_mil_acmil_gated` | 0.712 | 0.566 | 233.000 | 0.774 | 0.603 | 208 | 0.062 | **FLAG** |
| `mil/mil_results/whisper_mil_acmil_gated` | 0.747 | 0.607 | 233.000 | 0.691 | 0.531 | 208 | -0.057 | **FLAG** |
| `pyannote/pyannote_age_stratified/12_16m/12_16m` | 0.736 | 0.686 | 233.000 | — | 1.000 | 1 | — |  |
| `mil/mil_results/whisper_mil_acmil` | 0.746 | 0.656 | 233.000 | 0.694 | 0.585 | 208 | -0.052 | **FLAG** |
| `mil/mil_results/wavlm_mil_acmil` | 0.717 | 0.558 | 233.000 | 0.761 | 0.563 | 208 | 0.044 |  |
| `ensemble_runs/metadata_router_learned` | 0.724 | 0.635 | 233.000 | 0.727 | 0.605 | 208 | 0.004 |  |
| `mil/mil_results/whisper_mil_lr3e-04_seed1` | 0.740 | 0.585 | 233.000 | 0.694 | 0.554 | 208 | -0.046 |  |
| `mil/mil_results/wavlm_mil_tsmil_concat` | 0.699 | 0.601 | 229.000 | 0.722 | 0.538 | 207 | 0.023 |  |
| `mil/mil_results/seg_mil/vbx_transformer` | 0.736 | 0.598 | 233.000 | 0.718 | 0.554 | 208 | -0.018 |  |
| `mil/mil_results/wavlm_mil_tsmil_film` | 0.698 | 0.557 | 229.000 | 0.721 | 0.527 | 207 | 0.023 |  |
| `mil/mil_results/seg_mil/pyannote_max` | 0.728 | 0.637 | 233.000 | 0.711 | 0.548 | 208 | -0.016 |  |
| `baselines/audio_llm_baseline_runs/qwen2_audio_7b_2shot` | 0.719 | 0.624 | 233.000 | 0.720 | 0.591 | 208 | 0.001 |  |
| `baselines/audio_llm_baseline_runs/qwen2_audio_7b` | 0.719 | 0.624 | 233.000 | 0.720 | 0.591 | 208 | 0.001 |  |
| `baselines/audio_llm_baseline_runs/qwen2_audio_7b_5shot` | 0.719 | 0.624 | 233.000 | 0.720 | 0.591 | 208 | 0.001 |  |
| `vbx_ecapa_enrollment_runs_kfold3_f2` | 0.688 | 0.605 | 356.000 | 0.739 | 0.607 | 307 | 0.051 | **FLAG** |
| `baselines/audio_llm_baseline_runs/qwen2_audio_7b_synth_2shot` | 0.704 | 0.549 | 233.000 | 0.709 | 0.583 | 208 | 0.005 |  |
| `baselines/audio_llm_baseline_runs/qwen2_audio_7b_synth_2shot_v1` | 0.704 | 0.549 | 233.000 | 0.709 | 0.583 | 208 | 0.005 |  |
| `vbx_ecapa_enrollment_runs_kfold3_f1` | 0.700 | 0.586 | 384.000 | 0.709 | 0.589 | 341 | 0.008 |  |
| `baselines/audio_llm_baseline_runs/qwen25_omni_7b_target_2shot` | 0.734 | 0.581 | 233.000 | 0.655 | 0.520 | 208 | -0.080 | **FLAG** |
| `baselines/audio_llm_baseline_runs/qwen25_omni_7b_target_5shot` | 0.734 | 0.581 | 233.000 | 0.655 | 0.520 | 208 | -0.080 | **FLAG** |
| `mil/mil_results/wavlm_mil_hardneg_synth` | 0.677 | 0.500 | 233.000 | 0.744 | 0.494 | 208 | 0.067 | **FLAG** |
| `mil/mil_results/whisper_mil_cross_child_synth` | 0.695 | 0.530 | 274.000 | 0.729 | 0.582 | 205 | 0.035 |  |
| `ensemble_runs/metadata_router_rule` | 0.741 | 0.717 | 233.000 | 0.616 | 0.640 | 208 | -0.125 | **FLAG** |
| `pyannote_ecapa_enrollment_runs_kfold3_f2` | 0.678 | 0.597 | 356.000 | 0.706 | 0.625 | 307 | 0.029 |  |
| `pyannote/vbx_age_stratified/12_16m/12_16m` | 0.704 | 0.623 | 233.000 | — | 1.000 | 1 | — |  |
| `mil/mil_results/whisper_base_mil` | 0.716 | 0.624 | 233.000 | 0.663 | 0.577 | 208 | -0.053 | **FLAG** |
| `mil/mil_results/seg_mil/usc_sail_max` | 0.726 | 0.571 | 233.000 | 0.659 | 0.563 | 208 | -0.067 | **FLAG** |
| `pyannote/usc_sail_age_stratified/34_38m/34_38m` | — | — | — | 0.698 | 0.617 | 207 | — |  |
| `pyannote_ecapa_enrollment_runs_kfold3_f1` | 0.689 | 0.627 | 384.000 | 0.686 | 0.600 | 341 | -0.003 |  |
| `mil/mil_results/wavlm_mil_layersum` | 0.669 | 0.514 | 233.000 | 0.743 | 0.500 | 208 | 0.074 | **FLAG** |
| `vbx_ecapa_enrollment_runs_kfold3_f0` | 0.675 | 0.582 | 421.000 | 0.699 | 0.629 | 373 | 0.024 |  |
| `sortformer_ecapa_enrollment_runs` | 0.699 | 0.644 | 233.000 | 0.639 | 0.568 | 208 | -0.061 | **FLAG** |
| `mil/mil_results/seg_mil/pyannote_gmap` | 0.689 | 0.647 | 233.000 | 0.696 | 0.560 | 208 | 0.007 |  |
| `mil/mil_results/wavlm_mil_cross_child` | 0.678 | 0.500 | 274.000 | 0.731 | 0.529 | 205 | 0.054 | **FLAG** |
| `pseudo_frame/results/wavlm_pseudo_frame_c1distill` | 0.682 | 0.518 | 233.000 | 0.724 | 0.511 | 208 | 0.041 |  |
| `mil/mil_results/seg_mil/pyannote_dsmil` | 0.688 | 0.633 | 233.000 | 0.696 | 0.563 | 208 | 0.008 |  |
| `mil/mil_results/seg_mil/pyannote_gated_attention` | 0.687 | 0.636 | 233.000 | 0.695 | 0.563 | 208 | 0.008 |  |
| `mil/mil_results/seg_mil/pyannote_attention` | 0.686 | 0.644 | 233.000 | 0.690 | 0.571 | 208 | 0.003 |  |
| `sortformer_ecapa_enrollment_runs_kfold3_f2` | 0.627 | 0.580 | 356.000 | 0.750 | 0.661 | 307 | 0.123 | **FLAG** |
| `sortformer_ecapa_enrollment_runs_kfold3_f0` | 0.632 | 0.580 | 421.000 | 0.727 | 0.647 | 373 | 0.095 | **FLAG** |
| `pyannote_ecapa_enrollment_runs_kfold3_f0` | 0.646 | 0.613 | 421.000 | 0.703 | 0.645 | 373 | 0.057 | **FLAG** |
| `pyannote_ecapa_enrollment_runs` | 0.736 | 0.686 | 233.000 | 0.549 | 0.568 | 208 | -0.187 | **FLAG** |
| `vbx_ecapa_enrollment_runs` | 0.704 | 0.646 | 233.000 | 0.599 | 0.537 | 208 | -0.105 | **FLAG** |
| `sortformer_ecapa_enrollment_runs_kfold3_f1` | 0.639 | 0.560 | 384.000 | 0.680 | 0.632 | 341 | 0.041 |  |
| `mil/mil_results/wavlm_mil_tinyvox` | 0.654 | 0.522 | 233.000 | 0.692 | 0.526 | 208 | 0.038 |  |
| `mil/mil_results/whisper_mil_acmil_max_kfold3_f0` | 0.623 | 0.562 | 421.000 | 0.724 | 0.579 | 373 | 0.102 | **FLAG** |
| `mil/mil_results/seg_mil/pyannote_exp_softmax_pool` | 0.676 | 0.644 | 233.000 | 0.661 | 0.591 | 208 | -0.015 |  |
| `joint_asr_diar_ecapa_enrollment_runs` | 0.583 | 0.572 | 233.000 | 0.736 | 0.664 | 208 | 0.153 | **FLAG** |
| `pyannote/pyannote_enrollment_runs` | 0.719 | 0.670 | 233.000 | 0.537 | 0.568 | 208 | -0.182 | **FLAG** |
| `mil/mil_results/whisper_mil_cross_child_synth_cap2x` | 0.641 | 0.530 | 274.000 | 0.697 | 0.562 | 205 | 0.056 | **FLAG** |
| `mil/mil_results/seg_mil_synth/usc_sail_synth_combined_gated_attention` | 0.672 | 0.567 | 233.000 | 0.646 | 0.531 | 208 | -0.026 |  |
| `whisper-modeling/usc_sail_enrollment_runs` | 0.639 | 0.620 | 233.000 | 0.698 | 0.628 | 208 | 0.059 | **FLAG** |
| `mil/mil_results/wavlm_mil_hardneg_synth_v1` | 0.648 | 0.500 | 233.000 | 0.663 | 0.500 | 208 | 0.015 |  |
| `mil/mil_results/wavlm_mil_kfold3_f2` | 0.611 | 0.521 | 356.000 | 0.731 | 0.554 | 307 | 0.120 | **FLAG** |
| `mil/mil_results/wavlm_mil_kfold3_f0` | 0.637 | 0.516 | 421.000 | 0.668 | 0.517 | 373 | 0.031 |  |
| `baselines/baseline_results/whisper_attn_lw` | 0.590 | 0.561 | 274.000 | 0.704 | 0.611 | 205 | 0.114 | **FLAG** |
| `mil/mil_results/wavlm_mil_hardneg` | 0.614 | 0.500 | 233.000 | 0.688 | 0.500 | 208 | 0.074 | **FLAG** |
| `pyannote/usc_sail_age_stratified/12_16m/12_16m` | 0.639 | 0.612 | 233.000 | — | 1.000 | 1 | — |  |
| `mil/mil_results/seg_mil_synth_v1/usc_sail_synth_combined_transformer` | 0.617 | 0.553 | 233.000 | 0.655 | 0.537 | 208 | 0.038 |  |
| `mil/mil_results/seg_mil_synth_v1/usc_sail_synth_combined_gated_attention` | 0.632 | 0.563 | 233.000 | 0.642 | 0.557 | 208 | 0.010 |  |
| `mil/mil_results/whisper_mil_cross_child_synth_v3_2k` | 0.614 | 0.525 | 274.000 | 0.670 | 0.497 | 205 | 0.056 | **FLAG** |
| `mil/mil_results/wavlm_mil_cross_child_synth_v1` | 0.639 | 0.507 | 274.000 | 0.600 | 0.529 | 205 | -0.039 |  |
| `mil/mil_results/wavlm_mil_kfold3_f1` | 0.627 | 0.523 | 384.000 | 0.613 | 0.519 | 341 | -0.014 |  |
| `mil/mil_results/seg_mil/usc_sail_gmap` | 0.635 | 0.553 | 233.000 | 0.576 | 0.537 | 208 | -0.059 | **FLAG** |
| `mil/mil_results/seg_mil/usc_sail_dsmil` | 0.635 | 0.567 | 233.000 | 0.565 | 0.546 | 208 | -0.070 | **FLAG** |
| `mil/mil_results/seg_mil/pyannote_auto_pool` | 0.640 | 0.626 | 233.000 | 0.544 | 0.563 | 208 | -0.096 | **FLAG** |
| `mil/mil_results/seg_mil/pyannote_mean` | 0.639 | 0.626 | 233.000 | 0.543 | 0.563 | 208 | -0.097 | **FLAG** |
| `mil/mil_results/seg_mil/usc_sail_gated_attention` | 0.631 | 0.553 | 233.000 | 0.558 | 0.537 | 208 | -0.073 | **FLAG** |
| `pyannote/vbx_age_stratified/34_38m/34_38m` | — | — | — | 0.599 | 0.560 | 207 | — |  |
| `mil/mil_results/seg_mil/usc_sail_attention` | 0.630 | 0.553 | 233.000 | 0.552 | 0.537 | 208 | -0.077 | **FLAG** |
| `mil/mil_results/whisper_mil_layersum` | 0.604 | 0.514 | 233.000 | 0.543 | 0.500 | 208 | -0.061 | **FLAG** |
| `mil/mil_results/seg_mil_synth/usc_sail_synth_combined_transformer` | 0.568 | 0.553 | 233.000 | 0.632 | 0.537 | 208 | 0.064 | **FLAG** |
| `mil/mil_results/whisper_mil_cross_child_synth_v1` | 0.586 | 0.500 | 274.000 | 0.616 | 0.510 | 205 | 0.030 |  |
| `mil/mil_results/seg_mil/pyannote_top_k` | 0.620 | 0.615 | 233.000 | 0.518 | 0.554 | 208 | -0.102 | **FLAG** |
| `mil/mil_results/whisper_mil_lr1e-04_seed1` | 0.607 | 0.500 | 233.000 | 0.525 | 0.500 | 208 | -0.082 | **FLAG** |
| `mil/mil_results/seg_mil/usc_sail_exp_softmax_pool` | 0.604 | 0.553 | 233.000 | 0.528 | 0.537 | 208 | -0.075 | **FLAG** |
| `mil/mil_results/seg_mil/pyannote_transformer` | 0.585 | 0.615 | 233.000 | 0.544 | 0.554 | 208 | -0.041 |  |
| `video_asd_ecapa_enrollment_runs/talknet_asd` | 0.568 | 0.556 | 233.000 | 0.566 | 0.572 | 208 | -0.002 |  |
| `mil/mil_results/whisper_mil_lr1e-04_seed2` | 0.579 | 0.507 | 233.000 | 0.514 | 0.500 | 208 | -0.065 | **FLAG** |
| `mil/mil_results/wavlm_mil_cross_child_synth` | 0.548 | 0.507 | 274.000 | 0.608 | 0.529 | 205 | 0.060 | **FLAG** |
| `eend_eda_ecapa_enrollment_runs_kfold3_f2` | 0.564 | 0.550 | 356.000 | 0.577 | 0.576 | 307 | 0.013 |  |
| `mil/mil_results/whisper_mil_lr1e-04_seed42` | 0.605 | 0.518 | 233.000 | 0.465 | 0.511 | 208 | -0.141 | **FLAG** |
| `joint_asr_diar_sails_runs` | 0.466 | — | 234.000 | 0.609 | — | 207 | 0.143 | **FLAG** |
| `eend_eda_ecapa_enrollment_runs_kfold3_f0` | 0.543 | 0.560 | 421.000 | 0.553 | 0.581 | 373 | 0.010 |  |
| `pyannote/pyannote_age_stratified/34_38m/34_38m` | — | — | — | 0.550 | 0.570 | 207 | — |  |
| `mil/mil_results/seg_mil/usc_sail_auto_pool` | 0.570 | 0.553 | 233.000 | 0.513 | 0.537 | 208 | -0.057 | **FLAG** |
| `mil/mil_results/wavlm_mil_knnvc` | 0.538 | 0.500 | 233.000 | 0.564 | 0.500 | 208 | 0.026 |  |
| `eend_eda_ecapa_enrollment_runs` | 0.561 | 0.536 | 233.000 | 0.457 | 0.528 | 208 | -0.103 | **FLAG** |
| `mil/mil_results/seg_mil/usc_sail_transformer` | 0.529 | 0.553 | 233.000 | 0.507 | 0.537 | 208 | -0.022 |  |
| `eend_eda_ecapa_enrollment_runs_kfold3_f1` | 0.525 | 0.509 | 384.000 | 0.488 | 0.526 | 341 | -0.037 |  |
| `mil/mil_results/seg_mil/usc_sail_mean` | 0.512 | 0.553 | 233.000 | 0.517 | 0.537 | 208 | 0.005 |  |
| `mil/mil_results/wavlm_mil_child_adapted` | 0.500 | 0.500 | 233.000 | 0.500 | 0.500 | 208 | 0.000 |  |
| `baselines/audio_model_baseline_runs/cohere_transcribe` | 0.500 | 0.500 | 233.000 | 0.500 | 0.500 | 208 | 0.000 |  |
| `baselines/audio_model_baseline_runs/cohere_transcribe_cross_child` | 0.500 | 0.500 | 274.000 | 0.500 | 0.500 | 205 | 0.000 |  |
| `video_asd_ecapa_enrollment_runs/loconet_ecapa` | 0.500 | 0.500 | 233.000 | 0.500 | 0.500 | 208 | 0.000 |  |
| `sortformer_ecapa_enrollment_runs_cross_child` | 0.500 | 0.500 | 274.000 | 0.500 | 0.500 | 205 | 0.000 |  |
| `pyannote_ecapa_enrollment_runs_cross_child` | 0.500 | 0.500 | 274.000 | 0.500 | 0.500 | 205 | 0.000 |  |
| `vtc_kchi_ecapa_enrollment_runs_cross_child` | 0.500 | 0.500 | 274.000 | 0.500 | 0.500 | 205 | 0.000 |  |
| `vtc_ecapa_enrollment_runs_cross_child` | 0.500 | 0.500 | 274.000 | 0.500 | 0.500 | 205 | 0.000 |  |
| `eend_eda_ecapa_enrollment_runs_cross_child` | 0.500 | 0.500 | 274.000 | 0.500 | 0.500 | 205 | 0.000 |  |
| `mil/mil_results/seg_mil/usc_sail_top_k` | 0.538 | 0.553 | 233.000 | 0.447 | 0.537 | 208 | -0.090 | **FLAG** |
| `baselines/audio_model_baseline_runs/granite_speech_1b_cross_child` | 0.500 | 0.500 | 274.000 | 0.430 | 0.500 | 205 | -0.070 | **FLAG** |
| `baselines/audio_model_baseline_runs/granite_speech_1b` | 0.495 | 0.504 | 233.000 | 0.362 | 0.500 | 208 | -0.133 | **FLAG** |
| `mil/mil_results/seg_mil/vbx_noisy_or` | 0.354 | 0.500 | 233.000 | 0.396 | 0.500 | 208 | 0.042 |  |
| `mil/mil_results/seg_mil/usc_sail_noisy_or` | 0.313 | 0.500 | 233.000 | 0.328 | 0.500 | 208 | 0.015 |  |
| `mil/mil_results/seg_mil/pyannote_noisy_or` | 0.317 | 0.500 | 233.000 | 0.331 | 0.500 | 208 | 0.014 |  |
| `mil/mil_results/seg_mil/babar_vtc_noisy_or` | 0.306 | 0.500 | 233.000 | 0.326 | 0.500 | 208 | 0.020 |  |

## Flagged systems (|Δ AUROC| > 0.05)

| System | 14m AUROC | 36m AUROC | Δ |
|---|---|---|---|
| `joint_asr_diar_ecapa_enrollment_runs` | 0.583 | 0.736 | +0.153 |
| `joint_asr_diar_sails_runs` | 0.466 | 0.609 | +0.143 |
| `mil/mil_results/whisper_mil_cross_child_synth_v4` | 0.722 | 0.857 | +0.135 |
| `sortformer_ecapa_enrollment_runs_kfold3_f2` | 0.627 | 0.750 | +0.123 |
| `mil/mil_results/wavlm_mil_kfold3_f2` | 0.611 | 0.731 | +0.120 |
| `mil/mil_results/whisper_mil_acmil_max_kfold3_f2` | 0.737 | 0.852 | +0.115 |
| `baselines/baseline_results/whisper_attn_lw` | 0.590 | 0.704 | +0.114 |
| `mil/mil_results/whisper_mil_cross_child_synth_cap1x` | 0.767 | 0.872 | +0.106 |
| `mil/mil_results/whisper_mil_acmil_max_kfold3_f0` | 0.623 | 0.724 | +0.102 |
| `evaluation/cross_child_vtc_role_only` | 0.779 | 0.876 | +0.097 |
| `sortformer_ecapa_enrollment_runs_kfold3_f0` | 0.632 | 0.727 | +0.095 |
| `baselines/baseline_results/whisper_stats_lw` | 0.784 | 0.877 | +0.093 |
| `mil/mil_results/whisper_mil_cross_child_kfold3_f2` | 0.822 | 0.913 | +0.091 |
| `baselines/baseline_results/wavlm_stats_lw` | 0.766 | 0.852 | +0.086 |
| `ensemble_runs/cross_child_best_audio_mil` | 0.840 | 0.926 | +0.086 |
| `baselines/audio_llm_baseline_runs/qwen25_omni_7b_cross_child` | 0.786 | 0.869 | +0.083 |
| `mil/mil_results/whisper_mil_voiceconv_synth_half` | 0.779 | 0.860 | +0.081 |
| `ensemble_runs/cross_child_best_audio_mil_with_clap` | 0.846 | 0.923 | +0.077 |
| `mil/mil_results/whisper_medium_mil_kfold3_f2` | 0.839 | 0.916 | +0.076 |
| `mil/mil_results/whisper_mil_kfold3_f2` | 0.827 | 0.904 | +0.076 |
| `mil/mil_results/wavlm_mil_hardneg` | 0.614 | 0.688 | +0.074 |
| `mil/mil_results/wavlm_mil_layersum` | 0.669 | 0.743 | +0.074 |
| `baselines/baseline_results/whisper_attn_unfreeze2` | 0.803 | 0.875 | +0.072 |
| `mil/mil_results/whisper_mil_voiceconv_synth_full` | 0.784 | 0.855 | +0.070 |
| `mil/mil_results/whisper_mil_cross_child` | 0.844 | 0.914 | +0.070 |
| `mil/mil_results/wavlm_mil_hardneg_synth` | 0.677 | 0.744 | +0.067 |
| `baselines/baseline_results/whisper_attn` | 0.830 | 0.896 | +0.067 |
| `mil/mil_results/seg_mil_synth/usc_sail_synth_combined_transformer` | 0.568 | 0.632 | +0.064 |
| `pseudo_frame/results/wavlm_pseudo_frame_synth_v1` | 0.734 | 0.797 | +0.063 |
| `mil/mil_results/whisper_mil_acmil_max_kfold3_f1` | 0.718 | 0.781 | +0.063 |
| `baselines/baseline_results/whisper_attn_ptt` | 0.837 | 0.899 | +0.062 |
| `mil/mil_results/wavlm_mil_acmil_gated` | 0.712 | 0.774 | +0.062 |
| `mil/mil_results/whisper_mil_acmil_max` | 0.813 | 0.875 | +0.062 |
| `pseudo_frame/results/wavlm_pseudo_frame` | 0.802 | 0.863 | +0.061 |
| `mil/mil_results/wavlm_mil_cross_child_synth` | 0.548 | 0.608 | +0.060 |
| `baselines/baseline_results/wavlm_mean` | 0.782 | 0.842 | +0.060 |
| `whisper-modeling/usc_sail_enrollment_runs` | 0.639 | 0.698 | +0.059 |
| `pyannote_ecapa_enrollment_runs_kfold3_f0` | 0.646 | 0.703 | +0.057 |
| `baselines/audio_llm_baseline_runs/qwen25_omni_7b_2shot` | 0.739 | 0.795 | +0.057 |
| `mil/mil_results/whisper_mil_cross_child_synth_v3_2k` | 0.614 | 0.670 | +0.056 |
| `mil/mil_results/whisper_mil_cross_child_synth_cap2x` | 0.641 | 0.697 | +0.056 |
| `pseudo_frame/results/wavlm_pseudo_frame_kfold3_f2` | 0.757 | 0.812 | +0.055 |
| `mil/mil_results/whisper_mil_kfold3_f0` | 0.823 | 0.878 | +0.054 |
| `baselines/audio_llm_baseline_runs/qwen25_omni_7b_5shot` | 0.739 | 0.793 | +0.054 |
| `baselines/audio_llm_baseline_runs/qwen25_omni_7b` | 0.739 | 0.793 | +0.054 |
| `mil/mil_results/wavlm_mil_cross_child` | 0.678 | 0.731 | +0.054 |
| `vbx_ecapa_enrollment_runs_kfold3_f2` | 0.688 | 0.739 | +0.051 |
| `baselines/baseline_results/whisper_mean` | 0.844 | 0.894 | +0.050 |
| `ensemble_runs/advanced/mean` | 0.900 | 0.848 | -0.052 |
| `mil/mil_results/whisper_mil_acmil` | 0.746 | 0.694 | -0.052 |
| `mil/mil_results/whisper_base_mil` | 0.716 | 0.663 | -0.053 |
| `mil/mil_results/seg_mil/babar_vtc_gmap` | 0.813 | 0.760 | -0.053 |
| `mil/mil_results/seg_mil/babar_vtc_top_k` | 0.802 | 0.747 | -0.055 |
| `mil/mil_results/whisper_mil_acmil_gated` | 0.747 | 0.691 | -0.057 |
| `mil/mil_results/seg_mil/usc_sail_auto_pool` | 0.570 | 0.513 | -0.057 |
| `mil/mil_results/seg_mil/usc_sail_gmap` | 0.635 | 0.576 | -0.059 |
| `sortformer_ecapa_enrollment_runs` | 0.699 | 0.639 | -0.061 |
| `mil/mil_results/whisper_mil_layersum` | 0.604 | 0.543 | -0.061 |
| `mil/mil_results/hubert_large_mil` | 0.766 | 0.703 | -0.062 |
| `mil/mil_results/seg_mil/babar_vtc_gated_attention` | 0.832 | 0.767 | -0.064 |
| `mil/mil_results/seg_mil/babar_vtc_attention` | 0.813 | 0.748 | -0.064 |
| `mil/mil_results/whisper_mil_lr1e-04_seed2` | 0.579 | 0.514 | -0.065 |
| `mil/mil_results/seg_mil/usc_sail_max` | 0.726 | 0.659 | -0.067 |
| `mil/mil_results/seg_mil/babar_vtc_transformer` | 0.761 | 0.691 | -0.070 |
| `mil/mil_results/seg_mil/usc_sail_dsmil` | 0.635 | 0.565 | -0.070 |
| `baselines/audio_model_baseline_runs/granite_speech_1b_cross_child` | 0.500 | 0.430 | -0.070 |
| `mil/mil_results/seg_mil/usc_sail_gated_attention` | 0.631 | 0.558 | -0.073 |
| `baseline_results_seen_child/wavlm_stats_lw` | 0.800 | 0.727 | -0.073 |
| `mil/mil_results/seg_mil/usc_sail_exp_softmax_pool` | 0.604 | 0.528 | -0.075 |
| `mil/mil_results/seg_mil/usc_sail_attention` | 0.630 | 0.552 | -0.077 |
| `baselines/audio_llm_baseline_runs/qwen25_omni_7b_target_5shot` | 0.734 | 0.655 | -0.080 |
| `baselines/audio_llm_baseline_runs/qwen25_omni_7b_target_2shot` | 0.734 | 0.655 | -0.080 |
| `mil/mil_results/whisper_mil_lr1e-04_seed1` | 0.607 | 0.525 | -0.082 |
| `mil/mil_results/seg_mil/usc_sail_top_k` | 0.538 | 0.447 | -0.090 |
| `mil/mil_results/seg_mil/babar_vtc_max` | 0.802 | 0.707 | -0.095 |
| `mil/mil_results/seg_mil/pyannote_auto_pool` | 0.640 | 0.544 | -0.096 |
| `mil/mil_results/seg_mil/pyannote_mean` | 0.639 | 0.543 | -0.097 |
| `mil/mil_results/seg_mil/pyannote_top_k` | 0.620 | 0.518 | -0.102 |
| `eend_eda_ecapa_enrollment_runs` | 0.561 | 0.457 | -0.103 |
| `vbx_ecapa_enrollment_runs` | 0.704 | 0.599 | -0.105 |
| `ensemble_runs/metadata_router_rule` | 0.741 | 0.616 | -0.125 |
| `baselines/audio_model_baseline_runs/granite_speech_1b` | 0.495 | 0.362 | -0.133 |
| `mil/mil_results/whisper_mil_lr1e-04_seed42` | 0.605 | 0.465 | -0.141 |
| `pyannote/pyannote_enrollment_runs` | 0.719 | 0.537 | -0.182 |
| `pyannote_ecapa_enrollment_runs` | 0.736 | 0.549 | -0.187 |

*Interpretation prose to be added by chapter author (FR-022).*
