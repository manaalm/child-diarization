# Error Analysis Summary
## Child Vocalization Detection — MIT EECS MEng Thesis

*Seen-child test split, n=441 clips (335 positive, 106 negative). All analyses use val-tuned thresholds.*

---

## 1. Per-Diarizer False Positive / False Negative Rates

| Diarizer | TP | TN | FP | FN | FP rate | FN rate | Threshold |
|----------|----|----|----|----|---------|---------|-----------|
| BabAR | 281 | 79 | 27 | 54 | 6.1% | 12.2% | 0.12 |
| VTC-KCHI | 281 | 79 | 27 | 54 | 6.1% | 12.2% | 0.12 |
| Whisper-MIL | 303 | 60 | 46 | 32 | 10.4% | 7.3% | 0.50 |
| VTC | 305 | 59 | 47 | 30 | 10.7% | 6.8% | 0.13 |
| Pyannote | 301 | 40 | 66 | 34 | 15.0% | 7.7% | 0.16 |
| Sortformer | 301 | 29 | 77 | 34 | 17.5% | 7.7% | 0.10 |
| USC-SAIL | 320 | 29 | 77 | 15 | 17.5% | 3.4% | 0.21 |
| WavLM-MIL | 326 | 28 | 78 | 9 | 17.7% | **2.0%** | 0.50 |
| VBx | 311 | 27 | 79 | 24 | 17.9% | 5.4% | 0.175 |
| EEND-EDA | 312 | 14 | 92 | 23 | **20.9%** | 5.2% | 0.10 |
| TalkNet-ASD | 69 | 99 | 7 | 266 | 1.6% | **60.3%** | 0.10 |

*BabAR has the best-balanced error profile (6.1% FP / 12.2% FN). WavLM-MIL has the lowest FN rate (2.0%) at the cost of high FP (17.7%). TalkNet-ASD is a special case — extremely high recall at the expense of near-total FN dominance.*

**FP breakdown by interaction context (interactive / non-interactive):**

| Diarizer | FP interactive | FP non-interactive |
|----------|---------------|---------------------|
| BabAR | 19 | 8 |
| USC-SAIL | 54 | 23 |
| VTC | 37 | 10 |
| WavLM-MIL | 55 | 23 |
| Whisper-MIL | 37 | 9 |
| EEND-EDA | 57 | 35 |

*Interactive clips dominate FP errors across all diarizers — ~70% of FPs occur when the child is interacting with someone, likely because adult speech patterns during interaction resemble child vocalization contexts.*

**FP/FN breakdown by age band:**

| Diarizer | FP 14m | FP 36m | FN 14m | FN 36m |
|----------|--------|--------|--------|--------|
| BabAR | 18 | 9 | 24 | 30 |
| USC-SAIL | 51 | 26 | 9 | 6 |
| VTC | 30 | 17 | 18 | 12 |
| WavLM-MIL | 51 | 27 | 4 | 5 |
| Whisper-MIL | 28 | 18 | 21 | 11 |
| EEND-EDA | 60 | 32 | 10 | 13 |

*14-month clips generate more FPs than 36-month across all diarizers. FN distribution is more balanced by age.*

**FP/FN breakdown by n_children:**

| Diarizer | FP nc=1 | FP nc=2+ | FN nc=1 | FN nc=2+ |
|----------|---------|----------|---------|----------|
| BabAR | 15 | 11 | 43 | 9 |
| USC-SAIL | 51 | 20 | 13 | 2 |
| VTC | 23 | 19 | 28 | 2 |
| WavLM-MIL | 52 | 22 | 8 | 0 |
| Whisper-MIL | 25 | 17 | 31 | 0 |
| EEND-EDA | 64 | 22 | 19 | 3 |

*Multi-child (2+) clips concentrate FPs — other children vocalizing are mistaken for the target child. FNs are almost entirely single-child clips.*

---

## 2. Stratified F1 by Task Type

*All 11 diarizers. `unknown` = unlabeled or miscoded task in BIDS annotations.*

| Task | BabAR | VTC | USC-SAIL | Whisper-MIL | WavLM-MIL | Pyannote | VBx | Sortformer | EEND-EDA |
|------|-------|-----|----------|-------------|-----------|----------|-----|------------|----------|
| daily routine | **0.981** | **1.000** | 0.963 | 0.962 | **0.982** | 0.926 | 0.945 | 0.963 | 0.926 |
| social routine | 0.952 | 0.952 | 0.955 | 0.930 | 0.930 | 0.955 | **0.978** | 0.850 | 0.955 |
| gen. social comm. | 0.912 | 0.939 | 0.921 | 0.924 | 0.929 | 0.915 | 0.904 | 0.891 | 0.888 |
| book share | 0.857 | 0.857 | 0.857 | **0.923** | 0.875 | 0.800 | 0.875 | 0.800 | 0.800 |
| toy play | 0.887 | 0.876 | 0.875 | 0.875 | 0.848 | 0.828 | 0.836 | 0.831 | 0.823 |
| other | 0.893 | 0.933 | 0.848 | 0.857 | 0.875 | 0.786 | 0.812 | 0.825 | 0.783 |
| motor play | 0.800 | 0.818 | 0.834 | 0.837 | 0.827 | 0.786 | 0.808 | 0.757 | 0.808 |
| special occasion | 0.737 | 0.727 | 0.750 | 0.720 | 0.720 | 0.783 | 0.750 | 0.720 | 0.750 |
| unknown | 0.000 | 0.000 | 0.000 | 0.824 | 0.842 | 0.842 | 0.462 | **0.947** | 0.625 |

**Key findings:**
- `unknown` task is a catastrophic failure for diarizer-based systems (BabAR/VTC/USC-SAIL F1=0.000) — these clips appear to have unusual acoustic properties. Neural systems (Sortformer, Pyannote, Whisper-MIL) are robust.
- `special occasion` is the hardest labeled task across all systems (~0.72–0.78), likely due to background noise, crowds, and multiple speakers.
- `daily routine` and `social routine` are consistently easiest (0.93–1.00).
- `motor play` is challenging for all (~0.75–0.84), possibly due to physical activity noise masking vocalizations.

---

## 3. Stratified F1 by Age Band, Interaction, N-Children, N-Adults

### By age band

| Group | BabAR | VTC | USC-SAIL | Whisper-MIL | WavLM-MIL | Pyannote | VBx | Sortformer | EEND-EDA |
|-------|-------|-----|----------|-------------|-----------|----------|-----|------------|----------|
| 14-month | 0.869 | 0.858 | 0.837 | 0.853 | 0.853 | 0.836 | 0.836 | 0.797 | 0.814 |
| 36-month | 0.879 | **0.917** | **0.912** | **0.917** | 0.913 | 0.879 | 0.880 | 0.891 | 0.876 |

*36-month consistently outperforms 14-month by ~3–8pp F1. The gap is largest for USC-SAIL (+7.5pp) and VTC (+5.9pp), suggesting these diarizers rely on acoustic features that mature with age.*

### By interaction presence

| Group | BabAR | VTC | USC-SAIL | Whisper-MIL | WavLM-MIL | Pyannote | VBx | Sortformer | EEND-EDA |
|-------|-------|-----|----------|-------------|-----------|----------|-----|------------|----------|
| Interactive (True) | 0.874 | 0.892 | 0.894 | 0.896 | 0.904 | 0.881 | 0.877 | 0.864 | 0.875 |
| Non-interactive (False) | 0.873 | 0.867 | 0.778 | 0.833 | 0.778 | 0.729 | 0.760 | 0.746 | 0.710 |

*Non-interactive clips are harder for most systems — likely because these are clips where the child is present but not interacting (e.g., playing alone), and the acoustic context provides less signal. Whisper-MIL and WavLM-MIL are most affected (−6pp for non-interactive).*

### By number of children in scene

| n_children | BabAR | VTC | USC-SAIL | Whisper-MIL | WavLM-MIL | Pyannote | VBx | Sortformer | EEND-EDA |
|------------|-------|-----|----------|-------------|-----------|----------|-----|------------|----------|
| 1 | 0.887 | 0.905 | 0.890 | 0.896 | 0.898 | 0.862 | 0.869 | 0.849 | 0.859 |
| 2+ | 0.825 | 0.822 | 0.811 | 0.847 | 0.819 | 0.838 | 0.813 | 0.820 | 0.784 |

*Multi-child scenes cause a consistent ~5–7pp F1 drop. Whisper-MIL degrades least (−4.9pp), Sortformer most (−2.9pp, smaller absolute gap).*

### By number of adults in scene

| n_adults | BabAR | VTC | USC-SAIL | Whisper-MIL | WavLM-MIL | Pyannote | VBx | Sortformer | EEND-EDA |
|----------|-------|-----|----------|-------------|-----------|----------|-----|------------|----------|
| 0 | **0.886** | **0.907** | 0.885 | **0.895** | 0.887 | 0.862 | 0.871 | 0.857 | 0.849 |
| 1 | 0.855 | 0.859 | **0.880** | 0.873 | **0.877** | 0.857 | 0.834 | 0.811 | 0.842 |
| 2+ | 0.545 | 0.533 | 0.556 | 0.750 | 0.778 | 0.750 | 0.706 | 0.750 | 0.778 |

*2+ adults causes a severe drop for diarizer-based systems (BabAR/VTC 0.54–0.55). MIL-based systems (Whisper-MIL, WavLM-MIL) degrade less (0.75–0.78). Very few clips have 2+ adults (n=14), so these are noisy estimates.*

### By face visibility (video quality proxy)

| Group | BabAR | VTC | USC-SAIL | Whisper-MIL | WavLM-MIL | Pyannote | VBx |
|-------|-------|-----|----------|-------------|-----------|----------|-----|
| High (8–10) | 0.883 | 0.892 | 0.879 | 0.893 | 0.884 | 0.866 | 0.857 |
| Mid (5–7) | 0.862 | 0.891 | 0.880 | 0.891 | 0.898 | 0.903 | 0.906 |
| Low (1–4) | 0.873 | 0.879 | 0.862 | 0.870 | 0.865 | 0.800 | 0.813 |

*No strong monotonic relationship with face visibility for audio-only systems — this is expected, as face visibility is a video quality proxy. Mid-range visibility has unexpectedly high F1 for some systems.*

---

## 4. BabAR Combined Model — Detailed Error Analysis

*`babar_combined_runs/error_analysis/thesis_summary.json`. Best model: `pertp_logistic_diarizer_plus_phoneme`.*

| Metric | Value |
|--------|-------|
| Total clips | 441 |
| Total errors | 89 (20.2%) |
| False positives | 19 (4.3%) |
| False negatives | 70 (15.9%) |
| FN: silent child | 51 (72.9% of FN) |
| FN: vocal child missed | 19 (27.1% of FN) |
| FP rate in multi-child scenes | 25.0% |
| FP rate in single-child scenes | 14.9% |
| FN rate with interaction | 22.1% |
| FN rate without interaction | 14.8% |
| Children with perfect accuracy | 57 / 109 |
| Children below 70% accuracy | 34 / 109 |
| Mean KCHI duration in FP clips | 2.37 s |
| Mean KCHI duration in FN clips | 1.98 s |
| Easiest task | general social interaction |
| Hardest task | unknown |

*The dominant failure mode is false negatives on silent/minimally-vocalizing children (51/70 FN). The child is present but does not vocalize in the clip, yet the target-child enrollment prototype still produces a near-threshold cosine similarity. Short KCHI detections (~2s) drive most FPs.*

---

## 5. Cross-Diarizer Persistent Errors

*Files: `evaluation/cross_diarizer_errors/persistent_false_{positives,negatives}.csv`.*

### Persistent False Positives (104 unique clips)

Clips where the child is labeled negative (not vocalizing) but multiple diarizers predict positive:

| Diarizers agreeing | Count | Example context |
|-------------------|-------|-----------------|
| All 11 | 1 | Motor play, 14m, 1 adult 1 child, interactive, outside private |
| 10 / 11 | 8 | Motor play, toy play, special occasion; mostly 2+ children or interactive |
| 9 / 11 | 10 | Mixed tasks; motor play, general social, toy play |
| 8 / 11 | ~18 | — |

*The 1 clip failing all 11 diarizers (child A2P7X9N8L7, motorplay run-07, 14-month): 1 adult + 1 child, interactive, outside private location. The other child present is almost certainly the source of detected vocalizations.*

**Top FP-driving factors:**
- 2+ children present (sibling/peer vocalizations trigger detection)
- Interactive context (child-directed speech overlaps with detection boundary)
- Motor play / outdoor locations (ambient noise + movement sounds)

### Persistent False Negatives (284 unique clips)

Clips where the child is labeled positive (vocalizing) but multiple diarizers miss:

| Diarizers missing | Count | Example context |
|-------------------|-------|-----------------|
| 10 / 11 | 3 | Unknown task, "other" task — non-interactive, outside public, single child |
| 8 / 11 | 3 | Motor play, book share — interactive, single child |
| 7 / 11 | 8 | Social routine, GSCI, unknown — mixed |
| 6 / 11 | ~15 | — |

*The 3 clips failing 10/11 diarizers are all non-interactive, single-child clips in public locations (unknown/other/toyplay tasks). The child may be vocalizing quietly or at a distance from the microphone.*

**Top FN-driving factors:**
- Non-interactive context (child vocalizing alone, no adult response)
- Public locations (background noise floor)
- Unknown/other/social routine tasks (atypical recording conditions)
- Short or quiet vocalizations

---

## 6. Synthetic Augmentation — Error Profile

*`synth_results/augmentation_experiments/default_14_18mo/error_counts.json`. Baseline = 0× ratio (real data only).*

| Category | Count | Notes |
|----------|-------|-------|
| Unchanged correct | 360 / 441 | 81.6% of clips unaffected |
| Unchanged errors (all ratios) | 81 / 441 | 18.4% — identical across 0×–10× |
| — Short vocalization errors | 44 | Child speaks <0.5s; enrollment prototype similarity too low |
| — Overlap errors | 23 | Target child in overlap with another speaker |
| — Adult background FP | 7 | Adult background speech pattern misclassified |

| Age band | Clips | Error rate (base) | Error rate (best ratio) | Δ |
|----------|-------|-------------------|--------------------------|---|
| 14_month | 234 | 17.9% | 17.9% | 0.0 |
| 36_month | 207 | 18.8% | 18.8% | 0.0 |

*Synthetic augmentation does not change any individual clip's error status. The 81 hard clips are structurally resistant to the augmentation strategy used (BabAR RTTM + frozen ECAPA encoder). Short vocalizations (44/81) represent the largest single addressable error category.*

---

## 7. Audio-Visual Fusion — Error Analysis

*`av_fusion/av_results/manual_only/error_analysis_summary.json`. Gated AV model vs. audio-only baseline.*

| Error mode | Count | Mean |Δprob| | Mean vis. eligibility |
|------------|-------|-------------|----------------------|
| AV corrected FN (helped_fn) | 22 | 0.184 | 0.890 |
| AV introduced FP (hurt_fp) | 23 | 0.182 | 0.870 |
| Multi-face ambiguous | 14 | 0.005 | 0.603 |
| AV corrected FP (helped_fp) | 0 | — | — |
| AV introduced FN (hurt_fn) | 0 | — | — |
| Off-camera miss | 0 | — | — |

*AV fusion corrects 22 FNs but introduces 23 new FPs — a near-exact wash. Both error modes occur at high visual eligibility scores (~0.87–0.89), meaning the eligibility gate does not effectively screen out unreliable visual inputs. The manual face visibility annotation is insufficient to distinguish when video should be trusted for classification. Multi-face ambiguous clips (14) are low-impact (mean |Δprob|=0.005).*

---

## 8. Summary of Hard Cases

Across all analyses, the following categories are consistently the hardest:

| Factor | Effect | Most-affected diarizers |
|--------|--------|------------------------|
| 2+ adults in scene | F1 drops to 0.53–0.78 | BabAR, VTC, USC-SAIL |
| 2+ children in scene | F1 drops ~5–7pp | All |
| Unknown/unlabeled task | F1 = 0.000 for diarizer-based | BabAR, VTC, USC-SAIL, VBx |
| Special occasion | F1 ~ 0.72–0.78 | All |
| Motor play | F1 ~ 0.76–0.84 | All |
| Non-interactive, public, outdoor | Persistent FN | All |
| Short vocalizations (<0.5s) | 44 unresolvable errors | All |
| Overlap speech | 23 unresolvable errors | All |
| 14-month cohort | −3 to −8pp vs. 36-month | All |

**Hard clip signature:** non-interactive, public/outdoor, 14-month, unknown/motor play task, single child vocalizing briefly or quietly. These 81 clips remain errors regardless of model, augmentation strategy, or ensemble configuration.
