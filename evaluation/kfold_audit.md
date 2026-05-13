# K-fold audit (spec 022 US2 / FR-008)

**Verdict from source code**:

The within-child k-fold splitter is `whisper-modeling/make_kfold_seen_child_split.py`. Its docstring (lines 9-11) states explicitly:

> *"This preserves the within-child paradigm (the same 109 children appear in train/val/test of every fold), so training scripts that accept `--split-dir` can simply point at the appropriate fold directory without code changes."*

The split mechanism (lines 107-113) iterates over each `(child_id, timepoint_norm)` group and assigns its clips to folds modulo k. Therefore every child appears in every fold's train, val, AND test partitions; the variance reported by the 3-fold k-fold is *clip-level* shuffle variance, not *child-level* generalisation variance.

This matches the PI's flagged concern: the existing 3-fold k-fold does NOT measure cross-child generalisation. Spec 022 US2 introduces group-stratified k-fold (children disjoint per fold) to fill that gap.

---

## Empirical confirmation

Inspected 11 systems with `*_kfold<k>_f<i>/` result dirs:

| System | n fold dirs | Verdict |
|---|---|---|
| `fused_attn_unfreeze2` | 3 | WITHIN-CHILD (same children in train+test of every fold) |
| `fused_attn_unfreeze2_whisper_large` | 3 | WITHIN-CHILD (same children in train+test of every fold) |
| `fused_attn_unfreeze2_whisper_medium` | 3 | WITHIN-CHILD (same children in train+test of every fold) |
| `wavlm_mil` | 3 | WITHIN-CHILD (same children in train+test of every fold) |
| `wavlm_pseudo_frame` | 3 | WITHIN-CHILD (same children in train+test of every fold) |
| `whisper_medium_mil` | 3 | WITHIN-CHILD (same children in train+test of every fold) |
| `whisper_mil` | 3 | WITHIN-CHILD (same children in train+test of every fold) |
| `whisper_mil_acmil_max` | 3 | WITHIN-CHILD (same children in train+test of every fold) |
| `whisper_mil_cross_child` | 3 | WITHIN-CHILD (same children in train+test of every fold) |
| `whisper_mil_tsmil_concat` | 3 | WITHIN-CHILD (same children in train+test of every fold) |
| `whisper_pseudo_frame` | 3 | WITHIN-CHILD (same children in train+test of every fold) |

### Per-fold child-overlap detail

**fused_attn_unfreeze2**:

| fold | n_train_children | n_test_children | train∩test | train∩val | val∩test |
|---|---|---|---|---|---|
| 0 | 109 | 109 | 109 | 109 | 109 |
| 1 | 109 | 109 | 109 | 109 | 109 |
| 2 | 109 | 109 | 109 | 109 | 109 |

**fused_attn_unfreeze2_whisper_large**:

| fold | n_train_children | n_test_children | train∩test | train∩val | val∩test |
|---|---|---|---|---|---|
| 0 | 109 | 109 | 109 | 109 | 109 |
| 1 | 109 | 109 | 109 | 109 | 109 |
| 2 | 109 | 109 | 109 | 109 | 109 |

**fused_attn_unfreeze2_whisper_medium**:

| fold | n_train_children | n_test_children | train∩test | train∩val | val∩test |
|---|---|---|---|---|---|
| 0 | 109 | 109 | 109 | 109 | 109 |
| 1 | 109 | 109 | 109 | 109 | 109 |
| 2 | 109 | 109 | 109 | 109 | 109 |

**wavlm_mil**:

| fold | n_train_children | n_test_children | train∩test | train∩val | val∩test |
|---|---|---|---|---|---|
| 1 | 109 | 109 | 109 | 109 | 109 |
| 2 | 109 | 109 | 109 | 109 | 109 |
| 0 | 109 | 109 | 109 | 109 | 109 |

**wavlm_pseudo_frame**:

| fold | n_train_children | n_test_children | train∩test | train∩val | val∩test |
|---|---|---|---|---|---|
| 2 | 109 | 109 | 109 | 109 | 109 |
| 1 | 109 | 109 | 109 | 109 | 109 |
| 0 | 109 | 109 | 109 | 109 | 109 |

**whisper_medium_mil**:

| fold | n_train_children | n_test_children | train∩test | train∩val | val∩test |
|---|---|---|---|---|---|
| 1 | 109 | 109 | 109 | 109 | 109 |
| 0 | 109 | 109 | 109 | 109 | 109 |
| 2 | 109 | 109 | 109 | 109 | 109 |

**whisper_mil**:

| fold | n_train_children | n_test_children | train∩test | train∩val | val∩test |
|---|---|---|---|---|---|
| 0 | 109 | 109 | 109 | 109 | 109 |
| 1 | 109 | 109 | 109 | 109 | 109 |
| 2 | 109 | 109 | 109 | 109 | 109 |

**whisper_mil_acmil_max**:

| fold | n_train_children | n_test_children | train∩test | train∩val | val∩test |
|---|---|---|---|---|---|
| 1 | 109 | 109 | 109 | 109 | 109 |
| 2 | 109 | 109 | 109 | 109 | 109 |
| 0 | 109 | 109 | 109 | 109 | 109 |

**whisper_mil_cross_child**:

| fold | n_train_children | n_test_children | train∩test | train∩val | val∩test |
|---|---|---|---|---|---|
| 0 | 109 | 109 | 109 | 109 | 109 |
| 1 | 109 | 109 | 109 | 109 | 109 |
| 2 | 109 | 109 | 109 | 109 | 109 |

**whisper_mil_tsmil_concat**:

| fold | n_train_children | n_test_children | train∩test | train∩val | val∩test |
|---|---|---|---|---|---|
| 0 | 109 | 109 | 109 | 109 | 109 |
| 1 | 109 | 109 | 109 | 109 | 109 |
| 2 | 109 | 109 | 109 | 109 | 109 |

**whisper_pseudo_frame**:

| fold | n_train_children | n_test_children | train∩test | train∩val | val∩test |
|---|---|---|---|---|---|
| 0 | 109 | 109 | 109 | 109 | 109 |
| 1 | 109 | 109 | 109 | 109 | 109 |
| 2 | 109 | 109 | 109 | 109 | 109 |

---

## Implication for the headline k-fold table

The within-child 3-fold AUROC numbers in `CLAUDE.md` (e.g., Whisper pseudo-frame 0.884±0.020) are *clip-level shuffle variance within the same 109-child population*, not held-out-child generalisation. They are not statistically defensible as cross-child generalisation estimates.

Spec 022 US2 introduces `evaluation/group_stratified_kfold.py` using `sklearn.model_selection.StratifiedGroupKFold(n_splits=5, random_state=42)` with `groups=child_id`. Once that lands, the legacy within-child numbers will be relabelled `Within-child 3-fold (legacy)` in CLAUDE.md and the new group-stratified numbers will be added alongside.
