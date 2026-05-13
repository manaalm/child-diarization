# Schema: `specs/022-pi-thesis-revisions/bids_vs_spreadsheet_diff.csv`

Per-row provenance log of BIDS-vs-spreadsheet timepoint disagreements (US1 FR-002). Sits next to the spec, not under canonical results dirs, because it is a one-time correction artefact tied to this spec.

## Columns

| Column | Type | Description |
|---|---|---|
| `child_id` | str | SAILS child identifier (matches the `ID` column in `anotated_processed.csv`). |
| `bids_sub_id` | str | BIDS subject directory name, e.g., `sub-A1H3H9Y3T1`. |
| `clip_id` | str | Clip identifier (uniquely identifies a row in `anotated_processed.csv`). |
| `audio_path` | str | Absolute path to the 16kHz mono WAV. |
| `bids_session_id` | str | Session directory name parsed from `audio_path`, e.g., `ses-01`, `ses-02`, or `unknown`. |
| `bids_timepoint` | enum | Derived from session ID + `participants.tsv`: `14_month`, `36_month`, or `unknown`. |
| `spreadsheet_timepoint` | enum | Read from `anotated_processed.csv` `timepoint` column, normalised: `14_month`, `36_month`, or `unknown`. |
| `agree` | bool | `bids_timepoint == spreadsheet_timepoint` AND neither is `unknown`. |
| `rationale_if_disagree` | str | One of `non-standard-session-id`, `spreadsheet-missing`, `bids-missing`, `participants-tsv-disagrees`, `unknown-other`, or empty if `agree == true`. |
| `decision` | enum | Action taken: `keep-bids` (default), `keep-spreadsheet` (rare; flagged for human review), `drop-row` (both unknown). |
| `affected_existing_splits` | str | JSON list of split names that previously contained this row, e.g., `["train", "test"]`. |
| `affected_existing_systems` | str | JSON list of system result dirs that referenced this row via `test_predictions.csv`. |

## Validation

- Every row whose `agree == false` MUST have a non-empty `rationale_if_disagree`.
- Every row with `decision == "drop-row"` MUST have `bids_timepoint == "unknown"` AND `spreadsheet_timepoint == "unknown"`.
- `affected_existing_splits` and `affected_existing_systems` MUST be non-empty for any row that was present in the previous `master_with_split.csv`.

## Aggregation summary (companion `bids_vs_spreadsheet_diff_summary.json`)

A small companion JSON summarises the diff:

```json
{
  "n_rows_total": 3527,
  "n_agree": 3491,
  "n_disagree": 36,
  "n_drop": 0,
  "by_rationale": {
    "non-standard-session-id": 12,
    "spreadsheet-missing": 18,
    "bids-missing": 6,
    "participants-tsv-disagrees": 0
  },
  "by_disagreement_pair": {
    "bids=14_month,spreadsheet=36_month": 8,
    "bids=36_month,spreadsheet=14_month": 4,
    "bids=14_month,spreadsheet=unknown": 18,
    "bids=unknown,spreadsheet=14_month": 6
  },
  "n_children_affected": 18,
  "regenerated_at": "2026-05-15T08:00:00Z"
}
```
