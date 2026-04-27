"""
Aggregate thesis tables from committed result files.

Reads evaluation/configs/thesis_tables.yaml, loads each referenced result file
(JSON or CSV), assembles rows into per-table CSV outputs under
evaluation/thesis_tables/.

Exits 1 with a missing-file report if any required_files entry is absent.
Never constructs numeric values manually — every cell comes from a source file.

Usage:
    python evaluation/aggregate_thesis_tables.py
    python evaluation/aggregate_thesis_tables.py \\
        --config evaluation/configs/thesis_tables.yaml \\
        --output-dir evaluation/thesis_tables/ \\
        --repo-root .

Exit codes:
    0 = success (all tables assembled)
    1 = missing required result files
    2 = config or other error
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml


def _load_yaml(path: str) -> list | dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_nested(obj: dict, dotted_key: str):
    """Resolve a dotted key like 'f0_stats.mean' into obj['f0_stats']['mean']."""
    parts = dotted_key.split(".")
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


def _load_json_source(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_csv_source(path: str, csv_filter: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if csv_filter:
        for col, val in csv_filter.items():
            if col in df.columns:
                df = df[df[col] == val]
    return df


def _check_required_files(required_files: list, repo_root: Path) -> list[str]:
    missing = []
    for rel_path in required_files:
        full = repo_root / rel_path
        if not full.exists():
            missing.append(str(rel_path))
    return missing


def _assemble_table_json_array(table_def: dict, repo_root: Path) -> pd.DataFrame:
    """Assemble a table where all rows come from a single JSON array source."""
    columns = table_def["columns"]
    key_map = table_def.get("key_map", {})
    source_rel = table_def["source"]
    source_path = repo_root / source_rel

    if not source_path.exists():
        print(f"  WARNING: source missing: {source_rel}", file=sys.stderr)
        return pd.DataFrame(columns=list(columns) + ["_source"])

    with open(source_path) as f:
        items = json.load(f)

    if not isinstance(items, list):
        print(f"  WARNING: expected JSON array in {source_rel}", file=sys.stderr)
        return pd.DataFrame(columns=list(columns) + ["_source"])

    assembled = []
    for item in items:
        row = {}
        for col, key in key_map.items():
            row[col] = _get_nested(item, key) if isinstance(item, dict) else None
        row["_source"] = str(source_rel)
        assembled.append(row)

    out_cols = list(columns) + ["_source"]
    records = [{c: r.get(c) for c in out_cols} for r in assembled]
    return pd.DataFrame(records, columns=out_cols)


def _assemble_table(table_def: dict, repo_root: Path) -> pd.DataFrame:
    # Dispatch to json_array handler if source_type is set
    if table_def.get("source_type") == "json_array":
        return _assemble_table_json_array(table_def, repo_root)

    columns = table_def["columns"]
    rows_defs = table_def["rows"]
    assembled = []

    for row_def in rows_defs:
        label = row_def["label"]
        source_rel = row_def["source"]
        key_map = row_def.get("key_map", {})
        meta = row_def.get("meta", {})
        csv_filter = row_def.get("csv_filter")

        source_path = repo_root / source_rel
        if not source_path.exists():
            print(f"  WARNING: source missing for row '{label}': {source_rel}",
                  file=sys.stderr)
            row = {c: None for c in columns}
            row.update(meta)
            row["_source"] = str(source_rel)
            row["_missing"] = True
            assembled.append(row)
            continue

        if source_path.suffix == ".json":
            data = _load_json_source(str(source_path))
            row = dict(meta)
            for col, key in key_map.items():
                row[col] = _get_nested(data, key)
        elif source_path.suffix == ".csv":
            df = _load_csv_source(str(source_path), csv_filter)
            if len(df) == 0:
                print(f"  WARNING: CSV source empty after filter for row '{label}': {source_rel}",
                      file=sys.stderr)
                row = {c: None for c in columns}
                row.update(meta)
            else:
                src_row = df.iloc[0]
                row = dict(meta)
                for col, key in key_map.items():
                    row[col] = src_row.get(key)
        else:
            print(f"  WARNING: unsupported source format for row '{label}': {source_rel}",
                  file=sys.stderr)
            row = {c: None for c in columns}
            row.update(meta)

        row["_source"] = str(source_rel)
        assembled.append(row)

    # Reorder to match columns, add provenance column
    out_cols = list(columns) + ["_source"]
    records = []
    for r in assembled:
        record = {c: r.get(c) for c in out_cols}
        records.append(record)

    return pd.DataFrame(records, columns=out_cols)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate thesis tables from committed result files."
    )
    parser.add_argument(
        "--config",
        default="",
        help="Path to thesis_tables.yaml (default: evaluation/configs/thesis_tables.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for assembled CSVs (default: evaluation/thesis_tables/).",
    )
    parser.add_argument(
        "--repo-root",
        default="",
        help="Repository root path (default: parent of this script's parent).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Assemble tables even when some required files are absent (default: exit 1).",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = Path(args.repo_root).resolve() if args.repo_root else here.parent
    config_path = args.config or str(here / "configs" / "thesis_tables.yaml")
    out_dir = Path(args.output_dir or str(here / "thesis_tables"))

    if not os.path.exists(config_path):
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(2)

    try:
        config = _load_yaml(config_path)
    except Exception as e:
        print(f"ERROR: Failed to parse config: {e}", file=sys.stderr)
        sys.exit(2)

    if not isinstance(config, list):
        print("ERROR: thesis_tables.yaml must be a YAML list of table definitions.",
              file=sys.stderr)
        sys.exit(2)

    # Separate required_files block from table definitions
    required_files: list[str] = []
    table_defs: list[dict] = []
    for entry in config:
        if isinstance(entry, dict) and "required_files" in entry and len(entry) == 1:
            required_files = entry["required_files"]
        elif isinstance(entry, dict) and "name" in entry:
            table_defs.append(entry)

    # Check required files
    missing = _check_required_files(required_files, repo_root)
    if missing:
        print(f"\nMISSING REQUIRED FILES ({len(missing)}):", file=sys.stderr)
        for f in missing:
            print(f"  ✗  {f}", file=sys.stderr)
        if not args.skip_missing:
            print("\nRun the relevant pipeline scripts to generate these files,",
                  file=sys.stderr)
            print("or use --skip-missing to assemble partial tables.", file=sys.stderr)
            sys.exit(1)
        else:
            print("  (--skip-missing: continuing with partial assembly)", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    assembled_tables = []

    for tbl in table_defs:
        name = tbl.get("name", "unnamed")
        title = tbl.get("title", name)
        print(f"\nAssembling: {name}")
        print(f"  {title}")

        try:
            df = _assemble_table(tbl, repo_root)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue

        out_path = out_dir / f"{name}.csv"
        df.to_csv(str(out_path), index=False)
        n_complete = int((~df.isnull().any(axis=1)).sum())
        print(f"  → {out_path.name}  ({n_complete}/{len(df)} rows complete)")
        assembled_tables.append({"table": name, "rows": len(df), "complete": n_complete,
                                  "path": str(out_path)})

    # Write an index of assembled tables
    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(assembled_tables, f, indent=2)

    n_tables = len(assembled_tables)
    print(f"\n{n_tables} table(s) written to {out_dir}/")
    print(f"Index: {index_path}")

    # Exit 1 if any table had missing rows (i.e., source files were absent)
    has_incomplete = any(t["complete"] < t["rows"] for t in assembled_tables)
    if has_incomplete and not args.skip_missing:
        print("\nWARNING: Some table rows have missing source data.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
