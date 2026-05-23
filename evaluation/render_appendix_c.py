"""Render auto-generated tables in thesis_v2/appendices/C_full_results.tex.

Reads evaluation/configs/appendix_c_tables.yaml and splices LaTeX rows between
``% BEGIN AUTO:<id>`` and ``% END AUTO:<id>`` markers in the .tex file. Numbers
come from each row's source file (per-system ``test_metrics_tuned.json``);
nothing is hand-typed.

Two row modes:
  * per_row: hand-listed rows; one source path each.
  * groupstrat_aggregate: glob ``<root>/<slug>_groupstrat3_f{0,1,2}/``, take
    mean ± std across folds that have ``test_metrics_tuned.json``; emit a
    ``k/3'' fold-count cell.

Usage:
    python evaluation/render_appendix_c.py
    python evaluation/render_appendix_c.py --dry-run    # print rendered rows
    python evaluation/render_appendix_c.py --check      # exit 1 if .tex would change
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TEX = REPO / "thesis_v2" / "appendices" / "C_full_results.tex"
DEFAULT_CFG = REPO / "evaluation" / "configs" / "appendix_c_tables.yaml"

MARKER_BEGIN = "% BEGIN AUTO:{id}"
MARKER_END = "% END AUTO:{id}"


def _fmt(x: float | None, decimals: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    return f"{x:.{decimals}f}"


def _bold(s: str, on: bool) -> str:
    return f"\\textbf{{{s}}}" if on else s


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _render_per_row(table: dict) -> list[str]:
    lines: list[str] = []
    cols: list[str] = table["columns"]
    key_map: dict = table["json_keys"]
    for row in table["rows"]:
        src = REPO / row["source"] / "test_metrics_tuned.json"
        if not src.exists():
            lines.append(f"% MISSING: {row['source']}/test_metrics_tuned.json")
            continue
        data = _load_json(src)
        bold = bool(row.get("bold_row", False))
        cells: list[str] = []
        for col in cols:
            if col in key_map:
                cells.append(_bold(_fmt(data.get(key_map[col])), bold))
            else:
                meta_val = row["meta"].get(col, "")
                cells.append(_bold(str(meta_val), bold))
        lines.append("  " + " & ".join(cells) + r" \\")
    return lines


def _render_groupstrat(table: dict) -> list[str]:
    lines: list[str] = []
    cols: list[str] = table["columns"]
    key_map: dict = table["json_keys"]
    root = REPO / table["root"]
    for sys_entry in table["systems"]:
        slug = sys_entry["slug"]
        display = sys_entry["display"]
        folds: dict[str, dict] = {}
        for fold in ("0", "1", "2"):
            fold_dir = root / f"{slug}_groupstrat3_f{fold}"
            json_path = fold_dir / "test_metrics_tuned.json"
            if json_path.exists():
                folds[fold] = _load_json(json_path)
        n_complete = len(folds)
        if n_complete == 0:
            lines.append(f"% SKIPPED (no folds): {slug}")
            continue
        cells: list[str] = []
        for col in cols:
            if col == "system":
                cells.append(display)
            elif col == "folds":
                cells.append(f"{n_complete}/3")
            elif col in key_map:
                values = [folds[f][key_map[col]] for f in folds if key_map[col] in folds[f]]
                if not values:
                    cells.append("---")
                elif n_complete == 1:
                    cells.append(_fmt(values[0]))
                else:
                    mean = sum(values) / len(values)
                    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1) if len(values) > 1 else 0.0
                    std = math.sqrt(var)
                    cells.append(f"{_fmt(mean)} $\\pm$ {_fmt(std)}")
            else:
                cells.append("")
        lines.append("  " + " & ".join(cells) + r" \\")
    return lines


def render_one(table: dict) -> list[str]:
    if table["mode"] == "per_row":
        return _render_per_row(table)
    if table["mode"] == "groupstrat_aggregate":
        return _render_groupstrat(table)
    raise ValueError(f"unknown mode: {table['mode']}")


def splice(tex: str, table_id: str, rendered: list[str]) -> tuple[str, bool]:
    begin = MARKER_BEGIN.format(id=table_id)
    end = MARKER_END.format(id=table_id)
    pattern = re.compile(
        rf"(^[ \t]*{re.escape(begin)}[^\n]*\n)(.*?)(^[ \t]*{re.escape(end)}[^\n]*\n)",
        re.DOTALL | re.MULTILINE,
    )
    block = "\n".join(rendered) + "\n"
    match = pattern.search(tex)
    if not match:
        raise SystemExit(
            f"marker pair {begin!r} / {end!r} not found in {DEFAULT_TEX.name}; "
            "add the markers around the data-row region of the table"
        )
    new_tex = tex[: match.start(2)] + block + tex[match.end(2) :]
    return new_tex, new_tex != tex


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--tex", default=str(DEFAULT_TEX))
    ap.add_argument("--dry-run", action="store_true",
                    help="print rendered rows, do not modify .tex")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if .tex would change (no write)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    default_tex = Path(args.tex)

    # Group tables by target .tex file so we read/write each file once.
    by_target: dict[Path, list[dict]] = {}
    for table in cfg["tables"]:
        target = (REPO / table["target_tex"]) if "target_tex" in table else default_tex
        by_target.setdefault(target, []).append(table)

    any_changed = False
    any_drift = False
    for tex_path, tables in by_target.items():
        tex = tex_path.read_text()
        changed_here = False
        for table in tables:
            table_id = table["id"]
            rendered = render_one(table)
            print(f"=== {table_id} ({len(rendered)} rows)  ->  {tex_path.name} ===")
            for line in rendered:
                print(line)
            if args.dry_run:
                continue
            tex, changed = splice(tex, table_id, rendered)
            changed_here = changed_here or changed
        if args.dry_run:
            continue
        if args.check:
            if changed_here:
                any_drift = True
                print(f"would update {tex_path.name}", file=sys.stderr)
            continue
        if changed_here:
            tex_path.write_text(tex)
            print(f"wrote {tex_path}")
            any_changed = True

    if args.dry_run:
        return 0
    if args.check:
        if any_drift:
            print("auto-rendered tables are stale; rerun without --check to apply",
                  file=sys.stderr)
            return 1
        print("all auto-rendered tables are up-to-date.")
        return 0
    if not any_changed:
        print("all auto-rendered tables already up-to-date; no writes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
