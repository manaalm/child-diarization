#!/usr/bin/env python3
"""Write a spec-021 README.md verdict header per contracts/readme_verdict_format.md.

The first three lines of the resulting README MUST be:
    # {story-id}: {title}
    **Verdict**: {VERDICT} - {rationale}.
    **Date**: YYYY-MM-DD | **SLURM**: <id> | **Spec**: 021-post-thesis-future-work / US{n}

If --append-summary is given, the existing body is preserved below the header.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

VERDICTS = {"POSITIVE", "NEGATIVE", "NULL", "BLOCKED"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=Path, help="Result dir to write README.md into")
    ap.add_argument("--story", required=True, help="e.g. US1, US2 (must match spec.md user-story IDs)")
    ap.add_argument("--title", required=True, help="One-line title for the dir")
    ap.add_argument("--verdict", required=True, choices=sorted(VERDICTS))
    ap.add_argument("--rationale", required=True, help="One-sentence rationale; no trailing period needed")
    ap.add_argument("--slurm", default="not yet", help="SLURM job id or 'not yet'")
    ap.add_argument("--date", default=dt.date.today().isoformat(),
                    help="ISO date; defaults to today")
    ap.add_argument("--summary", default="",
                    help="Optional Summary paragraph for the body")
    ap.add_argument("--cited", nargs="*", default=[],
                    help="Optional list of cited result paths to render as a bullet list")
    ap.add_argument("--baseline", default="",
                    help="Optional baseline name + path for the Comparator block")
    ap.add_argument("--ship-gate", default="",
                    help="Optional copy of the spec.md SC text")
    ap.add_argument("--notes", default="",
                    help="Optional methodology notes")
    args = ap.parse_args()

    d: Path = args.dir
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
    if not d.is_dir():
        print(f"FAIL: --dir is not a directory: {d}", file=sys.stderr)
        return 1

    rationale = args.rationale.rstrip(".")

    header = (
        f"# {args.story}: {args.title}\n\n"
        f"**Verdict**: {args.verdict} - {rationale}.\n\n"
        f"**Date**: {args.date} | **SLURM**: {args.slurm} | **Spec**: 021-post-thesis-future-work / {args.story}\n\n"
        "---\n\n"
    )

    body_parts: list[str] = []
    if args.summary:
        body_parts.append(f"## Summary\n\n{args.summary.strip()}\n")
    if args.cited:
        bullets = "\n".join(f"- `{c}`" for c in args.cited)
        body_parts.append(f"## Cited results\n\n{bullets}\n")
    if args.baseline or args.ship_gate:
        bl = f"- **Baseline**: {args.baseline}\n" if args.baseline else ""
        sg = f"- **Ship gate**: {args.ship_gate}\n" if args.ship_gate else ""
        body_parts.append(f"## Comparator\n\n{bl}{sg}")
    if args.notes:
        body_parts.append(f"## Notes\n\n{args.notes.strip()}\n")

    body = "\n".join(body_parts) if body_parts else ""
    out = header + body
    (d / "README.md").write_text(out)
    print(f"WROTE {d / 'README.md'} verdict={args.verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
