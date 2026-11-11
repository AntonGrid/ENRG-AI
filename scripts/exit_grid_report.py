#!/usr/bin/env python
"""The exit grid as one table, read from the audited half of each run: ``criteria.json``.

Why not the CSV
===============

``walkforward.csv`` is the human table, and it is enough to compare two books. It is not
enough to say *whether the comparison could settle anything*: a row can show eight trades and
+57 bps a trade and still be unreadable, and the CSV has nowhere to put the spread or the
count the gate would have needed. ``criteria.json`` has both, because it is the file the run
writes beside its own verdict — so this script reads that, and prints only what the audit
carries.

Usage
=====

    python scripts/exit_grid_report.py                  # every directory under data/grid
    python scripts/exit_grid_report.py atr2 d-base      # only the named ones
    OUT=data/grid python scripts/exit_grid_report.py

Columns: ``folds`` is how many test windows produced the pooled reading, ``σ bps`` is the
per-trade spread, ``± bps`` is what that spread can resolve at the trade count present, and
``needed`` is the count the gate would need at the same spread — ``never`` when the book has
no dispersion to reason from (zero or one trade). A row whose ``needed`` exceeds ``trades`` is
``unmeasured`` by definition, and printing it next to a passing row is the point of the table.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("OUT", "data/grid"))

HEADERS = (
    ("run", 26),
    ("ablation", 18),
    ("folds", 5),
    ("trades", 6),
    ("bps/trade", 9),
    ("t", 6),
    ("σ bps", 6),
    ("± bps", 6),
    ("needed", 7),
    ("verdict", 11),
)


def cell(value, width):
    return str(value).ljust(width)


def main(argv):
    wanted = set(argv)
    print("  ".join(cell(name, width) for name, width in HEADERS))
    print("  ".join("-" * width for _, width in HEADERS))
    found = 0
    for directory in sorted(ROOT.iterdir()):
        if wanted and directory.name not in wanted:
            continue
        path = directory / "criteria.json"
        if not path.is_file():
            # A run that died before its report is not a run: say nothing rather than guess.
            continue
        run = json.loads(path.read_text(encoding="utf-8")).get("run") or {}
        for row in run.get("summaries") or []:
            resolution = row.get("resolution") or {}
            needed = resolution.get("needed_trades")
            spread = resolution.get("sigma_bps")
            reach = resolution.get("min_edge_bps")
            values = (
                directory.name,
                row.get("ablation", ""),
                row.get("folds", ""),
                row.get("trades", ""),
                f"{float(row.get('edge_bps') or 0.0):+.1f}",
                f"{float(row.get('t_stat') or 0.0):+.2f}",
                "—" if spread is None else f"{float(spread):.0f}",
                "—" if reach is None else f"{float(reach):.0f}",
                "never" if needed is None else needed,
                row.get("verdict", ""),
            )
            print("  ".join(cell(value, width) for value, (_, width) in zip(values, HEADERS)))
            found += 1
    if not found:
        print(f"no criteria.json under {ROOT}: run scripts/exit_grid.sh first", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
