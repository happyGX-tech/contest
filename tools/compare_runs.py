#!/usr/bin/env python3
"""Compare launch20 run summaries side by side.

Usage:
    python tools/compare_runs.py screen_newcore_launch20_it3_E01_701 diag_e01_bc diag_e01_bonly

Each argument is a run id under ``results/runs/`` (or a literal run directory).
The first run is the reference: destruction order and steps are diffed against it.

The script only reads ``summary.json`` (plus ``1.json`` event totals when
``--events`` is passed) and prints a Markdown table so it can be pasted into a
plan/report.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def latest_run_dir(name: str) -> Path:
    path = Path(name)
    if path.is_dir():
        return path
    candidates = sorted((ROOT / "results" / "runs" / name).glob("*/summary.json"))
    if not candidates:
        raise SystemExit(f"no summary.json under results/runs/{name}")
    return candidates[-1].parent


def load(run_dir: Path) -> dict:
    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def score_fields(summary: dict) -> tuple[float, float, float, int, str]:
    score = summary.get("score", {}) or {}
    value = float(score.get("score", 0.0) or 0.0)
    weight = float(score.get("destroyed_weight", 0.0) or 0.0)
    total = float(score.get("total_objective_weight", 0.0) or 0.0)
    steps = int(summary.get("steps_executed", 0) or 0)
    term = str(summary.get("termination_reason", "") or "")
    return value, weight, total, steps, term


def destroyed(summary: dict) -> dict[int, int]:
    raw = (summary.get("score", {}) or {}).get("destruction_steps", {}) or {}
    return {int(k): int(v) for k, v in raw.items()}


def event_totals(run_dir: Path) -> tuple[int, int]:
    """Return (blue damage events, red platform losses) from 1.json."""

    files = sorted(run_dir.glob("*.json"))
    blue = red = 0
    for path in files:
        if path.name == "summary.json":
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"data_type"' not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("data_type") != "event":
                continue
            event = ((record.get("content") or {}).get("event") or {})
            entity = (event.get("entity") or {})
            source = (event.get("destroySrc") or event.get("source") or {})
            victim = int(entity.get("type", 0) or 0)
            attacker = int(source.get("type", 0) or 0)
            if victim in (9400, 9500, 9600) and attacker in (21000, 21001, 21002):
                blue += 1
            if victim in (21000, 21001, 21002) and attacker == 24000:
                red += 1
    return blue, red


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run id or run directory")
    parser.add_argument("--events", action="store_true", help="also count events from 1.json")
    args = parser.parse_args()

    rows = []
    for name in args.runs:
        run_dir = latest_run_dir(name)
        summary = load(run_dir)
        value, weight, total, steps, term = score_fields(summary)
        hits = destroyed(summary)
        blue = red = 0
        if args.events:
            blue, red = event_totals(run_dir)
        rows.append(
            {
                "name": run_dir.name if name == str(run_dir) else name,
                "dir": run_dir,
                "value": value,
                "weight": weight,
                "total": total,
                "steps": steps,
                "term": term,
                "hits": hits,
                "blue": blue,
                "red": red,
            }
        )

    print("| run | score | weight | steps | termination | blue damage events | red losses |")
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        blue = row["blue"] if args.events else "-"
        red = row["red"] if args.events else "-"
        print(
            f"| {row['name']} | {row['value']:.4f} | {row['weight']:.1f}/{row['total']:.0f} "
            f"| {row['steps']} | {row['term']} | {blue} | {red} |"
        )

    reference = rows[0]
    print()
    print(f"### destruction diff vs {reference['name']}")
    print("| target | ref step | " + " | ".join(row["name"] for row in rows[1:]))
    for target, step in sorted(reference["hits"].items(), key=lambda item: (item[1], item[0])):
        cells = []
        for row in rows[1:]:
            other = row["hits"].get(target)
            cells.append("-" if other is None else f"{other} ({other - step:+d})")
        print(f"| {target} | {step} | " + " | ".join(cells) + " |")
    extra = sorted({t for row in rows[1:] for t in row["hits"]} - set(reference["hits"]))
    for target in extra:
        cells = []
        for row in rows[1:]:
            other = row["hits"].get(target)
            cells.append("-" if other is None else str(other))
        print(f"| {target} (extra) | - | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
