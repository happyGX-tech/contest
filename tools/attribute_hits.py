"""Decompose a run's objective damage into per-kind hit counts."""

from __future__ import annotations

import argparse
import glob
import json

# target_type -> per-sortie damage quanta: (kind, damage)
QUANTA = {
    9400: [("H", 16.0), ("H*", 20.0), ("M", 4.0)],
    9600: [("H", 12.0), ("H*", 15.0), ("M", 3.0)],
    9500: [("L", 0.8)],
}


def _product(ranges):
    if not ranges:
        yield ()
        return
    for head in ranges[0]:
        for tail in _product(ranges[1:]):
            yield (head, *tail)


def decompose(damage: float, target_type: int) -> dict[str, int]:
    quanta = QUANTA[target_type]
    best = None
    ranges = [range(0, int(damage / value) + 2) for _, value in quanta]
    for counts in _product(ranges):
        total = sum(count * value for count, (_, value) in zip(counts, quanta))
        candidate = (round(abs(damage - total), 3), sum(counts), counts)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None or best[0] > 0.05:
        return {}
    return {kind: count for count, (kind, _) in zip(best[2], quanta) if count}


def attribute(run_id: str) -> dict:
    paths = sorted(glob.glob(f"results/runs/{run_id}/*/summary.json"))
    if not paths:
        raise SystemExit(f"no summary for {run_id}")
    summary = json.load(open(paths[-1]))
    hits: dict[str, int] = {}
    unexplained = 0.0
    for objective in summary["objectives"]:
        damaged = objective["initial_health"] - objective["final_health"]
        if damaged <= 0.01:
            continue
        part = decompose(damaged, objective["type"])
        if not part:
            unexplained += damaged
        for kind, count in part.items():
            hits[kind] = hits.get(kind, 0) + count
    return {
        "run": run_id,
        "score": round(summary["score"]["score"], 4),
        "hits": hits,
        "unexplained": round(unexplained, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    for run_id in args.runs:
        print(json.dumps(attribute(run_id), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
