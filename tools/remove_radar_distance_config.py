"""Remove obsolete per-map radar distances from competition cases.

The current engine owns the fixed 200 km radar rule. Keeping a second distance
in scenario JSON or case metadata creates two contradictory sources of truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RADAR_TYPE = 44000


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_radar_distance(scenario: dict) -> int:
    removed = 0
    for item in scenario["imagineProfile"]["entityList"]:
        if item["entity"]["entityType"] != RADAR_TYPE or not item.get("external"):
            continue
        external = json.loads(item["external"])
        radar = external.get("raderExternal", {})
        if "distance" in radar:
            radar.pop("distance")
            removed += 1
        item["external"] = json.dumps(external, ensure_ascii=False, separators=(",", ":"))
    return removed


def process_case(case_dir: Path) -> int:
    scenario_path = case_dir / "scenario.json"
    info_path = case_dir / "case_info.json"
    if not scenario_path.is_file() or not info_path.is_file():
        return 0

    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    removed = strip_radar_distance(scenario)
    write_json(scenario_path, scenario)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.pop("radar_ranges_km", None)
    for root in info.get("roots", []):
        root.pop("radar_km", None)
    info["scenario_sha256"] = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
    write_json(info_path, info)
    return removed


def find_cases(roots: list[Path]) -> tuple[Path, ...]:
    found: set[Path] = set()
    for root in roots:
        if (root / "scenario.json").is_file():
            found.add(root)
        found.update(path.parent for path in root.rglob("scenario.json"))
    return tuple(sorted(found))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    cases = find_cases(args.roots)
    removed = sum(process_case(case) for case in cases)
    print(f"cases={len(cases)} removed_distance_fields={removed}")


if __name__ == "__main__":
    main()
