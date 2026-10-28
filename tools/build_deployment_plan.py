"""Build the 8 x 4 x 3 engine deployment plan for the new-engine line.

Only the plan shape is ours; the engine still owns the authoritative schema and
validation (``grid_maps.deployment.validate_plan`` /
``compile_actions``).  The layout rule for the r10 line is:

* low sorties (21002) start on the northern columns, closest to the enemy nests
  they sweep toward;
* high/medium sorties (21000/21001) start on the southern columns, behind the low
  echelon.

Entity coordinates are ``lla = (lat, lon, alt)`` (see
``core/envengine/simulator/interfaces/isimulator.py``), so the four-corner red
area has ``x = latitude`` (min = south, max = north) and ``y = longitude``.  The
8x4 grid therefore walks latitude in ``col`` and longitude in ``row``: the
latitude bands are the columns, and it is the columns that must be split between
the lead (northern) and trailing (southern) echelon.

Usage:
    python tools/build_deployment_plan.py \
        --scenario scenarios/cases/launch20/easy/E01/scenario.json \
        --out results/deployment_plans/E01_r10.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grid_maps.deployment import compile_actions, validate_plan

REPO_ROOT = Path(__file__).resolve().parents[1]
ROWS = 8
COLS = 4
WEAPON_TYPES = (21000, 21001, 21002)
NORTH_COLS = (2, 3)
SOUTH_COLS = (0, 1)
COLUMNS_BY_TYPE = {21000: SOUTH_COLS, 21001: SOUTH_COLS, 21002: NORTH_COLS}
# HMLopt4launch20：把 21001 也放进靠近目标的东侧列，让它比 H 先到、先去吃
# 蓝方拦截弹，H 与 L 保留原来的效果。
MEDIUM_NEAR_COLUMNS = {21000: SOUTH_COLS, 21001: NORTH_COLS, 21002: NORTH_COLS}


def _counts_by_type(entities: dict) -> dict[int, int]:
    counts = {weapon_type: 0 for weapon_type in WEAPON_TYPES}
    for entity_id, item in entities.items():
        if not isinstance(entity_id, int):
            raise ValueError(f"entity id must be an int: {entity_id!r}")
        if not isinstance(item, dict):
            raise ValueError(f"entity {entity_id} must be a mapping")
        weapon_type = item.get("type")
        if weapon_type not in counts:
            raise ValueError(f"unsupported platform type {weapon_type!r} for entity {entity_id}")
        counts[weapon_type] += 1
    return counts


def build_plan(entities: dict, area=None, medium_near: bool = False) -> dict:
    """Return a ``version=1`` plan with one 8x4 matrix per platform type."""

    counts_by_type = _counts_by_type(entities)
    columns = MEDIUM_NEAR_COLUMNS if medium_near else COLUMNS_BY_TYPE
    counts: dict[str, list[list[int]]] = {}
    for weapon_type in WEAPON_TYPES:
        matrix = [[0] * COLS for _ in range(ROWS)]
        cells = [(row, col) for row in range(ROWS) for col in columns[weapon_type]]
        for index in range(counts_by_type[weapon_type]):
            row, col = cells[index % len(cells)]
            matrix[row][col] += 1
        counts[str(weapon_type)] = matrix
    return {"version": 1, "counts": counts}


def _load_scenario(path: str | Path) -> tuple[dict, list, list]:
    payload = json.loads(Path(path).read_text())
    profile = payload["imagineProfile"] if "imagineProfile" in payload else payload
    entities: dict[int, dict] = {}
    for item in profile["entityList"]:
        entity = item["entity"]
        weapon_type = int(entity["entityType"])
        if weapon_type not in WEAPON_TYPES:
            continue
        entities[int(entity["id"])] = {
            "type": weapon_type,
            "position": {"alt": float(entity["lla"]["z"])},
        }
    red_area = profile.get("redArea") or {}
    coordinates = red_area.get("coordinates")
    coordinates_hm = red_area.get("coordinatesHM") or coordinates
    if not coordinates:
        raise ValueError(f"scenario has no redArea coordinates: {path}")
    return entities, coordinates, coordinates_hm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, help="scenario.json path")
    parser.add_argument("--out", required=True, help="output plan path")
    parser.add_argument("--medium-near", action="store_true", help="put 21001 on the near columns")
    args = parser.parse_args(argv)

    entities, coordinates, coordinates_hm = _load_scenario(args.scenario)
    plan = build_plan(entities, medium_near=args.medium_near)
    expected = _counts_by_type(entities)
    validate_plan(plan, expected)
    actions = compile_actions(plan, entities, coordinates, coordinates_hm)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2) + "\n")
    non_empty = sum(
        1
        for weapon_type in WEAPON_TYPES
        for row in plan["counts"][str(weapon_type)]
        for value in row
        if value
    )
    print(
        f"wrote {out_path}: {len(entities)} platforms, {non_empty} non-empty cells, "
        f"{len(actions)} actions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
