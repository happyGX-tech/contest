"""Tests for the deployment-plan generator (S3).

The launch20 red area is a four-corner rectangle in ``lla = (lat, lon, alt)``
coordinates, so the 8x4 plan grid walks latitude in ``col`` and longitude in
``row``.  The enemy nests sit north of the red area, so the low sorties that
sweep the dark nests start on the northern columns and the high/medium sorties
start behind them on the southern columns.
"""

import json

import pytest

from grid_maps.deployment import compile_actions, validate_plan
from tools.build_deployment_plan import (
    NORTH_COLS,
    SOUTH_COLS,
    WEAPON_TYPES,
    build_plan,
    main,
)

AREA = [
    [
        [11.381719629926717, 1.9625],
        [11.381719629926717, 2.6975],
        [11.74134166473247, 2.6975],
        [11.74134166473247, 1.9625],
    ]
]


def _entities(high=17, medium=24, low=85):
    entities = {}
    entity_id = 1000
    for entity_type, count in ((21000, high), (21001, medium), (21002, low)):
        for _ in range(count):
            entities[entity_id] = {"type": entity_type, "position": {"alt": 0.0}}
            entity_id += 1
    return entities


def _expected(entities):
    return {
        weapon_type: sum(1 for item in entities.values() if item["type"] == weapon_type)
        for weapon_type in WEAPON_TYPES
    }


def _nonzero_cells(plan, weapon_type):
    matrix = plan["counts"][str(weapon_type)]
    return [(row, col) for row in range(8) for col in range(4) if matrix[row][col]]


def test_counts_are_conserved_per_type():
    entities = _entities()
    plan = build_plan(entities)
    for weapon_type, expected in _expected(entities).items():
        matrix = plan["counts"][str(weapon_type)]
        assert sum(map(sum, matrix)) == expected


def test_low_sorties_land_only_in_northern_columns():
    plan = build_plan(_entities())
    assert _nonzero_cells(plan, 21002)
    assert {col for _, col in _nonzero_cells(plan, 21002)} <= set(NORTH_COLS)


def test_high_and_medium_land_only_in_southern_columns():
    plan = build_plan(_entities())
    for weapon_type in (21000, 21001):
        assert _nonzero_cells(plan, weapon_type)
        assert {col for _, col in _nonzero_cells(plan, weapon_type)} <= set(SOUTH_COLS)


def test_the_two_bands_do_not_share_a_column():
    assert not (set(NORTH_COLS) & set(SOUTH_COLS))
    plan = build_plan(_entities())
    low_cols = {col for _, col in _nonzero_cells(plan, 21002)}
    for weapon_type in (21000, 21001):
        assert low_cols.isdisjoint({col for _, col in _nonzero_cells(plan, weapon_type)})


def test_plan_is_deterministic():
    entities = _entities()
    assert build_plan(dict(entities)) == build_plan(dict(entities))


def test_spread_is_even_inside_each_type_band():
    plan = build_plan(_entities())
    for weapon_type, cols in ((21002, NORTH_COLS), (21000, SOUTH_COLS), (21001, SOUTH_COLS)):
        matrix = plan["counts"][str(weapon_type)]
        values = [matrix[row][col] for row in range(8) for col in cols]
        assert max(values) - min(values) <= 1


def test_plan_passes_engine_validation():
    entities = _entities()
    plan = build_plan(entities)
    validate_plan(plan, _expected(entities))


def test_compile_actions_returns_one_row_per_platform():
    entities = _entities(high=2, medium=2, low=2)
    plan = build_plan(entities)
    rows = compile_actions(plan, entities, AREA, AREA)

    assert len(rows) == len(entities)
    assert {row[1] for row in rows} == set(entities)
    south, north, west, east = 11.381719629926717, 11.74134166473247, 1.9625, 2.6975
    for row in rows:
        assert row[0] == 0
        assert south <= row[2] <= north
        assert west <= row[3] <= east
        assert row[4] == pytest.approx(0.0)


def test_low_cells_sit_north_of_high_medium_cells():
    entities = _entities(high=2, medium=2, low=2)
    rows = compile_actions(build_plan(entities), entities, AREA, AREA)
    low_types = {eid for eid, item in entities.items() if item["type"] == 21002}
    latitude = {row[1]: row[2] for row in rows}
    low_lat = min(latitude[eid] for eid in low_types)
    other_lat = max(latitude[eid] for eid in set(latitude) - low_types)
    assert low_lat > other_lat


def test_unknown_platform_type_is_rejected():
    with pytest.raises(ValueError):
        build_plan({1: {"type": 999, "position": {"alt": 0.0}}})


def _write_scenario(path):
    scenario = {
        "imagineProfile": {
            "entityList": [
                {"entity": {"id": 1, "entityType": 21000, "lla": {"x": 11.5, "y": 2.2, "z": 0.0}}},
                {"entity": {"id": 2, "entityType": 21001, "lla": {"x": 11.5, "y": 2.2, "z": 0.0}}},
                {"entity": {"id": 3, "entityType": 21002, "lla": {"x": 11.5, "y": 2.2, "z": 0.0}}},
                {"entity": {"id": 4, "entityType": 9400, "lla": {"x": 12.2, "y": 2.3, "z": 0.0}}},
            ],
            "redArea": {"coordinates": AREA, "coordinatesHM": AREA},
        }
    }
    path.write_text(json.dumps(scenario))
    return path


def test_cli_writes_a_plan_the_engine_accepts(tmp_path, capsys):
    scenario_path = _write_scenario(tmp_path / "scenario.json")
    out_path = tmp_path / "plan.json"

    assert main(["--scenario", str(scenario_path), "--out", str(out_path)]) == 0

    plan = json.loads(out_path.read_text())
    validate_plan(plan, {21000: 1, 21001: 1, 21002: 1})
    assert "actions" in capsys.readouterr().out
