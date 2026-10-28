"""Contract tests for the 10-19 engine's official ``launch20`` case set."""

from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "scenarios" / "cases" / "launch20"
RED_TYPES = {"high": 21000, "medium": 21001, "low": 21002}
BLUE_TYPES = {"target": 9400, "site": 9600, "ship": 9500}
INTERCEPTOR_TYPE = 24000
RADAR_TYPE = 44000
EXPECTED_CASES = {"easy": 6, "medium": 6, "hard": 8}
EXPECTED_HP = {"target": 24.0, "site": 18.0, "ship": 1.2000000000000002}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def entities(scenario: dict) -> list[dict]:
    return [item["entity"] for item in scenario["imagineProfile"]["entityList"]]


def cases() -> list[Path]:
    return sorted(CASES.glob("*/*/scenario.json"))


class Launch20ScenarioTests(unittest.TestCase):
    def test_case_distribution_is_six_six_eight(self) -> None:
        for difficulty, count in EXPECTED_CASES.items():
            prefix = {"easy": "E", "medium": "M", "hard": "H"}[difficulty]
            self.assertEqual(
                [f"{prefix}{index:02d}" for index in range(1, count + 1)],
                sorted(path.name for path in (CASES / difficulty).iterdir() if path.is_dir()),
            )
        self.assertEqual(20, len(cases()))

    def test_case_metadata_matches_the_scenario(self) -> None:
        for scenario_path in cases():
            with self.subTest(case=scenario_path.parent.name):
                scenario = load(scenario_path)
                info = load(scenario_path.with_name("case_info.json"))
                self.assertTrue(scenario_path.with_name("deployment_area.json").is_file())
                difficulty = scenario_path.parents[1].name
                rows = entities(scenario)
                counts = Counter(row["entityType"] for row in rows)

                self.assertEqual(difficulty, info["difficulty"])
                self.assertEqual(
                    hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
                    info["scenario_sha256"],
                )
                for kind, entity_type in BLUE_TYPES.items():
                    self.assertEqual(info["root_counts"][kind], counts[entity_type])
                    self.assertTrue(
                        all(
                            row["survivePoints"] == EXPECTED_HP[kind]
                            for row in rows
                            if row["entityType"] == entity_type
                        )
                    )
                for kind, entity_type in RED_TYPES.items():
                    self.assertEqual(info["red_missile_counts"][kind], counts[entity_type])
                self.assertEqual(info["interceptor_total"], counts[INTERCEPTOR_TYPE])
                self.assertEqual(
                    info["blue_root_total"],
                    sum(info["root_counts"].values()),
                )
                self.assertEqual(info["radar_total"], counts[RADAR_TYPE])
                self.assertEqual(info["entity_total"], len(rows))
                self.assertEqual(
                    set(info["objective_ids"]),
                    {
                        row["id"]
                        for row in rows
                        if row["entityType"] in BLUE_TYPES.values()
                    },
                )
                profile = scenario["imagineProfile"]
                self.assertEqual(info["max_steps"] * profile["simStep"], profile["simEndLogicTime"])
                self.assertEqual(info["map_area"], profile["mapArea"])
                self.assertEqual(info["satelliteMaxUseCount"], profile["satelliteMaxUseCount"])
                by_id = {row["id"]: row for row in rows}
                for root in info["roots"]:
                    self.assertEqual(root["hp"], by_id[root["id"]]["survivePoints"])
                    self.assertEqual(root["lon"], by_id[root["id"]]["lla"]["x"])
                    self.assertEqual(root["lat"], by_id[root["id"]]["lla"]["y"])

    def test_entity_graph_locations_and_radar_configuration_are_valid(self) -> None:
        for scenario_path in cases():
            with self.subTest(case=scenario_path.parent.name):
                scenario = load(scenario_path)
                info = load(scenario_path.with_name("case_info.json"))
                rows = entities(scenario)
                by_id = {row["id"]: row for row in rows}
                self.assertEqual(len(rows), len(by_id))
                area = scenario["imagineProfile"]["mapArea"]
                for row in rows:
                    parent_id = row.get("parentId", -1)
                    if parent_id != -1:
                        self.assertIn(parent_id, by_id)
                        self.assertIn(row["id"], by_id[parent_id].get("childrenId", ()))
                    for child_id in row.get("childrenId", ()):
                        self.assertIn(child_id, by_id)
                        self.assertEqual(row["id"], by_id[child_id].get("parentId"))
                    if row["entityType"] in BLUE_TYPES.values():
                        self.assertLessEqual(area["lonMin"], row["lla"]["x"])
                        self.assertGreaterEqual(area["lonMax"], row["lla"]["x"])
                        self.assertLessEqual(area["latMin"], row["lla"]["y"])
                        self.assertGreaterEqual(area["latMax"], row["lla"]["y"])

                # Every sentinel nest carries exactly one co-located radar; ships carry none.
                site_ids = {row["id"] for row in rows if row["entityType"] == BLUE_TYPES["site"]}
                ship_ids = {row["id"] for row in rows if row["entityType"] == BLUE_TYPES["ship"]}
                for root_id in site_ids:
                    radar_ids = [
                        child
                        for child in by_id[root_id].get("childrenId", ())
                        if by_id[child]["entityType"] == RADAR_TYPE
                    ]
                    self.assertEqual(1, len(radar_ids))
                    self.assertEqual(by_id[root_id]["lla"], by_id[radar_ids[0]]["lla"])
                for root_id in ship_ids:
                    self.assertFalse(
                        any(
                            by_id[child]["entityType"] == RADAR_TYPE
                            for child in by_id[root_id].get("childrenId", ())
                        )
                    )

                # Case metadata must agree with the physical interceptor children.
                for root in info["roots"]:
                    interceptor_ids = [
                        child
                        for child in by_id[root["id"]].get("childrenId", ())
                        if by_id[child]["entityType"] == INTERCEPTOR_TYPE
                    ]
                    self.assertEqual(root["interceptors"], len(interceptor_ids))
                    for child_id in interceptor_ids:
                        self.assertEqual(by_id[root["id"]]["lla"], by_id[child_id]["lla"])

    def test_dark_nests_outnumber_mothers_in_every_case(self) -> None:
        for scenario_path in cases():
            with self.subTest(case=scenario_path.parent.name):
                info = load(scenario_path.with_name("case_info.json"))
                self.assertGreaterEqual(info["root_counts"]["target"], 4)
                ratio = info["root_counts"]["ship"] / info["root_counts"]["target"]
                self.assertAlmostEqual(info["dark_to_mother_ratio"], ratio, places=3)
                self.assertGreater(ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
