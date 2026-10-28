"""Tests for passive full-simulator mission outcome accounting."""

from __future__ import annotations

import unittest

from core.evaluation.mission_outcomes import MissionOutcomeTracker
from core.evaluation.run_summary import RunSummary


class _Commander:
    def target_id_for(self, _platform_id: int) -> int:
        return 51


class _Agent:
    entity_id = 1001
    launch_step = 1
    commander = _Commander()


class _BlueCommander:
    launched_list = {1001: [9001]}


class _Factory:
    def get_all_simulators(self):
        return [_BlueCommander()]


def _observation(red_health: float, red_x: float, target_health: float) -> dict:
    return {
        "entities": {
            1001: {
                "type": 21000,
                "health": red_health,
                "pos_ecf": {"x": red_x, "y": 0.0, "z": 0.0},
            },
            51: {
                "type": 9400,
                "health": target_health,
                "pos_ecf": {"x": 1_000.0, "y": 0.0, "z": 0.0},
            },
        }
    }


class MissionOutcomeTrackerTests(unittest.TestCase):
    def test_arrival_after_blue_engagement_counts_as_evade_and_hit(self) -> None:
        tracker = MissionOutcomeTracker([_Agent()], _Factory())
        tracker.start(_observation(1.0, 700.0, 32.0))
        tracker.update(_observation(0.0, 950.0, 16.0))

        result = tracker.finish(_observation(0.0, 950.0, 16.0))

        self.assertEqual(result["launched"], 1)
        self.assertEqual(result["engaged"], 1)
        self.assertEqual(result["evaded"], 1)
        self.assertEqual(result["intercepted"], 0)
        self.assertEqual(result["arrived"], 1)
        self.assertEqual(result["effective_hits"], 1)
        self.assertEqual(result["evasion_rate"], 1.0)
        self.assertEqual(result["arrival_rate"], 1.0)
        self.assertEqual(result["hit_rate"], 1.0)
        self.assertEqual(
            result["per_platform"]["1001"],
            {
                "target_id": 51,
                "engaged": True,
                "resolved": True,
                "arrived": True,
                "effective_hit": True,
                "alive_at_end": False,
                "minimum_target_distance_m": 50.0,
            },
        )

    def test_distant_engaged_loss_counts_as_intercepted(self) -> None:
        tracker = MissionOutcomeTracker([_Agent()], _Factory())
        tracker.start(_observation(1.0, 0.0, 32.0))
        tracker.update(_observation(0.0, 100.0, 32.0))

        result = tracker.finish(_observation(0.0, 100.0, 32.0))

        self.assertEqual(result["evaded"], 0)
        self.assertEqual(result["intercepted"], 1)
        self.assertEqual(result["arrived"], 0)
        self.assertEqual(result["effective_hits"], 0)

    def test_run_summary_includes_outcomes(self) -> None:
        tracker = MissionOutcomeTracker([_Agent()], _Factory())
        summary = RunSummary(
            scenario="test",
            policies={"red": "test"},
            outcome_tracker=tracker,
        )
        summary.start(_observation(1.0, 700.0, 32.0))
        final_observation = _observation(0.0, 950.0, 16.0)
        summary.update(1, final_observation)

        result = summary.build(
            final_observation,
            termination_reason="environment_done",
            red_launched=1,
        )

        self.assertEqual(result["outcomes"]["launched"], 1)
        self.assertEqual(result["outcomes"]["effective_hits"], 1)


if __name__ == "__main__":
    unittest.main()
