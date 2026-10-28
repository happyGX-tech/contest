"""Tests for the eastbound grid-sweep search policy (``r7_grid_sweep``)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from policies.red.baselines import (
    RED_POLICY_CHOICES,
    BaselineObservation,
    BaselineRules,
    PlatformState,
    TacticalMetrics,
    TargetPrior,
    build_red_baseline,
    distance_km,
)
from policies.red.contracts import Position
from policies.red.r7_grid_sweep import GridSweepLayer, R7GridSweepPolicy, select_lane_platforms


POLYGON = (
    Position(11.90, 2.00),
    Position(12.40, 2.00),
    Position(12.40, 2.80),
    Position(11.90, 2.80),
)


def _platform(platform_id: int, kind: str, lat: float, lon: float = 11.60) -> PlatformState:
    return PlatformState(platform_id, kind, Position(lon, lat, 1_000.0))


def _ship(entity_id: int, lat: float, lon: float, value: float = 1.0) -> TargetPrior:
    return TargetPrior(entity_id, 9500, Position(lon, lat, 0.0), value)


def _nest(entity_id: int, lat: float, lon: float, value: float = 5.0) -> TargetPrior:
    return TargetPrior(entity_id, 9400, Position(lon, lat, 0.0), value)


def _observation(platforms, targets, step: int = 0) -> BaselineObservation:
    return BaselineObservation(step, tuple(platforms), tuple(targets), TacticalMetrics())


class GridSweepRegistrationTest(unittest.TestCase):
    def test_policy_name_is_registered_with_the_builder(self):
        self.assertIn("r7_grid_sweep", RED_POLICY_CHOICES)
        policy = build_red_baseline("r7_grid_sweep", BaselineRules(), 7, POLYGON)
        self.assertIsInstance(policy, R7GridSweepPolicy)

    def test_builder_requires_a_public_polygon(self):
        with self.assertRaises(ValueError):
            build_red_baseline("r7_grid_sweep", BaselineRules(), 7, None)


class GridSweepLaneTest(unittest.TestCase):
    def test_each_low_platform_flies_east_along_its_own_latitude(self):
        platforms = [_platform(index, "L", 2.00 + index * 0.05) for index in range(12)]
        platforms += [_platform(100, "H", 2.1), _platform(101, "M", 2.2)]
        policy = R7GridSweepPolicy(BaselineRules(), POLYGON)

        assignments = policy.decide(_observation(platforms, [_nest(5000, 2.3, 12.2)]))

        self.assertEqual(len(assignments), 12)
        by_id = {item.entity_id: item for item in platforms}
        for assignment in assignments:
            self.assertEqual(by_id[assignment.platform_id].kind, "L")
            self.assertIsNotNone(assignment.destination)
            self.assertAlmostEqual(assignment.destination.lat, by_id[assignment.platform_id].position.lat)
            self.assertAlmostEqual(assignment.destination.lon, 12.399)
            self.assertEqual(assignment.launch_step, 0)

    def test_sweep_waits_for_the_lead_before_ordinary_packaging(self):
        platforms = [_platform(index, "L", 2.00 + index * 0.05) for index in range(12)]
        policy = R7GridSweepPolicy(BaselineRules(), POLYGON, sweep_lead_steps=240)

        policy.decide(_observation(platforms, [_nest(5000, 2.3, 12.2)], step=0))
        self.assertEqual(policy.layer.follow_on_step, 240)
        self.assertEqual(policy.decide(_observation(platforms, [_nest(5000, 2.3, 12.2)], step=5)), ())
        self.assertEqual(policy.decide(_observation(platforms, [_nest(5000, 2.3, 12.2)], step=239)), ())

    def test_follow_on_is_anchored_to_deployment_not_to_a_replan(self):
        # A split after a casualty replaces the planner mid-flight.  The wall
        # clock must not restart, otherwise the follow-on wave never launches.
        platforms = [_platform(index, "L", 2.00 + index * 0.05) for index in range(4)]
        platforms += [_platform(100, "H", 2.1)]
        nests = [_nest(5000, 2.3, 12.2)]
        policy = R7GridSweepPolicy(BaselineRules(), POLYGON, sweep_lead_steps=240)

        lanes = policy.decide(_observation(platforms, nests, step=300))
        self.assertTrue(lanes)
        self.assertTrue(all(item.destination is not None for item in lanes))
        self.assertEqual(policy.layer.follow_on_step, 240)
        self.assertFalse(policy.layer.holding(_observation(platforms, nests, step=305)))

    def test_plan_anchor_keeps_the_historical_relative_clock(self):
        # R7_FOLLOW_ON_ANCHOR=plan exists only so the old behaviour can be
        # re-run as an A/B control; it must reproduce the relative clock.
        platforms = [_platform(index, "L", 2.00 + index * 0.05) for index in range(4)]
        nests = [_nest(5000, 2.3, 12.2)]
        with mock.patch.dict(os.environ, {"R7_FOLLOW_ON_ANCHOR": "plan"}):
            policy = R7GridSweepPolicy(BaselineRules(), POLYGON, sweep_lead_steps=240)
            lanes = policy.decide(_observation(platforms, nests, step=300))
        self.assertTrue(lanes)
        self.assertEqual(policy.layer.follow_on_step, 540)
        self.assertTrue(policy.layer.holding(_observation(platforms, nests, step=305)))

    def test_lanes_are_evenly_spaced_and_within_detection_reach(self):
        platforms = [_platform(index, "L", 1.99 + index * (0.643 / 84.0)) for index in range(85)]
        selected = select_lane_platforms(platforms, 36, 12.50)

        self.assertEqual(len(selected), 36)
        latitudes = sorted(item.position.lat for item in selected)
        self.assertEqual(len(set(latitudes)), 36)
        gaps_km = [
            distance_km(Position(12.0, latitudes[index]), Position(12.0, latitudes[index + 1]))
            for index in range(len(latitudes) - 1)
        ]
        # Two detection radii (6 km) is the widest spacing that still overlaps
        # adjacent corridors, so every lane pair must stay below it.
        self.assertLessEqual(max(gaps_km), 6.0)
        self.assertLess(latitudes[0], 2.05)
        self.assertGreater(latitudes[-1], 2.57)

    def test_no_low_platforms_falls_back_to_ordinary_policy_immediately(self):
        platforms = [_platform(100, "H", 2.1), _platform(101, "M", 2.2)]
        policy = R7GridSweepPolicy(BaselineRules(), POLYGON)

        assignments = policy.decide(_observation(platforms, [_nest(5000, 2.3, 12.2)]))

        self.assertEqual(assignments, ())
        self.assertEqual(policy.layer.follow_on_step, 0)
        self.assertEqual(policy.layer.sweep_platform_ids, frozenset())


class GridSweepR9GatingTest(unittest.TestCase):
    def test_r9_without_the_knob_keeps_its_deterministic_layer(self):
        from policies.red.baselines import R9HierarchicalLearningPolicy

        policy = R9HierarchicalLearningPolicy(BaselineRules(), POLYGON)
        self.assertIsNone(policy.layer)

    def test_r9_with_the_knob_prepends_the_same_lane_plan(self):
        from policies.red.baselines import R9HierarchicalLearningPolicy

        policy = R9HierarchicalLearningPolicy(BaselineRules(), POLYGON, enable_grid_sweep=12)
        platforms = [_platform(index, "L", 2.00 + index * 0.05) for index in range(12)]
        platforms += [_platform(100, "H", 2.1)]
        assignments = policy.decide(_observation(platforms, [_nest(5000, 2.3, 12.2)]))
        self.assertEqual(len(assignments), 12)
        self.assertTrue(all(item.destination is not None for item in assignments))
        self.assertAlmostEqual(assignments[0].destination.lon, 12.399)

    def test_r9_requires_a_polygon_when_the_knob_is_set(self):
        from policies.red.baselines import R9HierarchicalLearningPolicy

        with self.assertRaises(ValueError):
            R9HierarchicalLearningPolicy(BaselineRules(), None, enable_grid_sweep=8)


class GridSweepRetargetTest(unittest.TestCase):
    def setUp(self):
        self.platforms = [_platform(index, "L", 2.10, lon=12.00) for index in range(4)]
        self.policy = R7GridSweepPolicy(BaselineRules(), POLYGON)
        self.policy.decide(_observation(self.platforms, [_nest(5000, 2.3, 12.2)]))
        self.scanner = self.platforms[0]

    def test_scanning_platform_claims_a_detected_ship_in_range(self):
        ship = _ship(9001, 2.10, 12.01)
        claim = self.policy.claim_retarget(self.scanner, (ship,))
        self.assertIsNotNone(claim)
        self.assertEqual(claim.entity_id, 9001)

    def test_retarget_is_capped_per_platform_and_per_ship(self):
        first = _ship(9001, 2.10, 12.01)
        second = _ship(9002, 2.10, 12.02)
        self.assertEqual(self.policy.claim_retarget(self.scanner, (first,)).entity_id, 9001)
        self.assertEqual(self.policy.claim_retarget(self.scanner, (first, second)).entity_id, 9002)
        self.assertIsNone(self.policy.claim_retarget(self.scanner, (first, second)))

    def test_ship_claim_is_shared_across_scanning_platforms(self):
        policy = R7GridSweepPolicy(BaselineRules(), POLYGON, max_claims_per_ship=2)
        policy.decide(_observation(self.platforms, [_nest(5000, 2.3, 12.2)]))
        ship = _ship(9001, 2.10, 12.01)
        for scanner in self.platforms[:2]:
            self.assertIsNotNone(policy.claim_retarget(scanner, (ship,)))
        self.assertIsNone(policy.claim_retarget(self.platforms[2], (ship,)))

    def test_default_ship_claims_match_the_three_hits_a_ship_needs(self):
        policy = R7GridSweepPolicy(BaselineRules(), POLYGON)
        policy.decide(_observation(self.platforms, [_nest(5000, 2.3, 12.2)]))
        ship = _ship(9001, 2.10, 12.01)
        for scanner in self.platforms[:3]:
            self.assertIsNotNone(policy.claim_retarget(scanner, (ship,)))
        self.assertIsNone(policy.claim_retarget(self.platforms[3], (ship,)))

    def test_only_sweeping_platforms_may_retarget(self):
        non_scanner = _platform(999, "L", 2.10, lon=12.00)
        ship = _ship(9001, 2.10, 12.01)
        self.assertIsNone(self.policy.claim_retarget(non_scanner, (ship,)))

    def test_ships_out_of_retarget_range_are_ignored(self):
        far_ship = _ship(9001, 2.10, 12.30)
        self.assertIsNone(self.policy.claim_retarget(self.scanner, (far_ship,)))

    def test_low_platforms_prefer_discovered_ships_over_public_targets(self):
        ship = _ship(9001, 2.30, 12.20, value=1.0)
        nest = _nest(5000, 2.25, 12.20, value=5.0)
        chosen = self.policy._best_target(self.scanner, (ship, nest), {})
        self.assertEqual(chosen.entity_id, 9001)

    def test_high_platforms_keep_the_normal_target_order(self):
        ship = _ship(9001, 2.30, 12.20, value=1.0)
        nest = _nest(5000, 2.25, 12.20, value=5.0)
        high = _platform(200, "H", 2.30)
        chosen = self.policy._best_target(high, (ship, nest), {})
        self.assertEqual(chosen.entity_id, 5000)


class R9GridSweepBaselineTest(unittest.TestCase):
    """The named ``r9_grid_sweep`` baseline keeps the sweep on; plain r9 stays off."""

    def _policy(self):
        return build_red_baseline("r9_grid_sweep", BaselineRules(), 7, POLYGON)

    def test_named_baseline_is_registered_and_turns_the_sweep_on(self):
        from policies.red.baselines import R9HierarchicalLearningPolicy

        self.assertIn("r9_grid_sweep", RED_POLICY_CHOICES)
        policy = self._policy()
        self.assertIsInstance(policy, R9HierarchicalLearningPolicy)
        self.assertIsNotNone(policy.layer)
        self.assertEqual(policy.layer.sweep_platform_target, 85)
        self.assertEqual(policy.layer.sweep_lead_steps, 240)

    def test_named_baseline_requires_a_public_polygon(self):
        with self.assertRaises(ValueError):
            build_red_baseline("r9_grid_sweep", BaselineRules(), 7, None)

    def test_environment_still_overrides_the_named_baseline(self):
        with mock.patch.dict(
            os.environ, {"R9_GRID_SWEEP_PLATFORMS": "12", "R9_GRID_SWEEP_LEAD": "60"}
        ):
            policy = self._policy()
        self.assertEqual(policy.layer.sweep_platform_target, 12)
        self.assertEqual(policy.layer.sweep_lead_steps, 60)

    def test_zero_platforms_turns_the_named_baseline_off(self):
        with mock.patch.dict(os.environ, {"R9_GRID_SWEEP_PLATFORMS": "0"}):
            policy = self._policy()
        self.assertIsNone(policy.layer)

    def test_named_baseline_sweeps_first_and_holds_until_the_lead(self):
        platforms = [_platform(index, "L", 2.00 + index * 0.05) for index in range(12)]
        policy = self._policy()

        assignments = policy.decide(
            _observation(platforms, [_nest(5000, 2.3, 12.2)], step=0)
        )

        self.assertEqual(len(assignments), 12)
        for assignment in assignments:
            self.assertIsNotNone(assignment.destination)
            self.assertGreater(assignment.destination.lon, 12.39)
        self.assertEqual(policy.layer.follow_on_step, 240)
        self.assertEqual(
            policy.decide(_observation(platforms, [_nest(5000, 2.3, 12.2)], step=10)), ()
        )

    def test_plain_r9_stays_off_without_the_environment_knob(self):
        from policies.red.baselines import R9HierarchicalLearningPolicy

        with mock.patch.dict(os.environ, {"R9_GRID_SWEEP_PLATFORMS": "0"}):
            policy = R9HierarchicalLearningPolicy(BaselineRules(), POLYGON)
        self.assertIsNone(policy.layer)


class SearchEnhancedAliasTest(unittest.TestCase):
    """``r9_search_enhanced`` is the delivery-facing name of the same policy."""

    def test_delivery_name_is_registered_and_enables_the_sweep(self):
        from policies.red.baselines import R9HierarchicalLearningPolicy

        self.assertIn("r9_search_enhanced", RED_POLICY_CHOICES)
        policy = build_red_baseline("r9_search_enhanced", BaselineRules(), 7, POLYGON)
        self.assertIsInstance(policy, R9HierarchicalLearningPolicy)
        self.assertEqual(policy.layer.sweep_platform_target, 85)
        self.assertEqual(policy.layer.sweep_lead_steps, 240)

    def test_delivery_name_requires_a_public_polygon(self):
        with self.assertRaises(ValueError):
            build_red_baseline("r9_search_enhanced", BaselineRules(), 7, None)

    def test_both_names_produce_the_same_lane_plan(self):
        platforms = [_platform(index, "L", 2.00 + index * 0.05) for index in range(12)]
        observation = _observation(platforms, [_nest(5000, 2.3, 12.2)], step=0)
        mechanism = build_red_baseline("r9_grid_sweep", BaselineRules(), 7, POLYGON)
        delivery = build_red_baseline("r9_search_enhanced", BaselineRules(), 7, POLYGON)

        left = mechanism.decide(observation)
        right = delivery.decide(observation)

        self.assertEqual(
            [(a.platform_id, a.target_id, a.launch_step, a.destination) for a in left],
            [(a.platform_id, a.target_id, a.launch_step, a.destination) for a in right],
        )


if __name__ == "__main__":
    unittest.main()
