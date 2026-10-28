"""Checks for R0--R9 using the complete core-provided target catalogue."""

from __future__ import annotations

import unittest

from policies.red.baselines import (
    BaselineObservation,
    BaselineRules,
    PlatformState,
    R0RandomPolicy,
    R1PriorityPolicy,
    R2StaticAssignmentPolicy,
    R3WaveSchedulePolicy,
    R4RollingRulePolicy,
    R5EventRollingPolicy,
    R6FrontloadDecoyPolicy,
    R7StrikePackagePolicy,
    R8SatellitePackagePolicy,
    R9HierarchicalLearningPolicy,
    TacticalMetrics,
    TargetPrior,
)
from policies.red.baselines import target_is_compatible
from policies.red.baselines import expected_damage
from policies.red.contracts import Position
from policies.red.commander import RedBaselineCommander
from policies.red.priors import initial_targets_from_observation
from policies.red.tracks import InitialCatalogueTrackFusion


def observation(step: int = 0) -> BaselineObservation:
    return BaselineObservation(
        step=step,
        platforms=(
            PlatformState(1, "H", Position(118.0, 22.0)),
            PlatformState(2, "M", Position(118.2, 22.0)),
            PlatformState(3, "L", Position(118.4, 22.0)),
        ),
        targets=(
            TargetPrior(101, 9400, Position(118.5, 22.0), value=10.0),
            TargetPrior(102, 9600, Position(119.0, 22.0), value=6.0),
            TargetPrior(103, 9500, Position(121.0, 24.0), value=3.0),
        ),
    )


class RedBaselineTests(unittest.TestCase):
    def test_initial_prior_uses_every_core_provided_target(self) -> None:
        raw = {
            "entities": {
                "1": {"side": 1, "type": 9500, "health": 1, "position": {"lon": 1, "lat": 2}},
                "2": {"side": 1, "type": 9400, "health": 1, "position": {"lon": 3, "lat": 4}},
                "3": {"side": 1, "type": 9600, "health": 1, "position": {"lon": 5, "lat": 6}},
            }
        }
        self.assertEqual([item.entity_id for item in initial_targets_from_observation(raw)], [1, 2, 3])

    def test_all_baselines_assign_only_core_provided_targets(self) -> None:
        rules = BaselineRules(target_capacity=2, replan_interval=3)
        policies = (
            R0RandomPolicy(rules, seed=7),
            R1PriorityPolicy(rules),
            R2StaticAssignmentPolicy(rules),
            R3WaveSchedulePolicy(rules),
            R4RollingRulePolicy(rules),
            R5EventRollingPolicy(rules),
            R6FrontloadDecoyPolicy(rules),
            R7StrikePackagePolicy(rules),
            R8SatellitePackagePolicy(rules),
            R9HierarchicalLearningPolicy(rules),
        )
        for policy in policies:
            with self.subTest(policy=type(policy).__name__):
                assignments = policy.decide(observation())
                self.assertTrue(assignments)
                self.assertTrue({item.target_id for item in assignments} <= {101, 102, 103})
                self.assertEqual(len({item.platform_id for item in assignments}), len(assignments))

    def test_optimizing_policy_uses_platform_target_capability(self) -> None:
        assignments = {item.platform_id: item.target_id for item in R1PriorityPolicy(BaselineRules()).decide(observation())}
        self.assertEqual(assignments[1], 101)
        self.assertEqual(assignments[2], 101)
        self.assertEqual(assignments[3], 103)

    def test_static_plan_is_retained_and_rolling_plan_replans(self) -> None:
        rules = BaselineRules(target_capacity=2, replan_interval=3)
        static = R2StaticAssignmentPolicy(rules)
        self.assertEqual(static.decide(observation()), static.decide(observation(step=5)))
        rolling = R4RollingRulePolicy(rules)
        self.assertEqual(rolling.decide(observation()), rolling.decide(observation(step=2)))

    def test_wave_schedule_is_fixed_and_splits_each_kind_across_waves(self) -> None:
        rules = BaselineRules(wave_count=3, wave_interval=40)
        platforms = tuple(
            PlatformState(entity_id=index + 1, kind="H", position=Position(118.0 + index / 10, 22.0))
            for index in range(6)
        )
        fixed = R3WaveSchedulePolicy(rules)
        first = fixed.decide(BaselineObservation(0, platforms, observation().targets))
        self.assertEqual(first, fixed.decide(BaselineObservation(50, platforms, observation().targets)))
        self.assertEqual(sorted({item.launch_step for item in first}), [0, 40, 80])

    def test_track_fusion_only_updates_initial_catalogue_target_positions(self) -> None:
        fusion = InitialCatalogueTrackFusion(observation().targets)
        changed = fusion.ingest({
            "self": {
                "detectInfo": {
                    103: {"entity_id": 103, "entity_type": 9500, "lla": {"x": 122.0, "y": 25.0, "z": 0.0}},
                    999: {"entity_id": 999, "entity_type": 9400, "lla": {"x": 1.0, "y": 1.0, "z": 0.0}},
                }
            }
        })
        positions = {item.entity_id: item.position for item in fusion.targets}
        self.assertTrue(changed)
        self.assertEqual(positions[103], Position(122.0, 25.0, 0.0))
        self.assertNotIn(999, positions)

    def test_r5_merges_events_until_its_replan_cooldown_expires(self) -> None:
        commander = RedBaselineCommander(
            targets=(TargetPrior(101, 9500, Position(120.0, 20.0)),),
            policy_name="r5_event_rolling",
        )
        commander.register_platform(1)
        first = {
            "step": 0,
            "entity_id": 1,
            "self": {
                "type": 21000,
                "health": 1,
                "isVisible": True,
                "position": {"lon": 118.0, "lat": 20.0, "alt": 0.0},
                "detectInfo": {},
            },
        }
        commander.begin_step((first,))
        commander.action_for(1, 0)
        self.assertEqual(commander.last_plan_step, 0)

        changed = {
            **first,
            "step": 1,
            "self": {
                **first["self"],
                "detectInfo": {
                    101: {
                        "entity_id": 101,
                        "entity_type": 9500,
                        "lla": {"x": 121.0, "y": 20.0, "z": 0.0},
                    }
                },
            },
        }
        commander.begin_step((changed,))
        commander.action_for(1, 1)
        self.assertEqual(commander.last_plan_step, 0)
        self.assertGreater(commander.event_revision, commander.planned_revision)

        commander.action_for(1, commander.rules.event_replan_cooldown)
        self.assertEqual(commander.last_plan_step, commander.rules.event_replan_cooldown)
        self.assertEqual(commander.event_revision, commander.planned_revision)

    def test_commander_builds_adaptive_metrics_from_isolated_reports(self) -> None:
        commander = RedBaselineCommander(observation().targets, "r6_frontload_decoy")
        commander.register_platform(1)
        commander.register_platform(2)
        commander.begin_step(({
            "step": 1,
            "entity_id": 1,
            "self": {
                "type": 21000,
                "health": 1,
                "isVisible": True,
                "position": {"lon": 118.0, "lat": 22.0, "alt": 0.0},
                "detectInfo": {
                    500: {"entity_id": 500, "entity_type": 24000},
                },
            },
        },))
        self.assertEqual(commander.tactical_metrics.observed_interceptor_count, 1)
        self.assertEqual(commander.tactical_metrics.lost_count, 1)
        self.assertGreater(commander.tactical_metrics.pressure, 0.0)

    def test_r6_rolls_from_a_bounded_low_probe_to_a_main_strike(self) -> None:
        policy = R6FrontloadDecoyPolicy(BaselineRules())
        probe = policy.decide(observation())
        self.assertEqual([item.platform_id for item in probe], [3])

        commit = policy.decide(BaselineObservation(
            step=10,
            platforms=(
                PlatformState(1, "H", Position(118.0, 22.0)),
                PlatformState(2, "M", Position(118.2, 22.0)),
            ),
            targets=observation().targets,
            metrics=TacticalMetrics(launched_count=1, alive_count=3, total_count=3),
        ))
        self.assertEqual([item.platform_id for item in commit], [1])

    def test_r6_holds_main_strike_when_observed_pressure_is_high(self) -> None:
        policy = R6FrontloadDecoyPolicy(BaselineRules())
        pressured = BaselineObservation(
            step=10,
            platforms=observation().platforms,
            targets=observation().targets,
            metrics=TacticalMetrics(
                observed_interceptor_count=2,
                launched_count=2,
                alive_count=3,
                total_count=3,
            ),
        )
        self.assertEqual([item.platform_id for item in policy.decide(pressured)], [3])

    def test_r7_pairs_h_and_m_on_the_same_strike_target(self) -> None:
        package_observation = BaselineObservation(
            step=10,
            platforms=(
                PlatformState(1, "H", Position(118.0, 22.0)),
                PlatformState(2, "H", Position(118.1, 22.0)),
                PlatformState(3, "M", Position(118.2, 22.0)),
                PlatformState(4, "M", Position(118.3, 22.0)),
                PlatformState(5, "L", Position(118.4, 22.0)),
            ),
            targets=observation().targets,
            metrics=TacticalMetrics(launched_count=1, alive_count=5, total_count=5),
        )
        assignments = {item.platform_id: item for item in R7StrikePackagePolicy(BaselineRules()).decide(package_observation)}
        self.assertEqual(assignments[1].target_id, assignments[3].target_id)
        self.assertLess(assignments[3].launch_step, assignments[1].launch_step)
        self.assertEqual(set(assignments), {1, 3})

    def test_r8_selects_one_h_satellite_leader_per_strike_target(self) -> None:
        package_observation = BaselineObservation(
            step=10,
            platforms=(
                PlatformState(1, "H", Position(118.0, 22.0)),
                PlatformState(2, "H", Position(118.1, 22.0)),
                PlatformState(3, "M", Position(118.2, 22.0)),
                PlatformState(4, "M", Position(118.3, 22.0)),
            ),
            targets=observation().targets,
            metrics=TacticalMetrics(launched_count=1, alive_count=4, total_count=4),
        )
        policy = R8SatellitePackagePolicy(BaselineRules())
        assignments = {item.platform_id: item for item in policy.decide(package_observation)}
        self.assertTrue(policy.satellite_platform_ids)
        for platform_id in policy.satellite_platform_ids:
            self.assertEqual(package_observation.platforms[platform_id - 1].kind, "H")
            self.assertIn(assignments[platform_id].target_id, {101, 102})


    def test_attack_roles_match_the_engine_authorization_table(self):
        """The engine authorizes H/M only for nests/sites and L only for ships."""

        self.assertTrue(target_is_compatible("H", 9400))
        self.assertTrue(target_is_compatible("H", 9600))
        self.assertTrue(target_is_compatible("M", 9400))
        self.assertTrue(target_is_compatible("M", 9600))
        self.assertTrue(target_is_compatible("L", 9500))
        for kind, target_type in (("H", 9500), ("M", 9500), ("L", 9400), ("L", 9600), ("H", 24000)):
            self.assertFalse(target_is_compatible(kind, target_type), (kind, target_type))



class CommanderWaypointClampTests(unittest.TestCase):
    """Search guidance must stay legal after the motion stack drifts off-map."""

    POLYGON = (
        Position(11.90, 2.00),
        Position(12.40, 2.00),
        Position(12.40, 2.80),
        Position(11.90, 2.80),
    )

    def test_clamp_pulls_a_drifted_waypoint_inside_the_map(self) -> None:
        commander = RedBaselineCommander(
            targets=(), policy_name="r7_grid_sweep", search_polygon=self.POLYGON
        )
        clamped = commander._clamp_waypoint(Position(11.00, 3.50))
        self.assertAlmostEqual(clamped.lon, 11.90 + 0.001)
        self.assertAlmostEqual(clamped.lat, 2.80 - 0.001)

    def test_lane_guidance_from_an_off_map_platform_is_clamped(self) -> None:
        commander = RedBaselineCommander(
            targets=(), policy_name="r7_grid_sweep", search_polygon=self.POLYGON
        )
        commander.register_platform(1)
        commander.begin_step(({
            "step": 0,
            "entity_id": 1,
            "self": {
                "type": 21002,
                "health": 1,
                "isVisible": True,
                "position": {"lon": 11.60, "lat": 3.50, "alt": 1_000.0},
                "detectInfo": {},
            },
        },))
        action = commander.action_for(1, 0)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], 2.0)
        self.assertGreaterEqual(action[2], 11.90 + 0.001 - 1e-9)
        self.assertLessEqual(action[2], 12.40 - 0.001 + 1e-9)
        self.assertGreaterEqual(action[3], 2.00 + 0.001 - 1e-9)
        self.assertLessEqual(action[3], 2.80 - 0.001 + 1e-9)


class DamageBudgetAllocationTests(unittest.TestCase):
    """A target's health bar, not just a sortie count, caps its allocation."""

    POLYGON = (
        Position(11.90, 2.00),
        Position(12.40, 2.00),
        Position(12.40, 2.80),
        Position(11.90, 2.80),
    )

    def test_capacity_table_matches_the_launch20_health_budget(self) -> None:
        rules = BaselineRules()
        # Health bar divided by the measured damage one sortie lands, not by the
        # nominal core damage: six heavies to empty a 24-health mother.
        self.assertEqual(rules.capacity("H", 9400), 6)
        self.assertEqual(rules.capacity("H", 9600), 6)
        self.assertEqual(rules.capacity("M", 9400), 120)
        self.assertEqual(rules.capacity("L", 9500), 9)
        self.assertEqual(rules.capacity("H", 9500), 0)

    def test_expected_damage_scales_the_core_table_by_reach(self) -> None:
        self.assertAlmostEqual(expected_damage("H", 9400), 4.64)
        self.assertAlmostEqual(expected_damage("M", 9600), 0.15)
        self.assertAlmostEqual(expected_damage("L", 9500), 0.144)

    @staticmethod
    def _observation(entity_id: int, entity_type: int) -> dict:
        return {
            "step": 0,
            "entity_id": entity_id,
            "self": {
                "type": entity_type,
                "health": 1.0,
                "isVisible": True,
                "position": {"lon": 11.95, "lat": 2.10 + entity_id * 0.01, "alt": 3_000.0},
                "detectInfo": {},
            },
        }

    def test_heavy_sorties_split_between_the_mother_and_the_sentry_nest(self) -> None:
        targets = (
            TargetPrior(101, 9400, Position(12.30, 2.30), value=10.0),
            TargetPrior(102, 9600, Position(12.30, 2.50), value=6.0),
        )
        commander = RedBaselineCommander(targets, "r1_priority", seed=1, search_polygon=self.POLYGON)
        observations = tuple(self._observation(index, 21000) for index in range(1, 9))
        commander.begin_step(observations)

        for entity_id in range(1, 9):
            commander.action_for(entity_id, 0)

        # Six heavies exhaust the mother's budget; the remaining two fall to the
        # untouched sentry nest instead of stacking a seventh salvo on a wreck.
        self.assertEqual(commander.assigned_by_target, {101: 6, 102: 2})
        self.assertAlmostEqual(commander.damage_by_target[101], 27.84, places=2)
        self.assertAlmostEqual(commander.damage_by_target[102], 6.96, places=2)

    def test_medium_sorties_cannot_saturate_a_mother(self) -> None:
        targets = (
            TargetPrior(101, 9400, Position(12.30, 2.30), value=10.0),
            TargetPrior(102, 9600, Position(12.30, 2.50), value=6.0),
        )
        commander = RedBaselineCommander(targets, "r1_priority", seed=1, search_polygon=self.POLYGON)
        observations = tuple(self._observation(index, 21001) for index in range(1, 11))
        commander.begin_step(observations)

        for entity_id in range(1, 11):
            commander.action_for(entity_id, 0)

        # A medium sortie lands about 0.2 damage, so ten of them stay engaged on
        # the nests: the mother keeps most of the salvo, but nothing is forced
        # into search or onto a wreck by a health bar they cannot empty.
        self.assertEqual(sum(commander.assigned_by_target.values()), 10)
        self.assertGreaterEqual(commander.assigned_by_target.get(101, 0), 6)
        self.assertLess(commander.damage_by_target[101], 24.0)

class DarkNestFollowOnWaypointTests(unittest.TestCase):
    """A finished lane cannot kill a 1.2 hp nest; later L sorties fly there."""

    POLYGON = (
        Position(11.90, 2.00),
        Position(12.40, 2.00),
        Position(12.40, 2.80),
        Position(11.90, 2.80),
    )

    def _commander(self, nests: tuple[TargetPrior, ...] = ()) -> RedBaselineCommander:
        commander = RedBaselineCommander(
            targets=(), policy_name="r7_grid_sweep", search_polygon=self.POLYGON
        )
        commander.targets = nests
        return commander

    def _nest(self, entity_id: int, lon: float, lat: float) -> TargetPrior:
        return TargetPrior(
            entity_id=entity_id,
            entity_type=9500,
            position=Position(lon, lat, 0.0),
            value=2.0,
        )

    @staticmethod
    def _platform(entity_id: int, kind: str, lon: float, lat: float) -> PlatformState:
        return PlatformState(
            entity_id=entity_id,
            kind=kind,
            position=Position(lon, lat, 1_000.0),
            alive=True,
        )

    def test_low_sortie_flies_to_a_known_unfinished_nest(self) -> None:
        nest = self._nest(168, 12.10, 2.40)
        commander = self._commander((nest,))
        waypoint = commander._search_waypoint(self._platform(7, "L", 11.95, 2.05), {})
        self.assertEqual(waypoint, nest.position)
        self.assertEqual(commander.dark_nest_sorties[168], 1)

    def test_heavy_sortie_keeps_the_polygon_sweep(self) -> None:
        nest = self._nest(168, 12.10, 2.40)
        commander = self._commander((nest,))
        waypoint = commander._search_waypoint(self._platform(3, "H", 11.95, 2.05), {})
        self.assertNotEqual(waypoint, nest.position)
        self.assertEqual(commander.dark_nest_sorties, {})

    def test_finished_nest_is_skipped(self) -> None:
        nest = self._nest(168, 12.10, 2.40)
        commander = self._commander((nest,))
        waypoint = commander._search_waypoint(
            self._platform(7, "L", 11.95, 2.05), {168: 1.2}
        )
        self.assertNotEqual(waypoint, nest.position)

    def test_sorties_spread_then_cap_per_nest(self) -> None:
        nests = (self._nest(168, 12.10, 2.40), self._nest(2883, 12.20, 2.10))
        commander = self._commander(nests)
        for entity_id in range(1, 7):
            commander._search_waypoint(
                self._platform(entity_id, "L", 11.95, 2.05), {}
            )
        # Six sorties over two nests: the first two go one per nest, and the
        # spare third sortie per nest absorbs a lane kill the commander cannot
        # observe; the fourth is refused so a dead nest stops draining L hulls.
        self.assertEqual(commander.dark_nest_sorties, {168: 3, 2883: 3})
        no_more = self._commander(nests)
        no_more.dark_nest_sorties = {168: 3, 2883: 3}
        waypoint = no_more._search_waypoint(self._platform(99, "L", 11.95, 2.05), {})
        self.assertNotEqual(waypoint, nests[0].position)
        self.assertNotEqual(waypoint, nests[1].position)


if __name__ == "__main__":
    unittest.main()
