"""Core-facing adapter for R0--R9 inside one engine-authorized comm domain.

The 10-19 engine changed the red action protocol: deployments start every unit
and the only legal in-flight red actions are ``0`` (maneuver), ``2`` (fly to a
public search position) and ``4`` (attack an objective the executor has been
authorized to see).  The engine also owns communication domains and asks for one
controller per component, so this commander never sees or plans across
disconnected units.

Task allocation remains deterministic (``policies.red.baselines``); the learned
motion stack runs underneath and only chooses each platform's evasion maneuver.
"""

from __future__ import annotations

import logging
import os

from .baselines import (
    TARGET_HEALTH,
    BaselineObservation,
    BaselineRules,
    PlatformState,
    RED_POLICY_CHOICES,
    TacticalMetrics,
    TargetPrior,
    build_red_baseline,
    distance_km,
    engagement_effectiveness,
    expected_damage,
    target_is_compatible,
)
from .contracts import Position
from .tracks import InitialCatalogueTrackFusion


logger = logging.getLogger(__name__)

_KIND_BY_TYPE = {21000: "H", 21001: "M", 21002: "L"}
_ACTION_SEARCH_POSITION = 2.0
_ACTION_ATTACK_TARGET = 4.0

# A medium sortie cruises at 120 m/s.  A fused track carries its sample step and
# velocity, so the commander can aim at the intercept point and refresh it every
# step: the engine accepts an in-flight search waypoint (action 2), which makes
# this pure command guidance with no core change.
_MEDIUM_KIND = "M"
_MEDIUM_SPEED_KM_S = 0.12

# A dark nest (9500) can only be attacked by the low platform that detected it,
# and its 1.2 health resists one 0.8 warhead.  Later L sorties are therefore
# steered to the last known nest position so they can re-detect and finish it.
_DARK_NEST_TARGET_TYPE = 9500
# Two warheads kill a 1.2 hp nest; one spare sortie covers a miss or a kill
# by a lane strike that the commander cannot observe.
_DARK_NEST_SORTIE_CAP = 3

# The engine rejects any search waypoint outside the map area.  The learned
# motion stack can drift a platform past an edge, and a lane replanned from that
# drifted latitude would then be illegal, so every emitted waypoint is inset.
_MAP_EDGE_INSET_DEGREES = 0.001

# Policies whose allocation is recomputed on a fixed cadence.
_ADAPTIVE_POLICIES = frozenset({
    "r6_frontload_decoy",
    "r7_strike_packages",
    "r7_static_search",
    "r7_grid_sweep",
    "r8_satellite_packages",
    "r9_hierarchical_learning",
    "r9_grid_sweep",
    "r9_search_enhanced",
})
# Policies that allocate once and never replan.
_ONCE_POLICIES = frozenset({"r2_static_assignment", "r3_wave_schedule"})
# Policies whose sorties keep the target id they were first given: the
# commander refreshes that objective's coordinates on every re-plan instead of
# re-targeting a saturated or dead objective.
_FIXED_ID_POLICIES = frozenset({"r11_lethal_package", "windowed_salvo"})


class RedBaselineCommander:
    """Plan target guidance from public priors and reports of one comm domain."""

    def __init__(
        self,
        targets: tuple[TargetPrior, ...],
        policy_name: str,
        seed: int = 1,
        search_polygon: tuple[Position, ...] | None = None,
    ) -> None:
        if policy_name not in RED_POLICY_CHOICES:
            raise ValueError(f"Unsupported red baseline '{policy_name}'")
        self.initial_targets = tuple(
            item for item in targets if item.entity_type in (9400, 9600)
        )
        self.policy_name = policy_name
        self.seed = seed
        self.search_polygon = search_polygon
        self.rules = BaselineRules()
        self.expected_platform_ids: set[int] = set()
        self.reset()

    def register_platform(self, entity_id: int) -> None:
        self.expected_platform_ids.add(int(entity_id))

    def begin_step(self, observations: tuple[dict, ...]) -> None:
        """Accept the complete live snapshot for this communication domain."""

        self.current_step = int(observations[0].get("step", 0)) if observations else 0
        observed_ids = {int(item["entity_id"]) for item in observations}
        if self.expected_platform_ids and not observed_ids <= self.expected_platform_ids:
            raise ValueError("Commander received a platform outside its communication domain")
        self.current_snapshot_ids = frozenset(observed_ids)
        self.absent_platform_ids = frozenset(self.expected_platform_ids - observed_ids)
        changed = False
        for observation in observations:
            self.report(observation)
            changed = self.track_fusion.ingest(observation) or changed
        for entity_id in self.absent_platform_ids:
            # A missing report removes the unit from this planning snapshot.
            # It does not confirm destruction or create an observed threat.
            if self.reports.pop(entity_id, None) is not None:
                changed = True
            self.observed_target_ids.pop(entity_id, None)
            self.guidance.pop(entity_id, None)
            self.last_guidance.pop(entity_id, None)
            previous_target = self.target_by_platform.pop(entity_id, None)
            if previous_target is not None:
                self.assigned_by_target[previous_target] = max(
                    0, self.assigned_by_target.get(previous_target, 1) - 1
                )
            self._forget_damage(entity_id, previous_target)
        self.targets = self.track_fusion.targets
        if changed:
            self.event_revision += 1
        self.tactical_metrics = self._build_tactical_metrics(observations)

    def _forget_damage(self, platform_id: int, target_id: int | None) -> None:
        """Return the expected damage of a dropped sortie to the target pool."""

        amount = self.damage_by_platform.pop(platform_id, 0.0)
        if target_id is None or not amount:
            return
        remaining = self.damage_by_target.get(target_id, 0.0) - amount
        if remaining > 0.0:
            self.damage_by_target[target_id] = remaining
        else:
            self.damage_by_target.pop(target_id, None)

    @staticmethod
    def _remaining_health(target: TargetPrior, damage: dict[int, float]) -> float:
        health = TARGET_HEALTH.get(target.entity_type)
        if not health:
            return 0.0
        return max(0.0, health - damage.get(target.entity_id, 0.0))

    def _marginal_score(
        self,
        platform: PlatformState,
        target: TargetPrior,
        damage: dict[int, float],
    ) -> float:
        """Mirror the policy-side score so both paths rank targets identically."""

        health = TARGET_HEALTH.get(target.entity_type) or 1.0
        expected = expected_damage(platform.kind, target.entity_type)
        useful = min(expected, self._remaining_health(target, damage))
        committed = min(1.0, damage.get(target.entity_id, 0.0) / health)
        return (
            self.rules.value_weight * target.value * useful / health
            - self.rules.distance_weight * distance_km(platform.position, target.position) / 100.0
            - self.rules.coverage_weight * committed
        )

    @staticmethod
    def _track_field(track, name: str, default):
        return track.get(name, default) if isinstance(track, dict) else getattr(track, name, default)

    def report(self, observation: dict) -> None:
        own = observation.get("self", {})
        entity_id = int(observation.get("entity_id", -1))
        kind = _KIND_BY_TYPE.get(int(own.get("type", 0)))
        position = own.get("position", {})
        if entity_id < 0 or kind is None or not isinstance(position, dict):
            return
        if self.expected_platform_ids and entity_id not in self.expected_platform_ids:
            raise ValueError("Platform is not registered in this communication domain")
        previous_ids = self.observed_target_ids.get(entity_id, set())
        current_ids = {
            int(self._track_field(track, "entity_id", raw_id)): int(self._track_field(track, "entity_type", -1))
            for raw_id, track in (own.get("detectInfo") or {}).items()
            if int(self._track_field(track, "entity_type", -1)) in (9400, 9500, 9600)
        }
        for target_id in sorted(current_ids.keys() - previous_ids):
            logger.info(
                "DIAG detect step=%d platform=%d target=%d entity_type=%d",
                self.current_step, entity_id, target_id, current_ids[target_id],
            )
        self.observed_target_ids[entity_id] = set(current_ids)
        self.reports[entity_id] = PlatformState(
            entity_id=entity_id,
            kind=kind,
            position=Position(
                lon=float(position.get("lon", 0.0)),
                lat=float(position.get("lat", 0.0)),
                alt=float(position.get("alt", 0.0)),
            ),
            alive=float(own.get("health", 0)) > 0 and bool(own.get("isVisible", True)),
        )

    def _build_tactical_metrics(self, observations: tuple[dict, ...]) -> TacticalMetrics:
        interceptor_ids: set[int] = set()
        for observation in observations:
            for raw_id, track in (observation.get("self", {}).get("detectInfo") or {}).items():
                if int(self._track_field(track, "entity_type", -1)) == 24000:
                    interceptor_ids.add(int(self._track_field(track, "entity_id", raw_id)))
        alive_count = sum(item.alive for item in self.reports.values())
        total_count = len(self.expected_platform_ids)
        # The 10-19 engine launches every unit at deployment, so "launched" no
        # longer means "left the nest".  It means the platform has been
        # committed to a strike and is flying an attack profile; search lanes
        # (action 2) are not staged sorties and must not dilute the pressure
        # denominator.
        engaged_count = sum(
            1
            for entity_id in self.target_by_platform
            if self.reports.get(entity_id) is not None and self.reports[entity_id].alive
        )
        return TacticalMetrics(
            observed_interceptor_count=len(interceptor_ids),
            launched_count=engaged_count,
            alive_count=alive_count,
            lost_count=max(0, total_count - alive_count),
            total_count=total_count,
            event_revision=self.event_revision,
            assigned_target_counts=tuple(sorted(self.assigned_by_target.items())),
            assigned_damage=tuple(sorted(self.damage_by_target.items())),
        )

    def action_for(self, entity_id: int, step: int) -> list[float] | None:
        """Return a target ID or a public search waypoint for one airborne unit."""

        entity_id = int(entity_id)
        if self.current_snapshot_ids is None:
            raise RuntimeError("begin_step must provide the domain snapshot before planning")
        if entity_id not in self.current_snapshot_ids:
            return None
        if self._should_plan(step):
            self._plan(step)
        action = self.guidance.pop(entity_id, None)
        return action if action is not None else self._retarget_action_for(entity_id)

    def _should_plan(self, step: int) -> bool:
        if not self.current_snapshot_ids:
            return False
        if self.last_plan_step < 0:
            return True
        if self.policy_name in _ONCE_POLICIES:
            return False
        if self.policy_name in _ADAPTIVE_POLICIES:
            return step - self.last_plan_step >= self.rules.adaptive_replan_interval
        if self.policy_name == "r5_event_rolling":
            return (
                self.event_revision > self.planned_revision
                and step - self.last_plan_step >= self.rules.event_replan_cooldown
            )
        return step - self.last_plan_step >= self.rules.replan_interval

    def _plan(self, step: int) -> None:
        platforms = tuple(sorted(self.reports.values(), key=lambda item: item.entity_id))
        decision = self.policy.decide(
            BaselineObservation(step, platforms, self.targets, self.tactical_metrics)
        )
        metrics = self.tactical_metrics
        domain = min(self.current_snapshot_ids) if self.current_snapshot_ids else 0
        logger.info(
            "DIAG plan step=%d domain=%d platforms=%d targets=%d launched=%d alive=%d lost=%d "
            "interceptors=%d pressure=%.3f decided=%d",
            step, domain, len(platforms), len(self.targets), metrics.launched_count,
            metrics.alive_count, metrics.lost_count, metrics.observed_interceptor_count,
            metrics.pressure, len(decision),
        )
        initial_plan = self.last_plan_step < 0
        if not initial_plan and self.policy_name == "r4_rolling_rules":
            decision = decision[
                : max(1, int(sum(item.alive for item in platforms) * 0.25))
            ]
        assignments = {item.platform_id: item for item in decision}
        targets = {item.entity_id: item for item in self.targets}
        if initial_plan:
            selected = tuple(item for item in platforms if item.alive)
            counts: dict[int, int] = {}
            damage: dict[int, float] = {}
            self.damage_by_platform = {}
            self.damage_by_target = {}
        else:
            selected = tuple(
                item for item in platforms
                if item.alive and item.entity_id in assignments
            )
            counts = dict(self.assigned_by_target)
            damage = dict(self.damage_by_target)
            for platform in selected:
                previous = self.target_by_platform.get(platform.entity_id)
                if previous is not None:
                    counts[previous] = max(0, counts.get(previous, 1) - 1)
                    amount = self.damage_by_platform.get(platform.entity_id, 0.0)
                    if amount:
                        remaining = damage.get(previous, 0.0) - amount
                        if remaining > 0.0:
                            damage[previous] = remaining
                        else:
                            damage.pop(previous, None)
                self.damage_by_platform.pop(platform.entity_id, None)

        for platform in selected:
            assignment = assignments.get(platform.entity_id)
            target = targets.get(assignment.target_id) if assignment is not None else None
            is_search = assignment is not None and assignment.destination is not None
            if is_search:
                target = None
                destination = assignment.destination
            else:
                # Fixed-id policies keep the objective they were given: their
                # packages are lethal on purpose and a sortie must never be
                # steered onto a different objective.  Other policies saturate
                # here, at the single point where guidance becomes an engine
                # command.
                fixed = self.policy_name in _FIXED_ID_POLICIES
                available = target is not None and self._target_is_available(platform, target)
                if fixed:
                    if not available:
                        # A cruise missile detonates on arrival, so a search
                        # waypoint is a suicide order that spends the salvo in
                        # empty air.  A fixed sortie that has nothing to attack
                        # keeps its launch course and waits for the package.
                        continue
                else:
                    saturated = target is not None and (
                        counts.get(target.entity_id, 0)
                        >= self.rules.capacity(platform.kind, target.entity_type)
                        or self._remaining_health(target, damage) <= 0
                    )
                    if target is None or saturated or not available:
                        target = self._best_target(platform, counts, damage)
                destination = (
                    target.position
                    if target is not None
                    else self._search_waypoint(platform, damage)
                )
            if target is not None:
                self.target_by_platform[platform.entity_id] = target.entity_id
                counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
                amount = expected_damage(platform.kind, target.entity_type)
                self.damage_by_platform[platform.entity_id] = amount
                damage[target.entity_id] = damage.get(target.entity_id, 0.0) + amount
                if self.logged_assignments.get(platform.entity_id) != target.entity_id:
                    self.logged_assignments[platform.entity_id] = target.entity_id
                    logger.info(
                        "DIAG attack_assign step=%d platform=%d kind=%s target=%d entity_type=%d",
                        step, platform.entity_id, platform.kind, target.entity_id, target.entity_type,
                    )
                action, signature = self._strike_guidance(platform, target, step)
            else:
                self.target_by_platform.pop(platform.entity_id, None)
                self.damage_by_platform.pop(platform.entity_id, None)
                destination = self._clamp_waypoint(destination)
                action = [
                    _ACTION_SEARCH_POSITION,
                    float(platform.entity_id),
                    destination.lon,
                    destination.lat,
                ]
                signature = (None, destination.lon, destination.lat)
            force_refresh = (
                self.policy_name in _FIXED_ID_POLICIES and target is not None
            )
            if (
                initial_plan
                or force_refresh
                or self.last_guidance.get(platform.entity_id) != signature
            ):
                self.guidance[platform.entity_id] = action
            self.last_guidance[platform.entity_id] = signature
        self.assigned_by_target = {
            target_id: count for target_id, count in counts.items() if count > 0
        }
        self.damage_by_target = {
            target_id: amount for target_id, amount in damage.items() if amount > 0.0
        }
        self.last_plan_step = step
        self.planned_revision = self.event_revision

    def _target_is_available(self, platform: PlatformState, target: TargetPrior) -> bool:
        return (
            target.alive
            and target_is_compatible(platform.kind, target.entity_type)
            and (
                target.entity_type in (9400, 9600)
                or target.entity_id in self.observed_target_ids.get(platform.entity_id, set())
            )
        )

    def _best_target(
        self,
        platform: PlatformState,
        counts: dict[int, int],
        damage: dict[int, float] | None = None,
    ) -> TargetPrior | None:
        damage = {} if damage is None else damage
        candidates = [
            target for target in self.targets
            if self._target_is_available(platform, target)
            and counts.get(target.entity_id, 0) < self.rules.capacity(platform.kind, target.entity_type)
            and self._remaining_health(target, damage) > 0
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda target: (
            self._marginal_score(platform, target, damage),
            -target.entity_id,
        ))

    def _search_waypoint(
        self,
        platform: PlatformState,
        damage: dict[int, float] | None = None,
    ) -> Position:
        if not self.search_polygon:
            return platform.position
        if platform.kind == "L":
            nest = self._next_dark_nest(platform, damage)
            if nest is not None:
                self.dark_nest_sorties[nest.entity_id] = (
                    self.dark_nest_sorties.get(nest.entity_id, 0) + 1
                )
                logger.info(
                    "DIAG nest_waypoint step=%d platform=%d target=%d",
                    self.current_step, platform.entity_id, nest.entity_id,
                )
                return nest.position
        center = Position(
            sum(point.lon for point in self.search_polygon) / len(self.search_polygon),
            sum(point.lat for point in self.search_polygon) / len(self.search_polygon),
            platform.position.alt,
        )
        vertex = self.search_polygon[platform.entity_id % len(self.search_polygon)]
        return Position(
            (center.lon + vertex.lon) / 2.0,
            (center.lat + vertex.lat) / 2.0,
            platform.position.alt,
        )

    def _next_dark_nest(
        self,
        platform: PlatformState,
        damage: dict[int, float] | None,
    ) -> TargetPrior | None:
        """Return a still-open dark nest worth a follow-on low sortie.

        The engine authorises a 9500 target per platform, so the L that already
        spent itself on a nest cannot finish the second warhead a 1.2 hp nest
        needs.  A later L that receives this nest's last known position flies
        there as a legal search command, re-detects the nest inside its own
        sensor range, fires, and the 100 m self-destruct radius lands the hit.
        Sorties are capped because a nest needs two hits; the cap bounds the
        waste when an early sortie already finished it.
        """

        damage = {} if damage is None else damage
        nests = [
            target for target in self.targets
            if target.entity_type == _DARK_NEST_TARGET_TYPE
            and target.alive
            and self._remaining_health(target, damage) > 0.0
            and self.dark_nest_sorties.get(target.entity_id, 0) < _DARK_NEST_SORTIE_CAP
        ]
        if not nests:
            return None
        return min(
            nests,
            key=lambda target: (
                self.dark_nest_sorties.get(target.entity_id, 0),
                distance_km(platform.position, target.position),
                target.entity_id,
            ),
        )

    def _clamp_waypoint(self, position: Position) -> Position:
        """Return a search waypoint that is guaranteed to lie inside the map.

        The engine rejects a search command whose target leaves the map area, but
        the learned motion stack can drift a platform past an edge; a grid lane
        replanned from that live latitude would then be illegal.  Clamp the
        emitted waypoint instead of cancelling it: an edge lane stays legal and
        still searches the corridor.
        """

        if not self.search_polygon:
            return position
        lons = [point.lon for point in self.search_polygon]
        lats = [point.lat for point in self.search_polygon]
        return Position(
            min(
                max(position.lon, min(lons) + _MAP_EDGE_INSET_DEGREES),
                max(lons) - _MAP_EDGE_INSET_DEGREES,
            ),
            min(
                max(position.lat, min(lats) + _MAP_EDGE_INSET_DEGREES),
                max(lats) - _MAP_EDGE_INSET_DEGREES,
            ),
            position.alt,
        )

    def _strike_guidance(self, platform, target, step):
        """Return the action and its dedup signature for one committed sortie."""

        return (
            [_ACTION_ATTACK_TARGET, float(platform.entity_id), float(target.entity_id), 0.0],
            (target.entity_id, target.position.lon, target.position.lat),
        )

    def _retarget_action_for(self, entity_id: int) -> list[float] | None:
        platform = self.reports.get(entity_id)
        if platform is None:
            return None
        candidates = tuple(sorted(
            (target for target in self.targets if self._target_is_available(platform, target)),
            key=lambda target: (distance_km(platform.position, target.position), target.entity_id),
        ))
        target = self.policy.claim_retarget(platform, candidates)
        if target is None:
            return None
        previous = self.target_by_platform.get(entity_id)
        if previous is not None:
            self.assigned_by_target[previous] = max(
                0, self.assigned_by_target.get(previous, 1) - 1
            )
        self.target_by_platform[entity_id] = target.entity_id
        self.assigned_by_target[target.entity_id] = (
            self.assigned_by_target.get(target.entity_id, 0) + 1
        )
        action, signature = self._strike_guidance(platform, target, self.current_step)
        self.last_guidance[entity_id] = signature
        return action

    def target_id_for(self, platform_id: int) -> int | None:
        """Return the platform's currently assigned objective, if any."""

        return self.target_by_platform.get(int(platform_id))

    def target_position_for(self, platform_id: int) -> Position | None:
        """Return the position of the assigned objective for the motion skill."""

        target_id = self.target_by_platform.get(int(platform_id))
        if target_id is None:
            return None
        target = next((item for item in self.targets if item.entity_id == target_id), None)
        return target.position if target is not None else None

    def should_use_satellite(self, platform_id: int) -> bool:
        """Whether this platform is the policy's satellite leader."""

        return int(platform_id) in getattr(self.policy, "satellite_platform_ids", frozenset())

    def learning_task_context(self, platform_id: int) -> tuple[float, float, float, float, float]:
        """Domain-local task features for the hierarchical learning actor."""

        target_id = self.target_id_for(platform_id)
        target = next((item for item in self.targets if item.entity_id == target_id), None)
        entity_type = target.entity_type if target is not None else -1
        return (
            float(entity_type == 9400),
            float(entity_type == 9500),
            float(entity_type == 9600),
            self.tactical_metrics.pressure,
            1.0 if self.reports.get(int(platform_id)) is not None else 0.0,
        )

    def reset(self) -> None:
        self.current_step = 0
        self.logged_assignments: dict[int, int] = {}
        self.reports: dict[int, PlatformState] = {}
        self.current_snapshot_ids: frozenset[int] | None = None
        self.absent_platform_ids: frozenset[int] = frozenset()
        self.assigned_by_target: dict[int, int] = {}
        self.damage_by_target: dict[int, float] = {}
        self.damage_by_platform: dict[int, float] = {}
        self.target_by_platform: dict[int, int] = {}
        self.guidance: dict[int, list[float]] = {}
        self.last_guidance: dict[int, tuple[int | None, float, float]] = {}
        self.observed_target_ids: dict[int, set[int]] = {}
        self.dark_nest_sorties: dict[int, int] = {}
        self.last_plan_step = -1
        self.event_revision = 0
        self.planned_revision = -1
        self.tactical_metrics = TacticalMetrics()
        self.policy = build_red_baseline(
            self.policy_name,
            self.rules,
            self.seed,
            self.search_polygon,
        )
        self.targets = self.initial_targets
        self.track_fusion = InitialCatalogueTrackFusion(self.initial_targets)
