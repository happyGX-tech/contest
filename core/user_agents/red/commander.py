"""Target assignment for one engine-authorized red communication domain."""

from __future__ import annotations

import math

from .baselines import (
    BaselineObservation,
    BaselineRules,
    PlatformState,
    RED_POLICY_CHOICES,
    TacticalMetrics,
    TargetPrior,
    build_red_baseline,
    distance_km,
    engagement_effectiveness,
    target_is_compatible,
)
from .contracts import Position
from .tracks import ReceivedTrackFusion


_KIND_BY_TYPE = {21000: "H", 21001: "M", 21002: "L"}
_ACTION_SEARCH_POSITION = 2.0
_ACTION_ATTACK_TARGET = 4.0


class RedBaselineCommander:
    """Plan from public priors and reports inside a single communication domain.

    TrainingEnv owns domain membership and replaces this controller when a
    domain merges or splits. It never combines reports from disconnected units.
    The engine independently resolves and validates target-ID actions.
    """

    def __init__(
        self,
        targets: tuple[TargetPrior, ...],
        policy_name: str,
        seed: int = 1,
        search_polygon: tuple[Position, ...] | None = None,
    ) -> None:
        if policy_name not in RED_POLICY_CHOICES:
            raise ValueError(f"Unsupported red baseline '{policy_name}'")
        self.initial_targets = tuple(item for item in targets if item.entity_type in (9400, 9600))
        self.policy_name = policy_name
        self.seed = seed
        self.search_polygon = search_polygon
        self.rules = BaselineRules()
        self.expected_platform_ids: set[int] = set()
        self.reset()

    def register_platform(self, entity_id: int) -> None:
        self.expected_platform_ids.add(int(entity_id))

    def begin_step(self, observations: tuple[dict, ...]) -> None:
        """Accept the complete live snapshot for this domain, once per step."""

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
        self.targets = self.track_fusion.targets
        if changed:
            self.event_revision += 1
        self.tactical_metrics = self._build_tactical_metrics(observations)

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
        self.observed_target_ids[entity_id] = {
            int(self._track_field(track, "entity_id", raw_id))
            for raw_id, track in (own.get("detectInfo") or {}).items()
            if int(self._track_field(track, "entity_type", -1)) in (9400, 9500, 9600)
        }
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
        return TacticalMetrics(
            observed_interceptor_count=len(interceptor_ids),
            alive_count=alive_count,
            event_revision=self.event_revision,
            assigned_target_counts=tuple(sorted(self.assigned_by_target.items())),
        )

    def action_for(self, entity_id: int, step: int) -> list[float] | None:
        """Return a target ID or a public search waypoint for an airborne unit."""

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
        if self.policy_name in {"r2_static_assignment", "r3_static_priority"}:
            return False
        if self.policy_name in {
            "r6_pressure_assignment", "r7_strike_packages", "r7_static_search",
            "r8_satellite_packages", "r9_hierarchical_learning",
        }:
            return step - self.last_plan_step >= self.rules.adaptive_replan_interval
        if self.policy_name == "r5_event_rolling":
            return (
                self.event_revision > self.planned_revision
                and step - self.last_plan_step >= self.rules.event_replan_cooldown
            )
        return step - self.last_plan_step >= self.rules.replan_interval

    def _plan(self, step: int) -> None:
        platforms = tuple(sorted(self.reports.values(), key=lambda item: item.entity_id))
        decision = self.policy.decide(BaselineObservation(step, platforms, self.targets, self.tactical_metrics))
        initial_plan = self.last_plan_step < 0
        if not initial_plan and self.policy_name == "r4_rolling_rules":
            decision = decision[:math.ceil(sum(item.alive for item in platforms) * 0.25)]
        assignments = {item.platform_id: item for item in decision}
        targets = {item.entity_id: item for item in self.targets}
        if initial_plan:
            selected = tuple(item for item in platforms if item.alive)
            counts: dict[int, int] = {}
        else:
            selected = tuple(item for item in platforms if item.alive and item.entity_id in assignments)
            counts = dict(self.assigned_by_target)
            for platform in selected:
                previous = self.target_by_platform.get(platform.entity_id)
                if previous is not None:
                    counts[previous] = max(0, counts.get(previous, 1) - 1)

        for platform in selected:
            assignment = assignments.get(platform.entity_id)
            target = targets.get(assignment.target_id) if assignment is not None else None
            is_search = assignment is not None and assignment.destination is not None
            if is_search:
                target = None
                destination = assignment.destination
            else:
                if target is None or not self._target_is_available(platform, target):
                    target = self._best_target(platform, counts)
                destination = target.position if target is not None else self._search_waypoint(platform)
            if target is not None:
                self.target_by_platform[platform.entity_id] = target.entity_id
                counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
                action = [_ACTION_ATTACK_TARGET, float(platform.entity_id), float(target.entity_id), 0.0]
                # A received position update requires a new command even if the
                # objective ID stays the same; the engine resolves that snapshot.
                signature = (target.entity_id, destination.lon, destination.lat)
            else:
                self.target_by_platform.pop(platform.entity_id, None)
                action = [_ACTION_SEARCH_POSITION, float(platform.entity_id), destination.lon, destination.lat]
                signature = (None, destination.lon, destination.lat)
            if initial_plan or self.last_guidance.get(platform.entity_id) != signature:
                self.guidance[platform.entity_id] = action
            self.last_guidance[platform.entity_id] = signature
        self.assigned_by_target = {target_id: count for target_id, count in counts.items() if count > 0}
        self.last_plan_step = step
        self.planned_revision = self.event_revision

    def _target_is_available(self, platform: PlatformState, target: TargetPrior) -> bool:
        return (
            target.alive
            and target_is_compatible(platform.kind, target.entity_type)
            and (target.entity_type in (9400, 9600)
                 or target.entity_id in self.observed_target_ids.get(platform.entity_id, set()))
        )

    def _best_target(self, platform: PlatformState, counts: dict[int, int]) -> TargetPrior | None:
        candidates = [
            target for target in self.targets
            if self._target_is_available(platform, target)
            and counts.get(target.entity_id, 0) < self.rules.target_capacity
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda target: (
            self.rules.value_weight * target.value * engagement_effectiveness(platform.kind, target.entity_type)
            - self.rules.distance_weight * distance_km(platform.position, target.position) / 100.0
            - self.rules.coverage_weight * counts.get(target.entity_id, 0),
            -target.entity_id,
        ))

    def _search_waypoint(self, platform: PlatformState) -> Position:
        if not self.search_polygon:
            return platform.position
        center = Position(
            sum(point.lon for point in self.search_polygon) / len(self.search_polygon),
            sum(point.lat for point in self.search_polygon) / len(self.search_polygon),
            platform.position.alt,
        )
        vertex = self.search_polygon[platform.entity_id % len(self.search_polygon)]
        return Position((center.lon + vertex.lon) / 2.0, (center.lat + vertex.lat) / 2.0, platform.position.alt)

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
            self.assigned_by_target[previous] = max(0, self.assigned_by_target.get(previous, 1) - 1)
        self.target_by_platform[entity_id] = target.entity_id
        self.assigned_by_target[target.entity_id] = self.assigned_by_target.get(target.entity_id, 0) + 1
        self.last_guidance[entity_id] = (target.entity_id, target.position.lon, target.position.lat)
        return [_ACTION_ATTACK_TARGET, float(entity_id), float(target.entity_id), 0.0]

    def target_id_for(self, platform_id: int) -> int | None:
        return self.target_by_platform.get(int(platform_id))

    def target_position_for(self, platform_id: int) -> Position | None:
        """Return the position of this platform's assigned objective, if any."""

        target_id = self.target_by_platform.get(int(platform_id))
        if target_id is None:
            return None
        target = next((item for item in self.targets if item.entity_id == target_id), None)
        return target.position if target is not None else None

    def should_use_satellite(self, platform_id: int) -> bool:
        return int(platform_id) in getattr(self.policy, "satellite_platform_ids", frozenset())

    def learning_task_context(self, platform_id: int) -> tuple[float, float, float, float, float]:
        """Domain-local task features for the decentralized R9 Actor."""

        target_id = self.target_id_for(platform_id)
        target = next((item for item in self.targets if item.entity_id == target_id), None)
        entity_type = target.entity_type if target is not None else -1
        return (
            float(entity_type == 9400), float(entity_type == 9500), float(entity_type == 9600),
            self.tactical_metrics.pressure,
            1.0 if self.reports.get(int(platform_id)) is not None else 0.0,
        )

    def reset(self) -> None:
        self.reports: dict[int, PlatformState] = {}
        self.current_snapshot_ids: frozenset[int] | None = None
        self.absent_platform_ids: frozenset[int] = frozenset()
        self.assigned_by_target: dict[int, int] = {}
        self.target_by_platform: dict[int, int] = {}
        self.guidance: dict[int, list[float]] = {}
        self.last_guidance: dict[int, tuple[int | None, float, float]] = {}
        self.observed_target_ids: dict[int, set[int]] = {}
        self.last_plan_step = -1
        self.event_revision = 0
        self.planned_revision = -1
        self.tactical_metrics = TacticalMetrics()
        self.policy = build_red_baseline(self.policy_name, self.rules, self.seed, self.search_polygon)
        self.targets = self.initial_targets
        self.track_fusion = ReceivedTrackFusion(self.initial_targets)
