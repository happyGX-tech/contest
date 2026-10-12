"""Core-facing adapter for R0--R8 without blue-side global observations."""

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
)
from .contracts import Position
from .tracks import InitialCatalogueTrackFusion


_KIND_BY_TYPE = {21000: "H", 21001: "M", 21002: "L"}


class RedBaselineCommander:
    """Coordinate launch allocation from own-platform reports and core priors."""

    def __init__(
        self,
        targets: tuple[TargetPrior, ...],
        policy_name: str,
        seed: int = 1,
        search_polygon: tuple[Position, ...] | None = None,
    ) -> None:
        if policy_name not in RED_POLICY_CHOICES:
            raise ValueError(f"Unsupported red baseline '{policy_name}'")
        self.initial_targets = targets
        self.targets = targets
        self.track_fusion = InitialCatalogueTrackFusion(targets)
        self.policy_name = policy_name
        self.seed = seed
        self.search_polygon = search_polygon
        self.rules = BaselineRules()
        self.policy = build_red_baseline(policy_name, self.rules, seed, search_polygon)
        self.reports: dict[int, PlatformState] = {}
        self.expected_platform_ids: set[int] = set()
        self.launched_ids: set[int] = set()
        self.assigned_by_target: dict[int, int] = {}
        self.target_by_platform: dict[int, int] = {}
        self.pending: dict[int, list[tuple[int, float, float]]] = {}
        self.observed_target_ids: dict[int, set[int]] = {}
        self.last_plan_step = -1
        self.event_revision = 0
        self.planned_revision = -1
        self.tactical_metrics = TacticalMetrics()

    def register_platform(self, entity_id: int) -> None:
        self.expected_platform_ids.add(int(entity_id))

    def begin_step(self, observations: tuple[dict, ...]) -> None:
        """Synchronize the legal reports available to the central red commander.

        The environment supplies only each live platform's already-isolated
        observation.  No blue global state is passed through this method.
        """

        observed_ids = {int(item.get("entity_id", -1)) for item in observations}
        state_changed = False
        for observation in observations:
            self.report(observation)
            state_changed = self.track_fusion.ingest(observation) or state_changed
        for entity_id in self.expected_platform_ids - observed_ids:
            prior = self.reports.get(entity_id)
            if prior is not None and prior.alive:
                self.reports[entity_id] = PlatformState(
                    prior.entity_id,
                    prior.kind,
                    prior.position,
                    alive=False,
                    launched=prior.launched,
                )
                state_changed = True
        self.tactical_metrics = self._build_tactical_metrics(observations)
        if state_changed:
            self.targets = self.track_fusion.targets
            self.event_revision += 1
            self.tactical_metrics = self._build_tactical_metrics(observations)

    def _build_tactical_metrics(self, observations: tuple[dict, ...]) -> TacticalMetrics:
        interceptor_ids: set[int] = set()
        for observation in observations:
            tracks = observation.get("self", {}).get("detectInfo") or {}
            for raw_id, track in tracks.items():
                entity_type = self._track_field(track, "entity_type", -1)
                if int(entity_type) == 24000:
                    interceptor_ids.add(int(self._track_field(track, "entity_id", raw_id)))
        alive_count = sum(item.alive for item in self.reports.values())
        total_count = len(self.expected_platform_ids)
        return TacticalMetrics(
            observed_interceptor_count=len(interceptor_ids),
            launched_count=len(self.launched_ids),
            alive_count=alive_count,
            lost_count=max(0, total_count - alive_count),
            total_count=total_count,
            event_revision=self.event_revision,
            assigned_target_counts=tuple(sorted(self.assigned_by_target.items())),
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
        observed_ship_ids = {
            int(self._track_field(track, "entity_id", raw_id))
            for raw_id, track in (own.get("detectInfo") or {}).items()
            if int(self._track_field(track, "entity_type", -1)) == 9500
        }
        self.observed_target_ids[entity_id] = observed_ship_ids
        self.reports[entity_id] = PlatformState(
            entity_id=entity_id,
            kind=kind,
            position=Position(
                lon=float(position.get("lon", 0.0)),
                lat=float(position.get("lat", 0.0)),
                alt=float(position.get("alt", 0.0)),
            ),
            alive=float(own.get("health", 0)) > 0 and bool(own.get("isVisible", True)),
            launched=entity_id in self.launched_ids,
        )

    def action_for(self, entity_id: int, step: int) -> list[float] | None:
        if self._should_plan(step):
            self._plan(step)
        if int(entity_id) in self.launched_ids:
            return self._retarget_action_for(int(entity_id))
        rows = self.pending.get(int(entity_id), [])
        due = [row for row in rows if row[0] <= step]
        future = [row for row in rows if row[0] > step]
        if future:
            self.pending[int(entity_id)] = future
        else:
            self.pending.pop(int(entity_id), None)
        if not due:
            return None
        _, lon, lat = due[0]
        self.launched_ids.add(int(entity_id))
        return [1.0, float(entity_id), lon, lat]

    def _should_plan(self, step: int) -> bool:
        if not self.targets or len(self.reports) < len(self.expected_platform_ids):
            return False
        if self.policy_name in {"r2_static_assignment", "r3_wave_schedule"}:
            return self.last_plan_step < 0
        if self.policy_name == "r11_launch_timing":
            # 对照实验：immediate/waves3 锁定静态计划只求解一次；
            # r9_timing 需要按 adaptive 周期重复触发释放节奏
            if self.last_plan_step < 0:
                return True
            return getattr(self.policy, "timing_mode", "") == "r9_timing" and (
                step - self.last_plan_step >= self.rules.adaptive_replan_interval
            )
        if self.policy_name in {
            "r6_frontload_decoy",
            "r7_strike_packages",
            "r7_static_search",
            "r8_satellite_packages",
            "r9_hierarchical_learning",
        }:
            return self.last_plan_step < 0 or step - self.last_plan_step >= self.rules.adaptive_replan_interval
        if self.policy_name == "r5_event_rolling":
            # A burst of reports represents one information event, not a burst
            # of allocations. Keep the newest revision pending until the
            # cooldown elapses, then replan once against that newest state.
            return self.last_plan_step < 0 or (
                self.event_revision > self.planned_revision
                and step - self.last_plan_step >= self.rules.event_replan_cooldown
            )
        return self.last_plan_step < 0 or step - self.last_plan_step >= self.rules.replan_interval

    def _plan(self, step: int) -> None:
        platforms = tuple(
            PlatformState(item.entity_id, item.kind, item.position, item.alive, item.entity_id in self.launched_ids)
            for item in self.reports.values()
        )
        decision = self.policy.decide(
            BaselineObservation(
                step=step,
                platforms=platforms,
                targets=self.targets,
                metrics=self.tactical_metrics,
            )
        )
        target_by_id = {item.entity_id: item for item in self.targets}
        rolling_limit = math.ceil(sum(item.alive and not item.launched for item in platforms) * 0.25)
        accepted = 0
        for assignment in decision:
            if self.policy_name == "r4_rolling_rules" and accepted >= rolling_limit:
                break
            if assignment.platform_id in self.launched_ids:
                continue
            if assignment.platform_id in self.pending:
                if self.policy_name not in {
                    "r5_event_rolling",
                    "r6_frontload_decoy",
                    "r7_strike_packages",
                    "r7_static_search",
                    "r8_satellite_packages",
                    "r9_hierarchical_learning",
                }:
                    continue
                self._drop_pending(assignment.platform_id)
            if self.assigned_by_target.get(assignment.target_id, 0) >= self.rules.target_capacity:
                continue
            target = target_by_id.get(assignment.target_id)
            if target is None:
                continue
            destination = assignment.destination or target.position
            self.pending.setdefault(assignment.platform_id, []).append(
                (assignment.launch_step, destination.lon, destination.lat)
            )
            self.assigned_by_target[assignment.target_id] = self.assigned_by_target.get(assignment.target_id, 0) + 1
            self.target_by_platform[assignment.platform_id] = assignment.target_id
            accepted += 1
        self.last_plan_step = step
        self.planned_revision = self.event_revision

    def _retarget_action_for(self, platform_id: int) -> list[float] | None:
        observed_ids = self.observed_target_ids.get(platform_id, set())
        platform = self.reports.get(platform_id)
        if platform is None or not observed_ids:
            return None
        candidates = tuple(
            sorted(
                (
                    target
                    for target in self.targets
                    if target.entity_id in observed_ids and target.entity_type == 9500
                ),
                key=lambda target: (
                    distance_km(platform.position, target.position),
                    target.entity_id,
                ),
            )
        )
        target = self.policy.claim_retarget(platform, candidates)
        if target is None:
            return None

        previous_target = self.target_by_platform.get(platform_id)
        if previous_target is not None:
            self.assigned_by_target[previous_target] = max(
                0,
                self.assigned_by_target.get(previous_target, 1) - 1,
            )
        self.target_by_platform[platform_id] = target.entity_id
        self.assigned_by_target[target.entity_id] = self.assigned_by_target.get(target.entity_id, 0) + 1
        return [2.0, float(platform_id), target.position.lon, target.position.lat]

    def _drop_pending(self, platform_id: int) -> None:
        previous_target = self.target_by_platform.pop(platform_id, None)
        self.pending.pop(platform_id, None)
        if previous_target is not None:
            self.assigned_by_target[previous_target] = max(
                0,
                self.assigned_by_target.get(previous_target, 1) - 1,
            )

    def target_id_for(self, platform_id: int) -> int | None:
        """Return this platform's currently assigned initial-catalogue target."""

        return self.target_by_platform.get(int(platform_id))

    def should_use_satellite(self, platform_id: int) -> bool:
        """Whether this launched platform is the R8 satellite leader.

        The policy selects at most one H platform per 9400/9600 strike target.
        This is deliberately separate from reactive self-protection satellite
        use in the motion layer.
        """

        leaders = getattr(self.policy, "satellite_platform_ids", frozenset())
        return int(platform_id) in leaders

    def learning_task_context(self, platform_id: int) -> tuple[float, float, float, float, float]:
        """Return R9's legal upper-layer context for one platform's Actor."""

        target_id = self.target_id_for(platform_id)
        target = next((item for item in self.targets if item.entity_id == target_id), None)
        target_type = target.entity_type if target is not None else -1
        return (
            float(target_type == 9400),
            float(target_type == 9500),
            float(target_type == 9600),
            self.tactical_metrics.pressure,
            len(self.launched_ids) / max(1, len(self.expected_platform_ids)),
        )

    def reset(self) -> None:
        self.reports.clear()
        self.launched_ids.clear()
        self.assigned_by_target.clear()
        self.target_by_platform.clear()
        self.pending.clear()
        self.observed_target_ids.clear()
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
