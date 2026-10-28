"""R0--R9 target-assignment policies for one communication domain."""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from math import cos, hypot, radians

from .contracts import Position


RED_POLICY_CHOICES = (
    "r0_random",
    "r1_priority",
    "r2_static_assignment",
    "r3_static_priority",
    "r4_rolling_rules",
    "r5_event_rolling",
    "r6_pressure_assignment",
    "r7_strike_packages",
    "r7_static_search",
    "r8_satellite_packages",
    "r9_hierarchical_learning",
)

INTERCEPTOR_PRESSURE_WEIGHT = 0.65


@dataclass(frozen=True)
class PlatformState:
    entity_id: int
    kind: str
    position: Position
    alive: bool = True


@dataclass(frozen=True)
class TargetPrior:
    """A public objective or a target admitted from received observations."""

    entity_id: int
    entity_type: int
    position: Position
    value: float = 1.0
    alive: bool = True


@dataclass(frozen=True)
class Assignment:
    platform_id: int
    target_id: int | None
    score: float
    destination: Position | None = None
    search_group_id: int | None = None


@dataclass(frozen=True)
class TacticalMetrics:
    """Legal, red-side indicators used by adaptive rolling baselines."""

    observed_interceptor_count: int = 0
    alive_count: int = 0
    event_revision: int = 0
    assigned_target_counts: tuple[tuple[int, int], ...] = ()

    @property
    def pressure(self) -> float:
        """Threat indicator derived only from this domain's received tracks."""

        interceptor_pressure = min(1.0, self.observed_interceptor_count / max(1, self.alive_count))
        return INTERCEPTOR_PRESSURE_WEIGHT * interceptor_pressure


@dataclass(frozen=True)
class BaselineObservation:
    step: int
    platforms: tuple[PlatformState, ...]
    targets: tuple[TargetPrior, ...]
    metrics: TacticalMetrics = field(default_factory=TacticalMetrics)

    def active_platforms(self) -> tuple[PlatformState, ...]:
        return tuple(item for item in self.platforms if item.alive)

    def active_targets(self) -> tuple[TargetPrior, ...]:
        return tuple(item for item in self.targets if item.alive)


@dataclass(frozen=True)
class BaselineRules:
    target_capacity: int = 10_000
    value_weight: float = 1.0
    distance_weight: float = 0.25
    coverage_weight: float = 0.75
    replan_interval: int = 20
    event_replan_cooldown: int = 5
    adaptive_replan_interval: int = 5
    probe_window: int = 10
    cautious_pressure: float = 0.20
    probe_reassignment_fraction: float = 0.15
    commit_reassignment_fraction: float = 0.30
    cautious_reassignment_fraction: float = 0.12
    static_search_group_size: int = 3
    static_search_group_spacing_km: float = 5.0
    static_search_low_fraction: float = 0.06
    static_search_min_platforms: int = 9
    static_search_max_range_km: float = 48.0
    static_search_danger_radius_km: float = 20.0


def target_is_compatible(platform_kind: str, target_type: int) -> bool:
    return target_type in ({9400, 9600} if platform_kind in {"H", "M"} else {9500} if platform_kind == "L" else set())

def engagement_effectiveness(platform_kind: str, target_type: int) -> float:
    """Base hit rates used by optimizing assignment policies."""

    return {
        "H": {9400: 0.8, 9600: 0.6, 9500: 0.0},
        "M": {9400: 0.8, 9600: 0.6, 9500: 0.0},
        "L": {9400: 0.05, 9600: 0.05, 9500: 0.8},
    }.get(platform_kind, {}).get(target_type, 0.0)


def distance_km(left: Position, right: Position) -> float:
    latitude_scale = 111.32
    x = (left.lon - right.lon) * latitude_scale * cos(radians((left.lat + right.lat) / 2))
    y = (left.lat - right.lat) * latitude_scale
    return hypot(x, y)


class RedBaseline:
    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        raise NotImplementedError

    def claim_retarget(
        self,
        platform: PlatformState,
        observed_targets: tuple[TargetPrior, ...],
    ) -> TargetPrior | None:
        """Optionally claim one legally observed target for an in-flight platform."""

        return None


class R0RandomPolicy(RedBaseline):
    def __init__(self, rules: BaselineRules, seed: int) -> None:
        self.rules = rules
        self.seed = seed

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        targets = tuple(sorted(observation.active_targets(), key=lambda item: item.entity_id))
        counts: dict[int, int] = {}
        assignments: list[Assignment] = []
        for platform in sorted(observation.active_platforms(), key=lambda item: item.entity_id):
            candidates = [target for target in targets if target_is_compatible(platform.kind, target.entity_type) and counts.get(target.entity_id, 0) < self.rules.target_capacity]
            if not candidates:
                continue
            rng = random.Random(self.seed + observation.step * 1009 + platform.entity_id)
            target = candidates[rng.randrange(len(candidates))]
            assignments.append(Assignment(platform.entity_id, target.entity_id, 0.0))
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
        return tuple(assignments)


class R1PriorityPolicy(RedBaseline):
    def __init__(self, rules: BaselineRules) -> None:
        self.rules = rules

    def score(self, platform: PlatformState, target: TargetPrior, assigned_count: int = 0) -> float:
        return self.rules.value_weight * target.value * engagement_effectiveness(platform.kind, target.entity_type) - self.rules.distance_weight * distance_km(platform.position, target.position) / 100.0 - self.rules.coverage_weight * assigned_count

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        targets = tuple(sorted(observation.active_targets(), key=lambda item: item.entity_id))
        counts: dict[int, int] = {}
        assignments: list[Assignment] = []
        platforms = sorted(observation.active_platforms(), key=lambda item: (item.kind, item.entity_id))
        for platform in platforms:
            candidates = [target for target in targets if target_is_compatible(platform.kind, target.entity_type) and counts.get(target.entity_id, 0) < self.rules.target_capacity]
            if not candidates:
                continue
            target = max(candidates, key=lambda item: (self.score(platform, item, counts.get(item.entity_id, 0)), -item.entity_id))
            score = self.score(platform, target, counts.get(target.entity_id, 0))
            assignments.append(Assignment(platform.entity_id, target.entity_id, score))
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
        return tuple(assignments)


class R2StaticAssignmentPolicy(R1PriorityPolicy):
    """Compute a target-capacity-constrained plan once, then keep it fixed."""

    def __init__(self, rules: BaselineRules) -> None:
        super().__init__(rules)
        self._plan: tuple[Assignment, ...] | None = None

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        if self._plan is not None:
            active_platform_ids = {item.entity_id for item in observation.active_platforms()}
            active_target_ids = {item.entity_id for item in observation.active_targets()}
            return tuple(item for item in self._plan if item.platform_id in active_platform_ids and item.target_id in active_target_ids)

        platforms = tuple(sorted(observation.active_platforms(), key=lambda item: item.entity_id))
        targets = tuple(sorted(observation.active_targets(), key=lambda item: item.entity_id))
        options = tuple(
            tuple(
                sorted(
                    ((self.score(platform, target), target) for target in targets if target_is_compatible(platform.kind, target.entity_type)),
                    key=lambda item: (-item[0], item[1].entity_id),
                )
            )
            for platform in platforms
        )
        selected = self._solve_static_pairs(platforms, options)
        selected.sort(key=lambda item: (item[1].kind, item[1].entity_id))
        self._plan = tuple(
            Assignment(platform.entity_id, target.entity_id, score)
            for score, platform, target in selected
        )
        return self._plan

    def _solve_static_pairs(
        self,
        platforms: tuple[PlatformState, ...],
        options: tuple[tuple[tuple[float, TargetPrior], ...], ...],
    ) -> list[tuple[float, PlatformState, TargetPrior]]:
        """Use exact assignment for small examples and deterministic greedy otherwise."""

        if len(platforms) > 12:
            counts: dict[int, int] = {}
            selected: list[tuple[float, PlatformState, TargetPrior]] = []
            for platform, platform_options in zip(platforms, options):
                candidate = next(
                    (
                        (score, target)
                        for score, target in platform_options
                        if score > 0 and counts.get(target.entity_id, 0) < self.rules.target_capacity
                    ),
                    None,
                )
                if candidate is None:
                    continue
                score, target = candidate
                counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
                selected.append((score, platform, target))
            return selected

        best_score = 0.0
        best: tuple[tuple[float, PlatformState, TargetPrior], ...] = ()
        counts: dict[int, int] = {}
        chosen: list[tuple[float, PlatformState, TargetPrior]] = []

        def visit(index: int, score_total: float) -> None:
            nonlocal best_score, best
            if index == len(platforms):
                candidate = tuple(chosen)
                if score_total > best_score + 1e-12:
                    best_score, best = score_total, candidate
                return
            visit(index + 1, score_total)
            for score, target in options[index]:
                if score <= 0 or counts.get(target.entity_id, 0) >= self.rules.target_capacity:
                    continue
                counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
                chosen.append((score, platforms[index], target))
                visit(index + 1, score_total + score)
                chosen.pop()
                counts[target.entity_id] -= 1

        visit(0, 0.0)
        return list(best)


class R4RollingRulePolicy(R1PriorityPolicy):
    def __init__(self, rules: BaselineRules) -> None:
        super().__init__(rules)
        self._last_replan_step: int | None = None
        self._decision: tuple[Assignment, ...] = ()

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        if self._last_replan_step is None or observation.step - self._last_replan_step >= self.rules.replan_interval:
            self._decision = super().decide(observation)
            self._last_replan_step = observation.step
        active_platform_ids = {item.entity_id for item in observation.active_platforms()}
        active_target_ids = {item.entity_id for item in observation.active_targets()}
        return tuple(item for item in self._decision if item.platform_id in active_platform_ids and item.target_id in active_target_ids)


class R5EventRollingPolicy(R1PriorityPolicy):
    """Reallocate only when the commander receives a legal state-change event."""


class R3StaticPriorityPolicy(R1PriorityPolicy):
    """Cache R1's greedy priority assignment without later reassignment."""

    def __init__(self, rules: BaselineRules) -> None:
        super().__init__(rules)
        self._plan: tuple[Assignment, ...] | None = None

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        if self._plan is None:
            self._plan = super().decide(observation)

        active_platform_ids = {item.entity_id for item in observation.active_platforms()}
        active_target_ids = {item.entity_id for item in observation.active_targets()}
        return tuple(
            item
            for item in self._plan
            if item.platform_id in active_platform_ids and item.target_id in active_target_ids
        )


class AdaptiveRollingPolicy(R1PriorityPolicy):
    """Shared rolling controller for R6--R8 using legal red-side metrics."""

    def _stage(self, observation: BaselineObservation) -> str:
        low_remaining = any(item.kind == "L" for item in observation.active_platforms())
        metrics = observation.metrics
        if low_remaining and observation.step < self.rules.probe_window:
            return "probe"
        if low_remaining and metrics.pressure >= self.rules.cautious_pressure:
            return "absorb"
        return "commit"

    def _reassignment_budget(self, observation: BaselineObservation, stage: str) -> int:
        remaining = len(observation.active_platforms())
        if remaining == 0:
            return 0
        if stage == "probe":
            fraction = self.rules.probe_reassignment_fraction
        elif observation.metrics.pressure >= self.rules.cautious_pressure:
            fraction = self.rules.cautious_reassignment_fraction
        else:
            fraction = self.rules.commit_reassignment_fraction
        return max(1, math.ceil(remaining * fraction))

    def _best_target(
        self,
        platform: PlatformState,
        targets: tuple[TargetPrior, ...],
        counts: dict[int, int],
    ) -> TargetPrior | None:
        candidates = [item for item in targets if target_is_compatible(platform.kind, item.entity_type) and counts.get(item.entity_id, 0) < self.rules.target_capacity]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (self.score(platform, item, counts.get(item.entity_id, 0)), -item.entity_id),
        )

    def _priority_assignments(
        self,
        observation: BaselineObservation,
        platforms: list[PlatformState],
        limit: int,
        counts: dict[int, int] | None = None,
    ) -> tuple[Assignment, ...]:
        targets = tuple(sorted(observation.active_targets(), key=lambda item: item.entity_id))
        counts = dict(observation.metrics.assigned_target_counts) if counts is None else counts
        assignments: list[Assignment] = []
        for platform in platforms:
            if len(assignments) >= limit:
                break
            target = self._best_target(platform, targets, counts)
            if target is None:
                continue
            score = self.score(platform, target, counts.get(target.entity_id, 0))
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
            assignments.append(Assignment(platform.entity_id, target.entity_id, score))
        return tuple(assignments)


class R6PressureAssignmentPolicy(AdaptiveRollingPolicy):
    """Prioritize L tasks under pressure; limit how many airborne units replan."""

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        stage = self._stage(observation)
        active = tuple(sorted(observation.active_platforms(), key=lambda item: item.entity_id))
        if stage in {"probe", "absorb"}:
            candidates = [item for item in active if item.kind == "L"]
        else:
            candidates = [item for kind in ("H", "M", "L") for item in active if item.kind == kind]
        return self._priority_assignments(observation, candidates, self._reassignment_budget(observation, stage))


class R7StrikePackagePolicy(AdaptiveRollingPolicy):
    """Pressure-aware reassignment with H/M pairs sharing a strike target."""

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        stage = self._stage(observation)
        active = tuple(sorted(observation.active_platforms(), key=lambda item: item.entity_id))
        budget = self._reassignment_budget(observation, stage)
        if stage in {"probe", "absorb"}:
            low = [item for item in active if item.kind == "L"]
            return self._priority_assignments(observation, low, budget)

        targets = tuple(sorted(observation.active_targets(), key=lambda item: item.entity_id))
        counts = dict(observation.metrics.assigned_target_counts)
        assignments: list[Assignment] = []
        high = [item for item in active if item.kind == "H"]
        medium = [item for item in active if item.kind == "M"]
        low = [item for item in active if item.kind == "L"]
        paired = min(len(high), len(medium))
        processed_pairs = 0
        for index in range(paired):
            if len(assignments) + 2 > budget:
                break
            h_platform, m_platform = high[index], medium[index]
            target = self._best_target(h_platform, targets, counts)
            if target is None:
                break
            h_score = self.score(h_platform, target, counts.get(target.entity_id, 0))
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
            m_score = self.score(m_platform, target, counts.get(target.entity_id, 0))
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
            assignments.extend((
                Assignment(h_platform.entity_id, target.entity_id, h_score),
                Assignment(m_platform.entity_id, target.entity_id, m_score),
            ))
            processed_pairs += 1

        paired_ids = {item.entity_id for item in high[:processed_pairs] + medium[:processed_pairs]}
        remaining = [item for item in high + medium + low if item.entity_id not in paired_ids]
        if len(assignments) < budget:
            assignments.extend(self._priority_assignments(observation, remaining, budget - len(assignments), counts))
        return tuple(sorted(assignments, key=lambda item: item.platform_id))


class R8SatellitePackagePolicy(R7StrikePackagePolicy):
    """R7 packages plus one planned H-platform satellite use per strike target."""

    def __init__(self, rules: BaselineRules) -> None:
        super().__init__(rules)
        self.satellite_platform_ids: frozenset[int] = frozenset()

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        assignments = super().decide(observation)
        platform_by_id = {item.entity_id: item for item in observation.platforms}
        target_by_id = {item.entity_id: item for item in observation.targets}
        selected: set[int] = set()
        covered_targets: set[int] = set()
        for assignment in assignments:
            platform = platform_by_id[assignment.platform_id]
            target = target_by_id[assignment.target_id]
            if (
                platform.kind == "H"
                and target.entity_type in {9400, 9600}
                and assignment.target_id not in covered_targets
            ):
                selected.add(assignment.platform_id)
                covered_targets.add(assignment.target_id)
        self.satellite_platform_ids = frozenset(selected)
        return assignments


class R9HierarchicalLearningPolicy(R8SatellitePackagePolicy):
    """R8's legal task layer for the hierarchical PPO/MAPPO motion policy.

    This class intentionally keeps target allocation deterministic.  The R9
    learning actor operates beneath it, choosing only each assigned platform's
    lateral maneuver from isolated observations plus this task context.
    """


def build_red_baseline(
    name: str,
    rules: BaselineRules,
    seed: int,
    search_polygon: tuple[Position, ...] | None = None,
) -> RedBaseline:
    if name == "r0_random":
        return R0RandomPolicy(rules, seed)
    if name == "r1_priority":
        return R1PriorityPolicy(rules)
    if name == "r2_static_assignment":
        return R2StaticAssignmentPolicy(rules)
    if name == "r3_static_priority":
        return R3StaticPriorityPolicy(rules)
    if name == "r4_rolling_rules":
        return R4RollingRulePolicy(rules)
    if name == "r5_event_rolling":
        return R5EventRollingPolicy(rules)
    if name == "r6_pressure_assignment":
        return R6PressureAssignmentPolicy(rules)
    if name == "r7_strike_packages":
        return R7StrikePackagePolicy(rules)
    if name == "r7_static_search":
        if search_polygon is None:
            raise ValueError("r7_static_search requires a public search polygon")
        from .r7_static_search import R7StaticSearchPolicy

        return R7StaticSearchPolicy(rules, search_polygon)
    if name == "r8_satellite_packages":
        return R8SatellitePackagePolicy(rules)
    if name == "r9_hierarchical_learning":
        return R9HierarchicalLearningPolicy(rules)
    raise ValueError(f"Unknown red baseline '{name}'")
