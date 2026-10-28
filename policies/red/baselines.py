"""Small, comparable R0--R8 launch-allocation baselines for core priors."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from math import cos, hypot, radians

from .contracts import Position


RED_POLICY_CHOICES = (
    "r0_random",
    "r1_priority",
    "r2_static_assignment",
    "r3_wave_schedule",
    "r4_rolling_rules",
    "r5_event_rolling",
    "r6_frontload_decoy",
    "r7_strike_packages",
    "r7_static_search",
    "r7_grid_sweep",
    "r8_satellite_packages",
    "r9_hierarchical_learning",
    "r9_grid_sweep",
    "r9_search_enhanced",
    "r10_uniform_satellite",
    "r11_lethal_package",
    "windowed_salvo",  # 别名，保留以兼容新命名文档
)


@dataclass(frozen=True)
class PlatformState:
    entity_id: int
    kind: str
    position: Position
    alive: bool = True
    launched: bool = False


@dataclass(frozen=True)
class TargetPrior:
    """One target from ``TrainingEnv._get_init_ship_observation()``."""

    entity_id: int
    entity_type: int
    position: Position
    value: float = 1.0
    alive: bool = True


@dataclass(frozen=True)
class Assignment:
    platform_id: int
    target_id: int
    launch_step: int
    score: float
    destination: Position | None = None
    search_group_id: int | None = None


@dataclass(frozen=True)
class TacticalMetrics:
    """Legal, red-side indicators used by adaptive rolling baselines."""

    observed_interceptor_count: int = 0
    launched_count: int = 0
    alive_count: int = 0
    lost_count: int = 0
    total_count: int = 0
    event_revision: int = 0
    assigned_target_counts: tuple[tuple[int, int], ...] = ()
    # Expected damage already committed per target.  Counts alone cannot tell a
    # mother covered by two H sorties (24 hp, 16 damage each) from one covered
    # by six M sorties, so the damage-aware baselines read this instead.
    assigned_damage: tuple[tuple[int, float], ...] = ()

    @property
    def pressure(self) -> float:
        """Exposure indicator from own losses and locally received tracks."""

        launched = max(1, self.launched_count)
        interceptor_pressure = min(1.0, self.observed_interceptor_count / launched)
        loss_pressure = self.lost_count / max(1, self.total_count)
        return min(1.0, 0.65 * interceptor_pressure + 0.35 * loss_pressure)


@dataclass(frozen=True)
class BaselineObservation:
    step: int
    platforms: tuple[PlatformState, ...]
    targets: tuple[TargetPrior, ...]
    metrics: TacticalMetrics = field(default_factory=TacticalMetrics)

    def active_platforms(self) -> tuple[PlatformState, ...]:
        return tuple(item for item in self.platforms if item.alive and not item.launched)

    def active_targets(self) -> tuple[TargetPrior, ...]:
        return tuple(item for item in self.targets if item.alive)


@dataclass(frozen=True)
class BaselineRules:
    target_capacity: int = 10_000
    min_launch_interval: int = 0
    wave_by_kind: tuple[tuple[str, int], ...] = (("H", 0), ("M", 10), ("L", 20))
    value_weight: float = 1.0
    distance_weight: float = 0.25
    coverage_weight: float = 0.75
    replan_interval: int = 20
    event_replan_cooldown: int = 5
    adaptive_replan_interval: int = 5
    probe_window: int = 10
    cautious_pressure: float = 0.20
    probe_release_fraction: float = 0.15
    commit_release_fraction: float = 0.30
    cautious_release_fraction: float = 0.12
    package_high_delay: int = 3
    wave_count: int = 3
    wave_interval: int = 40
    static_search_group_size: int = 3
    static_search_group_spacing_km: float = 5.0
    static_search_low_fraction: float = 0.06
    static_search_min_platforms: int = 9
    static_search_max_range_km: float = 48.0
    static_search_danger_radius_km: float = 20.0
    # Policy time is the TrainingEnv step index; final24 uses one second/step.
    static_search_lead_steps: int = 700

    def wave_delay(self, kind: str) -> int:
        return dict(self.wave_by_kind).get(kind, 0)

    def capacity(self, platform_kind: str, target_type: int) -> int:
        """How many sorties of this kind may be assigned to one target."""

        expected = damage_capacity(platform_kind, target_type)
        if expected <= 0:
            return 0
        return min(self.target_capacity, expected)


def engagement_effectiveness(platform_kind: str, target_type: int) -> float:
    """Core's base hit-rate table, used only by optimizing baselines."""

    return {
        "H": {9400: 0.8, 9600: 0.6, 9500: 0.0},
        "M": {9400: 0.8, 9600: 0.6, 9500: 0.0},
        "L": {9400: 0.05, 9600: 0.05, 9500: 0.8},
    }.get(platform_kind, {}).get(target_type, 0.0)


# launch20 objective health (scenarios/cases/launch20/<tier>/<map>/scenario.json)
# and the engine's per-kind damage points (H 20, M 5, L 1).  A platform is a
# single-use kamikaze, so a target must be covered by enough sorties to remove
# its whole health bar: two H sorties per mother (24 / (20 * 0.8) = 1.5 -> 2),
# two H sorties per sentry nest (18 / (20 * 0.6) = 1.5 -> 2) and two L sorties
# per dark nest (1.2 / (1 * 0.8) = 1.5 -> 2).
TARGET_HEALTH = {9400: 24.0, 9600: 18.0, 9500: 1.2}
PLATFORM_DAMAGE = {"H": 20.0, "M": 5.0, "L": 1.0}
# A sortie only counts once the single-use platform reaches the 100 m kill
# sphere.  Stage A measured how many launched sorties actually reach one, on
# launch20/E01 (it3, seed 701, 3000 steps): H 4/17 = 0.24, M 1/24 = 0.04,
# L 16/85 = 0.19.  The nominal hit table overstates a sortie by roughly 4x for
# heavy and medium launchers, so the health budget below is scaled by what
# actually lands; otherwise two optimistic H sorties would "saturate" a
# 24-health mother that really needs a whole salvo.
#
# 0.29 / 0.05 / 0.18 are the first-cut parameters the stage-B and stage-C
# calibration runs (`diag_b_E01`, `diag_c2_E01`) were measured with.  The full
# run gives 0.24 / 0.04 / 0.19; re-check them together with the capacity table
# against the sampled stage-D baseline before freezing this table.
APPROACH_EFFECTIVENESS = {"H": 0.29, "M": 0.05, "L": 0.18}

def expected_damage(platform_kind: str, target_type: int) -> float:
    """Mean damage one launched sortie actually delivers to this target type."""

    return (
        PLATFORM_DAMAGE.get(platform_kind, 0.0)
        * engagement_effectiveness(platform_kind, target_type)
        * APPROACH_EFFECTIVENESS.get(platform_kind, 1.0)
    )


def damage_capacity(platform_kind: str, target_type: int) -> int:
    """Sorties needed to remove one target's full health bar (0: cannot hit)."""

    health = TARGET_HEALTH.get(target_type)
    expected = expected_damage(platform_kind, target_type)
    if not health or expected <= 0:
        return 0
    return max(1, math.ceil(health / expected))


def target_is_compatible(platform_kind: str, target_type: int) -> bool:
    """Whether the 10-19 engine authorizes this platform kind to attack this type.

    Heavy and medium launchers may only attack nests/sites (9400/9600); the
    low-performance launcher is the sole platform allowed to attack hidden
    nests (9500).  The engine enforces the same table.
    """

    return target_type in {
        "H": frozenset({9400, 9600}),
        "M": frozenset({9400, 9600}),
        "L": frozenset({9500}),
    }.get(platform_kind, frozenset())


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
        for offset, platform in enumerate(sorted(observation.active_platforms(), key=lambda item: item.entity_id)):
            candidates = [target for target in targets if counts.get(target.entity_id, 0) < self.rules.target_capacity]
            if not candidates:
                break
            rng = random.Random(self.seed + observation.step * 1009 + platform.entity_id)
            target = candidates[rng.randrange(len(candidates))]
            assignments.append(Assignment(platform.entity_id, target.entity_id, observation.step + self.rules.wave_delay(platform.kind) + offset * self.rules.min_launch_interval, 0.0))
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
        platforms = sorted(observation.active_platforms(), key=lambda item: (self.rules.wave_delay(item.kind), item.entity_id))
        for offset, platform in enumerate(platforms):
            candidates = [target for target in targets if counts.get(target.entity_id, 0) < self.rules.target_capacity]
            if not candidates:
                break
            target = max(candidates, key=lambda item: (self.score(platform, item, counts.get(item.entity_id, 0)), -item.entity_id))
            score = self.score(platform, target, counts.get(target.entity_id, 0))
            assignments.append(Assignment(platform.entity_id, target.entity_id, observation.step + self.rules.wave_delay(platform.kind) + offset * self.rules.min_launch_interval, score))
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
                    ((self.score(platform, target), target) for target in targets),
                    key=lambda item: (-item[0], item[1].entity_id),
                )
            )
            for platform in platforms
        )
        selected = self._solve_static_pairs(platforms, options)
        selected.sort(key=lambda item: (self.rules.wave_delay(item[1].kind), item[1].entity_id))
        self._plan = tuple(
            Assignment(platform.entity_id, target.entity_id, observation.step + self.rules.wave_delay(platform.kind) + offset * self.rules.min_launch_interval, score)
            for offset, (score, platform, target) in enumerate(selected)
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


class R3WaveSchedulePolicy(R1PriorityPolicy):
    """Compute one capability-aware plan, then execute it in fixed waves.

    R3 deliberately has no new observation feedback.  It is the controlled
    multi-wave precursor to R4: each platform kind is split evenly across
    a fixed number of waves, and the launch plan is never recomputed.
    """

    def __init__(self, rules: BaselineRules) -> None:
        super().__init__(rules)
        self._plan: tuple[Assignment, ...] | None = None

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        if self._plan is None:
            initial = super().decide(observation)
            platform_by_id = {item.entity_id: item for item in observation.platforms}
            totals: dict[str, int] = {}
            for assignment in initial:
                platform = platform_by_id[assignment.platform_id]
                totals[platform.kind] = totals.get(platform.kind, 0) + 1

            seen: dict[str, int] = {}
            wave_count = max(1, self.rules.wave_count)
            scheduled: list[Assignment] = []
            for assignment in initial:
                platform = platform_by_id[assignment.platform_id]
                ordinal = seen.get(platform.kind, 0)
                seen[platform.kind] = ordinal + 1
                wave_index = min(
                    wave_count - 1,
                    ordinal * wave_count // max(1, totals[platform.kind]),
                )
                scheduled.append(
                    Assignment(
                        platform_id=assignment.platform_id,
                        target_id=assignment.target_id,
                        launch_step=(
                            observation.step
                            + self.rules.wave_delay(platform.kind)
                            + wave_index * self.rules.wave_interval
                        ),
                        score=assignment.score,
                    )
                )
            self._plan = tuple(scheduled)

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
        if low_remaining and (metrics.launched_count == 0 or observation.step < self.rules.probe_window):
            return "probe"
        if low_remaining and metrics.pressure >= self.rules.cautious_pressure:
            return "absorb"
        return "commit"

    def _release_budget(self, observation: BaselineObservation, stage: str) -> int:
        remaining = len(observation.active_platforms())
        if remaining == 0:
            return 0
        if stage == "probe":
            fraction = self.rules.probe_release_fraction
        elif observation.metrics.pressure >= self.rules.cautious_pressure:
            fraction = self.rules.cautious_release_fraction
        else:
            fraction = self.rules.commit_release_fraction
        return max(1, math.ceil(remaining * fraction))

    def _best_target(
        self,
        platform: PlatformState,
        targets: tuple[TargetPrior, ...],
        counts: dict[int, int],
        damage: dict[int, float] | None = None,
    ) -> TargetPrior | None:
        damage = {} if damage is None else damage
        candidates = [
            item for item in targets
            if counts.get(item.entity_id, 0) < self.rules.target_capacity
            and target_is_compatible(platform.kind, item.entity_type)
            and self._remaining_health(item, damage) > 0
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                self._marginal_score(platform, item, damage),
                -item.entity_id,
            ),
        )

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
        """Score one sortie by the health it can actually remove.

        A mother has 24 health and an H sortie is expected to remove about five,
        so the first five heavies are worth their full value and a sixth only
        removes the remainder.  The coverage penalty is the share of the health
        bar already committed: a flat per-sortie penalty would push a well
        covered mother's next heavy onto a healthy sentry nest, splitting the
        salvos that actually kill an objective.
        """

        health = TARGET_HEALTH.get(target.entity_type) or 1.0
        expected = expected_damage(platform.kind, target.entity_type)
        useful = min(expected, self._remaining_health(target, damage))
        committed = min(1.0, damage.get(target.entity_id, 0.0) / health)
        return (
            self.rules.value_weight * target.value * useful / health
            - self.rules.distance_weight * distance_km(platform.position, target.position) / 100.0
            - self.rules.coverage_weight * committed
        )

    def _priority_assignments(
        self,
        observation: BaselineObservation,
        platforms: list[PlatformState],
        limit: int,
        counts: dict[int, int] | None = None,
        damage: dict[int, float] | None = None,
    ) -> tuple[Assignment, ...]:
        targets = tuple(sorted(observation.active_targets(), key=lambda item: item.entity_id))
        counts = dict(observation.metrics.assigned_target_counts) if counts is None else counts
        if damage is None:
            damage = dict(observation.metrics.assigned_damage)
        else:
            damage = dict(damage)
        assignments: list[Assignment] = []
        for platform in platforms:
            if len(assignments) >= limit:
                break
            target = self._best_target(platform, targets, counts, damage)
            if target is None:
                break
            score = self._marginal_score(platform, target, damage)
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
            damage[target.entity_id] = damage.get(target.entity_id, 0.0) + expected_damage(
                platform.kind, target.entity_type
            )
            assignments.append(Assignment(platform.entity_id, target.entity_id, observation.step, score))
        return tuple(assignments)


class R6FrontloadDecoyPolicy(AdaptiveRollingPolicy):
    """Adaptively release a bounded L probe before rolling main-strike batches."""

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        stage = self._stage(observation)
        active = tuple(sorted(observation.active_platforms(), key=lambda item: item.entity_id))
        if stage in {"probe", "absorb"}:
            candidates = [item for item in active if item.kind == "L"]
        else:
            candidates = [item for kind in ("H", "M", "L") for item in active if item.kind == kind]
        return self._priority_assignments(observation, candidates, self._release_budget(observation, stage))


class R7StrikePackagePolicy(AdaptiveRollingPolicy):
    """R6's adaptive release controller with rolling H/M strike packages."""

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        stage = self._stage(observation)
        active = tuple(sorted(observation.active_platforms(), key=lambda item: item.entity_id))
        budget = self._release_budget(observation, stage)
        if stage in {"probe", "absorb"}:
            low = [item for item in active if item.kind == "L"]
            return self._priority_assignments(observation, low, budget)

        targets = tuple(sorted(observation.active_targets(), key=lambda item: item.entity_id))
        counts = dict(observation.metrics.assigned_target_counts)
        damage = dict(observation.metrics.assigned_damage)
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
            target = self._best_target(h_platform, targets, counts, damage)
            if target is None:
                break
            h_score = self._marginal_score(h_platform, target, damage)
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
            damage[target.entity_id] = damage.get(target.entity_id, 0.0) + expected_damage(
                h_platform.kind, target.entity_type
            )
            m_target = self._best_target(m_platform, targets, counts, damage)
            if m_target is None:
                break
            m_score = self._marginal_score(m_platform, m_target, damage)
            counts[m_target.entity_id] = counts.get(m_target.entity_id, 0) + 1
            damage[m_target.entity_id] = damage.get(m_target.entity_id, 0.0) + expected_damage(
                m_platform.kind, m_target.entity_type
            )
            assignments.extend((
                Assignment(h_platform.entity_id, target.entity_id, observation.step + self.rules.package_high_delay, h_score),
                Assignment(m_platform.entity_id, m_target.entity_id, observation.step, m_score),
            ))
            processed_pairs += 1

        paired_ids = {item.entity_id for item in high[:processed_pairs] + medium[:processed_pairs]}
        remaining = [item for item in high + medium + low if item.entity_id not in paired_ids]
        if len(assignments) < budget:
            assignments.extend(
                self._priority_assignments(observation, remaining, budget - len(assignments), counts, damage)
            )
        return tuple(sorted(assignments, key=lambda item: (item.launch_step, item.platform_id)))


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

    Setting ``R9_GRID_SWEEP_PLATFORMS`` (or ``enable_grid_sweep``) prepends the
    eastbound lane sweep used by ``r7_grid_sweep``: a wall of L platforms flies
    due east to find hidden 9500 ships, while the learned evasion policy keeps
    them alive underneath.  The sweep only changes allocation and destinations,
    never the motion stack.

    The named baselines ``r9_grid_sweep`` (mechanism name) and
    ``r9_search_enhanced`` (delivery name) pass ``enable_grid_sweep=True`` so the
    sweep is on by default (85 lanes), while plain ``r9_hierarchical_learning``
    stays bit-exact with the upstream baseline and only sweeps when the
    environment variable asks for it.
    """

    DEFAULT_SWEEP_PLATFORMS = 85
    DEFAULT_SWEEP_LEAD_STEPS = 240
    DEFAULT_SWEEP_CLAIMS_PER_SHIP = 3

    def __init__(
        self,
        rules: BaselineRules,
        search_polygon: tuple[Position, ...] | None = None,
        enable_grid_sweep: bool | int | None = None,
    ) -> None:
        super().__init__(rules)
        self.layer = None
        if enable_grid_sweep is True:
            platforms = int(
                os.environ.get("R9_GRID_SWEEP_PLATFORMS", str(self.DEFAULT_SWEEP_PLATFORMS))
            )
        elif enable_grid_sweep is not None:
            platforms = int(enable_grid_sweep)
        else:
            platforms = int(os.environ.get("R9_GRID_SWEEP_PLATFORMS", "0"))
        if platforms <= 0:
            return
        if search_polygon is None:
            raise ValueError("r9 grid sweep requires a public search polygon")
        from .r7_grid_sweep import GridSweepLayer

        self.layer = GridSweepLayer(
            search_polygon,
            sweep_platform_target=platforms,
            sweep_lead_steps=int(
                os.environ.get("R9_GRID_SWEEP_LEAD", str(self.DEFAULT_SWEEP_LEAD_STEPS))
            ),
            max_claims_per_ship=int(
                os.environ.get(
                    "R9_GRID_SWEEP_SHIP_CLAIMS", str(self.DEFAULT_SWEEP_CLAIMS_PER_SHIP)
                )
            ),
        )

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        if self.layer is not None:
            if not self.layer.planned:
                lane_platforms, east_lon = self.layer.lane_plan(observation)
                base = tuple(
                    Assignment(item.entity_id, -1, observation.step, 0.0)
                    for item in lane_platforms
                )
                assignments = self.layer.decorate(base, lane_platforms, east_lon)
                self.layer.mark_planned(observation.step, assignments)
                return assignments
            if self.layer.holding(observation):
                return ()
        return super().decide(observation)

    def claim_retarget(self, platform, observed_targets):
        if self.layer is not None:
            target = self.layer.claim(platform, observed_targets)
            if target is not None:
                return target
        return super().claim_retarget(platform, observed_targets)

    def _best_target(self, platform, targets, counts, damage=None):
        if self.layer is not None:
            ship = self.layer.preferred_ship(platform, targets, counts)
            if ship is not None:
                return ship
        return super()._best_target(platform, targets, counts, damage)


class R10UniformSatellitePolicy(R8SatellitePackagePolicy):
    """One-shot uniform assignment with a single H satellite leader.

    Heavy and medium sorties share the eleven surface objectives (four nests and
    seven sites) and rotate over them, so every objective receives an even share
    of the salvo.  Low sorties rotate over the ships the satellite reveals (one
    hit does 0.8 damage, so a dark nest needs two sorties).  Every live platform
    is assigned once and keeps its target; only a platform whose target died is
    re-issued, against a live target of the same class.  The commander re-plans
    every 20 steps, so re-issuing everyone each time would steer in-flight
    sorties onto a fresh target and spoil their terminal aim.

    Concentrating heavy fire on the nests instead was measured to score lower on
    the current launch20 maps: the rotating pool keeps the design's even split,
    which is also the only assignment that does not depend on a hit-rate guess.
    """

    _SURFACE_TYPES = (9400, 9600)
    _SHIP_TYPE = 9500

    def __init__(self, rules: BaselineRules) -> None:
        super().__init__(rules)
        self._assignments: dict[int, int] = {}
        self._cursor: dict[int, int] = {}
        self._satellite_leader: int | None = None

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        platforms = sorted(
            observation.active_platforms(), key=lambda item: item.entity_id
        )
        targets = sorted(
            observation.active_targets(), key=lambda item: (item.entity_type, item.entity_id)
        )
        pools: dict[int, list[TargetPrior]] = {}
        for target in targets:
            pools.setdefault(target.entity_type, []).append(target)
        alive_platforms = {item.entity_id for item in platforms}
        live_targets = {item.entity_id for item in targets}
        self._assignments = {
            platform_id: target_id
            for platform_id, target_id in self._assignments.items()
            if platform_id in alive_platforms
        }

        assignments: list[Assignment] = []
        for platform in platforms:
            current = self._assignments.get(platform.entity_id)
            if current is not None and current in live_targets:
                continue
            target = self._take_target(platform.kind, pools)
            if target is None:
                self._assignments.pop(platform.entity_id, None)
                continue
            self._assignments[platform.entity_id] = target.entity_id
            assignments.append(
                Assignment(
                    platform_id=platform.entity_id,
                    target_id=target.entity_id,
                    launch_step=observation.step,
                    score=0.0,
                )
            )

        self.satellite_platform_ids = frozenset(self._leader(platforms))
        return tuple(assignments)

    def _take_target(
        self, kind: str, pools: dict[int, list[TargetPrior]]
    ) -> TargetPrior | None:
        if kind == "L":
            pool = pools.get(self._SHIP_TYPE, [])
        else:
            pool = [
                target
                for entity_type in self._SURFACE_TYPES
                for target in pools.get(entity_type, [])
            ]
        if not pool:
            return None
        key = "L" if kind == "L" else "surface"
        cursor = self._cursor.get(key, 0)
        self._cursor[key] = cursor + 1
        return pool[cursor % len(pool)]
        return None

    def _leader(self, platforms: tuple[PlatformState, ...]) -> set[int]:
        alive = {item.entity_id for item in platforms}
        if self._satellite_leader is not None and self._satellite_leader in alive:
            return {self._satellite_leader}
        self._satellite_leader = next(
            (item.entity_id for item in platforms if item.kind == "H"), None
        )
        return {self._satellite_leader} if self._satellite_leader is not None else set()


class R11LethalPackagePolicy(R8SatellitePackagePolicy):
    """Fixed-id lethal fire packages (the agreed eight-point design).

    Heavy and medium sorties are allocated as fire packages sized to the
    objective instead of rotating one sortie per objective: a nest (24 hp)
    receives two heavy sorties (20 nominal damage each) and a site (18 hp) one,
    the medium salvo adds two sorties to every surface objective, and the heavy
    surplus reinforces the sites because a single heavy sortie only covers 60%
    of a site's hit points at the measured hit rate.  Low sorties split the
    detected dark nests evenly, but only once the one-shot satellite has
    published the whole surface catalogue: a partial catalogue (the first
    dark-nest track can arrive while the cluster knows a single ship) would
    concentrate the entire low salvo on one hull, and fixed ids would never
    free it again.  A platform keeps the target id it was first
    given: every re-plan re-emits the same platform/target pair so the
    commander can refresh that objective's coordinates (point six of the
    design), but the id never changes, so a rebuilding communication cluster
    cannot steer a sortie onto a different objective.  Dead targets are dropped
    instead of replaced.  The satellite is requested once by a single heavy
    leader, but only when step ``R11_SATELLITE_STEP`` (default 280) opens the
    window: the eye lasts three minutes, so requesting it at the opening step
    leaves it spent long before the heavy sorties arrive.  The leader is
    re-elected if it dies before that step.
    """

    _NEST_TYPE = 9400
    _SITE_TYPE = 9600
    _SHIP_TYPE = 9500
    _NEST_HEAVY = 2
    _SITE_HEAVY = 1
    _MEDIUM_PER_SURFACE = 2
    _MEDIUM_DAMAGE = 3
    _SATELLITE_MINUTES = 3
    _HEAVY_SPEED_KM_S = 0.2

    # The environment builds one commander -- and therefore one policy instance
    # -- per red communication domain and replaces it whenever domains merge or
    # split.  The agreed design keeps a platform's first target across those
    # rebuilds (design point five), so the assignment table, the package latch
    # and the satellite leader live on the class instead of on the instance.
    _shared_assignments: dict[int, int] = {}
    _shared_surface_planned = False
    _shared_satellite_leader: int | None = None
    _shared_satellite_requested = False

    @classmethod
    def reset_shared_state(cls) -> None:
        """Drop the cross-domain assignment table (new run, or a test)."""

        cls._shared_assignments.clear()
        cls._shared_surface_planned = False
        cls._shared_satellite_leader = None
        cls._shared_satellite_requested = False

    def __init__(self, rules: BaselineRules) -> None:
        super().__init__(rules)
        self._assignments = type(self)._shared_assignments
        # Point six: refresh the fixed target's coordinates on every re-plan.
        # The commander forwards a command only when that position changed, so
        # this never re-aims a live sortie onto a different objective.
        self._refresh_coordinates = os.environ.get(
            "R11_REFRESH_COORDINATES", "1"
        ).lower() not in {"0", "false", "off", "no"}
        # The satellite is a one-shot three-minute window.  Firing it at the
        # opening step spends the whole bonus while the heavy sorties are still
        # in the air (they arrive around step 300-560), so the request waits for
        # the step where it can actually cover the first strikes.
        self._satellite_step = int(os.environ.get("R11_SATELLITE_STEP", "280"))
        # The low salvo waits for the satellite catalogue.  The first dark-nest
        # track can arrive while this cluster still knows a single ship, and a
        # round robin over a one-element list then commits every low sortie to
        # that hull; the fixed ids never release them.  ``R11_LOW_ASSIGN_STEP``
        # is the first replan after the satellite's 280 request plus delivery,
        # and ``R11_LOW_FALLBACK_STEP`` still fires the salvo when the satellite
        # never opens (leader dead, request rejected, fusion lost).
        self._low_assign_step = int(os.environ.get("R11_LOW_ASSIGN_STEP", "285"))
        self._low_fallback_step = int(os.environ.get("R11_LOW_FALLBACK_STEP", "420"))
        # A catalogue with one or two ships is not a surface grid: keep holding
        # the salvo until enough distinct hulls are visible to spread it over.
        self._low_min_ships = int(os.environ.get("R11_LOW_MIN_SHIPS", "3"))
        # The medium sorties have no sensor of their own, so the only way they
        # can hit anything is to arrive while the satellite tracks are live.
        # Based in the eastern cells every 21001 sits 34-46 km from the two
        # nearest sites (ETA 280-390 s, inside the 280-460 s window).  With a
        # positive focus they stop paying the "two shots on every surface" tax
        # on the objectives the heavy salvo already kills and instead saturate
        # the sites they can actually reach; zero keeps the old behaviour.
        self._medium_focus = int(os.environ.get("R11_MEDIUM_FOCUS", "0"))
        # The heavy salvo kills an 18 hp site only when its hit lands inside the
        # satellite window (20 instead of 12).  Basing spreads the arrivals over
        # 280-560 s, so the two spare heavy sorties are redirected onto the
        # objectives whose predicted arrival falls outside the window.
        self._heavy_eta_guard = os.environ.get(
            "R11_HEAVY_ETA_GUARD", "0"
        ).lower() not in {"0", "false", "off", "no"}

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        platforms = sorted(
            observation.active_platforms(), key=lambda item: item.entity_id
        )
        targets = sorted(
            observation.active_targets(), key=lambda item: (item.entity_type, item.entity_id)
        )
        # The table is shared by every communication cluster: prune nothing here
        # (a single cluster only sees part of the fleet) and never rebind it --
        # other commanders hold the very same mapping.

        assignments: list[Assignment] = []
        nests = [item for item in targets if item.entity_type == self._NEST_TYPE]
        sites = [item for item in targets if item.entity_type == self._SITE_TYPE]
        if not type(self)._shared_surface_planned and (nests or sites):
            type(self)._shared_surface_planned = True
            assignments.extend(
                self._package_surfaces(platforms, nests, sites, observation.step)
            )
        ships = [item for item in targets if item.entity_type == self._SHIP_TYPE]
        if ships:
            assignments.extend(self._split_low(platforms, ships, observation.step))


        if observation.step < self._satellite_step or type(self)._shared_satellite_requested:
            self.satellite_platform_ids = frozenset()
        else:
            leaders = self._leader(platforms)
            self.satellite_platform_ids = frozenset(leaders)
            if leaders:
                type(self)._shared_satellite_requested = True
        decision: dict[int, Assignment] = {
            assignment.platform_id: assignment for assignment in assignments
        }
        if self._refresh_coordinates:
            live_target_ids = {item.entity_id for item in targets}
            for platform_id, target_id in sorted(self._assignments.items()):
                if target_id not in live_target_ids:
                    # The cluster has no track of it (destroyed, or not yet
                    # detected): hold the fixed id but stop commanding it.
                    continue
                decision.setdefault(
                    platform_id,
                    Assignment(
                        platform_id=platform_id,
                        target_id=target_id,
                        launch_step=observation.step,
                        score=0.0,
                    ),
                )
        return tuple(decision.values())

    def claim_retarget(
        self,
        platform,
        observed_targets,
    ):
        """Point six: keep the fixed id and let the engine re-resolve its position.

        The commander asks for one target per airborne platform on every step
        that has no queued guidance, and resolves the objective's coordinates
        from the executor's own track.  Returning the fixed id here is what
        turns the one-shot assignment into a continuously refreshed coordinate
        without ever aiming a different objective.
        """

        target_id = self._assignments.get(platform.entity_id)
        if target_id is None:
            return None
        return next((item for item in observed_targets if item.entity_id == target_id), None)

    def _package_surfaces(
        self,
        platforms: tuple[PlatformState, ...],
        nests: list[TargetPrior],
        sites: list[TargetPrior],
        step: int,
    ) -> list[Assignment]:
        heavy = [item.entity_id for item in platforms if item.kind == "H"]
        medium = [item.entity_id for item in platforms if item.kind == "M"]
        assignments: list[Assignment] = []

        def take(pool: list[int], target: TargetPrior, count: int) -> None:
            while pool and count > 0:
                assignments.append(self._assign(pool.pop(0), target.entity_id, step))
                count -= 1

        for target in nests:
            take(heavy, target, self._NEST_HEAVY)
        for target in sites:
            take(heavy, target, self._SITE_HEAVY)
        if self._heavy_eta_guard:
            # A heavy hit outside the satellite window pays 12 instead of 20,
            # which leaves an 18 hp site alive at 6 hp.  The spare sorties are
            # worth more on exactly those objectives than as a redundant hit
            # where the first shot already arrives inside the window.  Baseline
            # speed is 200 m/s, so the distance alone predicts the arrival.
            heavy_positions = {
                item.entity_id: item.position for item in platforms if item.kind == "H"
            }
            window_start = self._satellite_step
            window_end = window_start + self._SATELLITE_MINUTES * 60
            for target in sites:
                shooters = [
                    item
                    for item in assignments
                    if item.target_id == target.entity_id
                    and item.platform_id in heavy_positions
                ]
                if not shooters:
                    take(heavy, target, 1)
                    continue
                eta = min(
                    distance_km(heavy_positions[item.platform_id], target.position)
                    / self._HEAVY_SPEED_KM_S
                    for item in shooters
                )
                if eta < window_start + 15 or eta > window_end - 15:
                    take(heavy, target, 1)
        for target in sites:
            take(heavy, target, 1)
        for target in nests:
            take(heavy, target, 1)

        if self._medium_focus > 0 and medium and sites:
            # Nearest-first with a per-site cap.  The medium salvo has no radar
            # of its own, so a sortie only scores when its ETA lands inside the
            # satellite window; based in the eastern cells that means the two
            # or three sites closest to each sortie.  A cap stops the most
            # attractive site from eating the whole pool, and every platform
            # walks the ranking itself so basing (not id order) picks the hull.
            cap = max(1, -(-len(medium) // self._medium_focus))
            counts: dict[int, int] = {item.entity_id: 0 for item in sites}
            by_position = {item.entity_id: item.position for item in platforms}
            for entity_id in medium:
                ranked = sorted(
                    sites,
                    key=lambda item: distance_km(by_position[entity_id], item.position),
                )
                target = next(
                    (item for item in ranked if counts[item.entity_id] < cap),
                    ranked[0],
                )
                counts[target.entity_id] += 1
                assignments.append(self._assign(entity_id, target.entity_id, step))
            return assignments

        for target in nests + sites:
            take(medium, target, self._MEDIUM_PER_SURFACE)
        for target in nests:
            take(medium, target, 1)
        for target in sites:
            take(medium, target, 1)
        return assignments

    def _split_low(
        self,
        platforms: tuple[PlatformState, ...],
        ships: list[TargetPrior],
        step: int,
    ) -> list[Assignment]:
        """Spread the low salvo over the ships the satellite catalogue reveals.

        The salvo is held until the one-shot satellite has published the surface
        grid, then assigned nearest-first with an even quota per hull.  Holding
        is what keeps the fire plan honest: a cluster that fires at the first
        dark-nest track knows one ship, so every low sortie would fly (and
        detonate) on the same hull while the rest of the grid is never attacked.
        """

        lows = [
            platform for platform in platforms
            if platform.kind == "L" and platform.entity_id not in self._assignments
        ]
        if not lows or not ships:
            return []
        if step < self._low_fallback_step and (
            step < self._low_assign_step or len(ships) < self._low_min_ships
        ):
            return []
        # Balanced shares: 85 sorties over 10 hulls is 9/8 per hull, not
        # ``ceil`` (9,9,...,4) which would starve the last hulls.
        base, remainder = divmod(len(lows), len(ships))
        ordered = sorted(ships, key=lambda item: item.entity_id)
        quota = {
            ship.entity_id: base + (1 if index < remainder else 0)
            for index, ship in enumerate(ordered)
        }
        load: dict[int, int] = {}
        for target_id in self._assignments.values():
            load[target_id] = load.get(target_id, 0) + 1
        assignments: list[Assignment] = []
        for platform in sorted(lows, key=lambda item: item.entity_id):
            open_ships = [
                ship for ship in ships
                if load.get(ship.entity_id, 0) < quota.get(ship.entity_id, 0)
            ]
            if not open_ships:
                break
            target = min(
                open_ships,
                key=lambda ship: (
                    distance_km(platform.position, ship.position),
                    ship.entity_id,
                ),
            )
            load[target.entity_id] = load.get(target.entity_id, 0) + 1
            assignments.append(self._assign(platform.entity_id, target.entity_id, step))
        return assignments

    def _assign(self, platform_id: int, target_id: int, step: int) -> Assignment:
        self._assignments[platform_id] = target_id
        return Assignment(
            platform_id=platform_id,
            target_id=target_id,
            launch_step=step,
            score=0.0,
        )

    def _leader(self, platforms: tuple[PlatformState, ...]) -> set[int]:
        alive = {item.entity_id for item in platforms}
        leader = type(self)._shared_satellite_leader
        if leader is not None and leader in alive:
            return {leader}
        # Re-elect when the leader died before the satellite window opened (or
        # was never chosen): otherwise the one-shot request is silently lost.
        leader = next((item.entity_id for item in platforms if item.kind == "H"), None)
        type(self)._shared_satellite_leader = leader
        return {leader} if leader is not None else set()


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
    if name == "r3_wave_schedule":
        return R3WaveSchedulePolicy(rules)
    if name == "r4_rolling_rules":
        return R4RollingRulePolicy(rules)
    if name == "r5_event_rolling":
        return R5EventRollingPolicy(rules)
    if name == "r6_frontload_decoy":
        return R6FrontloadDecoyPolicy(rules)
    if name == "r7_strike_packages":
        return R7StrikePackagePolicy(rules)
    if name == "r7_static_search":
        if search_polygon is None:
            raise ValueError("r7_static_search requires a public search polygon")
        from .r7_static_search import R7StaticSearchPolicy

        return R7StaticSearchPolicy(rules, search_polygon)
    if name == "r7_grid_sweep":
        if search_polygon is None:
            raise ValueError("r7_grid_sweep requires a public search polygon")
        from .r7_grid_sweep import R7GridSweepPolicy

        return R7GridSweepPolicy(rules, search_polygon)
    if name == "r8_satellite_packages":
        return R8SatellitePackagePolicy(rules)
    if name == "r9_hierarchical_learning":
        return R9HierarchicalLearningPolicy(rules, search_polygon)
    if name in {"r9_grid_sweep", "r9_search_enhanced"}:
        # ``r9_search_enhanced`` is the delivery-facing name for the same policy;
        # ``r9_grid_sweep`` stays as the mechanism/experiment name.
        if search_polygon is None:
            raise ValueError(f"{name} requires a public search polygon")
        return R9HierarchicalLearningPolicy(rules, search_polygon, enable_grid_sweep=True)
    if name == "r10_uniform_satellite":
        return R10UniformSatellitePolicy(rules)
    if name in {"r11_lethal_package", "windowed_salvo"}:
        return R11LethalPackagePolicy(rules)
    raise ValueError(f"Unknown red baseline '{name}'")
