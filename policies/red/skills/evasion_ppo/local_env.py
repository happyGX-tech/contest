"""Small encounter environment for curriculum pretraining of the avoid skill.

The environment is intentionally independent from the battle simulator.  It
models only the contract owned by the mid-level skill: preserve the assigned
goal while avoiding up to two locally observed interceptors.  All quantities
that still need confirmation against the official SDK live in
``LocalAvoidConfig`` rather than being hidden in the dynamics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np


LEFT = 0
STRAIGHT = 1
RIGHT = 2
ACTION_COUNT = 3


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _heading(vector: np.ndarray) -> float:
    return math.atan2(float(vector[1]), float(vector[0]))


def _unit(heading: float) -> np.ndarray:
    return np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)


def closest_segment_distance(start: np.ndarray, end: np.ndarray) -> float:
    """Minimum distance to the origin over a linearly interpolated step."""

    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        return float(np.linalg.norm(start))
    fraction = float(np.clip(-np.dot(start, delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(start + fraction * delta))


@dataclass
class LocalAvoidConfig:
    """Physical assumptions and reward terms for Stage 0-3.

    Speeds follow the planning document and can be replaced once the official
    motion API is calibrated.  Distances are metres and time is seconds.
    """

    dt: float = 1.0
    horizon_steps: int = 60
    avoid_angle_deg: float = 15.0
    unit_turn_rates_deg_s: tuple[float, float, float] | None = None
    integrated_turn_actions: bool = False
    straight_recovery_deg: float = 15.0
    control_interval_sec: float = 1.0
    unit_speeds_mps: tuple[float, float, float] = (2000.0, 1200.0, 300.0)
    bug_speed_mps: float = 2500.0
    bug_turn_rate_deg_s: float = 12.0
    hit_radius_m: float = 100.0
    threat_distance_scale_m: float = 60_000.0
    goal_distance_scale_m: float = 100_000.0
    relative_speed_scale_mps: float = 5000.0
    t_cpa_scale_s: float = 60.0
    risk_t_cpa_s: float = 30.0
    risk_d_cpa_m: float = 3000.0
    risk_distance_m: float = 5000.0
    evade_clear_distance_m: float = 2500.0
    evade_clear_steps: int = 3
    arrival_radius_m: float = 500.0
    arrival_segment_check: bool = False
    goal_distance_min_m: float = 40_000.0
    goal_distance_max_m: float = 80_000.0
    bug_spawn_min_m: float = 15_000.0
    bug_spawn_max_m: float = 45_000.0
    stage2_bearing_limit_deg: float = 180.0
    stage2_bearing_min_deg: float = 0.0
    stage3_simultaneous_probability: float = 0.34
    stage3_pincer_probability: float = 0.33
    stage3_same_side_probability: float = 0.0
    stage3_delay_min_steps: int = 5
    stage3_delay_max_steps: int = 15
    stage3_pincer_min_deg: float = 45.0
    stage3_pincer_max_deg: float = 100.0
    stage3_trail_jitter_deg: float = 10.0
    dangerous_spawn_only: bool = False
    dangerous_max_cpa_m: float = 500.0
    progress_scale: float = 1.0
    progress_fraction_scale: float = 0.0
    hit_penalty: float = 20.0
    evade_bonus: float = 3.0
    deviation_penalty: float = 0.05
    oscillation_penalty: float = 0.02
    action_switch_penalty: float = 0.0
    turn_reversal_penalty: float = 0.0
    time_penalty: float = 0.001
    recovery_penalty: float = 0.05
    heading_error_penalty: float = 0.0
    maneuver_burst_limit_steps: int = 0
    excess_maneuver_penalty: float = 0.0
    arrival_bonus: float = 0.0
    timeout_penalty: float = 0.0
    recovery_bonus: float = 0.0
    recovery_heading_tolerance_deg: float = 5.0
    terminate_on_evade: bool = True
    retire_threats_on_evade: bool = False
    actor_risk_gate: bool = False
    actor_action_history: bool = False
    encode_track_source: bool = False
    peer_track_probability: float = 0.0
    satellite_track_probability: float = 0.0
    safety_improvement_scale: float = 0.0
    unsafe_straight_penalty: float = 0.0
    reward_clip: float = 25.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalAvoidEnv:
    """One human unit and up to two interceptor slots with a Gym-like API."""

    MAX_THREATS = 2
    SELF_FEATURES = 3
    GOAL_FEATURES = 3
    THREAT_FEATURES = 9
    CONTEXT_FEATURES = 6
    TYPE_FEATURES = 3
    ACTOR_OBS_DIM = (
        SELF_FEATURES
        + GOAL_FEATURES
        + MAX_THREATS * THREAT_FEATURES
        + CONTEXT_FEATURES
        + TYPE_FEATURES
    )
    CRITIC_EXTRA_PER_THREAT = 5
    CRITIC_STATE_DIM = ACTOR_OBS_DIM + MAX_THREATS * CRITIC_EXTRA_PER_THREAT

    def __init__(
        self,
        config: LocalAvoidConfig | None = None,
        *,
        stage: int = 0,
        seed: int = 0,
    ) -> None:
        if stage not in (0, 1, 2, 3):
            raise ValueError("LocalAvoidEnv supports only stages 0, 1, 2, and 3")
        self.config = config or LocalAvoidConfig()
        self.stage = int(stage)
        self.rng = np.random.default_rng(seed)
        self.own_position = np.zeros(2, dtype=np.float64)
        self.goal_position = np.zeros(2, dtype=np.float64)
        self.own_heading = 0.0
        self.unit_type = 0
        self.bugs: list[dict[str, Any]] = []
        self.step_count = 0
        self.previous_action = STRAIGHT
        self.initial_goal_distance = 0.0
        self.path_length = 0.0
        self.episode_return = 0.0
        self.had_threat = False
        self.clear_steps = 0
        self.closest_bug_distance = math.inf
        self._last_risk = False
        self.oscillation_count = 0
        self.heading_deviation_sum = 0.0
        self.maneuver_streak = 0
        self.evade_achieved = False
        self.recovery_achieved = False
        self.encounter_pattern = "none"

    @property
    def unit_speed(self) -> float:
        return float(self.config.unit_speeds_mps[self.unit_type])

    def reset(
        self,
        scenario: Mapping[str, Any] | None = None,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        scenario = dict(scenario or {})
        self.step_count = 0
        self.previous_action = STRAIGHT
        self.path_length = 0.0
        self.episode_return = 0.0
        self.had_threat = False
        self.clear_steps = 0
        self.closest_bug_distance = math.inf
        self._last_risk = False
        self.oscillation_count = 0
        self.heading_deviation_sum = 0.0
        self.maneuver_streak = 0
        self.evade_achieved = False
        self.recovery_achieved = False
        self.unit_type = int(scenario.get("unit_type", 0))
        if self.unit_type not in (0, 1, 2):
            raise ValueError("unit_type must be 0, 1, or 2")

        self.own_position = np.asarray(
            scenario.get("deploy_xy", (0.0, 0.0)), dtype=np.float64
        )
        goal_heading = float(scenario.get("goal_heading", self.rng.uniform(-math.pi, math.pi)))
        goal_distance = float(
            scenario.get(
                "goal_distance_m",
                self.rng.uniform(
                    self.config.goal_distance_min_m,
                    self.config.goal_distance_max_m,
                ),
            )
        )
        self.goal_position = np.asarray(
            scenario.get(
                "goal_xy", self.own_position + _unit(goal_heading) * goal_distance
            ),
            dtype=np.float64,
        )
        self.own_heading = float(scenario.get("own_heading", goal_heading))
        self.initial_goal_distance = self._goal_distance()
        self.bugs = []
        default_bug_count = 0 if self.stage == 0 else (2 if self.stage == 3 else 1)
        bug_count = int(scenario.get("bug_num", default_bug_count))
        if not 0 <= bug_count <= self.MAX_THREATS:
            raise ValueError("bug_num must be between 0 and 2")
        bearing_offsets: list[float | None] = [None] * bug_count
        release_steps = list(scenario.get("bug_release_steps", [0] * bug_count))
        if len(release_steps) != bug_count:
            raise ValueError("bug_release_steps must match bug_num")
        track_sources = scenario.get("bug_track_sources")
        if track_sources is not None and len(track_sources) != bug_count:
            raise ValueError("bug_track_sources must match bug_num")
        if (
            self.config.peer_track_probability < 0.0
            or self.config.satellite_track_probability < 0.0
            or self.config.peer_track_probability
            + self.config.satellite_track_probability
            > 1.0
        ):
            raise ValueError("track source probabilities must be nonnegative and sum to at most one")
        self.encounter_pattern = str(scenario.get("encounter_pattern", "custom"))
        if self.stage == 3 and bug_count == 2 and "bug_spawn_positions" not in scenario:
            draw = self.rng.random()
            simultaneous = self.config.stage3_simultaneous_probability
            pincer = simultaneous + self.config.stage3_pincer_probability
            if draw < simultaneous:
                self.encounter_pattern = "simultaneous"
                if (
                    self.config.stage3_same_side_probability > 0.0
                    and self.rng.random()
                    < self.config.stage3_same_side_probability
                ):
                    limit = math.radians(self.config.stage2_bearing_limit_deg)
                    minimum = math.radians(self.config.stage2_bearing_min_deg)
                    side = self.rng.choice((-1.0, 1.0))
                    bearing_offsets = [
                        side * self.rng.uniform(minimum, limit),
                        side * self.rng.uniform(minimum, limit),
                    ]
            elif draw < pincer:
                self.encounter_pattern = "pincer"
                low = math.radians(self.config.stage3_pincer_min_deg)
                high = math.radians(self.config.stage3_pincer_max_deg)
                bearing_offsets = [
                    -self.rng.uniform(low, high),
                    self.rng.uniform(low, high),
                ]
            else:
                self.encounter_pattern = "delayed_trail"
                base = self.rng.uniform(-math.pi, math.pi)
                jitter = math.radians(self.config.stage3_trail_jitter_deg)
                bearing_offsets = [base, wrap_angle(base + self.rng.uniform(-jitter, jitter))]
                release_steps[1] = int(
                    self.rng.integers(
                        self.config.stage3_delay_min_steps,
                        self.config.stage3_delay_max_steps + 1,
                    )
                )
        for index in range(bug_count):
            bug = self._spawn_bug(
                scenario,
                index,
                goal_heading,
                bearing_offset=bearing_offsets[index],
                release_step=int(release_steps[index]),
            )
            if track_sources is not None:
                source = str(track_sources[index]).strip().lower()
            else:
                draw = self.rng.random()
                if draw < self.config.satellite_track_probability:
                    source = "satellite"
                elif draw < (
                    self.config.satellite_track_probability
                    + self.config.peer_track_probability
                ):
                    source = "peer"
                else:
                    source = "local"
            if source not in {"local", "peer", "satellite"}:
                raise ValueError("bug track source must be local, peer, or satellite")
            bug["track_source"] = source
            if self.config.dangerous_spawn_only and bug["active"]:
                values = self._threat_values(bug)
                if (
                    values["closing"] <= 0.0
                    or values["d_cpa"] > self.config.dangerous_max_cpa_m
                ):
                    raise RuntimeError("dangerous encounter generation produced a safe track")
            self.bugs.append(bug)
        observation = self.get_actor_obs()
        return observation, {"critic_state": self.get_critic_state()}

    def _intercept_heading(
        self,
        position: np.ndarray,
        own_position: np.ndarray,
        own_velocity: np.ndarray,
    ) -> float:
        relative = own_position - position
        a = float(np.dot(own_velocity, own_velocity)) - self.config.bug_speed_mps**2
        b = 2.0 * float(np.dot(relative, own_velocity))
        c = float(np.dot(relative, relative))
        discriminant = max(0.0, b * b - 4.0 * a * c)
        roots = []
        if abs(a) > 1e-9:
            roots = [
                value
                for value in (
                    (-b - math.sqrt(discriminant)) / (2.0 * a),
                    (-b + math.sqrt(discriminant)) / (2.0 * a),
                )
                if value > 0.0
            ]
        intercept_time = min(roots) if roots else math.sqrt(c) / max(
            self.config.bug_speed_mps + float(np.linalg.norm(own_velocity)), 1.0
        )
        aim = own_position + own_velocity * intercept_time
        return _heading(aim - position)

    def _spawn_bug(
        self,
        scenario: Mapping[str, Any],
        index: int,
        goal_heading: float,
        *,
        bearing_offset: float | None = None,
        release_step: int = 0,
    ) -> dict[str, Any]:
        positions = scenario.get("bug_spawn_positions")
        velocities = scenario.get("bug_velocities")
        if positions is not None:
            position = np.asarray(positions[index], dtype=np.float64)
        else:
            distance = self.rng.uniform(
                self.config.bug_spawn_min_m, self.config.bug_spawn_max_m
            )
            if bearing_offset is not None:
                bearing = goal_heading + bearing_offset
            elif self.stage == 1:
                bearing = goal_heading + self.rng.uniform(
                    math.radians(-10.0), math.radians(10.0)
                )
            else:
                limit = math.radians(self.config.stage2_bearing_limit_deg)
                minimum = math.radians(self.config.stage2_bearing_min_deg)
                if minimum < 0.0 or minimum > limit:
                    raise ValueError(
                        "stage2_bearing_min_deg must be between zero and the bearing limit"
                    )
                if minimum == 0.0:
                    offset = self.rng.uniform(-limit, limit)
                else:
                    magnitude = self.rng.uniform(minimum, limit)
                    offset = magnitude * self.rng.choice((-1.0, 1.0))
                bearing = goal_heading + offset
            position = self.own_position + _unit(bearing) * distance
        if velocities is not None:
            velocity = np.asarray(velocities[index], dtype=np.float64)
            heading = _heading(velocity)
        elif self.config.dangerous_spawn_only:
            own_velocity = _unit(goal_heading) * self.unit_speed
            heading = self._intercept_heading(position, self.own_position, own_velocity)
        else:
            distance = float(np.linalg.norm(position - self.own_position))
            intercept_time = distance / max(self.config.bug_speed_mps + self.unit_speed, 1.0)
            aim = self.own_position + _unit(goal_heading) * self.unit_speed * intercept_time
            heading = _heading(aim - position)
        return {
            "position": position,
            "heading": heading,
            "active": release_step <= 0,
            "ever_active": release_step <= 0,
            "release_step": release_step,
            "reacquire_on_release": velocities is None,
        }

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal_position - self.own_position))

    def _own_velocity(self) -> np.ndarray:
        return _unit(self.own_heading) * self.unit_speed

    def _threat_values(self, bug: Mapping[str, Any]) -> dict[str, float]:
        relative_position = np.asarray(bug["position"]) - self.own_position
        distance = float(np.linalg.norm(relative_position))
        bug_velocity = _unit(float(bug["heading"])) * self.config.bug_speed_mps
        relative_velocity = bug_velocity - self._own_velocity()
        relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))
        closing = -float(np.dot(relative_position, relative_velocity)) / max(distance, 1e-6)
        t_cpa = max(
            0.0,
            -float(np.dot(relative_position, relative_velocity))
            / max(relative_speed_sq, 1e-6),
        )
        cpa = relative_position + relative_velocity * t_cpa
        return {
            "distance": distance,
            "bearing": wrap_angle(_heading(relative_position) - self.own_heading),
            "relative_speed": float(np.linalg.norm(relative_velocity)),
            "closing": closing,
            "t_cpa": t_cpa,
            "d_cpa": float(np.linalg.norm(cpa)),
        }

    def _risk(self) -> bool:
        for bug in self.bugs:
            if not bug["active"]:
                continue
            values = self._threat_values(bug)
            if self._is_risky_threat(values):
                return True
        return False

    def _is_risky_threat(self, values: Mapping[str, float]) -> bool:
        if values["distance"] <= self.config.risk_distance_m:
            return True
        return bool(
            values["closing"] > 0.0
            and values["t_cpa"] <= self.config.risk_t_cpa_s
            and values["d_cpa"] <= self.config.risk_d_cpa_m
        )

    def _risk_score(self) -> float:
        """Return a smooth collision-risk potential in the range [0, 1]."""

        survival = 1.0
        for bug in self.bugs:
            if not bug["active"]:
                continue
            values = self._threat_values(bug)
            if values["closing"] <= 0.0:
                continue
            miss_factor = math.exp(
                -values["d_cpa"] / max(self.config.risk_d_cpa_m, 1.0)
            )
            time_factor = math.exp(
                -values["t_cpa"] / max(self.config.risk_t_cpa_s, 1.0)
            )
            survival *= 1.0 - miss_factor * time_factor
        return 1.0 - survival

    def get_actor_obs(self) -> np.ndarray:
        goal_vector = self.goal_position - self.own_position
        goal_distance = float(np.linalg.norm(goal_vector))
        goal_angle = wrap_angle(_heading(goal_vector) - self.own_heading)
        speed_scale = max(self.config.unit_speeds_mps)
        features = [
            self.unit_speed / speed_scale,
            math.sin(self.own_heading),
            math.cos(self.own_heading),
            np.clip(goal_distance / self.config.goal_distance_scale_m, 0.0, 2.0),
            math.sin(goal_angle),
            math.cos(goal_angle),
        ]
        threats = []
        for bug in self.bugs:
            if not bug["active"]:
                continue
            values = self._threat_values(bug)
            if self.config.actor_risk_gate and not self._is_risky_threat(values):
                continue
            values["track_source"] = str(bug.get("track_source", "local"))
            threats.append((values["distance"], values))
        threats.sort(key=lambda item: item[1]["bearing"])
        sources = [0.0] * self.MAX_THREATS
        for slot in range(self.MAX_THREATS):
            if slot >= len(threats):
                features.extend([0.0] * self.THREAT_FEATURES)
                continue
            values = threats[slot][1]
            source = {
                "local": 1.0 / 3.0,
                "peer": 2.0 / 3.0,
                "satellite": 1.0,
            }[str(threats[slot][1].get("track_source", "local"))]
            if not self.config.encode_track_source:
                source = 1.0 / 3.0
            sources[slot] = source
            features.extend(
                [
                    1.0,
                    np.clip(values["distance"] / self.config.threat_distance_scale_m, 0.0, 2.0),
                    math.sin(values["bearing"]),
                    math.cos(values["bearing"]),
                    np.clip(values["relative_speed"] / self.config.relative_speed_scale_mps, 0.0, 2.0),
                    np.clip(values["closing"] / self.config.relative_speed_scale_mps, -2.0, 2.0),
                    np.clip(values["t_cpa"] / self.config.t_cpa_scale_s, 0.0, 2.0),
                    np.clip(values["d_cpa"] / self.config.threat_distance_scale_m, 0.0, 2.0),
                    source,
                ]
            )
        action_context = [0.0, 0.0, 0.0]
        if self.config.actor_action_history:
            action_context[self.previous_action] = 1.0
        features.extend(
            [
                *action_context,
                sources[0],
                sources[1],
                np.clip(self.step_count / max(self.config.horizon_steps, 1), 0.0, 1.0),
            ]
        )
        features.extend(float(self.unit_type == index) for index in range(3))
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (self.ACTOR_OBS_DIM,):
            raise RuntimeError(f"unexpected actor observation shape {result.shape}")
        return result

    def get_critic_state(self) -> np.ndarray:
        state = self.get_actor_obs().astype(np.float64).tolist()
        own_velocity = self._own_velocity()
        for slot in range(self.MAX_THREATS):
            if slot >= len(self.bugs) or not self.bugs[slot]["active"]:
                state.extend([0.0] * self.CRITIC_EXTRA_PER_THREAT)
                continue
            bug = self.bugs[slot]
            relative = (np.asarray(bug["position"]) - self.own_position) / self.config.threat_distance_scale_m
            velocity = (
                _unit(float(bug["heading"])) * self.config.bug_speed_mps - own_velocity
            ) / self.config.relative_speed_scale_mps
            state.extend([1.0, float(relative[0]), float(relative[1]), float(velocity[0]), float(velocity[1])])
        result = np.asarray(state, dtype=np.float32)
        if result.shape != (self.CRITIC_STATE_DIM,):
            raise RuntimeError(f"unexpected critic state shape {result.shape}")
        return result

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if int(action) not in (LEFT, STRAIGHT, RIGHT):
            raise ValueError("action must be LEFT=0, STRAIGHT=1, or RIGHT=2")
        action = int(action)
        for bug in self.bugs:
            if (
                not bug["active"]
                and not bug["ever_active"]
                and self.step_count >= int(bug["release_step"])
            ):
                bug["active"] = True
                bug["ever_active"] = True
                if bug["reacquire_on_release"]:
                    bug["heading"] = self._intercept_heading(
                        np.asarray(bug["position"]),
                        self.own_position,
                        self._own_velocity(),
                    )
        previous_goal_distance = self._goal_distance()
        previous_risk_score = self._risk_score()
        previous_own = self.own_position.copy()
        previous_bug_positions = [np.asarray(item["position"]).copy() for item in self.bugs]
        goal_heading = _heading(self.goal_position - self.own_position)
        turn_rate = self.config.avoid_angle_deg
        if self.config.unit_turn_rates_deg_s is not None:
            turn_rate = self.config.unit_turn_rates_deg_s[self.unit_type]
        residual = (
            (action - STRAIGHT)
            * math.radians(turn_rate)
            * self.config.control_interval_sec
        )
        if self.config.integrated_turn_actions:
            if action == STRAIGHT:
                recovery = float(
                    np.clip(
                        wrap_angle(goal_heading - self.own_heading),
                        -math.radians(self.config.straight_recovery_deg),
                        math.radians(self.config.straight_recovery_deg),
                    )
                )
                self.own_heading = wrap_angle(self.own_heading + recovery)
            else:
                self.own_heading = wrap_angle(self.own_heading + residual)
        else:
            self.own_heading = wrap_angle(goal_heading + residual)
        dt = self.config.control_interval_sec
        self.own_position += self._own_velocity() * dt
        self.path_length += float(np.linalg.norm(self.own_position - previous_own))

        max_turn = math.radians(self.config.bug_turn_rate_deg_s) * dt
        for bug in self.bugs:
            if not bug["active"]:
                continue
            desired = _heading(self.own_position - np.asarray(bug["position"]))
            bug_max_turn = max_turn * float(bug.get("turn_rate_scale", 1.0))
            turn = float(
                np.clip(
                    wrap_angle(desired - float(bug["heading"])),
                    -bug_max_turn,
                    bug_max_turn,
                )
            )
            bug["heading"] = wrap_angle(float(bug["heading"]) + turn)
            bug["position"] = np.asarray(bug["position"]) + _unit(float(bug["heading"])) * self.config.bug_speed_mps * dt

        self.step_count += 1
        if action == STRAIGHT:
            self.maneuver_streak = 0
        else:
            self.maneuver_streak += 1
        hit = False
        for index, bug in enumerate(self.bugs):
            if not bug["active"]:
                continue
            start_relative = previous_bug_positions[index] - previous_own
            end_relative = np.asarray(bug["position"]) - self.own_position
            closest = closest_segment_distance(start_relative, end_relative)
            self.closest_bug_distance = min(self.closest_bug_distance, closest)
            if closest <= self.config.hit_radius_m:
                hit = True
                bug["active"] = False

        current_goal_distance = self._goal_distance()
        arrived = current_goal_distance <= self.config.arrival_radius_m
        if self.config.arrival_segment_check and not arrived:
            arrived = (
                closest_segment_distance(
                    previous_own - self.goal_position,
                    self.own_position - self.goal_position,
                )
                <= self.config.arrival_radius_m
            )
        risk = self._risk()
        self.had_threat = self.had_threat or risk
        if self.had_threat and not risk and not hit:
            self.clear_steps += 1
        else:
            self.clear_steps = 0
        newly_evaded = bool(
            self.bugs
            and not hit
            and not self.evade_achieved
            and all(bool(bug["ever_active"]) for bug in self.bugs)
            and self.clear_steps >= self.config.evade_clear_steps
            and min(
                (self._threat_values(bug)["distance"] for bug in self.bugs if bug["active"]),
                default=math.inf,
            ) >= self.config.evade_clear_distance_m
        )
        if newly_evaded:
            self.evade_achieved = True
            if self.config.retire_threats_on_evade:
                for bug in self.bugs:
                    if bug["active"]:
                        bug["active"] = False

        heading_error = abs(wrap_angle(goal_heading - self.own_heading))
        newly_recovered = bool(
            self.evade_achieved
            and not self.recovery_achieved
            and not risk
            and heading_error
            <= math.radians(self.config.recovery_heading_tolerance_deg)
        )
        if newly_recovered:
            self.recovery_achieved = True

        evaded = bool(self.evade_achieved and not hit)
        terminated = bool(
            hit
            or arrived
            or (self.config.terminate_on_evade and newly_evaded)
        )
        truncated = bool(self.step_count >= self.config.horizon_steps and not terminated)

        reward_goal_distance = 0.0 if arrived else current_goal_distance
        progress_m = previous_goal_distance - reward_goal_distance
        progress_km = progress_m / 1000.0
        progress_fraction = progress_m / max(self.initial_goal_distance, 1.0)
        reward = (
            self.config.progress_scale * progress_km
            + self.config.progress_fraction_scale * progress_fraction
        )
        if hit:
            reward -= self.config.hit_penalty
        if newly_evaded:
            reward += self.config.evade_bonus
        if arrived:
            reward += self.config.arrival_bonus
        if truncated:
            reward -= self.config.timeout_penalty
        current_risk_score = self._risk_score()
        if not hit:
            reward += self.config.safety_improvement_scale * (
                previous_risk_score - current_risk_score
            )
        if action == STRAIGHT:
            reward -= self.config.unsafe_straight_penalty * previous_risk_score
        deviation = abs(action - STRAIGHT)
        reward -= self.config.deviation_penalty * deviation
        switched = action != self.previous_action
        reversed_turn = (
            self.previous_action in (LEFT, RIGHT)
            and action in (LEFT, RIGHT)
            and action != self.previous_action
        )
        reward -= self.config.oscillation_penalty * abs(action - self.previous_action)
        reward -= self.config.action_switch_penalty * float(switched)
        reward -= self.config.turn_reversal_penalty * float(reversed_turn)
        reward -= self.config.time_penalty
        if not risk:
            reward -= self.config.recovery_penalty * deviation
        reward -= self.config.heading_error_penalty * (heading_error / math.pi)
        if newly_recovered:
            reward += self.config.recovery_bonus
        if self.config.maneuver_burst_limit_steps > 0:
            excess = max(
                0,
                self.maneuver_streak - self.config.maneuver_burst_limit_steps,
            )
            reward -= self.config.excess_maneuver_penalty * excess
        reward = float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))
        self.episode_return += reward
        oscillated = reversed_turn
        self.oscillation_count += int(oscillated)
        self.heading_deviation_sum += abs(math.degrees(residual))
        self.previous_action = action
        self._last_risk = risk
        observation = self.get_actor_obs()
        extra_path = max(
            0.0,
            self.path_length - max(0.0, self.initial_goal_distance - current_goal_distance),
        )
        info = {
            "critic_state": self.get_critic_state(),
            "hit": hit,
            "evaded": evaded,
            "newly_evaded": newly_evaded,
            "arrived": arrived,
            "recovered": self.recovery_achieved,
            "risk": risk,
            "risk_score": current_risk_score,
            "risk_improvement": previous_risk_score - current_risk_score,
            "progress_km": progress_km,
            "progress_fraction": progress_fraction,
            "extra_path_m": extra_path,
            "extra_time_s": extra_path / max(self.unit_speed, 1.0),
            "recovery_time_s": self.clear_steps * dt,
            "oscillated": oscillated,
            "oscillation_rate": self.oscillation_count / max(self.step_count, 1),
            "mean_heading_deviation_deg": self.heading_deviation_sum / max(self.step_count, 1),
            "closest_bug_distance_m": self.closest_bug_distance,
            "episode_steps": self.step_count,
            "episode_return": self.episode_return,
            "maneuver_streak": self.maneuver_streak,
            "encounter_pattern": self.encounter_pattern,
            "active_threats": sum(bool(bug["active"]) for bug in self.bugs),
            "released_threats": sum(bool(bug["ever_active"]) for bug in self.bugs),
        }
        return observation, reward, terminated, truncated, info


class VectorLocalAvoidEnv:
    """Synchronous vector wrapper with automatic reset after terminal steps."""

    def __init__(
        self,
        num_envs: int,
        config: LocalAvoidConfig | tuple[LocalAvoidConfig, ...] | list[LocalAvoidConfig],
        *,
        stage: int,
        seed: int,
        stage_distribution: tuple[float, ...] | None = None,
    ) -> None:
        num_envs = int(num_envs)
        if isinstance(config, LocalAvoidConfig):
            configs = [config] * num_envs
        else:
            configs = list(config)
            if len(configs) != num_envs:
                raise ValueError("one environment config is required per vector slot")
        self.rng = np.random.default_rng(seed + 9176)
        if stage_distribution is None:
            self.stage_probabilities = None
        else:
            weights = np.asarray(stage_distribution, dtype=np.float64)
            if (
                weights.ndim != 1
                or not 1 <= weights.size <= 4
                or np.any(weights < 0.0)
                or weights.sum() <= 0.0
            ):
                raise ValueError(
                    "stage_distribution must contain one to four non-negative weights"
                )
            self.stage_probabilities = weights / weights.sum()
        self.envs = [
            LocalAvoidEnv(
                configs[index],
                stage=self._sample_stage(stage),
                seed=seed + index * 9973,
            )
            for index in range(num_envs)
        ]
        self.stage = int(stage)

    def _sample_stage(self, fallback: int) -> int:
        if self.stage_probabilities is None:
            return int(fallback)
        return int(self.rng.choice(len(self.stage_probabilities), p=self.stage_probabilities))

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        pairs = [env.reset() for env in self.envs]
        actor = np.stack([item[0] for item in pairs])
        critic = np.stack([item[1]["critic_state"] for item in pairs])
        return actor, critic

    def step(self, actions: np.ndarray):
        actor_rows = []
        critic_rows = []
        rewards = []
        dones = []
        infos = []
        for env, action in zip(self.envs, actions.tolist()):
            observation, reward, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            terminal_info = dict(info)
            terminal_info["terminated"] = terminated
            terminal_info["truncated"] = truncated
            terminal_info["stage"] = env.stage
            if done:
                env.stage = self._sample_stage(self.stage)
                observation, reset_info = env.reset()
                critic_state = reset_info["critic_state"]
            else:
                critic_state = info["critic_state"]
            actor_rows.append(observation)
            critic_rows.append(critic_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(terminal_info if done else info)
        return (
            np.stack(actor_rows),
            np.stack(critic_rows),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            infos,
        )
