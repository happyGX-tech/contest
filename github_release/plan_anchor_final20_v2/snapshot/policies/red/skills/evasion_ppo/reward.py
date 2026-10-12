"""Reward and event accounting for the PPO evasion skill."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from policies.red.learning.red_policy import HighLevelAction, PolicyTransition

from .encoder import EvasionObservationEncoder


@dataclass(frozen=True)
class ShapedEvasionReward:
    value: float
    trainable: bool
    skill_terminal: bool


@dataclass
class _Encounter:
    active: bool = False
    clear_steps: int = 0
    closest_distance_km: float = math.inf
    initial_context: dict[str, float] | None = None


class EvasionRewardShaper:
    """Reward survival of an observable incoming interceptor encounter."""

    def __init__(self, config) -> None:
        self.config = config
        self._encounters: dict[int, _Encounter] = {}
        self._previous_actions: dict[int, int] = {}
        self._episode_successes = 0
        self._episode_failures = 0
        self._episode_censored = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_censored = 0
        self._rolling = deque(maxlen=max(1, int(config.metric_window_events)))
        self._episode_actions = [0, 0, 0]
        self._total_actions = [0, 0, 0]
        self._episode_failure_distances: list[float] = []
        self._total_failure_distances: list[float] = []
        self._episode_failure_contexts: list[dict[str, float]] = []
        self._total_failure_contexts: list[dict[str, float]] = []

    @staticmethod
    def _distance_km(observation: np.ndarray) -> float:
        return EvasionObservationEncoder.nearest_distance(observation) * 100.0

    def begin_environment_step(self, _observation) -> None:
        return None

    def end_environment_step(self, _observation) -> None:
        return None

    def _resolve(self, agent_id: int, success: bool) -> None:
        encounter = self._encounters.setdefault(agent_id, _Encounter())
        encounter.active = False
        encounter.clear_steps = 0
        encounter.closest_distance_km = math.inf
        encounter.initial_context = None
        self._rolling.append(bool(success))
        if success:
            self._episode_successes += 1
            self._total_successes += 1
        else:
            self._episode_failures += 1
            self._total_failures += 1

    def shape(self, transition: PolicyTransition) -> ShapedEvasionReward:
        agent_id = int(transition.agent_id)
        encounter = self._encounters.setdefault(agent_id, _Encounter())
        current_threat = EvasionObservationEncoder.has_threat(transition.observation)
        next_threat = EvasionObservationEncoder.has_threat(transition.next_observation)
        current_distance_km = self._distance_km(transition.observation)
        next_distance_km = self._distance_km(transition.next_observation)
        current_closing = EvasionObservationEncoder.maximum_closing_speed(
            transition.observation
        )
        next_closing = EvasionObservationEncoder.maximum_closing_speed(
            transition.next_observation
        )
        current_cpa_km = (
            EvasionObservationEncoder.minimum_cpa_distance(transition.observation)
            * 100.0
        )
        next_cpa_km = (
            EvasionObservationEncoder.minimum_cpa_distance(transition.next_observation)
            * 100.0
        )

        if current_threat and not encounter.active:
            encounter.active = True
            encounter.clear_steps = 0
            encounter.closest_distance_km = current_distance_km
            encounter.initial_context = EvasionObservationEncoder.nearest_threat_context(
                transition.observation
            )
        if encounter.active and current_threat:
            encounter.closest_distance_km = min(
                encounter.closest_distance_km, current_distance_km
            )

        trainable = bool(encounter.active or current_threat)
        reward = 0.0
        terminal = False
        if trainable:
            action = int(transition.action)
            if 0 <= action < len(self._episode_actions):
                self._episode_actions[action] += 1
                self._total_actions[action] += 1
            reward += self.config.threat_survival_reward
            if current_threat and next_threat:
                distance_gain = np.clip(
                    (next_distance_km - current_distance_km)
                    / max(self.config.distance_reward_scale_km, 1e-6),
                    -1.0,
                    1.0,
                )
                closing_reduction = np.clip(current_closing - next_closing, -1.0, 1.0)
                cpa_gain = np.clip(
                    (next_cpa_km - current_cpa_km)
                    / max(self.config.cpa_reward_scale_km, 1e-6),
                    -1.0,
                    1.0,
                )
                reward += self.config.distance_gain_scale * float(distance_gain)
                reward += self.config.closing_reduction_scale * float(closing_reduction)
                reward += self.config.cpa_gain_scale * float(cpa_gain)

            if transition.action != int(HighLevelAction.STOP_MANEUVER):
                reward -= self.config.maneuver_penalty
            previous = self._previous_actions.get(agent_id)
            if (
                previous in {int(HighLevelAction.MANEUVER_LEFT), int(HighLevelAction.MANEUVER_RIGHT)}
                and transition.action
                in {int(HighLevelAction.MANEUVER_LEFT), int(HighLevelAction.MANEUVER_RIGHT)}
                and previous != transition.action
            ):
                reward -= self.config.direction_change_penalty

        self._previous_actions[agent_id] = int(transition.action)
        if encounter.active:
            intercepted = bool(
                transition.done
                and encounter.closest_distance_km
                <= self.config.intercept_failure_radius_km
            )
            if intercepted:
                self._episode_failure_distances.append(encounter.closest_distance_km)
                self._total_failure_distances.append(encounter.closest_distance_km)
                failure_context = EvasionObservationEncoder.nearest_threat_context(
                    transition.observation
                )
                failure_context.update({
                    f"initial_{key}": value
                    for key, value in (encounter.initial_context or {}).items()
                })
                failure_context["action"] = float(transition.action)
                self._episode_failure_contexts.append(failure_context)
                self._total_failure_contexts.append(failure_context)
                reward -= self.config.intercepted_penalty
                self._resolve(agent_id, False)
                terminal = True
            elif transition.done:
                reward += self.config.evaded_bonus
                self._resolve(agent_id, True)
                terminal = True
            elif not next_threat:
                encounter.clear_steps += 1
                if encounter.clear_steps >= self.config.clear_grace_steps:
                    reward += self.config.evaded_bonus
                    self._resolve(agent_id, True)
                    terminal = True
            else:
                encounter.clear_steps = 0

        return ShapedEvasionReward(
            value=float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip)),
            trainable=trainable,
            skill_terminal=terminal,
        )

    @staticmethod
    def wilson_lower(successes: int, attempts: int, z: float = 1.96) -> float:
        if attempts <= 0:
            return 0.0
        p = successes / attempts
        denominator = 1.0 + z * z / attempts
        centre = p + z * z / (2.0 * attempts)
        margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * attempts)) / attempts)
        return max(0.0, (centre - margin) / denominator)

    @classmethod
    def _summary(
        cls, successes: int, failures: int, censored: int = 0
    ) -> dict[str, float | int]:
        attempts = successes + failures
        return {
            "attempts": attempts,
            "successes": successes,
            "failures": failures,
            "censored": censored,
            "evasion_rate": successes / attempts if attempts else 0.0,
            "wilson_lower_95": cls.wilson_lower(successes, attempts),
        }

    def finish_episode(self) -> dict[str, dict[str, float | int]]:
        for _agent_id, encounter in tuple(self._encounters.items()):
            if encounter.active:
                self._episode_censored += 1
                self._total_censored += 1
        episode = self._summary(
            self._episode_successes, self._episode_failures, self._episode_censored
        )
        total = self._summary(
            self._total_successes, self._total_failures, self._total_censored
        )
        rolling_successes = sum(self._rolling)
        rolling = self._summary(rolling_successes, len(self._rolling) - rolling_successes)
        episode["action_counts"] = list(self._episode_actions)
        total["action_counts"] = list(self._total_actions)
        episode["failure_closest_km_mean"] = (
            float(np.mean(self._episode_failure_distances))
            if self._episode_failure_distances else None
        )
        total["failure_closest_km_mean"] = (
            float(np.mean(self._total_failure_distances))
            if self._total_failure_distances else None
        )
        for summary, contexts in (
            (episode, self._episode_failure_contexts),
            (total, self._total_failure_contexts),
        ):
            keys = (
                "distance_km", "los_right", "closing_mps", "tgo_s", "cpa_km",
                "initial_distance_km", "initial_los_right", "initial_closing_mps",
                "initial_tgo_s", "initial_cpa_km", "action",
            )
            summary["failure_context_mean"] = {
                key: float(np.mean([item[key] for item in contexts if key in item]))
                for key in keys
                if any(key in item for item in contexts)
            }
        self._episode_successes = 0
        self._episode_failures = 0
        self._episode_censored = 0
        self._episode_actions = [0, 0, 0]
        self._episode_failure_distances.clear()
        self._episode_failure_contexts.clear()
        return {"episode": episode, "total": total, "rolling": rolling}

    def reset_episode(self) -> None:
        self._encounters.clear()
        self._previous_actions.clear()
        self._episode_successes = 0
        self._episode_failures = 0
        self._episode_censored = 0
        self._episode_actions = [0, 0, 0]
        self._episode_failure_distances.clear()
        self._episode_failure_contexts.clear()

    def reset_statistics(self) -> None:
        self.reset_episode()
        self._total_successes = 0
        self._total_failures = 0
        self._total_censored = 0
        self._total_actions = [0, 0, 0]
        self._total_failure_distances.clear()
        self._total_failure_contexts.clear()
        self._rolling.clear()

    def statistics_state(self) -> dict:
        return {
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "total_censored": self._total_censored,
            "rolling": list(self._rolling),
            "total_actions": list(self._total_actions),
            "total_failure_distances": list(self._total_failure_distances),
            "total_failure_contexts": list(self._total_failure_contexts),
        }

    def load_statistics_state(self, state: dict) -> None:
        self._total_successes = int(state.get("total_successes", 0))
        self._total_failures = int(state.get("total_failures", 0))
        self._total_censored = int(state.get("total_censored", 0))
        self._rolling.clear()
        self._rolling.extend(bool(value) for value in state.get("rolling", ()))
        self._total_actions = [int(value) for value in state.get("total_actions", (0, 0, 0))]
        self._total_failure_distances = [
            float(value) for value in state.get("total_failure_distances", ())
        ]
        self._total_failure_contexts = [
            {str(key): float(value) for key, value in item.items()}
            for item in state.get("total_failure_contexts", ())
        ]
