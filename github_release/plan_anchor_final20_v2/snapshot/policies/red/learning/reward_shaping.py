"""Shared R9 lower-layer reward shaping for PPO and MAPPO."""

from __future__ import annotations

import numpy as np

from .red_policy import (
    DEFAULT_TARGET_SLOTS,
    HighLevelAction,
    ObservationEncoder,
    PolicyTransition,
)


class R9RewardShaper:
    """Combine team objective damage with local motion feedback."""

    OBJECTIVE_WEIGHTS = {9400: 5.0, 9500: 1.0, 9600: 2.0}
    TARGET_FEATURE_OFFSET = ObservationEncoder.SELF_FEATURES
    TARGET_DISTANCE_OFFSET = 2
    TARGET_SELECTED_OFFSET = 9
    HEALTH_OFFSET = 4

    def __init__(self, config):
        self.config = config
        self._initial_health: dict[int, float] = {}
        self._objective_types: dict[int, int] = {}
        self._score_before = 0.0
        self._team_reward = 0.0
        self._previous_actions: dict[int, int] = {}

    @staticmethod
    def _entities(observation) -> dict:
        return observation.get("entities", {}) if observation else {}

    def reset_episode(self) -> None:
        self._initial_health.clear()
        self._objective_types.clear()
        self._score_before = 0.0
        self._team_reward = 0.0
        self._previous_actions.clear()

    def _capture_objectives(self, observation) -> None:
        for raw_id, entity in self._entities(observation).items():
            entity_type = int(entity.get("type", 0))
            health = float(entity.get("health", 0.0))
            if entity_type not in self.OBJECTIVE_WEIGHTS or health <= 0.0:
                continue
            entity_id = int(raw_id)
            self._objective_types.setdefault(entity_id, entity_type)
            self._initial_health.setdefault(entity_id, health)

    def _competition_score(self, observation) -> float:
        self._capture_objectives(observation)
        if not self._initial_health:
            return 0.0
        entities = self._entities(observation)
        weighted_damage = 0.0
        total_weight = 0.0
        for entity_id, initial_health in self._initial_health.items():
            entity = entities.get(entity_id, entities.get(str(entity_id), {}))
            health = float(entity.get("health", initial_health))
            damage_fraction = np.clip(
                (initial_health - health) / max(initial_health, 1e-6), 0.0, 1.0
            )
            weight = self.OBJECTIVE_WEIGHTS[self._objective_types[entity_id]]
            weighted_damage += weight * float(damage_fraction)
            total_weight += weight
        return 100.0 * weighted_damage / max(total_weight, 1e-6)

    def begin_environment_step(self, observation) -> None:
        self._score_before = self._competition_score(observation)
        self._team_reward = 0.0

    def end_environment_step(self, observation) -> None:
        score_after = self._competition_score(observation)
        self._team_reward = max(0.0, score_after - self._score_before)

    @classmethod
    def _assigned_target_slot(cls, observation: np.ndarray) -> int | None:
        for slot in range(DEFAULT_TARGET_SLOTS):
            selected_index = (
                cls.TARGET_FEATURE_OFFSET
                + slot * ObservationEncoder.TARGET_FEATURES
                + cls.TARGET_SELECTED_OFFSET
            )
            if selected_index < observation.size and observation[selected_index] > 0.5:
                return slot
        return None

    def shape(self, transition: PolicyTransition) -> float:
        reward = self.config.objective_damage_scale * self._team_reward
        target_slot = self._assigned_target_slot(transition.observation)
        if target_slot is not None:
            distance_index = (
                self.TARGET_FEATURE_OFFSET
                + target_slot * ObservationEncoder.TARGET_FEATURES
                + self.TARGET_DISTANCE_OFFSET
            )
            if distance_index < transition.next_observation.size:
                progress = float(
                    transition.observation[distance_index]
                    - transition.next_observation[distance_index]
                )
                reward += self.config.target_progress_scale * float(
                    np.clip(progress, -0.02, 0.02)
                )

        disappeared = (
            transition.observation.size > self.HEALTH_OFFSET
            and transition.next_observation.size > self.HEALTH_OFFSET
            and transition.observation[self.HEALTH_OFFSET] > 0.0
            and transition.next_observation[self.HEALTH_OFFSET] <= 0.0
        )
        if disappeared and self._team_reward <= 0.0:
            reward -= self.config.intercepted_penalty

        if transition.action != int(HighLevelAction.STOP_MANEUVER):
            reward -= self.config.maneuver_penalty
        previous_action = self._previous_actions.get(transition.agent_id)
        if (
            previous_action in {
                int(HighLevelAction.MANEUVER_LEFT),
                int(HighLevelAction.MANEUVER_RIGHT),
            }
            and transition.action in {
                int(HighLevelAction.MANEUVER_LEFT),
                int(HighLevelAction.MANEUVER_RIGHT),
            }
            and previous_action != transition.action
        ):
            reward -= self.config.direction_change_penalty
        self._previous_actions[transition.agent_id] = transition.action
        return float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))
