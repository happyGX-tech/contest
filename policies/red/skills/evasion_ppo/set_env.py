"""Variable-cardinality encounter environment for set-attention avoidance."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .local_env import (
    LocalAvoidConfig,
    LocalAvoidEnv,
    STRAIGHT,
    _heading,
    _unit,
    wrap_angle,
)


class SetLocalAvoidEnv(LocalAvoidEnv):
    """Local encounter with zero to eight visible interceptor tracks.

    Arrays are padded only at the batching boundary. The valid bit in each
    threat token is the attention mask, so padding is never interpreted as a
    physical interceptor.
    """

    MAX_THREATS = 8
    SELF_FEATURES = 3
    GOAL_FEATURES = 3
    THREAT_FEATURES = 9
    CONTEXT_FEATURES = 4
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
    THREAT_START = SELF_FEATURES + GOAL_FEATURES
    CONTEXT_START = THREAT_START + MAX_THREATS * THREAT_FEATURES
    TYPE_START = CONTEXT_START + CONTEXT_FEATURES

    def __init__(
        self,
        config: LocalAvoidConfig | None = None,
        *,
        stage: int = 3,
        seed: int = 0,
        threat_count_range: tuple[int, int] = (0, 2),
        decoy_probability: float = 0.0,
        delayed_wave_probability: float = 0.0,
    ) -> None:
        super().__init__(config, stage=stage, seed=seed)
        low, high = (int(value) for value in threat_count_range)
        if not 0 <= low <= high <= self.MAX_THREATS:
            raise ValueError("threat_count_range must satisfy 0 <= low <= high <= 8")
        if not 0.0 <= decoy_probability <= 1.0:
            raise ValueError("decoy_probability must be in [0, 1]")
        if not 0.0 <= delayed_wave_probability <= 1.0:
            raise ValueError("delayed_wave_probability must be in [0, 1]")
        self.threat_count_range = (low, high)
        self.decoy_probability = float(decoy_probability)
        self.delayed_wave_probability = float(delayed_wave_probability)

    def reset(
        self,
        scenario: Mapping[str, Any] | None = None,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        values = dict(scenario or {})
        generated = "bug_num" not in values
        if generated:
            low, high = self.threat_count_range
            values["bug_num"] = int(self.rng.integers(low, high + 1))
        bug_count = int(values.get("bug_num", 0))
        if "bug_release_steps" not in values:
            releases = [0] * bug_count
            if bug_count >= 2 and self.rng.random() < self.delayed_wave_probability:
                first_wave = max(1, bug_count // 2)
                low = max(1, int(self.config.stage3_delay_min_steps))
                high = max(low, int(self.config.stage3_delay_max_steps))
                for index in range(first_wave, bug_count):
                    releases[index] = int(self.rng.integers(low, high + 1))
                values["encounter_pattern"] = "multi_wave"
            else:
                values["encounter_pattern"] = "variable_simultaneous"
            values["bug_release_steps"] = releases

        observation, info = super().reset(values, seed=None)
        requested_decoys = values.get("decoy_indices")
        if requested_decoys is None and generated:
            decoys = {
                index
                for index in range(len(self.bugs))
                if self.rng.random() < self.decoy_probability
            }
            if len(decoys) == len(self.bugs) and self.bugs:
                decoys.remove(int(self.rng.integers(0, len(self.bugs))))
        else:
            decoys = {int(index) for index in (requested_decoys or [])}

        for index, bug in enumerate(self.bugs):
            bug["is_decoy"] = index in decoys
            if bug["is_decoy"]:
                outward = np.asarray(bug["position"]) - self.own_position
                bug["heading"] = _heading(outward)
                bug["turn_rate_scale"] = 0.0
            else:
                bug["turn_rate_scale"] = 1.0
                bug["heading"] = self._intercept_heading(
                    np.asarray(bug["position"]),
                    self.own_position,
                    self._own_velocity(),
                )
        observation = self.get_actor_obs()
        return observation, {"critic_state": self.get_critic_state()}

    @staticmethod
    def _source_value(bug: Mapping[str, Any], encode: bool) -> float:
        if not encode:
            return 1.0 / 3.0
        return {
            "local": 1.0 / 3.0,
            "peer": 2.0 / 3.0,
            "satellite": 1.0,
        }[str(bug.get("track_source", "local"))]

    def _ordered_threats(self) -> list[tuple[dict[str, Any], dict[str, float]]]:
        threats: list[tuple[tuple[float, ...], dict[str, Any], dict[str, float]]] = []
        for bug in self.bugs:
            if not bug["active"]:
                continue
            values = self._threat_values(bug)
            if self.config.actor_risk_gate and not self._is_risky_threat(values):
                continue
            risky = self._is_risky_threat(values)
            closing = values["closing"] > 0.0
            rank = (
                float(not risky),
                float(not closing),
                values["t_cpa"] if closing else math.inf,
                values["d_cpa"],
                values["distance"],
            )
            threats.append((rank, bug, values))
        threats.sort(key=lambda item: item[0])
        return [(bug, values) for _, bug, values in threats[: self.MAX_THREATS]]

    @staticmethod
    def _threat_features(
        values: Mapping[str, float],
        source: float,
        config: LocalAvoidConfig,
    ) -> list[float]:
        return [
            1.0,
            float(np.clip(values["distance"] / config.threat_distance_scale_m, 0.0, 2.0)),
            math.sin(values["bearing"]),
            math.cos(values["bearing"]),
            float(np.clip(values["relative_speed"] / config.relative_speed_scale_mps, 0.0, 2.0)),
            float(np.clip(values["closing"] / config.relative_speed_scale_mps, -2.0, 2.0)),
            float(np.clip(values["t_cpa"] / config.t_cpa_scale_s, 0.0, 2.0)),
            float(np.clip(values["d_cpa"] / config.threat_distance_scale_m, 0.0, 2.0)),
            source,
        ]

    def get_actor_obs(self) -> np.ndarray:
        goal_vector = self.goal_position - self.own_position
        goal_distance = float(np.linalg.norm(goal_vector))
        goal_angle = wrap_angle(_heading(goal_vector) - self.own_heading)
        speed_scale = max(self.config.unit_speeds_mps)
        features = [
            self.unit_speed / speed_scale,
            math.sin(self.own_heading),
            math.cos(self.own_heading),
            float(np.clip(goal_distance / self.config.goal_distance_scale_m, 0.0, 2.0)),
            math.sin(goal_angle),
            math.cos(goal_angle),
        ]
        threats = self._ordered_threats()
        for slot in range(self.MAX_THREATS):
            if slot >= len(threats):
                features.extend([0.0] * self.THREAT_FEATURES)
                continue
            bug, values = threats[slot]
            source = self._source_value(bug, self.config.encode_track_source)
            features.extend(self._threat_features(values, source, self.config))
        action_context = [0.0, 0.0, 0.0]
        if self.config.actor_action_history:
            action_context[self.previous_action] = 1.0
        features.extend(
            [
                *action_context,
                float(np.clip(self.step_count / max(self.config.horizon_steps, 1), 0.0, 1.0)),
            ]
        )
        features.extend(float(self.unit_type == index) for index in range(3))
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (self.ACTOR_OBS_DIM,):
            raise RuntimeError(f"unexpected set actor observation shape {result.shape}")
        return result

    def get_critic_state(self) -> np.ndarray:
        state = self.get_actor_obs().astype(np.float64).tolist()
        own_velocity = self._own_velocity()
        threats = self._ordered_threats()
        for slot in range(self.MAX_THREATS):
            if slot >= len(threats):
                state.extend([0.0] * self.CRITIC_EXTRA_PER_THREAT)
                continue
            bug, _ = threats[slot]
            relative = (
                np.asarray(bug["position"]) - self.own_position
            ) / self.config.threat_distance_scale_m
            velocity = (
                _unit(float(bug["heading"])) * self.config.bug_speed_mps
                - own_velocity
            ) / self.config.relative_speed_scale_mps
            state.extend(
                [1.0, float(relative[0]), float(relative[1]), float(velocity[0]), float(velocity[1])]
            )
        result = np.asarray(state, dtype=np.float32)
        if result.shape != (self.CRITIC_STATE_DIM,):
            raise RuntimeError(f"unexpected set critic state shape {result.shape}")
        return result

    def to_legacy_actor_observation(self, observation: np.ndarray | None = None) -> np.ndarray:
        """Project a zero-to-two-threat state onto the frozen V13 ABI."""

        source = self.get_actor_obs() if observation is None else np.asarray(observation)
        blocks = source[self.THREAT_START:self.CONTEXT_START].reshape(
            self.MAX_THREATS, self.THREAT_FEATURES
        )
        present = blocks[blocks[:, 0] > 0.5]
        chosen = present[:2]
        features = source[:6].astype(np.float32).tolist()
        sources = [0.0, 0.0]
        for slot in range(2):
            if slot < len(chosen):
                features.extend(chosen[slot].tolist())
                sources[slot] = float(chosen[slot, -1])
            else:
                features.extend([0.0] * self.THREAT_FEATURES)
        features.extend(source[self.CONTEXT_START:self.CONTEXT_START + 3].tolist())
        features.extend(sources)
        features.append(float(source[self.CONTEXT_START + 3]))
        features.extend(source[self.TYPE_START:self.TYPE_START + 3].tolist())
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (33,):
            raise RuntimeError(f"unexpected legacy projection shape {result.shape}")
        return result


class VectorSetLocalAvoidEnv:
    """Synchronous vector wrapper for variable-cardinality encounters."""

    def __init__(
        self,
        num_envs: int,
        config: LocalAvoidConfig | Sequence[LocalAvoidConfig],
        *,
        stage: int,
        seed: int,
        threat_count_range: tuple[int, int] | Sequence[tuple[int, int]],
        decoy_probability: float,
        delayed_wave_probability: float,
    ) -> None:
        if isinstance(config, LocalAvoidConfig):
            configs = [config] * int(num_envs)
        else:
            configs = list(config)
            if len(configs) != int(num_envs):
                raise ValueError("one environment config is required per vector slot")
        if (
            len(threat_count_range) == 2
            and isinstance(threat_count_range[0], (int, np.integer))
        ):
            ranges = [tuple(int(value) for value in threat_count_range)] * int(num_envs)
        else:
            ranges = [tuple(int(value) for value in item) for item in threat_count_range]
            if len(ranges) != int(num_envs):
                raise ValueError("one threat-count range is required per vector slot")
        self.envs = [
            SetLocalAvoidEnv(
                configs[index],
                stage=stage,
                seed=seed + index * 9973,
                threat_count_range=ranges[index],
                decoy_probability=decoy_probability,
                delayed_wave_probability=delayed_wave_probability,
            )
            for index in range(int(num_envs))
        ]

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        pairs = [environment.reset() for environment in self.envs]
        return (
            np.stack([item[0] for item in pairs]),
            np.stack([item[1]["critic_state"] for item in pairs]),
        )

    def step(self, actions: np.ndarray):
        actor_rows = []
        critic_rows = []
        rewards = []
        dones = []
        infos = []
        for environment, action in zip(self.envs, actions.tolist()):
            observation, reward, terminated, truncated, info = environment.step(int(action))
            done = bool(terminated or truncated)
            terminal_info = dict(info)
            terminal_info["terminated"] = terminated
            terminal_info["truncated"] = truncated
            terminal_info["stage"] = environment.stage
            if done:
                observation, reset_info = environment.reset()
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
