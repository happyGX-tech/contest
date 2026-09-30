"""Inference adapter for standalone 33-feature local avoidance checkpoints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from policies.red.learning.red_policy import HighLevelAction, SharedPolicy

from .encoder import EvasionObservationEncoder
from .local_env import LocalAvoidConfig
from .local_ppo import LocalAvoidPPO


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


class LocalAvoidRuntimeEncoder:
    """Translate one missile's legal simulator observation to the local PPO ABI."""

    OBSERVATION_DIM = 33

    def __init__(
        self,
        *,
        entity_id: int,
        max_steps: int,
        config: LocalAvoidConfig,
    ) -> None:
        self.config = config
        self.max_steps = max(1, int(max_steps))
        self._base = EvasionObservationEncoder(
            entity_id=entity_id,
            max_steps=max_steps,
            max_track_age_steps=3,
            threat_range_m=config.threat_distance_scale_m,
            # The platform defines satellite tracks as globally authoritative.
            # Keep their provenance, but never discard their kinematics.
            allow_fused_tracks=True,
        )

    @staticmethod
    def _target_geometry(
        own_position: Mapping[str, Any], goal_position: Any
    ) -> tuple[float, float]:
        lon = float(own_position.get("lon", 0.0))
        lat = float(own_position.get("lat", 0.0))
        target_lon = float(_field(goal_position, "lon", lon))
        target_lat = float(_field(goal_position, "lat", lat))
        mean_latitude = math.radians((lat + target_lat) / 2.0)
        east_m = (target_lon - lon) * 111_320.0 * math.cos(mean_latitude)
        north_m = (target_lat - lat) * 110_570.0
        return math.hypot(east_m, north_m), math.atan2(north_m, east_m)

    @staticmethod
    def _ecef_heading(direction: np.ndarray, lon_deg: float, lat_deg: float) -> float | None:
        if float(np.linalg.norm(direction)) <= 1e-6:
            return None
        lon = math.radians(lon_deg)
        lat = math.radians(lat_deg)
        dx, dy, dz = (float(value) for value in direction)
        east = -math.sin(lon) * dx + math.cos(lon) * dy
        north = (
            -math.sin(lat) * math.cos(lon) * dx
            - math.sin(lat) * math.sin(lon) * dy
            + math.cos(lat) * dz
        )
        if math.hypot(east, north) <= 1e-6:
            return None
        return math.atan2(north, east)

    def encode(
        self,
        observation: Mapping[str, Any],
        *,
        launch_step: int,
        maneuver_state: int,
        goal_position: Any,
    ) -> np.ndarray:
        base = self._base.encode(
            observation,
            launch_step=launch_step,
            maneuver_state=maneuver_state,
        )
        own = observation.get("self") or {}
        own_position = own.get("position") or {}
        goal_distance, goal_heading = self._target_geometry(own_position, goal_position)
        own_heading = self._ecef_heading(
            base[12:15],
            float(own_position.get("lon", 0.0)),
            float(own_position.get("lat", 0.0)),
        )
        if own_heading is None:
            own_heading = goal_heading
        goal_angle = _wrap_angle(goal_heading - own_heading)

        entity_type = int(own.get("type", 0))
        type_slot = {21000: 0, 21001: 1, 21002: 2}.get(entity_type, 0)
        speed_scale = max(self.config.unit_speeds_mps)
        features: list[float] = [
            float(self.config.unit_speeds_mps[type_slot] / speed_scale),
            math.sin(own_heading),
            math.cos(own_heading),
            float(np.clip(goal_distance / self.config.goal_distance_scale_m, 0.0, 2.0)),
            math.sin(goal_angle),
            math.cos(goal_angle),
        ]

        threats: list[tuple[float, float, list[float]]] = []
        trusted_sources = self._base.threat_sources
        for slot in range(EvasionObservationEncoder.MAX_THREATS):
            offset = (
                EvasionObservationEncoder.THREAT_START
                + slot * EvasionObservationEncoder.THREAT_FEATURES
            )
            if base[offset] <= 0.5:
                continue
            distance = float(base[offset + 4] * self.config.threat_distance_scale_m)
            bearing_sin = float(-base[offset + 2])
            bearing_cos = float(base[offset + 1])
            bearing = math.atan2(bearing_sin, bearing_cos)
            relative_speed = float(np.linalg.norm(base[offset + 5:offset + 8]) * 2_000.0)
            closing = float(base[offset + 8] * 2_000.0)
            time_to_cpa = float(base[offset + 9] * 120.0)
            cpa_distance = float(base[offset + 10] * self.config.threat_distance_scale_m)
            risky = bool(
                distance <= self.config.risk_distance_m
                or (
                    closing > 0.0
                    and time_to_cpa <= self.config.risk_t_cpa_s
                    and cpa_distance <= self.config.risk_d_cpa_m
                )
            )
            if self.config.actor_risk_gate and not risky:
                continue
            threats.append(
                (
                    bearing,
                    (
                        trusted_sources[slot]
                        if slot < len(trusted_sources)
                        else 1.0 / 3.0
                    ),
                    [
                        1.0,
                        float(np.clip(distance / self.config.threat_distance_scale_m, 0.0, 2.0)),
                        bearing_sin,
                        bearing_cos,
                        float(np.clip(relative_speed / self.config.relative_speed_scale_mps, 0.0, 2.0)),
                        float(np.clip(closing / self.config.relative_speed_scale_mps, -2.0, 2.0)),
                        float(np.clip(time_to_cpa / self.config.t_cpa_scale_s, 0.0, 2.0)),
                        float(np.clip(cpa_distance / self.config.threat_distance_scale_m, 0.0, 2.0)),
                        1.0 / 3.0,
                    ],
                )
            )
        threats.sort(key=lambda item: item[0])
        sources = [0.0, 0.0]
        for slot in range(2):
            if slot < len(threats):
                source = (
                    threats[slot][1]
                    if self.config.encode_track_source
                    else 1.0 / 3.0
                )
                threat_features = threats[slot][2]
                threat_features[-1] = source
                features.extend(threat_features)
                sources[slot] = source
            else:
                features.extend([0.0] * 9)

        previous_action = int(np.clip(maneuver_state + 1, 0, 2))
        action_context = [0.0, 0.0, 0.0]
        if self.config.actor_action_history:
            action_context[previous_action] = 1.0
        elapsed = max(0, int(observation.get("step", 0)) - int(launch_step))
        features.extend(
            [
                *action_context,
                sources[0],
                sources[1],
                float(np.clip(elapsed / max(self.config.horizon_steps, 1), 0.0, 1.0)),
            ]
        )
        features.extend(float(type_slot == index) for index in range(3))
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (self.OBSERVATION_DIM,):
            raise RuntimeError(f"Unexpected local avoidance observation shape: {result.shape}")
        return result

    def reset(self) -> None:
        self._base.reset()


class LocalAvoidRuntimePolicy(SharedPolicy):
    """Deterministic full-simulator inference for the standalone S3 skill."""

    algorithm_name = "local_avoid_asymmetric_ppo_v1"
    is_evasion_skill = True
    requires_target_context = True
    requires_environment_hooks = True
    records_evaluation = False
    motion_learning_enabled = False
    # Satellite use belongs to the unchanged upper-layer commander. The local
    # actor consumes the resulting authoritative tracks while retaining their
    # provenance separately from local and communication-cluster observations.
    allows_satellite = True

    def __init__(self, *, max_steps: int = 1000) -> None:
        self.max_steps = int(max_steps)
        self.training = False
        self.model: LocalAvoidPPO | None = None
        self.environment_config: LocalAvoidConfig | None = None
        self._encoders: dict[int, LocalAvoidRuntimeEncoder] = {}
        self._entity_ids: dict[int, int] = {}

    def load(self, path: str) -> None:
        self.model, self.environment_config, _payload = LocalAvoidPPO.load(path, device="cpu")
        self.model.network.eval()
        self._encoders.clear()

    def register_agent(self, agent_id: int, entity_id: int, max_steps: int) -> None:
        if self.environment_config is None:
            self._entity_ids[int(agent_id)] = int(entity_id)
            return
        self._entity_ids[int(agent_id)] = int(entity_id)
        self._encoders[int(agent_id)] = LocalAvoidRuntimeEncoder(
            entity_id=entity_id,
            max_steps=max_steps,
            config=self.environment_config,
        )

    def encode_motion_observation(
        self,
        observation: Mapping[str, Any],
        *,
        agent_id: int,
        entity_id: int,
        launch_step: int,
        maneuver_state: int,
        goal_position: Any,
    ) -> np.ndarray:
        if self.environment_config is None:
            raise RuntimeError("Local avoidance model has not been loaded")
        encoder = self._encoders.get(int(agent_id))
        if encoder is None:
            self.register_agent(agent_id, entity_id, self.max_steps)
            encoder = self._encoders[int(agent_id)]
        return encoder.encode(
            observation,
            launch_step=launch_step,
            maneuver_state=maneuver_state,
            goal_position=goal_position,
        )

    @staticmethod
    def prepare_action_mask(observation: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        effective = np.asarray(action_mask, dtype=np.bool_).copy()
        if observation[6] <= 0.5 and observation[15] <= 0.5:
            effective[:] = False
            effective[int(HighLevelAction.STOP_MANEUVER)] = True
        return effective

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        if self.model is None:
            raise RuntimeError("Local avoidance model has not been loaded")
        actions, _log_probs, _values, _actor, _critic = self.model.act(
            np.asarray(observation, dtype=np.float32),
            np.zeros(43, dtype=np.float32),
            deterministic=True,
            update_normalizer=False,
        )
        selected = int(np.asarray(actions).item())
        if not bool(action_mask[selected]):
            valid = np.flatnonzero(action_mask)
            return int(valid[0]) if valid.size else int(HighLevelAction.STOP_MANEUVER)
        return selected

    def set_training(self, training: bool) -> None:
        if training:
            raise ValueError("local_avoid_s3 is an inference-only motion skill")
        self.training = False

    def reset_episode(self) -> None:
        for encoder in self._encoders.values():
            encoder.reset()

    def save(self, path: str) -> None:
        raise NotImplementedError("local_avoid_s3 checkpoints are produced by standalone training")
