"""Inference adapter for standalone 33-feature local avoidance checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
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


@dataclass(frozen=True, slots=True)
class MissionGateConfig:
    """Mission-aware guard around the frozen Stage-3 actor."""

    recovery_deadband_deg: float = 4.0
    terminal_goal_distance_m: tuple[float, float, float] = (12_000.0, 10_000.0, 8_000.0)
    late_flight_fraction: float = 0.85
    engage_distance_m: tuple[float, float, float] = (1_000.0, 850.0, 600.0)
    engage_t_cpa_s: tuple[float, float, float] = (45.0, 32.0, 20.0)
    engage_d_cpa_m: tuple[float, float, float] = (3_000.0, 2_000.0, 900.0)
    emergency_distance_m: tuple[float, float, float] = (750.0, 600.0, 450.0)
    terminal_risk_scale: float = 0.65


def _only_action(action_mask: np.ndarray, action: HighLevelAction) -> np.ndarray:
    effective = np.asarray(action_mask, dtype=np.bool_).copy()
    effective[:] = False
    effective[int(action)] = True
    return effective


def _missile_type_slot(observation: np.ndarray) -> int:
    encoded = np.asarray(observation[30:33], dtype=np.float32)
    if encoded.size == 3 and encoded.max(initial=0.0) > 0.5:
        return int(np.argmax(encoded))
    return 0


def _has_urgent_threat(
    observation: np.ndarray,
    environment: LocalAvoidConfig,
    gate: MissionGateConfig,
    *,
    type_slot: int,
    terminal_mode: bool,
) -> bool:
    scale = gate.terminal_risk_scale if terminal_mode else 1.0
    t_limit = gate.engage_t_cpa_s[type_slot] * scale
    d_limit = gate.engage_d_cpa_m[type_slot] * scale
    for offset in (6, 15):
        if float(observation[offset]) <= 0.5:
            continue
        distance = float(observation[offset + 1]) * environment.threat_distance_scale_m
        closing = float(observation[offset + 5]) * environment.relative_speed_scale_mps
        time_to_cpa = float(observation[offset + 6]) * environment.t_cpa_scale_s
        cpa_distance = float(observation[offset + 7]) * environment.threat_distance_scale_m
        if distance <= gate.emergency_distance_m[type_slot]:
            return True
        if distance <= gate.engage_distance_m[type_slot] * scale:
            return True
        if closing > 0.0 and 0.0 <= time_to_cpa <= t_limit and cpa_distance <= d_limit:
            return True
    return False


def _recovery_action_mask(
    observation: np.ndarray,
    action_mask: np.ndarray,
    *,
    deadband_deg: float,
) -> np.ndarray:
    goal_error = math.atan2(float(observation[4]), float(observation[5]))
    if abs(goal_error) <= math.radians(deadband_deg):
        action = HighLevelAction.STOP_MANEUVER
    elif goal_error > 0.0:
        action = HighLevelAction.MANEUVER_LEFT
    else:
        action = HighLevelAction.MANEUVER_RIGHT
    return _only_action(action_mask, action)


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
        self._recovery_pending: dict[int, bool] = {}
        self._current_agent_id: int | None = None
        self.gate_mode = os.getenv("LOCAL_AVOID_GATE_MODE", "mission_v12").strip().lower()
        if self.gate_mode not in {"legacy", "mission_v12"}:
            raise ValueError(
                "LOCAL_AVOID_GATE_MODE must be either 'legacy' or 'mission_v12'"
            )
        self.mission_gate = MissionGateConfig()

    def load(self, path: str) -> None:
        self.model, self.environment_config, _payload = LocalAvoidPPO.load(path, device="cpu")
        self.model.network.eval()
        self._encoders.clear()
        self._recovery_pending.clear()
        self._current_agent_id = None

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
        agent_id = int(agent_id)
        encoder = self._encoders.get(agent_id)
        if encoder is None:
            self.register_agent(agent_id, entity_id, self.max_steps)
            encoder = self._encoders[agent_id]
        self._current_agent_id = agent_id
        if int(maneuver_state) != 0:
            self._recovery_pending[agent_id] = True
        return encoder.encode(
            observation,
            launch_step=launch_step,
            maneuver_state=maneuver_state,
            goal_position=goal_position,
        )

    def prepare_action_mask(
        self, observation: np.ndarray, action_mask: np.ndarray
    ) -> np.ndarray:
        if self.gate_mode == "legacy":
            effective = np.asarray(action_mask, dtype=np.bool_).copy()
            if observation[6] <= 0.5 and observation[15] <= 0.5:
                return _only_action(effective, HighLevelAction.STOP_MANEUVER)
            return effective

        if self.environment_config is None:
            raise RuntimeError("Local avoidance model has not been loaded")
        type_slot = _missile_type_slot(observation)
        goal_distance = (
            float(observation[3]) * self.environment_config.goal_distance_scale_m
        )
        terminal_mode = bool(
            goal_distance <= self.mission_gate.terminal_goal_distance_m[type_slot]
            or float(observation[29]) >= self.mission_gate.late_flight_fraction
        )
        if _has_urgent_threat(
            observation,
            self.environment_config,
            self.mission_gate,
            type_slot=type_slot,
            terminal_mode=terminal_mode,
        ):
            return np.asarray(action_mask, dtype=np.bool_).copy()
        agent_id = self._current_agent_id
        needs_recovery = bool(
            agent_id is not None and self._recovery_pending.get(agent_id, False)
        )
        if not needs_recovery:
            return _only_action(action_mask, HighLevelAction.STOP_MANEUVER)
        goal_error = math.atan2(float(observation[4]), float(observation[5]))
        if abs(goal_error) <= math.radians(self.mission_gate.recovery_deadband_deg):
            self._recovery_pending[agent_id] = False
        return _recovery_action_mask(
            observation,
            action_mask,
            deadband_deg=self.mission_gate.recovery_deadband_deg,
        )

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
        self._recovery_pending.clear()
        self._current_agent_id = None

    def save(self, path: str) -> None:
        raise NotImplementedError("local_avoid_s3 checkpoints are produced by standalone training")
