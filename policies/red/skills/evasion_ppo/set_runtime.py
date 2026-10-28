"""Full-simulator adapter for the variable-cardinality threat-set ABI."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .encoder import EvasionObservationEncoder
from .local_env import LocalAvoidConfig
from .set_env import SetLocalAvoidEnv


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


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


def _token_is_risky(token: Sequence[float], config: LocalAvoidConfig) -> bool:
    distance = float(token[1]) * config.threat_distance_scale_m
    closing = float(token[5]) * config.relative_speed_scale_mps
    time_to_cpa = float(token[6]) * config.t_cpa_scale_s
    cpa_distance = float(token[7]) * config.threat_distance_scale_m
    return bool(
        distance <= config.risk_distance_m
        or (
            closing > 0.0
            and time_to_cpa <= config.risk_t_cpa_s
            and cpa_distance <= config.risk_d_cpa_m
        )
    )


def _token_is_urgent(
    token: Sequence[float],
    config: LocalAvoidConfig,
    gate: Any,
    *,
    type_slot: int,
    terminal_mode: bool,
) -> bool:
    scale = gate.terminal_risk_scale if terminal_mode else 1.0
    distance = float(token[1]) * config.threat_distance_scale_m
    closing = float(token[5]) * config.relative_speed_scale_mps
    time_to_cpa = float(token[6]) * config.t_cpa_scale_s
    cpa_distance = float(token[7]) * config.threat_distance_scale_m
    return bool(
        distance <= gate.emergency_distance_m[type_slot]
        or distance <= gate.engage_distance_m[type_slot] * scale
        or (
            closing > 0.0
            and 0.0 <= time_to_cpa <= gate.engage_t_cpa_s[type_slot] * scale
            and cpa_distance <= gate.engage_d_cpa_m[type_slot] * scale
        )
    )


class SetLocalAvoidRuntimeEncoder:
    OBSERVATION_DIM = SetLocalAvoidEnv.ACTOR_OBS_DIM

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
            allow_fused_tracks=True,
        )
        self.last_diagnostics: dict[str, int] = {}

    @staticmethod
    def _candidate_token(
        candidate: tuple[float, float, tuple[float, ...]],
        config: LocalAvoidConfig,
    ) -> list[float]:
        distance, source, values = candidate
        bearing_sin = float(-values[2])
        bearing_cos = float(values[1])
        relative_speed = float(np.linalg.norm(values[5:8]) * 2_000.0)
        closing = float(values[8] * 2_000.0)
        time_to_cpa = float(values[9] * 120.0)
        cpa_distance = float(values[10] * config.threat_distance_scale_m)
        return [
            1.0,
            float(np.clip(distance / config.threat_distance_scale_m, 0.0, 2.0)),
            bearing_sin,
            bearing_cos,
            float(np.clip(relative_speed / config.relative_speed_scale_mps, 0.0, 2.0)),
            float(np.clip(closing / config.relative_speed_scale_mps, -2.0, 2.0)),
            float(np.clip(time_to_cpa / config.t_cpa_scale_s, 0.0, 2.0)),
            float(np.clip(cpa_distance / config.threat_distance_scale_m, 0.0, 2.0)),
            float(source if config.encode_track_source else 1.0 / 3.0),
        ]

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
        goal_distance, goal_heading = _target_geometry(own_position, goal_position)
        own_heading = _ecef_heading(
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
        tokens = [
            self._candidate_token(candidate, self.config)
            for candidate in self._base.threat_candidates
        ]
        tokens.sort(
            key=lambda token: (
                not _token_is_risky(token, self.config),
                float(token[6]),
                float(token[7]),
                float(token[1]),
            )
        )
        encoded_tokens = tokens[: SetLocalAvoidEnv.MAX_THREATS]
        for slot in range(SetLocalAvoidEnv.MAX_THREATS):
            features.extend(
                encoded_tokens[slot]
                if slot < len(encoded_tokens)
                else [0.0] * SetLocalAvoidEnv.THREAT_FEATURES
            )
        previous_action = int(np.clip(maneuver_state + 1, 0, 2))
        action_context = [0.0, 0.0, 0.0]
        if self.config.actor_action_history:
            action_context[previous_action] = 1.0
        elapsed = max(0, int(observation.get("step", 0)) - int(launch_step))
        features.extend(
            [
                *action_context,
                float(np.clip(elapsed / max(self.config.horizon_steps, 1), 0.0, 1.0)),
            ]
        )
        features.extend(float(type_slot == index) for index in range(3))
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (self.OBSERVATION_DIM,):
            raise RuntimeError(f"unexpected set runtime observation shape {result.shape}")
        self.last_diagnostics = {
            "detected_interceptors": int(self._base.detected_interceptor_count),
            "eligible_interceptors": len(tokens),
            "risky_interceptors": sum(
                int(_token_is_risky(token, self.config)) for token in tokens
            ),
            "encoded_interceptors": len(encoded_tokens),
            "truncated_interceptors": max(0, len(tokens) - len(encoded_tokens)),
        }
        return result

    def reset(self) -> None:
        self._base.reset()
        self.last_diagnostics = {}


def set_mission_gate_action_masks(
    observations: np.ndarray,
    environment: LocalAvoidConfig | Sequence[LocalAvoidConfig],
    gate: Any,
) -> np.ndarray:
    values = np.asarray(observations, dtype=np.float32)
    single = values.ndim == 1
    rows = values.reshape(1, -1) if single else values
    if rows.ndim != 2 or rows.shape[1] != SetLocalAvoidEnv.ACTOR_OBS_DIM:
        raise ValueError("set mission gate expects 85-feature observations")
    if isinstance(environment, LocalAvoidConfig):
        configs = [environment] * rows.shape[0]
    else:
        configs = list(environment)
        if len(configs) != rows.shape[0]:
            raise ValueError("one mission-gate config is required per observation")
    masks = np.zeros((rows.shape[0], 3), dtype=np.bool_)
    for row_index, (observation, config) in enumerate(zip(rows, configs)):
        type_slot = int(np.argmax(observation[SetLocalAvoidEnv.TYPE_START:SetLocalAvoidEnv.TYPE_START + 3]))
        goal_distance = float(observation[3]) * config.goal_distance_scale_m
        terminal_mode = bool(goal_distance <= gate.terminal_goal_distance_m[type_slot])
        blocks = observation[
            SetLocalAvoidEnv.THREAT_START:SetLocalAvoidEnv.CONTEXT_START
        ].reshape(SetLocalAvoidEnv.MAX_THREATS, SetLocalAvoidEnv.THREAT_FEATURES)
        urgent = any(
            token[0] > 0.5
            and _token_is_urgent(
                token,
                config,
                gate,
                type_slot=type_slot,
                terminal_mode=terminal_mode,
            )
            for token in blocks
        )
        if urgent:
            masks[row_index, :] = True
        else:
            masks[row_index, 1] = True
    return masks[0] if single else masks


def set_terminal_lockout_action_masks(
    observations: np.ndarray,
    environment: LocalAvoidConfig | Sequence[LocalAvoidConfig],
    gate: Any,
) -> np.ndarray:
    """Terminal lockout for the set-observation actor (31 features).

    Mirrors :func:`set_mission_gate_action_masks` but uses the lockout
    thresholds: inside the terminal envelope only ``STOP_MANEUVER`` is allowed
    unless an interceptor is inside the lockout emergency radius and closing.
    """
    values = np.asarray(observations, dtype=np.float32)
    single = values.ndim == 1
    rows = values.reshape(1, -1) if single else values
    if rows.ndim != 2 or rows.shape[1] != SetLocalAvoidEnv.ACTOR_OBS_DIM:
        raise ValueError("set terminal lockout expects 31-feature observations")
    if isinstance(environment, LocalAvoidConfig):
        configs = [environment] * rows.shape[0]
    else:
        configs = list(environment)
        if len(configs) != rows.shape[0]:
            raise ValueError("one mission-gate config is required per observation")
    masks = np.zeros((rows.shape[0], 3), dtype=np.bool_)
    for row_index, (observation, config) in enumerate(zip(rows, configs)):
        type_slot = int(
            np.argmax(
                observation[
                    SetLocalAvoidEnv.TYPE_START : SetLocalAvoidEnv.TYPE_START + 3
                ]
            )
        )
        goal_distance = float(observation[3]) * config.goal_distance_scale_m
        terminal = bool(goal_distance <= gate.lockout_goal_distance_m[type_slot])
        if not terminal:
            masks[row_index, :] = True
            continue
        blocks = observation[
            SetLocalAvoidEnv.THREAT_START : SetLocalAvoidEnv.CONTEXT_START
        ].reshape(SetLocalAvoidEnv.MAX_THREATS, SetLocalAvoidEnv.THREAT_FEATURES)
        urgent = any(
            token[0] > 0.5
            and float(token[1]) * config.threat_distance_scale_m
            <= gate.lockout_emergency_distance_m[type_slot]
            and float(token[5]) * config.relative_speed_scale_mps > 0.0
            for token in blocks
        )
        if urgent:
            masks[row_index, :] = True
        else:
            masks[row_index, 1] = True
    return masks[0] if single else masks


def set_observation_diagnostics(
    observation: np.ndarray,
    environment: LocalAvoidConfig,
    gate: Any,
) -> dict[str, int]:
    values = np.asarray(observation, dtype=np.float32)
    type_slot = int(
        np.argmax(values[SetLocalAvoidEnv.TYPE_START:SetLocalAvoidEnv.TYPE_START + 3])
    )
    goal_distance = float(values[3]) * environment.goal_distance_scale_m
    terminal_mode = bool(goal_distance <= gate.terminal_goal_distance_m[type_slot])
    blocks = values[
        SetLocalAvoidEnv.THREAT_START:SetLocalAvoidEnv.CONTEXT_START
    ].reshape(SetLocalAvoidEnv.MAX_THREATS, SetLocalAvoidEnv.THREAT_FEATURES)
    present = [token for token in blocks if token[0] > 0.5]
    return {
        "encoded_interceptors": len(present),
        "urgent_interceptors": sum(
            int(
                _token_is_urgent(
                    token,
                    environment,
                    gate,
                    type_slot=type_slot,
                    terminal_mode=terminal_mode,
                )
            )
            for token in present
        ),
    }
