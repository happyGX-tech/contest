"""Inference adapter for standalone 33-feature local avoidance checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from policies.red.learning.red_policy import HighLevelAction, PolicyTransition, SharedPolicy

from .encoder import EvasionObservationEncoder
from .local_env import LocalAvoidConfig
from .local_ppo import LocalAvoidPPO
from .recurrent_ppo import RecurrentLocalAvoidPPO
from .set_attention_ppo import SetAttentionLocalAvoidPPO
from .fixed_slot_ppo import FixedSlotLocalAvoidPPO, V13ResidualLocalAvoidPPO
from .set_env import SetLocalAvoidEnv
from .set_runtime import (
    SetLocalAvoidRuntimeEncoder,
    _token_is_urgent,
    _token_is_risky,
    set_mission_gate_action_masks,
    set_observation_diagnostics,
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True, slots=True)
class MissionGateConfig:
    """Mission-aware deployment guard around the frozen Stage-3 actor.

    Values are ordered H/M/L.  The gate deliberately keeps the learned actor
    unchanged: it only decides whether an interceptor track is urgent enough to
    justify evasive control, or whether the missile should recover toward its
    assigned target.
    """

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


def _recovery_action_mask(
    observation: np.ndarray,
    action_mask: np.ndarray,
    *,
    deadband_deg: float,
) -> np.ndarray:
    """Turn back toward the assigned target after an evasive manoeuvre."""

    goal_error = math.atan2(float(observation[4]), float(observation[5]))
    if abs(goal_error) <= math.radians(deadband_deg):
        action = HighLevelAction.STOP_MANEUVER
    # The native cruise-missile body Z axis has the opposite sign to the
    # mathematical heading used by the standalone 2-D trainer.
    elif goal_error > 0.0:
        action = HighLevelAction.MANEUVER_LEFT
    else:
        action = HighLevelAction.MANEUVER_RIGHT
    return _only_action(action_mask, action)


def _missile_type_slot(observation: np.ndarray) -> int:
    if np.asarray(observation).shape[-1] == SetLocalAvoidEnv.ACTOR_OBS_DIM:
        start = SetLocalAvoidEnv.TYPE_START
    else:
        start = 30
    encoded = np.asarray(observation[start:start + 3], dtype=np.float32)
    return int(np.argmax(encoded)) if encoded.size == 3 and encoded.max(initial=0.0) > 0.5 else 0


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


def mission_gate_action_masks(
    observations: np.ndarray,
    environment: LocalAvoidConfig | Sequence[LocalAvoidConfig],
    gate: MissionGateConfig | None = None,
) -> np.ndarray:
    """Build deployment-equivalent PPO masks for standalone training.

    The local environment's straight action restores the goal heading, so its
    single allowed non-urgent action models both native straight flight and the
    deterministic recovery branch used by :class:`LocalAvoidRuntimePolicy`.
    """

    values = np.asarray(observations, dtype=np.float32)
    single = values.ndim == 1
    rows = values.reshape(1, -1) if single else values
    if rows.ndim != 2 or rows.shape[1] != LocalAvoidRuntimeEncoder.OBSERVATION_DIM:
        raise ValueError("mission gate expects observations with 33 features")
    if isinstance(environment, LocalAvoidConfig):
        configs = [environment] * rows.shape[0]
    else:
        configs = list(environment)
        if len(configs) != rows.shape[0]:
            raise ValueError("one mission-gate config is required per observation")
    gate = gate or MissionGateConfig()
    masks = np.zeros((rows.shape[0], 3), dtype=np.bool_)
    for index, (observation, config) in enumerate(zip(rows, configs)):
        type_slot = _missile_type_slot(observation)
        goal_distance = float(observation[3]) * config.goal_distance_scale_m
        terminal_mode = bool(
            goal_distance <= gate.terminal_goal_distance_m[type_slot]
            or float(observation[29]) >= gate.late_flight_fraction
        )
        if _has_urgent_threat(
            observation,
            config,
            gate,
            type_slot=type_slot,
            terminal_mode=terminal_mode,
        ):
            masks[index, :] = True
        else:
            masks[index, int(HighLevelAction.STOP_MANEUVER)] = True
    return masks[0] if single else masks


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
        self.last_diagnostics: dict[str, int] = {}
        self.last_threat_tokens: tuple[tuple[float, ...], ...] = ()

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
        all_tokens = tuple(
            tuple(SetLocalAvoidRuntimeEncoder._candidate_token(candidate, self.config))
            for candidate in self._base.threat_candidates
        )
        self.last_threat_tokens = all_tokens
        self.last_diagnostics = {
            "detected_interceptors": int(self._base.detected_interceptor_count),
            "eligible_interceptors": len(all_tokens),
            "risky_interceptors": sum(
                int(_token_is_risky(token, self.config)) for token in all_tokens
            ),
            "encoded_interceptors": min(2, len(threats)),
            "truncated_interceptors": max(0, len(all_tokens) - 2),
        }
        return result

    def reset(self) -> None:
        self._base.reset()
        self.last_diagnostics = {}
        self.last_threat_tokens = ()


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
        self.model: (
            LocalAvoidPPO
            | RecurrentLocalAvoidPPO
            | SetAttentionLocalAvoidPPO
            | FixedSlotLocalAvoidPPO
            | V13ResidualLocalAvoidPPO
            | None
        ) = None
        self.environment_config: LocalAvoidConfig | None = None
        self._encoders: dict[
            int, LocalAvoidRuntimeEncoder | SetLocalAvoidRuntimeEncoder
        ] = {}
        self._entity_ids: dict[int, int] = {}
        self._recovery_pending: dict[int, bool] = {}
        self._recurrent_hidden: dict[int, np.ndarray] = {}
        self._current_agent_id: int | None = None
        trace_path = os.getenv("LOCAL_AVOID_TRACE_PATH", "").strip()
        self._trace_path = Path(trace_path) if trace_path else None
        self._trace_trajectories: dict[int, list[dict[str, Any]]] = {}
        self._trace_next_observations: dict[int, np.ndarray] = {}
        self._trace_step_diagnostics: dict[int, dict[str, int]] = {}
        self._prefetched_observations: dict[
            int, tuple[int, int, np.ndarray, dict[str, int]]
        ] = {}
        self._trace_initialized = False
        self.records_evaluation = self._trace_path is not None
        self.motion_learning_enabled = self._trace_path is not None
        self._rollout_mode = (
            os.getenv("LOCAL_AVOID_ROLLOUT_MODE", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._rollout_seed = int(
            os.getenv(
                "LOCAL_AVOID_ROLLOUT_SEED",
                os.getenv("RED_POLICY_SEED", "0"),
            )
        )
        if self._rollout_mode and self._trace_path is None:
            raise ValueError(
                "LOCAL_AVOID_ROLLOUT_MODE requires LOCAL_AVOID_TRACE_PATH"
            )
        configured_gate_mode = os.getenv("LOCAL_AVOID_GATE_MODE")
        self._gate_mode_explicit = configured_gate_mode is not None
        self.gate_mode = (configured_gate_mode or "mission_v12").strip().lower()
        if self.gate_mode not in {"none", "legacy", "mission_v12"}:
            raise ValueError(
                "LOCAL_AVOID_GATE_MODE must be 'none', 'legacy', or 'mission_v12'"
            )
        self.mission_gate = MissionGateConfig()

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algorithm = payload.get("algorithm")
        if algorithm == RecurrentLocalAvoidPPO.algorithm_name:
            loader = RecurrentLocalAvoidPPO
        elif algorithm == SetAttentionLocalAvoidPPO.algorithm_name:
            loader = SetAttentionLocalAvoidPPO
        elif algorithm == FixedSlotLocalAvoidPPO.algorithm_name:
            loader = FixedSlotLocalAvoidPPO
        elif algorithm == V13ResidualLocalAvoidPPO.algorithm_name:
            loader = V13ResidualLocalAvoidPPO
        else:
            loader = LocalAvoidPPO
        self.model, self.environment_config, _payload = loader.load(path, device="cpu")
        if self._rollout_mode:
            torch.manual_seed(self._rollout_seed)
        if not self._gate_mode_explicit:
            recommended_gate_mode = str(
                _payload.get("metrics", {}).get("deployment_gate_mode", "")
            ).strip().lower()
            if recommended_gate_mode in {"none", "legacy", "mission_v12"}:
                self.gate_mode = recommended_gate_mode
        self.model.network.eval()
        self._encoders.clear()
        self._recovery_pending.clear()
        self._recurrent_hidden.clear()
        self._trace_step_diagnostics.clear()
        self._current_agent_id = None

    def _initialize_trace(self) -> None:
        if self._trace_path is None or self._trace_initialized:
            return
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            os.getenv("LOCAL_AVOID_TRACE_OVERWRITE", "0").strip() == "1"
            and self._trace_path.exists()
        ):
            self._trace_path.unlink()
        self._trace_initialized = True

    def _flush_trace(self, agent_id: int, *, incomplete: bool = False) -> None:
        if self._trace_path is None or self.environment_config is None:
            return
        steps = self._trace_trajectories.pop(int(agent_id), None)
        if not steps:
            return
        self._initialize_trace()
        final_observation = np.asarray(steps[-1]["observation"], dtype=np.float32)
        type_slot = _missile_type_slot(final_observation)
        goal_distance_m = (
            float(final_observation[3])
            * self.environment_config.goal_distance_scale_m
        )
        arrival_margin_m = (
            self.environment_config.arrival_radius_m
            + 2.0
            * self.environment_config.unit_speeds_mps[type_slot]
            * self.environment_config.control_interval_sec
        )
        if incomplete:
            terminal = "incomplete"
        elif goal_distance_m <= arrival_margin_m:
            terminal = "arrived_proxy"
        else:
            terminal = "lost_proxy"
        next_observation = self._trace_next_observations.pop(
            int(agent_id), np.zeros_like(final_observation)
        )
        payload = {
            "format": "local-avoid-runtime-trajectory-v3",
            "agent_id": int(agent_id),
            "entity_id": int(self._entity_ids.get(int(agent_id), -1)),
            "missile_type_slot": int(type_slot),
            "terminal": terminal,
            "terminal_goal_distance_m": goal_distance_m,
            "rollout_mode": bool(self._rollout_mode),
            "rollout_seed": int(self._rollout_seed),
            "gate_mode": self.gate_mode,
            "next_observation": np.asarray(
                next_observation, dtype=np.float32
            ).tolist(),
            "steps": steps,
        }
        assert self._trace_path is not None
        with gzip.open(self._trace_path, "at", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
            trace_file.write("\n")

    def observe(self, transition: PolicyTransition) -> None:
        if self._trace_path is None:
            return
        agent_id = int(transition.agent_id)
        row = {
                "observation": np.asarray(
                    transition.observation, dtype=np.float32
                ).tolist(),
                "action": int(transition.action),
                "action_mask": np.asarray(
                    transition.action_mask, dtype=np.bool_
                ).astype(np.uint8).tolist(),
                "reward": float(transition.reward),
                "done": bool(transition.done),
            }
        diagnostics = self._trace_step_diagnostics.get(agent_id)
        if diagnostics:
            row["threat_diagnostics"] = dict(diagnostics)
        self._trace_trajectories.setdefault(agent_id, []).append(row)
        self._trace_next_observations[agent_id] = np.asarray(
            transition.next_observation, dtype=np.float32
        ).copy()
        if transition.done:
            self._flush_trace(agent_id)

    def register_agent(self, agent_id: int, entity_id: int, max_steps: int) -> None:
        if self.environment_config is None:
            self._entity_ids[int(agent_id)] = int(entity_id)
            return
        self._entity_ids[int(agent_id)] = int(entity_id)
        encoder_class = (
            SetLocalAvoidRuntimeEncoder
            if isinstance(
                self.model, (SetAttentionLocalAvoidPPO, FixedSlotLocalAvoidPPO)
            )
            else LocalAvoidRuntimeEncoder
        )
        self._encoders[int(agent_id)] = encoder_class(
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
        encoder = self._encoders.get(int(agent_id))
        if encoder is None:
            self.register_agent(agent_id, entity_id, self.max_steps)
            encoder = self._encoders[int(agent_id)]
        self._current_agent_id = agent_id
        if int(maneuver_state) != 0:
            self._recovery_pending[agent_id] = True
        step = int(observation.get("step", -1))
        prefetched = self._prefetched_observations.pop(agent_id, None)
        if (
            prefetched is not None
            and prefetched[0] == step
            and prefetched[1] == int(maneuver_state)
        ):
            encoded = prefetched[2].copy()
            diagnostics = dict(prefetched[3])
        else:
            encoded = encoder.encode(
                observation,
                launch_step=launch_step,
                maneuver_state=maneuver_state,
                goal_position=goal_position,
            )
            diagnostics = dict(encoder.last_diagnostics)
        if isinstance(encoder, SetLocalAvoidRuntimeEncoder):
            diagnostics.update(
                set_observation_diagnostics(
                    encoded,
                    self.environment_config,
                    self.mission_gate,
                )
            )
        else:
            type_slot = _missile_type_slot(encoded)
            goal_distance = (
                float(encoded[3]) * self.environment_config.goal_distance_scale_m
            )
            terminal_mode = bool(
                goal_distance <= self.mission_gate.terminal_goal_distance_m[type_slot]
                or float(encoded[29]) >= self.mission_gate.late_flight_fraction
            )
            diagnostics["urgent_interceptors"] = sum(
                int(
                    _token_is_urgent(
                        token,
                        self.environment_config,
                        self.mission_gate,
                        type_slot=type_slot,
                        terminal_mode=terminal_mode,
                    )
                )
                for token in encoder.last_threat_tokens
            )
        self._trace_step_diagnostics[agent_id] = diagnostics
        return encoded

    def encode_next_motion_observation(
        self,
        observation: Mapping[str, Any],
        *,
        agent_id: int,
        entity_id: int,
        launch_step: int,
        maneuver_state: int,
        goal_position: Any,
    ) -> np.ndarray:
        """Encode a post-step state once and reuse it at the next decision.

        The threat encoder is stateful.  Encoding here supplies the real
        bootstrap state for truncated rollouts; caching the result prevents the
        next action request from advancing the track cache a second time.
        """

        if not observation.get("self"):
            actor_dim = int(getattr(self.model.config, "actor_obs_dim", 0))
            return np.zeros(actor_dim, dtype=np.float32)
        agent_id = int(agent_id)
        encoder = self._encoders.get(agent_id)
        if encoder is None:
            self.register_agent(agent_id, entity_id, self.max_steps)
            encoder = self._encoders[agent_id]
        encoded = encoder.encode(
            observation,
            launch_step=launch_step,
            maneuver_state=maneuver_state,
            goal_position=goal_position,
        )
        diagnostics = dict(encoder.last_diagnostics)
        self._prefetched_observations[agent_id] = (
            int(observation.get("step", -1)),
            int(maneuver_state),
            encoded.copy(),
            diagnostics,
        )
        return encoded

    def prepare_action_mask(
        self, observation: np.ndarray, action_mask: np.ndarray
    ) -> np.ndarray:
        if self.gate_mode == "none":
            return np.asarray(action_mask, dtype=np.bool_).copy()
        if np.asarray(observation).shape[-1] == SetLocalAvoidEnv.ACTOR_OBS_DIM:
            if self.environment_config is None:
                raise RuntimeError("Local avoidance model has not been loaded")
            gate_mask = set_mission_gate_action_masks(
                observation,
                self.environment_config,
                self.mission_gate,
            )
            if bool(np.asarray(gate_mask).all()):
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
        elapsed_fraction = float(observation[29])
        terminal_mode = bool(
            goal_distance <= self.mission_gate.terminal_goal_distance_m[type_slot]
            or elapsed_fraction >= self.mission_gate.late_flight_fraction
        )
        urgent = _has_urgent_threat(
            observation,
            self.environment_config,
            self.mission_gate,
            type_slot=type_slot,
            terminal_mode=terminal_mode,
        )
        if urgent:
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
        if isinstance(self.model, RecurrentLocalAvoidPPO):
            if self._current_agent_id is None:
                raise RuntimeError("recurrent local avoidance requires an active agent")
            agent_id = self._current_agent_id
            hidden = self._recurrent_hidden.get(agent_id)
            if hidden is None:
                hidden = self.model.initial_hidden(1)[0]
            (
                actions,
                _log_probs,
                _values,
                _actor,
                _critic,
                next_hidden,
            ) = self.model.act(
                np.asarray(observation, dtype=np.float32),
                np.zeros(43, dtype=np.float32),
                hidden,
                deterministic=True,
                update_normalizer=False,
                action_masks=np.asarray(action_mask, dtype=np.bool_),
            )
            self._recurrent_hidden[agent_id] = np.asarray(next_hidden, dtype=np.float32)
        else:
            critic_dim = int(self.model.config.critic_state_dim)
            critic_state = np.zeros(critic_dim, dtype=np.float32)
            actor_values = np.asarray(observation, dtype=np.float32)
            critic_state[: min(actor_values.size, critic_dim)] = actor_values[
                : min(actor_values.size, critic_dim)
            ]
            actions, _log_probs, _values, _actor, _critic = self.model.act(
                actor_values,
                critic_state,
                deterministic=not self._rollout_mode,
                update_normalizer=False,
                action_masks=np.asarray(action_mask, dtype=np.bool_),
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
        for agent_id in tuple(self._trace_trajectories):
            self._flush_trace(agent_id, incomplete=True)
        for encoder in self._encoders.values():
            encoder.reset()
        self._recovery_pending.clear()
        self._recurrent_hidden.clear()
        self._trace_step_diagnostics.clear()
        self._trace_next_observations.clear()
        self._prefetched_observations.clear()
        self._current_agent_id = None

    def finalize_episode(self, *, truncated: bool = False) -> None:
        for agent_id in tuple(self._trace_trajectories):
            self._flush_trace(agent_id, incomplete=truncated)
        self._trace_step_diagnostics.clear()
        self._prefetched_observations.clear()
        self._current_agent_id = None

    def save(self, path: str) -> None:
        raise NotImplementedError("local_avoid_s3 checkpoints are produced by standalone training")
