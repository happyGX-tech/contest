"""Local observation encoder for independent interceptor evasion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class _TrackAge:
    timestamp: int
    last_changed_step: int


class EvasionObservationEncoder:
    """Encode only own-ship state and locally sensed interceptor tracks.

    Fused tracks are intentionally rejected by default. Parameter sharing is
    allowed, but missiles neither consume peer observations nor a global state.
    """

    MAX_THREATS = 3
    SELF_FEATURES = 15
    THREAT_FEATURES = 12
    OBSERVATION_DIM = SELF_FEATURES + MAX_THREATS * THREAT_FEATURES

    TIME_INDEX = 0
    HEALTH_INDEX = 4
    MANEUVER_START = 8
    SPEED_INDEX = 11
    THREAT_START = SELF_FEATURES
    THREAT_VALID_OFFSET = 0
    THREAT_DISTANCE_OFFSET = 4
    THREAT_CLOSING_OFFSET = 8
    THREAT_TGO_OFFSET = 9
    THREAT_CPA_OFFSET = 10

    def __init__(
        self,
        *,
        entity_id: int,
        max_steps: int,
        max_track_age_steps: int = 3,
        threat_range_m: float = 100_000.0,
        allow_fused_tracks: bool = False,
        maximum_cpa_distance_m: float | None = None,
        maximum_time_to_go_s: float | None = None,
        minimum_closing_speed_mps: float = 0.0,
    ) -> None:
        self.entity_id = int(entity_id)
        self.max_steps = max(1, int(max_steps))
        self.max_track_age_steps = max(0, int(max_track_age_steps))
        self.threat_range_m = max(float(threat_range_m), 1.0)
        self.allow_fused_tracks = bool(allow_fused_tracks)
        self.maximum_cpa_distance_m = (
            None
            if maximum_cpa_distance_m is None
            else max(float(maximum_cpa_distance_m), 0.0)
        )
        self.maximum_time_to_go_s = (
            None
            if maximum_time_to_go_s is None
            else max(float(maximum_time_to_go_s), 0.0)
        )
        self.minimum_closing_speed_mps = max(float(minimum_closing_speed_mps), 0.0)
        self._previous_position: np.ndarray | None = None
        self._previous_step: int | None = None
        self._cached_step: int | None = None
        self._cached_vector: np.ndarray | None = None
        self._cached_threat_sources: tuple[float, ...] = ()
        self._cached_threat_candidates: tuple[
            tuple[float, float, tuple[float, ...]], ...
        ] = ()
        self._detected_interceptor_count = 0
        self._track_ages: dict[int, _TrackAge] = {}

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _vector(cls, value: Any) -> np.ndarray:
        return np.asarray(
            [
                float(cls._field(value, "x", 0.0)),
                float(cls._field(value, "y", 0.0)),
                float(cls._field(value, "z", 0.0)),
            ],
            dtype=np.float64,
        )

    @classmethod
    def threat_valid_indices(cls) -> tuple[int, ...]:
        return tuple(
            cls.THREAT_START + slot * cls.THREAT_FEATURES + cls.THREAT_VALID_OFFSET
            for slot in range(cls.MAX_THREATS)
        )

    @classmethod
    def nearest_distance(cls, observation: np.ndarray) -> float:
        distances = []
        for slot in range(cls.MAX_THREATS):
            offset = cls.THREAT_START + slot * cls.THREAT_FEATURES
            if observation[offset + cls.THREAT_VALID_OFFSET] > 0.5:
                distances.append(float(observation[offset + cls.THREAT_DISTANCE_OFFSET]))
        return min(distances, default=1.0)

    @classmethod
    def maximum_closing_speed(cls, observation: np.ndarray) -> float:
        values = []
        for slot in range(cls.MAX_THREATS):
            offset = cls.THREAT_START + slot * cls.THREAT_FEATURES
            if observation[offset + cls.THREAT_VALID_OFFSET] > 0.5:
                values.append(float(observation[offset + cls.THREAT_CLOSING_OFFSET]))
        return max(values, default=0.0)

    @classmethod
    def minimum_cpa_distance(cls, observation: np.ndarray) -> float:
        values = []
        for slot in range(cls.MAX_THREATS):
            offset = cls.THREAT_START + slot * cls.THREAT_FEATURES
            if observation[offset + cls.THREAT_VALID_OFFSET] > 0.5:
                values.append(float(observation[offset + cls.THREAT_CPA_OFFSET]))
        return min(values, default=1.0)

    @classmethod
    def has_threat(cls, observation: np.ndarray) -> bool:
        return any(observation[index] > 0.5 for index in cls.threat_valid_indices())

    @classmethod
    def nearest_threat_context(cls, observation: np.ndarray) -> dict[str, float]:
        candidates = []
        for slot in range(cls.MAX_THREATS):
            offset = cls.THREAT_START + slot * cls.THREAT_FEATURES
            if observation[offset + cls.THREAT_VALID_OFFSET] <= 0.5:
                continue
            candidates.append((observation[offset + cls.THREAT_DISTANCE_OFFSET], offset))
        if not candidates:
            return {}
        _, offset = min(candidates, key=lambda item: float(item[0]))
        return {
            "distance_km": float(observation[offset + cls.THREAT_DISTANCE_OFFSET] * 100.0),
            "los_right": float(observation[offset + 2]),
            "closing_mps": float(observation[offset + cls.THREAT_CLOSING_OFFSET] * 2000.0),
            "tgo_s": float(observation[offset + cls.THREAT_TGO_OFFSET] * 120.0),
            "cpa_km": float(observation[offset + cls.THREAT_CPA_OFFSET] * 100.0),
        }

    def reset(self) -> None:
        self._previous_position = None
        self._previous_step = None
        self._cached_step = None
        self._cached_vector = None
        self._cached_threat_sources = ()
        self._cached_threat_candidates = ()
        self._detected_interceptor_count = 0
        self._track_ages.clear()

    @property
    def threat_sources(self) -> tuple[float, ...]:
        """Return source confidences aligned with the encoded threat slots.

        Local tracks use 1/3, communication-cluster tracks 2/3, and global
        satellite tracks 1.0. The engine has already resolved same-ID
        conflicts in favor of the satellite track before this encoder runs.
        """

        return self._cached_threat_sources

    @property
    def threat_candidates(
        self,
    ) -> tuple[tuple[float, float, tuple[float, ...]], ...]:
        """Return every fresh interceptor candidate before fixed-slot truncation.

        Each item contains ``(distance_m, source, features)`` and is ordered by
        distance. This side channel lets newer set-based policies consume a
        variable number of legal tracks without changing the legacy 51-feature
        ABI used by existing checkpoints.
        """

        return self._cached_threat_candidates

    @property
    def detected_interceptor_count(self) -> int:
        """Number of interceptor IDs present before freshness/risk filtering."""

        return self._detected_interceptor_count

    def _own_velocity(self, position: np.ndarray, step: int) -> np.ndarray:
        if self._previous_position is None or self._previous_step is None:
            return np.zeros(3, dtype=np.float64)
        elapsed = step - self._previous_step
        if elapsed <= 0:
            return np.zeros(3, dtype=np.float64)
        return (position - self._previous_position) / float(elapsed)

    def _fresh_track(self, track_id: int, timestamp: int, step: int) -> bool:
        age = self._track_ages.get(track_id)
        if age is None or timestamp != age.timestamp:
            self._track_ages[track_id] = _TrackAge(timestamp, step)
            return True
        return step - age.last_changed_step <= self.max_track_age_steps

    def encode(
        self,
        observation: Mapping[str, Any],
        *,
        launch_step: int,
        maneuver_state: int,
    ) -> np.ndarray:
        step = int(observation.get("step", 0))
        if self._cached_step == step and self._cached_vector is not None:
            return self._cached_vector.copy()

        own = observation.get("self") or {}
        position = self._vector(own.get("pos_ecf") or {})
        own_velocity = self._own_velocity(position, step)
        own_speed = float(np.linalg.norm(own_velocity))
        own_direction = own_velocity / max(own_speed, 1e-6)
        local_up = position / max(float(np.linalg.norm(position)), 1e-6)
        local_right = np.cross(own_direction, local_up)
        right_norm = float(np.linalg.norm(local_right))
        if right_norm <= 1e-6:
            local_right = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            local_right /= right_norm
        local_normal = np.cross(local_right, own_direction)
        local_normal /= max(float(np.linalg.norm(local_normal)), 1e-6)
        entity_type = int(own.get("type", 0))
        lla = own.get("position") or {}
        stage = float(own.get("stage", -1.0))
        maneuver = int(np.clip(maneuver_state, -1, 1))
        elapsed = max(0, step - int(launch_step))

        features = [
            float(np.clip(step / self.max_steps, 0.0, 1.0)),
            float(entity_type == 21000),
            float(entity_type == 21001),
            float(entity_type == 21002),
            float(np.clip(float(own.get("health", 0.0)) / 100.0, 0.0, 1.0)),
            float(np.clip(float(lla.get("alt", 0.0)) / 20_000.0, -1.0, 1.0)),
            float(np.clip(stage / 5.0, -0.2, 1.0)),
            float(np.clip(elapsed / self.max_steps, 0.0, 1.0)),
            float(maneuver == -1),
            float(maneuver == 0),
            float(maneuver == 1),
            float(np.clip(own_speed / 2_000.0, 0.0, 2.0)),
            float(own_direction[0]),
            float(own_direction[1]),
            float(own_direction[2]),
        ]

        communication_ids = {
            int(value) for value in (own.get("commRangeInfo") or [])
        }
        threats: list[tuple[float, float, list[float]]] = []
        self._detected_interceptor_count = 0
        for raw_id, track in (own.get("detectInfo") or {}).items():
            if int(self._field(track, "entity_type", -1)) != 24000:
                continue
            self._detected_interceptor_count += 1
            detect_from = int(self._field(track, "detect_from", -1))
            if not self.allow_fused_tracks and detect_from != self.entity_id:
                continue
            if detect_from == self.entity_id:
                source = 1.0 / 3.0
            elif detect_from in communication_ids:
                source = 2.0 / 3.0
            else:
                source = 1.0
            track_id = int(self._field(track, "entity_id", raw_id))
            timestamp = int(self._field(track, "time", 0))
            if not self._fresh_track(track_id, timestamp, step):
                continue
            relative_position = self._vector(self._field(track, "pos_ecf", {})) - position
            distance = float(np.linalg.norm(relative_position))
            if distance <= 1e-6:
                continue
            line_of_sight = relative_position / distance
            interceptor_velocity = self._vector(self._field(track, "vel_ecf", {}))
            relative_velocity = interceptor_velocity - own_velocity
            closing_speed = max(0.0, -float(np.dot(relative_position, relative_velocity)) / distance)
            relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))
            time_to_go = max(
                0.0,
                -float(np.dot(relative_position, relative_velocity))
                / max(relative_speed_sq, 1e-6),
            )
            cpa_vector = relative_position + relative_velocity * time_to_go
            cpa_distance = float(np.linalg.norm(cpa_vector))
            if closing_speed < self.minimum_closing_speed_mps:
                continue
            if (
                self.maximum_time_to_go_s is not None
                and time_to_go > self.maximum_time_to_go_s
            ):
                continue
            if (
                self.maximum_cpa_distance_m is not None
                and cpa_distance > self.maximum_cpa_distance_m
            ):
                continue
            local_los = np.asarray(
                [
                    np.dot(line_of_sight, own_direction),
                    np.dot(line_of_sight, local_right),
                    np.dot(line_of_sight, local_normal),
                ]
            )
            local_relative_velocity = np.asarray(
                [
                    np.dot(relative_velocity, own_direction),
                    np.dot(relative_velocity, local_right),
                    np.dot(relative_velocity, local_normal),
                ]
            )
            normalized_relative_velocity = np.clip(
                local_relative_velocity / 2_000.0, -2.0, 2.0
            )
            threats.append(
                (
                    distance,
                    source,
                    [
                        1.0,
                        float(local_los[0]),
                        float(local_los[1]),
                        float(local_los[2]),
                        float(np.clip(distance / self.threat_range_m, 0.0, 2.0)),
                        float(normalized_relative_velocity[0]),
                        float(normalized_relative_velocity[1]),
                        float(normalized_relative_velocity[2]),
                        float(np.clip(closing_speed / 2_000.0, 0.0, 2.0)),
                        float(np.clip(time_to_go / 120.0, 0.0, 2.5)),
                        float(np.clip(cpa_distance / self.threat_range_m, 0.0, 2.0)),
                        float(closing_speed > 0.0),
                    ],
                )
            )

        threats.sort(key=lambda item: item[0])
        self._cached_threat_candidates = tuple(
            (float(distance), float(source), tuple(float(value) for value in values))
            for distance, source, values in threats
        )
        self._cached_threat_sources = tuple(
            item[1] for item in threats[: self.MAX_THREATS]
        )
        for slot in range(self.MAX_THREATS):
            if slot < len(threats):
                features.extend(threats[slot][2])
            else:
                features.extend([0.0] * self.THREAT_FEATURES)

        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (self.OBSERVATION_DIM,):
            raise RuntimeError(f"Unexpected evasion observation shape: {vector.shape}")
        self._previous_position = position
        self._previous_step = step
        self._cached_step = step
        self._cached_vector = vector
        return vector.copy()
