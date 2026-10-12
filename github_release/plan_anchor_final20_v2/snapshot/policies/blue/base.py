# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
import math


@dataclass(frozen=True)
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class ThreatTrack:
    target_id: int
    name: str
    type: int
    lla: Vector3
    pos_ecf: Vector3
    vel_ecf: Vector3
    detect_time: int


@dataclass(frozen=True)
class InterceptorState:
    interceptor_id: int
    lla: Vector3
    pos_ecf: Vector3
    vel_ecf: Vector3
    available: bool
    max_launch_angle_deg: float = 180.0


@dataclass(frozen=True)
class DefendedAsset:
    asset_id: int
    name: str
    lla: Vector3
    pos_ecf: Vector3
    health: float
    value: float


@dataclass
class BlueObservation:
    sim_time: float
    targets: list[ThreatTrack]
    interceptors: list[InterceptorState]
    assets: list[DefendedAsset]
    launched_map: dict[int, list[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class InterceptorAssignment:
    interceptor_id: int
    target_id: int
    target_pos_ecf: Vector3
    target_vel_ecf: Vector3
    priority: float = 0.0


@dataclass(frozen=True)
class ScheduledInterceptorAssignment:
    interceptor_id: int
    target_id: int
    target_pos_ecf: Vector3
    target_vel_ecf: Vector3
    decision_time: float
    planned_launch_time: float
    estimated_intercept_time: float
    priority: float = 0.0
    coordination_id: str | None = None
    coordination_mode: str | None = None
    spatial_mode: str | None = None
    launch_pos_ecf: Vector3 | None = None
    estimated_intercept_pos_ecf: Vector3 | None = None


@dataclass(frozen=True)
class PlannedInterceptorAssignment:
    """An internal delayed release owned entirely by a blue policy."""

    assignment: InterceptorAssignment
    due_time: float
    wave_index: int = 0


class BluePolicy(ABC):
    name = "base"
    plans_launch_time = False
    preserves_pending_launches = False

    def __init__(self, max_shots_per_target: int = 5):
        self.max_shots_per_target = max(1, int(max_shots_per_target))

    @abstractmethod
    def decide(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorAssignment | ScheduledInterceptorAssignment]:
        pass

    def available_interceptors(self, observation: BlueObservation) -> list[InterceptorState]:
        return [item for item in observation.interceptors if item.available]

    def remaining_need(self, observation: BlueObservation, target_id: int) -> int:
        launched_count = len(observation.launched_map.get(target_id, []))
        return max(0, self.max_shots_per_target - launched_count)


class ScheduledBluePolicy(BluePolicy):
    """Stateful scheduler for policies that delay or split launch decisions."""

    def __init__(self, max_shots_per_target: int = 5):
        super().__init__(max_shots_per_target=max_shots_per_target)
        self._pending: list[PlannedInterceptorAssignment] = []
        self._last_sim_time = float("-inf")

    def available_interceptors_for_planning(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorState]:
        self._reset_if_new_round(observation)
        reserved = {item.assignment.interceptor_id for item in self._pending}
        launched = self._launched_interceptor_ids(observation)
        return [
            item
            for item in self.available_interceptors(observation)
            if item.interceptor_id not in reserved
            and item.interceptor_id not in launched
        ]

    def remaining_need_with_pending(
        self,
        observation: BlueObservation,
        target_id: int,
    ) -> int:
        pending_count = sum(
            item.assignment.target_id == target_id
            for item in self._pending
        )
        return max(
            0,
            self.remaining_need(observation, target_id) - pending_count,
        )

    def schedule_assignments(
        self,
        observation: BlueObservation,
        plans: list[PlannedInterceptorAssignment],
    ) -> None:
        self._drop_invalid_pending(observation)
        reserved = {item.assignment.interceptor_id for item in self._pending}
        pending_by_target: dict[int, int] = {}
        for item in self._pending:
            target_id = item.assignment.target_id
            pending_by_target[target_id] = (
                pending_by_target.get(target_id, 0) + 1
            )

        for plan in plans:
            interceptor_id = plan.assignment.interceptor_id
            target_id = plan.assignment.target_id
            if interceptor_id in reserved:
                continue
            remaining = (
                self.remaining_need(observation, target_id)
                - pending_by_target.get(target_id, 0)
            )
            if remaining <= 0:
                continue
            self._pending.append(plan)
            reserved.add(interceptor_id)
            pending_by_target[target_id] = (
                pending_by_target.get(target_id, 0) + 1
            )

    def release_due_assignments(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorAssignment]:
        self._drop_invalid_pending(observation)
        available_ids = {
            item.interceptor_id
            for item in self.available_interceptors(observation)
        } - self._launched_interceptor_ids(observation)
        launched_by_target = {
            target_id: len(interceptors)
            for target_id, interceptors in observation.launched_map.items()
        }
        target_by_id = {
            target.target_id: target
            for target in observation.targets
        }
        released: list[InterceptorAssignment] = []
        future: list[PlannedInterceptorAssignment] = []
        used_interceptors: set[int] = set()

        for plan in sorted(
            self._pending,
            key=lambda item: (
                item.due_time,
                item.wave_index,
                item.assignment.target_id,
                item.assignment.interceptor_id,
            ),
        ):
            assignment = plan.assignment
            if plan.due_time > observation.sim_time:
                future.append(plan)
                continue
            if (
                assignment.interceptor_id not in available_ids
                or assignment.interceptor_id in used_interceptors
            ):
                continue
            current_count = launched_by_target.get(assignment.target_id, 0)
            if current_count >= self.max_shots_per_target:
                continue
            current_target = target_by_id.get(assignment.target_id)
            if current_target is not None:
                assignment = replace(
                    assignment,
                    target_pos_ecf=current_target.pos_ecf,
                    target_vel_ecf=current_target.vel_ecf,
                )
            released.append(assignment)
            used_interceptors.add(assignment.interceptor_id)
            launched_by_target[assignment.target_id] = current_count + 1

        self._pending = future
        return released

    def _drop_invalid_pending(self, observation: BlueObservation) -> None:
        available_ids = {
            item.interceptor_id
            for item in self.available_interceptors(observation)
        } - self._launched_interceptor_ids(observation)
        self._pending = [
            item
            for item in self._pending
            if item.assignment.interceptor_id in available_ids
            and self.remaining_need(observation, item.assignment.target_id) > 0
        ]

    def _reset_if_new_round(self, observation: BlueObservation) -> None:
        if observation.sim_time < self._last_sim_time:
            self._pending.clear()
        self._last_sim_time = observation.sim_time

    @staticmethod
    def _launched_interceptor_ids(observation: BlueObservation) -> set[int]:
        return {
            interceptor_id
            for interceptors in observation.launched_map.values()
            for interceptor_id in interceptors
        }


def distance_m(a: Vector3, b: Vector3, ignore_z: bool = False) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = 0.0 if ignore_z else a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def speed_mps(v: Vector3, default: float = 1.0) -> float:
    speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    return speed if speed > 1e-6 else default
