# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .base import (
    BlueObservation,
    BluePolicy,
    InterceptorAssignment,
    InterceptorState,
    ScheduledInterceptorAssignment,
    ThreatTrack,
    Vector3,
    distance_m,
    speed_mps,
)
from .b3_min_cost_assignment import B3MinCostAssignmentPolicy
from .config import DEFAULT_MAX_SHOTS_PER_TARGET


@dataclass(frozen=True)
class _Candidate:
    assignment: ScheduledInterceptorAssignment
    cost: float


@dataclass(frozen=True)
class _ThreatAssessment:
    impact_time: float
    protection_value: float


class B4JointTimingAssignmentPolicy(BluePolicy):
    name = "b4_joint_timing_assignment"
    plans_launch_time = True
    preserves_pending_launches = True

    MAX_LAUNCH_RANGE_M = 600_000.0
    INTERCEPTOR_SPEED_MPS = 1200.0
    PLANNING_HORIZON_MS = 30_000
    PLANNING_STEP_MS = 2_000
    SAFETY_MARGIN_MS = 5_000
    SOLVE_TIME_LIMIT_S = 3.0
    BASE_AMMO_COST = 0.01
    INTERCEPT_TIME_COST = 0.05
    SECOND_SHOT_VALUE = 0.6

    def __init__(self, max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET):
        super().__init__(max_shots_per_target=max_shots_per_target)
        self.immediate_policy = B3MinCostAssignmentPolicy(
            max_shots_per_target=max_shots_per_target,
        )
        self.last_solve_stats: dict[str, float | int | str | bool] = {}

    def reset(self) -> None:
        self.last_solve_stats = {}
        self.immediate_policy.reset()

    def decide(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorAssignment | ScheduledInterceptorAssignment]:
        started_at = time.perf_counter()
        candidates = []
        available = self.available_interceptors(observation)
        for target in observation.targets:
            for interceptor in available:
                candidates.extend(self._candidates(observation, target, interceptor))
        if not candidates:
            self.last_solve_stats = {
                "status": "no_candidates",
                "optimal": True,
                "fallback": False,
                "candidate_count": 0,
                "observed_target_count": len(observation.targets),
                "candidate_target_count": 0,
                "selected_target_count": 0,
                "selected_assignment_count": 0,
                "solve_time_s": time.perf_counter() - started_at,
                "mip_gap": 0.0,
            }
            return []

        candidate_target_ids = {
            item.assignment.target_id
            for item in candidates
        }
        shot_levels = [
            (target.target_id, level)
            for target in observation.targets
            if target.target_id in candidate_target_ids
            for level in range(
                1,
                self.remaining_need(observation, target.target_id) + 1,
            )
        ]
        objective = self._objective(
            observation,
            candidates,
            shot_levels,
        )
        result = milp(
            c=objective,
            integrality=np.ones(len(objective)),
            bounds=Bounds(0.0, 1.0),
            constraints=self._constraints(
                observation,
                candidates,
                shot_levels,
            ),
            options={
                "mip_rel_gap": 0.0,
                "time_limit": self.SOLVE_TIME_LIMIT_S,
            },
        )
        if result.x is None:
            fallback_assignments = self.immediate_policy.decide(observation)
            self.last_solve_stats = {
                "status": "b3_immediate",
                "optimal": False,
                "fallback": True,
                "candidate_count": len(candidates),
                "observed_target_count": len(observation.targets),
                "candidate_target_count": len(candidate_target_ids),
                "selected_target_count": len(
                    {item.target_id for item in fallback_assignments}
                ),
                "selected_assignment_count": len(fallback_assignments),
                "solve_time_s": time.perf_counter() - started_at,
                "mip_gap": float("inf"),
            }
            return fallback_assignments
        selected = [
            candidate.assignment
            for index, candidate in enumerate(candidates)
            if result.x[index] > 0.5
        ]
        self.last_solve_stats = {
            "status": "optimal" if result.status == 0 else "feasible",
            "optimal": result.status == 0,
            "fallback": False,
            "candidate_count": len(candidates),
            "observed_target_count": len(observation.targets),
            "candidate_target_count": len(candidate_target_ids),
            "selected_target_count": len(
                {item.target_id for item in selected}
            ),
            "selected_assignment_count": len(selected),
            "solve_time_s": time.perf_counter() - started_at,
            "mip_gap": float(result.mip_gap),
        }
        return sorted(
            selected,
            key=lambda item: (
                item.planned_launch_time,
                -item.priority,
                item.target_id,
                item.interceptor_id,
            ),
        )

    def _candidates(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
        interceptor: InterceptorState,
    ) -> list[_Candidate]:
        feasible_slots = self._feasible_slots(observation.sim_time, target, interceptor)
        assessment = self._assess_threat(observation, target)
        if assessment is None:
            return []
        candidates = []
        for planned_time, target_pos, _ in feasible_slots:
            intercept_duration = self._relative_intercept_duration_ms(
                interceptor.pos_ecf,
                target_pos,
                target.vel_ecf,
            )
            if intercept_duration is None:
                continue
            estimated_intercept_time = planned_time + intercept_duration
            if (
                estimated_intercept_time + self.SAFETY_MARGIN_MS
                >= assessment.impact_time
            ):
                continue
            cost = self._assignment_cost(
                observation,
                target,
                planned_time,
                intercept_duration,
                assessment,
            )
            candidates.append(
                _Candidate(
                    assignment=ScheduledInterceptorAssignment(
                        interceptor_id=interceptor.interceptor_id,
                        target_id=target.target_id,
                        target_pos_ecf=target_pos,
                        target_vel_ecf=target.vel_ecf,
                        decision_time=observation.sim_time,
                        planned_launch_time=planned_time,
                        estimated_intercept_time=estimated_intercept_time,
                        priority=-cost,
                        launch_pos_ecf=interceptor.pos_ecf,
                        estimated_intercept_pos_ecf=self._project_intercept_position(
                            target_pos,
                            target.vel_ecf,
                            intercept_duration,
                        ),
                    ),
                    cost=cost,
                )
            )
        return candidates

    def _objective(
        self,
        observation: BlueObservation,
        candidates: list[_Candidate],
        shot_levels: list[tuple[int, int]],
    ) -> np.ndarray:
        assessments = {
            target.target_id: self._assess_threat(observation, target)
            for target in observation.targets
        }
        shot_values = [
            -assessments[target_id].protection_value
            * (
                1.0
                if level == 1
                else self.SECOND_SHOT_VALUE ** (level - 1)
            )
            for target_id, level in shot_levels
        ]
        return np.array(
            [candidate.cost for candidate in candidates]
            + shot_values
        )

    def _constraints(
        self,
        observation: BlueObservation,
        candidates: list[_Candidate],
        shot_levels: list[tuple[int, int]],
    ) -> LinearConstraint:
        group_indices: dict[tuple, list[int]] = {}
        group_limits: dict[tuple, float] = {}
        for index, item in enumerate(candidates):
            assignment = item.assignment
            interceptor_key = ("interceptor", assignment.interceptor_id)
            target_key = ("target", assignment.target_id)
            for key in (interceptor_key, target_key):
                group_indices.setdefault(key, []).append(index)
            group_limits[interceptor_key] = 1.0
            group_limits[target_key] = float(
                self.remaining_need(observation, assignment.target_id)
            )

        groups = [
            (group_indices[key], group_limits[key])
            for key in sorted(group_indices)
        ]

        rows = []
        columns = []
        values = []
        lower_bounds = []
        upper_bounds = []
        for row, (indices, _) in enumerate(groups):
            rows.extend([row] * len(indices))
            columns.extend(indices)
            values.extend([1.0] * len(indices))
            lower_bounds.append(0.0)
            upper_bounds.append(groups[row][1])

        target_ids = sorted({target_id for target_id, _ in shot_levels})
        for target_id in target_ids:
            row = len(lower_bounds)
            target_candidates = [
                index
                for index, item in enumerate(candidates)
                if item.assignment.target_id == target_id
            ]
            target_levels = [
                len(candidates) + index
                for index, (item_target_id, _) in enumerate(shot_levels)
                if item_target_id == target_id
            ]
            rows.extend([row] * (len(target_candidates) + len(target_levels)))
            columns.extend(target_candidates + target_levels)
            values.extend(
                [1.0] * len(target_candidates)
                + [-1.0] * len(target_levels)
            )
            lower_bounds.append(0.0)
            upper_bounds.append(0.0)

        for index, (target_id, level) in enumerate(shot_levels):
            if level == 1:
                continue
            previous_index = shot_levels.index((target_id, level - 1))
            row = len(lower_bounds)
            rows.extend([row, row])
            columns.extend(
                [
                    len(candidates) + index,
                    len(candidates) + previous_index,
                ]
            )
            values.extend([1.0, -1.0])
            lower_bounds.append(float("-inf"))
            upper_bounds.append(0.0)

        matrix = coo_matrix(
            (values, (rows, columns)),
            shape=(len(lower_bounds), len(candidates) + len(shot_levels)),
        )
        return LinearConstraint(
            matrix,
            np.array(lower_bounds),
            np.array(upper_bounds),
        )

    def _feasible_slots(
        self,
        sim_time: float,
        target: ThreatTrack,
        interceptor: InterceptorState,
    ) -> list[tuple[float, Vector3, float]]:
        slots = []
        for delay_ms in range(0, self.PLANNING_HORIZON_MS + 1, self.PLANNING_STEP_MS):
            target_pos = self._project_position(target, delay_ms)
            if not self._launch_eligible(interceptor, target_pos, target.vel_ecf):
                continue
            intercept_duration = (
                distance_m(interceptor.pos_ecf, target_pos)
                / self.INTERCEPTOR_SPEED_MPS
                * 1000.0
            )
            slots.append((sim_time + delay_ms, target_pos, intercept_duration))
        return slots

    def _assignment_cost(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
        planned_time: float,
        intercept_duration: float,
        assessment: _ThreatAssessment | None = None,
    ) -> float:
        if assessment is None:
            assessment = self._assess_threat(observation, target)
        if assessment is None:
            return float("inf")
        impact_window = max(
            assessment.impact_time - observation.sim_time,
            1.0,
        )
        intercept_elapsed = (
            planned_time
            + intercept_duration
            - observation.sim_time
        )
        return (
            self.BASE_AMMO_COST
            + self.INTERCEPT_TIME_COST
            * intercept_elapsed
            / impact_window
        )

    def _assess_threat(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
    ) -> _ThreatAssessment | None:
        target_speed = self._length(target.vel_ecf)
        if target_speed <= 1e-6 or not observation.assets:
            return None

        def alignment(asset) -> float:
            direction = Vector3(
                asset.pos_ecf.x - target.pos_ecf.x,
                asset.pos_ecf.y - target.pos_ecf.y,
                asset.pos_ecf.z - target.pos_ecf.z,
            )
            distance = self._length(direction)
            if distance <= 1e-6:
                return 1.0
            return (
                target.vel_ecf.x * direction.x
                + target.vel_ecf.y * direction.y
                + target.vel_ecf.z * direction.z
            ) / (target_speed * distance)

        asset = max(observation.assets, key=alignment)
        impact_distance = distance_m(target.pos_ecf, asset.pos_ecf)
        impact_duration_ms = impact_distance / target_speed * 1000.0
        damage = self._expected_damage(target.type, asset.name)
        damage_ratio = min(damage / max(asset.health, 1e-6), 1.0)
        urgency = 1.0 + min(
            2.0,
            self.PLANNING_HORIZON_MS / max(impact_duration_ms, 1.0),
        )
        return _ThreatAssessment(
            impact_time=observation.sim_time + impact_duration_ms,
            protection_value=asset.value * damage_ratio * urgency,
        )

    @staticmethod
    def _expected_damage(target_type: int, asset_name: str) -> float:
        is_ship = asset_name.startswith("暗巢")
        is_site = asset_name.startswith("哨卫巢")
        if target_type == 21002:
            return 0.8 if is_ship else 0.0
        if target_type == 21000:
            return 0.0 if is_ship else 20.0 * (0.6 if is_site else 0.8)
        if target_type == 21001:
            return 0.0 if is_ship else 5.0 * (0.6 if is_site else 0.8)
        return 0.0

    def _relative_intercept_duration_ms(
        self,
        interceptor_pos: Vector3,
        target_pos: Vector3,
        target_vel: Vector3,
    ) -> float | None:
        relative_x = target_pos.x - interceptor_pos.x
        relative_y = target_pos.y - interceptor_pos.y
        relative_z = target_pos.z - interceptor_pos.z
        a = (
            target_vel.x * target_vel.x
            + target_vel.y * target_vel.y
            + target_vel.z * target_vel.z
            - self.INTERCEPTOR_SPEED_MPS * self.INTERCEPTOR_SPEED_MPS
        )
        b = 2.0 * (
            relative_x * target_vel.x
            + relative_y * target_vel.y
            + relative_z * target_vel.z
        )
        c = (
            relative_x * relative_x
            + relative_y * relative_y
            + relative_z * relative_z
        )
        if abs(a) < 1e-9:
            if abs(b) < 1e-9:
                return None
            roots = [-c / b]
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant < 0:
                return None
            root = math.sqrt(discriminant)
            roots = [
                (-b - root) / (2.0 * a),
                (-b + root) / (2.0 * a),
            ]
        positive_roots = [value for value in roots if value > 0]
        if not positive_roots:
            return None
        return min(positive_roots) * 1000.0

    @staticmethod
    def _project_intercept_position(
        target_pos: Vector3,
        target_vel: Vector3,
        intercept_duration_ms: float,
    ) -> Vector3:
        duration_s = intercept_duration_ms / 1000.0
        return Vector3(
            target_pos.x + target_vel.x * duration_s,
            target_pos.y + target_vel.y * duration_s,
            target_pos.z + target_vel.z * duration_s,
        )

    def _asset_arrival_time(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
    ) -> float | None:
        if not observation.assets:
            return None
        nearest_asset = min(
            observation.assets,
            key=lambda asset: distance_m(asset.pos_ecf, target.pos_ecf),
        )
        travel_time_ms = (
            distance_m(nearest_asset.pos_ecf, target.pos_ecf)
            / speed_mps(target.vel_ecf, default=300.0)
            * 1000.0
        )
        return observation.sim_time + travel_time_ms

    def _threat_score(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
    ) -> float:
        type_score = {
            21000: 30.0,
            21001: 20.0,
            21002: 10.0,
        }.get(target.type, 1.0)
        if not observation.assets:
            return type_score
        nearest_asset = min(
            observation.assets,
            key=lambda asset: distance_m(asset.pos_ecf, target.pos_ecf),
        )
        eta = (
            distance_m(nearest_asset.pos_ecf, target.pos_ecf)
            / speed_mps(target.vel_ecf, default=300.0)
        )
        return type_score + nearest_asset.value / max(eta, 1.0) * 1000.0

    @staticmethod
    def _project_position(target: ThreatTrack, delay_ms: int) -> Vector3:
        delay_s = delay_ms / 1000.0
        return Vector3(
            target.pos_ecf.x + target.vel_ecf.x * delay_s,
            target.pos_ecf.y + target.vel_ecf.y * delay_s,
            target.pos_ecf.z + target.vel_ecf.z * delay_s,
        )

    def _launch_eligible(
        self,
        interceptor: InterceptorState,
        target_pos: Vector3,
        target_vel: Vector3,
    ) -> bool:
        relative = Vector3(
            interceptor.pos_ecf.x - target_pos.x,
            interceptor.pos_ecf.y - target_pos.y,
            interceptor.pos_ecf.z - target_pos.z,
        )
        if self._length(relative) >= self.MAX_LAUNCH_RANGE_M:
            return False
        return self._angle_deg(relative, target_vel) < interceptor.max_launch_angle_deg

    @staticmethod
    def _length(vector: Vector3) -> float:
        return math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)

    @classmethod
    def _angle_deg(cls, left: Vector3, right: Vector3) -> float:
        denominator = cls._length(left) * cls._length(right)
        if denominator == 0:
            return 0.0
        cosine = (left.x * right.x + left.y * right.y + left.z * right.z) / denominator
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
