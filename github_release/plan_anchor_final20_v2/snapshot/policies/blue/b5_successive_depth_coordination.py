# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .base import BlueObservation, BluePolicy, ScheduledInterceptorAssignment, Vector3
from .b4_joint_timing_assignment import B4JointTimingAssignmentPolicy, _Candidate


@dataclass
class _Engagement:
    coordination_id: str
    first: ScheduledInterceptorAssignment
    second: ScheduledInterceptorAssignment | None = None


@dataclass(frozen=True)
class _StageCandidate:
    candidate: _Candidate
    role: str
    coordination_id: str
    objective_cost: float
    intercept_gap_ms: float = 0.0
    depth_separation_m: float = 0.0


class B5SuccessiveDepthCoordinationPolicy(B4JointTimingAssignmentPolicy):
    name = "b5_successive_depth_coordination"
    preserves_pending_launches = True

    MIN_INTERCEPT_GAP_MS = 5_000.0
    CANDIDATES_PER_LAUNCH_SLOT = 4

    def __init__(self, max_shots_per_target: int = 2):
        if max_shots_per_target < 2:
            raise ValueError(
                "B5 successive-depth coordination requires "
                "max_shots_per_target >= 2"
            )
        BluePolicy.__init__(self, max_shots_per_target=max_shots_per_target)
        self.last_solve_stats: dict[str, float | int | str | bool] = {}
        self._engagements: dict[int, _Engagement] = {}
        self._last_sim_time = float("-inf")

    def decide(
        self,
        observation: BlueObservation,
    ) -> list[ScheduledInterceptorAssignment]:
        started_at = time.perf_counter()
        self._synchronize_engagements(observation)
        available = self.available_interceptors(observation)
        stage_candidates: list[_StageCandidate] = []

        for target in observation.targets:
            engagement = self._engagements.get(target.target_id)
            if engagement is None:
                candidates = self._target_candidates(
                    observation,
                    target,
                    available,
                )
                if len({
                    item.assignment.interceptor_id
                    for item in candidates
                }) < 2:
                    continue
                assessment = self._assess_threat(observation, target)
                if assessment is None:
                    continue
                coordination_id = (
                    f"{int(observation.sim_time)}:{target.target_id}"
                )
                stage_candidates.extend(
                    _StageCandidate(
                        candidate=item,
                        role="first",
                        coordination_id=coordination_id,
                        objective_cost=(
                            item.cost - assessment.protection_value
                        ),
                    )
                    for item in candidates
                )
                continue

            if not self._first_was_launched(observation, engagement):
                continue
            if engagement.second is not None:
                continue

            assessment = self._assess_threat(observation, target)
            if assessment is None:
                continue
            for item in self._target_candidates(
                observation,
                target,
                available,
            ):
                if (
                    item.assignment.interceptor_id
                    == engagement.first.interceptor_id
                ):
                    continue
                intercept_gap = (
                    item.assignment.estimated_intercept_time
                    - engagement.first.estimated_intercept_time
                )
                if intercept_gap < self.MIN_INTERCEPT_GAP_MS:
                    continue
                depth_separation = self._depth_separation(
                    engagement.first,
                    item.assignment,
                    target.vel_ecf,
                )
                if depth_separation <= 0.0:
                    continue
                stage_candidates.append(
                    _StageCandidate(
                        candidate=item,
                        role="second",
                        coordination_id=engagement.coordination_id,
                        objective_cost=(
                            item.cost
                            - assessment.protection_value
                            * self.SECOND_SHOT_VALUE
                        ),
                        intercept_gap_ms=intercept_gap,
                        depth_separation_m=depth_separation,
                    )
                )

        if not stage_candidates:
            self._set_stats(
                "no_candidates",
                started_at,
                observation,
                [],
                [],
            )
            return []

        result = milp(
            c=np.array([
                item.objective_cost
                for item in stage_candidates
            ]),
            integrality=np.ones(len(stage_candidates)),
            bounds=Bounds(0.0, 1.0),
            constraints=self._constraints(stage_candidates),
            options={
                "mip_rel_gap": 0.0,
                "time_limit": self.SOLVE_TIME_LIMIT_S,
            },
        )
        if result.x is None:
            self._set_stats(
                "infeasible",
                started_at,
                observation,
                stage_candidates,
                [],
                optimal=False,
                mip_gap=float("inf"),
            )
            return []

        selected = [
            item
            for index, item in enumerate(stage_candidates)
            if result.x[index] > 0.5
        ]
        assignments = []
        for item in selected:
            assignment = replace(
                item.candidate.assignment,
                coordination_id=item.coordination_id,
                coordination_mode="staggered",
                spatial_mode="layered",
            )
            target_id = assignment.target_id
            if item.role == "first":
                self._engagements[target_id] = _Engagement(
                    coordination_id=item.coordination_id,
                    first=assignment,
                )
            else:
                self._engagements[target_id].second = assignment
            assignments.append(assignment)

        self._set_stats(
            "optimal" if result.status == 0 else "feasible",
            started_at,
            observation,
            stage_candidates,
            selected,
            optimal=result.status == 0,
            mip_gap=float(result.mip_gap),
        )
        return sorted(
            assignments,
            key=lambda item: (
                item.planned_launch_time,
                item.target_id,
                item.interceptor_id,
            ),
        )

    def _synchronize_engagements(
        self,
        observation: BlueObservation,
    ) -> None:
        if observation.sim_time < self._last_sim_time:
            self._engagements.clear()
        self._last_sim_time = observation.sim_time
        visible_target_ids = {
            target.target_id
            for target in observation.targets
        }
        for target_id in list(self._engagements):
            if target_id not in visible_target_ids:
                del self._engagements[target_id]
                continue
            engagement = self._engagements[target_id]
            launched_ids = set(
                observation.launched_map.get(target_id, [])
            )
            if (
                observation.sim_time
                >= engagement.first.planned_launch_time
                and engagement.first.interceptor_id not in launched_ids
            ):
                del self._engagements[target_id]
                continue
            if (
                engagement.second is not None
                and observation.sim_time
                >= engagement.second.planned_launch_time
                and engagement.second.interceptor_id not in launched_ids
            ):
                engagement.second = None

    @staticmethod
    def _first_was_launched(
        observation: BlueObservation,
        engagement: _Engagement,
    ) -> bool:
        return (
            observation.sim_time
            >= engagement.first.planned_launch_time
            and engagement.first.interceptor_id
            in observation.launched_map.get(
                engagement.first.target_id,
                [],
            )
        )

    def _target_candidates(
        self,
        observation,
        target,
        available,
    ) -> list[_Candidate]:
        candidates = [
            candidate
            for interceptor in available
            for candidate in self._candidates(
                observation,
                target,
                interceptor,
            )
        ]
        return self._reduce_candidates(candidates)

    def _reduce_candidates(
        self,
        candidates: list[_Candidate],
    ) -> list[_Candidate]:
        by_launch_time: dict[float, list[_Candidate]] = {}
        for candidate in candidates:
            by_launch_time.setdefault(
                candidate.assignment.planned_launch_time,
                [],
            ).append(candidate)

        reduced = []
        for launch_time in sorted(by_launch_time):
            used_interceptors: set[int] = set()
            for candidate in sorted(
                by_launch_time[launch_time],
                key=lambda item: (
                    item.cost,
                    item.assignment.interceptor_id,
                ),
            ):
                interceptor_id = candidate.assignment.interceptor_id
                if interceptor_id in used_interceptors:
                    continue
                reduced.append(candidate)
                used_interceptors.add(interceptor_id)
                if (
                    len(used_interceptors)
                    >= self.CANDIDATES_PER_LAUNCH_SLOT
                ):
                    break
        return reduced

    @classmethod
    def _depth_separation(
        cls,
        first: ScheduledInterceptorAssignment,
        second: ScheduledInterceptorAssignment,
        target_velocity: Vector3,
    ) -> float:
        first_point = first.estimated_intercept_pos_ecf
        second_point = second.estimated_intercept_pos_ecf
        speed = cls._length(target_velocity)
        if first_point is None or second_point is None or speed <= 1e-9:
            return 0.0
        return (
            (second_point.x - first_point.x) * target_velocity.x
            + (second_point.y - first_point.y) * target_velocity.y
            + (second_point.z - first_point.z) * target_velocity.z
        ) / speed

    @staticmethod
    def _constraints(
        candidates: list[_StageCandidate],
    ) -> LinearConstraint:
        groups: dict[tuple[str, int], list[int]] = {}
        for index, item in enumerate(candidates):
            assignment = item.candidate.assignment
            groups.setdefault(
                ("target", assignment.target_id),
                [],
            ).append(index)
            groups.setdefault(
                ("interceptor", assignment.interceptor_id),
                [],
            ).append(index)

        rows = []
        columns = []
        for row, key in enumerate(sorted(groups)):
            indices = groups[key]
            rows.extend([row] * len(indices))
            columns.extend(indices)
        matrix = coo_matrix(
            (np.ones(len(rows)), (rows, columns)),
            shape=(len(groups), len(candidates)),
        )
        return LinearConstraint(matrix, 0.0, 1.0)

    def _set_stats(
        self,
        status: str,
        started_at: float,
        observation: BlueObservation,
        candidates: list[_StageCandidate],
        selected: list[_StageCandidate],
        *,
        optimal: bool = True,
        mip_gap: float = 0.0,
    ) -> None:
        selected_seconds = [
            item
            for item in selected
            if item.role == "second"
        ]
        pair_count = len(selected_seconds)
        self.last_solve_stats = {
            "status": status,
            "optimal": optimal,
            "fallback": False,
            "candidate_count": len(candidates),
            "observed_target_count": len(observation.targets),
            "candidate_target_count": len({
                item.candidate.assignment.target_id
                for item in candidates
            }),
            "selected_target_count": len({
                item.candidate.assignment.target_id
                for item in selected
            }),
            "selected_assignment_count": len(selected),
            "selected_first_shot_count": sum(
                item.role == "first"
                for item in selected
            ),
            "selected_second_shot_count": pair_count,
            "solve_time_s": time.perf_counter() - started_at,
            "mip_gap": mip_gap,
            "selected_pair_count": pair_count,
            "selected_temporal_pair_count": pair_count,
            "selected_staggered_pair_count": pair_count,
            "selected_synchronized_pair_count": 0,
            "selected_spatial_pair_count": pair_count,
            "selected_concentrated_pair_count": 0,
            "selected_layered_pair_count": pair_count,
            "selected_crossfire_pair_count": 0,
            "selected_spatiotemporal_pair_count": pair_count,
            "intercept_gap_ms_total": sum(
                item.intercept_gap_ms
                for item in selected_seconds
            ),
            "intercept_separation_m_total": sum(
                item.depth_separation_m
                for item in selected_seconds
            ),
            "approach_angle_deg_total": 0.0,
            "asset_clearance_m_total": 0.0,
            "coordination_mode": "successive_depth",
        }
