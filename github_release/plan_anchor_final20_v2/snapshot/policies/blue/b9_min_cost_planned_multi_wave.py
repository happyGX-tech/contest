# -*- coding: utf-8 -*-
from __future__ import annotations

import os

from .base import (
    BlueObservation,
    InterceptorAssignment,
    InterceptorState,
    PlannedInterceptorAssignment,
    ScheduledBluePolicy,
    ThreatTrack,
    distance_m,
    speed_mps,
)


class B9MinCostPlannedMultiWavePolicy(ScheduledBluePolicy):
    """Min-cost allocation with a planning delay and wave releases."""

    name = "b9_min_cost_planned_multi_wave"

    def __init__(
        self,
        max_shots_per_target: int = 5,
        planning_delay: float | None = None,
        wave_gap: float | None = None,
        wave_size: int | None = None,
    ):
        super().__init__(max_shots_per_target=max_shots_per_target)
        self.planning_delay = (
            _read_float("BLUE_PLAN_DELAY", 5_000.0)
            if planning_delay is None
            else planning_delay
        )
        self.wave_gap = (
            _read_float("BLUE_WAVE_GAP", 5_000.0)
            if wave_gap is None
            else wave_gap
        )
        self.wave_size = max(
            1,
            _read_int("BLUE_WAVE_SIZE", 1)
            if wave_size is None
            else int(wave_size),
        )

    def decide(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorAssignment]:
        available_by_id = {
            item.interceptor_id: item
            for item in self.available_interceptors_for_planning(observation)
        }
        remaining_need = {
            target.target_id: self.remaining_need_with_pending(
                observation,
                target.target_id,
            )
            for target in observation.targets
        }
        assigned_count = {
            target.target_id: 0
            for target in observation.targets
        }
        plans: list[PlannedInterceptorAssignment] = []

        while available_by_id and any(
            need > 0 for need in remaining_need.values()
        ):
            best_pair: tuple[
                float,
                ThreatTrack,
                InterceptorState,
            ] | None = None
            for target in observation.targets:
                if remaining_need.get(target.target_id, 0) <= 0:
                    continue
                for interceptor in available_by_id.values():
                    cost = self._assignment_cost(
                        observation,
                        target,
                        interceptor,
                        assigned_count[target.target_id],
                    )
                    if best_pair is None or cost < best_pair[0]:
                        best_pair = (cost, target, interceptor)

            if best_pair is None:
                break

            cost, target, interceptor = best_pair
            available_by_id.pop(interceptor.interceptor_id, None)
            remaining_need[target.target_id] -= 1
            wave_index = (
                assigned_count[target.target_id] // self.wave_size
            )
            assigned_count[target.target_id] += 1
            plans.append(
                PlannedInterceptorAssignment(
                    assignment=InterceptorAssignment(
                        interceptor_id=interceptor.interceptor_id,
                        target_id=target.target_id,
                        target_pos_ecf=target.pos_ecf,
                        target_vel_ecf=target.vel_ecf,
                        priority=-cost,
                    ),
                    due_time=(
                        observation.sim_time
                        + self.planning_delay
                        + wave_index * self.wave_gap
                    ),
                    wave_index=wave_index,
                )
            )

        self.schedule_assignments(observation, plans)
        return self.release_due_assignments(observation)

    def _assignment_cost(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
        interceptor: InterceptorState,
        already_assigned: int,
    ) -> float:
        intercept_distance = distance_m(
            interceptor.pos_ecf,
            target.pos_ecf,
        )
        intercept_time = intercept_distance / 1200.0
        asset_distance = self._nearest_asset_distance(
            observation,
            target,
        )
        target_urgency = self._target_urgency(observation, target)
        overkill_penalty = already_assigned * 50_000.0
        return (
            intercept_time * 1000.0
            + asset_distance * 0.05
            + overkill_penalty
            - target_urgency * 100_000.0
        )

    @staticmethod
    def _nearest_asset_distance(
        observation: BlueObservation,
        target: ThreatTrack,
    ) -> float:
        if not observation.assets:
            return 0.0
        return min(
            distance_m(asset.pos_ecf, target.pos_ecf)
            for asset in observation.assets
        )

    @staticmethod
    def _target_urgency(
        observation: BlueObservation,
        target: ThreatTrack,
    ) -> float:
        if not observation.assets:
            return 1.0
        nearest_asset = min(
            observation.assets,
            key=lambda asset: distance_m(
                asset.pos_ecf,
                target.pos_ecf,
            ),
        )
        eta = distance_m(
            nearest_asset.pos_ecf,
            target.pos_ecf,
        ) / speed_mps(target.vel_ecf, default=300.0)
        return nearest_asset.value / max(eta, 1.0)


def _read_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _read_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default
