# -*- coding: utf-8 -*-
from __future__ import annotations

from .config import DEFAULT_MAX_SHOTS_PER_TARGET

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
        max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET,
        planning_delay: float = 1_000.0,
        wave_gap: float = 5_000.0,
        wave_size: int = 1,
    ):
        super().__init__(max_shots_per_target=max_shots_per_target)
        self.planning_delay = planning_delay
        self.wave_gap = wave_gap
        self.wave_size = max(1, int(wave_size))

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

        # These terms are constant throughout this decision. Cache per target
        # occurrence, not ID: observations can contain distinct tracks for one ID.
        # Keep both encounter order and the original floating-point association.
        costs_by_target = []
        for target in observation.targets:
            if not available_by_id or remaining_need[target.target_id] <= 0:
                continue
            asset_cost = self._nearest_asset_distance(observation, target) * 0.05
            urgency_cost = self._target_urgency(observation, target) * 100_000.0
            pair_costs = {
                interceptor.interceptor_id: (
                    distance_m(interceptor.pos_ecf, target.pos_ecf) / 1200.0
                ) * 1000.0 + asset_cost
                for interceptor in available_by_id.values()
            }
            costs_by_target.append((target, pair_costs, urgency_cost))

        while available_by_id and any(
            need > 0 for need in remaining_need.values()
        ):
            best_pair: tuple[
                float,
                ThreatTrack,
                InterceptorState,
            ] | None = None
            for target, pair_costs, urgency_cost in costs_by_target:
                if remaining_need.get(target.target_id, 0) <= 0:
                    continue
                overkill_penalty = assigned_count[target.target_id] * 50_000.0
                for interceptor in available_by_id.values():
                    cost = (
                        pair_costs[interceptor.interceptor_id]
                        + overkill_penalty
                        - urgency_cost
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
