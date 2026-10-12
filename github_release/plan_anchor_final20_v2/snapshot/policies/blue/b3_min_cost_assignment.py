# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import (
    BlueObservation,
    BluePolicy,
    InterceptorAssignment,
    InterceptorState,
    ThreatTrack,
    distance_m,
    speed_mps,
)


class B3MinCostAssignmentPolicy(BluePolicy):
    name = "b3_min_cost_assignment"

    def decide(self, observation: BlueObservation) -> list[InterceptorAssignment]:
        assignments: list[InterceptorAssignment] = []
        available_by_id = {
            item.interceptor_id: item
            for item in self.available_interceptors(observation)
        }
        remaining_need = {
            target.target_id: self.remaining_need(observation, target.target_id)
            for target in observation.targets
        }
        assigned_count = {target.target_id: 0 for target in observation.targets}

        while available_by_id and any(need > 0 for need in remaining_need.values()):
            best_pair: tuple[float, ThreatTrack, InterceptorState] | None = None
            for target in observation.targets:
                if remaining_need.get(target.target_id, 0) <= 0:
                    continue
                for interceptor in available_by_id.values():
                    cost = self._assignment_cost(observation, target, interceptor, assigned_count[target.target_id])
                    if best_pair is None or cost < best_pair[0]:
                        best_pair = (cost, target, interceptor)

            if best_pair is None:
                break

            cost, target, interceptor = best_pair
            available_by_id.pop(interceptor.interceptor_id, None)
            remaining_need[target.target_id] -= 1
            assigned_count[target.target_id] += 1
            assignments.append(
                InterceptorAssignment(
                    interceptor_id=interceptor.interceptor_id,
                    target_id=target.target_id,
                    target_pos_ecf=target.pos_ecf,
                    target_vel_ecf=target.vel_ecf,
                    priority=-cost,
                )
            )

        return assignments

    def _assignment_cost(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
        interceptor: InterceptorState,
        already_assigned: int,
    ) -> float:
        intercept_distance = distance_m(interceptor.pos_ecf, target.pos_ecf)
        intercept_time = intercept_distance / 1200.0
        asset_distance = self._nearest_asset_distance(observation, target)
        target_urgency = self._target_urgency(observation, target)
        overkill_penalty = already_assigned * 50000.0
        return intercept_time * 1000.0 + asset_distance * 0.05 + overkill_penalty - target_urgency * 100000.0

    @staticmethod
    def _nearest_asset_distance(observation: BlueObservation, target: ThreatTrack) -> float:
        if not observation.assets:
            return 0.0
        return min(distance_m(asset.pos_ecf, target.pos_ecf) for asset in observation.assets)

    @staticmethod
    def _target_urgency(observation: BlueObservation, target: ThreatTrack) -> float:
        if not observation.assets:
            return 1.0
        nearest_asset = min(observation.assets, key=lambda asset: distance_m(asset.pos_ecf, target.pos_ecf))
        eta = distance_m(nearest_asset.pos_ecf, target.pos_ecf) / speed_mps(target.vel_ecf, default=300.0)
        return nearest_asset.value / max(eta, 1.0)
