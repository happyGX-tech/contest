# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import (
    BlueObservation,
    BluePolicy,
    InterceptorAssignment,
    ThreatTrack,
    distance_m,
    speed_mps,
)


class B2ThreatPriorityPolicy(BluePolicy):
    name = "b2_threat_priority"

    def decide(self, observation: BlueObservation) -> list[InterceptorAssignment]:
        assignments: list[InterceptorAssignment] = []
        available = self.available_interceptors(observation)
        targets = sorted(
            observation.targets,
            key=lambda target: self._threat_score(observation, target),
            reverse=True,
        )

        for target in targets:
            if not available:
                break
            need = min(self._desired_shots(target), self.remaining_need(observation, target.target_id), len(available))
            priority = self._threat_score(observation, target)
            for _ in range(need):
                interceptor = min(
                    available,
                    key=lambda item: distance_m(item.pos_ecf, target.pos_ecf),
                )
                available.remove(interceptor)
                assignments.append(
                    InterceptorAssignment(
                        interceptor_id=interceptor.interceptor_id,
                        target_id=target.target_id,
                        target_pos_ecf=target.pos_ecf,
                        target_vel_ecf=target.vel_ecf,
                        priority=priority,
                    )
                )

        return assignments

    def _desired_shots(self, target: ThreatTrack) -> int:
        if target.type == 21000:
            return min(self.max_shots_per_target, 3)
        if target.type == 21001:
            return min(self.max_shots_per_target, 2)
        return 1

    def _threat_score(self, observation: BlueObservation, target: ThreatTrack) -> float:
        if not observation.assets:
            return self._type_weight(target)

        nearest_asset = min(
            observation.assets,
            key=lambda asset: distance_m(asset.pos_ecf, target.pos_ecf),
        )
        distance_to_asset = distance_m(nearest_asset.pos_ecf, target.pos_ecf)
        time_to_asset = distance_to_asset / speed_mps(target.vel_ecf, default=300.0)
        asset_term = nearest_asset.value / max(time_to_asset, 1.0)
        return self._type_weight(target) + asset_term * 1000.0

    @staticmethod
    def _type_weight(target: ThreatTrack) -> float:
        if target.type == 21000:
            return 30.0
        if target.type == 21001:
            return 20.0
        if target.type == 21002:
            return 10.0
        return 1.0
