# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import BlueObservation, BluePolicy, InterceptorAssignment, distance_m


class B1NearestInterceptorPolicy(BluePolicy):
    name = "b1_nearest_interceptor"

    def decide(self, observation: BlueObservation) -> list[InterceptorAssignment]:
        assignments: list[InterceptorAssignment] = []
        available = self.available_interceptors(observation)
        targets = sorted(observation.targets, key=lambda item: (item.detect_time, item.target_id))

        for target in targets:
            if not available:
                break
            need = min(self.remaining_need(observation, target.target_id), len(available))
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
                    )
                )

        return assignments
