# -*- coding: utf-8 -*-
from __future__ import annotations

import random

from .base import BlueObservation, BluePolicy, InterceptorAssignment


class B0FixedRatioRandomPolicy(BluePolicy):
    name = "b0_fixed_ratio_random"

    def __init__(self, max_shots_per_target: int = 5, seed: int | None = None):
        super().__init__(max_shots_per_target=max_shots_per_target)
        self._rng = random.Random(seed)

    def decide(self, observation: BlueObservation) -> list[InterceptorAssignment]:
        assignments: list[InterceptorAssignment] = []
        available = self.available_interceptors(observation)

        for target in observation.targets:
            if not available:
                break
            need = min(self.remaining_need(observation, target.target_id), len(available))
            for _ in range(need):
                index = self._rng.randrange(len(available))
                interceptor = available.pop(index)
                assignments.append(
                    InterceptorAssignment(
                        interceptor_id=interceptor.interceptor_id,
                        target_id=target.target_id,
                        target_pos_ecf=target.pos_ecf,
                        target_vel_ecf=target.vel_ecf,
                    )
                )

        return assignments
