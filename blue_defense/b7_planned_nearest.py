# -*- coding: utf-8 -*-
from __future__ import annotations

from .config import DEFAULT_MAX_SHOTS_PER_TARGET

from .base import (
    BlueObservation,
    InterceptorAssignment,
    PlannedInterceptorAssignment,
    ScheduledBluePolicy,
    distance_m,
)


class B7PlannedNearestPolicy(ScheduledBluePolicy):
    """Nearest-interceptor allocation released after a planning delay."""

    name = "b7_planned_nearest"

    def __init__(
        self,
        max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET,
        planning_delay: float = 1_000.0,
    ):
        super().__init__(max_shots_per_target=max_shots_per_target)
        self.planning_delay = planning_delay

    def decide(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorAssignment]:
        available = self.available_interceptors_for_planning(observation)
        plans: list[PlannedInterceptorAssignment] = []
        targets = sorted(
            observation.targets,
            key=lambda item: (item.detect_time, item.target_id),
        )

        for target in targets:
            if not available:
                break
            need = min(
                self.remaining_need_with_pending(
                    observation,
                    target.target_id,
                ),
                len(available),
            )
            for _ in range(need):
                interceptor = min(
                    available,
                    key=lambda item: distance_m(
                        item.pos_ecf,
                        target.pos_ecf,
                    ),
                )
                available.remove(interceptor)
                plans.append(
                    PlannedInterceptorAssignment(
                        assignment=InterceptorAssignment(
                            interceptor_id=interceptor.interceptor_id,
                            target_id=target.target_id,
                            target_pos_ecf=target.pos_ecf,
                            target_vel_ecf=target.vel_ecf,
                        ),
                        due_time=(
                            observation.sim_time + self.planning_delay
                        ),
                    )
                )

        self.schedule_assignments(observation, plans)
        return self.release_due_assignments(observation)
