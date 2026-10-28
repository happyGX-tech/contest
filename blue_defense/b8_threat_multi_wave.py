# -*- coding: utf-8 -*-
from __future__ import annotations

from .config import DEFAULT_MAX_SHOTS_PER_TARGET

from .base import (
    BlueObservation,
    InterceptorAssignment,
    PlannedInterceptorAssignment,
    ScheduledBluePolicy,
    ThreatTrack,
    distance_m,
    speed_mps,
)


class B8ThreatMultiWavePolicy(ScheduledBluePolicy):
    """Threat-priority allocation released in successive waves."""

    name = "b8_threat_multi_wave"

    def __init__(
        self,
        max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET,
        wave_gap: float = 10_000.0,
        wave_size: int = 1,
    ):
        super().__init__(max_shots_per_target=max_shots_per_target)
        self.wave_gap = wave_gap
        self.wave_size = max(1, int(wave_size))

    def decide(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorAssignment]:
        available = self.available_interceptors_for_planning(observation)
        plans: list[PlannedInterceptorAssignment] = []
        targets = sorted(
            observation.targets,
            key=lambda target: self._threat_score(observation, target),
            reverse=True,
        )

        for target in targets:
            if not available:
                break
            need = min(
                self._desired_shots(target),
                self.remaining_need_with_pending(
                    observation,
                    target.target_id,
                ),
                len(available),
            )
            priority = self._threat_score(observation, target)
            for index in range(need):
                interceptor = min(
                    available,
                    key=lambda item: distance_m(
                        item.pos_ecf,
                        target.pos_ecf,
                    ),
                )
                available.remove(interceptor)
                wave_index = index // self.wave_size
                plans.append(
                    PlannedInterceptorAssignment(
                        assignment=InterceptorAssignment(
                            interceptor_id=interceptor.interceptor_id,
                            target_id=target.target_id,
                            target_pos_ecf=target.pos_ecf,
                            target_vel_ecf=target.vel_ecf,
                            priority=priority,
                        ),
                        due_time=(
                            observation.sim_time
                            + wave_index * self.wave_gap
                        ),
                        wave_index=wave_index,
                    )
                )

        self.schedule_assignments(observation, plans)
        return self.release_due_assignments(observation)

    def _desired_shots(self, target: ThreatTrack) -> int:
        if target.type == 21000:
            return min(self.max_shots_per_target, 3)
        if target.type == 21001:
            return min(self.max_shots_per_target, 2)
        return 1

    def _threat_score(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
    ) -> float:
        if not observation.assets:
            return self._type_weight(target)

        nearest_asset = min(
            observation.assets,
            key=lambda asset: distance_m(
                asset.pos_ecf,
                target.pos_ecf,
            ),
        )
        distance_to_asset = distance_m(
            nearest_asset.pos_ecf,
            target.pos_ecf,
        )
        time_to_asset = distance_to_asset / speed_mps(
            target.vel_ecf,
            default=300.0,
        )
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
