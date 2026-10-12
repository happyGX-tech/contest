# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import (
    BlueObservation,
    BluePolicy,
    DefendedAsset,
    InterceptorAssignment,
    InterceptorState,
    PlannedInterceptorAssignment,
    ScheduledBluePolicy,
    ScheduledInterceptorAssignment,
    ThreatTrack,
    Vector3,
)
from .b0_fixed_ratio_random import B0FixedRatioRandomPolicy
from .b1_nearest_interceptor import B1NearestInterceptorPolicy
from .b2_threat_priority import B2ThreatPriorityPolicy
from .b3_min_cost_assignment import B3MinCostAssignmentPolicy
from .b4_joint_timing_assignment import B4JointTimingAssignmentPolicy
from .b5_successive_depth_coordination import (
    B5SuccessiveDepthCoordinationPolicy,
)
from .b6_mixed_fire_coordination import (
    B6MixedFireCoordinationPolicy,
)
from .b7_planned_nearest import B7PlannedNearestPolicy
from .b8_threat_multi_wave import B8ThreatMultiWavePolicy
from .b9_min_cost_planned_multi_wave import (
    B9MinCostPlannedMultiWavePolicy,
)


_POLICIES = {
    B0FixedRatioRandomPolicy.name: B0FixedRatioRandomPolicy,
    B1NearestInterceptorPolicy.name: B1NearestInterceptorPolicy,
    B2ThreatPriorityPolicy.name: B2ThreatPriorityPolicy,
    B3MinCostAssignmentPolicy.name: B3MinCostAssignmentPolicy,
    B4JointTimingAssignmentPolicy.name: B4JointTimingAssignmentPolicy,
    B5SuccessiveDepthCoordinationPolicy.name: B5SuccessiveDepthCoordinationPolicy,
    B6MixedFireCoordinationPolicy.name: B6MixedFireCoordinationPolicy,
    B7PlannedNearestPolicy.name: B7PlannedNearestPolicy,
    B8ThreatMultiWavePolicy.name: B8ThreatMultiWavePolicy,
    B9MinCostPlannedMultiWavePolicy.name: B9MinCostPlannedMultiWavePolicy,
}


def build_blue_policy(
    name: str | None = None,
    max_shots_per_target: int = 5,
    seed: int | None = None,
) -> BluePolicy:
    policy_name = (name or B0FixedRatioRandomPolicy.name).strip().lower().replace("-", "_")
    policy_cls = _POLICIES.get(policy_name)
    if policy_cls is None:
        valid_names = ", ".join(sorted(_POLICIES))
        raise ValueError(f"Unknown blue policy '{name}'. Valid policies: {valid_names}")

    if policy_cls is B0FixedRatioRandomPolicy:
        return policy_cls(max_shots_per_target=max_shots_per_target, seed=seed)
    return policy_cls(max_shots_per_target=max_shots_per_target)


__all__ = [
    "BlueObservation",
    "BluePolicy",
    "DefendedAsset",
    "B0FixedRatioRandomPolicy",
    "InterceptorAssignment",
    "InterceptorState",
    "B1NearestInterceptorPolicy",
    "B2ThreatPriorityPolicy",
    "B3MinCostAssignmentPolicy",
    "B4JointTimingAssignmentPolicy",
    "B5SuccessiveDepthCoordinationPolicy",
    "B6MixedFireCoordinationPolicy",
    "B7PlannedNearestPolicy",
    "B8ThreatMultiWavePolicy",
    "B9MinCostPlannedMultiWavePolicy",
    "PlannedInterceptorAssignment",
    "ScheduledBluePolicy",
    "ScheduledInterceptorAssignment",
    "ThreatTrack",
    "Vector3",
    "build_blue_policy",
]
