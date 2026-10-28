# -*- coding: utf-8 -*-
"""Built-in blue defense controller and dependency-light public contracts."""

from __future__ import annotations

from importlib import import_module

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
from .config import BlueDefenseConfig
from .contracts import BlueDecision, BlueDefenseControllerProtocol
from .controller import BlueDefenseController
from .information import BlueInformationStore
from .registry import BLUE_POLICY_CHOICES, blue_policy_names, build_blue_policy


_LAZY_POLICY_EXPORTS = {
    "B0FixedRatioRandomPolicy": "b0_fixed_ratio_random",
    "B1NearestInterceptorPolicy": "b1_nearest_interceptor",
    "B2ThreatPriorityPolicy": "b2_threat_priority",
    "B3MinCostAssignmentPolicy": "b3_min_cost_assignment",
    "B4JointTimingAssignmentPolicy": "b4_joint_timing_assignment",
    "B5SuccessiveDepthCoordinationPolicy": "b5_successive_depth_coordination",
    "B6MixedFireCoordinationPolicy": "b6_mixed_fire_coordination",
    "B7PlannedNearestPolicy": "b7_planned_nearest",
    "B8ThreatMultiWavePolicy": "b8_threat_multi_wave",
    "B9MinCostPlannedMultiWavePolicy": "b9_min_cost_planned_multi_wave",
}


def __getattr__(name: str):
    module_name = _LAZY_POLICY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


__all__ = [
    "BlueDefenseConfig",
    "BlueDefenseController",
    "BlueDefenseControllerProtocol",
    "BlueDecision",
    "BlueInformationStore",
    "BlueObservation",
    "BLUE_POLICY_CHOICES",
    "BluePolicy",
    "DefendedAsset",
    "InterceptorAssignment",
    "InterceptorState",
    "PlannedInterceptorAssignment",
    "ScheduledBluePolicy",
    "ScheduledInterceptorAssignment",
    "ThreatTrack",
    "Vector3",
    "blue_policy_names",
    "build_blue_policy",
    *_LAZY_POLICY_EXPORTS,
]
