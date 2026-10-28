# -*- coding: utf-8 -*-
"""Lazy registry for the built-in blue baselines."""

from __future__ import annotations

from importlib import import_module

from .base import BluePolicy
from .config import DEFAULT_MAX_SHOTS_PER_TARGET, BlueDefenseConfig


_POLICIES: dict[str, tuple[str, str]] = {
    "b0_fixed_ratio_random": ("b0_fixed_ratio_random", "B0FixedRatioRandomPolicy"),
    "b1_nearest_interceptor": ("b1_nearest_interceptor", "B1NearestInterceptorPolicy"),
    "b2_threat_priority": ("b2_threat_priority", "B2ThreatPriorityPolicy"),
    "b3_min_cost_assignment": ("b3_min_cost_assignment", "B3MinCostAssignmentPolicy"),
    "b4_joint_timing_assignment": ("b4_joint_timing_assignment", "B4JointTimingAssignmentPolicy"),
    "b5_successive_depth_coordination": (
        "b5_successive_depth_coordination",
        "B5SuccessiveDepthCoordinationPolicy",
    ),
    "b6_mixed_fire_coordination": (
        "b6_mixed_fire_coordination",
        "B6MixedFireCoordinationPolicy",
    ),
    "b7_planned_nearest": ("b7_planned_nearest", "B7PlannedNearestPolicy"),
    "b8_threat_multi_wave": ("b8_threat_multi_wave", "B8ThreatMultiWavePolicy"),
    "b9_min_cost_planned_multi_wave": (
        "b9_min_cost_planned_multi_wave",
        "B9MinCostPlannedMultiWavePolicy",
    ),
}

BLUE_POLICY_CHOICES = tuple(_POLICIES)


def blue_policy_names() -> tuple[str, ...]:
    return tuple(sorted(_POLICIES))


def build_blue_policy(
    config: BlueDefenseConfig | str | None = None,
    max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET,
    seed: int | None = None,
    *,
    name: str | None = None,
) -> BluePolicy:
    """Instantiate one baseline without importing unrelated solver modules."""

    if not isinstance(config, BlueDefenseConfig):
        if name is not None and config is not None:
            raise TypeError("pass either config/policy name or name=, not both")
        config = BlueDefenseConfig(
            policy_name=name or config or "b0_fixed_ratio_random",
            max_shots_per_target=max_shots_per_target,
            seed=seed,
        )

    policy_spec = _POLICIES.get(config.policy_name)
    if policy_spec is None:
        valid_names = ", ".join(blue_policy_names())
        raise ValueError(
            f"Unknown blue policy '{config.policy_name}'. Valid policies: {valid_names}"
        )

    module_name, class_name = policy_spec
    module = import_module(f"{__package__}.{module_name}")
    policy_cls = getattr(module, class_name)
    kwargs: dict[str, object] = {
        "max_shots_per_target": config.max_shots_per_target,
    }
    if config.policy_name == "b0_fixed_ratio_random":
        kwargs["seed"] = config.seed
    elif config.policy_name == "b7_planned_nearest":
        kwargs["planning_delay"] = config.planning_delay_ms
    elif config.policy_name == "b8_threat_multi_wave":
        kwargs.update(
            wave_gap=config.wave_gap_ms,
            wave_size=config.wave_size,
        )
    elif config.policy_name == "b9_min_cost_planned_multi_wave":
        kwargs.update(
            planning_delay=config.planning_delay_ms,
            wave_gap=config.wave_gap_ms,
            wave_size=config.wave_size,
        )
    return policy_cls(**kwargs)
