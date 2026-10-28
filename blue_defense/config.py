# -*- coding: utf-8 -*-
"""Explicit configuration for the built-in blue defense controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


DEFAULT_MAX_SHOTS_PER_TARGET = 2


@dataclass(frozen=True)
class BlueDefenseConfig:
    """Explicit settings for built-in blue defense.

    Time values are expressed in simulation milliseconds. B0's random stream
    remains continuous across episode resets unless ``reset_rng_on_episode``
    is enabled. Configuration is supplied by the caller, without environment
    variable overrides.
    """

    policy_name: str = "b0_fixed_ratio_random"
    max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET
    seed: int | None = None
    min_cost_decision_interval_ms: float = 5.0
    joint_decision_interval_ms: float = 5_000.0
    planning_delay_ms: float = 1_000.0
    wave_gap_ms: float | None = None
    wave_size: int = 1
    reset_rng_on_episode: bool = False
    asset_values: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = self.policy_name.strip().lower().replace("-", "_")
        object.__setattr__(self, "policy_name", normalized)
        if self.wave_gap_ms is None:
            wave_gap = 10_000.0 if normalized == "b8_threat_multi_wave" else 5_000.0
            object.__setattr__(self, "wave_gap_ms", wave_gap)
        object.__setattr__(
            self,
            "max_shots_per_target",
            max(1, int(self.max_shots_per_target)),
        )
        object.__setattr__(self, "wave_size", max(1, int(self.wave_size)))
        object.__setattr__(
            self,
            "asset_values",
            {int(key): float(value) for key, value in self.asset_values.items()},
        )
