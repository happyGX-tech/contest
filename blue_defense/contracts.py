# -*- coding: utf-8 -*-
"""Lightweight boundary types shared with the simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .base import (
    BlueObservation,
    InterceptorAssignment,
    ScheduledInterceptorAssignment,
)


@dataclass(frozen=True)
class BlueDecision:
    """One completed controller evaluation.

    ``None`` from ``decide`` means the event was throttled and must not replace
    existing plans.  An empty ``assignments`` tuple means a real evaluation
    completed but selected no launch.
    """

    assignments: tuple[
        InterceptorAssignment | ScheduledInterceptorAssignment,
        ...,
    ] = ()
    solve_stats: dict[str, float | int | str | bool] = field(default_factory=dict)


@runtime_checkable
class BlueDefenseControllerProtocol(Protocol):
    name: str
    plans_launch_time: bool
    preserves_pending_launches: bool

    def decide(self, observation: BlueObservation) -> BlueDecision | None:
        """评估信息库快照；尚未到达决策时间时返回None。"""

    def reset(self) -> None:
        """Reset all episode-local decisions and pending plans."""

    def asset_value(self, entity_id: int, entity_type: int) -> float:
        """Return the controller's configured relative protection value."""
