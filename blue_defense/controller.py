# -*- coding: utf-8 -*-
"""Built-in central controller for blue defense baselines B0--B9."""

from __future__ import annotations

from .base import BlueObservation
from .config import BlueDefenseConfig
from .contracts import BlueDecision
from .registry import build_blue_policy


class BlueDefenseController:
    """管理策略配置、策略状态和信息库快照的决策间隔。"""

    def __init__(self, config: BlueDefenseConfig | None = None):
        self.config = config or BlueDefenseConfig()
        self._policy = build_blue_policy(self.config)
        self._last_decision_time = float("-inf")

    @property
    def name(self) -> str:
        return self._policy.name

    @property
    def plans_launch_time(self) -> bool:
        return bool(self._policy.plans_launch_time)

    @property
    def preserves_pending_launches(self) -> bool:
        return bool(self._policy.preserves_pending_launches)

    def decide(self, observation: BlueObservation) -> BlueDecision | None:
        """在决策间隔允许时评估当前信息库快照。"""

        interval_ms = self._decision_interval_ms()
        if observation.sim_time - self._last_decision_time < interval_ms:
            return None

        self._last_decision_time = observation.sim_time
        assignments = tuple(self._policy.decide(observation))
        stats = dict(getattr(self._policy, "last_solve_stats", {}))
        return BlueDecision(assignments=assignments, solve_stats=stats)

    def reset(self) -> None:
        """Clear episode plans; optionally restart B0's seeded random stream."""

        self._last_decision_time = float("-inf")
        if self.config.reset_rng_on_episode and self.name == "b0_fixed_ratio_random":
            self._policy = build_blue_policy(self.config)
            return
        self._policy.reset()

    def asset_value(self, entity_id: int, entity_type: int) -> float:
        if entity_id in self.config.asset_values:
            return float(self.config.asset_values[entity_id])
        return {
            9400: 10.0,
            9600: 6.0,
            9500: 3.0,
        }.get(entity_type, 1.0)

    def _decision_interval_ms(self) -> float:
        if self.name == "b3_min_cost_assignment":
            return self.config.min_cost_decision_interval_ms
        if self.plans_launch_time:
            return self.config.joint_decision_interval_ms
        return 0.0
