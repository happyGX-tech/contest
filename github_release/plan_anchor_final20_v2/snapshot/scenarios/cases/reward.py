"""Unified weighted-damage scoring for competition scenarios."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


# 不同蓝方根节点的任务价值：暗巢:哨卫巢:计分母巢 = 1:2:5。
OBJECTIVE_KIND_WEIGHTS = {
    "target": 5.0,
    "site": 2.0,
    "ship": 1.0,
}
DEFAULT_OBJECTIVE_WEIGHT = 1.0


@dataclass(frozen=True, slots=True)
class RewardPolicy:
    scenario_id: str
    objective_ids: tuple[int, ...]
    max_steps: int
    scenario_path: Path
    objective_weights: tuple[tuple[int, float], ...] = ()
    objective_initial_health: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    score: float
    raw_score: float
    K: float
    T: float
    completed: bool
    destroyed_ids: tuple[int, ...]
    destroyed_weight: float
    total_objective_weight: float
    objective_weights: tuple[tuple[int, float], ...]
    destruction_steps: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["destroyed_ids"] = list(self.destroyed_ids)
        value["objective_weights"] = {
            str(entity_id): weight
            for entity_id, weight in self.objective_weights
        }
        value["destruction_steps"] = {
            str(entity_id): step
            for entity_id, step in self.destruction_steps
        }
        return value


def _entity_health(
    entities: Mapping[int | str, Mapping[str, Any]],
    entity_id: int,
) -> float:
    entity = entities.get(entity_id, entities.get(str(entity_id)))
    if entity is None:
        return float("inf")
    return float(entity.get("health", float("inf")))


def load_reward_policy(scenario: str | Path) -> RewardPolicy | None:
    """Load judge inputs from the ``case_info.json`` beside a scenario.

    Non-competition scenarios do not have ``case_info.json`` and therefore
    return ``None`` instead of changing their original execution behaviour.
    """
    raw_path = str(scenario)
    if raw_path.startswith(("http://", "https://")):
        return None

    scenario_path = Path(raw_path).resolve()
    info_path = scenario_path.with_name("case_info.json")
    if not info_path.is_file():
        return None

    info = json.loads(info_path.read_text(encoding="utf-8"))
    expected_hash = str(info.get("scenario_sha256", ""))
    scenario_bytes = scenario_path.read_bytes()
    # Git may check JSON out with CRLF on Windows. Treat newline conversion as
    # transport-level normalization while still rejecting content changes.
    lf_bytes = scenario_bytes.replace(b"\r\n", b"\n")
    hash_candidates = {
        hashlib.sha256(scenario_bytes).hexdigest(),
        hashlib.sha256(lf_bytes).hexdigest(),
        hashlib.sha256(lf_bytes.replace(b"\n", b"\r\n")).hexdigest(),
    }
    if expected_hash and expected_hash not in hash_candidates:
        raise ValueError(
            f"场景校验失败：{scenario_path} 与 case_info.json 的 SHA-256 不一致"
        )

    objective_ids = tuple(int(value) for value in info["objective_ids"])
    roots_by_id = {
        int(root["id"]): root
        for root in info.get("roots", [])
        if "id" in root
    }
    objective_weights = tuple(
        (
            entity_id,
            float(
                OBJECTIVE_KIND_WEIGHTS.get(
                    str(roots_by_id.get(entity_id, {}).get("kind", "")),
                    DEFAULT_OBJECTIVE_WEIGHT,
                )
            ),
        )
        for entity_id in objective_ids
    )
    objective_initial_health = tuple(
        (
            entity_id,
            float(roots_by_id.get(entity_id, {}).get("hp", 0.0)),
        )
        for entity_id in objective_ids
    )
    max_steps = int(info["max_steps"])
    if not objective_ids:
        raise ValueError(f"{info_path} 没有计分母巢")
    if max_steps <= 0:
        raise ValueError(f"{info_path} 的 max_steps 必须大于 0")
    if any(weight <= 0 for _, weight in objective_weights):
        raise ValueError(f"{info_path} 包含非正数的目标权重")
    if any(health <= 0 for _, health in objective_initial_health):
        raise ValueError(f"{info_path} 包含非正数的目标初始血量")

    return RewardPolicy(
        scenario_id=str(info["scenario_id"]),
        objective_ids=objective_ids,
        max_steps=max_steps,
        scenario_path=scenario_path,
        objective_weights=objective_weights,
        objective_initial_health=objective_initial_health,
    )


def calculate_reward(
    *,
    policy: RewardPolicy,
    target_health: Mapping[int | str, float],
    destruction_steps: Mapping[int | str, int] | None = None,
) -> RewardBreakdown:
    """Calculate the 0--100 weighted damage ratio, independent of time.

    Each objective contributes its own fractional health loss multiplied by
    the configured kind weight.  ``K`` and ``raw_score`` both expose that
    normalized damage value. ``T`` is retained as a compatibility field and
    is always zero; destruction steps are retained only for result reporting.
    """
    destroyed_ids = tuple(
        entity_id
        for entity_id in policy.objective_ids
        if float(
            target_health.get(
                entity_id,
                target_health.get(str(entity_id), float("inf")),
            )
        )
        <= 0.0
    )
    weight_pairs = policy.objective_weights or tuple(
        (entity_id, DEFAULT_OBJECTIVE_WEIGHT)
        for entity_id in policy.objective_ids
    )
    weight_by_id = dict(weight_pairs)
    total_objective_weight = sum(weight_by_id.values())
    if total_objective_weight <= 0:
        raise ValueError("计分目标总权重必须大于 0")

    destroyed_weight = sum(weight_by_id[entity_id] for entity_id in destroyed_ids)
    initial_pairs = policy.objective_initial_health or tuple(
        (entity_id, 1.0)
        for entity_id in policy.objective_ids
    )
    initial_health_by_id = dict(initial_pairs)
    weighted_damage = 0.0
    for entity_id in policy.objective_ids:
        initial_health = float(initial_health_by_id.get(entity_id, 0.0))
        if initial_health <= 0:
            raise ValueError(f"计分目标 {entity_id} 的初始血量必须大于 0")
        final_health = float(
            target_health.get(
                entity_id,
                target_health.get(str(entity_id), initial_health),
            )
        )
        damage_fraction = max(0.0, min(1.0, (initial_health - final_health) / initial_health))
        weighted_damage += weight_by_id[entity_id] * damage_fraction
    k_value = weighted_damage / total_objective_weight
    completed = len(destroyed_ids) == len(policy.objective_ids)

    supplied_steps = destruction_steps or {}
    normalized_steps: dict[int, int] = {}
    for entity_id in destroyed_ids:
        raw_step = supplied_steps.get(
            entity_id,
            supplied_steps.get(str(entity_id)),
        )
        if raw_step is None:
            raw_step = policy.max_steps
        step = max(0, min(policy.max_steps, int(raw_step)))
        normalized_steps[entity_id] = step

    t_value = 0.0
    raw_score = max(0.0, min(1.0, k_value))
    return RewardBreakdown(
        score=100.0 * max(0.0, min(1.0, raw_score)),
        raw_score=raw_score,
        K=k_value,
        T=t_value,
        completed=completed,
        destroyed_ids=destroyed_ids,
        destroyed_weight=destroyed_weight,
        total_objective_weight=total_objective_weight,
        objective_weights=tuple(weight_pairs),
        destruction_steps=tuple(sorted(normalized_steps.items())),
    )


class RewardTracker:
    """Accumulate one round of state without changing the simulator."""

    def __init__(self, policy: RewardPolicy) -> None:
        self.policy = policy
        self.destruction_steps: dict[int, int] = {}

    def check_completion(
        self,
        step: int,
        observation: Mapping[str, Any],
    ) -> None:
        """Record each objective's first destroyed step.

        Repeated observations do not overwrite the first destroyed step.
        """
        entities = observation.get("entities", {})
        for entity_id in self.policy.objective_ids:
            if (
                entity_id not in self.destruction_steps
                and _entity_health(entities, entity_id) <= 0
            ):
                self.destruction_steps[entity_id] = int(step)

    def finish(self, observation: Mapping[str, Any]) -> RewardBreakdown:
        entities = observation.get("entities", {})
        target_health = {
            entity_id: _entity_health(entities, entity_id)
            for entity_id in self.policy.objective_ids
        }
        return calculate_reward(
            policy=self.policy,
            target_health=target_health,
            destruction_steps=self.destruction_steps,
        )
