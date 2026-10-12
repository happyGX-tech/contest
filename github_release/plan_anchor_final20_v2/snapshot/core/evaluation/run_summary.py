"""Collect and persist a compact, single-run simulation summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_RED_SIDE = 0
_BLUE_SIDE = 1
_RED_PLATFORM_TYPES = {21000, 21001, 21002}


def _entity_key(entity_id: int | str) -> str:
    return str(entity_id)


def _health(entity: Mapping[str, Any] | None) -> float:
    if not entity:
        return 0.0
    return float(entity.get("health", 0.0))


class RunSummary:
    """Track only the information needed for one finished simulation round."""

    def __init__(
        self,
        *,
        scenario: str,
        policies: Mapping[str, str],
        reward_tracker=None,
    ) -> None:
        self.scenario = scenario
        self.policies = dict(policies)
        self.reward_tracker = reward_tracker
        self.initial_entities: dict[int | str, dict[str, Any]] = {}
        self.steps_executed = 0

    def start(self, observation: Mapping[str, Any]) -> None:
        entities = observation.get("entities", {})
        self.initial_entities = {
            entity_id: dict(entity)
            for entity_id, entity in entities.items()
            if isinstance(entity, Mapping)
        }

    def update(self, step: int, observation: Mapping[str, Any]) -> None:
        self.steps_executed = int(step)
        if self.reward_tracker is not None:
            self.reward_tracker.check_completion(step, observation)

    def build(
        self,
        final_observation: Mapping[str, Any],
        *,
        termination_reason: str,
        red_launched: int | None,
    ) -> dict[str, Any]:
        entities = final_observation.get("entities", {})
        final_entities = {
            entity_id: entity
            for entity_id, entity in entities.items()
            if isinstance(entity, Mapping)
        }

        red_platforms = self._side_entities(self.initial_entities, _RED_SIDE, _RED_PLATFORM_TYPES)
        blue_entities = self._side_entities(self.initial_entities, _BLUE_SIDE)
        objectives, score = self._objectives_and_score(final_entities, final_observation)

        return {
            "scenario": self.scenario,
            "policies": self.policies,
            "steps_executed": self.steps_executed,
            "termination_reason": termination_reason,
            "score": score,
            "red": {
                "platforms_total": self._initial_side_count(_RED_SIDE, _RED_PLATFORM_TYPES),
                "launched": red_launched,
                "alive": sum(_health(self._lookup(final_entities, entity_id)) > 0 for entity_id in red_platforms),
                "lost": sum(_health(self._lookup(final_entities, entity_id)) <= 0 for entity_id in red_platforms),
            },
            "blue": {
                "entities_total": self._initial_side_count(_BLUE_SIDE),
                "alive": sum(_health(self._lookup(final_entities, entity_id)) > 0 for entity_id in blue_entities),
                "destroyed": sum(_health(self._lookup(final_entities, entity_id)) <= 0 for entity_id in blue_entities),
                "initial_health_total": self._initial_side_health(_BLUE_SIDE),
                "remaining_health_total": sum(_health(self._lookup(final_entities, entity_id)) for entity_id in blue_entities),
            },
            "objectives": objectives,
        }

    def write(
        self,
        output_dir: str | Path,
        summary: Mapping[str, Any],
        filename: str = "summary.json",
    ) -> Path:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / filename
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _objectives_and_score(
        self,
        entities: Mapping[int | str, Mapping[str, Any]],
        final_observation: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        if self.reward_tracker is None:
            return [], None

        breakdown = self.reward_tracker.finish(final_observation)
        destroyed_steps = dict(breakdown.destruction_steps)
        objectives: list[dict[str, Any]] = []
        for entity_id, weight in breakdown.objective_weights:
            initial = self._lookup(self.initial_entities, entity_id)
            final = self._lookup(entities, entity_id)
            objectives.append(
                {
                    "id": int(entity_id),
                    "name": (final or initial or {}).get("nameChn", ""),
                    "type": (final or initial or {}).get("type"),
                    "weight": float(weight),
                    "initial_health": _health(initial),
                    "final_health": _health(final),
                    "destroyed_step": destroyed_steps.get(entity_id),
                }
            )
        return objectives, breakdown.to_dict()

    @staticmethod
    def _lookup(
        entities: Mapping[int | str, Mapping[str, Any]], entity_id: int,
    ) -> Mapping[str, Any] | None:
        return entities.get(entity_id, entities.get(_entity_key(entity_id)))

    @staticmethod
    def _side_entities(
        entities: Mapping[int | str, Mapping[str, Any]],
        side: int,
        types: set[int] | None = None,
    ) -> dict[int | str, Mapping[str, Any]]:
        return {
            entity_id: entity
            for entity_id, entity in entities.items()
            if int(entity.get("side", -1)) == side
            and (types is None or int(entity.get("type", -1)) in types)
        }

    def _initial_side_count(self, side: int, types: set[int] | None = None) -> int:
        return len(self._side_entities(self.initial_entities, side, types))

    def _initial_side_health(self, side: int) -> float:
        return sum(_health(entity) for entity in self._side_entities(self.initial_entities, side).values())
