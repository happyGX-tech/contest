"""Public mother-nest and sentinel-nest positions available before deployment."""

from __future__ import annotations

from .contracts import Position
from .baselines import TargetPrior
from .objectives import objective_value


def initial_targets_from_observation(initial_observation: dict) -> tuple[TargetPrior, ...]:
    """Read public objective types; hidden nests require a received detection."""

    targets: list[TargetPrior] = []
    for raw_id, entity in initial_observation.get("entities", {}).items():
        location = entity.get("position", {})
        entity_type = int(entity["type"])
        if entity_type not in (9400, 9600):
            continue
        targets.append(
            TargetPrior(
                entity_id=int(raw_id),
                entity_type=entity_type,
                position=Position(
                    lon=float(location.get("lon", 0.0)),
                    lat=float(location.get("lat", 0.0)),
                    alt=float(location.get("alt", 0.0)),
                ),
                value=objective_value(entity_type),
            )
        )
    return tuple(sorted(targets, key=lambda item: item.entity_id))
