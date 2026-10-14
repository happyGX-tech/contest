"""Legal red-side track fusion from isolated ``detectInfo`` observations."""

from __future__ import annotations

from .baselines import TargetPrior
from .contracts import Position
from .objectives import objective_value


class InitialCatalogueTrackFusion:
    """Refine public targets and admit legally detected hidden unmanned ships.

    ``DetectInfo`` currently carries an ID, type and kinematics, but no target
    health.  This component therefore updates location only; it never infers
    hidden damage or silently removes a target. A hidden 9500 target is added
    only after it appears in an isolated platform observation.
    """

    def __init__(self, targets: tuple[TargetPrior, ...]) -> None:
        self._targets = {item.entity_id: item for item in targets}

    @property
    def targets(self) -> tuple[TargetPrior, ...]:
        return tuple(sorted(self._targets.values(), key=lambda item: item.entity_id))

    def ingest(self, observation: dict) -> bool:
        """Fuse one platform's received tracks and return whether a track moved."""

        changed = False
        for raw_id, track in (observation.get("self", {}).get("detectInfo") or {}).items():
            entity_id = int(self._field(track, "entity_id", raw_id))
            current = self._targets.get(entity_id)
            if current is None:
                entity_type = int(self._field(track, "entity_type", -1))
                lla = self._field(track, "lla", None)
                if entity_type != 9500 or lla is None:
                    continue
                self._targets[entity_id] = TargetPrior(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    position=Position(
                        lon=float(self._field(lla, "x", 0.0)),
                        lat=float(self._field(lla, "y", 0.0)),
                        alt=float(self._field(lla, "z", 0.0)),
                    ),
                    value=objective_value(entity_type),
                )
                changed = True
                continue
            entity_type = int(self._field(track, "entity_type", current.entity_type))
            if entity_type != current.entity_type:
                continue
            lla = self._field(track, "lla", None)
            if lla is None:
                continue
            position = Position(
                lon=float(self._field(lla, "x", current.position.lon)),
                lat=float(self._field(lla, "y", current.position.lat)),
                alt=float(self._field(lla, "z", current.position.alt)),
            )
            if position == current.position:
                continue
            self._targets[entity_id] = TargetPrior(
                entity_id=current.entity_id,
                entity_type=current.entity_type,
                position=position,
                value=current.value,
                alive=current.alive,
            )
            changed = True
        return changed

    @staticmethod
    def _field(value, name: str, default):
        return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)
