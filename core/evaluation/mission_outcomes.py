"""Passive full-simulator outcome accounting for red missile evaluations."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any, Iterable, Mapping


RED_PLATFORM_TYPES = {21000, 21001, 21002}


def _lookup(entities: Mapping[Any, Any], entity_id: int) -> Mapping[str, Any]:
    value = entities.get(entity_id, entities.get(str(entity_id), {}))
    return value if isinstance(value, Mapping) else {}


def _health(entity: Mapping[str, Any]) -> float:
    return float(entity.get("health", 0.0))


def _position(entity: Mapping[str, Any]) -> tuple[float, float, float] | None:
    value = entity.get("pos_ecf")
    if not isinstance(value, Mapping):
        return None
    return (float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0)))


def _distance(left: tuple[float, float, float] | None, right: tuple[float, float, float] | None) -> float:
    if left is None or right is None:
        return float("inf")
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


class MissionOutcomeTracker:
    """Classify launches, blue engagements, arrivals, and effective hits.

    The engine resolves target contact at 100 m.  Observations are sampled once
    per one-second step, so the tracker uses a 500 m arrival tolerance to cover
    the final 100-200 m travelled between adjacent observations.
    """

    def __init__(self, agents: Iterable[Any], simulator_factory: Any, *, arrival_tolerance_m: float = 500.0) -> None:
        self._agents = {
            int(agent.entity_id): agent
            for agent in agents
            if int(getattr(agent, "entity_id", -1)) >= 0
        }
        self._simulator_factory = simulator_factory
        self.arrival_tolerance_m = float(arrival_tolerance_m)
        self._previous_entities: dict[Any, Any] = {}
        self._launched: set[int] = set()
        self._engaged: set[int] = set()
        self._resolved: set[int] = set()
        self._arrived: set[int] = set()
        self._effective_hits: set[int] = set()
        self._target_by_platform: dict[int, int] = {}
        self._minimum_target_distance: dict[int, float] = defaultdict(lambda: float("inf"))

    def start(self, observation: Mapping[str, Any]) -> None:
        self._previous_entities = dict(observation.get("entities", {}))

    def _sync_assignments(self) -> None:
        for platform_id, agent in self._agents.items():
            if int(getattr(agent, "launch_step", -1)) >= 0:
                self._launched.add(platform_id)
            commander = getattr(agent, "commander", None)
            target_id_for = getattr(commander, "target_id_for", None)
            if target_id_for is None:
                continue
            target_id = target_id_for(platform_id)
            if target_id is not None:
                self._target_by_platform[platform_id] = int(target_id)

        for simulator in self._simulator_factory.get_all_simulators():
            launched_list = getattr(simulator, "launched_list", None)
            if not isinstance(launched_list, Mapping):
                continue
            self._engaged.update(int(target_id) for target_id, values in launched_list.items() if values)

    def update(self, observation: Mapping[str, Any]) -> None:
        self._sync_assignments()
        entities = observation.get("entities", {})
        arrivals_by_target: dict[int, list[int]] = defaultdict(list)

        for platform_id in self._launched:
            target_id = self._target_by_platform.get(platform_id)
            if target_id is None:
                continue
            current = _lookup(entities, platform_id)
            target = _lookup(entities, target_id)
            distance = _distance(_position(current), _position(target))
            self._minimum_target_distance[platform_id] = min(
                self._minimum_target_distance[platform_id], distance
            )
            previous = _lookup(self._previous_entities, platform_id)
            if (
                platform_id not in self._resolved
                and _health(previous) > 0.0
                and _health(current) <= 0.0
            ):
                self._resolved.add(platform_id)
                if self._minimum_target_distance[platform_id] <= self.arrival_tolerance_m:
                    self._arrived.add(platform_id)
                    arrivals_by_target[target_id].append(platform_id)

        for target_id, platform_ids in arrivals_by_target.items():
            previous_health = _health(_lookup(self._previous_entities, target_id))
            current_health = _health(_lookup(entities, target_id))
            if current_health < previous_health - 1e-9:
                self._effective_hits.update(platform_ids)

        self._previous_entities = dict(entities)

    def finish(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        self._sync_assignments()
        entities = observation.get("entities", {})
        launched = len(self._launched)
        arrivals = len(self._arrived)
        effective_hits = len(self._effective_hits)
        engaged_resolved = self._engaged & self._resolved
        engaged_arrivals = engaged_resolved & self._arrived
        intercepted = engaged_resolved - self._arrived
        engaged_censored = self._engaged - self._resolved
        alive = sum(_health(_lookup(entities, platform_id)) > 0.0 for platform_id in self._launched)

        by_type: dict[str, dict[str, int | float]] = {}
        for entity_type in sorted(RED_PLATFORM_TYPES):
            members = {
                platform_id
                for platform_id in self._launched
                if int(_lookup(entities, platform_id).get("type", -1)) == entity_type
            }
            type_arrivals = members & self._arrived
            type_hits = members & self._effective_hits
            by_type[str(entity_type)] = {
                "launched": len(members),
                "arrived": len(type_arrivals),
                "arrival_rate": len(type_arrivals) / max(len(members), 1),
                "effective_hits": len(type_hits),
                "hit_rate_per_launch": len(type_hits) / max(len(members), 1),
            }

        per_platform = {
            str(platform_id): {
                "target_id": self._target_by_platform.get(platform_id),
                "engaged": platform_id in self._engaged,
                "resolved": platform_id in self._resolved,
                "arrived": platform_id in self._arrived,
                "effective_hit": platform_id in self._effective_hits,
                "alive_at_end": _health(_lookup(entities, platform_id)) > 0.0,
                "minimum_target_distance_m": (
                    self._minimum_target_distance[platform_id]
                    if self._minimum_target_distance[platform_id] < float("inf")
                    else None
                ),
            }
            for platform_id in sorted(self._launched)
        }

        return {
            "definitions": {
                "evasion_rate": "engaged red missiles that reached their assigned target / resolved engaged red missiles",
                "arrival_rate": "red missiles reaching the assigned target within the 500 m sampled-state tolerance / launched red missiles",
                "hit_rate": "arrivals coincident with actual assigned-target health loss / arrivals",
            },
            "launched": launched,
            "engaged": len(self._engaged),
            "engaged_resolved": len(engaged_resolved),
            "engaged_censored": len(engaged_censored),
            "evaded": len(engaged_arrivals),
            "intercepted": len(intercepted),
            "evasion_rate": len(engaged_arrivals) / max(len(engaged_resolved), 1),
            "arrived": arrivals,
            "arrival_rate": arrivals / max(launched, 1),
            "effective_hits": effective_hits,
            "hit_rate": effective_hits / max(arrivals, 1),
            "hit_rate_per_launch": effective_hits / max(launched, 1),
            "alive_at_end": alive,
            "arrival_tolerance_m": self.arrival_tolerance_m,
            "by_type": by_type,
            "per_platform": per_platform,
        }
