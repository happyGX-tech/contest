"""Minimal contracts for red planning under partial observability.

This module deliberately contains no global target catalogue and no attack
policy. Core owns the authoritative observation and communication graph;
planning code can only operate on search areas and tracks it has been given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from typing import Mapping


@dataclass(frozen=True)
class Position:
    lon: float
    lat: float
    alt: float = 0.0


@dataclass(frozen=True)
class PlatformState:
    platform_id: int
    kind: str
    position: Position
    launched: bool = False
    alive: bool = True


@dataclass(frozen=True)
class SearchArea:
    """A scenario-provided search region, not a hidden target coordinate."""

    area_id: str
    center: Position
    radius_km: float
    prior_detection_value: float = 1.0

    def __post_init__(self) -> None:
        if self.radius_km <= 0:
            raise ValueError("search area radius must be positive")
        if self.prior_detection_value < 0:
            raise ValueError("search-area prior must be non-negative")


class TrackSource(StrEnum):
    ORGANIC_SENSOR = "organic_sensor"
    RELAY = "relay"
    SATELLITE = "satellite"
    PRE_MISSION = "pre_mission"


@dataclass(frozen=True)
class Track:
    """An authoritative enemy track produced by core, never by a policy."""

    track_id: str
    entity_id: int
    position: Position
    observed_step: int
    source: TrackSource
    visible_to: frozenset[int] = field(default_factory=frozenset)
    expires_step: int | None = None

    def is_usable_by(self, platform_id: int, step: int) -> bool:
        return platform_id in self.visible_to and (self.expires_step is None or step <= self.expires_step)


@dataclass(frozen=True)
class PlanningObservation:
    """The complete legal input of a partial-observation red planner."""

    step: int
    platforms: tuple[PlatformState, ...]
    search_areas: tuple[SearchArea, ...]
    tracks: tuple[Track, ...]

    def available_search_areas(self, platform_id: int) -> tuple[SearchArea, ...]:
        platform = next((item for item in self.platforms if item.platform_id == platform_id), None)
        if platform is None or not platform.alive or platform.launched:
            return ()
        return self.search_areas

    def available_tracks(self, platform_id: int) -> tuple[Track, ...]:
        platform = next((item for item in self.platforms if item.platform_id == platform_id), None)
        if platform is None or not platform.alive:
            return ()
        return tuple(item for item in self.tracks if item.is_usable_by(platform_id, self.step))


@dataclass(frozen=True)
class SearchLaunch:
    platform_id: int
    area_id: str


@dataclass(frozen=True)
class StrikeLaunch:
    platform_id: int
    track_id: str


@dataclass(frozen=True)
class Retarget:
    platform_id: int
    track_id: str


PlanningAction = SearchLaunch | StrikeLaunch | Retarget


def action_is_legal(action: PlanningAction, observation: PlanningObservation) -> bool:
    """Validate intent against the information that core has authorized."""

    if isinstance(action, SearchLaunch):
        return any(item.area_id == action.area_id for item in observation.available_search_areas(action.platform_id))
    return any(item.track_id == action.track_id for item in observation.available_tracks(action.platform_id))


def track_index(observation: PlanningObservation) -> Mapping[str, Track]:
    return {item.track_id: item for item in observation.tracks}
