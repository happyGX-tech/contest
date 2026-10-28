"""Coordinate value shared by red assignment and search geometry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    """Engine coordinates; scenario conversion owns the displayed kilometre axes."""

    lon: float
    lat: float
    alt: float = 0.0
