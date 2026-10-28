"""Red planning values aligned with the competition scoring weights."""

from __future__ import annotations


OBJECTIVE_VALUE_BY_TYPE = {
    9400: 5.0,
    9600: 2.0,
    9500: 1.0,
}


def objective_value(entity_type: int) -> float:
    return OBJECTIVE_VALUE_BY_TYPE.get(int(entity_type), 1.0)
