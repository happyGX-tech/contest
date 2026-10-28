"""Red assignment and per-platform execution inside a communication domain."""

from importlib import import_module

from .baselines import RED_POLICY_CHOICES
from .commander import RedBaselineCommander
from .contracts import Position
from .priors import initial_targets_from_observation

__all__ = [
    "Position",
    "RED_POLICY_CHOICES",
    "AttackMissileAgent",
    "DeployAgent",
    "RedBaselineCommander",
    "initial_targets_from_observation",
]


def __getattr__(name):
    modules = {
        "AttackMissileAgent": ".attack_missile_agent",
        "DeployAgent": ".deploy_agent",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
