"""Contestant agents. Importing this package does not load the native engine."""

from importlib import import_module

__all__ = ["AttackMissileAgent", "DeployAgent"]


def __getattr__(name):
    modules = {
        "AttackMissileAgent": ".red.attack_missile_agent",
        "DeployAgent": ".red.deploy_agent",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
