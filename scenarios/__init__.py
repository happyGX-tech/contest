"""Bundled competition scenarios and scoring metadata.

The engine normally runs with ``core`` as its script directory, so its
``scenarios`` package is the environment-owned one under ``core/``.  When it is
launched through ``-m core.main`` from the repository root this package shadows
it, so the engine-facing API is forwarded here to keep both launch modes
identical (data files stay under ``scenarios/cases``).
"""

from core.scenarios import (
    CASES_ROOT,
    SCENARIO_ROOT,
    ScenarioBundle,
    load_scenario,
    resolve_deployment_constraints,
    resolve_scenario_path,
)

__all__ = [
    "CASES_ROOT",
    "SCENARIO_ROOT",
    "ScenarioBundle",
    "load_scenario",
    "resolve_deployment_constraints",
    "resolve_scenario_path",
]
