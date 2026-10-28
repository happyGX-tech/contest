"""Resolve scenario IDs and load judge-side configuration without the engine."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .cases import RewardPolicy, load_reward_policy


SCENARIO_ROOT = Path(__file__).resolve().parent
CASES_ROOT = SCENARIO_ROOT / "cases"


@dataclass(frozen=True)
class ScenarioBundle:
    """Environment-owned metadata; never pass this bundle to contestant agents."""

    path: Path | str
    reward_policy: RewardPolicy | None
    deployment_area: dict[str, Any] | None = None
    case_info: dict[str, Any] | None = None


def resolve_scenario_path(scenario: str | Path = "default") -> Path | str:
    """Accept a bundled ID, a JSON path, or an engine-supported HTTP(S) URL."""
    value = str(scenario)
    if value.startswith(("http://", "https://")):
        return value
    if value == "default":
        return SCENARIO_ROOT / "platform.json"

    # IDs have the same meaning from the CLI and programmatic callers.
    bundled = (CASES_ROOT / value / "scenario.json").resolve()
    # CASES_ROOT may be reached through a symlink (the project keeps one shared
    # scenarios/cases tree), so compare against its resolved form.
    if bundled.is_relative_to(CASES_ROOT.resolve()) and bundled.is_file():
        return bundled
    external = Path(value).expanduser()
    if external.suffix.lower() == ".json" and external.is_file():
        return external.resolve()
    if external.name == value and external.suffix.lower() == ".json":
        stored = SCENARIO_ROOT / external.name
        if stored.is_file():
            return stored

    # Keep the entry point's historical default independent of cwd.
    normalized = value.replace("\\", "/").removeprefix("./")
    if normalized.startswith("scenarios/"):
        local = (SCENARIO_ROOT / normalized.removeprefix("scenarios/")).resolve()
        if local.is_relative_to(SCENARIO_ROOT) and local.suffix.lower() == ".json" and local.is_file():
            return local
    raise ValueError(
        f"Unknown scenario '{value}'. Use e.g. launch20/easy/E01, easy/E01, "
        "default, or an existing scenario.json path."
    )


def _optional_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Scenario metadata must be a JSON object: {path}")
    return value


def load_scenario(scenario: str | Path = "default") -> ScenarioBundle:
    """Load deployment and scoring metadata beside the selected map.

    The engine continues to read the scene JSON through its existing loader.
    Hash validation of competition scenes is performed by load_reward_policy.
    Remote scenes keep the engine's previous behavior without local metadata.
    """
    path = resolve_scenario_path(scenario)
    if isinstance(path, str):
        return ScenarioBundle(path=path, reward_policy=None)
    return ScenarioBundle(
        path=path,
        reward_policy=load_reward_policy(path),
        deployment_area=_optional_object(path.with_name("deployment_area.json")),
        case_info=_optional_object(path.with_name("case_info.json")),
    )
