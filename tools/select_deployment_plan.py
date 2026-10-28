"""Pick the deployment-plan variant per map: the M-as-decoy layout is not
uniformly better, so the choice is part of the strategy, not the launcher.

Usage:
    python tools/select_deployment_plan.py --scenario scenarios/cases/launch20/mid/M01/scenario.json
    python tools/select_deployment_plan.py --map M01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_ROOT = REPO_ROOT / "results" / "deployment_plans"
POLICY_PATH = PLAN_ROOT / "plan_policy.json"


def load_policy() -> dict:
    return json.loads(POLICY_PATH.read_text())


BAND_BY_PREFIX = {"E": "easy", "M": "medium", "H": "hard"}


def plan_path_for(map_id: str, band: str = "") -> Path:
    policy = load_policy()
    selected_band = band or BAND_BY_PREFIX.get(map_id[:1].upper(), "")
    variant = policy.get("bands", {}).get(selected_band, policy.get("default"))
    if variant not in policy["variants"]:
        raise SystemExit(f"unknown plan variant {variant!r} for map {map_id}")
    path = PLAN_ROOT / f"{map_id}_{variant}.json"
    if not path.exists():
        raise SystemExit(f"missing plan {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="")
    parser.add_argument("--scenario", default="")
    parser.add_argument("--band", default="", choices=["", "easy", "medium", "hard"])
    args = parser.parse_args()
    map_id = args.map
    band = args.band
    if not map_id and args.scenario:
        scenario = Path(args.scenario)
        map_id = scenario.parent.name
        if not band and scenario.parent.parent.name in ("easy", "medium", "hard"):
            band = scenario.parent.parent.name
    if not map_id:
        raise SystemExit("--map or --scenario is required")
    print(plan_path_for(map_id, band).relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
