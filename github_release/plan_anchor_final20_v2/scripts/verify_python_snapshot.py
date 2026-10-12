#!/usr/bin/env python3
"""Import-smoke the published Python replay modules without native envengine."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "snapshot"
sys.path.insert(0, str(SNAPSHOT / "core"))
sys.path.insert(0, str(SNAPSHOT))


def main() -> None:
    from policies.red.learning.plan_skill import (
        DeterministicPlanPolicy,
        load_plan_spec,
    )
    from policies.red.learning.unified.action_spec import FlightPhase
    from policies.red.learning.unified.info_center import InfoCenter
    from policies.red.learning.unified.unified_encoder import UnifiedObsEncoder
    from policies.red.learning.unified.unified_networks import CentralCritic
    from policies.red.skills.evasion_ppo import LocalAvoidRuntimePolicy

    objects = (
        DeterministicPlanPolicy,
        FlightPhase,
        InfoCenter,
        UnifiedObsEncoder,
        CentralCritic,
        LocalAvoidRuntimePolicy,
    )
    modules = sorted({item.__module__ for item in objects})
    for name in modules:
        module = sys.modules[name]
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(SNAPSHOT.resolve()):
            raise RuntimeError(f"module resolved outside snapshot: {module_path}")

    plan = load_plan_spec(ROOT / "plans/E01/A00_balanced.json")
    local_avoid = LocalAvoidRuntimePolicy(max_steps=3000)
    local_avoid.load(
        SNAPSHOT / "models/local_avoid_ppo_v13_gate_aligned_s3_best.pt"
    )
    local_avoid.set_training(False)

    print(json.dumps({
        "status": "ok",
        "imports_checked": len(objects),
        "modules": modules,
        "A00_plan_id": plan.plan_id,
        "checkpoint_load_checked": True,
        "local_avoid_gate_mode": local_avoid.gate_mode,
        "native_envengine_required_for_full_replay": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
