#!/usr/bin/env python3
"""Fail-closed integrity checks for the PlanSpec-v2 publication bundle."""

from __future__ import annotations

import ast
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SNAPSHOT_SOURCES = (
    "snapshot/run.py",
    "snapshot/core/evaluation/run_summary.py",
    "snapshot/core/user_agents/base_agent/base_agent.py",
    "snapshot/core/user_agents/unified_missile_agent.py",
    "snapshot/grid_maps/agent.py",
    "snapshot/policies/blue/b0_fixed_ratio_random.py",
    "snapshot/policies/red/baselines.py",
    "snapshot/policies/red/tracks.py",
    "snapshot/policies/red/learning/plan_skill/executor.py",
    "snapshot/policies/red/learning/plan_skill/spec.py",
    "snapshot/policies/red/learning/unified/action_spec.py",
    "snapshot/policies/red/learning/unified/info_center.py",
    "snapshot/policies/red/learning/unified/unified_encoder.py",
    "snapshot/policies/red/learning/unified/unified_networks.py",
    "snapshot/policies/red/skills/evasion_ppo/local_runtime.py",
)

RECOVERED_DEPENDENCY_HASHES = {
    "snapshot/policies/red/learning/unified/action_spec.py":
        "701616e14b25c68d4961ff55cb39aa97cfda131d28a77995839500ff33cf2698",
    "snapshot/policies/red/learning/unified/info_center.py":
        "5c8dd3de1abcc43b8fb072eff61d57d6345dddf1b9b661b943e784e93b87d09e",
    "snapshot/policies/red/learning/unified/unified_encoder.py":
        "55f75812970c54dee3f7394c6b8abe64ce6a4b1c5c3d7fadd7d97f021e144fdc",
    "snapshot/policies/red/learning/unified/unified_networks.py":
        "8261ab18fa99456ad5453e0ba88e43d355174cc372f0ef14c36daa9d187b3032",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_protocol_files() -> int:
    protocol = json.loads(
        (ROOT / "provenance/protocol_identity.json").read_text(encoding="utf-8")
    )
    bundle_paths = {
        "evasion_model": ROOT / (
            "snapshot/models/local_avoid_ppo_v13_gate_aligned_s3_best.pt"
        ),
        "plan_executor": ROOT / (
            "snapshot/policies/red/learning/plan_skill/executor.py"
        ),
        "plan_schema": ROOT / (
            "snapshot/policies/red/learning/plan_skill/spec.py"
        ),
        "unified_agent": ROOT / (
            "snapshot/core/user_agents/unified_missile_agent.py"
        ),
        "core_main": ROOT / "snapshot/core/main.py",
    }
    for name, path in bundle_paths.items():
        expected = protocol["shared_files"][name]["sha256"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{name} hash mismatch: {actual} != {expected}")

    for scenario, metadata in protocol["scenarios"].items():
        path = ROOT / "snapshot/scenarios/cases" / scenario / "scenario.json"
        actual = sha256(path)
        if actual != metadata["sha256"]:
            raise ValueError(f"scenario hash mismatch for {scenario}")
    return len(bundle_paths) + len(protocol["scenarios"])


def check_snapshot_sources() -> int:
    """Check the published Python-side replay dependency closure.

    This is intentionally a syntax/presence check.  Importing ``core/main.py``
    still requires the separately distributed native simulator package.
    """

    for relative in REQUIRED_SNAPSHOT_SOURCES:
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"missing replay dependency: {relative}")

    for relative, expected in RECOVERED_DEPENDENCY_HASHES.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(
                f"recovered dependency hash mismatch for {relative}: "
                f"{actual} != {expected}"
            )

    sources = sorted((ROOT / "snapshot").rglob("*.py"))
    for path in sources:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            raise ValueError(f"invalid Python snapshot source: {path}") from error
    return len(sources)


def check_plans() -> int:
    root_manifest = json.loads(
        (ROOT / "plans/manifest.json").read_text(encoding="utf-8")
    )
    if root_manifest["expected_runs"] != 1200:
        raise ValueError("root manifest expected_runs is not 1200")
    if root_manifest["map_count"] != 20 or root_manifest["anchor_count"] != 12:
        raise ValueError("root manifest is not the 20-map x 12-anchor matrix")

    checked = 0
    for map_row in root_manifest["maps"]:
        manifest_path = ROOT / "plans" / map_row["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(manifest["anchors"]) != 12:
            raise ValueError(f"wrong anchor count in {manifest_path}")
        for anchor in manifest["anchors"]:
            plan_path = manifest_path.parent / anchor["plan_file"]
            tokens = json.loads(plan_path.read_text(encoding="utf-8"))
            canonical = json.dumps(
                tokens, ensure_ascii=False, separators=(",", ":")
            )
            plan_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if plan_id != anchor["plan_id"]:
                raise ValueError(f"plan_id mismatch in {plan_path}")
            checked += 1
    if checked != 240:
        raise ValueError(f"expected 240 plan files, got {checked}")
    return checked


def read_scores() -> dict[tuple[str, int, str], float]:
    with (ROOT / "data/scores.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1200:
        raise ValueError(f"expected 1200 scores, got {len(rows)}")
    scores = {
        (row["scenario_id"], int(row["seed"]), row["anchor_id"]):
        float(row["score"])
        for row in rows
    }
    if len(scores) != 1200:
        raise ValueError("score keys are not unique")
    if len({key[0] for key in scores}) != 20:
        raise ValueError("score export does not contain 20 maps")
    if len({key[1] for key in scores}) != 5:
        raise ValueError("score export does not contain 5 shared seeds")
    if len({key[2] for key in scores}) != 12:
        raise ValueError("score export does not contain 12 anchors")
    return scores


def check_score_formula(scores: dict[tuple[str, int, str], float]) -> int:
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    with (ROOT / "data/objective_health.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            key = (row["scenario_id"], int(row["seed"]), row["anchor_id"])
            groups[key].append(row)
    if set(groups) != set(scores):
        raise ValueError("objective-health keys do not match score keys")

    for key, objectives in groups.items():
        weighted_damage = 0.0
        total_weight = 0.0
        for objective in objectives:
            weight = float(objective["weight"])
            initial = float(objective["initial_health"])
            final = float(objective["final_health"])
            damage_fraction = max(0.0, min(1.0, (initial - final) / initial))
            weighted_damage += weight * damage_fraction
            total_weight += weight
        calculated = 100.0 * weighted_damage / total_weight
        if not math.isclose(calculated, scores[key], rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"score formula mismatch for {key}: {calculated} != {scores[key]}"
            )
    return len(groups)


def main() -> None:
    protocol_files = check_protocol_files()
    snapshot_sources = check_snapshot_sources()
    plans = check_plans()
    scores = read_scores()
    formula_runs = check_score_formula(scores)
    baseline = [
        value for key, value in scores.items() if key[2] == "A00_balanced"
    ]
    print(json.dumps({
        "status": "ok",
        "protocol_files_checked": protocol_files,
        "snapshot_python_files_checked": snapshot_sources,
        "plans_checked": plans,
        "score_rows_checked": len(scores),
        "formula_runs_checked": formula_runs,
        "A00_runs": len(baseline),
        "A00_mean": sum(baseline) / len(baseline),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
