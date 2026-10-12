#!/usr/bin/env python3
"""Fail-closed integrity checks for the PlanSpec-v2 publication bundle."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    plans = check_plans()
    scores = read_scores()
    formula_runs = check_score_formula(scores)
    baseline = [
        value for key, value in scores.items() if key[2] == "A00_balanced"
    ]
    print(json.dumps({
        "status": "ok",
        "protocol_files_checked": protocol_files,
        "plans_checked": plans,
        "score_rows_checked": len(scores),
        "formula_runs_checked": formula_runs,
        "A00_runs": len(baseline),
        "A00_mean": sum(baseline) / len(baseline),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
