#!/usr/bin/env python3
"""Run the 12-anchor x 20-map x paired-seed PlanSpec-v2 matrix.

Every persistent artifact remains under this repository on the yunpan mount.
The runner is resumable and reuses a summary only after the plan ID and
deterministic execution ledger pass the consistency gate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANS = ROOT / "experiments/plan_skill/anchors_final20_all_v2"
DEFAULT_PREFIX = "plan_anchor_12x5_final20_all_v2"
_PRINT_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{ROOT}:{ROOT / 'core'}",
        "PYTHONHASHSEED": "0",
        "RED_LEARNING_TRAIN": "0",
        "SAVE_TRAIN_BEST": "0",
        "EVAL_SAMPLE": "0",
        "SEARCH_ROWS": "4",
        "SEARCH_COLS": "8",
        "TOPFORMER_CPU_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "EVASION_MODEL": str(
            ROOT / "models/local_avoid_ppo_v13_gate_aligned_s3_best.pt"
        ),
        "LOCAL_AVOID_GATE_MODE": "mission_v12",
        "PLAN_EARLY_STOP": "1",
        "LD_PRELOAD": ":".join((
            str(ROOT / "core/envengine/sdk/Util/_UtilsPy.so"),
            str(ROOT / (
                "core/envengine/simulator/models_bak/"
                "CompCruiseMissileModel/libUtils.so"
            )),
        )),
        "LD_LIBRARY_PATH": ":".join((
            "/home/ubuntu/yunpan/miniconda3/envs/competition/lib",
            str(ROOT / "core/envengine/simulator/models/ACMMissileModel"),
            str(ROOT / "core/envengine/simulator/models/CompCruiseMissileModel"),
            str(ROOT / "core/envengine/simulator/models/HXDMissileModel"),
            str(ROOT / "core/envengine/simulator/models/OrbitModel"),
            str(ROOT / "core/envengine/simulator/models/PAC2Model"),
            str(ROOT / "core/envengine/simulator/models/SM6_1BMissileModel"),
            str(ROOT / "core/envengine/sdk/Util"),
        )),
    })
    return env


def _latest_summary(run_id: str) -> Path | None:
    matches = sorted(glob.glob(str(
        ROOT / "results" / "runs" / run_id / "*" / "summary.json"
    )))
    return Path(matches[-1]) if matches else None


def _execution_gate(
    summary_path: Path, expected_plan_id: str
) -> tuple[bool, str]:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid summary: {error}"
    execution = payload.get("plan_execution") or {}
    launch = execution.get("launch_goal") or {}
    deployment = execution.get("deployment") or {}
    if execution.get("plan_id") != expected_plan_id:
        return False, "plan_id mismatch"
    if execution.get("trainable_parameters") != 0:
        return False, "executor unexpectedly has trainable parameters"
    if not deployment.get("all_match"):
        return False, "deployment mismatch"
    inconsistent = [
        unit for unit in launch.get("units", ())
        if unit.get("status") not in (
            "exact", "held_reserve", "not_triggered",
            "not_needed_score_frozen",
        )
        and not (
            unit.get("status") == "not_launched"
            and (unit.get("blocked") or {}).get("reason")
            == "planned_target_not_in_legal_candidate_set"
            and unit.get("actual") is None
        )
    ]
    if inconsistent:
        return False, "launch/goal mismatch"
    if execution.get("semantic_retarget_violations"):
        return False, "semantic retarget violation"
    if not isinstance(payload.get("score"), dict):
        return False, "missing official score"
    blocked = sum(
        unit.get("status") == "not_launched"
        and (unit.get("blocked") or {}).get("reason")
        == "planned_target_not_in_legal_candidate_set"
        for unit in launch.get("units", ())
    )
    return True, "ok" if blocked == 0 else f"ok ({blocked} targets unavailable)"


def _protocol_identity(
    python_bin: Path, map_manifests: list[tuple[Path, dict]]
) -> dict:
    shared_paths = {
        "evasion_model": ROOT / "models/local_avoid_ppo_v13_gate_aligned_s3_best.pt",
        "plan_executor": ROOT / "policies/red/learning/plan_skill/executor.py",
        "plan_schema": ROOT / "policies/red/learning/plan_skill/spec.py",
        "unified_agent": ROOT / "core/user_agents/unified_missile_agent.py",
        "core_main": ROOT / "core/main.py",
    }
    scenarios = {}
    for _path, manifest in map_manifests:
        scenario = str(manifest["scenario"])
        scenario_path = ROOT / "scenarios/cases" / scenario / "scenario.json"
        scenarios[scenario] = {
            "path": str(scenario_path), "sha256": _sha256(scenario_path)
        }
    return {
        "python": str(python_bin.resolve()),
        "shared_files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in shared_paths.items()
        },
        "scenarios": scenarios,
        "red_policy": "r12_topformer deterministic PlanSpec-v2 executor",
        "red_motion_policy": "local_avoid_s3 mission_v12 inference",
        "blue_policy": "b0_fixed_ratio_random",
        "blue_interceptor_ratio": 2,
        "max_steps": 3000,
        "satellite_rule": "fixed_lowest_live_unused_h",
        "hidden_response": "all legal discoveries then distinct L groups",
        "early_stop": "score frozen under deterministic plan",
    }


def _run_one(
    *,
    python_bin: Path,
    prefix: str,
    map_manifest_path: Path,
    map_manifest: dict,
    anchor: dict,
    seed: int,
    log_dir: Path,
    force: bool,
) -> dict:
    scenario = str(map_manifest["scenario"])
    scenario_id = str(map_manifest["scenario_id"])
    anchor_id = str(anchor["anchor_id"])
    plan_id = str(anchor["plan_id"])
    run_id = f"{prefix}_{scenario_id}_{anchor_id}_s{seed}"
    existing = _latest_summary(run_id)
    if existing is not None and not force:
        ok, reason = _execution_gate(existing, plan_id)
        if ok:
            with _PRINT_LOCK:
                print(
                    f"REUSE {scenario_id} {anchor_id} seed={seed}", flush=True
                )
            return {
                "scenario": scenario,
                "scenario_id": scenario_id,
                "difficulty": map_manifest["difficulty"],
                "anchor_id": anchor_id,
                "seed": seed,
                "run_id": run_id,
                "status": "reused",
                "reason": reason,
                "summary": str(existing),
            }

    plan_path = (map_manifest_path.parent / anchor["plan_file"]).resolve()
    command = [
        str(python_bin), str(ROOT / "run.py"), "run",
        "--scenario", scenario,
        "--seed", str(seed),
        "--red-policy", "r12_topformer",
        "--red-motion-policy", "local_avoid_s3",
        "--blue-policy", "b0_fixed_ratio_random",
        "--blue-interceptor-ratio", "2",
        "--plan-spec", str(plan_path),
        "--run-id", run_id,
        "--", "--total-rounds", "1", "--max-steps", "3000",
        "--render-mode", "none", "--disable-log-color",
    ]
    env = _runtime_env()
    env["PYTHONHASHSEED"] = str(seed)
    log_path = log_dir / f"{run_id}.log"
    started = datetime.now().isoformat(timespec="seconds")
    with _PRINT_LOCK:
        print(f"START {scenario_id} {anchor_id} seed={seed}", flush=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    summary_path = _latest_summary(run_id)
    ok = False
    reason = "summary missing"
    if completed.returncode == 0 and summary_path is not None:
        ok, reason = _execution_gate(summary_path, plan_id)
    elif completed.returncode != 0:
        reason = f"process exit {completed.returncode}"
    status = "completed" if ok else "failed"
    with _PRINT_LOCK:
        print(
            f"{status.upper()} {scenario_id} {anchor_id} seed={seed} "
            f"reason={reason}",
            flush=True,
        )
    return {
        "scenario": scenario,
        "scenario_id": scenario_id,
        "difficulty": map_manifest["difficulty"],
        "anchor_id": anchor_id,
        "seed": seed,
        "run_id": run_id,
        "status": status,
        "reason": reason,
        "returncode": completed.returncode,
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "log": str(log_path),
        "summary": str(summary_path) if summary_path is not None else None,
    }


def _select_map_manifests(
    plan_dir: Path, root_manifest: dict, requested_maps: list[str] | None
) -> list[tuple[Path, dict]]:
    requested = set(requested_maps or ())
    known = {
        str(row["scenario_id"]): row for row in root_manifest["maps"]
    }
    unknown = requested - set(known)
    if unknown:
        raise ValueError(f"unknown maps: {sorted(unknown)}")
    rows = [
        row for row in root_manifest["maps"]
        if not requested or str(row["scenario_id"]) in requested
    ]
    result = []
    for row in rows:
        path = (plan_dir / row["manifest"]).resolve()
        result.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--maps", nargs="+")
    parser.add_argument("--anchors", nargs="+")
    parser.add_argument("--max-jobs", type=int, default=16)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(
            "/home/ubuntu/yunpan/miniconda3/envs/competition/bin/python"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_jobs <= 0:
        raise ValueError("--max-jobs must be positive")
    plan_dir = args.plans.resolve()
    root_manifest = json.loads(
        (plan_dir / "manifest.json").read_text(encoding="utf-8")
    )
    map_manifests = _select_map_manifests(
        plan_dir, root_manifest, args.maps
    )
    seeds = args.seeds or [
        int(value) for value in root_manifest["common_seeds"]
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")

    known_anchors = {
        str(row["anchor_id"]) for row in root_manifest["anchors"]
    }
    requested_anchors = set(args.anchors or ())
    unknown_anchors = requested_anchors - known_anchors
    if unknown_anchors:
        raise ValueError(f"unknown anchors: {sorted(unknown_anchors)}")

    jobs = []
    for map_manifest_path, map_manifest in map_manifests:
        anchors = [
            anchor for anchor in map_manifest["anchors"]
            if not requested_anchors
            or str(anchor["anchor_id"]) in requested_anchors
        ]
        for seed in seeds:
            for anchor in anchors:
                jobs.append({
                    "map_manifest_path": map_manifest_path,
                    "map_manifest": map_manifest,
                    "anchor": anchor,
                    "seed": seed,
                })

    log_dir = ROOT / "results/runs" / f"{args.prefix}_driver_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    index_path = log_dir / "run_index.json"
    records = []
    protocol_identity = _protocol_identity(args.python, map_manifests)
    with ThreadPoolExecutor(max_workers=args.max_jobs) as pool:
        futures = [pool.submit(
            _run_one,
            python_bin=args.python.resolve(),
            prefix=args.prefix,
            map_manifest_path=job["map_manifest_path"],
            map_manifest=job["map_manifest"],
            anchor=job["anchor"],
            seed=job["seed"],
            log_dir=log_dir,
            force=args.force,
        ) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            index_path.write_text(json.dumps({
                "prefix": args.prefix,
                "plans": str(plan_dir),
                "maps": [
                    row[1]["scenario_id"] for row in map_manifests
                ],
                "seeds": seeds,
                "expected_runs": len(jobs),
                "protocol_identity": protocol_identity,
                "records": sorted(records, key=lambda row: (
                    row["scenario_id"], row["seed"], row["anchor_id"]
                )),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    failed = [row for row in records if row["status"] == "failed"]
    print(json.dumps({
        "expected": len(jobs),
        "completed_or_reused": len(records) - len(failed),
        "failed": len(failed),
        "run_index": str(index_path),
    }, ensure_ascii=False, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
