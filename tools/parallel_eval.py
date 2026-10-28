#!/usr/bin/env python3
"""Parallel evaluator for engine cells (plan 65, stage 1-3).

Runs independent ``run.py run`` subprocesses concurrently: one cell = one game
(scenario x blue policy x motion policy x seed). Existing ``summary.json`` files
are reused, so re-running a matrix costs seconds. The scheduler is dynamic:
every cell is submitted at once and the pool takes them as workers free up, which
naturally mixes short E/M games with long H games.

Does not touch ``core/``: each worker is a separate ``run.py run`` process with its
own ``--run-id`` and ``OMP_NUM_THREADS=1``.

Usage:
    tools/parallel_eval.py --map E01,M03,H07 --blue b0,b3,b7,b9 \
        --ckpt models/local_avoid_ppo_v13_gate_aligned_s3_best.pt --jobs 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def scenario_for(map_id: str, scenario_root: Path) -> tuple[str, Path]:
    """Return (scenario path, case directory) for a map id under scenario_root."""
    matches = sorted(scenario_root.glob(f"*/*{map_id}/scenario.json")) + sorted(
        scenario_root.glob(f"*/*/{map_id}/scenario.json")
    )
    if not matches:
        raise SystemExit(f"scenario not found for map {map_id!r} under {scenario_root}")
    case_dir = matches[0].parent
    relative = case_dir.relative_to(scenario_root)
    # run.py resolves a bare "hard/H07" against final20 only when it exists, but a
    # namespaced "final20/hard/H07" is unambiguous for every suite root.
    return f"{scenario_root.name}/{str(relative).replace(os.sep, '/')}", case_dir


def find_summary(runs_root: Path, run_id: str) -> Path | None:
    matches = sorted(runs_root.glob(f"{run_id}/*/summary.json"))
    return matches[0] if matches else None


def worker_environment() -> dict[str, str]:
    env = dict(os.environ)
    repo = str(ROOT)
    env["PYTHONPATH"] = f"{repo}:{repo}/core" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    env["LD_LIBRARY_PATH"] = (
        f"{repo}/core:{repo}/core/envengine/sdk/Util:"
        "/home/amax/miniconda3/envs/macomp/lib"
        + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else "")
    )
    env["OMP_NUM_THREADS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def run_cell(task: dict, runs_root: Path, timeout: int) -> dict:
    run_id = task["run_id"]
    cached = find_summary(runs_root, run_id)
    if cached is not None:
        summary = json.loads(cached.read_text(encoding="utf-8"))
        return {**task, "summary": str(cached), "cached": True,
                "score": float(summary["score"]["score"]),
                "steps": int(summary.get("steps_executed", 0))}
    log_path = Path(f"/tmp/parallel_eval_{run_id.replace('/', '_')}.log")
    command = [
        PY,
        str(ROOT / "run.py"),
        "run",
        "--scenario", task["scenario"],
        "--red-policy", "r9_hierarchical_learning",
        "--blue-policy", task["blue_policy"],
        "--red-motion-policy", task["red_motion_policy"],
        "--red-learning-model", task["ckpt"],
        "--seed", str(task["seed"]),
        "--run-id", run_id,
        "--",
        "--total-rounds", "1",
        "--render-mode", "none",
        "--disable-log-color",
    ]
    started = time.monotonic()
    with log_path.open("wb") as log:
        try:
            result = subprocess.run(
                command, cwd=ROOT, env=worker_environment(), stdout=log,
                stderr=subprocess.STDOUT, timeout=timeout,
            )
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = -9
    elapsed = time.monotonic() - started
    found = find_summary(runs_root, run_id)
    if found is None:
        return {**task, "summary": None, "cached": False, "exit": exit_code,
                "seconds": round(elapsed, 1), "score": None, "steps": None,
                "log": str(log_path)}
    summary = json.loads(found.read_text(encoding="utf-8"))
    return {**task, "summary": str(found), "cached": False, "exit": exit_code,
            "seconds": round(elapsed, 1), "score": float(summary["score"]["score"]),
            "steps": int(summary.get("steps_executed", 0)), "log": str(log_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="E01,M03,H07", help="comma separated map ids")
    parser.add_argument("--blue", default="b0,b3,b7,b9", help="comma separated blue policies")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--red-motion-policy", default="local_avoid_s3")
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--run-prefix", default="parallel_eval")
    parser.add_argument("--scenario-root", type=Path,
                        default=ROOT / "scenarios/cases/final20")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    blue_aliases = {
        "b0": "b0_fixed_ratio_random",
        "b3": "b3_min_cost_assignment",
        "b7": "b7_planned_nearest",
        "b9": "b9_min_cost_planned_multi_wave",
    }
    tasks = []
    for map_id in [item.strip() for item in args.map.split(",") if item.strip()]:
        scenario, _case_dir = scenario_for(map_id, args.scenario_root)
        for blue in [item.strip() for item in args.blue.split(",") if item.strip()]:
            blue_policy = blue_aliases.get(blue, blue)
            run_id = f"{args.run_prefix}/seed_{args.seed:04d}/{map_id}_{blue}_s{args.seed}"
            tasks.append({
                "map": map_id,
                "blue": blue,
                "blue_policy": blue_policy,
                "scenario": scenario,
                "ckpt": args.ckpt,
                "red_motion_policy": args.red_motion_policy,
                "seed": args.seed,
                "run_id": run_id,
            })

    print(f"parallel_eval: {len(tasks)} cells, jobs={args.jobs}, "
          f"timeout={args.timeout}s", flush=True)
    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(run_cell, task, args.runs_root, args.timeout): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            results.append(result)
            if result["score"] is None:
                print(f"  FAIL  {task['map']}/{task['blue']} exit={result.get('exit')} "
                      f"{result.get('seconds')}s log={result.get('log')}", flush=True)
            else:
                mark = "cached" if result["cached"] else f"{result['seconds']}s"
                print(f"  OK    {task['map']}/{task['blue']} score={result['score']:.2f} "
                      f"steps={result['steps']} ({mark})", flush=True)

    elapsed = time.monotonic() - started
    order = {(t["map"], t["blue"]): i for i, t in enumerate(tasks)}
    results.sort(key=lambda item: order[(item["map"], item["blue"])])
    print()
    print(f"{'map':4} {'blue':4} {'score':>7} {'steps':>6} {'time':>7}  status")
    for result in results:
        status = "cached" if result["cached"] else (
            "ok" if result["score"] is not None else f"FAIL exit={result.get('exit')}"
        )
        score = f"{result['score']:.2f}" if result["score"] is not None else "-"
        print(f"{result['map']:4} {result['blue']:4} {score:>7} "
              f"{str(result['steps']):>6} {str(result.get('seconds', '-')):>7}  {status}")
    scores = [item["score"] for item in results if item["score"] is not None]
    if scores:
        print(f"\nmean {sum(scores) / len(scores):.2f} over {len(scores)} cells; "
              f"wall {elapsed:.0f}s")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"tasks": results, "wall_seconds": round(elapsed, 1)},
                       indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"json -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
