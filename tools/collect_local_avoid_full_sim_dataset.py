#!/usr/bin/env python3
"""Build a permanent CPU-collected, multi-map local-avoid trajectory library."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policies.red.skills.evasion_ppo.trajectory_dataset import (  # noqa: E402
    archive_policy,
    archive_trace,
    archive_training_run,
    atomic_json,
    initialize_dataset,
    sha256_file,
    verify_dataset,
)
from tools.train_local_avoid_full_sim import (  # noqa: E402
    DEFAULT_IMAGE,
    DEFAULT_V10_TRAIN_SCENARIOS,
    WorkerTask,
    append_jsonl,
    run_worker,
    scenario_schedule,
    select_runtime,
)


DEFAULT_DATASET_ROOT = ROOT / "datasets/local_avoid_full_sim_v1"


def add_dataset_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    importer = subparsers.add_parser(
        "import-run", help="Archive all existing traces from a training run."
    )
    importer.add_argument("run_dir", type=Path)
    add_dataset_root(importer)

    verifier = subparsers.add_parser(
        "verify", help="Verify hashes and trace metadata for the full dataset."
    )
    add_dataset_root(verifier)

    collector = subparsers.add_parser(
        "collect", help="Run CPU simulator workers and archive each trace immediately."
    )
    add_dataset_root(collector)
    collector.add_argument("--model", type=Path, required=True)
    collector.add_argument(
        "--scenarios", default=",".join(DEFAULT_V10_TRAIN_SCENARIOS)
    )
    collector.add_argument("--rounds", type=int, default=1)
    collector.add_argument("--games-per-round", type=int, default=16)
    collector.add_argument("--workers", type=int, default=12)
    collector.add_argument("--seed-start", type=int, default=13001)
    collector.add_argument("--max-steps", type=int, default=1200)
    collector.add_argument("--timeout-seconds", type=float, default=7200.0)
    collector.add_argument("--minimum-free-gb", type=float, default=10.0)
    collector.add_argument("--blue-policy", default="b0_fixed_ratio_random")
    collector.add_argument(
        "--objective-motion", choices=("static", "rule"), default="static"
    )
    collector.add_argument(
        "--gate-mode", choices=("none", "legacy", "mission_v12"), default="none"
    )
    collector.add_argument(
        "--runtime", choices=("auto", "host", "docker"), default="auto"
    )
    collector.add_argument("--docker-image", default=DEFAULT_IMAGE)
    collector.add_argument(
        "--python-env",
        type=Path,
        default=None,
        help=(
            "Python/Conda environment mounted into Docker. Defaults to "
            "$COMPETITION_PYTHON_ENV or the environment running this script."
        ),
    )
    collector.add_argument("--run-name", default="local_avoid_cpu_dataset")
    args = parser.parse_args()
    if args.command == "collect":
        if min(args.rounds, args.games_per_round, args.workers, args.max_steps) <= 0:
            parser.error("rounds/games/workers/max-steps must be positive")
        if args.timeout_seconds <= 0.0 or args.minimum_free_gb < 0.0:
            parser.error("timeout must be positive and minimum-free-gb non-negative")
    return args


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024.0**3)


def collect(args: argparse.Namespace) -> int:
    dataset_root = initialize_dataset(args.dataset_root)
    args.model = args.model.resolve()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if free_gb(dataset_root) < args.minimum_free_gb:
        raise RuntimeError(
            f"dataset filesystem has only {free_gb(dataset_root):.2f} GiB free"
        )

    scenarios = [value.strip() for value in args.scenarios.split(",") if value.strip()]
    if not scenarios:
        raise ValueError("at least one scenario is required")
    for scenario in scenarios:
        scenario_path = ROOT / "scenarios" / "cases" / scenario / "scenario.json"
        if not scenario_path.is_file():
            raise FileNotFoundError(scenario_path)

    policy = archive_policy(dataset_root, args.model)
    policy_snapshot = dataset_root / policy["policy_object"]
    args.runtime = select_runtime(args.runtime)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = dataset_root / "runs" / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    history_path = run_dir / "collection_history.jsonl"
    status_path = run_dir / "run_status.json"
    protocol = {
        "status": "running",
        "pipeline": "complete_simulator_trajectory_collector_v1",
        "created_at": timestamp,
        "dataset_root": str(dataset_root),
        "policy_source": str(args.model),
        "policy_snapshot": str(policy_snapshot),
        "policy_sha256": policy["policy_sha256"],
        "scenarios": scenarios,
        "rounds": args.rounds,
        "games_per_round": args.games_per_round,
        "workers": args.workers,
        "seed_start": args.seed_start,
        "max_steps": args.max_steps,
        "timeout_seconds": args.timeout_seconds,
        "blue_policy": args.blue_policy,
        "objective_motion": args.objective_motion,
        "gate_mode": args.gate_mode,
        "runtime": args.runtime,
    }
    atomic_json(run_dir / "protocol.json", protocol)
    atomic_json(status_path, protocol)
    append_jsonl(history_path, {"event": "start", **protocol})

    all_workers: list[dict[str, Any]] = []
    for round_index in range(1, args.rounds + 1):
        if free_gb(dataset_root) < args.minimum_free_gb:
            raise RuntimeError(
                f"free space fell below {args.minimum_free_gb:.2f} GiB"
            )
        round_scenarios = scenario_schedule(
            scenarios, args.games_per_round, round_index
        )
        tasks = []
        for index, scenario in enumerate(round_scenarios):
            seed = args.seed_start + (round_index - 1) * args.games_per_round + index
            map_id = scenario.rsplit("/", 1)[-1]
            tasks.append(
                WorkerTask(
                    iteration=round_index,
                    worker_index=index % args.workers,
                    seed=seed,
                    checkpoint_path=policy_snapshot,
                    trace_path=(
                        run_dir
                        / "incoming"
                        / f"round_{round_index:04d}"
                        / f"{map_id}_seed_{seed:05d}.jsonl.gz"
                    ),
                    log_path=(
                        run_dir
                        / "logs"
                        / f"round_{round_index:04d}"
                        / f"{map_id}_seed_{seed:05d}.log"
                    ),
                    run_id=(
                        f"{args.run_name}_r{round_index:04d}_{map_id}_s{seed:05d}"
                    ),
                    scenario=scenario,
                )
            )

        round_results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_worker, args, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                trace_path = Path(str(result.get("trace_path", "")))
                archive_result = None
                if trace_path.is_file():
                    archive_result = archive_trace(
                        dataset_root,
                        trace_path,
                        metadata={
                            "source_kind": "cpu_collector",
                            "source_run": str(run_dir),
                            "run_id": result.get("run_id"),
                            "iteration": round_index,
                            "scenario": result.get("scenario"),
                            "seed": result.get("seed"),
                            "blue_policy": args.blue_policy,
                            "objective_motion": args.objective_motion,
                            "gate_mode": args.gate_mode,
                            "policy_sha256": policy["policy_sha256"],
                            "policy_object": policy["policy_object"],
                        },
                        source_error=result.get("error"),
                    )
                event = {
                    "event": "worker",
                    "completed": completed,
                    **result,
                    "archive": (
                        {
                            "trace_sha256": archive_result["trace_sha256"],
                            "object_path": archive_result["object_path"],
                            "usable_for_outcome_training": archive_result[
                                "usable_for_outcome_training"
                            ],
                        }
                        if archive_result is not None
                        else None
                    ),
                }
                round_results.append(event)
                all_workers.append(event)
                append_jsonl(history_path, event)
                print(
                    f"ROUND {round_index}/{args.rounds} WORKER "
                    f"{completed}/{len(tasks)} scenario={result.get('scenario')} "
                    f"status={'error' if result.get('error') else 'archived'}",
                    flush=True,
                )
        round_results.sort(key=lambda value: int(value["seed"]))
        atomic_json(
            run_dir / f"workers_round_{round_index:04d}.json",
            {"workers": round_results},
        )
        atomic_json(
            status_path,
            {
                **protocol,
                "status": "running",
                "completed_rounds": round_index,
                "workers_completed": len(all_workers),
            },
        )

    if sha256_file(policy_snapshot) != policy["policy_sha256"]:
        raise RuntimeError("archived policy snapshot changed during collection")
    failures = sum(bool(item.get("error")) for item in all_workers)
    final = {
        **protocol,
        "status": "complete",
        "run_dir": str(run_dir),
        "workers_completed": len(all_workers),
        "worker_failures": failures,
        "usable_archives": sum(
            bool((item.get("archive") or {}).get("usable_for_outcome_training"))
            for item in all_workers
        ),
        "quarantined_archives": sum(
            item.get("archive") is not None
            and not bool(item["archive"].get("usable_for_outcome_training"))
            for item in all_workers
        ),
    }
    atomic_json(run_dir / "collection_summary.json", final)
    atomic_json(status_path, final)
    append_jsonl(history_path, {"event": "complete", **final})
    print(f"COLLECTION_COMPLETE {run_dir}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "import-run":
        summary = archive_training_run(args.dataset_root, args.run_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "verify":
        summary = verify_dataset(args.dataset_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["valid"] else 1
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
