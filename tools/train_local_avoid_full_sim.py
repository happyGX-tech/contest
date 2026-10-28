#!/usr/bin/env python3
"""Iterative local-avoid PPO training backed only by the complete simulator.

This entry point supports both the legacy fixed-two-slot MLP and Set-Attention
checkpoints. It runs complete competition games in isolated worker processes,
collects compact per-missile runtime trajectories, performs centralized PPO
updates, writes TensorBoard/JSONL metrics, saves checkpoints, and reload-validates
each update.

The ``--smoke`` preset deliberately truncates games.  It validates plumbing,
not convergence or policy quality.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policies.red.skills.evasion_ppo.full_sim_rollout import (  # noqa: E402
    annotate_runtime_trace_outcomes,
    build_full_sim_rollout,
    load_runtime_trajectories,
)
from policies.red.skills.evasion_ppo.local_ppo import LocalAvoidPPO  # noqa: E402
from policies.red.skills.evasion_ppo.set_attention_ppo import (  # noqa: E402
    SetAttentionLocalAvoidPPO,
)


DEFAULT_IMAGE = "docker.m.daocloud.io/library/python:3.11-slim-trixie"
SUPPORTED_POLICY_LOADERS = {
    "local_avoid_asymmetric_ppo_v1": LocalAvoidPPO,
    SetAttentionLocalAvoidPPO.algorithm_name: SetAttentionLocalAvoidPPO,
}


def load_full_sim_policy(path: Path, *, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    algorithm = str(payload.get("algorithm", ""))
    try:
        loader = SUPPORTED_POLICY_LOADERS[algorithm]
    except KeyError as error:
        raise ValueError(
            f"unsupported full-simulator PPO checkpoint algorithm: {algorithm!r}"
        ) from error
    return loader.load(path, device=device)


def policy_observation_contract(policy: Any) -> dict[str, int | str]:
    actor_dim = int(policy.config.actor_obs_dim)
    critic_dim = int(policy.config.critic_state_dim)
    if (actor_dim, critic_dim) == (33, 43):
        return {
            "name": "fixed2",
            "architecture": "mlp",
            "max_threat_slots": 2,
            "actor_obs_dim": actor_dim,
            "critic_state_dim": critic_dim,
        }
    if (actor_dim, critic_dim) == (85, 125):
        return {
            "name": "set8",
            "architecture": "set_attention",
            "max_threat_slots": 8,
            "actor_obs_dim": actor_dim,
            "critic_state_dim": critic_dim,
        }
    raise ValueError(
        "unsupported local-avoid observation contract: "
        f"actor={actor_dim}, critic={critic_dim}"
    )


@dataclass(frozen=True)
class WorkerTask:
    iteration: int
    worker_index: int
    seed: int
    checkpoint_path: Path
    trace_path: Path
    log_path: Path
    run_id: str
    scenario: str
    cross_trace_path: Path | None = None


DEFAULT_V10_TRAIN_SCENARIOS = (
    *(f"launch20/easy/E{index:02d}" for index in range(1, 6)),
    *(f"launch20/medium/M{index:02d}" for index in range(1, 6)),
    *(f"launch20/hard/H{index:02d}" for index in range(1, 7)),
)
DEFAULT_V10_HOLDOUT_SCENARIOS = (
    "launch20/easy/E06",
    "launch20/medium/M06",
    "launch20/hard/H07",
    "launch20/hard/H08",
)
DEFAULT_V10_HOLDOUT_SEEDS = (11901, 11902, 11903)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-model",
        type=Path,
        default=ROOT / "models/local_avoid_set_attention_v9_fullsim_static_w16.pt",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=ROOT / "models/local_avoid_set_attention_v10_fullsim_multimap.pt",
    )
    parser.add_argument(
        "--policy-name",
        default="local_avoid_set_attention_v10",
        help="Version label stored in protocol and checkpoint metadata.",
    )
    parser.add_argument(
        "--observation-contract",
        choices=("auto", "fixed2", "set8"),
        default="auto",
        help="Fail closed unless the input checkpoint uses the requested ABI.",
    )
    parser.add_argument("--scenario", default="launch20/easy/E01")
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_V10_TRAIN_SCENARIOS),
        help="Comma-separated stratified map pool; overrides --scenario.",
    )
    parser.add_argument(
        "--heldout-scenarios",
        default=",".join(DEFAULT_V10_HOLDOUT_SCENARIOS),
        help="Comma-separated maps reserved from training and recorded in metadata.",
    )
    parser.add_argument("--blue-policy", default="b0_fixed_ratio_random")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=8101)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument(
        "--tier-holdout",
        choices=("none", "easy", "medium", "hard"),
        default="none",
        help="Train on the other two tiers and reserve the whole named tier.",
    )
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--clip-range-value", type=float, default=None)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--reward-clip", type=float, default=None)
    parser.add_argument("--advantage-standardize", action="store_true")
    parser.add_argument("--anneal-lr", action="store_true")
    parser.add_argument("--anneal-final-scale", type=float, default=0.1)
    parser.add_argument("--entropy-coef-final", type=float, default=None)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run the sampled evaluation every N iterations (0 disables it).",
    )
    parser.add_argument(
        "--eval-final-cells",
        default="H07/b0",
        help=(
            "Cells evaluated once against the final checkpoint (empty string "
            "disables the final check). The expensive hard cell belongs here "
            "instead of --eval-cells so mid-training checks stay cheap."
        ),
    )
    parser.add_argument("--eval-cells", default="E01/b0,M03/b0")
    parser.add_argument("--eval-jobs", type=int, default=2)
    parser.add_argument("--eval-timeout", type=float, default=1800.0)
    parser.add_argument("--eval-seed", type=int, default=701)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--behavior-batch-size", type=int, default=32768)
    parser.add_argument(
        "--reward-mode",
        choices=(
            "success_conditioned",
            "normalized_distance",
            "distance_outcome",
            "timely_arrival",
            "score_weighted",
            "progress_plus_score",
        ),
        default="success_conditioned",
        help=(
            "success_conditioned zeroes failed trajectories; normalized_distance "
            "uses only normalized target-distance reduction; distance_outcome "
            "adds a compact real-outcome terminal ladder; timely_arrival treats "
            "arrival and effective damage identically and penalizes delay/failure; "
            "score_weighted keeps success_conditioned and scales every return by "
            "the judge's objective weight of that platform's target; "
            "progress_plus_score keeps the plain progress return and adds a "
            "terminal bonus on real effective hits scaled by the target weight."
        ),
    )
    parser.add_argument(
        "--score-bonus-coef",
        type=float,
        default=1.0,
        help=(
            "Extra terminal reward for a real effective hit under "
            "--reward-mode progress_plus_score: coef * objective_weight / 5."
        ),
    )
    parser.add_argument(
        "--initial-trace-dir",
        type=Path,
        default=None,
        help="Reuse iteration-1 trajectories collected by the same input checkpoint.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runtime", choices=("auto", "host", "docker"), default="auto")
    parser.add_argument("--docker-image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--python-env",
        type=Path,
        default=None,
        help=(
            "Python/Conda environment mounted into Docker. Defaults to "
            "$COMPETITION_PYTHON_ENV or the environment running this script."
        ),
    )
    parser.add_argument("--objective-motion", choices=("static", "rule"), default="static")
    parser.add_argument("--gate-mode", choices=("none", "legacy", "mission_v12"), default="none")
    parser.add_argument("--run-name", default="local_avoid_set_attention_v10_fullsim_multimap")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/local_avoid_set_attention_v10_fullsim_multimap",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.workers = min(args.workers, 3)
        args.games = min(args.games, 3)
        args.max_steps = min(args.max_steps, 16)
        args.timeout_seconds = min(args.timeout_seconds, 300.0)
        args.iterations = 1
        args.ppo_epochs = 1
        args.minibatch_size = 256
        args.run_name = f"{args.run_name}_pipeline_smoke"
    if (
        args.workers <= 0
        or args.games <= 0
        or args.iterations <= 0
        or args.behavior_batch_size <= 0
        or args.max_steps <= 1
        or (args.gamma is not None and not 0.0 < args.gamma <= 1.0)
        or (args.gae_lambda is not None and not 0.0 < args.gae_lambda <= 1.0)
    ):
        parser.error(
            "workers/games/iterations must be positive, max-steps must exceed "
            "one, and gamma/gae-lambda must be in (0, 1]"
        )
    return args


def scenario_schedule(
    scenarios: list[str], games: int, iteration: int
) -> list[str]:
    """Rotate a near-equal allocation across every tier present in the pool."""

    if len(scenarios) == 1:
        return scenarios * games
    groups = {
        "easy": [value for value in scenarios if "/easy/" in value],
        "medium": [value for value in scenarios if "/medium/" in value],
        "hard": [value for value in scenarios if "/hard/" in value],
    }
    tier_names = tuple(name for name in ("easy", "medium", "hard") if groups[name])
    if not tier_names:
        return [
            scenarios[((iteration - 1) * games + index) % len(scenarios)]
            for index in range(games)
        ]
    tier_offsets = {
        name: (iteration - 1) * ((games + len(tier_names) - 1) // len(tier_names))
        for name in tier_names
    }
    selected = []
    tier_local_counts = {name: 0 for name in tier_names}
    for index in range(games):
        tier = tier_names[(index + iteration - 1) % len(tier_names)]
        choices = groups[tier]
        choice_index = tier_offsets[tier] + tier_local_counts[tier]
        selected.append(choices[choice_index % len(choices)])
        tier_local_counts[tier] += 1
    return selected


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_runtime(requested: str) -> str:
    if requested != "auto":
        return requested
    _name, version = platform.libc_ver()
    try:
        parts = tuple(int(item) for item in version.split(".")[:2])
    except ValueError:
        parts = (0, 0)
    return "host" if parts >= (2, 38) else "docker"


def container_path(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT)
    return str(Path("/workspace") / relative)


def competition_python_env(args: argparse.Namespace) -> Path:
    """Resolve the relocatable Python environment used by Docker workers."""

    configured = getattr(args, "python_env", None)
    if configured is None:
        configured = os.getenv("COMPETITION_PYTHON_ENV")
    environment = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(sys.executable).resolve().parents[1]
    )
    python = environment / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(
            f"competition Python environment is missing {python}; pass "
            "--python-env or set COMPETITION_PYTHON_ENV"
        )
    return environment


def simulator_command(
    args: argparse.Namespace, task: WorkerTask
) -> tuple[list[str], dict[str, str], str | None]:
    launch_trace_path = task.trace_path.with_name(
        task.trace_path.name.removesuffix(".jsonl.gz") + ".launches.jsonl"
    )
    common = [
        "run",
        "--scenario", task.scenario,
        "--blue-policy", args.blue_policy,
        "--red-policy", "r9_hierarchical_learning",
        "--red-motion-policy", "local_avoid_s3",
        "--red-learning-model", str(task.checkpoint_path.resolve()),
        "--seed", str(task.seed),
        "--run-id", task.run_id,
        "--",
        "--total-rounds", "1",
        "--max-steps", str(args.max_steps),
        "--render-mode", "none",
        "--disable-log-color",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "BLUE_OBJECTIVE_MOTION": args.objective_motion,
            "LOCAL_AVOID_GATE_MODE": args.gate_mode,
            "LOCAL_AVOID_ROLLOUT_MODE": "1",
            "LOCAL_AVOID_TRACE_OVERWRITE": "1",
            "LOCAL_AVOID_TRACE_PATH": str(task.trace_path.resolve()),
            "NATIVE_ENCOUNTER_LAUNCH_TRACE_PATH": str(launch_trace_path.resolve()),
            "NATIVE_ENCOUNTER_SOURCE_SCENARIO": task.scenario,
            "NATIVE_ENCOUNTER_SOURCE_RUN_ID": task.run_id,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if task.cross_trace_path is not None:
        environment.update({
            "LOCAL_AVOID_CROSS_ENV_TRACE_PATH": str(task.cross_trace_path.resolve()),
            "LOCAL_AVOID_CROSS_ENV_TRACE_OVERWRITE": "1",
        })
    if args.runtime == "host":
        return [sys.executable, str(ROOT / "run.py"), *common], environment, None

    checkpoint_index = common.index(str(task.checkpoint_path.resolve()))
    common[checkpoint_index] = container_path(task.checkpoint_path)
    trace_in_container = container_path(task.trace_path)
    launch_trace_in_container = container_path(launch_trace_path)
    python_env = competition_python_env(args)
    container_name = f"fullsim-ppo-w{task.worker_index}-s{task.seed}-{os.getpid()}"
    library_path = ":".join(
        (
            "/opt/competition-env/lib",
            "/workspace/core",
            "/workspace/core/envengine/simulator/models/ACMMissileModel",
            "/workspace/core/envengine/simulator/models/CompCruiseMissileModel",
            "/workspace/core/envengine/simulator/models/HXDMissileModel",
            "/workspace/core/envengine/simulator/models/OrbitModel",
            "/workspace/core/envengine/simulator/models/PAC2Model",
            "/workspace/core/envengine/simulator/models/SM6_1BMissileModel",
            "/workspace/core/envengine/sdk/Util",
        )
    )
    command = [
        "docker", "run", "--rm", "--name", container_name,
        "-v", f"{python_env}:/opt/competition-env:ro",
        "-v", f"{ROOT}:/workspace",
        "-w", "/workspace",
        "-e", "PYTHONPATH=/workspace/core",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "PYTHONUNBUFFERED=1",
        "-e", "OMP_NUM_THREADS=1",
        "-e", "MKL_NUM_THREADS=1",
        "-e", f"BLUE_OBJECTIVE_MOTION={args.objective_motion}",
        "-e", f"LOCAL_AVOID_GATE_MODE={args.gate_mode}",
        "-e", "LOCAL_AVOID_ROLLOUT_MODE=1",
        "-e", "LOCAL_AVOID_TRACE_OVERWRITE=1",
        "-e", f"LOCAL_AVOID_TRACE_PATH={trace_in_container}",
        "-e", f"NATIVE_ENCOUNTER_LAUNCH_TRACE_PATH={launch_trace_in_container}",
        "-e", f"NATIVE_ENCOUNTER_SOURCE_SCENARIO={task.scenario}",
        "-e", f"NATIVE_ENCOUNTER_SOURCE_RUN_ID={task.run_id}",
        "-e", f"LD_LIBRARY_PATH={library_path}",
        args.docker_image,
        "/opt/competition-env/bin/python", "/workspace/run.py", *common,
    ]
    if task.cross_trace_path is not None:
        cross_trace_in_container = container_path(task.cross_trace_path)
        image_index = command.index(args.docker_image)
        command[image_index:image_index] = [
            "-e", f"LOCAL_AVOID_CROSS_ENV_TRACE_PATH={cross_trace_in_container}",
            "-e", "LOCAL_AVOID_CROSS_ENV_TRACE_OVERWRITE=1",
        ]
    return command, environment, container_name


def terminate(process: subprocess.Popen[Any], container_name: str | None) -> None:
    if container_name:
        subprocess.run(
            ["docker", "kill", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def run_worker(args: argparse.Namespace, task: WorkerTask) -> dict[str, Any]:
    launch_trace_path = task.trace_path.with_name(
        task.trace_path.name.removesuffix(".jsonl.gz") + ".launches.jsonl"
    )
    task.trace_path.parent.mkdir(parents=True, exist_ok=True)
    if task.cross_trace_path is not None:
        task.cross_trace_path.parent.mkdir(parents=True, exist_ok=True)
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    command, environment, container_name = simulator_command(args, task)
    started = time.monotonic()
    with task.log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            terminate(process, container_name)
            task_data = asdict(task)
            task_data["checkpoint_path"] = str(task.checkpoint_path)
            if task.cross_trace_path is not None:
                task_data["cross_trace_path"] = str(task.cross_trace_path)
            return {
                **task_data,
                "trace_path": str(task.trace_path),
                "log_path": str(task.log_path),
                "elapsed_seconds": time.monotonic() - started,
                "error": "timeout",
            }
    elapsed = time.monotonic() - started
    task_data = asdict(task)
    task_data["checkpoint_path"] = str(task.checkpoint_path)
    if task.cross_trace_path is not None:
        task_data["cross_trace_path"] = str(task.cross_trace_path)
    result = {
        **task_data,
        "trace_path": str(task.trace_path),
        "log_path": str(task.log_path),
        "elapsed_seconds": elapsed,
        "return_code": return_code,
        "trace_bytes": task.trace_path.stat().st_size if task.trace_path.exists() else 0,
        "launch_trace_path": str(launch_trace_path),
        "launch_trace_bytes": (
            launch_trace_path.stat().st_size if launch_trace_path.exists() else 0
        ),
        "cross_trace_bytes": (
            task.cross_trace_path.stat().st_size
            if task.cross_trace_path is not None and task.cross_trace_path.exists()
            else 0
        ),
    }
    if return_code != 0 or not task.trace_path.is_file():
        result["error"] = f"exit={return_code}, trace={task.trace_path.exists()}"
    else:
        final_summary = None
        for line in task.log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("FINAL_SUMMARY "):
                final_summary = json.loads(line.removeprefix("FINAL_SUMMARY "))
        if final_summary is None:
            result["error"] = "missing FINAL_SUMMARY"
        else:
            final_outcomes = final_summary.get("outcomes", {})
            result["outcome_counts"] = annotate_runtime_trace_outcomes(
                task.trace_path,
                final_summary,
                alive_at_end_terminal=(
                    "timeout" if args.reward_mode == "timely_arrival" else "incomplete"
                ),
            )
            result["score"] = final_summary.get("score")
            result["outcomes"] = {
                key: value
                for key, value in final_outcomes.items()
                if key not in {"definitions", "per_platform", "by_type"}
            }
            per_platform = final_outcomes.get("per_platform", {})
            result["outcomes"]["sampled_damage_zone_100"] = sum(
                1
                for outcome in per_platform.values()
                if isinstance(outcome, dict)
                and outcome.get("minimum_target_distance_m") is not None
                and float(outcome["minimum_target_distance_m"]) <= 100.0
            )
            if result["outcome_counts"].get("missing_outcome", 0):
                result["error"] = (
                    "MissionOutcomeTracker has no label for "
                    f"{result['outcome_counts']['missing_outcome']} trajectories"
                )
    return result


def scalar_metrics(values: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def aggregate_actual_outcomes(
    worker_results: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate the exact evaluator counters used for model acceptance."""

    keys = (
        "launched",
        "engaged",
        "engaged_resolved",
        "evaded",
        "intercepted",
        "arrived",
        "effective_hits",
        "alive_at_end",
        "sampled_damage_zone_100",
    )
    counts = {
        key: float(
            sum(float(item.get("outcomes", {}).get(key, 0.0)) for item in worker_results)
        )
        for key in keys
    }
    scores = [
        float(item["score"]["score"])
        for item in worker_results
        if isinstance(item.get("score"), dict) and "score" in item["score"]
    ]
    return {
        **counts,
        "games": float(len(worker_results)),
        "score_mean": float(np.mean(scores)) if scores else 0.0,
        "evasion_rate": counts["evaded"] / max(counts["engaged_resolved"], 1.0),
        "arrival_rate": counts["arrived"] / max(counts["launched"], 1.0),
        "hit_rate": counts["effective_hits"] / max(counts["arrived"], 1.0),
        "hit_rate_per_launch": (
            counts["effective_hits"] / max(counts["launched"], 1.0)
        ),
        "sampled_damage_zone_100_rate": (
            counts["sampled_damage_zone_100"] / max(counts["launched"], 1.0)
        ),
    }


BLUE_POLICY_CHOICES = {
    "b0": "b0_fixed_ratio_random",
    "b3": "b3_min_cost_assignment",
    "b7": "b7_planned_nearest",
    "b9": "b9_min_cost_planned_multi_wave",
}
NEWLINE = "\n"


def tier_of(scenario: str) -> str:
    for tier in ("easy", "medium", "hard"):
        if f"/{tier}/" in scenario:
            return tier
    return "unknown"


def discover_tier_maps(tier: str) -> list[str]:
    root = ROOT / "scenarios" / "cases" / "launch20" / tier
    if not root.is_dir():
        return []
    return sorted(
        f"launch20/{tier}/{path.name}"
        for path in root.iterdir()
        if (path / "scenario.json").is_file()
    )


def resolve_scenario_pools(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Resolve the training pool, optionally holding out one whole tier."""

    if args.tier_holdout == "none":
        scenarios = (
            [value.strip() for value in args.scenarios.split(",") if value.strip()]
            if args.scenarios
            else [args.scenario]
        )
        heldout_scenarios = [
            value.strip() for value in args.heldout_scenarios.split(",") if value.strip()
        ]
        return scenarios, heldout_scenarios
    heldout_scenarios = discover_tier_maps(args.tier_holdout)
    if not heldout_scenarios:
        raise FileNotFoundError(
            f"no maps found for tier {args.tier_holdout!r} under "
            f"{ROOT / 'scenarios' / 'cases' / 'launch20' / args.tier_holdout}"
        )
    scenarios = [
        map_path
        for tier in ("easy", "medium", "hard")
        if tier != args.tier_holdout
        for map_path in discover_tier_maps(tier)
    ]
    if not scenarios:
        raise ValueError("tier hold-out left the training pool empty")
    return scenarios, heldout_scenarios


def apply_ppo_tricks(policy: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Turn the requested update-side switches on; return what was applied."""

    config = policy.config
    applied: dict[str, Any] = {}
    if args.target_kl is not None:
        config.target_kl = float(args.target_kl)
        applied["target_kl"] = config.target_kl
    if args.clip_range_value is not None:
        config.clip_range_value = float(args.clip_range_value)
        applied["clip_range_value"] = config.clip_range_value
    if args.dual_clip is not None:
        config.dual_clip = float(args.dual_clip)
        applied["dual_clip"] = config.dual_clip
    if args.reward_clip is not None:
        config.reward_clip = float(args.reward_clip)
        applied["reward_clip"] = config.reward_clip
    if args.advantage_standardize:
        config.advantage_standardize = True
        applied["advantage_standardize"] = True
    return applied


def aggregate_by_tier(worker_results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """The acceptance counters, split per map tier."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in worker_results:
        grouped.setdefault(tier_of(str(item.get("scenario", ""))), []).append(item)
    return {
        tier: aggregate_actual_outcomes(items)
        for tier, items in sorted(grouped.items())
    }


def parse_eval_cells(spec: str) -> list[tuple[str, str, str]]:
    cells: list[tuple[str, str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        map_id, _, blue = item.partition("/")
        tier = {"E": "easy", "M": "medium", "H": "hard"}.get(map_id[:1].upper())
        if tier is None or not blue:
            raise ValueError(f"malformed eval cell {item!r}; expected e.g. E01/b0")
        cells.append(
            (
                f"{map_id}_{blue}",
                f"launch20/{tier}/{map_id}",
                BLUE_POLICY_CHOICES.get(blue, blue),
            )
        )
    return cells


def run_eval_cell(
    args: argparse.Namespace,
    cell_name: str,
    scenario: str,
    blue_policy: str,
    checkpoint_path: Path,
    iteration: int,
) -> dict[str, float]:
    log_dir = args.results_root / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"iteration_{iteration:04d}_{cell_name}.log"
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        "run",
        "--scenario", scenario,
        "--blue-policy", blue_policy,
        "--red-policy", "r9_hierarchical_learning",
        "--red-motion-policy", "local_avoid_s3",
        "--red-learning-model", str(Path(checkpoint_path).resolve()),
        "--seed", str(args.eval_seed),
        "--run-id", f"{args.run_name}_eval_i{iteration:04d}_{cell_name}",
        "--",
        "--total-rounds", "1",
        "--max-steps", str(args.max_steps),
        "--render-mode", "none",
        "--disable-log-color",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "BLUE_OBJECTIVE_MOTION": args.objective_motion,
            "LOCAL_AVOID_GATE_MODE": args.gate_mode,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=args.eval_timeout,
            check=False,
        )
    summary = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("FINAL_SUMMARY "):
            summary = json.loads(line.removeprefix("FINAL_SUMMARY "))
    seconds = time.monotonic() - started
    if completed.returncode != 0 or summary is None:
        print(f"EVAL {cell_name} failed exit={completed.returncode}", flush=True)
        return {"score": float("nan"), "seconds": seconds, "failed": 1.0}
    score = summary.get("score")
    if isinstance(score, dict):
        score = score.get("score", 0.0)
    outcomes = summary.get("outcomes", {})
    return {
        "score": float(score or 0.0),
        "arrived": float(outcomes.get("arrived", 0.0)),
        "intercepted": float(outcomes.get("intercepted", 0.0)),
        "evaded": float(outcomes.get("evaded", 0.0)),
        "seconds": seconds,
        "failed": 0.0,
    }


def evaluate_checkpoint(
    args: argparse.Namespace,
    checkpoint_path: Path,
    iteration: int,
    cells_raw: str | None = None,
) -> dict[str, float]:
    cells = parse_eval_cells(cells_raw or args.eval_cells)
    metrics: dict[str, float] = {}
    scores: list[float] = []
    with ThreadPoolExecutor(max_workers=max(1, args.eval_jobs)) as executor:
        futures = {
            executor.submit(
                run_eval_cell, args, name, scenario, blue, checkpoint_path, iteration
            ): name
            for name, scenario, blue in cells
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - report and continue
                print(f"EVAL {name} raised {error}", flush=True)
                result = {"score": float("nan"), "failed": 1.0}
            for key, value in result.items():
                metrics[f"{name}/{key}"] = float(value)
            if np.isfinite(result.get("score", float("nan"))):
                scores.append(float(result["score"]))
            print(f"EVAL {name} score={result.get('score')}", flush=True)
    metrics["mean_score"] = float(np.mean(scores)) if scores else float("nan")
    return metrics


def tensorboard_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter(log_dir=str(log_dir))


def main() -> int:
    args = parse_args()
    args.input_model = args.input_model.resolve()
    args.output_model = args.output_model.resolve()
    args.results_root = args.results_root.resolve()
    scenarios, heldout_scenarios = resolve_scenario_pools(args)
    overlap = sorted(set(scenarios) & set(heldout_scenarios))
    if overlap:
        raise ValueError(f"training and heldout scenarios overlap: {overlap}")
    for scenario in [*scenarios, *heldout_scenarios]:
        scenario_path = ROOT / "scenarios" / "cases" / scenario / "scenario.json"
        if not scenario_path.is_file():
            raise FileNotFoundError(scenario_path)
    if args.initial_trace_dir is not None:
        args.initial_trace_dir = args.initial_trace_dir.resolve()
    if not args.input_model.is_file():
        raise FileNotFoundError(args.input_model)
    if args.initial_trace_dir is not None:
        reused_traces = sorted(args.initial_trace_dir.glob("*.jsonl.gz"))
        if len(reused_traces) != args.games:
            raise ValueError(
                f"initial trace directory must contain exactly {args.games} traces; "
                f"found {len(reused_traces)} in {args.initial_trace_dir}"
            )
    if args.output_model.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing output checkpoint: {args.output_model}"
        )
    args.runtime = select_runtime(args.runtime)

    policy, environment_config, checkpoint_payload = load_full_sim_policy(
        args.input_model, device=args.device
    )
    observation_contract = policy_observation_contract(policy)
    if (
        args.observation_contract != "auto"
        and observation_contract["name"] != args.observation_contract
    ):
        raise ValueError(
            f"input checkpoint uses {observation_contract['name']!r}, expected "
            f"{args.observation_contract!r}"
        )
    if not args.smoke and policy.device.type != "cuda":
        raise RuntimeError(
            "formal full-simulator PPO requires a CUDA learner; pass "
            "--device cuda:N from a host context with GPU access"
        )
    if args.learning_rate is not None:
        policy.config.learning_rate = float(args.learning_rate)
        for group in policy.optimizer.param_groups:
            group["lr"] = float(args.learning_rate)
    if args.ppo_epochs is not None:
        policy.config.ppo_epochs = int(args.ppo_epochs)
    if args.minibatch_size is not None:
        policy.config.minibatch_size = int(args.minibatch_size)
    # Only override the checkpoint's discount factors when explicitly asked:
    # the trainer defaults used to silently replace the R3 recipe (0.99/0.95)
    # with 0.999/0.97, which changes the PPO objective without any log line.
    if args.gamma is not None:
        policy.config.gamma = float(args.gamma)
    if args.gae_lambda is not None:
        policy.config.gae_lambda = float(args.gae_lambda)
    if args.entropy_coef is not None:
        policy.config.entropy_coef = float(args.entropy_coef)
    applied_tricks = apply_ppo_tricks(policy, args)
    base_learning_rate = float(policy.config.learning_rate)
    base_entropy_coef = float(policy.config.entropy_coef)

    # V10 changes both the reward definition and the rollout distribution.
    # Keep the learned V9 network/observation statistics, but do not carry
    # Adam moments or reward-scale statistics from the old objective.
    policy.optimizer = torch.optim.Adam(
        (
            parameter
            for parameter in policy.network.parameters()
            if parameter.requires_grad
        ),
        lr=float(policy.config.learning_rate),
    )
    policy.reward_rms.mean = np.zeros_like(policy.reward_rms.mean)
    policy.reward_rms.var = np.ones_like(policy.reward_rms.var)
    policy.reward_rms.count = 1e-4
    policy.discounted_reward = np.zeros_like(policy.discounted_reward)

    reward_config = {
        "progress_scale": float(environment_config.progress_scale),
        "progress_fraction_scale": float(environment_config.progress_fraction_scale),
        "hit_penalty": float(environment_config.hit_penalty),
        "evade_bonus": float(environment_config.evade_bonus),
        "arrival_bonus": float(environment_config.arrival_bonus),
        "timeout_penalty": float(environment_config.timeout_penalty),
        "deviation_penalty": float(environment_config.deviation_penalty),
        "oscillation_penalty": float(environment_config.oscillation_penalty),
        "action_switch_penalty": float(environment_config.action_switch_penalty),
        "turn_reversal_penalty": float(environment_config.turn_reversal_penalty),
        "time_penalty": float(environment_config.time_penalty),
        "recovery_penalty": float(environment_config.recovery_penalty),
        "heading_error_penalty": float(environment_config.heading_error_penalty),
        "maneuver_burst_limit_steps": int(
            environment_config.maneuver_burst_limit_steps
        ),
        "excess_maneuver_penalty": float(
            environment_config.excess_maneuver_penalty
        ),
        "recovery_bonus": float(environment_config.recovery_bonus),
        "safety_improvement_scale": float(
            environment_config.safety_improvement_scale
        ),
        "unsafe_straight_penalty": float(
            environment_config.unsafe_straight_penalty
        ),
        "arrival_radius_m": float(environment_config.arrival_radius_m),
        "reward_clip": float(environment_config.reward_clip),
        "normalize_reward": bool(policy.config.normalize_reward),
    }
    expected_v8 = {
        "progress_scale": 0.0,
        "progress_fraction_scale": 1.0,
        "hit_penalty": 2.0,
        "evade_bonus": 0.0,
        "arrival_bonus": 5.0,
        "timeout_penalty": 1.0,
        "deviation_penalty": 0.0,
        "oscillation_penalty": 0.0,
        "action_switch_penalty": 0.002,
        "turn_reversal_penalty": 0.01,
        "time_penalty": 0.0,
        "recovery_penalty": 0.0,
        "heading_error_penalty": 0.0,
        "maneuver_burst_limit_steps": 0,
        "excess_maneuver_penalty": 0.0,
        "recovery_bonus": 0.0,
        "safety_improvement_scale": 0.0,
        "unsafe_straight_penalty": 0.0,
        "arrival_radius_m": 100.0,
        "reward_clip": 10.0,
        "normalize_reward": True,
    }
    if args.reward_mode in {
        "success_conditioned",
        "score_weighted",
        "progress_plus_score",
    } and (
        reward_config != expected_v8
    ):
        raise ValueError(
            "input checkpoint does not carry the exact V8 static-minimal reward: "
            f"{reward_config}"
        )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = args.results_root / f"{args.run_name}_{timestamp}"
    traces_dir = run_dir / "traces"
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=False)
    history_path = run_dir / "training_history.jsonl"
    status_path = run_dir / "run_status.json"
    reward_name = {
        "success_conditioned": "v10_real_success_conditioned_target_distance_v2",
        "normalized_distance": "normalized_target_distance_reduction_only_v2",
        "distance_outcome": "normalized_target_distance_plus_real_outcome_v1",
        "timely_arrival": "normalized_target_distance_timely_arrival_v1",
        "score_weighted": "success_conditioned_x_official_objective_weight_v1",
        "progress_plus_score": "success_conditioned_plus_weighted_hit_bonus_v1",
    }[args.reward_mode]
    distance_outcome = args.reward_mode == "distance_outcome"
    timely_arrival = args.reward_mode == "timely_arrival"
    score_weighted = args.reward_mode == "score_weighted"
    score_bonus_mode = args.reward_mode == "progress_plus_score"
    effective_reward_config = {
        "normalized_target_distance_reduction": 1.0,
        "terminal_distance_source": "runtime_terminal_goal_distance_m",
        "success_conditioned": args.reward_mode in {
            "success_conditioned",
            "score_weighted",
            "progress_plus_score",
        },
        "score_bonus_coef": float(args.score_bonus_coef) if score_bonus_mode else None,
        "score_bonus_scale": ("coef * objective_weight / 5.0" if score_bonus_mode else None),
        "mission_objective_weight_scale": (
            "official_judge_weight_target5_site2_ship1_clip_0.25_8"
            if score_weighted
            else None
        ),
        "intercepted_or_lost_trajectory_return": (
            0.0
            if args.reward_mode
            in {"success_conditioned", "score_weighted", "progress_plus_score"}
            else (
                "intercepted=0.20*distance_progress,lost_or_timeout=0"
                if timely_arrival
                else "distance_progress"
            )
        ),
        "terminal_source": "mission_outcome_tracker",
        "proxy_terminals_allowed": False,
        "terminal_rewards_mutually_exclusive": distance_outcome or timely_arrival,
        "effective_hit_bonus": 0.35 if distance_outcome else 0.0,
        "arrival_without_effective_hit_bonus": 0.20 if distance_outcome else 0.0,
        "interception_penalty": (
            -0.25 if distance_outcome else (-0.15 if timely_arrival else 0.0)
        ),
        "lost_or_timeout_terminal_reward": -0.30 if timely_arrival else 0.0,
        "timeout_penalty": -0.30 if timely_arrival else 0.0,
        "arrival_and_effective_hit_equivalent": timely_arrival,
        "timely_arrival_bonus": 0.10 if timely_arrival else 0.0,
        "intercepted_progress_scale": 0.20 if timely_arrival else 1.0,
        "lost_timeout_progress_scale": 0.0 if timely_arrival else 1.0,
        "time_cost_budget_at_horizon": -0.05 if timely_arrival else 0.0,
        "official_arrival_tolerance_m": 500.0,
        "sampled_damage_zone_diagnostic_m": 100.0,
        "action_switch_penalty": 0.0,
        "turn_reversal_penalty": 0.0,
    }
    training_tiers = [
        name
        for name in ("easy", "medium", "hard")
        if any(f"/{name}/" in scenario for scenario in scenarios)
    ]
    protocol = {
        "status": "running",
        "pipeline": "complete_simulator_local_avoid_ppo_v6",
        "policy_name": args.policy_name,
        "source_algorithm": checkpoint_payload.get("algorithm"),
        "observation_contract": observation_contract,
        "smoke": bool(args.smoke),
        "input_model": str(args.input_model),
        "input_sha256": fingerprint(args.input_model),
        "output_model": str(args.output_model),
        "scenario": args.scenario,
        "scenarios": scenarios,
        "heldout_scenarios": heldout_scenarios,
        "heldout_seeds": list(DEFAULT_V10_HOLDOUT_SEEDS),
        "map_split": f"{len(scenarios)}_train_{len(heldout_scenarios)}_strict_holdout",
        "tier_holdout": args.tier_holdout,
        "applied_tricks": applied_tricks,
        "acceptance_baselines": ["straight", "input_checkpoint"],
        "acceptance_metrics": (
            ["arrival_rate", "evasion_rate", "sampled_damage_zone_100_rate"]
            if timely_arrival
            else ["arrival_rate", "evasion_rate", "hit_rate_per_launch"]
        ),
        "map_sampling": "rotating_tier_balanced_" + "_".join(training_tiers),
        "blue_policy": args.blue_policy,
        "objective_motion": args.objective_motion,
        "gate_mode": args.gate_mode,
        "runtime": args.runtime,
        "learner_device": str(policy.device),
        "workers": args.workers,
        "games_per_iteration": args.games,
        "iterations": args.iterations,
        "max_steps": args.max_steps,
        "reward": reward_name,
        "reward_mode": args.reward_mode,
        "source_checkpoint_reward_config": reward_config,
        "effective_reward_config": effective_reward_config,
        "ppo_config": {
            "learning_rate": float(policy.config.learning_rate),
            "ppo_epochs": int(policy.config.ppo_epochs),
            "minibatch_size": int(policy.config.minibatch_size),
            "gamma": float(policy.config.gamma),
            "gae_lambda": float(policy.config.gae_lambda),
            "clip_range": float(policy.config.clip_range),
            "value_coef": float(policy.config.value_coef),
            "entropy_coef": float(policy.config.entropy_coef),
        },
        "behavior_batch_size": args.behavior_batch_size,
        "initial_trace_dir": (
            str(args.initial_trace_dir) if args.initial_trace_dir is not None else None
        ),
        "type_sampling": "balanced_h_m_l_median_resample",
        "incomplete_rollouts_used": bool(
            args.smoke and args.reward_mode != "timely_arrival"
        ),
        "smoke_incomplete_note": (
            (
                "completed smoke horizon is a penalized timeout under timely_arrival"
                if timely_arrival
                else "plumbing-only time-limit bootstrap; formal training excludes incomplete"
            )
            if args.smoke
            else None
        ),
        "v8_constraint": "none",
        "optimizer_state": "reset_for_v10_objective",
        "reward_normalizer_state": "reset_for_v10_objective",
        "micro_simulator_samples": 0,
        "created_at": timestamp,
    }
    atomic_json(run_dir / "protocol.json", protocol)
    atomic_json(status_path, protocol)
    append_jsonl(history_path, {"event": "start", **protocol})

    writer = tensorboard_writer(run_dir / "tensorboard")
    iteration_summaries: list[dict[str, Any]] = []
    current_checkpoint = args.input_model
    base_policy_version = int(
        checkpoint_payload.get("metrics", {}).get("policy_version", 0)
    )
    for iteration in range(1, args.iterations + 1):
        iteration_trace_dir = traces_dir / f"iteration_{iteration:04d}"
        iteration_log_dir = logs_dir / f"iteration_{iteration:04d}"
        tasks = []
        worker_results: list[dict[str, Any]] = []
        if iteration == 1 and args.initial_trace_dir is not None:
            source_run_dir = args.initial_trace_dir.parent.parent
            source_workers_path = source_run_dir / "workers_iteration_0001.json"
            if not source_workers_path.is_file():
                raise FileNotFoundError(source_workers_path)
            source_workers = json.loads(
                source_workers_path.read_text(encoding="utf-8")
            ).get("workers", [])
            if len(source_workers) != args.games:
                raise ValueError(
                    f"reused worker manifest has {len(source_workers)} workers; "
                    f"expected {args.games}"
                )
            worker_results = [
                {**item, "iteration": iteration, "reused": True}
                for item in source_workers
            ]
            print(
                f"ITERATION {iteration}/{args.iterations} REUSE "
                f"traces={args.initial_trace_dir}",
                flush=True,
            )
            for completed, result in enumerate(worker_results, start=1):
                append_jsonl(
                    history_path,
                    {"event": "worker", "completed": completed, **result},
                )
        else:
            iteration_scenarios = scenario_schedule(
                scenarios, args.games, iteration
            )
            for index in range(args.games):
                seed = args.seed_start + (iteration - 1) * args.games + index
                scenario = iteration_scenarios[index]
                map_id = scenario.rsplit("/", 1)[-1]
                tasks.append(
                    WorkerTask(
                        iteration=iteration,
                        worker_index=index % args.workers,
                        seed=seed,
                        checkpoint_path=current_checkpoint,
                        trace_path=iteration_trace_dir / f"{map_id}_seed_{seed:05d}.jsonl.gz",
                        log_path=iteration_log_dir / f"{map_id}_seed_{seed:05d}.log",
                        run_id=f"{args.run_name}_i{iteration:04d}_{map_id}_s{seed:05d}",
                        scenario=scenario,
                    )
                )
            print(
                f"ITERATION {iteration}/{args.iterations} COLLECT "
                f"checkpoint={current_checkpoint}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(run_worker, args, task): task for task in tasks
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    worker_results.append(result)
                    append_jsonl(
                        history_path,
                        {"event": "worker", "completed": completed, **result},
                    )
                    print(
                        f"WORKER {completed}/{len(tasks)} seed={result['seed']} "
                        f"status={'error' if 'error' in result else 'complete'}",
                        flush=True,
                    )
        worker_results.sort(key=lambda item: int(item["seed"]))
        atomic_json(
            run_dir / f"workers_iteration_{iteration:04d}.json",
            {"workers": worker_results},
        )
        failures = [item for item in worker_results if "error" in item]
        if failures:
            failed = {
                **protocol,
                "status": "failed",
                "failed_iteration": iteration,
                "completed_iterations": iteration_summaries,
                "failures": failures,
            }
            atomic_json(status_path, failed)
            if writer is not None:
                writer.close()
            raise RuntimeError(
                f"{len(failures)} complete-simulator workers failed in "
                f"iteration {iteration}"
            )

        trace_paths = [Path(item["trace_path"]) for item in worker_results]
        trajectories = load_runtime_trajectories(trace_paths)
        rollout, rollout_stats = build_full_sim_rollout(
            policy,
            trajectories,
            environment_config=environment_config,
            behavior_batch_size=args.behavior_batch_size,
            include_incomplete=bool(args.smoke),
            require_real_outcomes=True,
            balance_types=True,
            balance_seed=args.seed_start + iteration,
            reward_mode=args.reward_mode,
            reward_horizon_steps=args.max_steps,
            score_bonus_coef=args.score_bonus_coef,
        )
        before = {
            name: value.detach().cpu().clone()
            for name, value in policy.network.state_dict().items()
        }
        if args.anneal_lr and args.iterations > 1:
            progress = (iteration - 1) / float(args.iterations - 1)
            scale = 1.0 - (1.0 - args.anneal_final_scale) * progress
            policy.config.learning_rate = base_learning_rate * scale
            for group in policy.optimizer.param_groups:
                group["lr"] = base_learning_rate * scale
            if args.entropy_coef_final is not None:
                policy.config.entropy_coef = base_entropy_coef + (
                    args.entropy_coef_final - base_entropy_coef
                ) * progress
        update_metrics = policy.update(rollout, expert_bc_coef=0.0)
        policy.total_steps += int(rollout_stats["samples"])
        changed_parameters = sum(
            int(not torch.equal(before[name], value.detach().cpu()))
            for name, value in policy.network.state_dict().items()
        )
        if changed_parameters <= 0:
            raise RuntimeError("PPO update completed without changing any parameter")
        if not all(
            np.isfinite(value) for value in scalar_metrics(update_metrics).values()
        ):
            raise RuntimeError("PPO update produced a non-finite metric")

        actual_outcomes = aggregate_actual_outcomes(worker_results)
        metrics = {
            "training_source": "complete_simulator_only",
            "reward_version": reward_name,
            "reward_mode": args.reward_mode,
            "micro_simulator_samples": 0,
            "deployment_gate_mode": "none",
            "objective_motion": args.objective_motion,
            "scenarios": scenarios,
            "blue_policy": args.blue_policy,
            "policy_version": base_policy_version + iteration,
            "full_sim_iteration": iteration,
            "rollout_samples": int(rollout_stats["samples"]),
            "optimized_samples": int(len(rollout["actions"])),
            "changed_parameter_tensors": int(changed_parameters),
            "source_checkpoint_sha256": protocol["input_sha256"],
            "source_algorithm": protocol["source_algorithm"],
            "observation_contract": observation_contract,
            "pipeline_smoke": bool(args.smoke),
            "incomplete_rollouts_used": bool(
                args.smoke and args.reward_mode != "timely_arrival"
            ),
            "terminal_source": "mission_outcome_tracker",
            "proxy_terminals_allowed": False,
            "type_balancing": "median_resample_h_m_l",
            "v8_constraint": "none",
        }
        policy.save(
            args.output_model,
            environment_config=environment_config,
            metrics=metrics,
        )
        snapshot_path = run_dir / "checkpoints" / f"iteration_{iteration:04d}.pt"
        policy.save(
            snapshot_path,
            environment_config=environment_config,
            metrics=metrics,
        )
        reloaded, _reloaded_environment, reloaded_payload = load_full_sim_policy(
            args.output_model, device="cpu"
        )
        finite_tensors = all(
            bool(torch.isfinite(value).all())
            for value in reloaded.network.state_dict().values()
        )
        if not finite_tensors:
            raise RuntimeError("reloaded checkpoint contains non-finite parameters")

        step = int(policy.total_steps)
        mean_worker_seconds = float(
            np.mean([item["elapsed_seconds"] for item in worker_results])
        )
        if writer is not None:
            for key, value in scalar_metrics(rollout_stats).items():
                writer.add_scalar(f"full_sim_rollout/{key}", value, step)
            for key, value in scalar_metrics(update_metrics).items():
                writer.add_scalar(f"ppo/{key}", value, step)
            for key, value in scalar_metrics(actual_outcomes).items():
                writer.add_scalar(f"actual_outcomes/{key}", value, step)
            writer.add_scalar(
                "pipeline/changed_parameter_tensors", changed_parameters, step
            )
            writer.add_scalar(
                "pipeline/mean_worker_seconds", mean_worker_seconds, step
            )
            writer.add_scalar(
                "pipeline/learning_rate", float(policy.config.learning_rate), step
            )
            writer.add_scalar(
                "pipeline/entropy_coef", float(policy.config.entropy_coef), step
            )
            for tier_name, tier_stats in aggregate_by_tier(worker_results).items():
                for key, value in scalar_metrics(tier_stats).items():
                    writer.add_scalar(f"tier_{tier_name}/{key}", value, step)
            writer.flush()

        eval_metrics: dict[str, float] = {}
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            eval_metrics = evaluate_checkpoint(args, args.output_model, iteration)
            if writer is not None:
                for key, value in eval_metrics.items():
                    if np.isfinite(value):
                        writer.add_scalar(f"eval/{key}", value, step)
                writer.flush()
            print(
                f"EVAL mean_score={eval_metrics.get('mean_score')}",
                flush=True,
            )

        iteration_summary = {
            "iteration": iteration,
            "step": step,
            "checkpoint": str(snapshot_path),
            "output_sha256": fingerprint(args.output_model),
            "worker_results": worker_results,
            "rollout": rollout_stats,
            "actual_outcomes": actual_outcomes,
            "ppo": scalar_metrics(update_metrics),
            "changed_parameter_tensors": changed_parameters,
            "reload_valid": finite_tensors,
            "saved_policy_version": reloaded_payload.get("metrics", {}).get(
                "policy_version"
            ),
            "mean_worker_seconds": mean_worker_seconds,
        }
        iteration_summary["eval"] = eval_metrics
        iteration_summary["applied_tricks"] = applied_tricks
        iteration_summaries.append(iteration_summary)
        append_jsonl(history_path, {"event": "iteration", **iteration_summary})
        atomic_json(
            status_path,
            {
                **protocol,
                "status": "running",
                "current_iteration": iteration,
                "latest": iteration_summary,
            },
        )
        current_checkpoint = args.output_model
        print(
            f"ITERATION_COMPLETE {iteration}/{args.iterations} "
            f"samples={len(rollout['actions'])} step={step}",
            flush=True,
        )

    final_eval_metrics: dict[str, float] = {}
    if str(args.eval_final_cells).strip() and not args.smoke:
        final_eval_metrics = evaluate_checkpoint(
            args,
            args.output_model,
            args.iterations,
            cells_raw=args.eval_final_cells,
        )
        if writer is not None:
            for key, value in final_eval_metrics.items():
                if np.isfinite(value):
                    writer.add_scalar(f"eval_final/{key}", value, step)
        print(
            f"EVAL_FINAL mean_score={final_eval_metrics.get('mean_score')}",
            flush=True,
        )

    if writer is not None:
        writer.close()

    final = {
        **protocol,
        "status": "complete",
        "run_dir": str(run_dir),
        "output_model": str(args.output_model),
        "output_sha256": fingerprint(args.output_model),
        "device": str(policy.device),
        "completed_iterations": len(iteration_summaries),
        "iterations_summary": iteration_summaries,
        "final_eval": final_eval_metrics,
        "tensorboard": str(run_dir / "tensorboard"),
    }
    atomic_json(run_dir / "pipeline_summary.json", final)
    atomic_json(status_path, final)
    append_jsonl(history_path, {"event": "complete", **final})
    print(f"PIPELINE_COMPLETE {run_dir}", flush=True)
    print(f"MODEL {args.output_model}", flush=True)
    print(f"SAMPLES {len(rollout['actions'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
