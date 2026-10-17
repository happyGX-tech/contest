#!/usr/bin/env python3
"""Evaluate Trans-V8 against the V13 baseline in the full simulator.

The default protocol keeps the R9 upper policy, map, blue policy, seed and
objective-motion mode paired between the two local-avoid checkpoints.  Every
game runs in its own process and has a wall-clock timeout so a stalled native
simulator cannot block the whole matrix indefinitely.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

from evaluate_motion_matrix import BLUE_POLICIES, Task, load_maps, summarize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V8 = ROOT / "models/local_avoid_set_attention_v8_static_minimal_r2_best.pt"
DEFAULT_V13 = ROOT / "models/local_avoid_ppo_v13_gate_aligned_s3_best.pt"
DEFAULT_MAPS = "E01,M03,H07"
DEFAULT_BLUE_POLICIES = (
    "b0_fixed_ratio_random,b3_min_cost_assignment,"
    "b7_planned_nearest,b9_min_cost_planned_multi_wave"
)
CONDITION_ORDER = ("v13", "v8")
MISSILE_TYPE_LABELS = {"21000": "H", "21001": "M", "21002": "L"}
DEFAULT_GATE_MODES = {"v13": "mission_v12", "v8": "none", "v9": "none"}
RED_UPPER_POLICY = "r9_hierarchical_learning"
RED_MOTION_POLICY = "local_avoid_s3"
CORE_DEFAULT_MAX_STEPS = 1000
CACHE_PROTOCOL_FILENAME = "evaluation_protocol.json"
CACHE_PROTOCOL_SCHEMA_VERSION = 2
ACCEPTED_TERMINATION_REASONS = ("environment_done", "time_limit")
SUMMARY_OUTCOME_COUNT_KEYS = (
    "launched",
    "engaged",
    "engaged_resolved",
    "engaged_censored",
    "evaded",
    "intercepted",
    "arrived",
    "effective_hits",
    "alive_at_end",
)
COMPETITION_ENV_HOST_ROOT = Path(
    "/home/ubuntu/yunpan/miniconda3/envs/competition"
)
COMPETITION_PYTHON_HOST = COMPETITION_ENV_HOST_ROOT / "bin" / "python"
COMPETITION_ENV_CONTAINER_ROOT = "/opt/competition-env"
COMPETITION_PYTHON_CONTAINER = f"{COMPETITION_ENV_CONTAINER_ROOT}/bin/python"

# The aggregate code identity deliberately covers the simulator entry point and
# all Python/native implementation files which can affect a game.  This is
# stronger than a Git commit ID in a dirty research checkout.
CODE_IDENTITY_GLOBS = (
    "run.py",
    "core/**/*.py",
    "core/**/*.so",
    "core/**/*.pyd",
    "policies/**/*.py",
    "scenarios/**/*.py",
    "scenarios/cases/**/*.py",
    "tools/evaluate_motion_matrix.py",
    "tools/evaluate_v8_vs_v13_full_sim.py",
)
OBSERVATION_IDENTITY_FILES = (
    "policies/red/skills/evasion_ppo/encoder.py",
    "policies/red/skills/evasion_ppo/local_runtime.py",
    "policies/red/skills/evasion_ppo/set_env.py",
    "policies/red/skills/evasion_ppo/set_runtime.py",
)


@dataclass(frozen=True)
class Condition:
    label: str
    checkpoint: Path
    gate_mode: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class EvaluationTask:
    condition: str
    checkpoint: str
    checkpoint_sha256: str
    gate_mode: str
    map_id: str
    tier: str
    scenario: str
    blue_policy: str
    seed: int
    scenario_sha256: str
    scenario_case_info_sha256: str | None
    max_steps: int

    def simulator_task(self) -> Task:
        return Task(
            self.map_id,
            self.tier,
            self.scenario,
            self.blue_policy,
            RED_MOTION_POLICY,
            self.seed,
        )


_ACTIVE_PROCESSES: dict[subprocess.Popen[Any], str | None] = {}
_ACTIVE_LOCK = threading.Lock()


class EvaluationCancelled(RuntimeError):
    """Raised by a worker which observes matrix cancellation before launch."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Optional explicit simulator horizon; omitted resolves the scenario "
            "competition limit (or the core default)."
        ),
    )
    parser.add_argument("--kill-grace-seconds", type=float, default=10.0)
    parser.add_argument(
        "--runtime",
        choices=("auto", "host", "docker"),
        default="auto",
        help="Execution runtime. Auto uses Docker when host glibc is older than 2.38.",
    )
    parser.add_argument(
        "--docker-image",
        default="docker.m.daocloud.io/library/python:3.11-slim-trixie",
    )
    parser.add_argument("--seeds", default="713")
    parser.add_argument("--maps", default=DEFAULT_MAPS)
    parser.add_argument("--blue-policies", default=DEFAULT_BLUE_POLICIES)
    parser.add_argument(
        "--objective-motion",
        choices=("static", "rule"),
        default="static",
        help="Blue objective motion. V8 was trained only with static objectives.",
    )
    parser.add_argument("--v8-checkpoint", type=Path, default=DEFAULT_V8)
    parser.add_argument("--v13-checkpoint", type=Path, default=DEFAULT_V13)
    parser.add_argument(
        "--v8-gate-mode",
        choices=("none", "legacy", "mission_v12"),
        default="none",
    )
    parser.add_argument(
        "--v13-gate-mode",
        choices=("none", "legacy", "mission_v12"),
        default="mission_v12",
    )
    parser.add_argument(
        "--run-prefix",
        help=(
            "Run-ID prefix. Motion, checkpoint, max_steps and protocol fingerprint "
            "are appended."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Report directory; defaults under results/evaluations_set_attention_v8.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run new games even when an exact protocol-sidecar cache hit exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the complete protocol without starting simulations.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not math.isfinite(args.kill_grace_seconds) or args.kill_grace_seconds < 0:
        parser.error("--kill-grace-seconds must be finite and non-negative")
    return args


def resolve_under_root(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_content_identity(root: Path, *, name: str) -> dict[str, Any]:
    """Hash every file/symlink in a mounted runtime directory.

    The evaluator mounts the complete competition conda environment, so the
    interpreter binary alone is not an adequate runtime identity.  Only the
    aggregate is retained in the protocol to avoid a very large JSON payload.
    """

    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise RuntimeError(f"Runtime environment is not a directory: {resolved_root}")
    aggregate = hashlib.sha256()
    file_count = 0
    symlink_count = 0
    total_bytes = 0
    for path in sorted(resolved_root.rglob("*"), key=lambda item: str(item)):
        relative = str(path.relative_to(resolved_root))
        if path.is_symlink():
            entry: dict[str, Any] = {
                "path": relative,
                "kind": "symlink",
                "target": os.readlink(path),
            }
            symlink_count += 1
            if path.exists() and path.resolve().is_file():
                entry["target_sha256"] = file_sha256(path)
                entry["target_size_bytes"] = path.stat().st_size
        elif path.is_file():
            size = path.stat().st_size
            entry = {
                "path": relative,
                "kind": "file",
                "sha256": file_sha256(path),
                "size_bytes": size,
            }
            file_count += 1
            total_bytes += size
        else:
            continue
        aggregate.update(canonical_json(entry).encode("utf-8"))
        aggregate.update(b"\n")
    return {
        "name": name,
        "algorithm": "sha256-content-and-relative-path",
        "sha256": aggregate.hexdigest(),
        "root": str(resolved_root),
        "file_count": file_count,
        "symlink_count": symlink_count,
        "total_file_bytes": total_bytes,
    }


def canonical_json(payload: Any) -> str:
    """Return the stable JSON representation used by protocol fingerprints."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def is_full_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def normalize_runtime_environment_identity(
    runtime: str,
    docker_image: str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Validate and copy an injected runtime identity without performing I/O."""

    normalized = json.loads(canonical_json(identity))
    if normalized.get("runtime") != runtime:
        raise ValueError("runtime environment identity does not match selected runtime")

    if runtime == "docker":
        if normalized.get("docker_image") != docker_image:
            raise ValueError("runtime environment identity has a different Docker image")
        image_id = normalized.get("docker_image_id")
        if (
            not isinstance(image_id, str)
            or not image_id.startswith("sha256:")
            or not is_full_sha256(image_id.removeprefix("sha256:"))
        ):
            raise ValueError("runtime environment identity has no full Docker image ID")
        repo_digests = normalized.get("docker_repo_digests")
        if not isinstance(repo_digests, list) or not all(
            isinstance(item, str) for item in repo_digests
        ):
            raise ValueError("runtime environment identity has invalid RepoDigests")
        if normalized.get("docker_run_reference") != image_id:
            raise ValueError("Docker run reference is not the inspected immutable image ID")
        environment_tree = normalized.get("competition_environment_tree")
        if not isinstance(environment_tree, dict) or not is_full_sha256(
            environment_tree.get("sha256")
        ):
            raise ValueError("mounted competition environment has no full tree SHA-256")
        executable = normalized.get("mounted_competition_python")
    elif runtime == "host":
        if not isinstance(normalized.get("platform"), dict):
            raise ValueError("host runtime environment identity has no platform")
        executable = normalized.get("python_executable")
    else:
        raise ValueError(f"Unsupported resolved runtime: {runtime}")

    if not isinstance(executable, dict) or not is_full_sha256(
        executable.get("sha256")
    ):
        raise ValueError("runtime Python executable has no full SHA-256")
    return normalized


def task_protocol_identity(
    task: EvaluationTask,
    *,
    objective_motion: str,
    runtime: str,
    docker_image: str,
    workers: int,
    timeout_seconds: float,
    kill_grace_seconds: float,
    runtime_environment_identity: dict[str, Any],
    code_identity: dict[str, Any],
    observation_encoding_identity: dict[str, Any],
    red_upper_policy: str = RED_UPPER_POLICY,
    red_motion_policy: str = RED_MOTION_POLICY,
) -> dict[str, Any]:
    """Build the complete, JSON-only identity of one simulator task.

    This function intentionally performs no file or process I/O.  Callers
    resolve hashes and step limits first, making the protocol identity easy to
    unit test and impossible to infer from a stale result directory.
    """

    for label, digest in (
        ("checkpoint", task.checkpoint_sha256),
        ("scenario", task.scenario_sha256),
    ):
        if not is_full_sha256(digest):
            raise ValueError(f"{label} SHA-256 must be a full lowercase digest")
    if task.max_steps <= 0:
        raise ValueError("resolved max_steps must be positive")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if not math.isfinite(kill_grace_seconds) or kill_grace_seconds < 0:
        raise ValueError("kill_grace_seconds must be finite and non-negative")

    # Canonical round-tripping copies and validates nested identity structures
    # without making this pure function depend on their concrete dict types.
    normalized_code_identity = json.loads(canonical_json(code_identity))
    normalized_observation_identity = json.loads(
        canonical_json(observation_encoding_identity)
    )
    normalized_runtime_identity = normalize_runtime_environment_identity(
        runtime,
        docker_image,
        runtime_environment_identity,
    )
    return {
        "schema_version": CACHE_PROTOCOL_SCHEMA_VERSION,
        "condition": task.condition,
        "checkpoint": {
            "path": task.checkpoint,
            "sha256": task.checkpoint_sha256,
        },
        "scenario": {
            "reference": task.scenario,
            "map_id": task.map_id,
            "tier": task.tier,
            "sha256": task.scenario_sha256,
            "case_info_sha256": task.scenario_case_info_sha256,
        },
        "objective_motion": objective_motion,
        "red_upper_policy": red_upper_policy,
        "red_motion_policy": red_motion_policy,
        "gate_mode": task.gate_mode,
        "blue_policy": task.blue_policy,
        "seed": int(task.seed),
        "max_steps": int(task.max_steps),
        "execution": {
            "workers": int(workers),
            "wall_clock_timeout_seconds": float(timeout_seconds),
            "kill_grace_seconds": float(kill_grace_seconds),
            "simulator_total_rounds": 1,
            "accepted_termination_reasons": list(ACCEPTED_TERMINATION_REASONS),
            "time_limit_requires_exact_max_steps": True,
        },
        "runtime": runtime,
        # Record the exact requested image string even for host runs.  Runtime
        # says whether it was used; retaining the string avoids an ambiguous
        # identity when the same report is later replayed under Docker.
        "docker_image": str(docker_image),
        "runtime_environment_identity": normalized_runtime_identity,
        "code_identity": normalized_code_identity,
        "observation_encoding_identity": normalized_observation_identity,
    }


def task_protocol_fingerprint(identity: dict[str, Any]) -> str:
    """Hash a task identity without consulting mutable external state."""

    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def task_identity_from_args(
    args: argparse.Namespace,
    task: EvaluationTask,
    runtime_environment_identity: dict[str, Any],
    code_identity: dict[str, Any],
    observation_encoding_identity: dict[str, Any],
) -> dict[str, Any]:
    """Build one task identity from the fully resolved matrix arguments."""

    return task_protocol_identity(
        task,
        objective_motion=args.objective_motion,
        runtime=args.runtime,
        docker_image=args.docker_image,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        kill_grace_seconds=args.kill_grace_seconds,
        runtime_environment_identity=runtime_environment_identity,
        code_identity=code_identity,
        observation_encoding_identity=observation_encoding_identity,
    )


def task_run_id(prefix: str, task: EvaluationTask, identity: dict[str, Any]) -> str:
    """Create a readable run ID with step limit and protocol fingerprint."""

    gate_tag = (
        ""
        if task.gate_mode == DEFAULT_GATE_MODES.get(task.condition)
        else f"_g{task.gate_mode}"
    )
    fingerprint = task_protocol_fingerprint(identity)
    return (
        f"{prefix}_{identity['objective_motion']}_{task.condition}{gate_tag}_"
        f"{task.checkpoint_sha256[:10]}_ms{task.max_steps}_"
        f"{task.map_id.lower()}_{task.blue_policy.split('_', 1)[0]}_"
        f"s{task.seed:04d}_pf{fingerprint[:12]}"
    )


def resolve_scenario_file(scenario: str) -> Path:
    """Resolve a scenario reference exactly as the repository run wrapper does."""

    if scenario == "default":
        return ROOT / "core" / "scenarios" / "platform.json"
    bundled = ROOT / "scenarios" / "cases" / scenario / "scenario.json"
    if bundled.is_file():
        return bundled.resolve()
    external = Path(scenario).expanduser()
    if external.is_file() and external.suffix.lower() == ".json":
        return external.resolve()
    raise ValueError(f"Unknown scenario: {scenario}")


def resolve_scenario_protocol(
    scenario: str,
    explicit_max_steps: int | None,
) -> tuple[str, str | None, int]:
    """Resolve scenario/case-info hashes and the actual simulator horizon."""

    scenario_path = resolve_scenario_file(scenario)
    scenario_digest = file_sha256(scenario_path)
    case_info_path = scenario_path.with_name("case_info.json")
    case_info_digest = file_sha256(case_info_path) if case_info_path.is_file() else None
    if explicit_max_steps is not None:
        resolved_max_steps = int(explicit_max_steps)
    elif case_info_path.is_file():
        case_info = json.loads(case_info_path.read_text(encoding="utf-8"))
        resolved_max_steps = int(case_info["max_steps"])
    else:
        resolved_max_steps = CORE_DEFAULT_MAX_STEPS
    if resolved_max_steps <= 0:
        raise ValueError(f"Resolved non-positive max_steps for {scenario}: {resolved_max_steps}")
    return scenario_digest, case_info_digest, resolved_max_steps


def files_identity(
    paths: Iterable[Path],
    *,
    name: str,
    scope: Iterable[str],
    include_file_hashes: bool = False,
) -> dict[str, Any]:
    """Build a deterministic content identity for a collection of files."""

    entries: dict[str, str] = {}
    for path in sorted({candidate.resolve() for candidate in paths}):
        relative = str(path.relative_to(ROOT))
        entries[relative] = file_sha256(path)
    aggregate = hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()
    identity: dict[str, Any] = {
        "name": name,
        "algorithm": "sha256",
        "sha256": aggregate,
        "file_count": len(entries),
        "scope": list(scope),
    }
    if include_file_hashes:
        identity["files"] = entries
    return identity


def current_source_identities() -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash the current simulator code and observation-encoding implementation."""

    code_paths: set[Path] = set()
    for pattern in CODE_IDENTITY_GLOBS:
        code_paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    code_identity = files_identity(
        code_paths,
        name="full_simulator_source_tree",
        scope=CODE_IDENTITY_GLOBS,
    )
    observation_paths = tuple(ROOT / relative for relative in OBSERVATION_IDENTITY_FILES)
    observation_identity = files_identity(
        observation_paths,
        name="local_avoid_runtime_observation_encoding",
        scope=OBSERVATION_IDENTITY_FILES,
        include_file_hashes=True,
    )
    return code_identity, observation_identity


def python_executable_identity(path: Path) -> dict[str, Any]:
    """Hash the exact interpreter binary used or mounted by the evaluator."""

    try:
        resolved_path = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"Python executable is unavailable: {path}") from error
    if not resolved_path.is_file():
        raise RuntimeError(f"Python executable is not a file: {resolved_path}")
    return {
        "configured_path": str(path),
        "resolved_path": str(resolved_path),
        "sha256": file_sha256(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
    }


def inspect_docker_image(image: str) -> dict[str, Any]:
    """Resolve a mutable Docker reference to its local immutable identity."""

    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Cannot inspect Docker image {image!r}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no detail"
        raise RuntimeError(
            f"Cannot inspect Docker image {image!r} (exit={completed.returncode}): "
            f"{detail}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Docker inspect returned invalid JSON for {image!r}") from error
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise RuntimeError(f"Docker inspect returned an unexpected payload for {image!r}")
    inspected = payload[0]
    image_id = inspected.get("Id")
    repo_digests = inspected.get("RepoDigests") or []
    if not isinstance(repo_digests, list):
        raise RuntimeError(f"Docker inspect returned invalid RepoDigests for {image!r}")
    return {
        "docker_image_id": image_id,
        "docker_repo_digests": sorted(str(item) for item in repo_digests),
    }


def resolve_runtime_environment_identity(
    runtime: str,
    docker_image: str,
) -> dict[str, Any]:
    """Resolve runtime facts once, before any simulator task can launch."""

    if runtime == "docker":
        docker_identity = inspect_docker_image(docker_image)
        identity: dict[str, Any] = {
            "runtime": "docker",
            "docker_image": docker_image,
            **docker_identity,
            "docker_run_reference": docker_identity["docker_image_id"],
            "competition_environment_mount": {
                "host_path": str(COMPETITION_ENV_HOST_ROOT),
                "container_path": COMPETITION_ENV_CONTAINER_ROOT,
                "read_only": True,
            },
            "competition_environment_tree": directory_content_identity(
                COMPETITION_ENV_HOST_ROOT,
                name="mounted_competition_conda_environment",
            ),
            "mounted_competition_python": {
                **python_executable_identity(COMPETITION_PYTHON_HOST),
                "container_path": COMPETITION_PYTHON_CONTAINER,
            },
        }
    elif runtime == "host":
        libc_name, libc_version = platform.libc_ver()
        identity = {
            "runtime": "host",
            "python_executable": python_executable_identity(Path(sys.executable)),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "libc_name": libc_name,
                "libc_version": libc_version,
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
            },
        }
    else:
        raise ValueError(f"Unsupported resolved runtime: {runtime}")
    return normalize_runtime_environment_identity(runtime, docker_image, identity)


def assert_task_files_unchanged(task: EvaluationTask) -> None:
    """Verify the checkpoint and scenario content captured for one task."""

    checkpoint = Path(task.checkpoint)
    if not checkpoint.is_file() or file_sha256(checkpoint) != task.checkpoint_sha256:
        raise RuntimeError(f"Checkpoint changed after protocol capture: {checkpoint}")
    scenario_sha, case_info_sha, resolved_steps = resolve_scenario_protocol(
        task.scenario,
        task.max_steps,
    )
    if (
        scenario_sha != task.scenario_sha256
        or case_info_sha != task.scenario_case_info_sha256
        or resolved_steps != task.max_steps
    ):
        raise RuntimeError(f"Scenario changed after protocol capture: {task.scenario}")


def assert_execution_inputs_unchanged(
    args: argparse.Namespace,
    task: EvaluationTask,
    runtime_environment_identity: dict[str, Any],
    code_identity: dict[str, Any],
    observation_encoding_identity: dict[str, Any],
) -> None:
    """Fail closed if any live input no longer matches the task fingerprint."""

    assert_task_files_unchanged(task)
    current_code, current_observation = current_source_identities()
    if canonical_json(current_code) != canonical_json(code_identity):
        raise RuntimeError("Simulator source tree changed after protocol capture")
    if canonical_json(current_observation) != canonical_json(
        observation_encoding_identity
    ):
        raise RuntimeError("Observation encoding changed after protocol capture")

    if args.runtime == "docker":
        current_docker = inspect_docker_image(args.docker_image)
        if current_docker["docker_image_id"] != runtime_environment_identity.get(
            "docker_image_id"
        ) or current_docker["docker_repo_digests"] != runtime_environment_identity.get(
            "docker_repo_digests"
        ):
            raise RuntimeError("Docker image reference changed after protocol capture")
        current_python = python_executable_identity(COMPETITION_PYTHON_HOST)
        expected_python = runtime_environment_identity.get(
            "mounted_competition_python", {}
        )
        if current_python["sha256"] != expected_python.get("sha256"):
            raise RuntimeError(
                "Mounted competition Python changed after protocol capture"
            )


def assert_runtime_environment_unchanged(
    args: argparse.Namespace,
    expected: dict[str, Any],
) -> None:
    """Re-hash the full mounted runtime once before accepting the matrix."""

    current = resolve_runtime_environment_identity(args.runtime, args.docker_image)
    if canonical_json(current) != canonical_json(expected):
        raise RuntimeError("Runtime environment changed during evaluation")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _summary_error(path: Path, detail: str) -> ValueError:
    return ValueError(f"Invalid simulator summary {path}: {detail}")


def _mapping(value: Any, *, path: Path, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _summary_error(path, f"{label} must be an object")
    return value


def _nonnegative_int(value: Any, *, path: Path, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _summary_error(path, f"{label} must be a non-negative integer")
    return value


def _finite_number(
    value: Any,
    *,
    path: Path,
    label: str,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _summary_error(path, f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        qualifier = "non-negative finite" if nonnegative else "finite"
        raise _summary_error(path, f"{label} must be a {qualifier} number")
    return number


def _rate(
    value: Any,
    numerator: int,
    denominator: int,
    *,
    path: Path,
    label: str,
) -> None:
    observed = _finite_number(value, path=path, label=label, nonnegative=True)
    expected = numerator / max(denominator, 1)
    if observed > 1.0 or not math.isclose(
        observed, expected, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise _summary_error(
            path,
            f"{label}={observed} is inconsistent with {numerator}/{denominator}",
        )


def validate_summary_payload(
    summary: Any,
    task: EvaluationTask,
    *,
    path: Path,
) -> dict[str, Any]:
    """Validate all fields used to accept and aggregate one simulator game."""

    summary = _mapping(summary, path=path, label="root")
    if summary.get("scenario") != task.map_id:
        raise _summary_error(
            path,
            f"scenario {summary.get('scenario')!r} does not match {task.map_id!r}",
        )
    policies = _mapping(summary.get("policies"), path=path, label="policies")
    expected_policies = {
        "red": RED_UPPER_POLICY,
        "red_motion": RED_MOTION_POLICY,
        "blue": task.blue_policy,
    }
    for name, expected in expected_policies.items():
        if policies.get(name) != expected:
            raise _summary_error(
                path,
                f"policies.{name}={policies.get(name)!r} does not match {expected!r}",
            )

    steps = _nonnegative_int(
        summary.get("steps_executed"), path=path, label="steps_executed"
    )
    if steps < 1 or steps > task.max_steps:
        raise _summary_error(
            path, f"steps_executed={steps} is outside [1, {task.max_steps}]"
        )
    termination_reason = summary.get("termination_reason")
    if termination_reason not in ACCEPTED_TERMINATION_REASONS:
        raise _summary_error(
            path, f"termination_reason={termination_reason!r} is unsupported"
        )
    if termination_reason == "time_limit" and steps != task.max_steps:
        raise _summary_error(
            path,
            f"time_limit ended at {steps}, expected {task.max_steps}",
        )

    score = _mapping(summary.get("score"), path=path, label="score")
    for name in ("score", "raw_score", "K", "T", "destroyed_weight"):
        _finite_number(
            score.get(name), path=path, label=f"score.{name}", nonnegative=True
        )
    total_objective_weight = _finite_number(
        score.get("total_objective_weight"),
        path=path,
        label="score.total_objective_weight",
        nonnegative=True,
    )
    if total_objective_weight <= 0.0:
        raise _summary_error(path, "score.total_objective_weight must be positive")
    if not isinstance(score.get("completed"), bool):
        raise _summary_error(path, "score.completed must be boolean")
    destroyed_ids = score.get("destroyed_ids")
    if not isinstance(destroyed_ids, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in destroyed_ids
    ):
        raise _summary_error(path, "score.destroyed_ids must be an integer list")
    if len(set(destroyed_ids)) != len(destroyed_ids):
        raise _summary_error(path, "score.destroyed_ids contains duplicates")
    objective_weights = _mapping(
        score.get("objective_weights"), path=path, label="score.objective_weights"
    )
    normalized_weights = {
        str(key): _finite_number(
            value,
            path=path,
            label=f"score.objective_weights.{key}",
            nonnegative=True,
        )
        for key, value in objective_weights.items()
    }
    weight_total = sum(normalized_weights.values())
    if not math.isclose(
        weight_total, total_objective_weight, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise _summary_error(path, "objective weights do not sum to total weight")
    destruction_steps = _mapping(
        score.get("destruction_steps"), path=path, label="score.destruction_steps"
    )
    for key, value in destruction_steps.items():
        destroyed_step = _nonnegative_int(
            value, path=path, label=f"score.destruction_steps.{key}"
        )
        if destroyed_step > task.max_steps:
            raise _summary_error(path, f"destruction step for {key} exceeds max_steps")
    destroyed_keys = {str(entity_id) for entity_id in destroyed_ids}
    if set(destruction_steps) != destroyed_keys:
        raise _summary_error(path, "destruction_steps keys do not match destroyed_ids")
    if not destroyed_keys.issubset(normalized_weights):
        raise _summary_error(path, "destroyed_ids are absent from objective_weights")
    expected_destroyed_weight = sum(
        normalized_weights[entity_id] for entity_id in destroyed_keys
    )
    if not math.isclose(
        float(score["destroyed_weight"]),
        expected_destroyed_weight,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise _summary_error(path, "destroyed_weight is inconsistent with destroyed_ids")
    if score["completed"] != (destroyed_keys == set(normalized_weights)):
        raise _summary_error(path, "score.completed is inconsistent with destroyed_ids")

    outcomes = _mapping(summary.get("outcomes"), path=path, label="outcomes")
    _mapping(outcomes.get("definitions"), path=path, label="outcomes.definitions")
    counts = {
        name: _nonnegative_int(
            outcomes.get(name), path=path, label=f"outcomes.{name}"
        )
        for name in SUMMARY_OUTCOME_COUNT_KEYS
    }
    if counts["engaged"] != counts["engaged_resolved"] + counts["engaged_censored"]:
        raise _summary_error(path, "engaged != engaged_resolved + engaged_censored")
    if counts["engaged_resolved"] != counts["evaded"] + counts["intercepted"]:
        raise _summary_error(path, "engaged_resolved != evaded + intercepted")
    for child, parent in (
        ("engaged", "launched"),
        ("arrived", "launched"),
        ("effective_hits", "arrived"),
        ("alive_at_end", "launched"),
        ("evaded", "arrived"),
    ):
        if counts[child] > counts[parent]:
            raise _summary_error(path, f"{child} exceeds {parent}")
    _rate(
        outcomes.get("evasion_rate"),
        counts["evaded"],
        counts["engaged_resolved"],
        path=path,
        label="outcomes.evasion_rate",
    )
    _rate(
        outcomes.get("arrival_rate"),
        counts["arrived"],
        counts["launched"],
        path=path,
        label="outcomes.arrival_rate",
    )
    _rate(
        outcomes.get("hit_rate"),
        counts["effective_hits"],
        counts["arrived"],
        path=path,
        label="outcomes.hit_rate",
    )
    _rate(
        outcomes.get("hit_rate_per_launch"),
        counts["effective_hits"],
        counts["launched"],
        path=path,
        label="outcomes.hit_rate_per_launch",
    )
    _finite_number(
        outcomes.get("arrival_tolerance_m"),
        path=path,
        label="outcomes.arrival_tolerance_m",
        nonnegative=True,
    )

    by_type = _mapping(outcomes.get("by_type"), path=path, label="outcomes.by_type")
    if set(by_type) != set(MISSILE_TYPE_LABELS):
        raise _summary_error(path, "outcomes.by_type does not contain exactly H/M/L")
    type_totals = {"launched": 0, "arrived": 0, "effective_hits": 0}
    for type_id in MISSILE_TYPE_LABELS:
        values = _mapping(
            by_type[type_id], path=path, label=f"outcomes.by_type.{type_id}"
        )
        type_counts = {
            name: _nonnegative_int(
                values.get(name),
                path=path,
                label=f"outcomes.by_type.{type_id}.{name}",
            )
            for name in type_totals
        }
        if type_counts["arrived"] > type_counts["launched"]:
            raise _summary_error(path, f"type {type_id} arrived exceeds launched")
        if type_counts["effective_hits"] > type_counts["arrived"]:
            raise _summary_error(path, f"type {type_id} hits exceeds arrived")
        _rate(
            values.get("arrival_rate"),
            type_counts["arrived"],
            type_counts["launched"],
            path=path,
            label=f"outcomes.by_type.{type_id}.arrival_rate",
        )
        _rate(
            values.get("hit_rate_per_launch"),
            type_counts["effective_hits"],
            type_counts["launched"],
            path=path,
            label=f"outcomes.by_type.{type_id}.hit_rate_per_launch",
        )
        for name, value in type_counts.items():
            type_totals[name] += value
    for name, total in type_totals.items():
        if total != counts[name]:
            raise _summary_error(path, f"by_type {name} total does not match outcomes")

    per_platform = _mapping(
        outcomes.get("per_platform"), path=path, label="outcomes.per_platform"
    )
    if len(per_platform) != counts["launched"]:
        raise _summary_error(path, "per_platform size does not match launched")
    platform_counts = {name: 0 for name in SUMMARY_OUTCOME_COUNT_KEYS[1:]}
    for platform_id, raw_values in per_platform.items():
        values = _mapping(
            raw_values, path=path, label=f"outcomes.per_platform.{platform_id}"
        )
        flags: dict[str, bool] = {}
        for name in ("engaged", "resolved", "arrived", "effective_hit", "alive_at_end"):
            value = values.get(name)
            if not isinstance(value, bool):
                raise _summary_error(path, f"platform {platform_id} {name} must be boolean")
            flags[name] = value
        distance = values.get("minimum_target_distance_m")
        if distance is not None:
            _finite_number(
                distance,
                path=path,
                label=f"platform {platform_id} minimum_target_distance_m",
                nonnegative=True,
            )
        if flags["arrived"] and not flags["resolved"]:
            raise _summary_error(path, f"platform {platform_id} arrived but is unresolved")
        if flags["effective_hit"] and not flags["arrived"]:
            raise _summary_error(path, f"platform {platform_id} hit without arrival")
        platform_counts["engaged"] += int(flags["engaged"])
        platform_counts["engaged_resolved"] += int(
            flags["engaged"] and flags["resolved"]
        )
        platform_counts["engaged_censored"] += int(
            flags["engaged"] and not flags["resolved"]
        )
        platform_counts["evaded"] += int(
            flags["engaged"] and flags["resolved"] and flags["arrived"]
        )
        platform_counts["intercepted"] += int(
            flags["engaged"] and flags["resolved"] and not flags["arrived"]
        )
        platform_counts["arrived"] += int(flags["arrived"])
        platform_counts["effective_hits"] += int(flags["effective_hit"])
        platform_counts["alive_at_end"] += int(flags["alive_at_end"])
    for name, total in platform_counts.items():
        if total != counts[name]:
            raise _summary_error(path, f"per_platform {name} total does not match outcomes")

    red = _mapping(summary.get("red"), path=path, label="red")
    red_counts = {
        name: _nonnegative_int(red.get(name), path=path, label=f"red.{name}")
        for name in ("platforms_total", "launched", "alive", "lost")
    }
    if red_counts["alive"] + red_counts["lost"] != red_counts["platforms_total"]:
        raise _summary_error(path, "red alive + lost does not match platforms_total")
    if red_counts["launched"] > red_counts["platforms_total"]:
        raise _summary_error(path, "red launched exceeds platforms_total")
    if red_counts["launched"] != counts["launched"]:
        raise _summary_error(path, "red launched does not match outcomes.launched")
    if counts["alive_at_end"] > red_counts["alive"]:
        raise _summary_error(path, "outcomes alive_at_end exceeds red alive")
    blue = _mapping(summary.get("blue"), path=path, label="blue")
    blue_counts = {
        name: _nonnegative_int(blue.get(name), path=path, label=f"blue.{name}")
        for name in ("entities_total", "alive", "destroyed")
    }
    if blue_counts["alive"] + blue_counts["destroyed"] != blue_counts["entities_total"]:
        raise _summary_error(path, "blue alive + destroyed does not match entities_total")
    for name in ("initial_health_total", "remaining_health_total"):
        _finite_number(blue.get(name), path=path, label=f"blue.{name}", nonnegative=True)
    if not isinstance(summary.get("objectives"), list):
        raise _summary_error(path, "objectives must be a list")
    return summary


def read_validated_summary(
    path: Path, task: EvaluationTask
) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        summary = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise _summary_error(path, f"cannot read JSON: {error}") from error
    validated = validate_summary_payload(summary, task, path=path)
    return validated, hashlib.sha256(raw).hexdigest()


def valid_summary(path: Path, task: EvaluationTask) -> bool:
    try:
        read_validated_summary(path, task)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def write_protocol_sidecar(
    run_dir: Path,
    summary_path: Path,
    identity: dict[str, Any],
    *,
    expected_summary_sha256: str | None = None,
) -> Path:
    """Bind exactly one completed summary to its full task protocol."""

    run_root = run_dir.resolve()
    resolved_summary = summary_path.resolve()
    try:
        relative_summary = resolved_summary.relative_to(run_root)
    except ValueError as error:
        raise ValueError(f"Summary is outside run directory: {summary_path}") from error
    actual_summary_sha256 = file_sha256(resolved_summary)
    if (
        expected_summary_sha256 is not None
        and actual_summary_sha256 != expected_summary_sha256
    ):
        raise RuntimeError(f"Summary changed before cache commit: {resolved_summary}")
    fingerprint = task_protocol_fingerprint(identity)
    sidecar_path = run_dir / CACHE_PROTOCOL_FILENAME
    atomic_json(
        sidecar_path,
        {
            "schema_version": CACHE_PROTOCOL_SCHEMA_VERSION,
            "protocol_fingerprint": fingerprint,
            "protocol_identity": identity,
            "summary_path": str(relative_summary),
            "summary_sha256": actual_summary_sha256,
        },
    )
    return sidecar_path


def reusable_summary(
    run_dir: Path,
    identity: dict[str, Any],
    task: EvaluationTask,
    *,
    force: bool = False,
) -> Path | None:
    """Return a cache hit only when the sidecar matches byte-for-byte identity."""

    if force:
        return None
    sidecar_path = run_dir / CACHE_PROTOCOL_FILENAME
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_fingerprint = task_protocol_fingerprint(identity)
    if sidecar.get("schema_version") != CACHE_PROTOCOL_SCHEMA_VERSION:
        return None
    if sidecar.get("protocol_fingerprint") != expected_fingerprint:
        return None
    if sidecar.get("protocol_identity") != identity:
        return None
    if task_protocol_fingerprint(sidecar["protocol_identity"]) != expected_fingerprint:
        return None
    relative_summary = sidecar.get("summary_path")
    if not isinstance(relative_summary, str) or Path(relative_summary).is_absolute():
        return None
    run_root = run_dir.resolve()
    candidate = (run_dir / relative_summary).resolve()
    try:
        candidate.relative_to(run_root)
    except ValueError:
        return None
    expected_summary_sha256 = sidecar.get("summary_sha256")
    if not isinstance(expected_summary_sha256, str) or not candidate.is_file():
        return None
    try:
        actual_summary_sha256 = file_sha256(candidate)
    except OSError:
        return None
    if actual_summary_sha256 != expected_summary_sha256 or not valid_summary(
        candidate, task
    ):
        return None
    return candidate


def latest_valid_summary(
    run_dir: Path,
    task: EvaluationTask,
    *,
    exclude: set[Path] | None = None,
) -> Path | None:
    excluded = exclude or set()
    candidates = sorted(
        run_dir.glob("*/summary.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        if path.resolve() in excluded:
            continue
        if valid_summary(path, task):
            return path
    return None


def task_slug(task: EvaluationTask) -> str:
    blue = task.blue_policy.split("_", 1)[0]
    return f"{task.condition}_{task.map_id.lower()}_{blue}_s{task.seed:04d}"


def select_runtime(requested: str) -> str:
    if requested != "auto":
        return requested
    _libc_name, libc_version = platform.libc_ver()
    try:
        components = tuple(int(item) for item in libc_version.split(".")[:2])
    except ValueError:
        components = (0, 0)
    return "host" if components >= (2, 38) else "docker"


def terminate_process(
    process: subprocess.Popen[Any],
    grace_seconds: float,
    container_name: str | None = None,
) -> None:
    if process.poll() is not None:
        return
    if container_name is not None:
        subprocess.run(
            ["docker", "kill", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def terminate_all_active(
    grace_seconds: float,
    *,
    cancel_event: threading.Event | None = None,
) -> None:
    """Linearize matrix cancellation with launch and snapshot active children."""

    with _ACTIVE_LOCK:
        if cancel_event is not None:
            cancel_event.set()
        processes = tuple(_ACTIVE_PROCESSES.items())
    for process, container_name in processes:
        terminate_process(process, grace_seconds, container_name)


def start_process_unless_cancelled(
    command: list[str],
    *,
    cancel_event: threading.Event,
    container_name: str | None,
    log_file: Any,
    env: dict[str, str],
) -> subprocess.Popen[Any]:
    """Atomically arbitrate cancellation against subprocess launch/registration."""

    with _ACTIVE_LOCK:
        if cancel_event.is_set():
            raise EvaluationCancelled("evaluation cancelled before simulator launch")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _ACTIVE_PROCESSES[process] = container_name
    return process


def container_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(ROOT)
    except ValueError as error:
        raise ValueError(
            f"Docker checkpoints must be inside the project root: {path}"
        ) from error
    return str(Path("/workspace") / relative)


def simulator_command(
    args: argparse.Namespace,
    task: EvaluationTask,
    run_id: str,
) -> tuple[list[str], str | None]:
    common = [
        "run",
        "--scenario",
        task.scenario,
        "--blue-policy",
        task.blue_policy,
        "--red-policy",
        RED_UPPER_POLICY,
        "--red-motion-policy",
        RED_MOTION_POLICY,
        "--red-learning-model",
        task.checkpoint,
        "--seed",
        str(task.seed),
        "--run-id",
        run_id,
        "--",
        "--total-rounds",
        "1",
        "--max-steps",
        str(task.max_steps),
    ]
    common.extend([
        "--render-mode",
        "none",
        "--disable-log-color",
    ])
    if args.runtime == "host":
        return [sys.executable, str(ROOT / "run.py"), *common], None

    container_name = (
        f"v8v13-{task.condition}-{task.map_id.lower()}-"
        f"{task.blue_policy.split('_', 1)[0]}-s{task.seed:04d}-"
        f"{task.checkpoint_sha256[:6]}-{os.getpid()}"
    )
    common[common.index(task.checkpoint)] = container_path(Path(task.checkpoint))
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
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "-v",
        f"{COMPETITION_ENV_HOST_ROOT}:{COMPETITION_ENV_CONTAINER_ROOT}:ro",
        "-v",
        f"{ROOT}:/workspace",
        "-w",
        "/workspace",
        "-e",
        "PYTHONPATH=/workspace/core",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "MKL_NUM_THREADS=1",
        "-e",
        "RED_LEARNING_TRAIN=0",
        "-e",
        f"BLUE_OBJECTIVE_MOTION={args.objective_motion}",
        "-e",
        f"LOCAL_AVOID_GATE_MODE={task.gate_mode}",
        "-e",
        f"LD_LIBRARY_PATH={library_path}",
        getattr(args, "docker_run_reference", args.docker_image),
        COMPETITION_PYTHON_CONTAINER,
        "/workspace/run.py",
        *common,
    ]
    return command, container_name


def simulator_environment(
    args: argparse.Namespace, task: EvaluationTask
) -> dict[str, str]:
    """Build a scrubbed host environment with evaluation mode forced on."""

    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith(("LOCAL_AVOID_", "BLUE_")):
            env.pop(key, None)
    env.pop("RED_LEARNING_TRAIN", None)
    env["BLUE_OBJECTIVE_MOTION"] = args.objective_motion
    env["LOCAL_AVOID_GATE_MODE"] = task.gate_mode
    env["RED_LEARNING_TRAIN"] = "0"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    python_lib = str(Path(sys.executable).resolve().parents[1] / "lib")
    current_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{python_lib}:{current_library_path}"
        if current_library_path
        else python_lib
    )
    return env


def summary_record(
    task: EvaluationTask,
    summary_path: Path,
    *,
    run_id: str,
    protocol_fingerprint: str,
    cached: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    summary, summary_sha256 = read_validated_summary(summary_path, task)
    outcomes = summary["outcomes"]
    steps = summary["steps_executed"]
    termination_reason = summary["termination_reason"]
    outcome_record = {key: outcomes.get(key) for key in outcomes}
    per_platform = outcomes.get("per_platform", {})
    outcome_record["sampled_damage_zone_100"] = sum(
        1
        for outcome in per_platform.values()
        if isinstance(outcome, dict)
        and outcome.get("minimum_target_distance_m") is not None
        and float(outcome["minimum_target_distance_m"]) <= 100.0
    )
    return {
        **asdict(task),
        "run_id": run_id,
        "protocol_fingerprint": protocol_fingerprint,
        "cached": cached,
        "elapsed_seconds": elapsed_seconds,
        "score": float(summary["score"]["score"]),
        "completed": bool(summary["score"]["completed"]),
        "steps": steps,
        "termination_reason": termination_reason,
        "summary_path": str(summary_path),
        "summary_sha256": summary_sha256,
        "outcomes": outcome_record,
    }


def run_task(
    args: argparse.Namespace,
    task: EvaluationTask,
    output_dir: Path,
    cancel_event: threading.Event,
    runtime_environment_identity: dict[str, Any],
    code_identity: dict[str, Any],
    observation_encoding_identity: dict[str, Any],
) -> dict[str, Any]:
    if cancel_event.is_set():
        raise EvaluationCancelled("evaluation cancelled before task preparation")
    prefix = args.run_prefix or "full_sim_v8_vs_v13"
    identity = task_identity_from_args(
        args,
        task,
        runtime_environment_identity=runtime_environment_identity,
        code_identity=code_identity,
        observation_encoding_identity=observation_encoding_identity,
    )
    protocol_fingerprint = task_protocol_fingerprint(identity)
    run_id = task_run_id(prefix, task, identity)
    run_dir = ROOT / "results" / "runs" / run_id
    assert_execution_inputs_unchanged(
        args,
        task,
        runtime_environment_identity,
        code_identity,
        observation_encoding_identity,
    )
    cached_summary = reusable_summary(run_dir, identity, task, force=args.force)
    if cached_summary is not None:
        return summary_record(
            task,
            cached_summary,
            run_id=run_id,
            protocol_fingerprint=protocol_fingerprint,
            cached=True,
            elapsed_seconds=0.0,
        )

    # Docker writes timestamped result children as root.  Create the parent as
    # the evaluator user first so the protocol sidecar remains writable.
    run_dir.mkdir(parents=True, exist_ok=True)
    old_summaries = {path.resolve() for path in run_dir.glob("*/summary.json")}
    command, container_name = simulator_command(args, task, run_id)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task_slug(task)}.log"
    env = simulator_environment(args, task)

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = start_process_unless_cancelled(
            command,
            cancel_event=cancel_event,
            container_name=container_name,
            log_file=log_file,
            env=env,
        )
        try:
            try:
                return_code = process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                terminate_process(process, args.kill_grace_seconds, container_name)
                return {
                    **asdict(task),
                    "run_id": run_id,
                    "protocol_fingerprint": protocol_fingerprint,
                    "error": f"timeout after {args.timeout_seconds:.1f}s",
                    "timed_out": True,
                    "elapsed_seconds": time.monotonic() - started,
                    "log_path": str(log_path),
                }
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE_PROCESSES.pop(process, None)

    elapsed = time.monotonic() - started
    summary_path = latest_valid_summary(run_dir, task, exclude=old_summaries)
    if return_code != 0 or summary_path is None:
        return {
            **asdict(task),
            "run_id": run_id,
            "protocol_fingerprint": protocol_fingerprint,
            "error": f"exit={return_code}, fresh_summary={summary_path}",
            "elapsed_seconds": elapsed,
            "log_path": str(log_path),
        }
    assert_execution_inputs_unchanged(
        args,
        task,
        runtime_environment_identity,
        code_identity,
        observation_encoding_identity,
    )
    record = summary_record(
        task,
        summary_path,
        run_id=run_id,
        protocol_fingerprint=protocol_fingerprint,
        cached=False,
        elapsed_seconds=elapsed,
    )
    return record


def commit_completed_task_sidecars(
    records: Iterable[dict[str, Any]],
    args: argparse.Namespace,
    runtime_environment_identity: dict[str, Any],
    code_identity: dict[str, Any],
    observation_encoding_identity: dict[str, Any],
) -> list[Path]:
    """Publish fresh task caches only after the matrix-wide final guard passes."""

    committed: list[Path] = []
    prefix = args.run_prefix or "full_sim_v8_vs_v13"
    for record in records:
        if "error" in record or record.get("cached") is not False:
            continue
        task = EvaluationTask(
            **{key: record[key] for key in EvaluationTask.__dataclass_fields__}
        )
        identity = task_identity_from_args(
            args,
            task,
            runtime_environment_identity,
            code_identity,
            observation_encoding_identity,
        )
        fingerprint = task_protocol_fingerprint(identity)
        expected_run_id = task_run_id(prefix, task, identity)
        if record.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"Task fingerprint changed before cache commit: {task_slug(task)}")
        if record.get("run_id") != expected_run_id:
            raise RuntimeError(f"Task run ID changed before cache commit: {task_slug(task)}")
        summary_path = Path(str(record["summary_path"]))
        _summary, summary_sha256 = read_validated_summary(summary_path, task)
        if summary_sha256 != record.get("summary_sha256"):
            raise RuntimeError(f"Summary changed before cache commit: {summary_path}")
        run_dir = ROOT / "results" / "runs" / expected_run_id
        committed.append(
            write_protocol_sidecar(
                run_dir,
                summary_path,
                identity,
                expected_summary_sha256=summary_sha256,
            )
        )
    return committed


def grouped(
    records: Iterable[dict[str, Any]], keys: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        key = "|".join(str(row[item]) for item in keys)
        buckets.setdefault(key, []).append(row)
    return {key: summarize(rows) for key, rows in sorted(buckets.items())}


def summarize_by_type(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, int]] = {}
    for row in records:
        condition = str(row["condition"])
        for type_id, label in MISSILE_TYPE_LABELS.items():
            values = row["outcomes"].get("by_type", {}).get(type_id, {})
            key = f"{condition}|{label}"
            bucket = totals.setdefault(
                key, {"launched": 0, "arrived": 0, "effective_hits": 0}
            )
            for count_key in bucket:
                bucket[count_key] += int(values.get(count_key, 0))
    result: dict[str, dict[str, Any]] = {}
    for key, counts in totals.items():
        result[key] = {
            **counts,
            "arrival_rate": counts["arrived"] / max(counts["launched"], 1),
            "hit_rate": counts["effective_hits"] / max(counts["arrived"], 1),
            "hit_rate_per_launch": counts["effective_hits"]
            / max(counts["launched"], 1),
        }
    return result


def paired_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conditions_by_case: dict[tuple[str, str, int], set[str]] = {}
    for row in records:
        case = (str(row["map_id"]), str(row["blue_policy"]), int(row["seed"]))
        conditions_by_case.setdefault(case, set()).add(str(row["condition"]))
    paired_cases = {
        case
        for case, conditions in conditions_by_case.items()
        if set(CONDITION_ORDER).issubset(conditions)
    }
    return [
        row
        for row in records
        if (str(row["map_id"]), str(row["blue_policy"]), int(row["seed"]))
        in paired_cases
    ]


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def metric_row(label: str, metrics: dict[str, Any] | None) -> str:
    if metrics is None:
        return f"| {label} | NA | NA | NA | NA | NA | NA | NA |"
    return (
        f"| {label} | {metrics['games']} | {metrics['score_mean']:.2f} | "
        f"{pct(metrics['evasion_rate'])} | {pct(metrics['arrival_rate'])} | "
        f"{pct(metrics['hit_rate'])} | {pct(metrics['hit_rate_per_launch'])} | "
        f"{metrics['intercepted']} |"
    )


def delta_row(v13: dict[str, Any] | None, v8: dict[str, Any] | None) -> str:
    if v13 is None or v8 is None:
        return "| V8 - V13 | NA | NA | NA | NA | NA | NA | NA |"
    return (
        f"| V8 - V13 | - | {v8['score_mean'] - v13['score_mean']:+.2f} | "
        f"{pct(v8['evasion_rate'] - v13['evasion_rate'])} | "
        f"{pct(v8['arrival_rate'] - v13['arrival_rate'])} | "
        f"{pct(v8['hit_rate'] - v13['hit_rate'])} | "
        f"{pct(v8['hit_rate_per_launch'] - v13['hit_rate_per_launch'])} | "
        f"{v8['intercepted'] - v13['intercepted']:+d} |"
    )


def table(
    lines: list[str],
    title: str,
    v13: dict[str, Any] | None,
    v8: dict[str, Any] | None,
) -> None:
    lines.extend(
        [
            f"### {title}",
            "",
            "| 条件 | 局数 | 均分 | 规避率 | 到达率 | 到达后命中率 | 每发命中率 | 被拦截数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            metric_row("V13", v13),
            metric_row("V8", v8),
            delta_row(v13, v8),
            "",
        ]
    )


def render_markdown(report: dict[str, Any]) -> str:
    protocol = report["protocol"]
    lines = [
        "# V8 与 V13 完整仿真评估",
        "",
        f"- 状态：`{report['status']}`",
        f"- 有效局数：{protocol['valid_games']} / {protocol['expected_games']}",
        f"- 成对样本：{protocol['paired_cases']} 组（V13/V8 各一局）",
        f"- 地图：{', '.join(protocol['maps'])}",
        "- 蓝方：" + ", ".join(item.split("_", 1)[0].upper() for item in protocol["blue_policies"]),
        f"- 种子：{', '.join(str(item) for item in protocol['seeds'])}",
        f"- 每局最大步数：`{protocol['max_steps']}`",
        f"- 蓝方目标运动：`{protocol['objective_motion']}`",
        f"- 上层 / 机动：`{protocol['red_upper_policy']}` / "
        f"`{protocol['red_motion_policy']}`",
        f"- 门控：`V13={protocol['conditions']['v13']['gate_mode']}`，"
        f"`V8={protocol['conditions']['v8']['gate_mode']}`",
        "- 到达后命中率：有效损毁数 / 到达目标区的导弹数；不是微仿真的接触代理。",
        "",
        "## 总体（仅同地图、同蓝方、同 seed 的成对有效局）",
        "",
    ]
    overall = report["overall"]
    table(lines, "全部", overall.get("v13"), overall.get("v8"))
    lines.extend(["## 按地图", ""])
    by_map = report["by_condition_map"]
    for map_id in protocol["maps"]:
        table(
            lines,
            map_id,
            by_map.get(f"v13|{map_id}"),
            by_map.get(f"v8|{map_id}"),
        )
    lines.extend(["## 按蓝方策略", ""])
    by_blue = report["by_condition_blue"]
    for blue_policy in protocol["blue_policies"]:
        table(
            lines,
            blue_policy.split("_", 1)[0].upper(),
            by_blue.get(f"v13|{blue_policy}"),
            by_blue.get(f"v8|{blue_policy}"),
        )
    lines.extend(
        [
            "## 按 H/M/L 型号",
            "",
            "| 型号 | 条件 | 发射数 | 到达数 | 到达率 | 有效损毁数 | 到达后命中率 | 每发命中率 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_type = report["by_condition_type"]
    for type_label in MISSILE_TYPE_LABELS.values():
        for condition in CONDITION_ORDER:
            values = by_type.get(f"{condition}|{type_label}")
            if values is None:
                continue
            lines.append(
                f"| {type_label} | {condition.upper()} | {values['launched']} | "
                f"{values['arrived']} | {pct(values['arrival_rate'])} | "
                f"{values['effective_hits']} | {pct(values['hit_rate'])} | "
                f"{pct(values['hit_rate_per_launch'])} |"
            )
    lines.append("")
    if report["failures"]:
        lines.extend(["## 失败或超时任务", ""])
        for row in report["failures"]:
            failed_task = EvaluationTask(
                **{
                    key: row[key]
                    for key in EvaluationTask.__dataclass_fields__
                }
            )
            lines.append(f"- `{task_slug(failed_task)}`: {row['error']}")
        lines.append("")
    return "\n".join(lines)


def protocol_max_steps(tasks: Iterable[EvaluationTask]) -> int | list[int]:
    values = sorted({int(task.max_steps) for task in tasks})
    if len(values) == 1:
        return values[0]
    return values


def protocol_scenarios(tasks: Iterable[EvaluationTask]) -> dict[str, dict[str, Any]]:
    return {
        task.map_id: {
            "reference": task.scenario,
            "sha256": task.scenario_sha256,
            "case_info_sha256": task.scenario_case_info_sha256,
            "max_steps": task.max_steps,
        }
        for task in tasks
    }


def task_protocol_fingerprints(
    tasks: Iterable[EvaluationTask],
    args: argparse.Namespace,
    runtime_environment_identity: dict[str, Any],
    code_identity: dict[str, Any],
    observation_encoding_identity: dict[str, Any],
) -> dict[str, str]:
    return {
        task_slug(task): task_protocol_fingerprint(
            task_identity_from_args(
                args,
                task,
                runtime_environment_identity,
                code_identity,
                observation_encoding_identity,
            )
        )
        for task in tasks
    }


def build_report(
    tasks: list[EvaluationTask],
    records: list[dict[str, Any]],
    conditions: tuple[Condition, ...],
    args: argparse.Namespace,
    runtime_environment_identity: dict[str, Any],
    code_identity: dict[str, Any],
    observation_encoding_identity: dict[str, Any],
) -> dict[str, Any]:
    valid = [row for row in records if "error" not in row]
    failures = [row for row in records if "error" in row]
    paired = paired_records(valid)
    paired_cases = len(paired) // len(CONDITION_ORDER)
    overall = {
        condition: (
            summarize(rows)
            if (rows := [row for row in paired if row["condition"] == condition])
            else None
        )
        for condition in CONDITION_ORDER
    }
    return {
        "status": "complete" if not failures and len(valid) == len(tasks) else "partial",
        "protocol": {
            "maps": list(dict.fromkeys(task.map_id for task in tasks)),
            "blue_policies": list(dict.fromkeys(task.blue_policy for task in tasks)),
            "seeds": sorted({task.seed for task in tasks}),
            "max_steps": protocol_max_steps(tasks),
            "max_steps_by_map": {
                task.map_id: task.max_steps for task in tasks
            },
            "objective_motion": args.objective_motion,
            "red_upper_policy": RED_UPPER_POLICY,
            "red_motion_policy": RED_MOTION_POLICY,
            "scenarios": protocol_scenarios(tasks),
            "conditions": {
                condition.label: {
                    "checkpoint": str(condition.checkpoint),
                    "checkpoint_sha256": condition.checkpoint_sha256,
                    "gate_mode": condition.gate_mode,
                }
                for condition in conditions
            },
            "expected_games": len(tasks),
            "valid_games": len(valid),
            "paired_cases": paired_cases,
            "timeout_seconds_per_game": args.timeout_seconds,
            "kill_grace_seconds": args.kill_grace_seconds,
            "termination_contract": {
                "accepted_reasons": list(ACCEPTED_TERMINATION_REASONS),
                "time_limit_requires_exact_max_steps": True,
                "simulator_total_rounds": 1,
            },
            "runtime": args.runtime,
            "docker_image": args.docker_image,
            "runtime_environment_identity": runtime_environment_identity,
            "code_identity": code_identity,
            "observation_encoding_identity": observation_encoding_identity,
            "task_protocol_fingerprints": task_protocol_fingerprints(
                tasks,
                args,
                runtime_environment_identity,
                code_identity,
                observation_encoding_identity,
            ),
            "isolated_process_per_game": True,
        },
        "overall": overall,
        "by_condition_map": grouped(paired, ("condition", "map_id")),
        "by_condition_blue": grouped(paired, ("condition", "blue_policy")),
        "by_condition_type": summarize_by_type(paired),
        "failures": failures,
    }


def protocol_preview(
    tasks: list[EvaluationTask],
    conditions: tuple[Condition, ...],
    args: argparse.Namespace,
    runtime_environment_identity: dict[str, Any],
    code_identity: dict[str, Any],
    observation_encoding_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "objective_motion": args.objective_motion,
        "red_upper_policy": RED_UPPER_POLICY,
        "red_motion_policy": RED_MOTION_POLICY,
        "maps": list(dict.fromkeys(task.map_id for task in tasks)),
        "blue_policies": list(dict.fromkeys(task.blue_policy for task in tasks)),
        "seeds": sorted({task.seed for task in tasks}),
        "max_steps": protocol_max_steps(tasks),
        "max_steps_by_map": {task.map_id: task.max_steps for task in tasks},
        "scenarios": protocol_scenarios(tasks),
        "conditions": {
            condition.label: {
                "checkpoint": str(condition.checkpoint),
                "checkpoint_sha256": condition.checkpoint_sha256,
                "gate_mode": condition.gate_mode,
            }
            for condition in conditions
        },
        "games": len(tasks),
        "workers": args.workers,
        "timeout_seconds_per_game": args.timeout_seconds,
        "kill_grace_seconds": args.kill_grace_seconds,
        "termination_contract": {
            "accepted_reasons": list(ACCEPTED_TERMINATION_REASONS),
            "time_limit_requires_exact_max_steps": True,
            "simulator_total_rounds": 1,
        },
        "runtime": args.runtime,
        "docker_image": args.docker_image,
        "runtime_environment_identity": runtime_environment_identity,
        "code_identity": code_identity,
        "observation_encoding_identity": observation_encoding_identity,
        "task_protocol_fingerprints": task_protocol_fingerprints(
            tasks,
            args,
            runtime_environment_identity,
            code_identity,
            observation_encoding_identity,
        ),
    }


def parse_unique_csv(raw: str, *, label: str) -> tuple[str, ...]:
    """Parse a non-empty comma list and reject duplicate matrix axes."""

    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"At least one {label} is required")
    duplicates = sorted({item for item in values if values.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {label}: {', '.join(duplicates)}")
    return values


def parse_unique_nonnegative_int_csv(raw: str, *, label: str) -> tuple[int, ...]:
    """Parse integer axes canonically, then reject aliases and negatives."""

    values = parse_unique_csv(raw, label=label)
    try:
        normalized = tuple(int(item) for item in values)
    except ValueError as error:
        raise ValueError(f"Invalid {label}: {error}") from error
    negatives = sorted({value for value in normalized if value < 0})
    if negatives:
        raise ValueError(
            f"Negative {label}: {', '.join(str(value) for value in negatives)}"
        )
    duplicates = sorted(
        {value for value in normalized if normalized.count(value) > 1}
    )
    if duplicates:
        raise ValueError(
            f"Duplicate {label}: {', '.join(str(value) for value in duplicates)}"
        )
    return normalized


def invalidate_completion_marker(output_dir: Path) -> bool:
    """Remove a stale success marker before a new matrix starts."""

    marker = output_dir / "COMPLETED.json"
    if not marker.exists():
        return False
    marker.unlink()
    return True


def resolve_output_dir(args: argparse.Namespace, seeds: tuple[int, ...]) -> Path:
    if args.output_dir is not None:
        return resolve_under_root(args.output_dir)
    seed_tag = "_".join(f"s{seed:04d}" for seed in seeds)
    nondefault_gate_tag = "".join(
        f"_{label}_g{gate_mode}"
        for label, gate_mode in (
            ("v13", args.v13_gate_mode),
            ("v8", args.v8_gate_mode),
        )
        if gate_mode != DEFAULT_GATE_MODES[label]
    )
    return (
        ROOT
        / "results"
        / "evaluations_set_attention_v8"
        / f"full_sim_{args.objective_motion}_v8_vs_v13{nondefault_gate_tag}_{seed_tag}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.runtime = select_runtime(args.runtime)
    map_names = parse_unique_csv(args.maps, label="map")
    selected_maps = set(map_names)
    maps = load_maps(selected_maps)
    blue_policies = parse_unique_csv(args.blue_policies, label="blue policy")
    unknown_blue = sorted(set(blue_policies) - set(BLUE_POLICIES))
    if unknown_blue:
        raise ValueError(f"Unknown blue policies: {', '.join(unknown_blue)}")
    seeds = parse_unique_nonnegative_int_csv(args.seeds, label="seed")
    output_dir = resolve_output_dir(args, seeds)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        invalidate_completion_marker(output_dir)
        atomic_json(
            output_dir / "run_status.json",
            {"status": "preflight", "completed_games": 0},
        )

    runtime_environment_identity = resolve_runtime_environment_identity(
        args.runtime,
        args.docker_image,
    )
    if args.runtime == "docker":
        args.docker_run_reference = runtime_environment_identity[
            "docker_run_reference"
        ]

    v8_checkpoint = resolve_under_root(args.v8_checkpoint)
    v13_checkpoint = resolve_under_root(args.v13_checkpoint)
    for checkpoint in (v8_checkpoint, v13_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    conditions = (
        Condition("v13", v13_checkpoint, args.v13_gate_mode, file_sha256(v13_checkpoint)),
        Condition("v8", v8_checkpoint, args.v8_gate_mode, file_sha256(v8_checkpoint)),
    )
    scenario_protocols = {
        scenario: resolve_scenario_protocol(scenario, args.max_steps)
        for _map_id, _tier, scenario in maps
    }
    code_identity, observation_encoding_identity = current_source_identities()
    tasks = [
        EvaluationTask(
            condition=condition.label,
            checkpoint=str(condition.checkpoint),
            checkpoint_sha256=condition.checkpoint_sha256,
            gate_mode=condition.gate_mode,
            map_id=map_id,
            tier=tier,
            scenario=scenario,
            blue_policy=blue_policy,
            seed=seed,
            scenario_sha256=scenario_protocols[scenario][0],
            scenario_case_info_sha256=scenario_protocols[scenario][1],
            max_steps=scenario_protocols[scenario][2],
        )
        for map_id, tier, scenario in maps
        for blue_policy in blue_policies
        for seed in seeds
        for condition in conditions
    ]
    if not tasks:
        raise RuntimeError("Evaluation matrix resolved to zero tasks")
    preview = protocol_preview(
        tasks,
        conditions,
        args,
        runtime_environment_identity,
        code_identity,
        observation_encoding_identity,
    )
    preview["matrix_protocol_fingerprint"] = hashlib.sha256(
        canonical_json(preview).encode("utf-8")
    ).hexdigest()
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    atomic_json(output_dir / "protocol.json", preview)
    records: list[dict[str, Any]] = []
    atomic_json(output_dir / "per_run.partial.json", records)
    atomic_json(
        output_dir / "run_status.json",
        {"status": "running", "completed_games": 0, "expected_games": len(tasks)},
    )
    interrupted = False
    cancel_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures: dict[Any, EvaluationTask] = {}
    try:
        futures = {
            executor.submit(
                run_task,
                args,
                task,
                output_dir,
                cancel_event,
                runtime_environment_identity,
                code_identity,
                observation_encoding_identity,
            ): task
            for task in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                record = future.result()
            except Exception as error:  # Keep the rest of a long matrix alive.
                identity = task_identity_from_args(
                    args,
                    task,
                    runtime_environment_identity,
                    code_identity,
                    observation_encoding_identity,
                )
                record = {
                    **asdict(task),
                    "run_id": task_run_id(
                        args.run_prefix or "full_sim_v8_vs_v13", task, identity
                    ),
                    "protocol_fingerprint": task_protocol_fingerprint(identity),
                    "error": f"{type(error).__name__}: {error}",
                }
            records.append(record)
            result_label = (
                "failed"
                if "error" in record
                else "cached" if record.get("cached") else "done"
            )
            atomic_json(output_dir / "per_run.partial.json", records)
            atomic_json(
                output_dir / "run_status.json",
                {
                    "status": "running",
                    "completed_games": index,
                    "expected_games": len(tasks),
                    "last_task": task_slug(task),
                    "last_result": result_label,
                },
            )
            print(
                f"PROGRESS {index}/{len(tasks)} {task_slug(task)} "
                f"{result_label}",
                flush=True,
            )
    except KeyboardInterrupt:
        interrupted = True
        terminate_all_active(
            args.kill_grace_seconds,
            cancel_event=cancel_event,
        )
        for future in futures:
            future.cancel()
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    if interrupted:
        atomic_json(
            output_dir / "run_status.json",
            {
                "status": "interrupted",
                "completed_games": len(records),
                "expected_games": len(tasks),
            },
        )
        print(f"INTERRUPTED {output_dir}", flush=True)
        return 130

    try:
        assert_runtime_environment_unchanged(args, runtime_environment_identity)
        final_code_identity, final_observation_identity = current_source_identities()
        if canonical_json(final_code_identity) != canonical_json(code_identity):
            raise RuntimeError("Simulator source tree changed during evaluation")
        if canonical_json(final_observation_identity) != canonical_json(
            observation_encoding_identity
        ):
            raise RuntimeError("Observation encoding changed during evaluation")
        checked_task_files: set[tuple[str, str]] = set()
        for task in tasks:
            key = (task.checkpoint, task.scenario)
            if key not in checked_task_files:
                assert_task_files_unchanged(task)
                checked_task_files.add(key)
        commit_completed_task_sidecars(
            records,
            args,
            runtime_environment_identity,
            code_identity,
            observation_encoding_identity,
        )
    except Exception as error:
        atomic_json(
            output_dir / "run_status.json",
            {
                "status": "protocol_guard_failed",
                "completed_games": len(records),
                "expected_games": len(tasks),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        print(f"PROTOCOL_GUARD_FAILED {output_dir}: {error}", flush=True)
        return 1

    report = build_report(
        tasks,
        records,
        conditions,
        args,
        runtime_environment_identity,
        code_identity,
        observation_encoding_identity,
    )
    report["protocol"]["matrix_protocol_fingerprint"] = preview[
        "matrix_protocol_fingerprint"
    ]
    valid = sorted(
        (row for row in records if "error" not in row),
        key=lambda row: (
            row["condition"], row["map_id"], row["blue_policy"], row["seed"]
        ),
    )
    atomic_json(output_dir / "comparison.json", report)
    atomic_json(output_dir / "per_run.json", valid)
    (output_dir / "COMPARISON.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    atomic_json(
        output_dir / "run_status.json",
        {
            "status": report["status"],
            "completed_games": len(records),
            "valid_games": len(valid),
            "expected_games": len(tasks),
            "matrix_protocol_fingerprint": preview[
                "matrix_protocol_fingerprint"
            ],
            "report": str(output_dir / "COMPARISON.md"),
        },
    )
    if report["status"] == "complete":
        atomic_json(
            output_dir / "COMPLETED.json",
            {
                "status": "complete",
                "valid_games": len(valid),
                "expected_games": len(tasks),
                "matrix_protocol_fingerprint": preview[
                    "matrix_protocol_fingerprint"
                ],
                "report": str(output_dir / "COMPARISON.md"),
            },
        )
    print(f"FULL_SIM_EVALUATION {output_dir}", flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
