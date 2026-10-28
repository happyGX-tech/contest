"""Unified local entry point for the competition workspace."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scenarios.cases import load_reward_policy
from policies.red.baselines import RED_POLICY_CHOICES


ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
SCENARIOS = ROOT / "scenarios" / "cases"
BLUE_POLICY_CHOICES = (
    "b0_fixed_ratio_random",
    "b1_nearest_interceptor",
    "b2_threat_priority",
    "b3_min_cost_assignment",
    "b4_joint_timing_assignment",
    "b5_successive_depth_coordination",
    "b6_mixed_fire_coordination",
    "b7_planned_nearest",
    "b8_threat_multi_wave",
    "b9_min_cost_planned_multi_wave",
)
RED_MOTION_POLICY_CHOICES = (
    "straight",
    "reactive_evasion",
    "random_masked",
    "ppo",
    "mappo",
    "local_avoid_s3",
)


def scenario_path(case_id: str) -> Path:
    if case_id == "default":
        return CORE / "scenarios" / "platform.json"

    # Bundled case IDs always win. This keeps the current nine-map design
    # authoritative even when a similarly named legacy path exists.
    candidate = SCENARIOS / case_id / "scenario.json"
    if candidate.is_file():
        return candidate

    external = Path(case_id).expanduser()
    if external.is_file() and external.suffix.lower() == ".json":
        return external.resolve()

    raise ValueError(
        f"Unknown scenario '{case_id}'. Use e.g. easy/E01, medium/M01, "
        "hard/H01, or an existing scenario.json path."
    )


def competition_step_limit(scenario: Path) -> int | None:
    """Return the scenario's scoring step limit, if it publishes one."""

    policy = load_reward_policy(scenario)
    return None if policy is None else int(policy.max_steps)


def cmd_run(args: argparse.Namespace) -> int:
    """Run one round through the 10-19 engine entry point.

    The engine now owns the red action protocol, the blue baselines and the
    scoring metadata, so this wrapper forwards explicit CLI flags instead of
    environment variables.  Paths and the shared library search path are set
    here so ``core/main.py`` can import the workspace ``policies`` package.
    """

    try:
        scenario = scenario_path(args.scenario)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    output = (ROOT / "results" / "runs" / args.run_id).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # argparse keeps the separator used before REMAINDER.  It is meaningful to
    # this wrapper only; forwarding it makes core/main.py reject its arguments.
    core_args = args.core_args[1:] if args.core_args[:1] == ["--"] else args.core_args
    command = [
        sys.executable,
        str(CORE / "main.py"),
        "--scenario", str(scenario),
        "--output-dir", str(output),
        "--red-policy", args.red_policy,
        "--red-motion-policy", args.red_motion_policy,
        "--blue-policy", args.blue_policy,
    ]
    if "--render-mode" not in core_args:
        command.extend(["--render-mode", "none"])
    if "--max-steps" not in core_args:
        limit = competition_step_limit(scenario)
        if limit is not None:
            command.extend(["--max-steps", str(limit)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.red_learning_model is not None:
        command.extend(["--red-learning-model", str(Path(args.red_learning_model).resolve())])
    if args.red_learning_train:
        command.append("--red-learning-train")
    if args.blue_interceptor_ratio is not None:
        command.extend(["--blue-interceptor-ratio", str(args.blue_interceptor_ratio)])
    if args.blue_plan_delay is not None:
        command.extend(["--blue-plan-delay", str(args.blue_plan_delay)])
    if args.blue_wave_gap is not None:
        command.extend(["--blue-wave-gap", str(args.blue_wave_gap)])
    if args.blue_wave_size is not None:
        command.extend(["--blue-wave-size", str(args.blue_wave_size)])
    if args.blue_min_cost_decision_interval is not None:
        command.extend(["--blue-min-cost-decision-interval", str(args.blue_min_cost_decision_interval)])
    command.extend(core_args)

    if any(
        arg == "--deployment-plan" or arg.startswith("--deployment-plan=")
        for arg in command
    ):
        # ``core/user_agents/red/grid_deploy_agent.py`` imports
        # ``scenarios.grid_maps``, which only exists in the repository-root
        # ``scenarios`` package (the engine's own package lives under
        # ``core/``).  Launching through ``-m`` keeps that root package first
        # on sys.path; other runs keep the original script invocation.
        command = [sys.executable, "-m", "core.main", *command[2:]]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), str(CORE), env.get("PYTHONPATH", "")) if part
    )
    runtime_library_path = Path(sys.executable).resolve().parent.parent / "lib"
    env["LD_LIBRARY_PATH"] = os.pathsep.join([
        str(CORE),
        str(CORE / "envengine" / "sdk" / "Util"),
        str(runtime_library_path),
        env.get("LD_LIBRARY_PATH", ""),
    ])
    env.setdefault("OMP_NUM_THREADS", "1")
    return subprocess.call(command, cwd=ROOT, env=env)


def cmd_ui(_: argparse.Namespace) -> int:
    from ui.server import serve
    serve(ROOT)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Competition platform workspace")
    subparsers = parser.add_subparsers(required=True)
    run_parser = subparsers.add_parser("run", help="Run a blue policy against a scenario")
    run_parser.add_argument(
        "--blue-policy",
        default="b0_fixed_ratio_random",
        choices=BLUE_POLICY_CHOICES,
        help="Blue interception policy (default: b0_fixed_ratio_random)",
    )
    run_parser.add_argument(
        "--blue-interceptor-ratio",
        type=int,
        help="Maximum interceptors assigned per detected red platform",
    )
    run_parser.add_argument(
        "--blue-plan-delay",
        type=float,
        help="Planning delay for B7/B9, in seconds",
    )
    run_parser.add_argument(
        "--blue-wave-gap",
        type=float,
        help="Gap between B8/B9 interceptor waves, in seconds",
    )
    run_parser.add_argument(
        "--blue-wave-size",
        type=int,
        help="Interceptors released per target in each blue wave",
    )
    run_parser.add_argument(
        "--blue-min-cost-decision-interval",
        type=int,
        help="B3 minimum decision interval in simulation-time units",
    )
    run_parser.add_argument(
        "--red-policy",
        default="r0_random",
        choices=RED_POLICY_CHOICES,
        help="Red initial-catalogue baseline (default: r0_random)",
    )
    run_parser.add_argument(
        "--red-motion-policy",
        default="reactive_evasion",
        choices=RED_MOTION_POLICY_CHOICES,
        help="Red post-launch motion policy (default: reactive_evasion)",
    )
    run_parser.add_argument(
        "--red-learning-model",
        help="PPO/MAPPO checkpoint path; required when --red-motion-policy is ppo or mappo",
    )
    run_parser.add_argument(
        "--red-learning-train",
        action="store_true",
        help="Train the selected PPO/MAPPO policy; use --red-learning-model as the output checkpoint path",
    )
    run_parser.add_argument("--scenario", default="default", help="default or easy/E01 … hard/H03")
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--run-id", default="manual")
    run_parser.add_argument("core_args", nargs=argparse.REMAINDER, help="Arguments forwarded to core/main.py")
    run_parser.set_defaults(handler=cmd_run)
    ui_parser = subparsers.add_parser("ui", help="Start the local plugin control page")
    ui_parser.set_defaults(handler=cmd_ui)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    raise SystemExit(arguments.handler(arguments))
