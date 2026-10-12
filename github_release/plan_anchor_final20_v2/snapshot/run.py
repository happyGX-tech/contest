"""Unified local entry point for the competition workspace."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scenarios.cases import load_reward_policy


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
RED_POLICY_CHOICES = (
    "r0_random",
    "r1_priority",
    "r2_static_assignment",
    "r3_wave_schedule",
    "r4_rolling_rules",
    "r5_event_rolling",
    "r6_frontload_decoy",
    "r7_strike_packages",
    "r7_static_search",
    "r8_satellite_packages",
    "r9_hierarchical_learning",
    "r10_unified",
    "r11_launch_timing",
    "r12_topformer",
)
RED_MOTION_POLICY_CHOICES = (
    "straight",
    "reactive_evasion",
    "unified",
    "random_masked",
    "ppo",
    "mappo",
    "ppo_evasion",
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


def with_competition_step_limit(scenario: Path, core_args: list[str]) -> list[str]:
    has_explicit_limit = any(
        value == "--max-steps" or value.startswith("--max-steps=")
        for value in core_args
    )
    if has_explicit_limit:
        return core_args

    policy = load_reward_policy(scenario)
    if policy is None:
        return core_args
    return ["--max-steps", str(policy.max_steps), *core_args]


def cmd_run(args: argparse.Namespace) -> int:
    if args.red_policy == "r9_hierarchical_learning" and args.red_motion_policy not in {
        "random_masked", "ppo", "mappo",
    }:
        print(
            "r9_hierarchical_learning requires --red-motion-policy "
            "random_masked, ppo, or mappo",
            file=sys.stderr,
        )
        return 2
    if args.red_policy == "r10_unified" and args.red_motion_policy != "unified":
        print(
            "r10_unified requires --red-motion-policy unified",
            file=sys.stderr,
        )
        return 2
    if args.plan_spec is not None and args.red_policy != "r12_topformer":
        print("--plan-spec requires --red-policy r12_topformer", file=sys.stderr)
        return 2
    if args.plan_spec is not None and args.red_learning_train:
        print("--plan-spec is evaluation-only and cannot train", file=sys.stderr)
        return 2
    try:
        scenario = scenario_path(args.scenario)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    output = ROOT / "results" / "runs" / args.run_id
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "main.py", "--scenario", str(scenario), "--output-dir", str(output)]
    if args.plan_spec is not None:
        command.extend(["--plan-spec", str(Path(args.plan_spec).resolve())])
    # argparse keeps the separator used before REMAINDER.  It is meaningful to
    # this wrapper only; forwarding it makes core/main.py reject its arguments.
    core_args = args.core_args[1:] if args.core_args[:1] == ["--"] else args.core_args
    command.extend(with_competition_step_limit(scenario, core_args))
    env = os.environ.copy()
    env["BLUE_POLICY"] = args.blue_policy
    env["RED_POLICY"] = args.red_policy
    env["RED_MOTION_POLICY"] = args.red_motion_policy
    if args.blue_interceptor_ratio is not None:
        env["BLUE_INTERCEPTOR_RATIO"] = str(args.blue_interceptor_ratio)
    if args.blue_plan_delay is not None:
        env["BLUE_PLAN_DELAY"] = str(args.blue_plan_delay * 1000.0)
    if args.blue_wave_gap is not None:
        env["BLUE_WAVE_GAP"] = str(args.blue_wave_gap * 1000.0)
    if args.blue_wave_size is not None:
        env["BLUE_WAVE_SIZE"] = str(args.blue_wave_size)
    if args.blue_min_cost_decision_interval is not None:
        env["BLUE_MIN_COST_DECISION_INTERVAL"] = str(
            args.blue_min_cost_decision_interval
        )
    reward_policy = load_reward_policy(scenario)
    if reward_policy is not None:
        env["BLUE_ASSET_VALUES"] = json.dumps(
            dict(reward_policy.objective_weights)
        )
    if args.red_learning_model is not None:
        env["RED_LEARNING_MODEL"] = str(Path(args.red_learning_model).resolve())
    if args.red_learning_train:
        env["RED_LEARNING_TRAIN"] = "1"
    if args.seed is not None:
        env["BLUE_POLICY_SEED"] = str(args.seed)
        env["RED_POLICY_SEED"] = str(args.seed)
        env["SIMULATION_SEED"] = str(args.seed)
    return subprocess.call(command, cwd=CORE, env=env)


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
    run_parser.add_argument(
        "--plan-spec",
        help="Canonical PlanSpec token-list JSON for deterministic replay",
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
