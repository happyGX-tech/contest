"""Single command-line entry point for the competition simulation."""

from __future__ import annotations

import argparse
from datetime import datetime
from functools import partial
import json
import logging
import math
from pathlib import Path
import random
import time
from urllib.parse import urlparse

from blue_defense import BLUE_POLICY_CHOICES, BlueDefenseConfig
from scenarios import ScenarioBundle, load_scenario, resolve_deployment_constraints
from policies.red import RED_POLICY_CHOICES
from user_agents.red.learning.runtime import LEARNING_MOTION_CHOICES


ROOT = Path(__file__).resolve().parent
RED_PLATFORM_TYPES = (21000, 21001, 21002)
RED_MOTION_POLICY_CHOICES = ("straight", "reactive_evasion", *LEARNING_MOTION_CHOICES)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number of seconds")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Competition simulation")
    scenario = parser.add_argument_group("Scenario and execution")
    scenario.add_argument("--scenario", default="default",
                          help="default, launch20/easy/E01, a JSON path, or an HTTP(S) URL")
    scenario.add_argument("--deployment-plan", type=Path,
                          help="Optional validated grid deployment plan JSON")
    scenario.add_argument("--total-rounds", type=positive_int, default=1,
                          help="Number of rounds (default: 1)")
    scenario.add_argument("--max-steps", type=positive_int,
                          help="Round limit (default: scenario limit, or 1000 without scoring metadata)")
    scenario.add_argument("--render-mode", choices=("human", "none"), default="human",
                          help="Rendering mode (default: human)")
    scenario.add_argument("--seed", type=int,
                          help="Shared random seed for simulation and policies (0 through 4294967295)")

    red = parser.add_argument_group("Red agents")
    red.add_argument("--red-policy", choices=RED_POLICY_CHOICES, default="r0_random")
    red.add_argument("--red-motion-policy", choices=RED_MOTION_POLICY_CHOICES,
                     default="reactive_evasion")
    red.add_argument("--red-learning-model", type=Path,
                     help="PPO/MAPPO checkpoint; required for inference and training output")
    red.add_argument("--red-learning-train", action="store_true",
                     help="Enable PPO/MAPPO training")

    defaults = BlueDefenseConfig()
    blue = parser.add_argument_group("Built-in blue defense (all time options are in seconds)")
    blue.add_argument("--blue-policy", choices=BLUE_POLICY_CHOICES, default=defaults.policy_name)
    blue.add_argument("--blue-interceptor-ratio", type=positive_int,
                      default=defaults.max_shots_per_target,
                      help="Maximum assigned interceptors per target (default: 2)")
    blue.add_argument("--blue-plan-delay", type=nonnegative_seconds,
                      default=defaults.planning_delay_ms / 1000,
                      help="B7/B9 planning delay in seconds (default: 1)")
    blue.add_argument("--blue-wave-gap", type=nonnegative_seconds,
                      default=None,
                      help="B8/B9 wave interval in seconds (default: B8 10, B9 5)")
    blue.add_argument("--blue-wave-size", type=positive_int, default=defaults.wave_size,
                      help="Interceptors per target in each wave (default: 1)")
    blue.add_argument("--blue-min-cost-decision-interval", type=nonnegative_seconds,
                      default=defaults.min_cost_decision_interval_ms / 1000,
                      help="B3 decision interval in seconds (default: 0.005)")
    blue.add_argument("--blue-joint-decision-interval", type=nonnegative_seconds,
                      default=defaults.joint_decision_interval_ms / 1000,
                      help="B4/B5/B6 decision interval in seconds (default: 5)")
    blue.add_argument("--blue-reset-rng-on-episode", action="store_true",
                      help="Restart the blue random stream from its seed each round")

    output = parser.add_argument_group("Output")
    output.add_argument("--output-dir", type=Path, default=ROOT / "results/runs/manual",
                        help="Result directory; a timestamped subdirectory is created (default: results/runs/manual)")
    output.add_argument("--batch-size", type=positive_int, default=100)
    output.add_argument("--log-frequency", type=positive_int, default=1,
                        help="Write state once every N steps (default: 1)")
    output.add_argument("--disable-log-color", action="store_false",
                        dest="enable_log_color", default=True)
    output.add_argument("--verbose", action="store_true")
    for name in ("config", "state", "event", "ai-action"):
        output.add_argument(f"--enable-{name}", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def prepare_run(args: argparse.Namespace) -> ScenarioBundle:
    """Resolve caller-relative inputs and validate settings before loading native code."""
    bundle = load_scenario(args.scenario)
    if args.max_steps is None:
        args.max_steps = bundle.reward_policy.max_steps if bundle.reward_policy else 1000
    for name in ("output_dir", "deployment_plan", "red_learning_model"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.deployment_plan is not None and not args.deployment_plan.is_file():
        raise ValueError(f"Deployment plan does not exist: {args.deployment_plan}")
    if args.seed is not None and not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("--seed must be between 0 and 4294967295")
    if args.red_policy == "r9_hierarchical_learning" and args.red_motion_policy not in LEARNING_MOTION_CHOICES:
        raise ValueError("r9_hierarchical_learning requires --red-motion-policy random_masked, ppo, or mappo")
    if args.red_learning_train and args.red_motion_policy not in {"ppo", "mappo"}:
        raise ValueError("--red-learning-train requires --red-motion-policy ppo or mappo")
    if args.red_motion_policy in {"ppo", "mappo", "local_avoid_s3"}:
        if args.red_learning_model is None:
            raise ValueError("--red-learning-model is required for PPO/MAPPO/local_avoid_s3")
        if not args.red_learning_train and not args.red_learning_model.is_file():
            raise ValueError(f"Learning checkpoint does not exist: {args.red_learning_model}")
    elif args.red_learning_model is not None:
        raise ValueError("--red-learning-model requires --red-motion-policy ppo/mappo/local_avoid_s3")
    return bundle


def build_blue_config(args: argparse.Namespace, bundle: ScenarioBundle) -> BlueDefenseConfig:
    """Convert the single CLI configuration to the engine's millisecond units."""
    return BlueDefenseConfig(
        policy_name=args.blue_policy,
        max_shots_per_target=args.blue_interceptor_ratio,
        seed=args.seed,
        min_cost_decision_interval_ms=args.blue_min_cost_decision_interval * 1000,
        joint_decision_interval_ms=args.blue_joint_decision_interval * 1000,
        planning_delay_ms=args.blue_plan_delay * 1000,
        wave_gap_ms=None if args.blue_wave_gap is None else args.blue_wave_gap * 1000,
        wave_size=args.blue_wave_size,
        reset_rng_on_episode=args.blue_reset_rng_on_episode,
        asset_values=dict(bundle.reward_policy.objective_weights) if bundle.reward_policy else {},
    )


def read_profile(source: str):
    """Read the engine profile from a local file or an HTTP(S) endpoint."""
    from envengine import Profile

    if urlparse(source).scheme in {"http", "https"}:
        import requests

        response = requests.get(source, timeout=30)
        response.raise_for_status()
        data = response.json()
        return Profile.from_dict(data.get("data", data))
    path = Path(source)
    if not path.is_file():
        raise ValueError(f"Scenario file does not exist: {source}")
    return Profile.from_dict(json.loads(path.read_text(encoding="utf-8")))


def run_simulation(args: argparse.Namespace, bundle: ScenarioBundle) -> None:
    # Native modules are loaded only after argument validation; --help stays lightweight.
    import numpy as np

    from blue_defense import BlueDefenseController
    from envengine import TrainingEnv
    from envengine.sdk.log import LogManager
    from envengine.sdk.writer import WriteConfig, init_writer, write_immediately
    from evaluation import RunSummary
    from scenarios.cases import RewardTracker
    from user_agents.red import AttackMissileAgent, DeployAgent
    from policies.red import (
        Position, RedBaselineCommander, initial_targets_from_observation,
    )
    from user_agents.red.learning import build_learning_motion_policy

    if args.seed is None:
        args.seed = random.SystemRandom().randrange(2**32)
    random.seed(args.seed)
    np.random.seed(args.seed)
    profile = read_profile(str(bundle.path))
    coordinates, coordinates_hm, launch_altitude = resolve_deployment_constraints(
        profile, bundle.deployment_area, args.deployment_plan,
    )
    blue_config = build_blue_config(args, bundle)
    hierarchical_learning = args.red_policy == "r9_hierarchical_learning"
    learning_policy = None
    if args.red_motion_policy in LEARNING_MOTION_CHOICES:
        learning_policy = build_learning_motion_policy(
            args.red_motion_policy,
            seed=args.seed,
            model_path=str(args.red_learning_model) if args.red_learning_model else None,
            max_steps=args.max_steps,
            training=args.red_learning_train,
            observation_dim=90 if hierarchical_learning else 85,
        )

    LogManager(color_enabled=args.enable_log_color)
    write_config = WriteConfig(
        output_dir=str(args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")),
        log_frequency=args.log_frequency,
        verbose=args.verbose,
        batch_size=args.batch_size,
        enable_config=args.enable_config,
        enable_state=args.enable_state,
        enable_event=args.enable_event,
        enable_ai_action=args.enable_ai_action,
    )
    writer = init_writer(write_config)
    env = None
    try:
        logging.info("Scenario: %s; rounds: %s; steps: %s; seed: %s",
                     bundle.path, args.total_rounds, args.max_steps, args.seed)
        logging.info("Policies: red=%s, motion=%s, blue=%s",
                     args.red_policy, args.red_motion_policy, blue_config.policy_name)
        env = TrainingEnv(profile, render_mode=args.render_mode)
        env.max_steps = args.max_steps
        env.set_deployment_constraints(coordinates, coordinates_hm, launch_altitude)
        simulators = env.engine.simulator_factory.get_all_simulators()
        for simulator in env.engine.simulator_factory.get_simulators_by_type(35000):
            simulator.set_defense_controller(BlueDefenseController(blue_config))

        public_observation = env.get_public_targets_observation()
        public_targets = initial_targets_from_observation(public_observation)
        area = profile.imagineProfile.mapArea
        env.set_team_controller_factory(partial(
            RedBaselineCommander,
            public_targets,
            policy_name=args.red_policy,
            seed=args.seed,
            search_polygon=(
                Position(area.lonMin, area.latMin), Position(area.lonMax, area.latMin),
                Position(area.lonMax, area.latMax), Position(area.lonMin, area.latMax),
            ),
        ))
        if args.red_learning_train:
            env.set_shared_learning_policy(learning_policy)
        for index, simulator in enumerate(simulators):
            entity = simulator.entity_ext.entity
            if entity.entityType in RED_PLATFORM_TYPES:
                env.agent_manager.register_agent(AttackMissileAgent(
                    index + 1, entity.id, public_observation,
                    commander=None,
                    motion_policy=args.red_motion_policy,
                    learning_policy=learning_policy,
                    learning_max_steps=args.max_steps,
                    hierarchical_learning=hierarchical_learning,
                ))

        if args.deployment_plan is not None:
            from user_agents.red.grid_deploy_agent import GridDeployAgent

            plan = json.loads(args.deployment_plan.read_text(encoding="utf-8"))
            entities = {
                item.entity.id: {
                    "type": item.entity.entityType,
                    "position": {"alt": launch_altitude if launch_altitude is not None else item.entity.lla.z},
                }
                for item in profile.imagineProfile.entityList
                if item.entity.entityType in RED_PLATFORM_TYPES
            }
            deploy_agent = GridDeployAgent(-1, -1, {"entities": entities}, coordinates, coordinates_hm, plan)
        else:
            deploy_agent = DeployAgent(-1, -1, {}, coordinates, coordinates_hm)
        env.agent_manager.register_agent(deploy_agent)

        for round_index in range(args.total_rounds):
            initial_observation = env.reset()
            summary_builder = RunSummary(
                scenario=bundle.reward_policy.scenario_id if bundle.reward_policy else str(args.scenario),
                policies={"red": args.red_policy, "red_motion": args.red_motion_policy, "blue": blue_config.policy_name},
                reward_tracker=RewardTracker(bundle.reward_policy) if bundle.reward_policy else None,
            )
            summary_builder.start(initial_observation)
            final_observation = initial_observation
            termination_reason = "time_limit"
            env.red_model_deploy()
            start_time = time.perf_counter()
            for step in range(1, args.max_steps + 1):
                final_observation, _, done, _ = env.step()
                summary_builder.update(step, final_observation)
                if step % 200 == 0:
                    logging.info("Round %s, step %s", round_index + 1, step)
                if done:
                    termination_reason = "time_limit" if step >= args.max_steps else "environment_done"
                    break
            write_immediately()
            red_launched = sum(
                getattr(agent, "launch_step", -1) >= 0
                for agent in env.agent_manager.get_all_agents()
            )
            summary = summary_builder.build(
                final_observation, termination_reason=termination_reason, red_launched=red_launched,
            )
            filename = "summary.json" if args.total_rounds == 1 else f"summary_round_{round_index + 1}.json"
            path = summary_builder.write(write_config.output_dir, summary, filename)
            print("FINAL_SUMMARY " + json.dumps(summary, ensure_ascii=False))
            logging.info("Round %s finished in %.3f seconds; summary: %s",
                         round_index + 1, time.perf_counter() - start_time, path)

        if args.red_learning_train:
            args.red_learning_model.parent.mkdir(parents=True, exist_ok=True)
            learning_policy.save(str(args.red_learning_model))
            logging.info("Learning checkpoint saved: %s", args.red_learning_model)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            writer.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        bundle = prepare_run(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    run_simulation(args, bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
