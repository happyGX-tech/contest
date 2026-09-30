"""Mission-aware Stage-3 PPO recovery training on the current speed scale."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policies.red.skills.evasion_ppo.local_env import VectorLocalAvoidEnv
from policies.red.skills.evasion_ppo.local_ppo import (
    LocalAvoidPPO,
    RunningMeanStd,
    collect_rollout,
    evaluate_policy,
    summarize_episodes,
)


PHASES = (
    {
        "name": "single_recovery",
        "bug_spawn_min_m": 8_000.0,
        "bug_spawn_max_m": 16_000.0,
        "bug_speed_mps": 220.0,
        "stage_distribution": (0.0, 0.25, 0.75, 0.0),
        "hard_replay_fraction": 0.0,
    },
    {
        "name": "easy_double_recovery",
        "bug_spawn_min_m": 7_000.0,
        "bug_spawn_max_m": 15_000.0,
        "bug_speed_mps": 230.0,
        "stage_distribution": (0.0, 0.0, 0.35, 0.65),
        "hard_replay_fraction": 0.10,
    },
    {
        "name": "hard_double_transfer",
        "bug_spawn_min_m": 5_000.0,
        "bug_spawn_max_m": 12_000.0,
        "bug_speed_mps": 240.0,
        "stage_distribution": (0.0, 0.0, 0.20, 0.80),
        "hard_replay_fraction": 0.40,
    },
)


def parse_steps(value: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("phase steps must be integers") from error
    if len(values) != 3 or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("provide three positive phase budgets")
    return values


def selection_score(stage3: dict[str, float], stage2: dict[str, float]) -> float:
    action_imbalance = abs(
        stage3.get("left_action_ratio", 0.0)
        - stage3.get("right_action_ratio", 0.0)
    )
    return (
        1.50 * stage3["arrival_rate"]
        + 0.75 * stage3["evade_success_rate"]
        + 0.20 * stage3["recovery_rate"]
        - 0.75 * stage3["hit_rate"]
        - 0.50 * stage3["timeout_rate"]
        - 0.10 * min(stage3["mean_extra_path_length_m"] / 20_000.0, 2.0)
        - 0.05 * action_imbalance
        + 0.20 * stage2["arrival_rate"]
        + 0.10 * stage2["evade_success_rate"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        type=Path,
        default=ROOT / "models/local_avoid_ppo_stage3_selected.pt",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/local_avoid_ppo_v10_mission_s3.pt",
    )
    parser.add_argument(
        "--phase-steps",
        type=parse_steps,
        default=parse_steps("262144,262144,393216"),
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encode-track-source", action="store_true")
    parser.add_argument("--preserve-normalizers", action="store_true")
    parser.add_argument("--peer-track-probability", type=float, default=0.0)
    parser.add_argument("--satellite-track-probability", type=float, default=0.0)
    parser.add_argument("--strict-mission", action="store_true")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "results/local_avoid_ppo/tensorboard_v10_mission_s3",
    )
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if (
        args.peer_track_probability < 0.0
        or args.satellite_track_probability < 0.0
        or args.peer_track_probability + args.satellite_track_probability > 1.0
    ):
        parser.error("track source probabilities must be nonnegative and sum to at most one")

    if args.smoke:
        args.phase_steps = (32, 32, 32)
        args.num_envs = 4
        args.rollout_steps = 8
        args.minibatch_size = 16
        args.ppo_epochs = 1
        args.eval_episodes = 4
        args.eval_interval = 1
        args.disable_tensorboard = True

    policy, previous_config, _payload = LocalAvoidPPO.load(
        args.resume, device=args.device
    )
    policy.config.num_parallel_envs = args.num_envs
    policy.config.rollout_steps = args.rollout_steps
    policy.config.minibatch_size = args.minibatch_size
    policy.config.ppo_epochs = args.ppo_epochs
    policy.config.learning_rate = args.learning_rate
    policy.config.clip_range = 0.10
    policy.config.entropy_coef = 0.01
    policy.config.symmetry_actor_coef = 0.02
    policy.config.symmetry_critic_coef = 0.01
    policy.config.permutation_actor_coef = 0.02
    policy.config.permutation_critic_coef = 0.01
    policy.config.freeze_observation_normalizer = False
    policy.optimizer = torch.optim.Adam(
        policy.network.parameters(), lr=args.learning_rate
    )
    if not args.preserve_normalizers:
        policy.actor_rms = RunningMeanStd((policy.config.actor_obs_dim,))
        policy.critic_rms = RunningMeanStd((policy.config.critic_state_dim,))
    policy.reward_rms = RunningMeanStd(())
    policy.discounted_reward = np.zeros(args.num_envs, dtype=np.float64)
    policy.update_count = 0
    policy.total_steps = 0
    policy.stage = 3

    # Current core commands use 20G/15G/20G at approximately 200/120/100 m/s.
    # The turn rates are the planar a/v approximation, with L capped at 90 deg/s.
    base_config = replace(
        previous_config,
        horizon_steps=900,
        unit_speeds_mps=(200.0, 120.0, 100.0),
        bug_speed_mps=250.0,
        unit_turn_rates_deg_s=(56.1, 70.2, 90.0),
        integrated_turn_actions=True,
        straight_recovery_deg=90.0,
        goal_distance_min_m=40_000.0,
        goal_distance_max_m=80_000.0,
        bug_spawn_min_m=3_000.0,
        bug_spawn_max_m=10_000.0,
        threat_distance_scale_m=10_000.0,
        relative_speed_scale_mps=500.0,
        risk_t_cpa_s=45.0,
        risk_d_cpa_m=3_000.0,
        risk_distance_m=1_000.0,
        dangerous_spawn_only=True,
        dangerous_max_cpa_m=500.0,
        actor_risk_gate=True,
        encode_track_source=args.encode_track_source,
        peer_track_probability=args.peer_track_probability,
        satellite_track_probability=args.satellite_track_probability,
        terminate_on_evade=False,
        retire_threats_on_evade=True,
        maneuver_burst_limit_steps=2,
        progress_scale=1.5,
        hit_penalty=150.0,
        evade_bonus=25.0,
        arrival_bonus=150.0,
        timeout_penalty=100.0,
        deviation_penalty=0.02,
        oscillation_penalty=0.50,
        recovery_penalty=0.75,
        heading_error_penalty=1.0,
        excess_maneuver_penalty=3.0,
        recovery_bonus=20.0,
        recovery_heading_tolerance_deg=5.0,
        safety_improvement_scale=25.0,
        unsafe_straight_penalty=0.50,
        reward_clip=200.0,
        stage3_same_side_probability=0.15,
    )
    if args.strict_mission:
        base_config = replace(
            base_config,
            progress_scale=2.0,
            hit_penalty=200.0,
            evade_bonus=10.0,
            arrival_bonus=250.0,
            timeout_penalty=200.0,
            recovery_penalty=1.5,
            heading_error_penalty=2.0,
            excess_maneuver_penalty=5.0,
            recovery_bonus=30.0,
        )
    if args.smoke:
        base_config = replace(
            base_config,
            horizon_steps=50,
            goal_distance_min_m=2_000.0,
            goal_distance_max_m=4_000.0,
            bug_spawn_min_m=1_000.0,
            bug_spawn_max_m=2_000.0,
        )
    full_config = replace(
        base_config,
        stage3_simultaneous_probability=0.34,
        stage3_pincer_probability=0.33,
    )

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = args.log_dir / f"{args.model.stem}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "training_history.jsonl"
    history_file = history_path.open("a", encoding="utf-8")
    writer = None
    if not args.disable_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(run_dir))

    best_path = args.model.with_name(f"{args.model.stem}_best{args.model.suffix}")
    best_score = float("-inf")
    final_metrics: dict[str, float] = {}

    def evaluate_and_save(phase: str, phase_config, offset: int) -> None:
        nonlocal best_score, final_metrics
        stage3 = evaluate_policy(
            policy,
            full_config,
            stage=3,
            episodes=args.eval_episodes,
            seed=args.seed + 900_000 + offset,
        )
        stage2 = evaluate_policy(
            policy,
            full_config,
            stage=2,
            episodes=args.eval_episodes,
            seed=args.seed + 1_100_000 + offset,
        )
        straight = evaluate_policy(
            policy,
            full_config,
            stage=3,
            episodes=args.eval_episodes,
            seed=args.seed + 900_000 + offset,
            straight_baseline=True,
        )
        curriculum = evaluate_policy(
            policy,
            phase_config,
            stage=3,
            episodes=args.eval_episodes,
            seed=args.seed + 1_300_000 + offset,
        )
        score = selection_score(stage3, stage2) + 0.25 * selection_score(
            curriculum, stage2
        )
        final_metrics = stage3
        event = {
            "phase": phase,
            "total_steps": policy.total_steps,
            "stage3": stage3,
            "stage2_retention": stage2,
            "straight_stage3": straight,
            "curriculum_stage3": curriculum,
            "selection_score": score,
        }
        print("LOCAL_AVOID_V10_EVAL " + json.dumps(event), flush=True)
        history_file.write(json.dumps({"evaluation": event}) + "\n")
        history_file.flush()
        if writer is not None:
            for key, value in stage3.items():
                writer.add_scalar(f"eval_stage3/{key}", value, policy.total_steps)
            writer.add_scalar("eval_stage3/selection_score", score, policy.total_steps)
        if score > best_score:
            best_score = score
            policy.save(best_path, environment_config=full_config, metrics=event)

    try:
        evaluate_and_save("initial", full_config, 0)
        for phase_index, (phase, budget) in enumerate(
            zip(PHASES, args.phase_steps)
        ):
            phase_config = replace(
                base_config,
                bug_spawn_min_m=phase["bug_spawn_min_m"],
                bug_spawn_max_m=phase["bug_spawn_max_m"],
                bug_speed_mps=phase["bug_speed_mps"],
                threat_distance_scale_m=phase["bug_spawn_max_m"],
                stage3_simultaneous_probability=0.34,
                stage3_pincer_probability=0.33,
            )
            hard_count = round(args.num_envs * phase["hard_replay_fraction"])
            environment = VectorLocalAvoidEnv(
                args.num_envs,
                [full_config] * hard_count
                + [phase_config] * (args.num_envs - hard_count),
                stage=3,
                seed=args.seed + phase_index * 100_000,
                stage_distribution=phase["stage_distribution"],
            )
            actor_observation, critic_state = environment.reset()
            phase_start = policy.total_steps
            completed_episodes: list[dict] = []
            while policy.total_steps - phase_start < budget:
                rollout, actor_observation, critic_state, completed = collect_rollout(
                    policy, environment, actor_observation, critic_state
                )
                completed_episodes.extend(completed)
                train_metrics = policy.update(rollout)
                skill = summarize_episodes(completed_episodes[-5000:])
                record = {
                    "phase": phase["name"],
                    "total_steps": policy.total_steps,
                    "phase_steps": policy.total_steps - phase_start,
                    "train": train_metrics,
                    "skill": skill,
                }
                print("LOCAL_AVOID_V10_TRAIN " + json.dumps(record), flush=True)
                history_file.write(json.dumps(record) + "\n")
                history_file.flush()
                if policy.update_count % args.eval_interval == 0:
                    evaluate_and_save(
                        phase["name"], phase_config, policy.update_count
                    )
            evaluate_and_save(
                phase["name"], phase_config, 50_000 + phase_index
            )

        policy.save(args.model, environment_config=full_config, metrics=final_metrics)
    finally:
        history_file.close()
        if writer is not None:
            writer.close()

    print(f"LOCAL_AVOID_V10_MODEL {args.model}")
    print(f"LOCAL_AVOID_V10_BEST {best_path}")
    print(f"LOCAL_AVOID_V10_HISTORY {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
