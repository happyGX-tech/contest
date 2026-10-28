"""A/B training driver for the S3 avoidance skill.

One entry point for the whole experiment series (PPO implementation details, GRU,
Transformer): every arm is the same code path with a different flag set, so two arms
differ in exactly the flag under test and nothing else.

Protocol (see local_plan/10_PPO_trick审计与实验计划.md §六):
- warm start from a shared checkpoint, same seed, same env budget for every arm;
- the full three-phase V10/V13 curriculum by default (1,048,576 env steps);
- a checkpoint at the end of every phase, so a cheaper screen can reuse the phase-2
  endpoint instead of paying for phase 3;
- no engine required and no writes outside --out-dir.

Config is imported from `tools/train_local_avoid_ppo_v10_mission_s3.py` (PHASES) and
mirrors its `base_config`, so the arms stay in lockstep with the real trainer.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policies.red.skills.evasion_ppo.local_env import (  # noqa: E402
    LocalAvoidConfig,
    VectorLocalAvoidEnv,
)
from policies.red.skills.evasion_ppo.local_ppo import (  # noqa: E402
    LocalAvoidPPO,
    RunningMeanStd,
    collect_rollout,
    summarize_episodes,
)
from tools.train_local_avoid_ppo_calib import base_config  # noqa: E402
from tools.train_local_avoid_ppo_v10_mission_s3 import PHASES  # noqa: E402


def parse_phase_steps(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("phase steps must be integers") from error
    if not steps or any(item < 0 for item in steps):
        raise argparse.ArgumentTypeError("provide non-negative phase budgets")
    return steps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True,
                        help="shared warm-start checkpoint for every arm")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--phase-steps", type=parse_phase_steps,
                        default=(262_144, 262_144, 393_216),
                        help="per-phase env-step budgets; a 0 skips the phase")
    parser.add_argument("--strict-mission", action="store_true", default=True)
    parser.add_argument("--no-strict-mission", dest="strict_mission",
                        action="store_false")
    parser.add_argument("--preserve-normalizers", action="store_true")
    parser.add_argument("--peer-track-probability", type=float, default=0.25)
    parser.add_argument("--satellite-track-probability", type=float, default=0.50)
    # --- the switches under test; every default reproduces the legacy recipe ---
    parser.add_argument("--target-kl", type=float, default=0.0)
    parser.add_argument("--clip-range-value", type=float, default=None)
    parser.add_argument("--orthogonal-init", action="store_true")
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--anneal-lr", action="store_true",
                        help="linear LR decay to 0 over --phase-steps' total")
    parser.add_argument("--advantage-standardize", action="store_true")
    parser.add_argument("--reward-clip", type=float, default=None,
                        help="clip the normalised discounted return")
    parser.add_argument("--entropy-coef-final", type=float, default=None)
    parser.add_argument(
        "--tensorboard-logdir",
        type=Path,
        default=None,
        help="TensorBoard event directory; defaults to results/tensorboard/<run-name>",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=0,
        help="Frame-window length for the GRU temporal model; 0 keeps the MLP.",
    )
    parser.add_argument(
        "--use-attention", action="store_true",
        help="Add a self-attention block over the frame window.",
    )
    parser.add_argument(
        "--attention-layers", type=int, default=1,
        help="Transformer depth of the window attention block.",
    )
    parser.add_argument(
        "--gate-pooling", action="store_true",
        help="Learned temporal gate pooling instead of mean pooling.",
    )
    return parser.parse_args(argv)


def apply_flags(policy: LocalAvoidPPO, args: argparse.Namespace, total_steps: int) -> None:
    policy.config.target_kl = args.target_kl
    policy.config.clip_range_value = args.clip_range_value
    policy.config.dual_clip = args.dual_clip
    policy.config.orthogonal_init = args.orthogonal_init
    policy.config.total_steps = total_steps if args.anneal_lr else 0
    policy.config.advantage_standardize = args.advantage_standardize
    policy.config.reward_clip = args.reward_clip
    policy.config.entropy_coef_final = (
        args.entropy_coef_final
        if args.entropy_coef_final is not None
        else policy.config.entropy_coef
    )


def build_environment(args: argparse.Namespace, phase: dict, base: LocalAvoidConfig):
    full_config = replace(
        base,
        stage3_simultaneous_probability=0.34,
        stage3_pincer_probability=0.33,
    )
    phase_config = replace(
        full_config,
        bug_spawn_min_m=phase["bug_spawn_min_m"],
        bug_spawn_max_m=phase["bug_spawn_max_m"],
        bug_speed_mps=phase["bug_speed_mps"],
        threat_distance_scale_m=phase["bug_spawn_max_m"],
    )
    hard_count = round(args.num_envs * phase["hard_replay_fraction"])
    environment = VectorLocalAvoidEnv(
        args.num_envs,
        [full_config] * hard_count + [phase_config] * (args.num_envs - hard_count),
        stage=3,
        seed=args.seed,
        stage_distribution=phase["stage_distribution"],
    )
    return environment, full_config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.out_dir / "training_history.jsonl"
    history = history_path.open("a", encoding="utf-8")
    tensorboard_dir = args.tensorboard_logdir or (
        ROOT / "results" / "tensorboard" / args.run_name
    )
    tensorboard = SummaryWriter(log_dir=str(tensorboard_dir))
    settings = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    settings["tensorboard_logdir"] = str(tensorboard_dir)
    settings["started"] = datetime.now().isoformat(timespec="seconds")
    (args.out_dir / "configuration.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    overrides = None
    if args.sequence_length > 0:
        overrides = {
            "sequence_length": int(args.sequence_length),
            "use_attention": bool(args.use_attention),
            "window_attention_layers": int(args.attention_layers),
            "window_gate_pooling": bool(args.gate_pooling),
        }
    policy, _previous_config, _payload = LocalAvoidPPO.load(
        args.resume, device=args.device, overrides=overrides
    )
    # The V13 recipe: symmetry/permutation consistency coefficients and a fresh Adam
    # and reward normaliser, keeping the observation normalisers warm.
    policy.config.permutation_actor_coef = 0.02
    policy.config.permutation_critic_coef = 0.01
    policy.config.freeze_observation_normalizer = False
    policy.optimizer = torch.optim.Adam(policy.network.parameters(), lr=args.learning_rate)
    if not args.preserve_normalizers:
        policy.actor_rms = RunningMeanStd((policy.config.actor_obs_dim,))
        policy.critic_rms = RunningMeanStd((policy.config.critic_state_dim,))
    policy.reward_rms = RunningMeanStd(())
    policy.discounted_reward = np.zeros(args.num_envs, dtype=np.float64)
    policy.update_count = 0
    policy.total_steps = 0
    policy.stage = 3

    total_steps = int(sum(args.phase_steps))
    apply_flags(policy, args, total_steps)

    base = replace(
        base_config(args.strict_mission),
        encode_track_source=False,
        peer_track_probability=args.peer_track_probability,
        satellite_track_probability=args.satellite_track_probability,
    )

    per_update = args.num_envs * args.rollout_steps
    print(
        f"LOCAL_AB_START run={args.run_name} total_steps={total_steps} "
        f"per_update={per_update} device={args.device} out={args.out_dir}",
        flush=True,
    )

    for phase_index, phase in enumerate(PHASES):
        budget = args.phase_steps[phase_index] if phase_index < len(args.phase_steps) else 0
        if budget <= 0:
            print(f"LOCAL_AB_PHASE_SKIP phase={phase['name']}", flush=True)
            continue
        environment, full_config = build_environment(args, phase, base)
        actor_observation, critic_state = environment.reset()
        updates = max(1, budget // per_update)
        completed: list[dict] = []
        for update_index in range(1, updates + 1):
            rollout, actor_observation, critic_state, done = collect_rollout(
                policy, environment, actor_observation, critic_state
            )
            completed.extend(done)
            train_metrics = policy.update(rollout)
            skill = summarize_episodes(completed[-5000:])
            record = {
                "run": args.run_name,
                "phase": phase["name"],
                "phase_update": update_index,
                "update": policy.update_count,
                "total_steps": policy.total_steps,
                "train": train_metrics,
                "skill": skill,
            }
            history.write(json.dumps(record) + "\n")
            history.flush()
            tensorboard.add_scalar("phase", phase_index + 1, policy.update_count)
            tensorboard.add_scalar("train/total_steps", policy.total_steps, policy.update_count)
            for key in (
                "policy_loss",
                "value_loss",
                "entropy",
                "explained_variance",
                "approx_kl",
                "annealing_scale",
            ):
                value = train_metrics.get(key)
                if value is not None:
                    tensorboard.add_scalar(f"train/{key}", float(value), policy.update_count)
            for key in (
                "arrival_rate",
                "evade_success_rate",
                "hit_rate",
                "recovery_rate",
                "timeout_rate",
                "mean_extra_path_length_m",
            ):
                value = skill.get(key)
                if value is not None:
                    tensorboard.add_scalar(f"skill/{key}", float(value), policy.update_count)
            print(
                "LOCAL_AB_TRAIN " + json.dumps({
                    "run": args.run_name,
                    "phase": phase["name"],
                    "update": policy.update_count,
                    "total_steps": policy.total_steps,
                    "policy_loss": train_metrics.get("policy_loss"),
                    "value_loss": train_metrics.get("value_loss"),
                    "entropy": train_metrics.get("entropy"),
                    "explained_variance": train_metrics.get("explained_variance"),
                    "approx_kl": train_metrics.get("approx_kl"),
                    "annealing_scale": train_metrics.get("annealing_scale"),
                    "kl_stopped_early": train_metrics.get("kl_stopped_early"),
                    "arrival_rate_recent": skill.get("arrival_rate"),
                    "evade_rate_recent": skill.get("evade_success_rate"),
                }),
                flush=True,
            )
        ckpt = args.out_dir / f"ckpt_phase{phase_index + 1}_{phase['name']}_steps{policy.total_steps}.pt"
        policy.save(ckpt, environment_config=full_config, metrics=record)
        print(f"LOCAL_AB_CKPT {ckpt}", flush=True)

    history.close()
    tensorboard.flush()
    tensorboard.close()
    print(
        f"LOCAL_AB_DONE run={args.run_name} updates={policy.update_count} "
        f"steps={policy.total_steps}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
