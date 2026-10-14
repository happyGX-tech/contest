"""Build PPO batches from trajectories collected by the complete simulator.

The module intentionally contains no simulator substitute.  It consumes the
runtime actor observations and actions emitted by ``LocalAvoidRuntimePolicy``
inside the complete competition environment, reconstructs the behavior-policy
log probabilities, replaces runtime distance proxies with MissionOutcomeTracker
labels, and builds the V10 success-conditioned target-distance reward.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import torch
from torch.distributions import Categorical

from .local_env import LocalAvoidConfig
from .set_attention_ppo import SetAttentionLocalAvoidPPO


SUPPORTED_TRACE_FORMATS = {
    "local-avoid-runtime-trajectory-v1",
    "local-avoid-runtime-trajectory-v2",
    "local-avoid-runtime-trajectory-v3",
}


def load_runtime_trajectories(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load complete-simulator runtime trajectories from gzip JSONL files."""

    trajectories: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"runtime trajectory does not exist: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("format") not in SUPPORTED_TRACE_FORMATS:
                    raise ValueError(
                        f"unsupported trace format at {path}:{line_number}: "
                        f"{payload.get('format')!r}"
                    )
                if not isinstance(payload.get("steps"), list):
                    raise ValueError(f"trace has no step list: {path}:{line_number}")
                trajectories.append(payload)
    return trajectories


def annotate_runtime_trace_outcomes(
    path: str | Path, summary: dict[str, Any]
) -> dict[str, int]:
    """Replace distance proxies with per-platform full-simulator outcomes."""

    trace_path = Path(path)
    trajectories = load_runtime_trajectories([trace_path])
    per_platform = summary.get("outcomes", {}).get("per_platform", {})
    counts: dict[str, int] = {}
    for trajectory in trajectories:
        outcome = per_platform.get(str(trajectory.get("entity_id", -1)))
        if not isinstance(outcome, dict):
            trajectory["terminal_proxy"] = trajectory.get("terminal")
            terminal = "missing_outcome"
            trajectory["terminal"] = terminal
            trajectory["terminal_source"] = "mission_outcome_tracker_missing"
            counts[terminal] = counts.get(terminal, 0) + 1
            continue
        if bool(outcome.get("effective_hit")):
            terminal = "effective_hit"
        elif bool(outcome.get("arrived")):
            terminal = "arrived"
        elif bool(outcome.get("resolved")):
            terminal = "intercepted" if bool(outcome.get("engaged")) else "lost"
        elif bool(outcome.get("alive_at_end")):
            terminal = "incomplete"
        else:
            terminal = "lost"
        trajectory["terminal_proxy"] = trajectory.get("terminal")
        trajectory["terminal"] = terminal
        trajectory["terminal_source"] = "mission_outcome_tracker"
        trajectory["mission_outcome"] = outcome
        counts[terminal] = counts.get(terminal, 0) + 1

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=trace_path.parent, prefix=f".{trace_path.name}.", delete=False
    ) as temporary_stream:
        temporary_path = Path(temporary_stream.name)
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as stream:
            for trajectory in trajectories:
                stream.write(
                    json.dumps(trajectory, ensure_ascii=True, separators=(",", ":"))
                )
                stream.write("\n")
        temporary_path.replace(trace_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return counts


def _critic_from_actor(
    observations: np.ndarray, critic_dim: int
) -> np.ndarray:
    result = np.zeros((len(observations), critic_dim), dtype=np.float32)
    width = min(observations.shape[1], critic_dim)
    result[:, :width] = observations[:, :width]
    return result


def _behavior_outputs(
    policy: SetAttentionLocalAvoidPPO,
    observations: np.ndarray,
    critic_states: np.ndarray,
    actions: np.ndarray,
    action_masks: np.ndarray,
    *,
    batch_size: int = 32768,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("behavior batch_size must be positive")
    normalized_actor = policy._normalize_actor(observations, False)
    normalized_critic = policy._normalize_critic(critic_states, False)
    log_prob_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            end = min(start + batch_size, len(observations))
            actor_tensor = torch.as_tensor(
                normalized_actor[start:end],
                dtype=torch.float32,
                device=policy.device,
            )
            critic_tensor = torch.as_tensor(
                normalized_critic[start:end],
                dtype=torch.float32,
                device=policy.device,
            )
            mask_tensor = torch.as_tensor(
                action_masks[start:end], dtype=torch.bool, device=policy.device
            )
            logits = policy.network.actor_logits(actor_tensor)
            logits = logits.masked_fill(
                ~mask_tensor, torch.finfo(logits.dtype).min
            )
            distribution = Categorical(logits=logits)
            action_tensor = torch.as_tensor(
                actions[start:end], dtype=torch.long, device=policy.device
            )
            log_prob_chunks.append(
                distribution.log_prob(action_tensor).cpu().numpy().astype(np.float32)
            )
            value_chunks.append(
                policy.network.value(critic_tensor).cpu().numpy().astype(np.float32)
            )
    return (
        normalized_actor,
        normalized_critic,
        np.concatenate(log_prob_chunks),
        np.concatenate(value_chunks),
    )


def _v10_raw_rewards(
    observations: np.ndarray,
    next_observation: np.ndarray,
    actions: np.ndarray,
    terminal: str,
    config: LocalAvoidConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Success-conditioned normalized target-distance reduction.

    Consecutive rows provide every intermediate post-step distance.  The v3
    trace carries the final post-step observation for a correct time-limit
    reward and value bootstrap.  A real arrival is treated as zero remaining
    distance, so its undiscounted distance return telescopes to one.  A real
    interception or loss receives zero for the whole trajectory: getting close
    before being destroyed is not a training success.
    """

    rewards = np.zeros(len(observations), dtype=np.float32)
    distances_m = (
        observations[:, 3].astype(np.float64)
        * float(config.goal_distance_scale_m)
    )
    initial_distance_m = max(float(distances_m[0]), 1.0)
    progress_fraction = np.zeros(len(observations), dtype=np.float64)
    if len(observations) > 1:
        progress_fraction[:-1] = (
            distances_m[:-1] - distances_m[1:]
        ) / initial_distance_m

    arrived = terminal in {"arrived", "effective_hit"}
    failed = terminal in {"intercepted", "lost"}
    incomplete = terminal in {"incomplete", "censored"}
    if arrived:
        progress_fraction[-1] = distances_m[-1] / initial_distance_m
    elif failed:
        progress_fraction.fill(0.0)
    elif incomplete and next_observation.size == observations.shape[1]:
        next_distance_m = (
            float(next_observation[3]) * float(config.goal_distance_scale_m)
        )
        progress_fraction[-1] = (
            distances_m[-1] - next_distance_m
        ) / initial_distance_m
    else:
        raise ValueError(f"unsupported V10 terminal outcome: {terminal!r}")

    del actions
    rewards += progress_fraction.astype(np.float32)
    np.clip(
        rewards,
        -float(config.reward_clip),
        float(config.reward_clip),
        out=rewards,
    )
    return rewards, {
        "progress_reward": float(progress_fraction.sum()),
        "successful_distance_reward": (
            float(progress_fraction.sum()) if arrived else 0.0
        ),
        "failed_zero_reward": float(failed),
        "arrival_bonus": 0.0,
        "hit_penalty": 0.0,
        "timeout_penalty": 0.0,
        "switch_penalty": 0.0,
        "reversal_penalty": 0.0,
    }


def _normalize_episode_rewards(
    policy: SetAttentionLocalAvoidPPO,
    raw_rewards: np.ndarray,
    dones: np.ndarray,
    lengths: list[int],
) -> np.ndarray:
    """Apply V8 reward-RMS normalization without mixing episode accumulators."""

    if not policy.config.normalize_reward:
        return raw_rewards.astype(np.float32, copy=True)
    discounted: list[float] = []
    cursor = 0
    for length in lengths:
        running = 0.0
        end = cursor + length
        for index in range(cursor, end):
            running = (
                running
                * float(policy.config.gamma)
                * (1.0 - float(dones[index]))
                + float(raw_rewards[index])
            )
            discounted.append(running)
        cursor = end
    policy.reward_rms.update(np.asarray(discounted, dtype=np.float64))
    return np.clip(
        raw_rewards / np.sqrt(policy.reward_rms.var + 1e-8), -10.0, 10.0
    ).astype(np.float32)


def build_full_sim_rollout(
    policy: SetAttentionLocalAvoidPPO,
    trajectories: Iterable[dict[str, Any]],
    *,
    environment_config: LocalAvoidConfig,
    behavior_batch_size: int = 32768,
    gamma: float | None = None,
    gae_lambda: float | None = None,
    include_incomplete: bool = True,
    require_real_outcomes: bool = True,
    balance_types: bool = False,
    balance_seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Convert complete-simulator trajectories into one on-policy PPO batch.

    Reward is the V10 objective: success-conditioned normalized target-distance
    reduction.  Formal training rejects proxy terminal labels.  Time-limit
    trajectories use their real final next observation for value bootstrap.
    Optionally, PPO samples are resampled to equal H/M/L counts.
    """

    gamma_value = float(policy.config.gamma if gamma is None else gamma)
    lambda_value = float(
        policy.config.gae_lambda if gae_lambda is None else gae_lambda
    )
    usable: list[dict[str, Any]] = []
    excluded_incomplete = 0
    for trajectory in trajectories:
        steps = trajectory.get("steps") or []
        if (
            require_real_outcomes
            and trajectory.get("terminal_source") != "mission_outcome_tracker"
        ):
            raise ValueError(
                "formal full-simulator PPO requires MissionOutcomeTracker "
                f"labels; got terminal={trajectory.get('terminal')!r}, "
                f"source={trajectory.get('terminal_source')!r}"
            )
        if trajectory.get("terminal") == "incomplete" and not include_incomplete:
            excluded_incomplete += 1
            continue
        if len(steps) >= 2:
            usable.append(trajectory)
    if not usable:
        raise ValueError("no complete-simulator trajectory contains at least two steps")

    flat_observations: list[np.ndarray] = []
    flat_actions: list[int] = []
    flat_masks: list[np.ndarray] = []
    flat_type_slots: list[int] = []
    lengths: list[int] = []
    type_counts = {0: 0, 1: 0, 2: 0}
    terminals: dict[str, int] = {}
    for trajectory in usable:
        steps = trajectory["steps"]
        lengths.append(len(steps))
        slot = int(trajectory.get("missile_type_slot", 0))
        type_counts[slot] = type_counts.get(slot, 0) + 1
        terminal = str(trajectory.get("terminal", "unknown"))
        terminals[terminal] = terminals.get(terminal, 0) + 1
        for step in steps:
            flat_observations.append(
                np.asarray(step["observation"], dtype=np.float32)
            )
            flat_actions.append(int(step["action"]))
            flat_masks.append(np.asarray(step["action_mask"], dtype=np.bool_))
            flat_type_slots.append(slot)

    observations = np.stack(flat_observations)
    if observations.shape[1] != policy.config.actor_obs_dim:
        raise ValueError(
            f"trace actor dimension {observations.shape[1]} does not match "
            f"checkpoint {policy.config.actor_obs_dim}"
        )
    actions = np.asarray(flat_actions, dtype=np.int64)
    action_masks = np.stack(flat_masks)
    type_slots = np.asarray(flat_type_slots, dtype=np.int64)
    critic_states = _critic_from_actor(
        observations, int(policy.config.critic_state_dim)
    )
    normalized_actor, normalized_critic, log_probs, values = _behavior_outputs(
        policy,
        observations,
        critic_states,
        actions,
        action_masks,
        batch_size=behavior_batch_size,
    )

    raw_rewards = np.zeros(len(observations), dtype=np.float32)
    dones = np.zeros(len(observations), dtype=np.float32)
    advantages = np.zeros(len(observations), dtype=np.float32)
    returns = np.zeros(len(observations), dtype=np.float32)
    component_totals = {
        "progress_reward": 0.0,
        "successful_distance_reward": 0.0,
        "failed_zero_reward": 0.0,
        "arrival_bonus": 0.0,
        "hit_penalty": 0.0,
        "timeout_penalty": 0.0,
        "switch_penalty": 0.0,
        "reversal_penalty": 0.0,
    }
    cursor = 0
    bootstrap_values: list[float] = []
    bootstrapped_incomplete = 0
    for trajectory, length in zip(usable, lengths):
        end = cursor + length
        episode_observations = observations[cursor:end]
        next_observation = np.asarray(
            trajectory.get("next_observation", []), dtype=np.float32
        )
        episode_rewards, components = _v10_raw_rewards(
            episode_observations,
            next_observation,
            actions[cursor:end],
            str(trajectory.get("terminal", "unknown")),
            environment_config,
        )
        raw_rewards[cursor:end] = episode_rewards
        for key, value in components.items():
            component_totals[key] += value
        incomplete = str(trajectory.get("terminal", "unknown")) in {
            "incomplete", "censored"
        }
        valid_next = next_observation.shape == (observations.shape[1],)
        dones[end - 1] = 0.0 if incomplete and valid_next else 1.0
        if incomplete and valid_next:
            next_critic = _critic_from_actor(
                next_observation.reshape(1, -1),
                int(policy.config.critic_state_dim),
            )
            bootstrap_values.append(float(policy.values(next_critic)[0]))
            bootstrapped_incomplete += 1
        else:
            bootstrap_values.append(0.0)
        cursor = end

    rewards = _normalize_episode_rewards(policy, raw_rewards, dones, lengths)
    cursor = 0
    for length, bootstrap_value in zip(lengths, bootstrap_values):
        end = cursor + length
        gae = 0.0
        for index in range(end - 1, cursor - 1, -1):
            next_value = (
                bootstrap_value if index == end - 1 else float(values[index + 1])
            )
            nonterminal = 1.0 - float(dones[index])
            delta = (
                float(rewards[index])
                + gamma_value * next_value * nonterminal
                - float(values[index])
            )
            gae = delta + gamma_value * lambda_value * nonterminal * gae
            advantages[index] = gae
        returns[cursor:end] = advantages[cursor:end] + values[cursor:end]
        cursor = end

    rollout = {
        "observations": normalized_actor,
        "critic_states": normalized_critic,
        "raw_observations": observations,
        "raw_critic_states": critic_states,
        "actions": actions,
        "action_masks": action_masks,
        "log_probs": log_probs,
        "values": values,
        "rewards": rewards.copy(),
        "raw_rewards": raw_rewards,
        "dones": dones,
        "returns": returns,
        "advantages": advantages,
        "type_slots": type_slots,
    }
    collected_samples = len(observations)
    optimized_type_counts = {
        slot: int(np.sum(type_slots == slot)) for slot in (0, 1, 2)
    }
    if balance_types:
        present = [slot for slot, count in optimized_type_counts.items() if count > 0]
        if len(present) != 3:
            raise ValueError(
                "type-balanced PPO requires H, M, and L samples in every iteration"
            )
        target_count = int(np.median(list(optimized_type_counts.values())))
        rng = np.random.default_rng(balance_seed)
        selected_parts = []
        for slot in (0, 1, 2):
            candidates = np.flatnonzero(type_slots == slot)
            selected_parts.append(
                rng.choice(
                    candidates,
                    size=target_count,
                    replace=len(candidates) < target_count,
                )
            )
        selected = np.concatenate(selected_parts)
        rng.shuffle(selected)
        for key, value in tuple(rollout.items()):
            if isinstance(value, np.ndarray) and value.shape[:1] == (collected_samples,):
                rollout[key] = value[selected]
        optimized_type_counts = {
            slot: int(np.sum(rollout["type_slots"] == slot)) for slot in (0, 1, 2)
        }
    stats = {
        "trajectories": float(len(usable)),
        "samples": float(collected_samples),
        "optimized_samples": float(len(rollout["actions"])),
        "balanced_h_samples": float(optimized_type_counts[0]),
        "balanced_m_samples": float(optimized_type_counts[1]),
        "balanced_l_samples": float(optimized_type_counts[2]),
        "bootstrapped_incomplete": float(bootstrapped_incomplete),
        "mean_trajectory_steps": float(np.mean(lengths)),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "raw_reward_mean": float(raw_rewards.mean()),
        "raw_reward_std": float(raw_rewards.std()),
        "mean_raw_episode_return": float(raw_rewards.sum() / len(usable)),
        "progress_positive_rate": float(np.mean(raw_rewards > 0.0)),
        "h_trajectories": float(type_counts.get(0, 0)),
        "m_trajectories": float(type_counts.get(1, 0)),
        "l_trajectories": float(type_counts.get(2, 0)),
        "arrived_proxy": float(terminals.get("arrived_proxy", 0)),
        "lost_proxy": float(terminals.get("lost_proxy", 0)),
        "incomplete": float(terminals.get("incomplete", 0)),
        "arrived": float(terminals.get("arrived", 0)),
        "effective_hit": float(terminals.get("effective_hit", 0)),
        "intercepted": float(terminals.get("intercepted", 0)),
        "lost": float(terminals.get("lost", 0)),
        "excluded_incomplete": float(excluded_incomplete),
        **component_totals,
    }
    return rollout, stats
