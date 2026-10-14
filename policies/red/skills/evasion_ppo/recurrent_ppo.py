"""Recurrent asymmetric PPO with a frozen feed-forward safety anchor."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .local_env import ACTION_COUNT, LocalAvoidConfig, STRAIGHT, VectorLocalAvoidEnv
from .local_ppo import LocalAvoidPPO, RunningMeanStd, summarize_episodes


ActionMaskFn = Callable[[np.ndarray], np.ndarray]
ExpertActionFn = Callable[[Any, np.ndarray], int]


@dataclass
class RecurrentPPOConfig:
    actor_obs_dim: int = 33
    critic_state_dim: int = 43
    action_dim: int = ACTION_COUNT
    recurrent_hidden_dim: int = 128
    learning_rate: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.10
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    expert_coef: float = 0.20
    anchor_kl_coef: float = 0.05
    max_grad_norm: float = 0.5
    rollout_steps: int = 128
    env_minibatch_size: int = 16
    ppo_epochs: int = 4
    num_parallel_envs: int = 256
    normalize_observation: bool = True
    normalize_reward: bool = True
    freeze_observation_normalizer: bool = True
    freeze_base_actor: bool = True
    seed: int = 20261006
    device: str = "auto"


class RecurrentLocalAvoidActorCritic(nn.Module):
    """Preserve the MLP actor and learn a recurrent residual over its features."""

    def __init__(
        self,
        config: RecurrentPPOConfig,
        *,
        actor_backbone: nn.Module,
        actor_head: nn.Linear,
        critic: nn.Module,
    ) -> None:
        super().__init__()
        self.actor_backbone = actor_backbone
        self.actor_head = actor_head
        self.actor_gru = nn.GRUCell(
            actor_head.in_features, config.recurrent_hidden_dim
        )
        self.recurrent_head = nn.Linear(config.recurrent_hidden_dim, config.action_dim)
        nn.init.zeros_(self.recurrent_head.weight)
        nn.init.zeros_(self.recurrent_head.bias)
        self.critic = critic
        if config.freeze_base_actor:
            for parameter in self.actor_backbone.parameters():
                parameter.requires_grad_(False)
            for parameter in self.actor_head.parameters():
                parameter.requires_grad_(False)

    def actor_step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        episode_start: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if episode_start is not None:
            hidden = hidden * (~episode_start.bool()).to(hidden.dtype).unsqueeze(-1)
        features = self.actor_backbone(observation)
        hidden = self.actor_gru(features, hidden)
        base_logits = self.actor_head(features)
        logits = base_logits + self.recurrent_head(hidden)
        return logits, hidden, base_logits

    def actor_sequence(
        self,
        observations: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = initial_hidden
        logits = []
        base_logits = []
        for step in range(observations.shape[0]):
            current, hidden, base = self.actor_step(
                observations[step], hidden, episode_starts[step]
            )
            logits.append(current)
            base_logits.append(base)
        return torch.stack(logits), torch.stack(base_logits)

    def value(self, critic_state: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_state).squeeze(-1)


class RecurrentLocalAvoidPPO:
    algorithm_name = "local_avoid_recurrent_ppo_v2"

    def __init__(
        self,
        config: RecurrentPPOConfig,
        network: RecurrentLocalAvoidActorCritic,
    ) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        device_name = config.device
        if device_name == "auto":
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.network = network.to(self.device)
        self.optimizer = torch.optim.Adam(
            (parameter for parameter in self.network.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
        )
        self.actor_rms = RunningMeanStd((config.actor_obs_dim,))
        self.critic_rms = RunningMeanStd((config.critic_state_dim,))
        self.reward_rms = RunningMeanStd(())
        self.discounted_reward = np.zeros(config.num_parallel_envs, dtype=np.float64)
        self.update_count = 0
        self.total_steps = 0
        self.stage = 3

    @classmethod
    def from_feedforward(cls, path: str | Path, *, device: str = "auto"):
        base, environment, payload = LocalAvoidPPO.load(path, device="cpu")
        actor_layers = list(base.network.actor.children())
        if len(actor_layers) < 2 or not isinstance(actor_layers[-1], nn.Linear):
            raise ValueError("feed-forward actor does not expose a linear output head")
        config = RecurrentPPOConfig(
            actor_obs_dim=base.config.actor_obs_dim,
            critic_state_dim=base.config.critic_state_dim,
            action_dim=base.config.action_dim,
            gamma=base.config.gamma,
            gae_lambda=base.config.gae_lambda,
            device=device,
        )
        network = RecurrentLocalAvoidActorCritic(
            config,
            actor_backbone=nn.Sequential(*[copy.deepcopy(layer) for layer in actor_layers[:-1]]),
            actor_head=copy.deepcopy(actor_layers[-1]),
            critic=copy.deepcopy(base.network.critic),
        )
        policy = cls(config, network)
        policy.actor_rms.load_state_dict(base.actor_rms.state_dict())
        policy.critic_rms.load_state_dict(base.critic_rms.state_dict())
        policy.reward_rms.load_state_dict(base.reward_rms.state_dict())
        return policy, environment, payload

    def initial_hidden(self, batch_size: int) -> np.ndarray:
        return np.zeros((int(batch_size), self.config.recurrent_hidden_dim), dtype=np.float32)

    def _normalize_actor(self, values: np.ndarray, update: bool) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not self.config.normalize_observation:
            return values
        if update:
            self.actor_rms.update(values)
        return self.actor_rms.normalize(values)

    def _normalize_critic(self, values: np.ndarray, update: bool) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not self.config.normalize_observation:
            return values
        if update:
            self.critic_rms.update(values)
        return self.critic_rms.normalize(values)

    def act(
        self,
        actor_observation: np.ndarray,
        critic_state: np.ndarray,
        hidden: np.ndarray | None = None,
        *,
        episode_start: np.ndarray | bool | None = None,
        deterministic: bool = False,
        update_normalizer: bool = False,
        action_masks: np.ndarray | None = None,
    ):
        raw_actor = np.asarray(actor_observation, dtype=np.float32)
        raw_critic = np.asarray(critic_state, dtype=np.float32)
        single = raw_actor.ndim == 1
        actor_rows = raw_actor.reshape(1, -1) if single else raw_actor
        critic_rows = raw_critic.reshape(1, -1) if single else raw_critic
        actor_normalized = self._normalize_actor(actor_rows, update_normalizer)
        critic_normalized = self._normalize_critic(critic_rows, update_normalizer)
        if hidden is None:
            hidden = self.initial_hidden(actor_rows.shape[0])
        hidden_rows = np.asarray(hidden, dtype=np.float32).reshape(
            actor_rows.shape[0], self.config.recurrent_hidden_dim
        )
        starts = np.zeros(actor_rows.shape[0], dtype=np.bool_)
        if episode_start is not None:
            starts = np.broadcast_to(
                np.asarray(episode_start, dtype=np.bool_), starts.shape
            ).copy()
        with torch.no_grad():
            logits, next_hidden, _base = self.network.actor_step(
                torch.as_tensor(actor_normalized, dtype=torch.float32, device=self.device),
                torch.as_tensor(hidden_rows, dtype=torch.float32, device=self.device),
                torch.as_tensor(starts, dtype=torch.bool, device=self.device),
            )
            if action_masks is not None:
                masks = np.asarray(action_masks, dtype=np.bool_)
                masks = masks.reshape(logits.shape)
                mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
                if not bool(torch.all(mask_tensor.any(dim=-1))):
                    raise ValueError("every action mask row must allow at least one action")
                logits = logits.masked_fill(~mask_tensor, torch.finfo(logits.dtype).min)
            distribution = Categorical(logits=logits)
            actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
            log_probs = distribution.log_prob(actions)
            values = self.network.value(
                torch.as_tensor(critic_normalized, dtype=torch.float32, device=self.device)
            )
        outputs = (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
            actor_normalized,
            critic_normalized,
            next_hidden.cpu().numpy(),
        )
        if not single:
            return outputs
        return tuple(value[0] for value in outputs)

    def values(self, critic_state: np.ndarray) -> np.ndarray:
        normalized = self._normalize_critic(critic_state, False)
        with torch.no_grad():
            result = self.network.value(
                torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            )
        return result.cpu().numpy()

    def normalize_rewards(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        if not self.config.normalize_reward:
            return rewards.astype(np.float32)
        if self.discounted_reward.shape != rewards.shape:
            self.discounted_reward = np.zeros_like(rewards, dtype=np.float64)
        self.discounted_reward = (
            self.discounted_reward * self.config.gamma * (1.0 - dones) + rewards
        )
        self.reward_rms.update(self.discounted_reward)
        return np.clip(
            rewards / np.sqrt(self.reward_rms.var + 1e-8), -10.0, 10.0
        ).astype(np.float32)

    def update(self, rollout: dict[str, np.ndarray]) -> dict[str, float]:
        device = self.device
        observations = torch.as_tensor(rollout["observations"], dtype=torch.float32, device=device)
        critic_states = torch.as_tensor(rollout["critic_states"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=device)
        old_log_probs = torch.as_tensor(rollout["log_probs"], dtype=torch.float32, device=device)
        returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)
        advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        action_masks = torch.as_tensor(rollout["action_masks"], dtype=torch.bool, device=device)
        starts = torch.as_tensor(rollout["episode_starts"], dtype=torch.bool, device=device)
        initial_hidden = torch.as_tensor(rollout["initial_hidden"], dtype=torch.float32, device=device)
        expert_actions = torch.as_tensor(rollout["expert_actions"], dtype=torch.long, device=device)
        expert_valid = torch.as_tensor(rollout["expert_valid"], dtype=torch.bool, device=device)
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "expert_loss": 0.0,
            "anchor_kl": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        updates = 0
        env_count = observations.shape[1]
        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(env_count, device=device)
            for start in range(0, env_count, self.config.env_minibatch_size):
                index = permutation[start:start + self.config.env_minibatch_size]
                logits, base_logits = self.network.actor_sequence(
                    observations[:, index], initial_hidden[index], starts[:, index]
                )
                masks = action_masks[:, index]
                floor = torch.finfo(logits.dtype).min
                masked_logits = logits.masked_fill(~masks, floor)
                distribution = Categorical(logits=masked_logits)
                new_log_probs = distribution.log_prob(actions[:, index])
                entropy = distribution.entropy().mean()
                ratio = torch.exp(new_log_probs - old_log_probs[:, index])
                unclipped = ratio * advantages[:, index]
                clipped = torch.clamp(
                    ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range
                ) * advantages[:, index]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                predicted = self.network.value(critic_states[:, index].reshape(-1, critic_states.shape[-1]))
                value_loss = 0.5 * (
                    predicted - returns[:, index].reshape(-1)
                ).pow(2).mean()
                valid = expert_valid[:, index]
                if bool(valid.any()):
                    expert_loss = nn.functional.cross_entropy(
                        masked_logits[valid], expert_actions[:, index][valid]
                    )
                    anchored_base = base_logits.masked_fill(~masks, floor)
                    base_prob = torch.softmax(anchored_base.detach()[valid], dim=-1)
                    anchor_kl = torch.sum(
                        base_prob
                        * (
                            torch.log(base_prob.clamp_min(1e-8))
                            - torch.log_softmax(masked_logits[valid], dim=-1)
                        ),
                        dim=-1,
                    ).mean()
                else:
                    expert_loss = torch.zeros((), device=device)
                    anchor_kl = torch.zeros((), device=device)
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                    + self.config.expert_coef * expert_loss
                    + self.config.anchor_kl_coef * anchor_kl
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    (parameter for parameter in self.network.parameters() if parameter.requires_grad),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                log_ratio = new_log_probs - old_log_probs[:, index]
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                totals["expert_loss"] += float(expert_loss.item())
                totals["anchor_kl"] += float(anchor_kl.item())
                totals["approx_kl"] += float(((ratio - 1.0) - log_ratio).mean().item())
                totals["clip_fraction"] += float(
                    (torch.abs(ratio - 1.0) > self.config.clip_range).float().mean().item()
                )
                updates += 1
        self.update_count += 1
        result = {key: value / max(updates, 1) for key, value in totals.items()}
        result["samples"] = float(observations.shape[0] * observations.shape[1])
        result["expert_samples"] = float(expert_valid.sum().item())
        result["reward_mean"] = float(rollout["raw_rewards"].mean())
        result["explained_variance"] = float(
            1.0
            - np.var(rollout["returns"] - rollout["values"])
            / max(np.var(rollout["returns"]), 1e-8)
        )
        return result

    def save(
        self,
        path: str | Path,
        *,
        environment_config: LocalAvoidConfig,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": self.algorithm_name,
            "ppo_config": asdict(self.config),
            "environment_config": environment_config.to_dict(),
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "actor_rms": self.actor_rms.state_dict(),
            "critic_rms": self.critic_rms.state_dict(),
            "reward_rms": self.reward_rms.state_dict(),
            "update_count": self.update_count,
            "total_steps": self.total_steps,
            "stage": self.stage,
            "metrics": metrics or {},
        }
        temporary = target.with_name(f".{target.name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != cls.algorithm_name:
            raise ValueError("checkpoint is not a recurrent local avoidance PPO model")
        config_data = dict(payload["ppo_config"])
        config_data["device"] = device
        config = RecurrentPPOConfig(**config_data)
        # Recreate the frozen MLP shape used by all current 33-feature checkpoints.
        backbone = nn.Sequential(
            nn.Linear(config.actor_obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        network = RecurrentLocalAvoidActorCritic(
            config,
            actor_backbone=backbone,
            actor_head=nn.Linear(128, config.action_dim),
            critic=nn.Sequential(
                nn.Linear(config.critic_state_dim, 256), nn.Tanh(),
                nn.Linear(256, 256), nn.Tanh(),
                nn.Linear(256, 128), nn.Tanh(),
                nn.Linear(128, 1),
            ),
        )
        policy = cls(config, network)
        policy.network.load_state_dict(payload["network"])
        if "optimizer" in payload:
            policy.optimizer.load_state_dict(payload["optimizer"])
        policy.actor_rms.load_state_dict(payload["actor_rms"])
        policy.critic_rms.load_state_dict(payload["critic_rms"])
        policy.reward_rms.load_state_dict(payload["reward_rms"])
        policy.update_count = int(payload.get("update_count", 0))
        policy.total_steps = int(payload.get("total_steps", 0))
        policy.stage = int(payload.get("stage", 3))
        environment_data = dict(payload["environment_config"])
        environment_data["unit_speeds_mps"] = tuple(environment_data["unit_speeds_mps"])
        if environment_data.get("unit_turn_rates_deg_s") is not None:
            environment_data["unit_turn_rates_deg_s"] = tuple(
                environment_data["unit_turn_rates_deg_s"]
            )
        return policy, LocalAvoidConfig(**environment_data), payload


def collect_recurrent_rollout(
    policy: RecurrentLocalAvoidPPO,
    environment: VectorLocalAvoidEnv,
    actor_observation: np.ndarray,
    critic_state: np.ndarray,
    hidden: np.ndarray,
    episode_starts: np.ndarray,
    *,
    action_mask_fn: ActionMaskFn | None = None,
    expert_action_fn: ExpertActionFn | None = None,
):
    keys = (
        "observations", "critic_states", "actions", "action_masks", "log_probs",
        "values", "rewards", "raw_rewards", "dones", "episode_starts",
        "expert_actions", "expert_valid",
    )
    rows: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    completed: list[dict[str, Any]] = []
    initial_hidden = hidden.copy()
    for _ in range(policy.config.rollout_steps):
        action_masks = (
            np.asarray(action_mask_fn(actor_observation), dtype=np.bool_)
            if action_mask_fn is not None
            else np.ones((actor_observation.shape[0], policy.config.action_dim), dtype=np.bool_)
        )
        if expert_action_fn is None:
            expert_actions = np.full(actor_observation.shape[0], STRAIGHT, dtype=np.int64)
        else:
            expert_actions = np.asarray(
                [
                    expert_action_fn(env, observation)
                    for env, observation in zip(environment.envs, actor_observation)
                ],
                dtype=np.int64,
            )
        invalid = ~action_masks[np.arange(len(expert_actions)), expert_actions]
        expert_actions[invalid] = np.argmax(action_masks[invalid], axis=1)
        expert_valid = action_masks.sum(axis=1) > 1
        actions, log_probs, values, normalized_actor, normalized_critic, next_hidden = policy.act(
            actor_observation,
            critic_state,
            hidden,
            episode_start=episode_starts,
            update_normalizer=not policy.config.freeze_observation_normalizer,
            action_masks=action_masks,
        )
        next_actor, next_critic, raw_rewards, dones, infos = environment.step(actions)
        rewards = policy.normalize_rewards(raw_rewards, dones)
        values_to_store = {
            "observations": normalized_actor,
            "critic_states": normalized_critic,
            "actions": actions,
            "action_masks": action_masks,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "raw_rewards": raw_rewards,
            "dones": dones,
            "episode_starts": episode_starts,
            "expert_actions": expert_actions,
            "expert_valid": expert_valid,
        }
        for key, value in values_to_store.items():
            rows[key].append(np.asarray(value).copy())
        completed.extend(info for info in infos if "terminated" in info or "truncated" in info)
        actor_observation, critic_state = next_actor, next_critic
        hidden = next_hidden
        episode_starts = dones.astype(np.bool_)
    stacked = {key: np.stack(value) for key, value in rows.items()}
    next_values = policy.values(critic_state)
    advantages = np.zeros_like(stacked["rewards"])
    gae = np.zeros(stacked["rewards"].shape[1], dtype=np.float32)
    for step in reversed(range(policy.config.rollout_steps)):
        bootstrap = next_values if step == policy.config.rollout_steps - 1 else stacked["values"][step + 1]
        nonterminal = 1.0 - stacked["dones"][step]
        delta = stacked["rewards"][step] + policy.config.gamma * bootstrap * nonterminal - stacked["values"][step]
        gae = delta + policy.config.gamma * policy.config.gae_lambda * nonterminal * gae
        advantages[step] = gae
    stacked["advantages"] = advantages
    stacked["returns"] = advantages + stacked["values"]
    stacked["initial_hidden"] = initial_hidden
    policy.total_steps += int(stacked["actions"].size)
    return stacked, actor_observation, critic_state, hidden, episode_starts, completed


def evaluate_recurrent_policy(
    policy: RecurrentLocalAvoidPPO,
    environment_config: LocalAvoidConfig,
    *,
    stage: int,
    episodes: int,
    seed: int,
    straight_baseline: bool = False,
    action_mask_fn: ActionMaskFn | None = None,
) -> dict[str, float]:
    num_envs = min(policy.config.num_parallel_envs, max(1, episodes))
    environment = VectorLocalAvoidEnv(num_envs, environment_config, stage=stage, seed=seed)
    actor, critic = environment.reset()
    hidden = policy.initial_hidden(num_envs)
    starts = np.ones(num_envs, dtype=np.bool_)
    completed: list[dict[str, Any]] = []
    selected: list[np.ndarray] = []
    while len(completed) < episodes:
        masks = (
            np.asarray(action_mask_fn(actor), dtype=np.bool_)
            if action_mask_fn is not None
            else np.ones((num_envs, policy.config.action_dim), dtype=np.bool_)
        )
        if straight_baseline:
            actions = np.full(num_envs, STRAIGHT, dtype=np.int64)
            # Keep recurrent state synchronized for a fair stateful evaluation.
            result = policy.act(actor, critic, hidden, episode_start=starts, deterministic=True, action_masks=masks)
            hidden = result[-1]
        else:
            actions, _logp, _value, _a, _c, hidden = policy.act(
                actor, critic, hidden, episode_start=starts, deterministic=True, action_masks=masks
            )
        actor, critic, _reward, dones, infos = environment.step(actions)
        starts = dones.astype(np.bool_)
        selected.append(actions.copy())
        completed.extend(info for info in infos if "terminated" in info or "truncated" in info)
    return summarize_episodes(completed[:episodes], np.concatenate(selected))
