"""Decentralized parameter-sharing PPO for red-side missile motion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .red_policy import ACTION_DIM, PolicyTransition, SharedPolicy
from .reward_shaping import R9RewardShaper


@dataclass
class PPOConfig:
    observation_dim: int = 85
    action_dim: int = ACTION_DIM
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    rollout_size: int = 8192
    min_update_size: int = 512
    objective_damage_scale: float = 1.0
    target_progress_scale: float = 2.0
    intercepted_penalty: float = 0.2
    maneuver_penalty: float = 0.001
    direction_change_penalty: float = 0.002
    reward_clip: float = 25.0
    seed: int = 0
    device: str = "auto"
    max_steps: int = 1000


PPORewardShaper = R9RewardShaper


class PPOActorCritic(nn.Module):
    """Use local observations for both the actor and decentralized critic."""

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(config.observation_dim, config.actor_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.actor_hidden_dim, config.actor_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.actor_hidden_dim, config.action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(config.observation_dim, config.critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.critic_hidden_dim, config.critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.critic_hidden_dim, 1),
        )

    def actor_forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)

    def critic_forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)


class PPOSharedPolicy(SharedPolicy):
    """Share one decentralized PPO actor-critic across all red missiles."""

    def __init__(self, config: PPOConfig | None = None):
        self.config = config or PPOConfig()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        device_name = self.config.device
        if device_name == "auto":
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.network = PPOActorCritic(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config.learning_rate
        )
        self.reward_shaper = PPORewardShaper(self.config)
        self._buffer: list[tuple[PolicyTransition, float, float]] = []
        self._pending: list[tuple[float, float]] = []
        self.update_count = 0
        self.transition_count = 0
        self.last_metrics: dict[str, float] = {}
        self.training = True

    def begin_environment_step(self, full_observation) -> None:
        self.reward_shaper.begin_environment_step(full_observation)

    def end_environment_step(self, full_observation) -> None:
        self.reward_shaper.end_environment_step(full_observation)

    def finish_environment_step(self) -> None:
        if self.training and len(self._buffer) >= self.config.rollout_size:
            self.update()

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            action_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self.network.actor_forward(observation_tensor)
            value = self.network.critic_forward(observation_tensor)
            logits = logits.masked_fill(~mask_tensor, torch.finfo(logits.dtype).min)
            distribution = Categorical(logits=logits)
            action = distribution.sample() if self.training else torch.argmax(logits, dim=-1)
            log_prob = distribution.log_prob(action)
        if self.training:
            self._pending.append((float(log_prob.item()), float(value.item())))
        return int(action.item())

    def observe(self, transition: PolicyTransition) -> None:
        if not self.training:
            return
        if not self._pending:
            raise RuntimeError("PPO收到transition前没有对应动作")
        log_prob, value = self._pending.pop(0)
        shaped_transition = PolicyTransition(
            agent_id=transition.agent_id,
            observation=transition.observation,
            action=transition.action,
            action_mask=transition.action_mask,
            reward=self.reward_shaper.shape(transition),
            next_observation=transition.next_observation,
            done=transition.done,
            global_state=None,
            next_global_state=None,
        )
        self._buffer.append((shaped_transition, log_prob, value))
        self.transition_count += 1

    def reset_episode(self) -> None:
        if not self.training:
            return
        if self._pending:
            raise RuntimeError("回合结束时仍有未匹配的PPO动作")
        if len(self._buffer) >= self.config.min_update_size:
            self.update()
        self.reward_shaper.reset_episode()

    def set_training(self, training: bool) -> None:
        self.training = bool(training)
        self.network.train(self.training)
        if not self.training:
            self._buffer.clear()
            self._pending.clear()

    def _advantages_and_returns(self, rewards, dones, values, next_values, agent_ids):
        advantages = torch.zeros_like(rewards)
        for agent_id in torch.unique(agent_ids).tolist():
            indices = torch.nonzero(agent_ids == agent_id, as_tuple=False).flatten()
            gae = torch.zeros((), dtype=torch.float32, device=self.device)
            for index in reversed(indices.tolist()):
                nonterminal = 1.0 - dones[index]
                delta = (
                    rewards[index]
                    + self.config.gamma * next_values[index] * nonterminal
                    - values[index]
                )
                gae = (
                    delta
                    + self.config.gamma
                    * self.config.gae_lambda
                    * nonterminal
                    * gae
                )
                advantages[index] = gae
        return advantages, advantages + values

    def update(self) -> dict[str, float]:
        if len(self._buffer) < self.config.min_update_size:
            return self.last_metrics

        transitions = [item[0] for item in self._buffer]
        observations = torch.as_tensor(
            np.stack([item.observation for item in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        next_observations = torch.as_tensor(
            np.stack([item.next_observation for item in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in transitions], dtype=torch.long, device=self.device
        )
        action_masks = torch.as_tensor(
            np.stack([item.action_mask for item in transitions]),
            dtype=torch.bool,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [item.reward for item in transitions], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            [item.done for item in transitions], dtype=torch.float32, device=self.device
        )
        agent_ids = torch.as_tensor(
            [item.agent_id for item in transitions], dtype=torch.long, device=self.device
        )
        old_log_probs = torch.as_tensor(
            [item[1] for item in self._buffer], dtype=torch.float32, device=self.device
        )
        old_values = torch.as_tensor(
            [item[2] for item in self._buffer], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            next_values = self.network.critic_forward(next_observations)
            advantages, returns = self._advantages_and_returns(
                rewards, dones, old_values, next_values, agent_ids
            )
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        updates = 0
        sample_count = observations.shape[0]
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.config.minibatch_size):
                indices = permutation[start:start + self.config.minibatch_size]
                logits = self.network.actor_forward(observations[indices])
                logits = logits.masked_fill(
                    ~action_masks[indices], torch.finfo(logits.dtype).min
                )
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(actions[indices])
                entropy = distribution.entropy().mean()
                ratio = torch.exp(new_log_probs - old_log_probs[indices])
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantages[indices]
                policy_loss = -torch.min(unclipped, clipped).mean()

                predicted_values = self.network.critic_forward(observations[indices])
                clipped_values = old_values[indices] + torch.clamp(
                    predicted_values - old_values[indices],
                    -self.config.value_clip_ratio,
                    self.config.value_clip_ratio,
                )
                value_losses = (predicted_values - returns[indices]).pow(2)
                clipped_value_losses = (clipped_values - returns[indices]).pow(2)
                value_loss = 0.5 * torch.maximum(
                    value_losses, clipped_value_losses
                ).mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                log_ratio = new_log_probs - old_log_probs[indices]
                totals["approx_kl"] += float(
                    ((ratio - 1.0) - log_ratio).mean().item()
                )
                totals["clip_fraction"] += float(
                    (torch.abs(ratio - 1.0) > self.config.clip_ratio)
                    .float()
                    .mean()
                    .item()
                )
                updates += 1

        self.update_count += 1
        self.last_metrics = {
            key: value / max(updates, 1) for key, value in totals.items()
        }
        self.last_metrics["samples"] = float(sample_count)
        self.last_metrics["reward_mean"] = float(rewards.mean().item())
        self.last_metrics["reward_std"] = float(rewards.std(unbiased=False).item())
        return_variance = torch.var(returns, unbiased=False)
        if return_variance > 1e-8:
            explained_variance = 1.0 - torch.var(
                returns - old_values, unbiased=False
            ) / return_variance
            self.last_metrics["explained_variance"] = float(
                explained_variance.item()
            )
        else:
            self.last_metrics["explained_variance"] = 0.0
        self._buffer.clear()
        return self.last_metrics

    def save(self, path: str) -> None:
        if self.training and len(self._buffer) >= self.config.min_update_size:
            self.update()
        payload = {
            "algorithm": "ppo",
            "config": asdict(self.config),
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "transition_count": self.transition_count,
            "last_metrics": self.last_metrics,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm", "ppo") != "ppo":
            raise ValueError("该checkpoint不是PPO模型")
        self.network.load_state_dict(checkpoint["network"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_count = int(checkpoint.get("update_count", 0))
        self.transition_count = int(checkpoint.get("transition_count", 0))
        self.last_metrics = dict(checkpoint.get("last_metrics", {}))
