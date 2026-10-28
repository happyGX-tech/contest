"""基于 PyTorch 的轻量参数共享 PPO 基线。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .red_policy import ACTION_DIM, PolicyTransition, SharedPolicy


@dataclass
class PPOConfig:
    observation_dim: int = 85
    action_dim: int = ACTION_DIM
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    rollout_size: int = 2048
    seed: int = 0
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(config.observation_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(config.hidden_dim, config.action_dim)
        self.critic = nn.Linear(config.hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        return self.actor(features), self.critic(features).squeeze(-1)


class PPOSharedPolicy(SharedPolicy):
    """所有红方导弹共享的一套 PPO Actor-Critic 参数。"""

    def __init__(self, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.device = torch.device(self.config.device)
        self.network = ActorCritic(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self._buffer = []
        self._pending = []
        self.update_count = 0
        self.transition_count = 0
        self.last_metrics = {}
        self.training = True

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.network(observation_tensor)
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
            raise RuntimeError("PPO 收到 transition 前没有对应的动作采样")
        log_prob, value = self._pending.pop(0)
        self._buffer.append((transition, log_prob, value))
        self.transition_count += 1
        if len(self._buffer) >= self.config.rollout_size:
            self.update()

    def reset_episode(self) -> None:
        if not self.training:
            return
        if self._pending:
            raise RuntimeError("回合结束时仍有未匹配的 PPO 动作")
        if self._buffer:
            self.update()

    def set_training(self, training: bool) -> None:
        """切换训练/评估模式；评估时使用 argmax 且不收集经验。"""
        self.training = bool(training)
        self.network.train(self.training)

    def update(self) -> dict[str, float]:
        if not self._buffer:
            return self.last_metrics

        observations = torch.as_tensor(
            np.stack([item[0].observation for item in self._buffer]),
            dtype=torch.float32,
            device=self.device,
        )
        next_observations = torch.as_tensor(
            np.stack([item[0].next_observation for item in self._buffer]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item[0].action for item in self._buffer], dtype=torch.long, device=self.device
        )
        action_masks = torch.as_tensor(
            np.stack([item[0].action_mask for item in self._buffer]),
            dtype=torch.bool,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [item[0].reward for item in self._buffer], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            [item[0].done for item in self._buffer], dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.as_tensor(
            [item[1] for item in self._buffer], dtype=torch.float32, device=self.device
        )
        old_values = torch.as_tensor(
            [item[2] for item in self._buffer], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            _, next_values = self.network(next_observations)
            returns = rewards + self.config.gamma * next_values * (1.0 - dones)
            advantages = returns - old_values
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        sample_count = observations.shape[0]
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.config.minibatch_size):
                indices = permutation[start:start + self.config.minibatch_size]
                logits, values = self.network(observations[indices])
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
                value_loss = torch.nn.functional.mse_loss(values, returns[indices])
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                updates += 1

        self._buffer.clear()
        self.update_count += 1
        self.last_metrics = {key: value / max(1, updates) for key, value in metrics.items()}
        self.last_metrics["samples"] = float(sample_count)
        return self.last_metrics

    def save(self, path: str) -> None:
        torch.save({
            "config": self.config.__dict__,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "transition_count": self.transition_count,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_count = int(checkpoint.get("update_count", 0))
        self.transition_count = int(checkpoint.get("transition_count", 0))
