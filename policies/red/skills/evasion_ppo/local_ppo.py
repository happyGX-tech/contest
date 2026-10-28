"""Asymmetric actor-critic PPO for the standalone avoidance curriculum."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .local_env import ACTION_COUNT, LocalAvoidConfig, STRAIGHT, VectorLocalAvoidEnv


ActionMaskFn = Callable[[np.ndarray], np.ndarray]
ExpertPriorFn = Callable[
    [np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
]


@dataclass
class LocalPPOConfig:
    actor_obs_dim: int = 33
    critic_state_dim: int = 43
    action_dim: int = ACTION_COUNT
    hidden_dims: tuple[int, int, int] = (256, 256, 128)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_steps: int = 256
    minibatch_size: int = 1024
    ppo_epochs: int = 5
    num_parallel_envs: int = 128
    normalize_observation: bool = True
    normalize_reward: bool = True
    freeze_observation_normalizer: bool = False
    symmetry_actor_coef: float = 0.0
    symmetry_critic_coef: float = 0.0
    permutation_actor_coef: float = 0.0
    permutation_critic_coef: float = 0.0
    # Update-side switches, all off by default. They only change how the
    # collected rollout is optimised, never what the actor observes.
    target_kl: float = 0.0
    clip_range_value: float | None = None
    dual_clip: float | None = None
    reward_clip: float | None = None
    advantage_standardize: bool = False
    seed: int = 20260924
    device: str = "auto"


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)
        batch_count = values.shape[0] if values.ndim else 1
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (
            m_a + m_b + np.square(delta) * self.count * batch_count / total
        ) / total
        self.mean, self.var, self.count = new_mean, new_var, total

    def normalize(self, values: np.ndarray, clip: float = 10.0) -> np.ndarray:
        return np.clip(
            (values - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip
        ).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


def _mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden:
        layers.extend((nn.Linear(previous, width), nn.Tanh()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


def mirror_actor_observations(values: np.ndarray) -> np.ndarray:
    """Reflect local observations across the missile's longitudinal axis."""
    result = np.asarray(values, dtype=np.float32).copy()
    if result.shape[-1] == 85:
        # Set-attention ABI: six ego/goal features, eight 9-float threat
        # tokens, action-history/elapsed context, and three type bits.
        result[..., 1] *= -1.0
        result[..., 4] *= -1.0
        threat_blocks = result[..., 6:78].reshape(result.shape[:-1] + (8, 9))
        threat_blocks[..., :, 2] *= -1.0
        action_history = result[..., 78:81].copy()
        result[..., 78] = action_history[..., 2]
        result[..., 80] = action_history[..., 0]
        return result
    if result.shape[-1] != 33:
        raise ValueError("actor observation must have 33 or 85 features")
    result[..., 1] *= -1.0
    result[..., 4] *= -1.0

    threat_blocks = result[..., 6:24].reshape(result.shape[:-1] + (2, 9))
    original_blocks = threat_blocks.copy()
    both_present = (
        (original_blocks[..., 0, 0] > 0.5)
        & (original_blocks[..., 1, 0] > 0.5)
    )
    threat_blocks[...] = np.where(
        both_present[..., None, None],
        original_blocks[..., ::-1, :],
        original_blocks,
    )
    threat_blocks[..., :, 2] *= -1.0

    original_sources = result[..., 27:29].copy()
    result[..., 27:29] = np.where(
        both_present[..., None], original_sources[..., ::-1], original_sources
    )
    return result


def mirror_critic_states(values: np.ndarray) -> np.ndarray:
    """Reflect privileged geometry while preserving each threat's identity slot."""
    result = np.asarray(values, dtype=np.float32).copy()
    if result.shape[-1] == 125:
        result[..., :85] = mirror_actor_observations(result[..., :85])
        threat_extras = result[..., 85:125].reshape(
            result.shape[:-1] + (8, 5)
        )
        threat_extras[..., :, 2] *= -1.0
        threat_extras[..., :, 4] *= -1.0
        return result
    if result.shape[-1] != 43:
        raise ValueError("critic state must have 43 or 125 features")
    result[..., :33] = mirror_actor_observations(result[..., :33])
    threat_extras = result[..., 33:43].reshape(result.shape[:-1] + (2, 5))
    threat_extras[..., :, 2] *= -1.0
    threat_extras[..., :, 4] *= -1.0
    return result


def permute_actor_threats(values: np.ndarray) -> np.ndarray:
    """Swap two present threat slots without changing physical semantics."""
    result = np.asarray(values, dtype=np.float32).copy()
    if result.shape[-1] != 33:
        raise ValueError("actor observation must have 33 features")
    blocks = result[..., 6:24].reshape(result.shape[:-1] + (2, 9))
    original_blocks = blocks.copy()
    both_present = (
        (original_blocks[..., 0, 0] > 0.5)
        & (original_blocks[..., 1, 0] > 0.5)
    )
    blocks[...] = np.where(
        both_present[..., None, None],
        original_blocks[..., ::-1, :],
        original_blocks,
    )
    original_sources = result[..., 27:29].copy()
    result[..., 27:29] = np.where(
        both_present[..., None], original_sources[..., ::-1], original_sources
    )
    return result


def permute_critic_threats(values: np.ndarray) -> np.ndarray:
    """Swap both actor slots and privileged threat-identity slots."""
    result = np.asarray(values, dtype=np.float32).copy()
    if result.shape[-1] != 43:
        raise ValueError("critic state must have 43 features")
    result[..., :33] = permute_actor_threats(result[..., :33])
    extras = result[..., 33:43].reshape(result.shape[:-1] + (2, 5))
    original_extras = extras.copy()
    both_present = (
        (original_extras[..., 0, 0] > 0.5)
        & (original_extras[..., 1, 0] > 0.5)
    )
    extras[...] = np.where(
        both_present[..., None, None],
        original_extras[..., ::-1, :],
        original_extras,
    )
    return result


class LocalAvoidActorCritic(nn.Module):
    """Actor consumes legal local features; critic consumes privileged state."""

    def __init__(self, config: LocalPPOConfig) -> None:
        super().__init__()
        self.actor = _mlp(config.actor_obs_dim, config.hidden_dims, config.action_dim)
        self.critic = _mlp(config.critic_state_dim, config.hidden_dims, 1)

    def actor_logits(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)

    def value(self, critic_state: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_state).squeeze(-1)


class LocalAvoidPPO:
    def __init__(self, config: LocalPPOConfig | None = None) -> None:
        self.config = config or LocalPPOConfig()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        device_name = self.config.device
        if device_name == "auto":
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.network = LocalAvoidActorCritic(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config.learning_rate
        )
        self.actor_rms = RunningMeanStd((self.config.actor_obs_dim,))
        self.critic_rms = RunningMeanStd((self.config.critic_state_dim,))
        self.reward_rms = RunningMeanStd(())
        self.discounted_reward = np.zeros(
            self.config.num_parallel_envs, dtype=np.float64
        )
        self.update_count = 0
        self.total_steps = 0
        self.stage = 0

    def _normalize_actor(self, values: np.ndarray, update: bool) -> np.ndarray:
        if not self.config.normalize_observation:
            return np.asarray(values, dtype=np.float32)
        if update:
            self.actor_rms.update(values)
        return self.actor_rms.normalize(values)

    def _normalize_critic(self, values: np.ndarray, update: bool) -> np.ndarray:
        if not self.config.normalize_observation:
            return np.asarray(values, dtype=np.float32)
        if update:
            self.critic_rms.update(values)
        return self.critic_rms.normalize(values)

    def act(
        self,
        actor_observation: np.ndarray,
        critic_state: np.ndarray,
        *,
        deterministic: bool = False,
        update_normalizer: bool = False,
        action_masks: np.ndarray | None = None,
        logit_biases: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        actor_normalized = self._normalize_actor(
            actor_observation, update_normalizer
        )
        critic_normalized = self._normalize_critic(critic_state, update_normalizer)
        actor_tensor = torch.as_tensor(
            actor_normalized, dtype=torch.float32, device=self.device
        )
        critic_tensor = torch.as_tensor(
            critic_normalized, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            logits = self.network.actor_logits(actor_tensor)
            if logit_biases is not None:
                biases = torch.as_tensor(
                    logit_biases, dtype=logits.dtype, device=self.device
                )
                if biases.shape != logits.shape:
                    raise ValueError(
                        f"logit bias shape {tuple(biases.shape)} does not match "
                        f"logits {tuple(logits.shape)}"
                    )
                logits = logits + biases
            if action_masks is not None:
                masks = torch.as_tensor(
                    action_masks, dtype=torch.bool, device=self.device
                )
                if masks.shape != logits.shape:
                    raise ValueError(
                        f"action mask shape {tuple(masks.shape)} does not match "
                        f"logits {tuple(logits.shape)}"
                    )
                if not bool(torch.all(masks.any(dim=-1))):
                    raise ValueError("every action mask row must allow at least one action")
                logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
            distribution = Categorical(logits=logits)
            actions = (
                torch.argmax(logits, dim=-1)
                if deterministic
                else distribution.sample()
            )
            log_probs = distribution.log_prob(actions)
            values = self.network.value(critic_tensor)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
            actor_normalized,
            critic_normalized,
        )

    def values(self, critic_state: np.ndarray) -> np.ndarray:
        normalized = self._normalize_critic(critic_state, False)
        with torch.no_grad():
            result = self.network.value(
                torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            )
        return result.cpu().numpy()

    def normalize_rewards(
        self, rewards: np.ndarray, dones: np.ndarray
    ) -> np.ndarray:
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

    def update(
        self,
        rollout: dict[str, np.ndarray],
        *,
        expert_bc_coef: float = 0.0,
    ) -> dict[str, float]:
        device = self.device
        observations = torch.as_tensor(
            rollout["observations"], dtype=torch.float32, device=device
        )
        critic_states = torch.as_tensor(
            rollout["critic_states"], dtype=torch.float32, device=device
        )
        actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=device)
        old_log_probs = torch.as_tensor(
            rollout["log_probs"], dtype=torch.float32, device=device
        )
        old_values = torch.as_tensor(
            rollout["values"], dtype=torch.float32, device=device
        )
        returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)
        if self.config.reward_clip is not None:
            returns = torch.clamp(
                returns, -self.config.reward_clip, self.config.reward_clip
            )
        advantages = torch.as_tensor(
            rollout["advantages"], dtype=torch.float32, device=device
        )
        if self.config.advantage_standardize:
            advantages = advantages / (advantages.std(unbiased=False) + 1e-8)
        else:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        action_masks = torch.as_tensor(
            rollout.get(
                "action_masks",
                np.ones((len(rollout["actions"]), self.config.action_dim), dtype=np.bool_),
            ),
            dtype=torch.bool,
            device=device,
        )
        logit_biases = torch.as_tensor(
            rollout.get(
                "logit_biases",
                np.zeros(
                    (len(rollout["actions"]), self.config.action_dim),
                    dtype=np.float32,
                ),
            ),
            dtype=torch.float32,
            device=device,
        )
        expert_actions = torch.as_tensor(
            rollout.get(
                "expert_actions",
                np.zeros(len(rollout["actions"]), dtype=np.int64),
            ),
            dtype=torch.long,
            device=device,
        )
        expert_valid = torch.as_tensor(
            rollout.get(
                "expert_valid",
                np.zeros(len(rollout["actions"]), dtype=np.bool_),
            ),
            dtype=torch.bool,
            device=device,
        )
        mirrored_observations = None
        mirrored_critic_states = None
        permuted_observations = None
        permuted_critic_states = None
        if self.config.symmetry_actor_coef > 0.0:
            mirrored_observations = torch.as_tensor(
                self._normalize_actor(
                    mirror_actor_observations(rollout["raw_observations"]), False
                ),
                dtype=torch.float32,
                device=device,
            )
        if self.config.symmetry_critic_coef > 0.0:
            mirrored_critic_states = torch.as_tensor(
                self._normalize_critic(
                    mirror_critic_states(rollout["raw_critic_states"]), False
                ),
                dtype=torch.float32,
                device=device,
            )
        if self.config.permutation_actor_coef > 0.0:
            permuted_observations = torch.as_tensor(
                self._normalize_actor(
                    permute_actor_threats(rollout["raw_observations"]), False
                ),
                dtype=torch.float32,
                device=device,
            )
        if self.config.permutation_critic_coef > 0.0:
            permuted_critic_states = torch.as_tensor(
                self._normalize_critic(
                    permute_critic_threats(rollout["raw_critic_states"]), False
                ),
                dtype=torch.float32,
                device=device,
            )
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "symmetry_actor_loss": 0.0,
            "symmetry_critic_loss": 0.0,
            "permutation_actor_loss": 0.0,
            "permutation_critic_loss": 0.0,
            "expert_bc_loss": 0.0,
        }
        updates = 0
        stopped_early = False
        sample_count = observations.shape[0]
        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(sample_count, device=device)
            for start in range(0, sample_count, self.config.minibatch_size):
                index = permutation[start:start + self.config.minibatch_size]
                bare_logits = self.network.actor_logits(observations[index])
                bare_masked_logits = bare_logits.masked_fill(
                    ~action_masks[index], torch.finfo(bare_logits.dtype).min
                )
                guided_logits = (bare_logits + logit_biases[index]).masked_fill(
                    ~action_masks[index], torch.finfo(bare_logits.dtype).min
                )
                distribution = Categorical(logits=guided_logits)
                new_log_probs = distribution.log_prob(actions[index])
                entropy = distribution.entropy().mean()
                ratio = torch.exp(new_log_probs - old_log_probs[index])
                unclipped = ratio * advantages[index]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                ) * advantages[index]
                policy_loss = -torch.minimum(unclipped, clipped)
                if self.config.dual_clip is not None:
                    policy_loss = -torch.maximum(
                        -policy_loss,
                        self.config.dual_clip * advantages[index].abs(),
                    )
                policy_loss = policy_loss.mean()
                predicted = self.network.value(critic_states[index])
                if self.config.clip_range_value is not None:
                    value_clip = self.config.clip_range_value
                    clipped_values = old_values[index] + torch.clamp(
                        predicted - old_values[index], -value_clip, value_clip
                    )
                    value_loss = 0.5 * torch.maximum(
                        (predicted - returns[index]).pow(2),
                        (clipped_values - returns[index]).pow(2),
                    ).mean()
                else:
                    value_loss = 0.5 * (predicted - returns[index]).pow(2).mean()
                symmetry_actor_loss = torch.zeros((), device=device)
                if mirrored_observations is not None:
                    mirrored_logits = self.network.actor_logits(
                        mirrored_observations[index]
                    )
                    mirrored_masks = action_masks[index][:, [2, 1, 0]]
                    mirrored_logits = mirrored_logits.masked_fill(
                        ~mirrored_masks, torch.finfo(mirrored_logits.dtype).min
                    )
                    target_probabilities = torch.softmax(
                        bare_masked_logits.detach(), dim=-1
                    )[:, [2, 1, 0]]
                    symmetry_actor_loss = torch.sum(
                        target_probabilities
                        * (
                            torch.log(target_probabilities.clamp_min(1e-8))
                            - torch.log_softmax(mirrored_logits, dim=-1)
                        ),
                        dim=-1,
                    ).mean()
                symmetry_critic_loss = torch.zeros((), device=device)
                if mirrored_critic_states is not None:
                    mirrored_values = self.network.value(
                        mirrored_critic_states[index]
                    )
                    symmetry_critic_loss = 0.5 * (
                        mirrored_values - predicted.detach()
                    ).pow(2).mean()
                permutation_actor_loss = torch.zeros((), device=device)
                if permuted_observations is not None:
                    permuted_logits = self.network.actor_logits(
                        permuted_observations[index]
                    )
                    permuted_logits = permuted_logits.masked_fill(
                        ~action_masks[index], torch.finfo(permuted_logits.dtype).min
                    )
                    target_probabilities = torch.softmax(
                        bare_masked_logits.detach(), dim=-1
                    )
                    permutation_actor_loss = torch.sum(
                        target_probabilities
                        * (
                            torch.log(target_probabilities.clamp_min(1e-8))
                            - torch.log_softmax(permuted_logits, dim=-1)
                        ),
                        dim=-1,
                    ).mean()
                permutation_critic_loss = torch.zeros((), device=device)
                if permuted_critic_states is not None:
                    permuted_values = self.network.value(permuted_critic_states[index])
                    permutation_critic_loss = 0.5 * (
                        permuted_values - predicted.detach()
                    ).pow(2).mean()
                expert_bc_loss = torch.zeros((), device=device)
                valid = expert_valid[index]
                if expert_bc_coef > 0.0 and bool(valid.any()):
                    expert_bc_loss = nn.functional.cross_entropy(
                        bare_masked_logits[valid], expert_actions[index][valid]
                    )
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                    + self.config.symmetry_actor_coef * symmetry_actor_loss
                    + self.config.symmetry_critic_coef * symmetry_critic_loss
                    + self.config.permutation_actor_coef * permutation_actor_loss
                    + self.config.permutation_critic_coef * permutation_critic_loss
                    + expert_bc_coef * expert_bc_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                log_ratio = new_log_probs - old_log_probs[index]
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                totals["approx_kl"] += float(
                    ((ratio - 1.0) - log_ratio).mean().item()
                )
                totals["clip_fraction"] += float(
                    (torch.abs(ratio - 1.0) > self.config.clip_range)
                    .float()
                    .mean()
                    .item()
                )
                totals["symmetry_actor_loss"] += float(symmetry_actor_loss.item())
                totals["symmetry_critic_loss"] += float(symmetry_critic_loss.item())
                totals["permutation_actor_loss"] += float(
                    permutation_actor_loss.item()
                )
                totals["permutation_critic_loss"] += float(
                    permutation_critic_loss.item()
                )
                totals["expert_bc_loss"] += float(expert_bc_loss.item())
                updates += 1
                if (
                    self.config.target_kl is not None
                    and self.config.target_kl > 0.0
                    and updates > 0
                ):
                    approximate_kl = totals["approx_kl"] / updates
                    if approximate_kl > 1.5 * self.config.target_kl:
                        stopped_early = True
                        break
            if stopped_early:
                break
        self.update_count += 1
        result = {key: value / max(updates, 1) for key, value in totals.items()}
        result["samples"] = float(sample_count)
        result["expert_samples"] = float(expert_valid.sum().item())
        result["expert_bc_coef"] = float(expert_bc_coef)
        result["kl_stopped_early"] = 1.0 if stopped_early else 0.0
        result["reward_mean"] = float(rollout["raw_rewards"].mean())
        result["reward_std"] = float(rollout["raw_rewards"].std())
        result["normalized_reward_mean"] = float(rollout["rewards"].mean())
        result["normalized_reward_std"] = float(rollout["rewards"].std())
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
            "algorithm": "local_avoid_asymmetric_ppo_v1",
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
        if payload.get("algorithm") != "local_avoid_asymmetric_ppo_v1":
            raise ValueError("checkpoint is not a local avoidance PPO model")
        config_data = dict(payload["ppo_config"])
        config_data["hidden_dims"] = tuple(config_data["hidden_dims"])
        config_data["device"] = device
        policy = cls(LocalPPOConfig(**config_data))
        policy.network.load_state_dict(payload["network"])
        if "optimizer" in payload:
            policy.optimizer.load_state_dict(payload["optimizer"])
        policy.actor_rms.load_state_dict(payload["actor_rms"])
        policy.critic_rms.load_state_dict(payload["critic_rms"])
        policy.reward_rms.load_state_dict(payload["reward_rms"])
        policy.update_count = int(payload.get("update_count", 0))
        policy.total_steps = int(payload.get("total_steps", 0))
        policy.stage = int(payload.get("stage", 0))
        environment_data = dict(payload["environment_config"])
        environment_data["unit_speeds_mps"] = tuple(environment_data["unit_speeds_mps"])
        if environment_data.get("unit_turn_rates_deg_s") is not None:
            environment_data["unit_turn_rates_deg_s"] = tuple(
                environment_data["unit_turn_rates_deg_s"]
            )
        return policy, LocalAvoidConfig(**environment_data), payload


def collect_rollout(
    policy: LocalAvoidPPO,
    environment: VectorLocalAvoidEnv,
    actor_observation: np.ndarray,
    critic_state: np.ndarray,
    *,
    action_mask_fn: ActionMaskFn | None = None,
    expert_prior_fn: ExpertPriorFn | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "observations",
            "critic_states",
            "raw_observations",
            "raw_critic_states",
            "actions",
            "action_masks",
            "log_probs",
            "values",
            "rewards",
            "raw_rewards",
            "dones",
            "logit_biases",
            "expert_actions",
            "expert_valid",
        )
    }
    completed: list[dict[str, Any]] = []
    for _ in range(policy.config.rollout_steps):
        action_masks = (
            np.asarray(action_mask_fn(actor_observation), dtype=np.bool_)
            if action_mask_fn is not None
            else np.ones(
                (actor_observation.shape[0], policy.config.action_dim),
                dtype=np.bool_,
            )
        )
        if expert_prior_fn is None:
            logit_biases = np.zeros(
                (actor_observation.shape[0], policy.config.action_dim),
                dtype=np.float32,
            )
            expert_actions = np.zeros(actor_observation.shape[0], dtype=np.int64)
            expert_valid = np.zeros(actor_observation.shape[0], dtype=np.bool_)
        else:
            logit_biases, expert_actions, expert_valid = expert_prior_fn(
                actor_observation
            )
            logit_biases = np.asarray(logit_biases, dtype=np.float32)
            expert_actions = np.asarray(expert_actions, dtype=np.int64)
            expert_valid = np.asarray(expert_valid, dtype=np.bool_)
        actions, log_probs, values, normalized_actor, normalized_critic = policy.act(
            actor_observation,
            critic_state,
            update_normalizer=not policy.config.freeze_observation_normalizer,
            action_masks=action_masks,
            logit_biases=logit_biases,
        )
        next_actor, next_critic, raw_rewards, dones, infos = environment.step(actions)
        rewards = policy.normalize_rewards(raw_rewards, dones)
        rows["observations"].append(normalized_actor)
        rows["critic_states"].append(normalized_critic)
        rows["raw_observations"].append(actor_observation.copy())
        rows["raw_critic_states"].append(critic_state.copy())
        rows["actions"].append(actions)
        rows["action_masks"].append(action_masks)
        rows["log_probs"].append(log_probs)
        rows["values"].append(values)
        rows["rewards"].append(rewards)
        rows["raw_rewards"].append(raw_rewards)
        rows["dones"].append(dones)
        rows["logit_biases"].append(logit_biases)
        rows["expert_actions"].append(expert_actions)
        rows["expert_valid"].append(expert_valid)
        completed.extend(
            info for info in infos if "terminated" in info or "truncated" in info
        )
        actor_observation, critic_state = next_actor, next_critic

    stacked = {key: np.stack(value) for key, value in rows.items()}
    next_values = policy.values(critic_state)
    advantages = np.zeros_like(stacked["rewards"])
    gae = np.zeros(stacked["rewards"].shape[1], dtype=np.float32)
    for step in reversed(range(policy.config.rollout_steps)):
        bootstrap = next_values if step == policy.config.rollout_steps - 1 else stacked["values"][step + 1]
        nonterminal = 1.0 - stacked["dones"][step]
        delta = (
            stacked["rewards"][step]
            + policy.config.gamma * bootstrap * nonterminal
            - stacked["values"][step]
        )
        gae = (
            delta
            + policy.config.gamma
            * policy.config.gae_lambda
            * nonterminal
            * gae
        )
        advantages[step] = gae
    returns = advantages + stacked["values"]
    rollout = {
        key: value.reshape((-1,) + value.shape[2:])
        for key, value in stacked.items()
    }
    rollout["advantages"] = advantages.reshape(-1)
    rollout["returns"] = returns.reshape(-1)
    policy.total_steps += int(rollout["actions"].shape[0])
    return rollout, actor_observation, critic_state, completed


def summarize_episodes(
    episodes: list[dict[str, Any]], actions: np.ndarray | None = None
) -> dict[str, float]:
    count = len(episodes)
    hits = sum(bool(item.get("hit")) for item in episodes)
    evades = sum(bool(item.get("evaded")) for item in episodes)
    arrivals = sum(bool(item.get("arrived")) for item in episodes)
    recoveries = sum(bool(item.get("recovered")) for item in episodes)
    timeouts = sum(bool(item.get("truncated")) for item in episodes)
    result = {
        "episodes": float(count),
        "evade_success_rate": evades / max(evades + hits, 1),
        "hit_rate": hits / max(count, 1),
        "arrival_rate": arrivals / max(count, 1),
        "recovery_rate": recoveries / max(count, 1),
        "timeout_rate": timeouts / max(count, 1),
        "mean_episode_return": float(
            np.mean([item.get("episode_return", 0.0) for item in episodes])
            if episodes else 0.0
        ),
        "mean_episode_length": float(
            np.mean([item.get("episode_steps", 0.0) for item in episodes])
            if episodes else 0.0
        ),
        "mean_extra_path_length_m": float(
            np.mean([item["extra_path_m"] for item in episodes]) if episodes else 0.0
        ),
        "mean_extra_time_s": float(
            np.mean([item.get("extra_time_s", 0.0) for item in episodes])
            if episodes else 0.0
        ),
        "mean_recovery_time_s": float(
            np.mean([item.get("recovery_time_s", 0.0) for item in episodes])
            if episodes else 0.0
        ),
        "mean_heading_deviation_deg": float(
            np.mean([item.get("mean_heading_deviation_deg", 0.0) for item in episodes])
            if episodes else 0.0
        ),
        "oscillation_rate": float(
            np.mean([item.get("oscillation_rate", 0.0) for item in episodes])
            if episodes else 0.0
        ),
    }
    if actions is not None and actions.size:
        for index, name in enumerate(("left", "straight", "right")):
            result[f"{name}_action_ratio"] = float(np.mean(actions == index))
    return result


def evaluate_policy(
    policy: LocalAvoidPPO,
    environment_config: LocalAvoidConfig,
    *,
    stage: int,
    episodes: int,
    seed: int,
    straight_baseline: bool = False,
    action_mask_fn: ActionMaskFn | None = None,
) -> dict[str, float]:
    num_envs = min(policy.config.num_parallel_envs, max(1, episodes))
    environment = VectorLocalAvoidEnv(
        num_envs, environment_config, stage=stage, seed=seed
    )
    actor_observation, critic_state = environment.reset()
    completed: list[dict[str, Any]] = []
    selected: list[np.ndarray] = []
    while len(completed) < episodes:
        if straight_baseline:
            actions = np.full(num_envs, STRAIGHT, dtype=np.int64)
        else:
            action_masks = (
                np.asarray(action_mask_fn(actor_observation), dtype=np.bool_)
                if action_mask_fn is not None
                else None
            )
            actions = policy.act(
                actor_observation,
                critic_state,
                deterministic=True,
                update_normalizer=False,
                action_masks=action_masks,
            )[0]
        actor_observation, critic_state, _reward, _done, infos = environment.step(actions)
        selected.append(actions.copy())
        completed.extend(
            info for info in infos if "terminated" in info or "truncated" in info
        )
    return summarize_episodes(completed[:episodes], np.concatenate(selected))
