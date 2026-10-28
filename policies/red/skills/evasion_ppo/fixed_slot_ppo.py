"""Fixed-slot MLP PPO for the ordered eight-threat observation ABI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .local_env import LocalAvoidConfig
from .local_ppo import LocalAvoidActorCritic, LocalAvoidPPO, LocalPPOConfig, _mlp
from .set_env import SetLocalAvoidEnv


@dataclass
class FixedSlotPPOConfig(LocalPPOConfig):
    actor_obs_dim: int = SetLocalAvoidEnv.ACTOR_OBS_DIM
    critic_state_dim: int = SetLocalAvoidEnv.CRITIC_STATE_DIM
    # Zero padding is part of the fixed-slot ABI. Per-feature running
    # normalization would turn empty slots into non-zero pseudo threats.
    normalize_observation: bool = False


class FixedSlotActorCritic(nn.Module):
    """Independent actor/critic MLPs over risk-ordered padded slots."""

    def __init__(self, config: FixedSlotPPOConfig) -> None:
        super().__init__()
        hidden = tuple(config.hidden_dims)
        if not hidden:
            raise ValueError("fixed-slot MLP requires at least one hidden layer")
        self.actor_backbone = _mlp(
            config.actor_obs_dim, hidden[:-1], hidden[-1]
        )
        self.actor_head = nn.Linear(hidden[-1], config.action_dim)
        self.critic_backbone = _mlp(
            config.critic_state_dim, hidden[:-1], hidden[-1]
        )
        self.critic_head = nn.Linear(hidden[-1], 1)

    def actor_logits(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor_head(torch.tanh(self.actor_backbone(observation)))

    def value(self, critic_state: torch.Tensor) -> torch.Tensor:
        return self.critic_head(
            torch.tanh(self.critic_backbone(critic_state))
        ).squeeze(-1)


class FixedSlotLocalAvoidPPO(LocalAvoidPPO):
    algorithm_name = "local_avoid_fixed_slot_mlp_ppo_v1"

    def __init__(self, config: FixedSlotPPOConfig | None = None) -> None:
        resolved = config or FixedSlotPPOConfig()
        if resolved.normalize_observation:
            raise ValueError(
                "fixed-slot observations keep padding at zero; normalization must be disabled"
            )
        super().__init__(resolved)
        self.config = resolved
        self.network = FixedSlotActorCritic(resolved).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config.learning_rate
        )

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
            "observation_contract": {
                "max_threats": SetLocalAvoidEnv.MAX_THREATS,
                "actor_dim": SetLocalAvoidEnv.ACTOR_OBS_DIM,
                "critic_dim": SetLocalAvoidEnv.CRITIC_STATE_DIM,
                "padding": "zero with valid bit",
                "slot_order": "runtime risk order",
                "architecture": "fixed_slot_mlp",
            },
        }
        temporary = target.with_name(f".{target.name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != cls.algorithm_name:
            raise ValueError("checkpoint is not a fixed-slot MLP avoidance model")
        config_data = dict(payload["ppo_config"])
        config_data["hidden_dims"] = tuple(config_data["hidden_dims"])
        config_data["device"] = device
        policy = cls(FixedSlotPPOConfig(**config_data))
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
        environment_data["unit_speeds_mps"] = tuple(
            environment_data["unit_speeds_mps"]
        )
        if environment_data.get("unit_turn_rates_deg_s") is not None:
            environment_data["unit_turn_rates_deg_s"] = tuple(
                environment_data["unit_turn_rates_deg_s"]
            )
        return policy, LocalAvoidConfig(**environment_data), payload


@dataclass
class V13ResidualPPOConfig(FixedSlotPPOConfig):
    """Eight-slot residual policy with an embedded frozen V13 actor."""

    teacher_actor_obs_dim: int = 33
    teacher_hidden_dims: tuple[int, int, int] = (256, 256, 128)
    input_threat_distance_scale_m: float = 10_000.0
    input_relative_speed_scale_mps: float = 500.0
    input_t_cpa_scale_s: float = 120.0
    teacher_threat_distance_scale_m: float = 10_000.0
    teacher_relative_speed_scale_mps: float = 500.0
    teacher_t_cpa_scale_s: float = 120.0
    teacher_risk_distance_m: float = 1_000.0
    teacher_risk_t_cpa_s: float = 45.0
    teacher_risk_d_cpa_m: float = 3_000.0
    residual_logit_scale: float = 1.0


class V13ResidualActorCritic(nn.Module):
    """Frozen two-threat V13 logits plus a trainable eight-slot correction."""

    def __init__(self, config: V13ResidualPPOConfig) -> None:
        super().__init__()
        teacher_config = LocalPPOConfig(
            actor_obs_dim=config.teacher_actor_obs_dim,
            hidden_dims=config.teacher_hidden_dims,
            action_dim=config.action_dim,
        )
        self.teacher_actor = LocalAvoidActorCritic(teacher_config).actor
        for parameter in self.teacher_actor.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("teacher_mean", torch.zeros(33))
        self.register_buffer("teacher_var", torch.ones(33))

        hidden = tuple(config.hidden_dims)
        self.residual_backbone = _mlp(
            config.actor_obs_dim, hidden[:-1], hidden[-1]
        )
        self.residual_head = nn.Linear(hidden[-1], config.action_dim)
        # The untrained model must be exactly V13, not V13 plus random noise.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.critic_backbone = _mlp(
            config.critic_state_dim, hidden[:-1], hidden[-1]
        )
        self.critic_head = nn.Linear(hidden[-1], 1)
        self.config = config

    def legacy_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """Project risk-ordered eight slots onto V13's frozen 33-D ABI."""

        blocks = observation[..., 6:78].reshape(*observation.shape[:-1], 8, 9)
        valid = blocks[..., :, 0] > 0.5
        distance = blocks[..., :, 1] * self.config.input_threat_distance_scale_m
        closing = blocks[..., :, 5] * self.config.input_relative_speed_scale_mps
        time_to_cpa = blocks[..., :, 6] * self.config.input_t_cpa_scale_s
        cpa_distance = blocks[..., :, 7] * self.config.input_threat_distance_scale_m
        risky = valid & (
            (distance <= self.config.teacher_risk_distance_m)
            | (
                (closing > 0.0)
                & (time_to_cpa <= self.config.teacher_risk_t_cpa_s)
                & (cpa_distance <= self.config.teacher_risk_d_cpa_m)
            )
        )
        selected = torch.zeros(
            *observation.shape[:-1], 2, 9,
            dtype=observation.dtype,
            device=observation.device,
        )
        count = torch.zeros(
            observation.shape[:-1], dtype=torch.long, device=observation.device
        )
        for slot in range(8):
            token = blocks[..., slot, :]
            is_risky = risky[..., slot]
            for target in range(2):
                take = is_risky & (count == target)
                selected[..., target, :] = torch.where(
                    take[..., None], token, selected[..., target, :]
                )
            count = count + is_risky.to(torch.long)
        selected[..., :, 8] = torch.where(
            selected[..., :, 0] > 0.5,
            torch.full_like(selected[..., :, 8], 1.0 / 3.0),
            torch.zeros_like(selected[..., :, 8]),
        )
        selected[..., :, 1] *= (
            self.config.input_threat_distance_scale_m
            / self.config.teacher_threat_distance_scale_m
        )
        selected[..., :, 4:6] *= (
            self.config.input_relative_speed_scale_mps
            / self.config.teacher_relative_speed_scale_mps
        )
        selected[..., :, 6] *= (
            self.config.input_t_cpa_scale_s / self.config.teacher_t_cpa_scale_s
        )
        selected[..., :, 7] *= (
            self.config.input_threat_distance_scale_m
            / self.config.teacher_threat_distance_scale_m
        )
        sources = selected[..., :, 8]
        zeros = torch.zeros(
            *observation.shape[:-1], 3,
            dtype=observation.dtype,
            device=observation.device,
        )
        return torch.cat(
            (
                observation[..., :6],
                selected.flatten(start_dim=-2),
                zeros,
                sources,
                observation[..., 81:82],
                observation[..., 82:85],
            ),
            dim=-1,
        )

    def teacher_logits(self, observation: torch.Tensor) -> torch.Tensor:
        legacy = self.legacy_observation(observation)
        normalized = torch.clamp(
            (legacy - self.teacher_mean) / torch.sqrt(self.teacher_var + 1e-8),
            -10.0,
            10.0,
        )
        return self.teacher_actor(normalized)

    def actor_logits(self, observation: torch.Tensor) -> torch.Tensor:
        base = self.teacher_logits(observation)
        residual_observation = observation.clone()
        # Slots 1-2 belong exclusively to the frozen V13 path.  The residual
        # may use ego/goal/context plus slots 3-8, but cannot rewrite V13 from
        # its own two threat inputs.
        residual_observation[..., 6:24] = 0.0
        extra_valid = (
            observation[..., 24:78]
            .reshape(*observation.shape[:-1], 6, 9)[..., :, 0]
            .gt(0.5)
            .any(dim=-1)
            .to(observation.dtype)
        )
        residual = self.residual_head(
            torch.tanh(self.residual_backbone(residual_observation))
        )
        return base + (
            self.config.residual_logit_scale * extra_valid[..., None] * residual
        )

    def value(self, critic_state: torch.Tensor) -> torch.Tensor:
        return self.critic_head(
            torch.tanh(self.critic_backbone(critic_state))
        ).squeeze(-1)


class V13ResidualLocalAvoidPPO(FixedSlotLocalAvoidPPO):
    algorithm_name = "local_avoid_fixed_slot_v13_residual_ppo_v1"

    def __init__(
        self,
        config: V13ResidualPPOConfig | None = None,
        *,
        teacher: LocalAvoidPPO | None = None,
    ) -> None:
        resolved = config or V13ResidualPPOConfig()
        LocalAvoidPPO.__init__(self, resolved)
        self.config = resolved
        self.network = V13ResidualActorCritic(resolved).to(self.device)
        if teacher is not None:
            self.initialize_from_teacher(teacher)
        self.optimizer = torch.optim.Adam(
            (parameter for parameter in self.network.parameters() if parameter.requires_grad),
            lr=self.config.learning_rate,
        )

    def initialize_from_teacher(self, teacher: LocalAvoidPPO) -> None:
        if tuple(teacher.config.hidden_dims) != tuple(self.config.teacher_hidden_dims):
            raise ValueError("V13 teacher hidden dimensions do not match residual config")
        self.network.teacher_actor.load_state_dict(teacher.network.actor.state_dict())
        self.network.teacher_mean.copy_(
            torch.as_tensor(teacher.actor_rms.mean, dtype=torch.float32)
        )
        self.network.teacher_var.copy_(
            torch.as_tensor(teacher.actor_rms.var, dtype=torch.float32)
        )
        for parameter in self.network.teacher_actor.parameters():
            parameter.requires_grad_(False)

    def save(
        self,
        path: str | Path,
        *,
        environment_config: LocalAvoidConfig,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        super().save(path, environment_config=environment_config, metrics=metrics)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["algorithm"] = self.algorithm_name
        payload["observation_contract"]["architecture"] = "v13_frozen_residual_mlp8"
        temporary = Path(path).with_name(f".{Path(path).name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != cls.algorithm_name:
            raise ValueError("checkpoint is not a V13 residual avoidance model")
        config_data = dict(payload["ppo_config"])
        config_data["hidden_dims"] = tuple(config_data["hidden_dims"])
        config_data["teacher_hidden_dims"] = tuple(config_data["teacher_hidden_dims"])
        config_data["device"] = device
        policy = cls(V13ResidualPPOConfig(**config_data))
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
