"""Permutation-invariant set-attention PPO for variable interceptor counts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .local_env import LocalAvoidConfig
from .local_ppo import LocalAvoidPPO, LocalPPOConfig
from .set_env import SetLocalAvoidEnv


@dataclass
class SetAttentionPPOConfig(LocalPPOConfig):
    actor_obs_dim: int = SetLocalAvoidEnv.ACTOR_OBS_DIM
    critic_state_dim: int = SetLocalAvoidEnv.CRITIC_STATE_DIM
    normalize_observation: bool = False
    symmetry_actor_coef: float = 0.0
    symmetry_critic_coef: float = 0.0
    permutation_actor_coef: float = 0.0
    permutation_critic_coef: float = 0.0
    token_dim: int = 64
    attention_heads: int = 4
    attention_layers: int = 2
    attention_ff_dim: int = 128


class _ThreatSetBackbone(nn.Module):
    def __init__(self, config: SetAttentionPPOConfig, *, privileged: bool) -> None:
        super().__init__()
        self.privileged = bool(privileged)
        self.max_threats = SetLocalAvoidEnv.MAX_THREATS
        self.threat_features = SetLocalAvoidEnv.THREAT_FEATURES
        self.extra_features = (
            SetLocalAvoidEnv.CRITIC_EXTRA_PER_THREAT if privileged else 0
        )
        ego_dim = 6 + SetLocalAvoidEnv.CONTEXT_FEATURES + SetLocalAvoidEnv.TYPE_FEATURES
        token_input = self.threat_features + self.extra_features
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, config.token_dim),
            nn.Tanh(),
            nn.LayerNorm(config.token_dim),
        )
        self.threat_encoder = nn.Sequential(
            nn.Linear(token_input, config.token_dim),
            nn.Tanh(),
            nn.Linear(config.token_dim, config.token_dim),
            nn.LayerNorm(config.token_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.attention_ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(
            layer,
            num_layers=config.attention_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.token_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        single = values.ndim == 1
        rows = values.unsqueeze(0) if single else values
        actor = rows[..., : SetLocalAvoidEnv.ACTOR_OBS_DIM]
        threats = actor[
            ...,
            SetLocalAvoidEnv.THREAT_START:SetLocalAvoidEnv.CONTEXT_START,
        ].reshape(-1, self.max_threats, self.threat_features)
        valid = threats[..., 0] > 0.5
        if self.privileged:
            extras = rows[..., SetLocalAvoidEnv.ACTOR_OBS_DIM:].reshape(
                -1, self.max_threats, self.extra_features
            )
            threat_inputs = torch.cat((threats, extras), dim=-1)
        else:
            threat_inputs = threats
        ego = torch.cat(
            (
                actor[..., :6],
                actor[..., SetLocalAvoidEnv.CONTEXT_START:SetLocalAvoidEnv.TYPE_START + 3],
            ),
            dim=-1,
        ).reshape(-1, 6 + SetLocalAvoidEnv.CONTEXT_FEATURES + SetLocalAvoidEnv.TYPE_FEATURES)
        ego_token = self.ego_encoder(ego).unsqueeze(1)
        threat_tokens = self.threat_encoder(threat_inputs)
        sequence = torch.cat((ego_token, threat_tokens), dim=1)
        padding = torch.cat(
            (
                torch.zeros((valid.shape[0], 1), dtype=torch.bool, device=valid.device),
                ~valid,
            ),
            dim=1,
        )
        encoded = self.attention(sequence, src_key_padding_mask=padding)
        output = self.output_norm(encoded[:, 0])
        return output[0] if single else output


class SetAttentionActorCritic(nn.Module):
    def __init__(self, config: SetAttentionPPOConfig) -> None:
        super().__init__()
        self.actor_backbone = _ThreatSetBackbone(config, privileged=False)
        self.critic_backbone = _ThreatSetBackbone(config, privileged=True)
        self.actor_head = nn.Sequential(
            nn.Linear(config.token_dim, 128),
            nn.Tanh(),
            nn.Linear(128, config.action_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(config.token_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def actor_logits(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self.actor_backbone(observation))

    def value(self, critic_state: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_backbone(critic_state)).squeeze(-1)


class SetAttentionLocalAvoidPPO(LocalAvoidPPO):
    algorithm_name = "local_avoid_set_attention_ppo_v1"

    def __init__(self, config: SetAttentionPPOConfig | None = None) -> None:
        resolved = config or SetAttentionPPOConfig()
        if resolved.normalize_observation:
            raise ValueError(
                "set-attention observations keep padding at zero; use LayerNorm instead"
            )
        super().__init__(resolved)
        self.config = resolved
        self.network = SetAttentionActorCritic(resolved).to(self.device)
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
                "padding": "valid-bit masked",
                "positional_encoding": False,
            },
        }
        temporary = target.with_name(f".{target.name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != cls.algorithm_name:
            raise ValueError("checkpoint is not a set-attention avoidance PPO model")
        config_data = dict(payload["ppo_config"])
        config_data["hidden_dims"] = tuple(config_data["hidden_dims"])
        config_data["device"] = device
        policy = cls(SetAttentionPPOConfig(**config_data))
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
