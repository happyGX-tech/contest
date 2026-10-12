"""Symmetry-augmented PPO evasion skill."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from policies.red.learning.red_policy import PolicyTransition

from .encoder import EvasionObservationEncoder
from .policy_v3 import EvasionPPOV3Config, EvasionPPOV3Policy


@dataclass
class EvasionPPOV4Config(EvasionPPOV3Config):
    entropy_coef: float = 0.01


class EvasionPPOV4Policy(EvasionPPOV3Policy):
    """Train on mirrored local geometry to prevent one-sided action collapse."""

    algorithm_name = "ppo_evasion_skill_v4"
    MIRROR_AGENT_OFFSET = 1_000_000

    def __init__(self, config: EvasionPPOV4Config | None = None) -> None:
        super().__init__(config or EvasionPPOV4Config())
        self.config: EvasionPPOV4Config

    @staticmethod
    def mirror_observation(observation: np.ndarray) -> np.ndarray:
        mirrored = np.asarray(observation, dtype=np.float32).copy()
        mirrored[8], mirrored[10] = observation[10], observation[8]
        for slot in range(EvasionObservationEncoder.MAX_THREATS):
            offset = (
                EvasionObservationEncoder.THREAT_START
                + slot * EvasionObservationEncoder.THREAT_FEATURES
            )
            mirrored[offset + 2] *= -1.0
            mirrored[offset + 6] *= -1.0
        return mirrored

    @staticmethod
    def mirror_action(action: int) -> int:
        return {0: 2, 1: 1, 2: 0}[int(action)]

    @staticmethod
    def mirror_action_mask(action_mask: np.ndarray) -> np.ndarray:
        mirrored = np.asarray(action_mask, dtype=np.bool_).copy()
        mirrored[0], mirrored[2] = action_mask[2], action_mask[0]
        return mirrored

    def observe(self, transition: PolicyTransition) -> None:
        buffer_size = len(self._buffer)
        super().observe(transition)
        if not self.training or len(self._buffer) == buffer_size:
            return

        shaped = self._buffer[-1][0]
        observation = self.mirror_observation(shaped.observation)
        next_observation = self.mirror_observation(shaped.next_observation)
        action = self.mirror_action(shaped.action)
        action_mask = self.mirror_action_mask(shaped.action_mask)
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            action_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self.network.actor_forward(observation_tensor)
            logits = logits.masked_fill(~mask_tensor, torch.finfo(logits.dtype).min)
            distribution = Categorical(logits=logits)
            action_tensor = torch.as_tensor([action], dtype=torch.long, device=self.device)
            log_prob = float(distribution.log_prob(action_tensor).item())
            value = float(self.network.critic_forward(observation_tensor).item())

        mirrored = PolicyTransition(
            agent_id=shaped.agent_id + self.MIRROR_AGENT_OFFSET,
            observation=observation,
            action=action,
            action_mask=action_mask,
            reward=shaped.reward,
            next_observation=next_observation,
            done=shaped.done,
        )
        self._buffer.append((mirrored, log_prob, value))
        self.transition_count += 1
