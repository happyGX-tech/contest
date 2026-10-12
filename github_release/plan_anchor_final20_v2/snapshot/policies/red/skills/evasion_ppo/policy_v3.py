"""Threat-gated PPO evasion skill built on the stable V2 interface."""

from __future__ import annotations

from dataclasses import dataclass

from .encoder import EvasionObservationEncoder
from .policy import EvasionPPOConfig, EvasionPPOPolicy


@dataclass
class EvasionPPOV3Config(EvasionPPOConfig):
    learning_rate: float = 1e-4
    entropy_coef: float = 0.005
    threat_survival_reward: float = 0.0
    distance_gain_scale: float = 0.01
    closing_reduction_scale: float = 0.05
    cpa_gain_scale: float = 0.5
    evaded_bonus: float = 5.0
    maneuver_penalty: float = 0.0
    direction_change_penalty: float = 0.02
    maximum_cpa_distance_km: float = 15.0
    maximum_time_to_go_s: float = 120.0
    minimum_closing_speed_mps: float = 50.0


class EvasionPPOV3Policy(EvasionPPOPolicy):
    """Ignore unrelated overflight tracks and optimize real incoming geometry."""

    algorithm_name = "ppo_evasion_skill_v3"

    def __init__(self, config: EvasionPPOV3Config | None = None) -> None:
        super().__init__(config or EvasionPPOV3Config())
        self.config: EvasionPPOV3Config

    def register_agent(self, agent_id: int, entity_id: int, max_steps: int) -> None:
        self._entity_ids[int(agent_id)] = int(entity_id)
        self._encoders[int(agent_id)] = EvasionObservationEncoder(
            entity_id=entity_id,
            max_steps=max_steps,
            max_track_age_steps=self.config.max_track_age_steps,
            allow_fused_tracks=False,
            maximum_cpa_distance_m=self.config.maximum_cpa_distance_km * 1000.0,
            maximum_time_to_go_s=self.config.maximum_time_to_go_s,
            minimum_closing_speed_mps=self.config.minimum_closing_speed_mps,
        )
