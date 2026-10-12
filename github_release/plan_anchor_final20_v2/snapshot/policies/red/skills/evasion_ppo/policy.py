"""PPO implementation packaged as an independent missile evasion skill."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from policies.red.learning.ppo_policy import PPOConfig, PPOSharedPolicy
from policies.red.learning.red_policy import HighLevelAction, PolicyTransition

from .encoder import EvasionObservationEncoder
from .reward import EvasionRewardShaper


@dataclass
class EvasionPPOConfig(PPOConfig):
    observation_dim: int = EvasionObservationEncoder.OBSERVATION_DIM
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    learning_rate: float = 2e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    entropy_coef: float = 0.02
    rollout_size: int = 4096
    min_update_size: int = 256
    minibatch_size: int = 256
    update_epochs: int = 6
    threat_survival_reward: float = 0.01
    distance_gain_scale: float = 0.02
    closing_reduction_scale: float = 0.04
    cpa_gain_scale: float = 0.25
    distance_reward_scale_km: float = 10.0
    cpa_reward_scale_km: float = 5.0
    intercepted_penalty: float = 10.0
    evaded_bonus: float = 3.0
    maneuver_penalty: float = 0.0005
    direction_change_penalty: float = 0.002
    reward_clip: float = 10.0
    clear_grace_steps: int = 3
    intercept_failure_radius_km: float = 2.0
    metric_window_events: int = 2000
    target_evasion_rate: float = 0.99
    target_min_events: int = 1000
    best_min_events: int = 200
    max_track_age_steps: int = 3


class EvasionPPOPolicy(PPOSharedPolicy):
    """Shared weights with strictly decentralized per-missile execution."""

    algorithm_name = "ppo_evasion_skill_v2"
    is_evasion_skill = True
    records_evaluation = True
    requires_environment_hooks = True
    allows_satellite = False

    def __init__(self, config: EvasionPPOConfig | None = None) -> None:
        super().__init__(config or EvasionPPOConfig())
        self.config: EvasionPPOConfig
        self.reward_shaper = EvasionRewardShaper(self.config)
        self._encoders: dict[int, EvasionObservationEncoder] = {}
        self._entity_ids: dict[int, int] = {}
        self._checkpoint_path: Path | None = None
        self._episode_count = 0
        self._best_wilson = 0.0
        self.last_evasion_metrics: dict = {}

    def register_agent(self, agent_id: int, entity_id: int, max_steps: int) -> None:
        self._entity_ids[int(agent_id)] = int(entity_id)
        self._encoders[int(agent_id)] = EvasionObservationEncoder(
            entity_id=entity_id,
            max_steps=max_steps,
            max_track_age_steps=self.config.max_track_age_steps,
            allow_fused_tracks=False,
        )

    def set_checkpoint_path(self, path: str | None) -> None:
        self._checkpoint_path = Path(path) if path else None

    def encode_motion_observation(
        self,
        observation,
        *,
        agent_id: int,
        entity_id: int,
        launch_step: int,
        maneuver_state: int,
    ) -> np.ndarray:
        encoder = self._encoders.get(int(agent_id))
        if encoder is None:
            self.register_agent(agent_id, entity_id, self.config.max_steps)
            encoder = self._encoders[int(agent_id)]
        return encoder.encode(
            observation,
            launch_step=launch_step,
            maneuver_state=maneuver_state,
        )

    def prepare_action_mask(
        self, observation: np.ndarray, action_mask: np.ndarray
    ) -> np.ndarray:
        effective_mask = np.asarray(action_mask, dtype=np.bool_).copy()
        if not EvasionObservationEncoder.has_threat(observation):
            effective_mask[:] = False
            effective_mask[int(HighLevelAction.STOP_MANEUVER)] = True
        return effective_mask

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        return super().select_action(observation, action_mask)

    def observe(self, transition: PolicyTransition) -> None:
        outcome = self.reward_shaper.shape(transition)
        if not self.training:
            return
        if not self._pending:
            raise RuntimeError("Evasion PPO received a transition without an action")
        log_prob, value = self._pending.pop(0)
        if not outcome.trainable:
            return
        shaped = PolicyTransition(
            agent_id=transition.agent_id,
            observation=transition.observation,
            action=transition.action,
            action_mask=transition.action_mask,
            reward=outcome.value,
            next_observation=transition.next_observation,
            done=bool(transition.done or outcome.skill_terminal),
        )
        self._buffer.append((shaped, log_prob, value))
        self.transition_count += 1

    def reset_episode(self) -> None:
        super().reset_episode()
        for encoder in self._encoders.values():
            encoder.reset()

    def set_training(self, training: bool) -> None:
        was_training = getattr(self, "training", True)
        super().set_training(training)
        if not training and was_training:
            self.reward_shaper.reset_statistics()

    def finish_episode(self, _final_observation) -> dict:
        self._episode_count += 1
        metrics = self.reward_shaper.finish_episode()
        rolling = metrics["rolling"]
        target_reached = bool(
            rolling["attempts"] >= self.config.target_min_events
            and rolling["evasion_rate"] >= self.config.target_evasion_rate
            and rolling["wilson_lower_95"] >= self.config.target_evasion_rate
        )
        metrics.update(
            {
                "episode_count": self._episode_count,
                "target_evasion_rate": self.config.target_evasion_rate,
                "target_min_events": self.config.target_min_events,
                "target_reached": target_reached,
                "training": self.training,
            }
        )
        self.last_evasion_metrics = metrics
        if (
            self.training
            and self._checkpoint_path is not None
            and rolling["attempts"] >= self.config.best_min_events
            and rolling["wilson_lower_95"] > self._best_wilson
        ):
            self._best_wilson = float(rolling["wilson_lower_95"])
            self._write_checkpoint(self._best_path())
        print("EVASION_METRICS " + json.dumps(metrics, ensure_ascii=True), flush=True)
        return metrics

    def _best_path(self) -> Path:
        assert self._checkpoint_path is not None
        return self._checkpoint_path.with_name(
            f"{self._checkpoint_path.stem}_best{self._checkpoint_path.suffix}"
        )

    def _payload(self) -> dict:
        return {
            "algorithm": self.algorithm_name,
            "config": asdict(self.config),
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "transition_count": self.transition_count,
            "last_metrics": self.last_metrics,
            "evasion_statistics": self.reward_shaper.statistics_state(),
            "episode_count": self._episode_count,
            "best_wilson": self._best_wilson,
            "last_evasion_metrics": self.last_evasion_metrics,
        }

    def _write_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save(self._payload(), temporary)
        temporary.replace(path)

    def save(self, path: str) -> None:
        if self.training and len(self._buffer) >= self.config.min_update_size:
            self.update()
        self._write_checkpoint(Path(path))

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm") != self.algorithm_name:
            raise ValueError("Checkpoint is not a PPO evasion skill model")
        self.network.load_state_dict(checkpoint["network"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_count = int(checkpoint.get("update_count", 0))
        self.transition_count = int(checkpoint.get("transition_count", 0))
        self.last_metrics = dict(checkpoint.get("last_metrics", {}))
        self.reward_shaper.load_statistics_state(checkpoint.get("evasion_statistics", {}))
        self._episode_count = int(checkpoint.get("episode_count", 0))
        self._best_wilson = float(checkpoint.get("best_wilson", 0.0))
        self.last_evasion_metrics = dict(checkpoint.get("last_evasion_metrics", {}))
