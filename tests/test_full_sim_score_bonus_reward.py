from __future__ import annotations

import unittest

import numpy as np

from policies.red.skills.evasion_ppo.full_sim_rollout import build_full_sim_rollout
from policies.red.skills.evasion_ppo.local_env import LocalAvoidConfig
from policies.red.skills.evasion_ppo.set_attention_ppo import (
    SetAttentionLocalAvoidPPO,
    SetAttentionPPOConfig,
)


def observation(distance: float) -> np.ndarray:
    value = np.zeros(85, dtype=np.float32)
    value[3] = distance
    value[5] = 1.0
    value[79] = 1.0
    value[82] = 1.0
    return value


def policy() -> SetAttentionLocalAvoidPPO:
    return SetAttentionLocalAvoidPPO(
        SetAttentionPPOConfig(
            device="cpu",
            token_dim=16,
            attention_heads=2,
            attention_layers=1,
            attention_ff_dim=32,
            hidden_dims=(32, 32, 16),
            ppo_epochs=1,
            minibatch_size=8,
        )
    )


def environment() -> LocalAvoidConfig:
    return LocalAvoidConfig(
        progress_scale=0.0,
        progress_fraction_scale=1.0,
        hit_penalty=2.0,
        arrival_bonus=5.0,
        timeout_penalty=1.0,
        action_switch_penalty=0.0,
        turn_reversal_penalty=0.0,
        reward_clip=10.0,
    )


def trajectory(terminal: str, weight: float) -> dict:
    return {
        "format": "local-avoid-runtime-trajectory-v3",
        "missile_type_slot": 0,
        "terminal": terminal,
        "terminal_source": "mission_outcome_tracker",
        "objective_id": 2749,
        "objective_weight": weight,
        "steps": [
            {
                "observation": observation(0.8).tolist(),
                "action": 1,
                "action_mask": [1, 1, 1],
                "reward": 0.0,
                "done": False,
            },
            {
                "observation": observation(0.3).tolist(),
                "action": 1,
                "action_mask": [1, 1, 1],
                "reward": 0.0,
                "done": True,
            },
        ],
    }


class ScoreBonusRewardTests(unittest.TestCase):
    def test_progress_plus_score_adds_weight_scaled_hit_bonus(self) -> None:
        batch, stats = build_full_sim_rollout(
            policy(),
            [trajectory("effective_hit", 5.0), trajectory("effective_hit", 1.0)],
            environment_config=environment(),
            behavior_batch_size=2,
            reward_mode="progress_plus_score",
            score_bonus_coef=1.0,
        )

        np.testing.assert_allclose(batch["raw_rewards"][:2].sum(), 2.0, atol=1e-5)
        np.testing.assert_allclose(batch["raw_rewards"][2:].sum(), 1.2, atol=1e-5)
        self.assertEqual(stats["score_bonus_mode"], 1.0)
        self.assertEqual(stats["score_bonus_trajectories"], 2.0)
        np.testing.assert_allclose(stats["score_bonus_total"], 1.2, atol=1e-5)

    def test_progress_plus_score_pays_no_bonus_when_intercepted(self) -> None:
        batch, stats = build_full_sim_rollout(
            policy(),
            [trajectory("intercepted", 5.0)],
            environment_config=environment(),
            behavior_batch_size=1,
            reward_mode="progress_plus_score",
            score_bonus_coef=1.0,
        )

        np.testing.assert_allclose(batch["raw_rewards"].sum(), 0.0, atol=1e-5)
        self.assertEqual(stats["score_bonus_trajectories"], 0.0)
        self.assertEqual(stats["score_bonus_total"], 0.0)

    def test_score_bonus_coef_scales_the_extra_term(self) -> None:
        batch, _ = build_full_sim_rollout(
            policy(),
            [trajectory("effective_hit", 5.0)],
            environment_config=environment(),
            behavior_batch_size=1,
            reward_mode="progress_plus_score",
            score_bonus_coef=0.5,
        )

        np.testing.assert_allclose(batch["raw_rewards"].sum(), 1.5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
