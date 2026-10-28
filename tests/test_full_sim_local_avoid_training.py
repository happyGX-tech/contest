from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from policies.red.learning.red_policy import PolicyTransition
from policies.red.skills.evasion_ppo.full_sim_rollout import (
    annotate_runtime_trace_outcomes,
    build_full_sim_rollout,
    load_runtime_trajectories,
)
from policies.red.skills.evasion_ppo.local_env import LocalAvoidConfig
from policies.red.skills.evasion_ppo.local_runtime import LocalAvoidRuntimePolicy
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


class FullSimulatorRolloutTests(unittest.TestCase):
    def test_failed_trajectory_receives_zero_v10_reward(self) -> None:
        policy = SetAttentionLocalAvoidPPO(
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
        trajectory = {
            "format": "local-avoid-runtime-trajectory-v2",
            "missile_type_slot": 0,
            "terminal": "intercepted",
            "terminal_source": "mission_outcome_tracker",
            "steps": [
                {
                    "observation": observation(0.8).tolist(),
                    "action": 1,
                    "action_mask": [1, 1, 1],
                    "reward": 999.0,
                    "done": False,
                },
                {
                    "observation": observation(0.7).tolist(),
                    "action": 1,
                    "action_mask": [1, 1, 1],
                    "reward": 999.0,
                    "done": True,
                },
            ],
        }

        environment = LocalAvoidConfig(
            progress_scale=0.0,
            progress_fraction_scale=1.0,
            hit_penalty=2.0,
            arrival_bonus=5.0,
            timeout_penalty=1.0,
            action_switch_penalty=0.002,
            turn_reversal_penalty=0.01,
            reward_clip=10.0,
        )
        rollout, stats = build_full_sim_rollout(
            policy,
            [trajectory],
            environment_config=environment,
            behavior_batch_size=1,
        )

        np.testing.assert_allclose(rollout["raw_rewards"], [0.0, 0.0], atol=1e-6)
        self.assertEqual(stats["trajectories"], 1.0)
        self.assertEqual(stats["intercepted"], 1.0)
        self.assertEqual(stats["failed_zero_reward"], 1.0)
        self.assertEqual(rollout["critic_states"].shape, (2, 125))

    def test_normalized_distance_reward_keeps_progress_before_interception(self) -> None:
        policy = SetAttentionLocalAvoidPPO(
            SetAttentionPPOConfig(
                device="cpu",
                token_dim=16,
                attention_heads=2,
                attention_layers=1,
                attention_ff_dim=32,
                hidden_dims=(32, 32, 16),
                ppo_epochs=1,
                minibatch_size=8,
                normalize_reward=False,
            )
        )
        trajectory = {
            "format": "local-avoid-runtime-trajectory-v3",
            "missile_type_slot": 0,
            "terminal": "intercepted",
            "terminal_source": "mission_outcome_tracker",
            # The complete simulator emits an all-zero next observation after
            # an entity disappears.  The runtime trace separately preserves
            # the real terminal goal distance.
            "next_observation": observation(0.0),
            "terminal_goal_distance_m": 60_000.0,
            "steps": [
                {
                    "observation": observation(0.8),
                    "action": 1,
                    "action_mask": [1, 1, 1],
                },
                {
                    "observation": observation(0.7),
                    "action": 1,
                    "action_mask": [1, 1, 1],
                },
            ],
        }

        rollout, stats = build_full_sim_rollout(
            policy,
            [trajectory],
            environment_config=LocalAvoidConfig(),
            reward_mode="normalized_distance",
        )

        np.testing.assert_allclose(
            rollout["raw_rewards"], [0.125, 0.125], atol=1e-6
        )
        self.assertAlmostEqual(stats["normalized_distance_reward"], 0.25)
        self.assertEqual(stats["failed_zero_reward"], 0.0)

    def test_distance_outcome_reward_adds_mutually_exclusive_terminal_ladder(self) -> None:
        policy = SetAttentionLocalAvoidPPO(
            SetAttentionPPOConfig(
                device="cpu",
                token_dim=16,
                attention_heads=2,
                attention_layers=1,
                attention_ff_dim=32,
                hidden_dims=(32, 32, 16),
                normalize_reward=False,
            )
        )
        environment = LocalAvoidConfig()

        def trajectory(terminal: str) -> dict:
            return {
                "format": "local-avoid-runtime-trajectory-v3",
                "terminal": terminal,
                "terminal_source": "mission_outcome_tracker",
                "terminal_goal_distance_m": 60_000.0,
                "missile_type_slot": 0,
                "steps": [
                    {"observation": observation(0.8), "action": 1, "action_mask": [1, 1, 1]},
                    {"observation": observation(0.7), "action": 1, "action_mask": [1, 1, 1]},
                ],
            }

        expected = {
            "effective_hit": (0.60, 0.35),
            "arrived": (0.45, 0.20),
            "intercepted": (0.00, -0.25),
            "lost": (0.25, 0.00),
        }
        for terminal, (expected_return, expected_terminal) in expected.items():
            with self.subTest(terminal=terminal):
                rollout, stats = build_full_sim_rollout(
                    policy,
                    [trajectory(terminal)],
                    environment_config=environment,
                    reward_mode="distance_outcome",
                )
                self.assertAlmostEqual(float(rollout["raw_rewards"].sum()), expected_return)
                self.assertAlmostEqual(stats["terminal_outcome_reward"], expected_terminal)

        _, hit_stats = build_full_sim_rollout(
            policy,
            [trajectory("effective_hit")],
            environment_config=environment,
            reward_mode="distance_outcome",
        )
        self.assertEqual(hit_stats["arrival_bonus"], 0.0)
        self.assertAlmostEqual(hit_stats["effective_hit_bonus"], 0.35)

    def test_timely_arrival_equates_arrival_and_damage_and_penalizes_failure(self) -> None:
        policy = SetAttentionLocalAvoidPPO(
            SetAttentionPPOConfig(
                device="cpu",
                token_dim=16,
                attention_heads=2,
                attention_layers=1,
                attention_ff_dim=32,
                hidden_dims=(32, 32, 16),
                normalize_reward=False,
            )
        )

        def trajectory(terminal: str) -> dict:
            return {
                "format": "local-avoid-runtime-trajectory-v3",
                "terminal": terminal,
                "terminal_source": "mission_outcome_tracker",
                "terminal_goal_distance_m": 60_000.0,
                "missile_type_slot": 0,
                "steps": [
                    {"observation": observation(0.8), "action": 1, "action_mask": [1, 1, 1]},
                    {"observation": observation(0.7), "action": 1, "action_mask": [1, 1, 1]},
                ],
            }

        returns = {}
        for terminal in ("effective_hit", "arrived", "intercepted", "lost", "timeout"):
            rollout, stats = build_full_sim_rollout(
                policy,
                [trajectory(terminal)],
                environment_config=LocalAvoidConfig(),
                reward_mode="timely_arrival",
                reward_horizon_steps=100,
            )
            returns[terminal] = float(rollout["raw_rewards"].sum())
            self.assertEqual(stats["effective_hit_bonus"], 0.0)

        self.assertAlmostEqual(returns["effective_hit"], returns["arrived"])
        self.assertAlmostEqual(returns["arrived"], 0.349, places=6)
        self.assertAlmostEqual(returns["intercepted"], -0.101, places=6)
        self.assertAlmostEqual(returns["lost"], -0.301, places=6)
        self.assertAlmostEqual(returns["timeout"], -0.301, places=6)
        self.assertGreater(returns["arrived"], returns["intercepted"])
        self.assertGreater(returns["intercepted"], returns["timeout"])

    def test_real_arrival_distance_reward_telescopes_to_one(self) -> None:
        policy = SetAttentionLocalAvoidPPO(
            SetAttentionPPOConfig(
                device="cpu",
                token_dim=16,
                attention_heads=2,
                attention_layers=1,
                attention_ff_dim=32,
                hidden_dims=(32, 32, 16),
            )
        )
        trajectory = {
            "format": "local-avoid-runtime-trajectory-v3",
            "terminal": "arrived",
            "terminal_source": "mission_outcome_tracker",
            "missile_type_slot": 0,
            "steps": [
                {"observation": observation(0.8), "action": 1, "action_mask": [1, 1, 1]},
                {"observation": observation(0.7), "action": 1, "action_mask": [1, 1, 1]},
            ],
        }

        rollout, stats = build_full_sim_rollout(
            policy,
            [trajectory],
            environment_config=LocalAvoidConfig(),
        )

        np.testing.assert_allclose(rollout["raw_rewards"], [0.125, 0.875], atol=1e-6)
        self.assertAlmostEqual(float(rollout["raw_rewards"].sum()), 1.0)
        self.assertAlmostEqual(stats["successful_distance_reward"], 1.0)

    def test_time_limit_trajectory_uses_next_observation_bootstrap(self) -> None:
        policy = SetAttentionLocalAvoidPPO(
            SetAttentionPPOConfig(
                device="cpu",
                token_dim=16,
                attention_heads=2,
                attention_layers=1,
                attention_ff_dim=32,
                hidden_dims=(32, 32, 16),
            )
        )
        incomplete = {
            "format": "local-avoid-runtime-trajectory-v3",
            "terminal": "incomplete",
            "terminal_source": "mission_outcome_tracker",
            "missile_type_slot": 0,
            "next_observation": observation(0.6),
            "steps": [
                {"observation": observation(0.8), "action": 1, "action_mask": [1, 1, 1]},
                {"observation": observation(0.7), "action": 1, "action_mask": [1, 1, 1]},
            ],
        }
        rollout, stats = build_full_sim_rollout(
            policy,
            [incomplete],
            environment_config=LocalAvoidConfig(),
        )

        self.assertEqual(len(rollout["actions"]), 2)
        self.assertEqual(stats["excluded_incomplete"], 0.0)
        self.assertEqual(stats["bootstrapped_incomplete"], 1.0)
        self.assertEqual(rollout["dones"][-1], 0.0)
        np.testing.assert_allclose(
            rollout["raw_rewards"], [0.125, 0.125], atol=1e-6
        )

    def test_proxy_terminal_is_rejected_by_formal_rollout(self) -> None:
        policy = SetAttentionLocalAvoidPPO(
            SetAttentionPPOConfig(
                device="cpu",
                token_dim=16,
                attention_heads=2,
                attention_layers=1,
                attention_ff_dim=32,
                hidden_dims=(32, 32, 16),
            )
        )
        trajectory = {
            "format": "local-avoid-runtime-trajectory-v3",
            "terminal": "arrived_proxy",
            "missile_type_slot": 0,
            "steps": [
                {"observation": observation(0.8), "action": 1, "action_mask": [1, 1, 1]},
                {"observation": observation(0.7), "action": 1, "action_mask": [1, 1, 1]},
            ],
        }

        with self.assertRaisesRegex(ValueError, "MissionOutcomeTracker"):
            build_full_sim_rollout(
                policy,
                [trajectory],
                environment_config=LocalAvoidConfig(),
            )

    def test_runtime_rollout_mode_requires_a_trace_and_writes_v3(self) -> None:
        with patch.dict(os.environ, {"LOCAL_AVOID_ROLLOUT_MODE": "1"}, clear=False):
            os.environ.pop("LOCAL_AVOID_TRACE_PATH", None)
            with self.assertRaisesRegex(ValueError, "LOCAL_AVOID_TRACE_PATH"):
                LocalAvoidRuntimePolicy()

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "rollout.jsonl.gz"
            with patch.dict(
                os.environ,
                {
                    "LOCAL_AVOID_ROLLOUT_MODE": "1",
                    "LOCAL_AVOID_TRACE_PATH": str(trace_path),
                    "LOCAL_AVOID_TRACE_OVERWRITE": "1",
                    "LOCAL_AVOID_GATE_MODE": "none",
                },
            ):
                runtime = LocalAvoidRuntimePolicy()
            runtime.environment_config = LocalAvoidConfig()
            runtime._entity_ids[1] = 101
            runtime.observe(
                PolicyTransition(
                    agent_id=1,
                    observation=observation(0.5),
                    action=1,
                    action_mask=np.ones(3, dtype=np.bool_),
                    reward=1.0,
                    next_observation=np.zeros(85, dtype=np.float32),
                    done=True,
                )
            )
            trajectories = load_runtime_trajectories([trace_path])

        self.assertEqual(trajectories[0]["format"], "local-avoid-runtime-trajectory-v3")
        self.assertTrue(trajectories[0]["rollout_mode"])
        self.assertEqual(trajectories[0]["gate_mode"], "none")
        self.assertEqual(len(trajectories[0]["next_observation"]), 85)

    def test_finalize_episode_flushes_truncated_active_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "truncated.jsonl.gz"
            with patch.dict(
                os.environ,
                {
                    "LOCAL_AVOID_ROLLOUT_MODE": "1",
                    "LOCAL_AVOID_TRACE_PATH": str(trace_path),
                    "LOCAL_AVOID_TRACE_OVERWRITE": "1",
                },
            ):
                runtime = LocalAvoidRuntimePolicy()
            runtime.environment_config = LocalAvoidConfig()
            runtime._entity_ids[2] = 202
            runtime.observe(
                PolicyTransition(
                    agent_id=2,
                    observation=observation(0.9),
                    action=1,
                    action_mask=np.ones(3, dtype=np.bool_),
                    reward=1.0,
                    next_observation=np.zeros(85, dtype=np.float32),
                    done=False,
                )
            )
            runtime.finalize_episode(truncated=True)
            trajectories = load_runtime_trajectories([trace_path])

        self.assertEqual(trajectories[0]["terminal"], "incomplete")

    def test_trace_annotation_uses_real_per_platform_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "outcome.jsonl.gz"
            trajectory = {
                "format": "local-avoid-runtime-trajectory-v3",
                "entity_id": 101,
                "terminal": "lost_proxy",
                "steps": [],
            }
            with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(trajectory) + "\n")
            counts = annotate_runtime_trace_outcomes(
                trace_path,
                {
                    "outcomes": {
                        "per_platform": {
                            "101": {
                                "engaged": True,
                                "resolved": True,
                                "arrived": True,
                                "effective_hit": True,
                                "alive_at_end": False,
                            }
                        }
                    }
                },
            )
            annotated = load_runtime_trajectories([trace_path])[0]

        self.assertEqual(counts, {"effective_hit": 1})
        self.assertEqual(annotated["terminal"], "effective_hit")
        self.assertEqual(annotated["terminal_proxy"], "lost_proxy")

    def test_trace_annotation_can_label_completed_alive_episode_as_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "timeout.jsonl.gz"
            trajectory = {
                "format": "local-avoid-runtime-trajectory-v3",
                "entity_id": 101,
                "terminal": "incomplete",
                "steps": [],
            }
            with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(trajectory) + "\n")
            counts = annotate_runtime_trace_outcomes(
                trace_path,
                {
                    "outcomes": {
                        "per_platform": {
                            "101": {
                                "engaged": False,
                                "resolved": False,
                                "arrived": False,
                                "effective_hit": False,
                                "alive_at_end": True,
                            }
                        }
                    }
                },
                alive_at_end_terminal="timeout",
            )
            annotated = load_runtime_trajectories([trace_path])[0]

        self.assertEqual(counts, {"timeout": 1})
        self.assertEqual(annotated["terminal"], "timeout")

    def test_trace_annotation_records_the_official_objective_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "weighted.jsonl.gz"
            trajectory = {
                "format": "local-avoid-runtime-trajectory-v3",
                "entity_id": 101,
                "terminal": "lost_proxy",
                "steps": [],
            }
            with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(trajectory) + "\n")
            annotate_runtime_trace_outcomes(
                trace_path,
                {
                    "outcomes": {
                        "per_platform": {
                            "101": {
                                "target_id": 2749,
                                "engaged": False,
                                "resolved": True,
                                "arrived": True,
                                "effective_hit": False,
                                "alive_at_end": False,
                            }
                        }
                    },
                    "score": {
                        "objective_weights": {"2749": 5.0, "2750": 1.0}
                    },
                },
            )
            annotated = load_runtime_trajectories([trace_path])[0]

        self.assertEqual(annotated["objective_id"], 2749)
        self.assertEqual(annotated["objective_weight"], 5.0)

    def test_score_weighted_reward_multiplies_return_by_objective_weight(self) -> None:
        policy = SetAttentionLocalAvoidPPO(
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
        environment = LocalAvoidConfig(
            progress_scale=0.0,
            progress_fraction_scale=1.0,
            hit_penalty=2.0,
            arrival_bonus=5.0,
            timeout_penalty=1.0,
            action_switch_penalty=0.0,
            turn_reversal_penalty=0.0,
            reward_clip=10.0,
        )

        def arrival(weight: float) -> dict:
            return {
                "format": "local-avoid-runtime-trajectory-v3",
                "missile_type_slot": 0,
                "terminal": "arrived",
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

        plain, _ = build_full_sim_rollout(
            policy,
            [arrival(1.0)],
            environment_config=environment,
            behavior_batch_size=1,
        )
        weighted, stats = build_full_sim_rollout(
            policy,
            [arrival(1.0), arrival(5.0)],
            environment_config=environment,
            behavior_batch_size=2,
            reward_mode="score_weighted",
        )

        np.testing.assert_allclose(plain["raw_rewards"].sum(), 1.0, atol=1e-5)
        np.testing.assert_allclose(
            weighted["raw_rewards"][:2].sum(), 1.0, atol=1e-5
        )
        np.testing.assert_allclose(
            weighted["raw_rewards"][2:].sum(), 5.0, atol=1e-5
        )
        self.assertEqual(stats["weighted_trajectories"], 1.0)
        self.assertEqual(stats["score_weighted"], 1.0)


if __name__ == "__main__":
    unittest.main()
