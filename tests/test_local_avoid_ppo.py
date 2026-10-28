"""Contract tests for Stage 0-3 local avoidance pretraining."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from policies.red.skills.evasion_ppo.local_env import (
    LEFT,
    RIGHT,
    STRAIGHT,
    LocalAvoidConfig,
    LocalAvoidEnv,
    VectorLocalAvoidEnv,
    closest_segment_distance,
)
from policies.red.skills.evasion_ppo.local_ppo import (
    LocalAvoidPPO,
    LocalPPOConfig,
    collect_rollout,
    mirror_actor_observations,
    mirror_critic_states,
    permute_actor_threats,
    permute_critic_threats,
)
from tools.diagnose_stage3_avoidance import _distance_band, _reactive_actions


class LocalAvoidEnvironmentTests(unittest.TestCase):
    def test_observation_and_privileged_state_shapes(self) -> None:
        environment = LocalAvoidEnv(stage=2, seed=4)
        observation, info = environment.reset()
        self.assertEqual(observation.shape, (33,))
        self.assertEqual(info["critic_state"].shape, (43,))
        self.assertTrue(np.isfinite(observation).all())
        self.assertTrue(np.isfinite(info["critic_state"]).all())

    def test_stage_zero_straight_flight_has_no_extra_path(self) -> None:
        config = LocalAvoidConfig(
            goal_distance_min_m=4000.0,
            goal_distance_max_m=4000.0,
            unit_speeds_mps=(1000.0, 1000.0, 1000.0),
            arrival_radius_m=100.0,
        )
        environment = LocalAvoidEnv(config, stage=0, seed=1)
        environment.reset({"goal_heading": 0.0, "goal_distance_m": 4000.0})
        done = False
        info = {}
        while not done:
            _obs, _reward, terminated, truncated, info = environment.step(STRAIGHT)
            done = terminated or truncated
        self.assertTrue(info["arrived"])
        self.assertAlmostEqual(info["extra_path_m"], 0.0, places=5)

    def test_residual_actions_are_symmetric(self) -> None:
        config = LocalAvoidConfig(unit_speeds_mps=(1000.0, 1000.0, 1000.0))
        left = LocalAvoidEnv(config, stage=0, seed=1)
        right = LocalAvoidEnv(config, stage=0, seed=1)
        scenario = {"goal_heading": 0.0, "goal_distance_m": 40_000.0}
        left.reset(scenario)
        right.reset(scenario)
        left.step(LEFT)
        right.step(RIGHT)
        self.assertAlmostEqual(left.own_position[0], right.own_position[0])
        self.assertAlmostEqual(left.own_position[1], -right.own_position[1])

    def test_continuous_hit_check_cannot_tunnel(self) -> None:
        self.assertEqual(
            closest_segment_distance(
                np.asarray([-1000.0, 0.0]), np.asarray([1000.0, 0.0])
            ),
            0.0,
        )

    def test_frontal_threat_exposes_positive_closing_and_finite_cpa(self) -> None:
        environment = LocalAvoidEnv(stage=1, seed=2)
        observation, _ = environment.reset(
            {
                "goal_heading": 0.0,
                "goal_distance_m": 50_000.0,
                "bug_num": 1,
                "bug_spawn_positions": [[20_000.0, 0.0]],
                "bug_velocities": [[-2500.0, 0.0]],
            }
        )
        threat_start = 6
        self.assertEqual(observation[threat_start], 1.0)
        self.assertGreater(observation[threat_start + 5], 0.0)
        self.assertTrue(math.isfinite(float(observation[threat_start + 6])))
        self.assertLess(float(observation[threat_start + 7]), 1e-5)

    def test_integrated_turn_accumulates_and_straight_recovers(self) -> None:
        config = LocalAvoidConfig(
            integrated_turn_actions=True,
            avoid_angle_deg=15.0,
            straight_recovery_deg=15.0,
        )
        environment = LocalAvoidEnv(config, stage=0, seed=8)
        environment.reset({"goal_heading": 0.0, "goal_distance_m": 100_000.0})
        environment.step(LEFT)
        environment.step(LEFT)
        self.assertAlmostEqual(math.degrees(environment.own_heading), -30.0)
        environment.step(STRAIGHT)
        self.assertAlmostEqual(math.degrees(environment.own_heading), -15.0)

    def test_dangerous_stage_two_spawns_intercepting_tracks(self) -> None:
        config = LocalAvoidConfig(
            dangerous_spawn_only=True,
            stage2_bearing_limit_deg=180.0,
        )
        environment = LocalAvoidEnv(config, stage=2, seed=9)
        for seed in range(64):
            environment.reset(seed=seed)
            values = environment._threat_values(environment.bugs[0])
            self.assertGreater(values["closing"], 0.0)
            self.assertLessEqual(values["d_cpa"], config.dangerous_max_cpa_m)

    def test_stage_two_bearing_band_excludes_front_and_rear(self) -> None:
        config = LocalAvoidConfig(
            stage2_bearing_min_deg=45.0,
            stage2_bearing_limit_deg=100.0,
        )
        environment = LocalAvoidEnv(config, stage=2, seed=9)
        for seed in range(64):
            environment.reset(seed=seed)
            bearing = abs(
                math.degrees(environment._threat_values(environment.bugs[0])["bearing"])
            )
            self.assertGreaterEqual(bearing, 45.0 - 1e-6)
            self.assertLessEqual(bearing, 100.0 + 1e-6)

    def test_stage_three_pincer_spawns_two_opposing_active_threats(self) -> None:
        config = LocalAvoidConfig(
            stage3_simultaneous_probability=0.0,
            stage3_pincer_probability=1.0,
        )
        environment = LocalAvoidEnv(config, stage=3, seed=11)
        observation, _info = environment.reset({"goal_heading": 0.0})
        bearings = [environment._threat_values(bug)["bearing"] for bug in environment.bugs]
        self.assertEqual(environment.encounter_pattern, "pincer")
        self.assertEqual(len(environment.bugs), 2)
        self.assertTrue(all(bug["active"] for bug in environment.bugs))
        self.assertLess(bearings[0], 0.0)
        self.assertGreater(bearings[1], 0.0)
        self.assertEqual(observation[6], 1.0)
        self.assertEqual(observation[15], 1.0)

    def test_stage_three_delayed_threat_prevents_early_success(self) -> None:
        config = LocalAvoidConfig(
            stage3_simultaneous_probability=0.0,
            stage3_pincer_probability=0.0,
            stage3_delay_min_steps=5,
            stage3_delay_max_steps=5,
            bug_spawn_min_m=45_000.0,
            bug_spawn_max_m=45_000.0,
            evade_clear_steps=1,
            evade_clear_distance_m=0.0,
        )
        environment = LocalAvoidEnv(config, stage=3, seed=12)
        observation, _info = environment.reset({"goal_heading": 0.0})
        self.assertEqual(environment.encounter_pattern, "delayed_trail")
        self.assertEqual(sum(bool(bug["active"]) for bug in environment.bugs), 1)
        self.assertEqual(observation[15], 0.0)
        for _step in range(5):
            observation, _reward, terminated, truncated, _info = environment.step(STRAIGHT)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        observation, _reward, _terminated, _truncated, info = environment.step(STRAIGHT)
        self.assertEqual(info["released_threats"], 2)
        self.assertEqual(observation[15], 1.0)

    def test_stage_three_same_side_sampling(self) -> None:
        config = LocalAvoidConfig(
            stage2_bearing_min_deg=45.0,
            stage2_bearing_limit_deg=100.0,
            stage3_simultaneous_probability=1.0,
            stage3_pincer_probability=0.0,
            stage3_same_side_probability=1.0,
        )
        environment = LocalAvoidEnv(config, stage=3, seed=21)
        for seed in range(32):
            environment.reset(seed=seed)
            bearings = [
                environment._threat_values(bug)["bearing"]
                for bug in environment.bugs
            ]
            self.assertGreater(bearings[0] * bearings[1], 0.0)

    def test_vector_stage_distribution_is_applied_after_reset(self) -> None:
        config = LocalAvoidConfig(horizon_steps=1)
        environment = VectorLocalAvoidEnv(
            4,
            config,
            stage=2,
            seed=10,
            stage_distribution=(0.0, 1.0, 0.0),
        )
        actor, _critic = environment.reset()
        _actor, _critic, _reward, dones, infos = environment.step(
            np.full(4, STRAIGHT, dtype=np.int64)
        )
        self.assertTrue(dones.all())
        self.assertEqual({1}, {int(info["stage"]) for info in infos})
        self.assertEqual(actor.shape, (4, 33))

    def test_vector_environment_accepts_per_slot_configs(self) -> None:
        normal = LocalAvoidConfig(stage3_same_side_probability=0.0)
        hard = LocalAvoidConfig(stage3_same_side_probability=1.0)
        environment = VectorLocalAvoidEnv(
            2,
            [normal, hard],
            stage=3,
            seed=23,
        )
        environment.reset()
        self.assertIs(environment.envs[0].config, normal)
        self.assertIs(environment.envs[1].config, hard)
        with self.assertRaises(ValueError):
            VectorLocalAvoidEnv(2, [normal], stage=3, seed=23)


class LocalAvoidPPOTests(unittest.TestCase):
    def test_stage_three_diagnostic_helpers(self) -> None:
        self.assertEqual(_distance_band(15_000.0), "[0km,25km)")
        self.assertEqual(_distance_band(40_000.0), "[35km,inf)")
        observations = np.zeros((2, 33), dtype=np.float32)
        observations[:, 6] = 1.0
        observations[0, 8] = 0.5
        observations[1, 8] = -0.5
        self.assertTrue(
            np.array_equal(_reactive_actions(observations), np.asarray([LEFT, RIGHT]))
        )

    def test_mirror_transform_is_an_involution_and_swaps_threat_order(self) -> None:
        actor = np.zeros(33, dtype=np.float32)
        actor[1] = 0.25
        actor[4] = -0.5
        actor[6:15] = np.arange(1, 10, dtype=np.float32)
        actor[15:24] = np.arange(11, 20, dtype=np.float32)
        actor[6] = actor[15] = 1.0
        actor[8] = -0.2
        actor[17] = 0.6
        actor[27:29] = (0.1, 0.2)

        mirrored = mirror_actor_observations(actor)
        self.assertAlmostEqual(mirrored[1], -actor[1])
        self.assertAlmostEqual(mirrored[4], -actor[4])
        self.assertAlmostEqual(mirrored[8], -actor[17])
        self.assertAlmostEqual(mirrored[17], -actor[8])
        self.assertTrue(np.allclose(mirrored[27:29], actor[27:29][::-1]))
        self.assertTrue(np.allclose(mirror_actor_observations(mirrored), actor))

        critic = np.concatenate(
            (actor, np.asarray([1.0, 2.0, 3.0, 4.0, 5.0] * 2, dtype=np.float32))
        )
        mirrored_critic = mirror_critic_states(critic)
        self.assertAlmostEqual(mirrored_critic[35], -critic[35])
        self.assertAlmostEqual(mirrored_critic[37], -critic[37])
        self.assertTrue(np.allclose(mirror_critic_states(mirrored_critic), critic))

    def test_threat_permutation_is_an_involution(self) -> None:
        actor = np.zeros(33, dtype=np.float32)
        actor[6:15] = np.arange(1, 10, dtype=np.float32)
        actor[15:24] = np.arange(11, 20, dtype=np.float32)
        actor[6] = actor[15] = 1.0
        actor[27:29] = (0.1, 0.2)
        permuted = permute_actor_threats(actor)
        self.assertTrue(np.allclose(permuted[6:15], actor[15:24]))
        self.assertTrue(np.allclose(permute_actor_threats(permuted), actor))

        critic = np.concatenate(
            (actor, np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 7.0, 8.0, 9.0, 10.0]))
        ).astype(np.float32)
        permuted_critic = permute_critic_threats(critic)
        self.assertTrue(np.allclose(permuted_critic[33:38], critic[38:43]))
        self.assertTrue(
            np.allclose(permute_critic_threats(permuted_critic), critic)
        )

    def test_smoke_rollout_update_and_checkpoint(self) -> None:
        env_config = LocalAvoidConfig(horizon_steps=8)
        config = LocalPPOConfig(
            num_parallel_envs=2,
            rollout_steps=4,
            minibatch_size=4,
            ppo_epochs=1,
            hidden_dims=(32, 32, 16),
            symmetry_actor_coef=0.05,
            symmetry_critic_coef=0.01,
            permutation_actor_coef=0.05,
            permutation_critic_coef=0.01,
            device="cpu",
        )
        policy = LocalAvoidPPO(config)
        environment = VectorLocalAvoidEnv(2, env_config, stage=1, seed=3)
        actor, critic = environment.reset()
        rollout, _actor, _critic, _episodes = collect_rollout(
            policy, environment, actor, critic
        )
        metrics = policy.update(rollout)
        self.assertEqual(metrics["samples"], 8.0)
        self.assertGreaterEqual(metrics["symmetry_actor_loss"], 0.0)
        self.assertGreaterEqual(metrics["symmetry_critic_loss"], 0.0)
        self.assertGreaterEqual(metrics["permutation_actor_loss"], 0.0)
        self.assertGreaterEqual(metrics["permutation_critic_loss"], 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.pt"
            policy.save(path, environment_config=env_config, metrics=metrics)
            restored, restored_env, _payload = LocalAvoidPPO.load(path, device="cpu")
        self.assertEqual(restored.update_count, 1)
        self.assertEqual(restored_env.horizon_steps, 8)

    def test_rollout_can_freeze_observation_normalizers(self) -> None:
        env_config = LocalAvoidConfig(horizon_steps=4)
        config = LocalPPOConfig(
            num_parallel_envs=2,
            rollout_steps=2,
            minibatch_size=2,
            ppo_epochs=1,
            hidden_dims=(16, 16, 8),
            freeze_observation_normalizer=True,
            device="cpu",
        )
        policy = LocalAvoidPPO(config)
        environment = VectorLocalAvoidEnv(2, env_config, stage=1, seed=31)
        actor, critic = environment.reset()
        actor_count = policy.actor_rms.count
        critic_count = policy.critic_rms.count
        collect_rollout(policy, environment, actor, critic)
        self.assertEqual(policy.actor_rms.count, actor_count)
        self.assertEqual(policy.critic_rms.count, critic_count)


if __name__ == "__main__":
    unittest.main()
