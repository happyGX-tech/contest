"""Differential tests against the unmodified frozen B9 policy.

Run with unittest discovery. The reference fixture preserves the pre-optimization
policy; no experiment directories or native simulator are required.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT.parent / "policies" / "blue"
REFERENCE_FILE = ROOT / "fixtures" / "b9_reference_5d473a91.py"
POLICY_FILE = "b9_min_cost_planned_multi_wave.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_policies():
    # A private package avoids importing the simulation engine through policies.
    package = types.ModuleType("_b9_equivalence")
    package.__path__ = [str(CANDIDATE)]
    sys.modules[package.__name__] = package
    base = load_module("_b9_equivalence.base", CANDIDATE / "base.py")
    old = load_module("_b9_equivalence.reference", REFERENCE_FILE)
    new = load_module("_b9_equivalence.candidate", CANDIDATE / POLICY_FILE)
    return base, old.B9MinCostPlannedMultiWavePolicy, new.B9MinCostPlannedMultiWavePolicy


BASE, REFERENCE, OPTIMIZED = load_policies()
V = BASE.Vector3


def target(identifier, pos, velocity=V(300, 0, 0)):
    return BASE.ThreatTrack(identifier, f"target_{identifier}", 1, V(), pos, velocity, 0)


def interceptor(identifier, pos=V(), available=True):
    return BASE.InterceptorState(identifier, V(), pos, V(), available)


def asset(identifier, pos, value=1.0):
    return BASE.DefendedAsset(identifier, f"asset_{identifier}", V(), pos, 24.0, value)


def serialized(items):
    return [asdict(item) for item in items]


class B9EquivalenceTests(unittest.TestCase):
    def compare_step(self, reference, optimized, observation, context=""):
        before = asdict(observation)
        old_observation = copy.deepcopy(observation)
        new_observation = copy.deepcopy(observation)
        old_released = reference.decide(old_observation)
        new_released = optimized.decide(new_observation)
        self.assertEqual(serialized(old_released), serialized(new_released), context)
        self.assertEqual(serialized(reference._pending), serialized(optimized._pending), context)
        self.assertEqual(reference._last_sim_time, optimized._last_sim_time, context)
        self.assertEqual(before, asdict(old_observation), "reference mutated the observation")
        self.assertEqual(before, asdict(new_observation), "optimized mutated the observation")
        return old_released

    def run_steps(self, observations, **kwargs):
        reference, optimized = REFERENCE(**kwargs), OPTIMIZED(**kwargs)
        for index, observation in enumerate(observations):
            self.compare_step(reference, optimized, observation, f"step={index}, params={kwargs}")

    def test_empty_inputs_and_assets(self):
        for targets, interceptors, assets in [
            ([], [], []),
            ([target(1, V(1000, 0, 0))], [], []),
            ([], [interceptor(10)], []),
            ([target(1, V(1000, 0, 0))], [interceptor(10), interceptor(11)], []),
        ]:
            with self.subTest(targets=len(targets), interceptors=len(interceptors)):
                self.run_steps([BASE.BlueObservation(0, targets, interceptors, assets)], planning_delay=0)

    def test_equal_costs_preserve_observation_order(self):
        # Deliberately non-sorted IDs distinguish encounter order from numeric ID order.
        targets = [target(9, V(1000, 0, 0)), target(2, V(1000, 0, 0))]
        interceptors = [interceptor(i) for i in [19, 3, 17, 1, 12, 10, 4, 5, 6, 7]]
        for shots in [1, 3, 5]:
            for wave_size in [1, 2, 3]:
                with self.subTest(shots=shots, wave_size=wave_size):
                    obs = BASE.BlueObservation(0, targets, interceptors, [])
                    self.run_steps([obs, replace(obs, sim_time=5000), replace(obs, sim_time=20000)],
                                   max_shots_per_target=shots, wave_size=wave_size,
                                   planning_delay=5000, wave_gap=5000)

    def test_same_distance_assets_have_different_values(self):
        targets = [target(1, V(), V()), target(2, V(20, 1, 0))]
        interceptors = [interceptor(i, V(i * 120, 0, 0)) for i in range(10, 28)]
        assets = [asset(7, V(-1000, 0, 0), 8), asset(2, V(1000, 0, 0), 1)]
        for asset_order in [assets, assets[::-1]]:
            self.run_steps([BASE.BlueObservation(0, targets, interceptors, asset_order)],
                           planning_delay=0, max_shots_per_target=5)

    def test_duplicate_target_and_interceptor_ids(self):
        targets = [target(3, V(500, 0, 0)), target(3, V(900, 50, 0)), target(1, V(100, 0, 0))]
        interceptors = [interceptor(20, V(-100, 0, 0)), interceptor(2, V()),
                        interceptor(20, V(200, 0, 0)), interceptor(5, V(300, 0, 0)),
                        interceptor(9, V(50, 0, 0)), interceptor(11, V(800, 0, 0))]
        obs = BASE.BlueObservation(0, targets, interceptors, [asset(1, V(1000, 0, 0), 3)])
        self.run_steps([obs, replace(obs, sim_time=6000), replace(obs, sim_time=11000)],
                       planning_delay=5000, wave_gap=5000, max_shots_per_target=3)

    def test_visibility_loss_reappearance_launch_feedback_losses_and_round_reset(self):
        targets = [target(4, V(1000, 500, 0)), target(2, V(900, -100, 0))]
        interceptors = [interceptor(i, V(i * 150, 0, 0)) for i in range(10, 28)]
        assets = [asset(1, V(2000, 0, 0), 5)]
        obs = BASE.BlueObservation(0, targets, interceptors, assets)
        observations = [
            obs,
            replace(obs, sim_time=3000, targets=[]),
            replace(obs, sim_time=5000, targets=[], interceptors=interceptors[1:]),
            replace(obs, sim_time=5500, targets=[replace(targets[0], pos_ecf=V(200, 40, 0))],
                    launched_map={4: [10]}, interceptors=interceptors[2:]),
            replace(obs, sim_time=10000, launched_map={4: [10], 2: [11]}, interceptors=interceptors[2:]),
            replace(obs, sim_time=21000, launched_map={4: [10, 12, 14, 16, 18]}),
            replace(obs, sim_time=0),
            replace(obs, sim_time=20000),
        ]
        self.run_steps(observations, planning_delay=5000, wave_gap=5000, max_shots_per_target=5)

    def test_randomized_multistep_scenes(self):
        for seed in range(160):
            rng = random.Random(seed)
            count = rng.randrange(1, 7)
            targets = [target(i + 1, V(rng.uniform(-80000, 80000), rng.uniform(-80000, 80000),
                                     rng.uniform(-1000, 1000)),
                              V(rng.uniform(-400, 400), rng.uniform(-400, 400), 0))
                       for i in range(count)]
            if seed % 7 == 0:
                targets.append(replace(targets[0], pos_ecf=V(50, 10, 1)))
            interceptors = [interceptor(100 + i, V(rng.uniform(-60000, 60000),
                                                rng.uniform(-60000, 60000), 0))
                            for i in range(rng.randrange(1, 20))]
            assets = [asset(i, V(rng.uniform(-90000, 90000), rng.uniform(-90000, 90000), 0),
                            rng.choice([1, 3, 8, 12])) for i in range(rng.randrange(9))]
            kwargs = dict(max_shots_per_target=rng.choice([1, 2, 5, 7]),
                          planning_delay=rng.choice([0.0, 1000.0, 5000.0]),
                          wave_gap=rng.choice([0.0, 700.0, 5000.0]),
                          wave_size=rng.choice([1, 2, 3]))
            reference, optimized = REFERENCE(**kwargs), OPTIMIZED(**kwargs)
            launched = {}
            with self.subTest(seed=seed, **kwargs):
                for step, now in enumerate([0, 1000, 4999, 5000, 11000, 16000, 0, 16000]):
                    if step == 6:
                        launched = {}
                    visible = [replace(item, pos_ecf=V(item.pos_ecf.x + step * 37,
                                                      item.pos_ecf.y - step * 23, item.pos_ecf.z))
                               for item in targets if rng.random() > 0.22]
                    rng.shuffle(visible)
                    observed_interceptors = [replace(item, available=rng.random() > 0.12)
                                             for item in interceptors]
                    rng.shuffle(observed_interceptors)
                    observation = BASE.BlueObservation(now, visible, observed_interceptors, assets,
                                                       copy.deepcopy(launched))
                    released = self.compare_step(reference, optimized, observation, f"seed={seed}, step={step}")
                    for assignment in released:
                        # Some accepted plans receive delayed launch feedback.
                        if rng.random() > 0.2:
                            launched.setdefault(assignment.target_id, []).append(assignment.interceptor_id)

    def test_distance_work_does_not_repeat_for_each_allocation(self):
        observation = synthetic_observation(target_count=6, interceptor_count=48, asset_count=12)
        original_distance = BASE.distance_m

        def decide_and_count(policy_class, shots):
            calls = 0

            def counted_distance(*args, **kwargs):
                nonlocal calls
                calls += 1
                return original_distance(*args, **kwargs)

            policy = policy_class(max_shots_per_target=shots, planning_delay=5000,
                                  wave_gap=5000, wave_size=1)
            policy_module = sys.modules[policy_class.__module__]
            with patch.object(BASE, "distance_m", counted_distance), \
                    patch.object(policy_module, "distance_m", counted_distance):
                released = policy.decide(copy.deepcopy(observation))
            return calls, (serialized(released), serialized(policy._pending), policy._last_sim_time)

        reference_counts, optimized_counts = [], []
        for shots in [1, 5]:
            with self.subTest(max_shots_per_target=shots):
                reference_count, reference_output = decide_and_count(REFERENCE, shots)
                optimized_count, optimized_output = decide_and_count(OPTIMIZED, shots)
                self.assertEqual(reference_output, optimized_output)
                # Each target performs two asset scans plus one distance to its
                # nearest asset, and one distance for every available interceptor.
                distance_budget = len(observation.targets) * (
                    2 * len(observation.assets) + 1 + len(observation.interceptors)
                )
                self.assertEqual(optimized_count, distance_budget)
                self.assertGreater(reference_count, optimized_count)
                reference_counts.append(reference_count)
                optimized_counts.append(optimized_count)
        # More assigned waves must not multiply the geometric work.
        self.assertEqual(optimized_counts[0], optimized_counts[1])
        self.assertGreater(reference_counts[1], reference_counts[0])

    def test_ecf_scale_and_small_distance_perturbations(self):
        center = 6_371_000.0
        targets = [target(i, V(center + (i % 2) * 1e-8, i * 1e-9, 0), V(300, 0, 0))
                   for i in [8, 2, 9, 1]]
        interceptors = [interceptor(i, V(center + 1000 + (i % 3) * 1e-8, 0, 0))
                        for i in [17, 11, 19, 12, 20, 14, 15, 16]]
        self.run_steps([BASE.BlueObservation(0, targets, interceptors,
                                            [asset(1, V(center + 100, 0, 0), 5)])],
                       planning_delay=0, wave_gap=100, max_shots_per_target=3)


def synthetic_observation(target_count, interceptor_count, asset_count):
    rng = random.Random(42)
    targets = [target(i, V(rng.uniform(-50000, 50000), rng.uniform(-50000, 50000), 100),
                      V(300, -20, 0)) for i in range(target_count)]
    interceptors = [interceptor(1000 + i, V(rng.uniform(-60000, 60000), rng.uniform(-60000, 60000), 0))
                    for i in range(interceptor_count)]
    assets = [asset(2000 + i, V(rng.uniform(-50000, 50000), rng.uniform(-50000, 50000), 0), 3)
              for i in range(asset_count)]
    return BASE.BlueObservation(0, targets, interceptors, assets)



if __name__ == "__main__":
    unittest.main()
