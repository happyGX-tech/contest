"""Tests for the training-side additions: tier hold-out, tricks, eval cells."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policies.red.skills.evasion_ppo.local_ppo import LocalAvoidPPO, LocalPPOConfig  # noqa: E402


def load_trainer():
    path = ROOT / "tools" / "train_local_avoid_full_sim.py"
    spec = importlib.util.spec_from_file_location("full_sim_trainer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAINER = load_trainer()


def make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "tier_holdout": "none",
        "scenarios": "launch20/easy/E01,launch20/medium/M01",
        "heldout_scenarios": "launch20/hard/H07",
        "scenario": "launch20/easy/E01",
        "target_kl": None,
        "clip_range_value": None,
        "dual_clip": None,
        "reward_clip": None,
        "advantage_standardize": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_default_trick_switches_are_off() -> None:
    config = LocalPPOConfig()
    assert config.target_kl == 0.0
    assert config.clip_range_value is None
    assert config.dual_clip is None
    assert config.reward_clip is None
    assert config.advantage_standardize is False


def test_parse_eval_cells_maps_ids_to_scenarios() -> None:
    cells = TRAINER.parse_eval_cells("E01/b0, M03/b3,H07/b9")
    assert [name for name, _, _ in cells] == ["E01_b0", "M03_b3", "H07_b9"]
    assert [scenario for _, scenario, _ in cells] == [
        "launch20/easy/E01",
        "launch20/medium/M03",
        "launch20/hard/H07",
    ]
    assert [policy for _, _, policy in cells] == [
        "b0_fixed_ratio_random",
        "b3_min_cost_assignment",
        "b9_min_cost_planned_multi_wave",
    ]


def test_parse_eval_cells_rejects_malformed() -> None:
    for spec in ("E01", "X01/b0"):
        try:
            TRAINER.parse_eval_cells(spec)
        except ValueError:
            continue
        raise AssertionError(f"{spec!r} should have been rejected")


def test_tier_holdout_pools_are_disjoint_and_complete() -> None:
    scenarios, heldout = TRAINER.resolve_scenario_pools(make_args(tier_holdout="easy"))
    assert heldout, "easy tier should have maps"
    assert all("/easy/" in item for item in heldout)
    assert not any("/easy/" in item for item in scenarios)
    assert not set(scenarios) & set(heldout)
    for tier in ("medium", "hard"):
        discovered = TRAINER.discover_tier_maps(tier)
        assert discovered, f"{tier} tier should have maps"
        assert set(discovered) <= set(scenarios)


def test_explicit_pools_still_work_without_holdout() -> None:
    scenarios, heldout = TRAINER.resolve_scenario_pools(make_args())
    assert scenarios == ["launch20/easy/E01", "launch20/medium/M01"]
    assert heldout == ["launch20/hard/H07"]


def test_apply_ppo_tricks_sets_only_requested_switches() -> None:
    policy = LocalAvoidPPO(LocalPPOConfig())
    args = make_args(
        target_kl=0.02,
        clip_range_value=0.2,
        dual_clip=-0.5,
        reward_clip=10.0,
        advantage_standardize=True,
    )
    applied = TRAINER.apply_ppo_tricks(policy, args)
    assert applied == {
        "target_kl": 0.02,
        "clip_range_value": 0.2,
        "dual_clip": -0.5,
        "reward_clip": 10.0,
        "advantage_standardize": True,
    }
    assert policy.config.target_kl == 0.02
    assert policy.config.dual_clip == -0.5
    assert policy.config.advantage_standardize is True


def test_apply_ppo_tricks_is_a_noop_by_default() -> None:
    policy = LocalAvoidPPO(LocalPPOConfig())
    applied = TRAINER.apply_ppo_tricks(policy, make_args())
    assert applied == {}
    assert policy.config.target_kl == 0.0


def test_aggregate_by_tier_splits_acceptance_counters() -> None:
    workers = [
        {
            "scenario": "launch20/easy/E01",
            "score": {"score": 80.0},
            "outcomes": {"launched": 10, "arrived": 8, "evaded": 4, "engaged_resolved": 5},
        },
        {
            "scenario": "launch20/medium/M01",
            "score": {"score": 60.0},
            "outcomes": {"launched": 10, "arrived": 6, "evaded": 2, "engaged_resolved": 5},
        },
    ]
    by_tier = TRAINER.aggregate_by_tier(workers)
    assert set(by_tier) == {"easy", "medium"}
    assert by_tier["easy"]["score_mean"] == 80.0
    assert by_tier["medium"]["arrival_rate"] == 0.6
