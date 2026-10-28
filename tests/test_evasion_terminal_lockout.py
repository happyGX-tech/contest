"""Terminal-phase lockout for the local-avoid actor.

The learned actor is rewarded for survival, so it keeps manoeuvring until the
missile self-destructs.  The terminal lockout gate removes lateral control once
the missile is close enough to its goal, unless an interceptor is inside the
emergency radius and still closing.
"""

import numpy as np
import pytest

from policies.red.learning.red_policy import HighLevelAction
from policies.red.skills.evasion_ppo.local_env import LocalAvoidConfig
from policies.red.skills.evasion_ppo.local_runtime import (
    LocalAvoidRuntimePolicy,
    MissionGateConfig,
    terminal_lockout_action_masks,
)
from policies.red.skills.evasion_ppo.set_env import SetLocalAvoidEnv
from policies.red.skills.evasion_ppo.set_runtime import set_terminal_lockout_action_masks

CONFIG = LocalAvoidConfig()
STOP = int(HighLevelAction.STOP_MANEUVER)
ONLY_STOP = [False, True, False]


def _legacy_obs(*, slot: int = 0, goal_m: float = 60_000.0, elapsed: float = 0.1, threats=()):
    obs = np.zeros(33, dtype=np.float32)
    obs[3] = goal_m / CONFIG.goal_distance_scale_m
    obs[29] = elapsed
    obs[30 + slot] = 1.0
    for offset, distance_m, closing_mps in threats:
        obs[offset] = 1.0
        obs[offset + 1] = distance_m / CONFIG.threat_distance_scale_m
        obs[offset + 5] = closing_mps / CONFIG.relative_speed_scale_mps
    return obs


def _set_obs(*, slot: int = 0, goal_m: float = 60_000.0, elapsed: float = 0.1, threats=()):
    obs = np.zeros(SetLocalAvoidEnv.ACTOR_OBS_DIM, dtype=np.float32)
    obs[3] = goal_m / CONFIG.goal_distance_scale_m
    obs[SetLocalAvoidEnv.CONTEXT_START + 3] = elapsed
    obs[SetLocalAvoidEnv.TYPE_START + slot] = 1.0
    for offset, distance_m, closing_mps in threats:
        obs[offset] = 1.0
        obs[offset + 1] = distance_m / CONFIG.threat_distance_scale_m
        obs[offset + 5] = closing_mps / CONFIG.relative_speed_scale_mps
    return obs


def test_stop_action_index_matches_engine_mapping():
    assert STOP == 1


def test_far_from_goal_keeps_full_manoeuvre_space():
    mask = terminal_lockout_action_masks(_legacy_obs(goal_m=60_000.0), CONFIG, MissionGateConfig())
    assert np.asarray(mask).tolist() == [True, True, True]


def test_terminal_without_threat_locks_lateral_control():
    mask = terminal_lockout_action_masks(_legacy_obs(goal_m=5_000.0), CONFIG, MissionGateConfig())
    assert np.asarray(mask).tolist() == ONLY_STOP


def test_elapsed_fraction_alone_never_locks_control():
    # obs[29] is elapsed/horizon_steps with a 60-step training horizon, not the
    # real flight time; a deployed missile flies for minutes, so this feature
    # must never close the gate on its own.
    mask = terminal_lockout_action_masks(
        _legacy_obs(goal_m=80_000.0, elapsed=0.9), CONFIG, MissionGateConfig()
    )
    assert np.asarray(mask).tolist() == [True, True, True]


def test_emergency_threat_inside_radius_keeps_control():
    mask = terminal_lockout_action_masks(
        _legacy_obs(goal_m=5_000.0, threats=((6, 400.0, 200.0),)),
        CONFIG,
        MissionGateConfig(),
    )
    assert np.asarray(mask).tolist() == [True, True, True]


def test_receding_emergency_threat_still_locks():
    mask = terminal_lockout_action_masks(
        _legacy_obs(goal_m=5_000.0, threats=((6, 400.0, -200.0),)),
        CONFIG,
        MissionGateConfig(),
    )
    assert np.asarray(mask).tolist() == ONLY_STOP


def test_thresholds_are_ordered_per_missile_type():
    # 9 km is terminal for H (12 km) but not yet for L (8 km).
    high = terminal_lockout_action_masks(_legacy_obs(slot=0, goal_m=9_000.0), CONFIG, MissionGateConfig())
    low = terminal_lockout_action_masks(_legacy_obs(slot=2, goal_m=9_000.0), CONFIG, MissionGateConfig())
    assert np.asarray(high).tolist() == ONLY_STOP
    assert np.asarray(low).tolist() == [True, True, True]


def test_set_attention_layout_is_supported():
    mask = set_terminal_lockout_action_masks(
        _set_obs(goal_m=5_000.0), CONFIG, MissionGateConfig()
    )
    assert np.asarray(mask).tolist() == ONLY_STOP


def test_gate_mode_terminal_lockout_routes_through_runtime(monkeypatch):
    monkeypatch.setenv("LOCAL_AVOID_GATE_MODE", "terminal_lockout")
    policy = LocalAvoidRuntimePolicy(max_steps=3000)
    policy.environment_config = CONFIG

    mask = policy.prepare_action_mask(_set_obs(goal_m=5_000.0), np.ones(3, dtype=np.bool_))

    assert policy.gate_mode == "terminal_lockout"
    assert np.asarray(mask).tolist() == ONLY_STOP


def test_unknown_gate_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("LOCAL_AVOID_GATE_MODE", "medieval")
    with pytest.raises(ValueError):
        LocalAvoidRuntimePolicy(max_steps=10)
