"""The mission gate must run on the engine's inference action path.

``core/user_agents/red/attack_missile_agent.py`` hands the encoded observation
and the policy mask straight to ``LocalAvoidRuntimePolicy.select_action`` (the
engine never calls ``prepare_action_mask`` itself in the 10-19 release), so the
policy is the only place that can enforce the gate.
"""

import numpy as np

from policies.red.learning.red_policy import HighLevelAction
from policies.red.skills.evasion_ppo.local_runtime import LocalAvoidRuntimePolicy

ONLY_STOP = [False, True, False]


class _RecordingModel:
    config = type("Config", (), {"critic_state_dim": 33})()

    def __init__(self):
        self.masks = []

    def act(self, actor_values, critic_state, *, deterministic, update_normalizer, action_masks):
        self.masks.append(np.array(action_masks, dtype=bool))
        return np.array([0]), None, None, None, None


def _gated_policy(model):
    policy = LocalAvoidRuntimePolicy(max_steps=1000)
    policy.model = model
    policy.prepare_action_mask = lambda observation, mask: np.array(ONLY_STOP)
    return policy


def test_select_action_applies_gate_mask_before_sampling():
    model = _RecordingModel()
    policy = _gated_policy(model)

    policy.select_action(np.zeros(33, dtype=np.float32), np.ones(3, dtype=bool))

    assert model.masks, "model.act was not called"
    assert model.masks[0].tolist() == ONLY_STOP


def test_select_action_falls_back_when_actor_picks_masked_action():
    model = _RecordingModel()  # always returns lateral action 0
    policy = _gated_policy(model)

    action = policy.select_action(np.zeros(33, dtype=np.float32), np.ones(3, dtype=bool))

    assert action == int(HighLevelAction.STOP_MANEUVER)
