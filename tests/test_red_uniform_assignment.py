"""Uniform one-shot baseline (r10) tests: even split, satellite leader, stickiness."""

from collections import Counter

from policies.red.baselines import (
    RED_POLICY_CHOICES,
    BaselineObservation,
    BaselineRules,
    PlatformState,
    TargetPrior,
    build_red_baseline,
)
from policies.red.contracts import Position

STEP = 120
RULES = BaselineRules()


def _platform(entity_id: int, kind: str, *, alive: bool = True) -> PlatformState:
    return PlatformState(
        entity_id=entity_id,
        kind=kind,
        position=Position(lon=11.5, lat=2.2),
        alive=alive,
    )


def _target(entity_id: int, entity_type: int, *, alive: bool = True) -> TargetPrior:
    return TargetPrior(
        entity_id=entity_id,
        entity_type=entity_type,
        position=Position(lon=12.2, lat=2.3),
        alive=alive,
    )


def _obs(platforms, targets, *, step: int = STEP) -> BaselineObservation:
    return BaselineObservation(step=step, platforms=tuple(platforms), targets=tuple(targets))


def _policy():
    return build_red_baseline("r10_uniform_satellite", RULES, seed=0)


def _platforms_by_kind(platforms, assignments):
    kind_by_id = {item.entity_id: item.kind for item in platforms}
    result: dict[str, list] = {}
    for item in assignments:
        result.setdefault(kind_by_id[item.platform_id], []).append(item)
    return result


def test_r10_is_registered_as_a_red_baseline():
    assert "r10_uniform_satellite" in RED_POLICY_CHOICES
    assert _policy() is not None


def test_high_sorties_split_evenly_over_visible_sites():
    platforms = [_platform(1000 + index, "H") for index in range(7)]
    targets = [_target(5000 + index, 9600) for index in range(3)]
    assignments = _policy().decide(_obs(platforms, targets))

    assert len(assignments) == 7
    assert Counter(item.target_id for item in assignments) == {5000: 3, 5001: 2, 5002: 2}
    assert all(5000 <= item.target_id < 5100 for item in assignments)


def test_medium_sorties_split_evenly_over_visible_nests():
    platforms = [_platform(2000 + index, "M") for index in range(6)]
    targets = [_target(6000 + index, 9400) for index in range(2)]
    assignments = _policy().decide(_obs(platforms, targets))

    assert len(assignments) == 6
    assert Counter(item.target_id for item in assignments) == {6000: 3, 6001: 3}


def test_high_falls_back_to_nests_when_no_sites_are_visible():
    platforms = [_platform(1000, "H")]
    targets = [_target(6000, 9400)]
    assignments = _policy().decide(_obs(platforms, targets))
    assert [item.target_id for item in assignments] == [6000]


def test_medium_falls_back_to_sites_when_no_nests_are_visible():
    platforms = [_platform(2000, "M")]
    targets = [_target(5000, 9600)]
    assignments = _policy().decide(_obs(platforms, targets))
    assert [item.target_id for item in assignments] == [5000]


def test_high_and_medium_share_the_eleven_surface_targets():
    high = [_platform(1000 + index, "H") for index in range(4)]
    medium = [_platform(2000 + index, "M") for index in range(4)]
    nests = [_target(6000 + index, 9400) for index in range(2)]
    sites = [_target(5000 + index, 9600) for index in range(3)]
    assignments = _policy().decide(_obs(high + medium, nests + sites))

    counts = Counter(item.target_id for item in assignments)
    assert set(counts) == {5000, 5001, 5002, 6000, 6001}
    assert sorted(counts.values()) == [1, 1, 2, 2, 2]


def test_low_sorties_rotate_evenly_over_alive_ships():
    platforms = [_platform(3000 + index, "L") for index in range(7)]
    targets = [_target(7000 + index, 9500) for index in range(3)]
    assignments = _policy().decide(_obs(platforms, targets))

    assert len(assignments) == 7
    assert Counter(item.target_id for item in assignments) == {7000: 3, 7001: 2, 7002: 2}


def test_low_sorties_wait_when_no_ship_is_visible():
    platforms = [_platform(3000, "L")]
    targets = [_target(6000, 9400), _target(5000, 9600)]
    assert _policy().decide(_obs(platforms, targets)) == ()


def test_high_medium_never_target_ships():
    platforms = [_platform(1000, "H"), _platform(2000, "M")]
    targets = [
        _target(7000, 9500),
        _target(6000, 9400),
        _target(5000, 9600),
    ]
    assignments = _policy().decide(_obs(platforms, targets))
    assert {item.target_id for item in assignments} <= {5000, 6000}


def test_assignment_is_issued_once_not_every_replan():
    platforms = [_platform(1000 + index, "H") for index in range(2)]
    targets = [_target(5000 + index, 9600) for index in range(2)]
    policy = _policy()
    observation = _obs(platforms, targets)

    first = policy.decide(observation)
    second = policy.decide(observation)

    assert len(first) == 2
    assert second == ()


def test_a_platform_whose_target_died_is_reissued_alone():
    platforms = [_platform(1000 + index, "H") for index in range(3)]
    targets = [_target(5000, 9600), _target(5001, 9600)]
    policy = _policy()
    policy.decide(_obs(platforms, targets))

    survivors = [_target(5000, 9600)]
    reissued = policy.decide(_obs(platforms, survivors))

    assert [item.platform_id for item in reissued] == [1001]
    assert reissued[0].target_id == 5000


def test_new_assignments_carry_the_current_step():
    policy = _policy()
    assignments = policy.decide(
        _obs([_platform(1000, "H")], [_target(5000, 9600)], step=200)
    )
    assert assignments[0].launch_step == 200


def test_exactly_one_high_sortie_is_the_satellite_leader():
    platforms = [_platform(1000 + index, "H") for index in range(3)]
    targets = [_target(5000 + index, 9600) for index in range(2)]
    policy = _policy()

    assignments = policy.decide(_obs(platforms, targets))
    leader = policy.satellite_platform_ids

    assert len(leader) == 1
    assert leader <= {item.platform_id for item in assignments}
    kinds = {item.entity_id: item.kind for item in platforms}
    assert {kinds[entity_id] for entity_id in leader} == {"H"}

    policy.decide(_obs(platforms, targets))
    assert policy.satellite_platform_ids == leader


def test_medium_only_force_registers_no_satellite():
    policy = _policy()
    policy.decide(_obs([_platform(2000, "M")], [_target(6000, 9400)]))
    assert policy.satellite_platform_ids == frozenset()


def test_decide_is_deterministic():
    platforms = [_platform(1000 + index, "L") for index in range(5)]
    targets = [_target(7000 + index, 9500) for index in range(2)]
    observation = _obs(platforms, targets)
    assert _policy().decide(observation) == _policy().decide(observation)


def test_no_live_platform_returns_nothing():
    policy = _policy()
    assert policy.decide(_obs([], [_target(5000, 9600)])) == ()
