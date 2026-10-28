"""Lethal-package baseline (r11) tests: package sizing, fixed ids, even L split."""

from collections import Counter

import pytest

from policies.red.baselines import (
    RED_POLICY_CHOICES,
    R11LethalPackagePolicy,
    BaselineObservation,
    BaselineRules,
    PlatformState,
    TargetPrior,
    build_red_baseline,
)
from policies.red.commander import RedBaselineCommander
from policies.red.contracts import Position

STEP = 120
RULES = BaselineRules()


@pytest.fixture(autouse=True)
def _fresh_shared_state():
    """The assignment table is class state: shared across commander rebuilds."""

    R11LethalPackagePolicy.reset_shared_state()
    yield
    R11LethalPackagePolicy.reset_shared_state()

NEST_IDS = (5000, 5001, 5002, 5003)
SITE_IDS = (6000, 6001, 6002, 6003, 6004, 6005, 6006)
SHIP_IDS = tuple(7000 + index for index in range(10))


def _platform(entity_id: int, kind: str, *, alive: bool = True) -> PlatformState:
    return PlatformState(
        entity_id=entity_id,
        kind=kind,
        position=Position(lon=11.5, lat=2.2),
        alive=alive,
    )


def _target(
    entity_id: int,
    entity_type: int,
    *,
    lon: float = 12.2,
    lat: float = 2.3,
    alive: bool = True,
) -> TargetPrior:
    return TargetPrior(
        entity_id=entity_id,
        entity_type=entity_type,
        position=Position(lon=lon, lat=lat),
        alive=alive,
    )


def _obs(platforms, targets, *, step: int = STEP) -> BaselineObservation:
    return BaselineObservation(step=step, platforms=tuple(platforms), targets=tuple(targets))


def _fleet() -> list[PlatformState]:
    platforms = [_platform(1000 + index, "H") for index in range(17)]
    platforms += [_platform(2000 + index, "M") for index in range(24)]
    platforms += [_platform(3000 + index, "L") for index in range(85)]
    return platforms


def _surfaces() -> list[TargetPrior]:
    return [_target(entity_id, 9400) for entity_id in NEST_IDS] + [
        _target(entity_id, 9600) for entity_id in SITE_IDS
    ]


def _ships() -> list[TargetPrior]:
    return [_target(entity_id, 9500, lon=11.65, lat=2.2) for entity_id in SHIP_IDS]


def _policy():
    return build_red_baseline("r11_lethal_package", RULES, seed=0)


def _kind(platforms, platform_id: int) -> str:
    return {item.entity_id: item.kind for item in platforms}[platform_id]


def _counter(assignments, platforms, kind: str) -> Counter:
    return Counter(
        item.target_id
        for item in assignments
        if _kind(platforms, item.platform_id) == kind
    )


def test_r11_is_registered_as_a_red_baseline():
    assert "r11_lethal_package" in RED_POLICY_CHOICES
    assert _policy() is not None


def test_every_surface_objective_receives_a_lethal_heavy_package():
    platforms = _fleet()
    assignments = _policy().decide(_obs(platforms, _surfaces()))

    heavy = _counter(assignments, platforms, "H")
    assert set(heavy) == set(NEST_IDS) | set(SITE_IDS)
    assert all(heavy[entity_id] >= 2 for entity_id in NEST_IDS)
    assert all(heavy[entity_id] >= 1 for entity_id in SITE_IDS)


def test_the_whole_medium_salvo_is_packaged_onto_the_surface_targets():
    platforms = _fleet()
    assignments = _policy().decide(_obs(platforms, _surfaces()))

    medium = _counter(assignments, platforms, "M")
    assert set(medium) == set(NEST_IDS) | set(SITE_IDS)
    assert all(medium[entity_id] >= 2 for entity_id in NEST_IDS + SITE_IDS)
    assert sum(medium.values()) == 24


def test_heavy_spare_goes_to_a_site_whose_hit_lands_outside_the_window(monkeypatch):
    # R11_HEAVY_ETA_GUARD=1：窗口外命中的重弹只打 12，18 血的阵地会留 6 血，
    # 所以两发富余弹要补给“预计到达时间落在天眼窗口外”的阵地。
    monkeypatch.setenv("R11_HEAVY_ETA_GUARD", "1")
    sites = [
        _target(6000, 9600, lon=12.0, lat=2.2),  # 约 55 km → ETA 277 s，早于窗口
        _target(6001, 9600, lon=12.2, lat=2.3),  # 约 78 km → ETA 392 s，窗口内
    ]
    platforms = [_platform(1000 + index, "H") for index in range(4)]

    assignments = _policy().decide(_obs(platforms, sites))

    heavy = _counter(assignments, platforms, "H")
    assert heavy[6000] == 3
    assert heavy[6001] == 1


def test_medium_focus_saturates_the_two_reachable_sites(monkeypatch):
    # R11_MEDIUM_FOCUS>0：中弹没有自己的雷达，只有落在天眼窗口内的弹才有意义，
    # 所以放弃“每个面目标 2 发”，改为把 24 发均分给最近的两个阵地。
    monkeypatch.setenv("R11_MEDIUM_FOCUS", "2")
    sites = [
        _target(6000, 9600, lon=12.0, lat=2.24),
        _target(6001, 9600, lon=12.1, lat=2.06),
        _target(6002, 9600, lon=12.6, lat=2.6),
    ]
    platforms = [_platform(2000 + index, "M") for index in range(24)]

    assignments = _policy().decide(_obs(platforms, sites))

    medium = _counter(assignments, platforms, "M")
    assert set(medium) == {6000, 6001}
    assert medium[6000] == 12
    assert medium[6001] == 12


def test_low_sorties_are_split_evenly_over_the_detected_dark_nests():
    platforms = _fleet()
    policy = _policy()
    # 低弹只有在天眼目录（R11_SATELLITE_STEP=280 之后的首个重规划）亮起后才分包。
    assignments = policy.decide(_obs(platforms, _surfaces() + _ships(), step=300))

    low = _counter(assignments, platforms, "L")
    assert set(low) == set(SHIP_IDS)
    assert sum(low.values()) == 85
    assert max(low.values()) - min(low.values()) <= 1


def test_low_sorties_take_the_nearest_hull_within_a_balanced_share():
    # 南北两条船：每发 L 都应挑自己最近的船，而不是按 id 轮流。
    platforms = [
        _platform(3000, "L", ),
        _platform(3001, "L"),
    ]
    platforms[0] = PlatformState(3000, "L", Position(lon=12.2, lat=2.05), True)
    platforms[1] = PlatformState(3001, "L", Position(lon=12.2, lat=2.45), True)
    ships = [
        TargetPrior(8000, 9500, Position(lon=12.2, lat=2.10)),
        TargetPrior(8001, 9500, Position(lon=12.2, lat=2.40)),
        TargetPrior(8002, 9500, Position(lon=12.2, lat=2.50)),
    ]
    assignments = _policy().decide(_obs(platforms, _surfaces() + ships, step=300))
    by_platform = {item.platform_id: item.target_id for item in assignments}
    assert by_platform[3000] == 8000
    assert by_platform[3001] in (8001, 8002)
    low = _counter(assignments, platforms, "L")
    assert max(low.values()) - min(low.values()) <= 1


def test_the_low_salvo_fires_anyway_when_the_satellite_never_opens():
    platforms = _fleet()
    policy = _policy()
    # 天眼负责人阵亡/请求被拒时，固定等待会让 85 发一发不出。
    assignments = policy.decide(_obs(platforms, _surfaces() + _ships(), step=420))
    low = _counter(assignments, platforms, "L")
    assert len(low) == len(SHIP_IDS)
    assert sum(low.values()) == 85


def test_low_sorties_wait_for_the_satellite_catalogue():
    platforms = _fleet()
    policy = _policy()
    first = policy.decide(_obs(platforms, _surfaces()))
    assert _counter(first, platforms, "L") == Counter()

    # 首条暗巢轨迹可能在簇内只认识一艘船时到达：此时分包会把 85 发全部钉死
    # 在这一艘上，而固定 ID 永远不会释放它们。天眼目录之前的船必须继续等。
    partial = policy.decide(_obs(platforms, _surfaces() + _ships()[:1], step=STEP + 20))
    assert _counter(partial, platforms, "L") == Counter()

    second = policy.decide(_obs(platforms, _surfaces() + _ships(), step=300))
    low = _counter(second, platforms, "L")
    assert set(low) == set(SHIP_IDS)
    assert sum(low.values()) == 85


def test_target_ids_stay_fixed_after_the_first_assignment():
    platforms = _fleet()
    policy = _policy()
    fixed = {
        item.platform_id: item.target_id
        for item in policy.decide(_obs(platforms, _surfaces() + _ships(), step=300))
    }
    assert len(fixed) == 126

    # 目标位置移动后，每个平台仍被重新发出，但目标 id 不变：坐标由 commander 刷新。
    moved = [
        _target(entity_id, entity_type, lon=12.25, lat=2.35)
        for entity_type, ids in ((9400, NEST_IDS), (9600, SITE_IDS))
        for entity_id in ids
    ]
    moved += _ships()
    refreshed = policy.decide(_obs(platforms, moved, step=STEP + 200))
    assert {item.platform_id: item.target_id for item in refreshed} == fixed
    assert len(refreshed) == len(fixed)


def test_a_dead_target_drops_its_sorties_instead_of_retargeting():
    platforms = _fleet()
    policy = _policy()
    fixed = {
        item.platform_id: item.target_id
        for item in policy.decide(_obs(platforms, _surfaces() + _ships(), step=300))
    }

    survivors = [target for target in _surfaces() + _ships() if target.entity_id != NEST_IDS[0]]
    refreshed = policy.decide(_obs(platforms, survivors, step=STEP + 200))
    # 没有任何平台被改成别的目标 id；打 5000 的平台不再被发命令。
    assert all(fixed[item.platform_id] == item.target_id for item in refreshed)
    assert NEST_IDS[0] not in {item.target_id for item in refreshed}


def test_a_rebuilt_commander_inherits_the_fixed_assignments():
    platforms = _fleet()
    opening = _policy().decide(_obs(platforms, _surfaces() + _ships(), step=300))

    # 通信簇重建意味着新的 commander/policy 实例：它必须继承既有分配，
    # 既不重新分包，也不把任何平台改成别的目标。
    rebuilt = _policy().decide(_obs(platforms, _surfaces() + _ships(), step=STEP + 200))
    assert {item.platform_id: item.target_id for item in rebuilt} == {
        item.platform_id: item.target_id for item in opening
    }
    assert len(rebuilt) == len(opening)


def test_only_one_heavy_platform_requests_the_satellite_inside_its_window():
    platforms = _fleet()
    policy = _policy()
    # 天眼只亮 180 秒：开局就点亮会在重弹到达（约 300-560 步）前耗尽。
    policy.decide(_obs(platforms, _surfaces(), step=STEP))
    assert policy.satellite_platform_ids == frozenset()

    policy.decide(_obs(platforms, _surfaces(), step=280))
    leaders = policy.satellite_platform_ids
    assert len(leaders) == 1
    assert _kind(platforms, next(iter(leaders))) == "H"

    # 请求只发一次：窗口开启后的下一轮重规划不再重复请求（引擎会告警）。
    policy.decide(_obs(platforms, _surfaces(), step=300))
    assert policy.satellite_platform_ids == frozenset()


def test_the_window_elects_a_live_leader_when_the_opening_leader_died(monkeypatch):
    monkeypatch.setenv("R11_SATELLITE_STEP", "280")
    platforms = _fleet()
    policy = _policy()
    # 开局长机（1000）在窗口打开前阵亡：到 280 步时只剩其它重弹，天眼不能因此丢掉。
    policy.decide(_obs(platforms, _surfaces(), step=120))
    assert policy.satellite_platform_ids == frozenset()

    survivors = [item for item in platforms if item.entity_id != 1000]
    policy.decide(_obs(survivors, _surfaces(), step=280))
    replacement = policy.satellite_platform_ids
    assert len(replacement) == 1
    assert next(iter(replacement)) in {1001, 1002}


def test_a_fixed_sortie_without_an_objective_keeps_its_launch_course():
    from policies.red.commander import RedBaselineCommander

    nest = TargetPrior(5000, 9400, Position(lon=12.0, lat=2.2))
    commander = RedBaselineCommander(targets=(nest,), policy_name="r11_lethal_package")
    commander.register_platform(1000)
    commander.register_platform(3000)
    observations = (
        _observation(0, 1000, 21000),
        _observation(0, 3000, 21002),
    )
    commander.begin_step(observations)
    # 重弹拿到了母巢；L 在开局还没有暗巢情报，若发搜索航点，它会飞到航点自爆。
    assert int(commander.action_for(1000, 0)[2]) == 5000
    assert commander.action_for(3000, 0) is None

def test_refresh_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("R11_REFRESH_COORDINATES", "0")
    platforms = _fleet()
    policy = _policy()
    first = policy.decide(_obs(platforms, _surfaces() + _ships(), step=300))
    assert len(first) == 126
    assert policy.decide(_obs(platforms, _surfaces() + _ships(), step=STEP + 200)) == ()


def _observation(step: int, entity_id: int, entity_type: int, *, lon=11.5, lat=2.2, detect=None):
    return {
        "step": step,
        "entity_id": entity_id,
        "self": {
            "type": entity_type,
            "health": 1,
            "isVisible": True,
            "position": {"lon": lon, "lat": lat, "alt": 0.0},
            "detectInfo": detect or {},
        },
    }


def test_the_commander_keeps_the_fixed_id_when_the_package_saturates():
    from policies.red.commander import RedBaselineCommander

    nest = TargetPrior(5000, 9400, Position(lon=12.0, lat=2.2))
    commander = RedBaselineCommander(targets=(nest,), policy_name="r11_lethal_package")
    platforms = [_platform(1000, "H"), _platform(1001, "H"), _platform(1002, "M")]
    for platform in platforms:
        commander.register_platform(platform.entity_id)
    first = _observation(0, 1000, 21000)
    rest = [_observation(0, 1001, 21000), _observation(0, 1002, 21001)]
    commander.begin_step((first, *rest))
    for platform in platforms:
        command = commander.action_for(platform.entity_id, 0)
        assert command is not None and int(command[2]) == 5000

    # 第二轮：母巢血量已被两发 H 覆盖，普通策略会改派，固定 ID 策略必须保持。
    second = [_observation(5, platform.entity_id, type_id) for platform, type_id in (
        (platforms[0], 21000), (platforms[1], 21000), (platforms[2], 21001)
    )]
    commander.begin_step(tuple(second))
    for platform in platforms:
        command = commander.action_for(platform.entity_id, 5)
        assert command is not None and int(command[2]) == 5000

    # 非重规划步也要拿到同一目标 ID 的命令：引擎会用执行机当前的情报刷新坐标。
    for step in (1, 2, 3):
        for value in second:
            value["step"] = step
        commander.begin_step(tuple(second))
        for platform in platforms:
            command = commander.action_for(platform.entity_id, step)
            assert command is not None and int(command[2]) == 5000


def test_claim_retarget_returns_the_fixed_id_and_stops_when_it_dies():
    policy = _policy()
    platforms = _fleet()
    policy.decide(_obs(platforms, _surfaces()))
    target = next(item for item in _surfaces() if item.entity_id == NEST_IDS[1])
    assigned = next(
        pid for pid, tid in policy._assignments.items() if tid == NEST_IDS[1]
    )
    alive = policy.claim_retarget(_platform(assigned, "H"), (target,))
    assert alive is not None and alive.entity_id == NEST_IDS[1]
    assert policy.claim_retarget(_platform(9999, "H"), (target,)) is None


def _track_observation(step, entity_id, lon, lat):
    return {
        "step": step,
        "self": {
            "detectInfo": {
                str(entity_id): {
                    "entity_id": entity_id,
                    "entity_type": 9600,
                    "lla": {"x": lon, "y": lat, "z": 0.0},
                }
            }
        },
    }


