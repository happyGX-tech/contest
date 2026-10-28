"""Generate twenty role-layered competition maps.

Red launch coordinates are immutable. Blue interceptor sites occupy explicit
forward defence lines, scoring targets occupy rear belts, and unmanned boats
are sampled along distinct map-wide patrol routes. This replaces the former
small-ring expansion around a few legacy anchors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from generate_final_24 import (
    BLUE_HP,
    DIFFICULTIES,
    INTERCEPTOR_TYPE,
    RADAR_TYPE,
    RED_TYPES,
    apply_health,
    expand_blue_roots,
    next_entity_id,
    resize_interceptors,
    resize_red_ammunition,
)
from generate_structural_25 import entity_map, root_metadata, set_formation_position, write_json
from remove_radar_distance_config import strip_radar_distance


Point = tuple[float, float]
PathGroup = tuple[tuple[Point, ...], ...]
MAP_AREA = {"lonMin": 115.8, "lonMax": 128.2, "latMin": 18.4, "latMax": 28.2}
MAX_STEPS = 3000
LEVEL_LABELS = {"easy": "简单", "medium": "中档", "hard": "复杂"}

# Exactly 20% more blue scoring roots than the preceding 20/30/40-root set.
ROOT_COUNTS = {
    "easy": {"target": 5, "site": 10, "ship": 9},
    "medium": {"target": 7, "site": 15, "ship": 14},
    "hard": {"target": 10, "site": 20, "ship": 18},
}

# Each row preserves the former high:medium:low:interceptor ratio 5:7:18:27.
# Scaling it with the root count keeps blue root HP near 20% of nominal damage.
RED_COUNTS = {
    "easy": {"high": 35, "medium": 49, "low": 126},
    "medium": {"high": 50, "medium": 70, "low": 180},
    "hard": {"high": 70, "medium": 98, "low": 252},
}
INTERCEPTOR_COUNTS = {"easy": 189, "medium": 270, "hard": 378}

# Fixed offsets are part of the map definition. They never depend on an
# experiment seed, so all policies see exactly the same deployment geometry.
FIXED_OFFSETS: tuple[Point, ...] = (
    (-0.16, 0.09), (0.11, -0.14), (0.06, 0.15), (-0.09, -0.07),
    (0.17, 0.02), (-0.04, 0.12), (0.13, -0.10), (-0.12, 0.05),
    (0.03, -0.16), (0.09, 0.11), (-0.15, -0.03), (0.14, 0.07),
)
FIXED_CROSS_LAYER = (1.10, 1.25, 1.40, 1.55, 1.20)


def path(*points: Point) -> tuple[Point, ...]:
    if len(points) < 2:
        raise ValueError("a placement path needs at least two points")
    return tuple(points)


@dataclass(frozen=True)
class MapSpec:
    case_id: str
    difficulty: str
    title: str
    design_intent: str
    route_options: tuple[str, ...]
    site_paths: PathGroup
    ship_paths: PathGroup
    target_paths: PathGroup


MAP_SPECS: tuple[MapSpec, ...] = (
    MapSpec(
        "E01", "easy", "单门扇形", "断墙控制中路，后续扇形支线给出直穿或上下绕行。",
        ("中央破门", "北翼外绕", "南翼外绕"),
        (path((119.7, 19.4), (119.7, 22.1)), path((119.7, 24.3), (119.7, 27.2)), path((120.5, 23.2), (123.0, 26.0)), path((120.5, 23.2), (123.0, 20.4))),
        (path((119.2, 26.8), (121.7, 25.2), (124.3, 26.5)), path((119.5, 19.6), (122.3, 21.0), (126.0, 19.8)), path((121.0, 23.1), (125.6, 23.8))),
        (path((125.4, 20.2), (126.8, 21.2)), path((126.0, 23.4), (127.1, 24.2)), path((125.3, 26.5), (126.8, 27.0))),
    ),
    MapSpec(
        "E02", "easy", "双层错门", "两道防线的缺口上下错置，最短路线必须换道。",
        ("先南后北换道", "北侧破前门", "南侧绕后门"),
        (path((119.5, 18.9), (119.5, 22.3)), path((119.5, 24.5), (119.5, 27.4)), path((122.1, 20.5), (122.1, 24.7)), path((122.1, 26.2), (122.1, 27.4))),
        (path((119.0, 27.0), (121.0, 24.6), (123.2, 25.6), (125.0, 23.2)), path((119.2, 20.0), (121.3, 21.4), (123.8, 19.4)), path((122.7, 27.0), (126.1, 26.0))),
        (path((125.3, 19.8), (127.0, 21.0)), path((125.8, 23.7), (127.2, 23.0)), path((125.2, 26.8), (126.9, 27.2))),
    ),
    MapSpec(
        "E03", "easy", "楔形漏斗", "两条斜向防线逐步收束，中轴承受最长连续覆盖。",
        ("漏斗中轴强突", "北侧外缘", "南侧外缘"),
        (path((119.1, 27.3), (120.8, 25.7), (123.2, 23.5)), path((119.1, 19.2), (120.8, 21.1), (123.2, 23.5))),
        (path((119.3, 26.2), (122.0, 26.8), (125.2, 25.3)), path((119.4, 20.3), (122.3, 19.5), (125.4, 21.4)), path((121.0, 23.5), (126.0, 23.5))),
        (path((125.0, 21.2), (126.8, 22.1)), path((125.5, 24.0), (127.2, 25.3)), path((126.0, 26.7), (127.3, 27.2))),
    ),
    MapSpec(
        "E04", "easy", "三岛跳转", "三组分离阵地形成跳点式覆盖，路线可在岛间切换。",
        ("北岛外缘", "中部跳转", "南岛外缘"),
        (path((119.4, 25.2), (120.4, 27.0)), path((120.3, 19.3), (121.4, 21.4)), path((121.9, 23.0), (123.5, 24.5))),
        (path((119.0, 22.6), (121.2, 24.0), (123.8, 22.0), (126.0, 24.1)), path((120.0, 27.3), (123.0, 26.4), (126.6, 27.0)), path((119.5, 19.1), (123.2, 20.0), (126.2, 19.3))),
        (path((125.0, 20.8), (126.6, 21.5)), path((125.4, 23.6), (127.1, 24.2)), path((125.3, 26.3), (126.9, 27.1))),
    ),
    MapSpec(
        "E05", "easy", "偏置新月", "偏南新月压迫正面，北侧长绕与弧心破点并存。",
        ("新月弧心破点", "北侧长绕", "南端切入"),
        (path((119.2, 20.0), (120.0, 22.0), (121.5, 23.2), (123.4, 23.7)), path((119.5, 19.0), (121.8, 19.5), (123.1, 21.0))),
        (path((119.1, 26.9), (122.0, 27.3), (125.7, 26.2)), path((120.0, 21.0), (123.0, 24.8), (126.4, 22.2)), path((119.4, 18.9), (124.0, 19.5))),
        (path((125.3, 20.4), (126.8, 21.4)), path((125.3, 23.7), (127.1, 24.0)), path((125.4, 26.0), (127.0, 27.1))),
    ),
    MapSpec(
        "E06", "easy", "三路栅栏", "短栅栏压迫北中南三路，三条路线覆盖长度不同。",
        ("北路短穿", "中路连续破点", "南路宽绕"),
        (path((119.4, 25.8), (121.5, 25.8)), path((120.0, 22.9), (122.8, 22.9)), path((119.2, 19.8), (120.8, 20.8), (122.2, 20.8))),
        (path((119.1, 27.2), (123.0, 26.6), (126.2, 25.5)), path((119.4, 23.8), (122.0, 24.0), (125.2, 23.0)), path((120.0, 19.0), (123.7, 20.0), (126.5, 19.2))),
        (path((125.3, 20.7), (126.7, 21.5)), path((125.2, 23.8), (127.0, 24.0)), path((125.5, 26.5), (127.0, 27.0))),
    ),
    MapSpec(
        "M01", "medium", "双门分流", "两座独立前门保护不同目标群，要求分配突防兵力。",
        ("北门集中突破", "南门集中突破", "双门同步牵制"),
        (path((119.3, 23.8), (120.4, 27.2)), path((119.3, 22.1), (120.4, 18.9)), path((121.8, 24.8), (123.5, 26.4)), path((121.8, 21.0), (123.5, 19.5))),
        (path((119.0, 27.4), (122.6, 25.0), (126.0, 27.0)), path((119.1, 19.0), (122.8, 21.2), (126.3, 19.5)), path((120.0, 23.0), (124.0, 23.0), (126.5, 24.0))),
        (path((125.0, 19.5), (127.1, 21.2)), path((125.2, 25.2), (127.2, 27.0))),
    ),
    MapSpec(
        "M02", "medium", "菱堡外环", "前中部菱形堡垒压住直线，外环提供不等长绕行。",
        ("击穿菱心", "北外环", "南外环"),
        (path((119.2, 23.1), (121.3, 27.0), (123.4, 23.1), (121.3, 19.1), (119.2, 23.1)),),
        (path((118.9, 26.8), (121.0, 27.5), (124.4, 26.0), (126.5, 27.0)), path((119.0, 19.4), (121.4, 18.9), (124.5, 20.5), (126.3, 19.4)), path((120.1, 23.0), (126.2, 23.0))),
        (path((125.2, 20.1), (127.1, 21.4)), path((125.1, 23.4), (127.3, 24.0)), path((125.3, 26.1), (127.1, 27.2))),
    ),
    MapSpec(
        "M03", "medium", "沙漏双颈", "两个收缩口串联，破坏顺序会改变后续通道宽度。",
        ("连续破双颈", "第一颈北绕", "第二颈南切"),
        (path((119.1, 27.2), (120.5, 24.0), (121.3, 23.0)), path((119.1, 19.1), (120.5, 22.0), (121.3, 23.0)), path((121.7, 23.0), (122.7, 25.0), (123.6, 27.0)), path((121.7, 23.0), (122.7, 21.0), (123.6, 19.1))),
        (path((119.0, 25.8), (121.0, 23.8), (124.0, 25.5), (126.2, 27.1)), path((119.2, 20.0), (121.5, 22.0), (124.2, 20.3), (126.4, 19.0)), path((121.0, 27.1), (124.5, 23.2), (126.0, 22.0))),
        (path((125.2, 19.8), (127.1, 21.0)), path((125.2, 23.2), (127.2, 23.8)), path((125.2, 26.0), (127.2, 27.1))),
    ),
    MapSpec(
        "M04", "medium", "北窗南堡", "南部多层密集防御，北部保留较远但较疏的入口。",
        ("南部逐层破点", "北窗远绕", "中路切换"),
        (path((119.2, 19.1), (120.5, 21.8), (123.6, 22.2)), path((119.4, 23.4), (121.6, 23.7), (123.2, 24.8)), path((120.5, 25.8), (122.0, 26.1))),
        (path((119.0, 27.3), (122.0, 27.1), (126.3, 26.0)), path((119.3, 20.0), (122.2, 21.0), (125.0, 19.2)), path((120.0, 24.5), (124.0, 23.4), (126.4, 24.6))),
        (path((125.1, 19.8), (127.2, 21.0)), path((125.4, 23.0), (127.2, 24.0)), path((125.1, 26.1), (127.1, 27.2))),
    ),
    MapSpec(
        "M05", "medium", "弧门侧切", "弧形前门把直线引向高压区，侧翼缺口要求提前转向。",
        ("弧心正面突破", "北端侧切", "南端长绕"),
        (path((119.0, 19.5), (119.8, 22.4), (121.5, 24.6), (123.7, 25.5)), path((120.0, 27.0), (122.0, 26.6))),
        (path((119.1, 26.2), (122.0, 24.0), (125.8, 26.8)), path((119.3, 19.0), (122.5, 20.0), (126.5, 19.8)), path((120.5, 23.0), (124.2, 22.4), (126.4, 24.0))),
        (path((125.2, 20.4), (127.2, 21.2)), path((125.3, 23.6), (127.3, 24.2)), path((125.1, 26.4), (127.0, 27.1))),
    ),
    MapSpec(
        "M06", "medium", "断续阶梯", "四段错位短防线构成阶梯，可逐段换道或攻击连接点。",
        ("沿阶梯逐段换道", "攻击中段连接点", "南北外缘长绕"),
        (path((119.2, 26.6), (120.5, 26.6)), path((120.3, 24.3), (121.8, 24.3)), path((121.3, 21.9), (122.9, 21.9)), path((122.3, 19.6), (123.7, 19.6))),
        (path((119.0, 27.5), (122.0, 25.3), (125.6, 27.0)), path((119.2, 19.0), (121.7, 20.7), (124.5, 18.9)), path((120.0, 23.0), (123.5, 23.5), (126.3, 22.4))),
        (path((125.1, 19.8), (127.3, 21.2)), path((125.4, 23.1), (127.1, 24.0)), path((125.1, 26.0), (127.2, 27.2))),
    ),
    MapSpec(
        "H01", "hard", "三层错位门", "三道纵深门的缺口交替变化，需要多次换道或逐层破门。",
        ("北南北连续换道", "逐层破门", "最外侧长绕"),
        (path((119.0, 18.9), (119.0, 22.4)), path((119.0, 24.2), (119.0, 27.5)), path((121.2, 20.0), (121.2, 24.8)), path((121.2, 26.2), (121.2, 27.4)), path((123.3, 18.9), (123.3, 21.5)), path((123.3, 23.2), (123.3, 27.4))),
        (path((118.9, 27.2), (121.0, 25.0), (123.0, 26.2), (126.2, 24.8)), path((119.0, 19.2), (121.5, 20.8), (124.0, 19.0), (126.5, 20.2)), path((119.5, 23.0), (122.5, 22.5), (125.8, 23.4))),
        (path((125.1, 19.4), (127.3, 21.2)), path((125.2, 22.8), (127.4, 24.2)), path((125.0, 26.1), (127.2, 27.3))),
    ),
    MapSpec(
        "H02", "hard", "S形防带", "连续错向防带迫使短航路多次转弯，外缘更长但覆盖较少。",
        ("沿S形缺口", "北外弧", "南外弧"),
        (path((119.0, 26.8), (120.5, 25.0), (121.0, 22.0), (122.0, 20.1)), path((121.0, 27.0), (122.1, 25.0), (122.7, 22.0), (123.7, 19.2))),
        (path((119.0, 19.0), (121.0, 21.8), (124.0, 20.2), (126.5, 22.0)), path((119.2, 27.4), (121.8, 25.0), (124.6, 27.0), (126.6, 25.3)), path((120.0, 23.7), (123.2, 23.0), (126.2, 23.8))),
        (path((125.1, 19.4), (127.3, 21.0)), path((125.1, 23.0), (127.4, 24.0)), path((125.1, 26.0), (127.2, 27.4))),
    ),
    MapSpec(
        "H03", "hard", "偏置双环", "前环与中环偏心叠加，直穿两环和单侧脱离代价不同。",
        ("双环重叠区强突", "北侧脱环", "南侧脱环"),
        (path((119.0, 23.0), (119.8, 26.2), (121.3, 27.1), (122.0, 24.0), (121.0, 20.2), (119.5, 19.2), (119.0, 23.0)), path((121.0, 23.8), (122.0, 26.0), (123.6, 25.0), (123.5, 21.0), (122.0, 19.7), (121.0, 23.8))),
        (path((119.1, 27.5), (122.0, 26.0), (125.8, 27.0)), path((119.0, 18.9), (122.3, 20.4), (126.2, 19.3)), path((120.0, 23.0), (123.5, 22.5), (126.5, 24.2))),
        (path((125.1, 19.4), (127.2, 21.2)), path((125.0, 22.9), (127.4, 24.0)), path((125.0, 26.2), (127.2, 27.4))),
    ),
    MapSpec(
        "H04", "hard", "鱼骨走廊", "纵深主干与斜向支骨构成关键节点，可断骨或穿越骨间通道。",
        ("主干连续压制", "北侧骨间穿越", "南侧骨间穿越"),
        (path((119.0, 23.1), (123.8, 23.1)), path((119.8, 23.1), (118.9, 26.5)), path((120.8, 23.1), (121.6, 27.1)), path((121.8, 23.1), (120.7, 19.0)), path((122.8, 23.1), (123.8, 19.3))),
        (path((119.0, 27.3), (121.7, 25.5), (124.0, 27.0), (126.2, 25.4)), path((119.1, 19.0), (121.6, 20.7), (124.2, 19.1), (126.6, 20.5)), path((120.0, 24.0), (124.0, 22.0), (126.4, 23.8))),
        (path((125.0, 19.5), (127.3, 21.1)), path((125.2, 22.8), (127.4, 24.1)), path((125.0, 26.1), (127.2, 27.4))),
    ),
    MapSpec(
        "H05", "hard", "三叉收束", "北中南三条防线在中后段收束，适合分波次选择支路。",
        ("北支路", "中央破点", "南支路", "三路分流"),
        (path((119.0, 27.3), (121.3, 25.5), (123.5, 23.2)), path((119.0, 23.2), (123.5, 23.2)), path((119.0, 18.9), (121.3, 20.8), (123.5, 23.2))),
        (path((119.0, 26.3), (121.5, 27.2), (124.5, 25.4), (126.5, 27.0)), path((119.0, 19.8), (121.4, 19.0), (124.3, 21.0), (126.5, 19.3)), path((119.2, 23.2), (126.4, 23.2))),
        (path((125.0, 19.4), (127.4, 21.2)), path((125.2, 22.8), (127.4, 24.0)), path((125.0, 26.1), (127.3, 27.4))),
    ),
    MapSpec(
        "H06", "hard", "钩形纵深", "横向前沿接续纵向钩臂，进入方向决定连续暴露时间。",
        ("钩形内侧短穿", "钩形外缘", "北端切入后南转"),
        (path((119.0, 26.6), (121.7, 26.6), (123.0, 24.8), (123.0, 19.0)), path((119.4, 23.8), (121.0, 22.5))),
        (path((119.0, 27.4), (122.5, 25.2), (125.8, 27.0)), path((119.2, 19.0), (121.3, 20.4), (124.0, 19.0), (126.5, 21.0)), path((120.0, 23.0), (124.5, 23.6), (126.4, 24.8))),
        (path((125.0, 19.4), (127.3, 21.0)), path((125.2, 22.8), (127.4, 24.1)), path((125.1, 26.2), (127.3, 27.4))),
    ),
    MapSpec(
        "H07", "hard", "五岛棋盘", "五个稀疏阵地岛形成非连续覆盖，路线依赖跳转顺序。",
        ("西北—中央跳转", "西南—东南跳转", "外围避岛"),
        (path((119.0, 25.5), (120.2, 27.2)), path((119.1, 19.2), (120.3, 21.1)), path((120.6, 22.7), (122.0, 24.1)), path((122.1, 26.1), (123.7, 27.0)), path((122.2, 19.2), (123.7, 21.0))),
        (path((119.0, 27.4), (121.5, 25.5), (124.4, 27.1), (126.4, 25.2)), path((119.0, 19.0), (121.8, 21.0), (124.3, 19.1), (126.6, 20.8)), path((119.5, 23.0), (122.4, 22.0), (125.0, 24.2), (126.5, 23.4))),
        (path((125.0, 19.4), (127.4, 21.1)), path((125.2, 22.8), (127.4, 24.1)), path((125.0, 26.1), (127.2, 27.4))),
    ),
    MapSpec(
        "H08", "hard", "双弧包络", "方向相反的双弧包络通路，局部缺口不会形成直线捷径。",
        ("连续穿越双弧", "北侧弧外", "南侧弧外"),
        (path((119.0, 19.4), (119.8, 22.4), (121.2, 24.7), (123.0, 26.4)), path((120.0, 27.2), (121.6, 25.1), (122.8, 22.2), (123.7, 19.1))),
        (path((119.0, 27.4), (121.6, 26.0), (124.2, 27.0), (126.5, 25.5)), path((119.0, 19.0), (121.8, 20.4), (124.5, 19.0), (126.6, 20.6)), path((119.4, 23.0), (122.5, 23.8), (125.2, 22.5), (126.5, 24.0))),
        (path((125.0, 19.4), (127.4, 21.2)), path((125.2, 22.8), (127.4, 24.2)), path((125.0, 26.1), (127.2, 27.4))),
    ),
)


def polyline_length(points: tuple[Point, ...]) -> float:
    return sum(math.dist(start, end) for start, end in zip(points, points[1:]))


def point_on_polyline(points: tuple[Point, ...], fraction: float) -> Point:
    lengths = [math.dist(start, end) for start, end in zip(points, points[1:])]
    wanted = min(max(fraction, 0.0), 1.0) * sum(lengths)
    travelled = 0.0
    for (start, end), length in zip(zip(points, points[1:]), lengths):
        if travelled + length >= wanted:
            ratio = 0.0 if length == 0.0 else (wanted - travelled) / length
            return (start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio)
        travelled += length
    return points[-1]


def allocate_paths(paths: PathGroup, count: int) -> tuple[int, ...]:
    """Allocate at least one root per path, then allocate by path length."""
    if count < len(paths):
        raise ValueError(f"cannot place {count} roots on {len(paths)} paths")
    lengths = [polyline_length(points) for points in paths]
    if any(length <= 0.0 for length in lengths):
        raise ValueError("placement paths must have non-zero length")
    remaining = count - len(paths)
    raw = [remaining * length / sum(lengths) for length in lengths]
    allocation = [1 + math.floor(value) for value in raw]
    missing = count - sum(allocation)
    order = sorted(range(len(paths)), key=lambda item: raw[item] - math.floor(raw[item]), reverse=True)
    for index in order[:missing]:
        allocation[index] += 1
    return tuple(allocation)


def sample_paths(
    paths: PathGroup,
    count: int,
    phase: float,
    *,
    fixed_phase: int,
    role: str,
) -> tuple[Point, ...]:
    """Sample complete routes and apply map-defined, non-random variation."""
    positions: list[Point] = []
    for path_index, (points, amount) in enumerate(zip(paths, allocate_paths(paths, count))):
        for ordinal in range(amount):
            lon, lat = point_on_polyline(points, (ordinal + 0.5) / amount)
            offset = 0.025 * math.sin(phase + path_index * 1.7 + ordinal * 2.1)
            fixed_lon, fixed_lat = FIXED_OFFSETS[(fixed_phase + len(positions)) % len(FIXED_OFFSETS)]
            positions.append((lon + fixed_lon, lat + offset + fixed_lat))

    # Front/rear placement is a tendency, not a hard boundary. A small,
    # deterministic subset crosses the nominal layer so maps contain forward
    # objectives and deeper defence nodes without losing their overall roles.
    if role in {"site", "target"}:
        outlier_count = max(1, round(count * (0.15 if role == "site" else 0.20)))
        # Deepest sites and shallowest targets cross their nominal role layer.
        # Selecting them explicitly guarantees a visible but bounded overlap.
        indices = sorted(range(count), key=lambda index: positions[index][0], reverse=role == "site")
        direction = 1.0 if role == "site" else -1.0
        for ordinal, index in enumerate(indices[:outlier_count]):
            lon, lat = positions[index]
            amount = FIXED_CROSS_LAYER[(fixed_phase + ordinal) % len(FIXED_CROSS_LAYER)]
            positions[index] = (lon + direction * amount, lat)
    return tuple(positions)


def case_positions(spec: MapSpec, roots: dict[str, tuple[int, ...]]) -> dict[int, Point]:
    fixed_phase = next(index for index, item in enumerate(MAP_SPECS) if item.case_id == spec.case_id)
    sampled = {
        "site": sample_paths(spec.site_paths, len(roots["site"]), 0.2, fixed_phase=fixed_phase, role="site"),
        "ship": sample_paths(spec.ship_paths, len(roots["ship"]), 1.0, fixed_phase=fixed_phase + 4, role="ship"),
        "target": sample_paths(spec.target_paths, len(roots["target"]), 1.8, fixed_phase=fixed_phase + 8, role="target"),
    }
    return {
        root_id: position
        for kind in ("site", "ship", "target")
        for root_id, position in zip(roots[kind], sampled[kind])
    }


def source_launch_positions(scenario: dict) -> dict[int, set[tuple[float, float, float]]]:
    result: dict[int, set[tuple[float, float, float]]] = {}
    for item in scenario["imagineProfile"]["entityList"]:
        entity = item["entity"]
        if entity["entityType"] not in RED_TYPES.values():
            continue
        lla = entity["lla"]
        result.setdefault(entity["entityType"], set()).add((lla["x"], lla["y"], lla.get("z", 0.0)))
    return result


def launch_marker_groups(scenario: dict) -> tuple[dict[str, object], ...]:
    """Merge physically coincident launch types into one unambiguous marker."""
    role_labels = {"high": "高", "medium": "中", "low": "低"}
    groups: list[dict[str, object]] = []
    locations = source_launch_positions(scenario)
    for role in ("high", "medium", "low"):
        entity_type = RED_TYPES[role]
        for lon, lat, _altitude in sorted(locations.get(entity_type, ())):
            matched = next(
                (
                    group
                    for group in groups
                    if math.dist((lon, lat), (float(group["lon"]), float(group["lat"]))) <= 0.03
                ),
                None,
            )
            if matched is None:
                groups.append({"roles": [role], "label": role_labels[role], "lon": lon, "lat": lat})
                continue
            roles = matched["roles"]
            assert isinstance(roles, list)
            previous_count = len(roles)
            matched["lon"] = (float(matched["lon"]) * previous_count + lon) / (previous_count + 1)
            matched["lat"] = (float(matched["lat"]) * previous_count + lat) / (previous_count + 1)
            roles.append(role)
            matched["label"] = "/".join(role_labels[item] for item in roles)
    return tuple(groups)


def root_health(counts: dict[str, int]) -> float:
    return sum(counts[kind] * BLUE_HP[kind] for kind in counts)


def nominal_red_damage(red_counts: dict[str, int]) -> float:
    return red_counts["high"] * 20.0 + red_counts["medium"] * 5.0 + red_counts["low"]


def haversine_km(a: Point, b: Point) -> float:
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    value = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * math.asin(math.sqrt(value))


def draw_dashed_line(draw, points: list[Point], *, fill: str, width: int = 2) -> None:
    for start, end in zip(points, points[1:]):
        length = math.dist(start, end)
        segments = max(1, int(length / 10.0))
        for index in range(0, segments, 2):
            begin, finish = index / segments, min((index + 1) / segments, 1.0)
            draw.line((start[0] + (end[0] - start[0]) * begin,
                       start[1] + (end[1] - start[1]) * begin,
                       start[0] + (end[0] - start[0]) * finish,
                       start[1] + (end[1] - start[1]) * finish), fill=fill, width=width)


def render_case_map(
    path_out: Path,
    spec: MapSpec,
    scenario: dict,
    roots: dict[str, tuple[int, ...]],
    positions: dict[int, Point],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1560, 980
    left, top, right, bottom = 100, 105, 1505, 880
    image = Image.new("RGB", (width, height), "#fcfbf8")
    draw = ImageDraw.Draw(image, "RGBA")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    title_font = ImageFont.truetype(str(font_path), 29) if font_path.is_file() else ImageFont.load_default()
    label_font = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()
    small_font = ImageFont.truetype(str(font_path), 13) if font_path.is_file() else ImageFont.load_default()
    lon_min, lon_max = MAP_AREA["lonMin"], MAP_AREA["lonMax"]
    lat_min, lat_max = MAP_AREA["latMin"], MAP_AREA["latMax"]

    def screen(lon: float, lat: float) -> Point:
        return (left + (lon - lon_min) / (lon_max - lon_min) * (right - left),
                bottom - (lat - lat_min) / (lat_max - lat_min) * (bottom - top))

    for lon in range(math.ceil(lon_min), math.floor(lon_max) + 1):
        x, _ = screen(lon, lat_min)
        draw.line((x, top, x, bottom), fill="#ddd8ce", width=1)
        draw.text((x - 17, bottom + 8), f"{lon}°E", font=small_font, fill="#625f59")
    for lat in range(math.ceil(lat_min), math.floor(lat_max) + 1):
        _, y = screen(lon_min, lat)
        draw.line((left, y, right, y), fill="#ddd8ce", width=1)
        draw.text((25, y - 8), f"{lat}°N", font=small_font, fill="#625f59")

    red_area = scenario["imagineProfile"].get("redArea", {})
    area_styles = (
        ("coordinatesHM", "#bb3e0325", "#9d2f02", "毁灭者、突击者部署区"),
        ("coordinates", "#d9770620", "#a45a00", "清道夫部署区"),
    )
    for key, fill, outline, area_label in area_styles:
        for polygon in red_area.get(key, ()):
            draw.polygon([screen(lon, lat) for lon, lat in polygon], fill=fill, outline=outline, width=2)
            centre_lon = sum(point[0] for point in polygon) / len(polygon)
            centre_lat = sum(point[1] for point in polygon) / len(polygon)
            x, y = screen(centre_lon, centre_lat)
            draw.text((x - 52, y - 8), area_label, font=small_font, fill=outline)

    for route in spec.site_paths:
        draw.line([screen(*point) for point in route], fill="#315f7d55", width=3)
    for route in spec.ship_paths:
        draw_dashed_line(draw, [screen(*point) for point in route], fill="#42766d70", width=2)
    for route in spec.target_paths:
        draw.line([screen(*point) for point in route], fill="#c47a2c55", width=2)

    for site_id in roots["site"]:
        lon, lat = positions[site_id]
        x, y = screen(lon, lat)
        rx = 200.0 / (111.32 * math.cos(math.radians(lat))) / (lon_max - lon_min) * (right - left)
        ry = 200.0 / 111.32 / (lat_max - lat_min) * (bottom - top)
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill="#53789b08", outline="#53789b36", width=1)
        draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill="#315f7d")
    for ship_id in roots["ship"]:
        x, y = screen(*positions[ship_id])
        draw.polygon(((x, y - 6), (x + 7, y), (x, y + 6), (x - 7, y)), fill="#35776b")
    for target_id in roots["target"]:
        x, y = screen(*positions[target_id])
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#ca7b27")

    for marker in launch_marker_groups(scenario):
        x, y = screen(float(marker["lon"]), float(marker["lat"]))
        draw.polygon(((x, y - 9), (x - 8, y + 7), (x + 8, y + 7)), fill="#a33a35")
        draw.text((x + 10, y - 18), f"{marker['label']}实际初始点", font=label_font, fill="#7b2522")

    counts = ROOT_COUNTS[spec.difficulty]
    draw.text((left, 28), f"{spec.case_id}  {LEVEL_LABELS[spec.difficulty]}  |  {spec.title}  |  200 km视野塔", font=title_font, fill="#293b34")
    draw.text((left, 67), f"母巢/哨卫巢/暗巢={counts['target']}/{counts['site']}/{counts['ship']}  候选方向：{'、'.join(spec.route_options)}", font=label_font, fill="#536057")
    draw.text((left + 360, top + 8), "前沿防御  →  中段机动  →  后方计分目标", font=label_font, fill="#6b6256")
    draw.text((left, 925), "■ 哨卫巢（浅圆为真实200 km视野塔）  ◆ 暗巢（虚线为分布带）  ● 母巢  ▲ 红方实际发射位置", font=small_font, fill="#4c514d")
    image.save(path_out)


def build_case(output_root: Path, spec: MapSpec, source_scenario: dict, source_info: dict) -> None:
    scenario = copy.deepcopy(source_scenario)
    roots = expand_blue_roots(scenario, ROOT_COUNTS[spec.difficulty])
    positions = case_positions(spec, roots)
    entities = entity_map(scenario)
    for root_id, (lon, lat) in positions.items():
        set_formation_position(root_id, lon, lat, entities)
    apply_health(entities, roots)

    source_launch = source_launch_positions(scenario)
    red_counts = RED_COUNTS[spec.difficulty]
    entity_id = resize_red_ammunition(scenario, red_counts, next_entity_id(entities))
    defense_roots = roots["site"] + roots["ship"]
    interceptor_total = INTERCEPTOR_COUNTS[spec.difficulty]
    _entity_id, interceptor_counts = resize_interceptors(scenario, interceptor_total, entity_id, defense_roots)
    strip_radar_distance(scenario)
    scenario["environmentProfile"]["projectName"] = spec.case_id
    scenario["imagineProfile"]["mapArea"] = copy.deepcopy(MAP_AREA)
    sim_step = int(scenario["imagineProfile"]["simStep"])
    scenario["imagineProfile"]["simEndLogicTime"] = MAX_STEPS * sim_step

    if source_launch_positions(scenario) != source_launch:
        raise AssertionError("red launch positions changed")
    for lon, lat in positions.values():
        if not (MAP_AREA["lonMin"] <= lon <= MAP_AREA["lonMax"] and MAP_AREA["latMin"] <= lat <= MAP_AREA["latMax"]):
            raise ValueError(f"{spec.case_id} has a blue root outside mapArea")

    case_dir = output_root / spec.difficulty / spec.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = case_dir / "scenario.json"
    write_json(scenario_path, scenario)
    checksum = hashlib.sha256(scenario_path.read_bytes()).hexdigest()

    entities = entity_map(scenario)
    interceptor_counts_str = {str(key): value for key, value in interceptor_counts.items()}
    root_rows = [root_metadata(root_id, kind, entities, interceptor_counts_str)
                 for kind in ("target", "site", "ship") for root_id in roots[kind]]
    for row in root_rows:
        row.pop("radar_km", None)

    counts = ROOT_COUNTS[spec.difficulty]
    blue_health = root_health(counts)
    red_damage = nominal_red_damage(red_counts)
    all_entities = [item["entity"] for item in scenario["imagineProfile"]["entityList"]]
    public_ids, hidden_ids = roots["target"] + roots["site"], roots["ship"]
    launch_points = [point for values in source_launch.values() for point in values]
    centre_launch = (sum(item[0] for item in launch_points) / len(launch_points),
                     sum(item[1] for item in launch_points) / len(launch_points))

    info = copy.deepcopy(source_info)
    info.pop("radar_ranges_km", None)
    info.update({
        "scenario_id": spec.case_id,
        "display_id": spec.case_id,
        "title": spec.title,
        "difficulty": spec.difficulty,
        "difficulty_name": LEVEL_LABELS[spec.difficulty],
        "entity_template_case": DIFFICULTIES[spec.difficulty].source_case,
        "topology_model": "role_layered_paths_v2",
        "placement_variation": {
            "model": "fixed_offset_table",
            "experiment_seed_dependent": False,
            "maximum_coordinate_offset_degrees": {"longitude": 0.17, "latitude": 0.16},
            "cross_layer_fraction": {"site": 0.15, "target": 0.20},
        },
        "design_intent": spec.design_intent,
        "expected_route_options": list(spec.route_options),
        "placement_paths": {"site": spec.site_paths, "ship": spec.ship_paths, "target": spec.target_paths},
        "red_launch_markers": launch_marker_groups(scenario),
        "max_steps": MAX_STEPS,
        "map_area": copy.deepcopy(MAP_AREA),
        "roots": root_rows,
        "scenario_sha256": checksum,
        "entity_total": len(all_entities),
        "blue_entity_total": sum(entity["sideId"] == 1 for entity in all_entities),
        "blue_root_total": sum(counts.values()),
        "root_counts": copy.deepcopy(counts),
        "radar_total": sum(entity["entityType"] == RADAR_TYPE for entity in all_entities),
        "interceptor_total": interceptor_total,
        "interceptor_counts": interceptor_counts_str,
        "objective_ids": list(public_ids + hidden_ids),
        "objective_total": sum(counts.values()),
        "public_objective_ids": list(public_ids),
        "hidden_objective_ids": list(hidden_ids),
        "red_missile_counts": copy.deepcopy(red_counts),
        "blue_hp_by_kind": copy.deepcopy(BLUE_HP),
        "blue_root_health_total": blue_health,
        "red_nominal_damage_total": red_damage,
        "blue_health_to_red_nominal_damage_ratio": blue_health / red_damage,
        "building_health_total": counts["target"] * BLUE_HP["target"] + counts["site"] * BLUE_HP["site"],
        "interceptor_to_high_medium_ratio": interceptor_total / (red_counts["high"] + red_counts["medium"]),
        "interceptor_to_low_ratio": interceptor_total / red_counts["low"],
        "farthest_blue_root_from_red_launch_centre_km": max(haversine_km(centre_launch, point) for point in positions.values()),
    })
    write_json(case_dir / "case_info.json", info)
    render_case_map(case_dir / "map_info.png", spec, scenario, roots, positions)


def render_contact_sheet(output_root: Path, specs: tuple[MapSpec, ...], filename: str, heading: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    cell_w, cell_h, columns = 900, 600, 2
    rows = math.ceil(len(specs) / columns)
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows + 78), "#f4f1eb")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    title_font = ImageFont.truetype(str(font_path), 28) if font_path.is_file() else ImageFont.load_default()
    label_font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    draw.text((24, 20), heading, font=title_font, fill="#2f3b35")
    for index, spec in enumerate(specs):
        row, col = divmod(index, columns)
        map_image = Image.open(output_root / spec.difficulty / spec.case_id / "map_info.png").convert("RGB")
        map_image.thumbnail((cell_w - 22, cell_h - 58))
        x = col * cell_w + (cell_w - map_image.width) // 2
        y = 78 + row * cell_h + 34
        sheet.paste(map_image, (x, y))
        draw.text((col * cell_w + 12, 78 + row * cell_h + 6), f"{spec.case_id}  {LEVEL_LABELS[spec.difficulty]}  {spec.title}", font=label_font, fill="#33413a")
    sheet.save(output_root / filename)


def render_overviews(output_root: Path) -> None:
    render_contact_sheet(output_root, MAP_SPECS, "final_20_overview.png", "20张分异化地图总览：6张简单 / 6张中档 / 8张复杂")
    for difficulty, filename in (("easy", "overview_easy.png"), ("medium", "overview_medium.png"), ("hard", "overview_hard.png")):
        selected = tuple(spec for spec in MAP_SPECS if spec.difficulty == difficulty)
        render_contact_sheet(output_root, selected, filename, f"{LEVEL_LABELS[difficulty]}地图总览（{len(selected)}张）")


def write_catalog(output_root: Path) -> None:
    lines = [
        "# 20张分异化竞赛地图", "", "## 固定数值规则", "",
        "- 图数按简单/中档/复杂=6/6/8配置。",
        "- 蓝方根实体在上一版基础上增加20%，三档分别为24/36/48个。",
        "- 母巢/哨卫巢/暗巢分别为5/10/9、7/15/14、10/20/18；暗巢均略少于哨卫巢。",
        "- 单体血量继续使用母巢16、哨卫巢12、暗巢0.8。",
        "- 三档红方毁灭者/突击者/清道夫与蓝方自爆虫群保持5:7:18:27。",
        "- 蓝方根实体总血量为红方全部名义伤害的19.35%～19.82%。",
        "- 地图范围扩大为115.8～128.2°E、18.4～28.2°N；红方发射坐标和发射区域未改变。",
        "- 母巢位于后方纵深，哨卫巢位于前中部，暗巢沿各图不同航带分散。",
        "- 正式时限统一为3000步。", "",
        "| 难度 | 图数 | 毁灭者/突击者/清道夫 | 自爆虫群 | 母巢/哨卫巢/暗巢 | 蓝方总血量 | 红方名义伤害 | 比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for difficulty in ("easy", "medium", "hard"):
        counts, red_counts = ROOT_COUNTS[difficulty], RED_COUNTS[difficulty]
        health, damage = root_health(counts), nominal_red_damage(red_counts)
        map_count = sum(item.difficulty == difficulty for item in MAP_SPECS)
        lines.append(f"| {LEVEL_LABELS[difficulty]} | {map_count} | {red_counts['high']}/{red_counts['medium']}/{red_counts['low']} | {INTERCEPTOR_COUNTS[difficulty]} | {counts['target']}/{counts['site']}/{counts['ship']} | {health:.1f} | {damage:.1f} | {health / damage:.2%} |")
    lines += ["", "## 地图结构", "", "| 地图 | 结构 | 空间作用 | 可预期路线 |", "|---|---|---|---|"]
    for spec in MAP_SPECS:
        lines.append(f"| {spec.case_id} | {spec.title} | {spec.design_intent} | {'；'.join(spec.route_options)} |")
    lines += ["", "运行示例：", "", "```powershell", ".\\run_local.cmd run --scenario final20/medium/M01 --red-policy r3_wave_schedule --blue-policy b2_threat_priority --seed 42 --run-id final20_M01 -- --total-rounds 1 --max-steps 3000 --render-mode human --disable-log-color", "```"]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("scenarios/cases"))
    parser.add_argument("--output", type=Path, default=Path("scenarios/cases/final20"))
    args = parser.parse_args()
    for spec in MAP_SPECS:
        difficulty = DIFFICULTIES[spec.difficulty]
        source_dir = args.source_root / spec.difficulty / difficulty.source_case
        source_scenario = json.loads((source_dir / "scenario.json").read_text(encoding="utf-8"))
        source_info = json.loads((source_dir / "case_info.json").read_text(encoding="utf-8"))
        build_case(args.output, spec, source_scenario, source_info)
    render_overviews(args.output)
    write_catalog(args.output)
    print(f"generated={len(MAP_SPECS)} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
