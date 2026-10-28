"""Build the final 24 structurally distinct competition maps.

Every blue root kind has one health value across all difficulty levels.  The
levels reuse E01/M01/H01 respectively, so difficulty grows through the number
of scoring entities and ammunition rather than arbitrary fractional health.
Red launch sites remain identical in all three source cases.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from generate_structural_25 import (
    LAYOUTS,
    RADAR_TYPE,
    SITE_IDS,
    Layout,
    ecef,
    entity_map,
    root_metadata,
    set_formation_position,
    write_json,
)
from remove_radar_distance_config import strip_radar_distance


RED_TYPES = {"high": 21000, "medium": 21001, "low": 21002}
INTERCEPTOR_TYPE = 24000
BLUE_ROOT_TYPES = {"target": 9400, "site": 9600, "ship": 9500}


@dataclass(frozen=True)
class DifficultySpec:
    name: str
    name_zh: str
    high: int
    medium: int
    low: int
    interceptors: int
    source_case: str

    @property
    def red_counts(self) -> dict[str, int]:
        return {"high": self.high, "medium": self.medium, "low": self.low}

    @property
    def building_damage(self) -> float:
        return self.high * 20.0 + self.medium * 5.0

DIFFICULTIES = {
    "easy": DifficultySpec("easy", "简单", 30, 42, 108, 162, "E01"),
    "medium": DifficultySpec("medium", "中等", 45, 63, 162, 243, "M01"),
    "hard": DifficultySpec("hard", "困难", 60, 84, 216, 324, "H01"),
}

# Baseline effective damage without satellite bonus:
# high -> target/site = 20 * 0.8 / 20 * 0.6, medium -> 5 * 0.8 / 5 * 0.6,
# low -> ship = 1 * 0.8.  These values therefore avoid half-useful follow-up
# rounds: one high or four medium rounds destroys a target/site, and one low
# round destroys a ship.
BLUE_HP = {"target": 16.0, "site": 12.0, "ship": 0.8}
DESIRED_ROOT_COUNTS = {
    "easy": {"target": 5, "site": 2, "ship": 2},
    "medium": {"target": 6, "site": 5, "ship": 2},
    "hard": {"target": 7, "site": 8, "ship": 3},
}


EXTRA_LAYOUTS = {
    "T26": Layout("T26", "螺旋折返", "覆盖区向内旋转，短路需连续换向，外绕路更长但转向更少。", (106, 310), ((120.0,24.8),(121.6,24.6),(121.7,21.6),(120.5,22.0)), ((119.5,20.5),(122.5,25.3)), ((123.0,25.2),(123.6,24.0),(123.6,22.0),(123.0,20.8))),
    "T27": Layout("T27", "三门不等宽", "南、中、北三个入口宽度和纵深不同，考察路径与压力分配。", (106,), ((120.0,24.9),(120.5,23.0),(120.0,21.1),(122.0,22.8)), ((121.0,25.5),(121.2,20.5)), ((123.1,25.2),(123.6,24.0),(123.5,22.0),(123.0,20.8))),
    "T28": Layout("T28", "钩形走廊", "防御链先横向压迫再纵向封锁，直线进入会连续穿越覆盖区。", (310,), ((120.0,25.0),(121.0,23.8),(122.0,22.7),(122.0,20.8)), ((119.5,20.5),(122.6,25.4)), ((123.0,25.2),(123.6,24.0),(123.5,22.0),(123.0,20.7))),
    "T29": Layout("T29", "偏置盆地", "母巢位于偏东的半包围区，西北入口较短但暴露时间长。", (54, 312), ((120.6,25.0),(121.8,24.2),(121.8,21.8),(120.6,21.0)), ((119.5,23.0),(122.6,23.0)), ((122.9,25.0),(123.5,23.8),(123.5,22.2),(122.9,21.0))),
    "T30": Layout("T30", "断续弧带", "防御节点形成不连续弧带，局部缺口可通行但会导向不同目标群。", (106, 310), ((120.0,25.0),(121.0,24.4),(122.0,23.2),(121.2,21.2)), ((119.5,20.5),(122.5,25.3)), ((123.0,25.2),(123.6,24.0),(123.5,22.0),(123.0,20.8))),
}


LAYOUT_BY_ID = {layout.case_id: layout for layout in LAYOUTS} | EXTRA_LAYOUTS

# Avoid mirrored and one-variable-at-a-time pairs. Each group mixes unrelated
# structures so the final set reads as 24 standalone tasks, not controls.
SELECTION = {
    "easy": (
        ("E01", "T01", "单节点破口与南北绕行"),
        ("E02", "T03", "短高压路与长低压路选择"),
        ("E03", "T10", "分叉后的兵力分流"),
        ("E04", "T13", "识别外环缺口及绕行成本"),
        ("E05", "T16", "前中后三层优先级"),
        ("E06", "T19", "双团簇中间缝的突防时机"),
        ("E07", "T22", "多缺口换道选择"),
        ("E08", "T26", "多次转向与外绕距离权衡"),
    ),
    "medium": (
        ("M01", "T02", "两个独立突破口的资源分配"),
        ("M02", "T05", "中心破点与菱形外绕"),
        ("M03", "T08", "S形换道与中段关键节点"),
        ("M04", "T12", "串联双颈的攻击顺序"),
        ("M05", "T14", "内弧强突与外弧远绕"),
        ("M06", "T17", "单侧长距离绕行价值"),
        ("M07", "T20", "通道逐步收窄时的前置分流"),
        ("M08", "T27", "三个不等宽入口的压力估计"),
    ),
    "hard": (
        ("H01", "T06", "斜向纵深链的中段断链"),
        ("H02", "T09", "Z形多次换道规划"),
        ("H03", "T15", "错位双门的跨层航路"),
        ("H04", "T23", "串联关键点的连续破坏"),
        ("H05", "T25", "短窄、中等破点和长绕三路决策"),
        ("H06", "T28", "钩形覆盖下的方向选择"),
        ("H07", "T29", "偏置半包围下的入口判断"),
        ("H08", "T30", "断续弧带缺口与目标匹配"),
    ),
}


def next_entity_id(entities: dict[int, dict]) -> int:
    return max(entities) + 1


def blue_root_ids(scenario: dict) -> dict[str, tuple[int, ...]]:
    """Return blue objective roots in their stable scenario order."""
    entities = [item["entity"] for item in scenario["imagineProfile"]["entityList"]]
    return {
        kind: tuple(entity["id"] for entity in entities if entity["sideId"] == 1 and entity["entityType"] == entity_type)
        for kind, entity_type in BLUE_ROOT_TYPES.items()
    }


def expand_blue_roots(scenario: dict, desired: dict[str, int]) -> dict[str, tuple[int, ...]]:
    """Clone existing objective formations until the requested counts exist."""
    items = scenario["imagineProfile"]["entityList"]
    entities = entity_map(scenario)
    roots = {kind: list(ids) for kind, ids in blue_root_ids(scenario).items()}
    entity_id = next_entity_id(entities)
    for kind, desired_count in desired.items():
        templates = tuple(roots[kind])
        while len(roots[kind]) < desired_count:
            template_id = templates[(len(roots[kind]) - len(templates)) % len(templates)]
            template = entities[template_id]
            root_item = copy.deepcopy(template)
            root = root_item["entity"]
            root_id = entity_id
            entity_id += 1
            root["id"] = root_id
            root["parentId"] = -1
            root["childrenId"] = []
            root["nameChn"] = f"{kind}_added_{root_id}"
            root["nameEn"] = f"{kind}_added_{root_id}"
            items.append(root_item)
            entities[root_id] = root_item

            for child_id in template["entity"].get("childrenId", ()):
                child_template = entities[child_id]
                if child_template["entity"]["entityType"] == INTERCEPTOR_TYPE:
                    continue
                child_item = copy.deepcopy(child_template)
                child = child_item["entity"]
                child["id"] = entity_id
                child["parentId"] = root_id
                child["childrenId"] = []
                child["nameChn"] = f"{child['entityType']}_{root_id}"
                child["nameEn"] = f"{child['entityType']}_{root_id}"
                root["childrenId"].append(entity_id)
                items.append(child_item)
                entities[entity_id] = child_item
                entity_id += 1
            roots[kind].append(root_id)
    return {kind: tuple(ids) for kind, ids in roots.items()}


def fit_positions(
    base: tuple[tuple[float, float], ...], count: int, kind: str,
) -> tuple[tuple[float, float], ...]:
    """Adapt a four-site/two-ship layout to the root count of a difficulty."""
    if count <= len(base):
        if count == 1:
            return (base[len(base) // 2],)
        indices = [round(index * (len(base) - 1) / (count - 1)) for index in range(count)]
        return tuple(base[index] for index in indices)

    positions = list(base)
    mean_lon = sum(lon for lon, _ in base) / len(base)
    mean_lat = sum(lat for _, lat in base) / len(base)
    if kind == "site":
        candidates = [
            ((base[0][0] + base[1][0]) / 2 + 0.12, (base[0][1] + base[1][1]) / 2 - 0.10),
            ((base[-2][0] + base[-1][0]) / 2 - 0.12, (base[-2][1] + base[-1][1]) / 2 + 0.10),
            ((base[0][0] + base[2][0]) / 2 - 0.12, (base[0][1] + base[2][1]) / 2 + 0.10),
            ((base[1][0] + base[3][0]) / 2 + 0.12, (base[1][1] + base[3][1]) / 2 - 0.10),
        ]
    elif kind == "target":
        candidates = [
            (mean_lon - 0.22, mean_lat),
            (mean_lon + 0.18, mean_lat + 0.55),
            (mean_lon + 0.18, mean_lat - 0.55),
            (mean_lon - 0.05, mean_lat + 1.00),
        ]
    else:
        candidates = [(mean_lon + 0.35, mean_lat), (mean_lon - 0.35, mean_lat)]
    positions.extend(candidates[: count - len(positions)])
    if len(positions) != count:
        raise ValueError(f"cannot fit {count} {kind} positions")
    return tuple(positions)


def case_positions(layout: Layout, roots: dict[str, tuple[int, ...]]) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    for kind, base in (("site", layout.sites), ("ship", layout.ships), ("target", layout.targets)):
        positions.update(zip(roots[kind], fit_positions(base, len(roots[kind]), kind)))
    return positions


def key_node_ids(layout: Layout, site_ids: tuple[int, ...]) -> tuple[int, ...]:
    key_ordinals = [SITE_IDS.index(site_id) for site_id in layout.key_nodes if site_id in SITE_IDS]
    if not key_ordinals:
        return ()
    scaled = [round(index * (len(site_ids) - 1) / (len(SITE_IDS) - 1)) for index in key_ordinals]
    return tuple(dict.fromkeys(site_ids[index] for index in scaled))


def resize_red_ammunition(scenario: dict, counts: dict[str, int], start_id: int) -> int:
    items = scenario["imagineProfile"]["entityList"]
    templates = {
        kind: copy.deepcopy(next(item for item in items if item["entity"]["entityType"] == entity_type))
        for kind, entity_type in RED_TYPES.items()
    }
    items[:] = [item for item in items if item["entity"]["entityType"] not in RED_TYPES.values()]
    entity_id = start_id
    for kind, count in counts.items():
        for ordinal in range(1, count + 1):
            item = copy.deepcopy(templates[kind])
            entity = item["entity"]
            entity["id"] = entity_id
            entity["parentId"] = -1
            entity["childrenId"] = []
            entity["nameChn"] = f"{kind}_{ordinal:03d}"
            entity["nameEn"] = f"{kind}_{ordinal:03d}"
            items.append(item)
            entity_id += 1
    return entity_id


def distribute(total: int, roots: tuple[int, ...]) -> dict[int, int]:
    quotient, remainder = divmod(total, len(roots))
    return {root_id: quotient + (index < remainder) for index, root_id in enumerate(roots)}


def resize_interceptors(
    scenario: dict, total: int, start_id: int, defense_root_ids: tuple[int, ...],
) -> tuple[int, dict[int, int]]:
    items = scenario["imagineProfile"]["entityList"]
    entities = entity_map(scenario)
    templates: dict[int, dict] = {}
    old_ids: set[int] = set()
    generic_template = copy.deepcopy(next(
        item for item in items if item["entity"]["entityType"] == INTERCEPTOR_TYPE
    ))
    for root_id in defense_root_ids:
        child_ids = entities[root_id]["entity"].get("childrenId", [])
        interceptor_ids = [child_id for child_id in child_ids if entities[child_id]["entity"]["entityType"] == INTERCEPTOR_TYPE]
        templates[root_id] = copy.deepcopy(entities[interceptor_ids[0]]) if interceptor_ids else copy.deepcopy(generic_template)
        old_ids.update(interceptor_ids)
        entities[root_id]["entity"]["childrenId"] = [child_id for child_id in child_ids if child_id not in interceptor_ids]
    items[:] = [item for item in items if item["entity"]["id"] not in old_ids]

    counts = distribute(total, defense_root_ids)
    entity_id = start_id
    entities = entity_map(scenario)
    for root_id, count in counts.items():
        root = entities[root_id]["entity"]
        for ordinal in range(1, count + 1):
            item = copy.deepcopy(templates[root_id])
            entity = item["entity"]
            entity["id"] = entity_id
            entity["parentId"] = root_id
            entity["childrenId"] = []
            entity["lla"] = copy.deepcopy(root["lla"])
            entity["posEcf"] = ecef(root["lla"]["x"], root["lla"]["y"], root["lla"].get("z", 0.0))
            entity["nameChn"] = f"自爆虫群_{root_id}_{ordinal:03d}"
            entity["nameEn"] = f"Interceptor_{root_id}_{ordinal:03d}"
            items.append(item)
            root["childrenId"].append(entity_id)
            entity_id += 1
    return entity_id, counts


def apply_health(entities: dict[int, dict], roots: dict[str, tuple[int, ...]]) -> None:
    for kind, entity_ids in roots.items():
        for entity_id in entity_ids:
            entities[entity_id]["entity"]["survivePoints"] = BLUE_HP[kind]


def render_case_map(
    path: Path, case_id: str, layout: Layout, roots: dict[str, tuple[int, ...]],
    positions: dict[int, tuple[float, float]], key_nodes: tuple[int, ...],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1200, 820
    left, top, right, bottom = 85, 75, 1150, 750
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image, "RGBA")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    title_font = ImageFont.truetype(str(font_path), 26) if font_path.is_file() else ImageFont.load_default()
    label_font = ImageFont.truetype(str(font_path), 15) if font_path.is_file() else ImageFont.load_default()
    small_font = ImageFont.truetype(str(font_path), 13) if font_path.is_file() else ImageFont.load_default()

    def point(lon: float, lat: float) -> tuple[float, float]:
        return left + (lon - 116.0) / 8.0 * (right - left), bottom - (lat - 20.0) / 6.0 * (bottom - top)

    for lon in range(116, 125):
        x, _ = point(lon, 20)
        draw.line((x, top, x, bottom), fill="#cbd5e1", width=1)
        draw.text((x - 15, bottom + 8), f"{lon}°E", font=small_font, fill="#64748b")
    for lat in range(20, 27):
        _, y = point(116, lat)
        draw.line((left, y, right, y), fill="#cbd5e1", width=1)
        draw.text((20, y - 8), f"{lat}°N", font=small_font, fill="#64748b")
    draw.rectangle((*point(116, 24.8), *point(117, 21.5)), fill="#ef444422", outline="#b91c1c", width=2)
    for lon, lat, color in ((117.869691, 25.588449, "#be123c"), (117.100999, 26.46319, "#f97316")):
        x, y = point(lon, lat)
        draw.polygon(((x, y - 9), (x - 8, y + 7), (x + 8, y + 7)), fill=color)

    for site_id in roots["site"]:
        lon, lat = positions[site_id]
        x, y = point(lon, lat)
        rx = 200.0 / (111.32 * math.cos(math.radians(lat))) / 8.0 * (right - left)
        ry = 200.0 / 111.32 / 6.0 * (bottom - top)
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill="#3b82f61a", outline="#2563eb", width=2)
        draw.rectangle((x - 7, y - 7, x + 7, y + 7), fill="#1d4ed8")
        draw.text((x + 9, y - 18), f"S{site_id}" + ("*" if site_id in key_nodes else ""), font=label_font, fill="#1e3a8a")
    for ship_id in roots["ship"]:
        x, y = point(*positions[ship_id])
        draw.polygon(((x, y - 8), (x + 8, y), (x, y + 8), (x - 8, y)), fill="#0f766e")
        draw.text((x + 9, y - 18), f"U{ship_id}", font=label_font, fill="#115e59")
    for target_id in roots["target"]:
        x, y = point(*positions[target_id])
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#d97706")
        draw.text((x + 9, y - 18), f"T{target_id}", font=label_font, fill="#92400e")

    draw.text((left, 22), f"{case_id}  {layout.name}  |  蓝方哨卫巢视野塔半径 200 km", font=title_font, fill="#0f172a")
    draw.text((left, 782), "*为关键节点；圆仅表示哨卫巢视野塔覆盖，暗巢按原始单图设计不挂视野塔", font=small_font, fill="#475569")
    image.save(path)


def build_case(
    *, output_root: Path, case_id: str, difficulty: str, source_layout: Layout,
    focus: str, source_scenario: dict, source_info: dict,
) -> None:
    spec = DIFFICULTIES[difficulty]
    scenario = copy.deepcopy(source_scenario)
    roots_by_kind = expand_blue_roots(scenario, DESIRED_ROOT_COUNTS[difficulty])
    positions = case_positions(source_layout, roots_by_kind)
    keys = key_node_ids(source_layout, roots_by_kind["site"])
    entities = entity_map(scenario)
    for root_id, (lon, lat) in positions.items():
        set_formation_position(root_id, lon, lat, entities)
    apply_health(entities, roots_by_kind)
    entity_id = resize_red_ammunition(scenario, spec.red_counts, next_entity_id(entities))
    defense_root_ids = roots_by_kind["site"] + roots_by_kind["ship"]
    entity_id, interceptor_counts = resize_interceptors(scenario, spec.interceptors, entity_id, defense_root_ids)
    del entity_id
    strip_radar_distance(scenario)
    scenario["environmentProfile"]["projectName"] = case_id

    case_dir = output_root / difficulty / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = case_dir / "scenario.json"
    write_json(scenario_path, scenario)
    sha = hashlib.sha256(scenario_path.read_bytes()).hexdigest()

    entities = entity_map(scenario)
    interceptor_counts_str = {str(key): value for key, value in interceptor_counts.items()}
    roots = [
        root_metadata(root_id, kind, entities, interceptor_counts_str)
        for kind in ("target", "site", "ship")
        for root_id in roots_by_kind[kind]
    ]
    for root in roots:
        root.pop("radar_km", None)

    all_entities = [item["entity"] for item in scenario["imagineProfile"]["entityList"]]
    info = copy.deepcopy(source_info)
    info.pop("radar_ranges_km", None)
    objective_ids = roots_by_kind["target"] + roots_by_kind["site"] + roots_by_kind["ship"]
    public_ids = roots_by_kind["target"] + roots_by_kind["site"]
    building_health = len(roots_by_kind["target"]) * BLUE_HP["target"] + len(roots_by_kind["site"]) * BLUE_HP["site"]
    info.update({
        "scenario_id": case_id,
        "display_id": case_id,
        "title": source_layout.name,
        "difficulty": difficulty,
        "difficulty_name": spec.name_zh,
        "mission_type": "route_and_key_node",
        "mission_type_name": "通道与关键节点",
        "topology_id": source_layout.case_id.lower(),
        "topology_name": source_layout.name,
        "design_intent": source_layout.intent,
        "evaluation_focus": focus,
        "structural_features": ["通道选择", "关键节点", "破点与绕行"],
        "difficulty_mechanism": f"{spec.name_zh}计分实体数量+弹药规模+独立空间拓扑；同类实体血量跨难度固定。",
        "key_node_ids": list(keys),
        "roots": roots,
        "scenario_sha256": sha,
        "entity_total": len(all_entities),
        "blue_entity_total": sum(entity["sideId"] == 1 for entity in all_entities),
        "blue_root_total": len(objective_ids),
        "root_counts": {kind: len(ids) for kind, ids in roots_by_kind.items()},
        "radar_total": sum(entity["entityType"] == RADAR_TYPE for entity in all_entities),
        "interceptor_total": spec.interceptors,
        "interceptor_counts": {str(key): value for key, value in interceptor_counts.items()},
        "objective_ids": list(objective_ids),
        "objective_total": len(objective_ids),
        "public_objective_ids": list(public_ids),
        "hidden_objective_ids": list(roots_by_kind["ship"]),
        "red_missile_counts": spec.red_counts,
        "blue_hp_by_kind": BLUE_HP,
        "ammo_rules": {
            "medium_to_high": spec.medium / spec.high,
            "low_to_high_plus_medium": spec.low / (spec.high + spec.medium),
            "interceptor_to_low": spec.interceptors / spec.low,
        },
        "building_damage_budget": spec.building_damage,
        "building_health_total": building_health,
        "building_health_to_damage_ratio": building_health / spec.building_damage,
    })
    write_json(case_dir / "case_info.json", info)
    render_case_map(case_dir / "map_info.png", case_id, source_layout, roots_by_kind, positions, keys)


def render_overview(output_root: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    cases = [(difficulty, case_id) for difficulty in SELECTION for case_id, _, _ in SELECTION[difficulty]]
    cell_w, cell_h = 420, 300
    sheet = Image.new("RGB", (cell_w * 4, cell_h * 6 + 60), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    title_font = ImageFont.truetype(str(font_path), 25) if font_path.is_file() else ImageFont.load_default()
    label_font = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()
    draw.text((20, 12), "最终24图：同类统一血量 + 三档计分实体/弹药规模", font=title_font, fill="#0f172a")
    for index, (difficulty, case_id) in enumerate(cases):
        row, col = divmod(index, 4)
        image = Image.open(output_root / difficulty / case_id / "map_info.png").convert("RGB")
        image.thumbnail((cell_w - 12, cell_h - 38))
        x = col * cell_w + (cell_w - image.width) // 2
        y = 60 + row * cell_h + 26
        sheet.paste(image, (x, y))
        draw.text((col * cell_w + 8, 60 + row * cell_h + 4), f"{case_id}  {DIFFICULTIES[difficulty].name_zh}", font=label_font, fill="#0f172a")
    sheet.save(output_root / "final_24_overview.png")


def write_catalog(output_root: Path) -> None:
    lines = [
        "# 最终24张竞赛地图",
        "",
        "## 统一规则",
        "",
        "- 不新增实体类型；简单/中等/困难分别以 E01/M01/H01 为模板，按难度复制既有类型的计分实体，红方发射场位置一致。",
        "- 视野塔距离不在地图配置，统一由核心使用 200 km。",
        "- 突击者 = 毁灭者×1.4；清道夫 = (毁灭者+突击者)×1.5；蓝方自爆虫群 = 清道夫×1.5。",
        "- 同类蓝方实体跨难度统一血量：母巢16、哨卫巢12、暗巢0.8。基准命中率下分别对应1枚毁灭者（或4枚突击者）、1枚毁灭者（或4枚突击者）、1枚清道夫。",
        "- 难度通过计分实体数、空间拓扑和弹药规模形成；12.84%由实体数量得到，不再通过放大单体血量实现。",
        "",
        "| 难度 | 毁灭者/突击者/清道夫 | 蓝方自爆虫群 | 母巢/哨卫巢/暗巢数 | 母巢/哨卫巢/暗巢血量 | 建筑血量/名义伤害 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for difficulty, spec in DIFFICULTIES.items():
        first_case = SELECTION[difficulty][0][0]
        info = json.loads((output_root / difficulty / first_case / "case_info.json").read_text(encoding="utf-8"))
        counts = info["root_counts"]
        lines.append(
            f"| {spec.name_zh} | {spec.high}/{spec.medium}/{spec.low} | {spec.interceptors} | "
            f"{counts['target']}/{counts['site']}/{counts['ship']} | 16/12/0.8 | "
            f"{info['building_health_to_damage_ratio']:.2%} |"
        )
    lines += ["", "## 地图结构与考察要点", "", "| 地图 | 难度 | 结构 | 考察要点 |", "|---|---|---|---|"]
    for difficulty, selected in SELECTION.items():
        for case_id, layout_id, focus in selected:
            lines.append(f"| {case_id} | {DIFFICULTIES[difficulty].name_zh} | {LAYOUT_BY_ID[layout_id].name} | {focus} |")
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("scenarios/cases"))
    parser.add_argument("--output", type=Path, default=Path("scenarios/cases/final24"))
    args = parser.parse_args()
    for difficulty, selected in SELECTION.items():
        spec = DIFFICULTIES[difficulty]
        source_dir = args.source_root / difficulty / spec.source_case
        source_scenario = json.loads((source_dir / "scenario.json").read_text(encoding="utf-8"))
        source_info = json.loads((source_dir / "case_info.json").read_text(encoding="utf-8"))
        for case_id, layout_id, focus in selected:
            build_case(
                output_root=args.output,
                case_id=case_id,
                difficulty=difficulty,
                source_layout=LAYOUT_BY_ID[layout_id],
                focus=focus,
                source_scenario=source_scenario,
                source_info=source_info,
            )
    render_overview(args.output)
    write_catalog(args.output)
    print(f"generated=24 output={args.output.resolve()}")


if __name__ == "__main__":
    main()
