"""Generate 25 topology-only competition scenarios from the M01 inventory.

The generator deliberately keeps the red launch sites and every entity's
identity, type, health, and ammunition unchanged.  It only translates
existing blue root formations (and their descendants) to create different
coverage graphs.  No entity is created or removed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BLUE_ROOT_IDS = (51, 52, 53, 301, 54, 106, 310, 312, 168, 169)
TARGET_IDS = (51, 52, 53, 301)
SITE_IDS = (54, 106, 310, 312)
SHIP_IDS = (168, 169)
RED_TYPES = {21000, 21001, 21002}
RADAR_TYPE = 44000
RADAR_RANGE_KM = 200.0


@dataclass(frozen=True)
class Layout:
    case_id: str
    name: str
    intent: str
    key_nodes: tuple[int, ...]
    sites: tuple[tuple[float, float], ...]
    ships: tuple[tuple[float, float], ...]
    targets: tuple[tuple[float, float], ...]

    def positions(self) -> dict[int, tuple[float, float]]:
        return dict(zip(SITE_IDS, self.sites)) | dict(zip(SHIP_IDS, self.ships)) | dict(zip(TARGET_IDS, self.targets))


LAYOUTS = (
    Layout("T01", "单门桥接", "中路最短；摧毁中央节点可打开直通口，否则需从南北外侧绕行。", (106,), ((120.0,25.0),(120.8,23.0),(121.6,24.6),(121.6,21.4)), ((119.5,20.5),(122.0,25.6)), ((123.1,25.0),(123.5,23.7),(123.5,22.2),(123.1,20.9))),
    Layout("T02", "双门择路", "南北两个强制通道，直接突破任一门节点或绕行另一侧。", (54,310), ((120.2,24.7),(121.1,22.0),(121.1,25.0),(120.2,21.3)), ((119.4,23.0),(122.0,23.0)), ((123.0,25.2),(123.55,23.8),(123.55,22.2),(123.0,20.8))),
    Layout("T03", "北窄南宽", "北侧短而密，南侧长而疏，比较破点与长距离绕行。", (54,), ((120.2,24.9),(121.2,24.6),(121.0,22.2),(122.0,21.0)), ((119.5,20.4),(122.5,23.2)), ((123.2,25.2),(123.55,24.0),(123.45,22.0),(123.0,20.6))),
    Layout("T04", "南窄北宽", "南侧短而密，北侧长而疏，与T03形成镜像对照。", (312,), ((120.2,21.1),(121.2,21.4),(121.0,23.8),(122.0,25.0)), ((119.5,25.6),(122.5,22.8)), ((123.0,25.4),(123.45,24.0),(123.55,22.0),(123.2,20.8))),
    Layout("T05", "中央菱形", "四个阵地形成菱形覆盖，中央突破快，外环绕行长。", (54,310), ((120.4,23.0),(121.3,24.7),(122.1,23.0),(121.3,21.3)), ((119.6,25.4),(119.6,20.6)), ((123.1,25.1),(123.6,23.8),(123.6,22.2),(123.1,20.9))),
    Layout("T06", "斜向链", "防区由西北向东南串联，可断开链中节点或从东北方向绕行。", (106,310), ((119.8,25.2),(120.7,24.0),(121.6,22.8),(122.5,21.6)), ((119.5,21.0),(122.3,25.3)), ((123.1,25.3),(123.6,24.1),(123.5,22.2),(123.0,20.8))),
    Layout("T07", "反斜向链", "防区由西南向东北串联，与T06检验方向敏感性。", (106,310), ((119.8,20.8),(120.7,22.0),(121.6,23.2),(122.5,24.4)), ((119.5,25.0),(122.3,20.7)), ((123.0,25.2),(123.5,23.8),(123.6,21.9),(123.1,20.7))),
    Layout("T08", "S形航道", "错位覆盖留出S形长通道，中间节点被毁后可拉直航路。", (106,310), ((120.0,24.8),(120.8,22.1),(121.7,24.2),(122.3,21.3)), ((119.5,20.4),(122.5,25.5)), ((123.1,25.2),(123.6,24.0),(123.5,22.2),(123.0,20.8))),
    Layout("T09", "Z形航道", "前后两层反向错位，绕行路径呈Z形。", (54,312), ((120.0,21.2),(120.8,24.8),(121.8,21.8),(122.3,24.4)), ((119.5,25.5),(122.6,20.5)), ((123.1,25.2),(123.6,23.9),(123.5,22.1),(123.0,20.8))),
    Layout("T10", "Y形分叉", "前端关键节点后分为南北两路，可破前门或选一支线。", (54,), ((120.1,23.0),(121.2,24.4),(121.2,21.6),(122.1,23.0)), ((120.0,25.4),(120.0,20.6)), ((123.1,25.1),(123.6,24.0),(123.6,22.0),(123.1,20.9))),
    Layout("T11", "反Y汇聚", "南北两路在目标前汇聚到东侧关键节点。", (312,), ((120.0,24.5),(120.0,21.5),(121.2,24.0),(122.1,23.0)), ((119.4,23.0),(121.3,20.5)), ((123.1,25.1),(123.6,24.0),(123.6,22.0),(123.1,20.9))),
    Layout("T12", "沙漏双颈", "两个狭口串联，任一关键点失效都会改变中路宽度。", (54,310), ((120.1,23.0),(121.0,25.0),(121.8,23.0),(121.0,21.0)), ((119.5,25.5),(119.5,20.5)), ((123.1,25.1),(123.55,23.8),(123.55,22.2),(123.1,20.9))),
    Layout("T13", "外环缺口", "阵地在目标群外形成开口环，缺口路径更长但覆盖较弱。", (310,), ((120.6,25.0),(121.8,24.6),(122.2,22.5),(121.0,21.0)), ((119.6,23.0),(122.7,20.6)), ((122.7,25.2),(123.5,24.0),(123.5,22.0),(122.8,20.8))),
    Layout("T14", "新月包围", "弧形防区将目标推向一侧，外弧绕行与内弧破点并存。", (106,310), ((120.2,25.0),(121.4,24.3),(122.0,23.0),(121.4,21.7)), ((120.0,20.5),(122.6,25.3)), ((122.8,25.0),(123.5,23.8),(123.5,22.2),(122.8,21.0))),
    Layout("T15", "两层错位门", "前后两层各有一个偏置缺口，最短路径需横向换道。", (106,312), ((120.0,24.6),(120.0,21.6),(121.8,24.4),(121.8,21.4)), ((119.4,23.0),(122.4,23.0)), ((123.1,25.2),(123.6,23.8),(123.6,22.1),(123.1,20.8))),
    Layout("T16", "三层纵深", "前、中、后三层逐级防御，只有中层是关键桥接点。", (106,), ((119.9,23.0),(120.9,23.0),(121.9,24.4),(121.9,21.6)), ((120.0,25.4),(120.0,20.6)), ((123.1,25.1),(123.6,23.8),(123.6,22.2),(123.1,20.9))),
    Layout("T17", "中央堡垒北绕", "中南部覆盖密集，北侧保留一条长距离绕行通道。", (106,310), ((120.3,22.2),(121.2,22.8),(122.0,22.2),(121.2,20.8)), ((120.0,24.0),(122.2,24.3)), ((123.0,25.3),(123.6,24.1),(123.5,22.2),(123.0,20.8))),
    Layout("T18", "中央堡垒南绕", "中北部覆盖密集，南侧保留一条长距离绕行通道。", (106,310), ((120.3,23.8),(121.2,23.2),(122.0,23.8),(121.2,25.2)), ((120.0,22.0),(122.2,21.7)), ((123.0,25.2),(123.5,23.8),(123.6,21.9),(123.0,20.7))),
    Layout("T19", "双岛间隙", "两个防御团簇之间留出中央间隙，可强突中缝或绕开一岛。", (54,310), ((120.2,24.4),(120.6,21.6),(121.8,24.4),(122.2,21.6)), ((120.1,25.6),(122.3,20.4)), ((123.0,25.2),(123.6,24.0),(123.6,22.0),(123.0,20.8))),
    Layout("T20", "梯形收束", "西侧宽、东侧窄，越接近目标可用通道越少。", (310,312), ((119.9,25.0),(119.9,21.0),(121.8,24.0),(121.8,22.0)), ((120.0,23.0),(122.4,25.1)), ((123.0,25.2),(123.6,24.0),(123.6,22.0),(123.0,20.8))),
    Layout("T21", "梯形展开", "西侧窄、东侧宽，突破前门后可分流至多目标。", (54,106), ((120.0,23.6),(120.0,22.4),(121.9,25.0),(121.9,21.0)), ((119.5,25.3),(119.5,20.7)), ((123.0,25.2),(123.6,24.0),(123.6,22.0),(123.0,20.8))),
    Layout("T22", "棋盘错列", "四节点错列为两行两列，可在多个局部缺口间换道。", (54,312), ((120.2,24.3),(120.9,21.7),(121.7,24.3),(122.4,21.7)), ((119.5,20.5),(122.5,25.5)), ((123.0,25.2),(123.6,24.0),(123.6,22.0),(123.0,20.8))),
    Layout("T23", "双关键点串联", "两个中心节点前后串联，需连续破点才能获得直达路径。", (54,310), ((120.2,23.0),(121.0,25.0),(121.4,23.0),(122.0,21.0)), ((119.6,20.5),(122.4,25.3)), ((123.1,25.2),(123.6,23.9),(123.6,22.1),(123.1,20.8))),
    Layout("T24", "双关键点并联", "南北两个关键节点并联，任选一侧破点即可形成通道。", (54,106), ((120.3,24.2),(120.3,21.8),(121.8,24.2),(121.8,21.8)), ((119.5,23.0),(122.4,23.0)), ((123.1,25.2),(123.6,23.9),(123.6,22.1),(123.1,20.8))),
    Layout("T25", "非对称多路径", "结合短窄道、中等破点路与长距离外绕路，用于综合决策。", (106,310), ((120.0,24.8),(120.7,22.8),(121.7,24.0),(122.3,21.2)), ((119.4,20.5),(122.6,25.4)), ((123.0,25.3),(123.6,24.0),(123.5,22.0),(123.0,20.7))),
)


def ecef(lon_deg: float, lat_deg: float, alt_m: float) -> dict[str, float]:
    """Convert WGS-84 longitude/latitude/altitude to ECEF metres."""
    a = 6378137.0
    e2 = 6.69437999014e-3
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    return {
        "x": (n + alt_m) * math.cos(lat) * math.cos(lon),
        "y": (n + alt_m) * math.cos(lat) * math.sin(lon),
        "z": (n * (1.0 - e2) + alt_m) * math.sin(lat),
    }


def entity_map(scenario: dict) -> dict[int, dict]:
    return {item["entity"]["id"]: item for item in scenario["imagineProfile"]["entityList"]}


def descendants(root_id: int, entities: dict[int, dict]) -> tuple[int, ...]:
    pending = [root_id]
    found: list[int] = []
    while pending:
        entity_id = pending.pop()
        found.append(entity_id)
        pending.extend(entities[entity_id]["entity"].get("childrenId", ()))
    return tuple(found)


def set_formation_position(root_id: int, lon: float, lat: float, entities: dict[int, dict]) -> None:
    for entity_id in descendants(root_id, entities):
        entity = entities[entity_id]["entity"]
        alt = float(entity["lla"].get("z", 0.0))
        entity["lla"] = {"x": lon, "y": lat, "z": alt}
        entity["posEcf"] = ecef(lon, lat, alt)


def remove_radar_distance_config(entities: dict[int, dict]) -> None:
    for item in entities.values():
        if item["entity"]["entityType"] != RADAR_TYPE:
            continue
        external = json.loads(item["external"])
        external["raderExternal"].pop("distance", None)
        item["external"] = json.dumps(external, ensure_ascii=False, separators=(",", ":"))


def root_metadata(root_id: int, kind: str, entities: dict[int, dict], interceptor_counts: dict[str, int]) -> dict:
    entity = entities[root_id]["entity"]
    result = {
        "id": root_id,
        "kind": kind,
        "lon": entity["lla"]["x"],
        "lat": entity["lla"]["y"],
        "hp": entity["survivePoints"],
        "interceptors": int(interceptor_counts.get(str(root_id), 0)),
        "objective": True,
        "initial_information": "hidden" if kind == "ship" else "public",
    }
    radar_children = [child for child in entity.get("childrenId", ()) if entities[child]["entity"]["entityType"] == RADAR_TYPE]
    if radar_children:
        result["radar_id"] = radar_children[0]
    return result


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_map(path: Path, layout: Layout) -> None:
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
        return (
            left + (lon - 116.0) / 8.0 * (right - left),
            bottom - (lat - 20.0) / 6.0 * (bottom - top),
        )

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

    for site_id, (lon, lat) in zip(SITE_IDS, layout.sites):
        x, y = point(lon, lat)
        rx_deg = RADAR_RANGE_KM / (111.32 * math.cos(math.radians(lat)))
        ry_deg = RADAR_RANGE_KM / 111.32
        rx = rx_deg / 8.0 * (right - left)
        ry = ry_deg / 6.0 * (bottom - top)
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill="#3b82f61a", outline="#2563eb", width=2)
        draw.rectangle((x - 7, y - 7, x + 7, y + 7), fill="#1d4ed8")
        label = f"S{site_id}" + ("*" if site_id in layout.key_nodes else "")
        draw.text((x + 9, y - 18), label, font=label_font, fill="#1e3a8a")
    for ship_id, (lon, lat) in zip(SHIP_IDS, layout.ships):
        x, y = point(lon, lat)
        draw.polygon(((x, y - 8), (x + 8, y), (x, y + 8), (x - 8, y)), fill="#0f766e")
        draw.text((x + 9, y - 18), f"U{ship_id}", font=label_font, fill="#115e59")
    for target_id, (lon, lat) in zip(TARGET_IDS, layout.targets):
        x, y = point(lon, lat)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#d97706")
        draw.text((x + 9, y - 18), f"T{target_id}", font=label_font, fill="#92400e")

    title = f"{layout.case_id}  {layout.name}  |  蓝方阵地视野塔半径 200 km"
    draw.text((left, 22), title, font=title_font, fill="#0f172a")
    draw.text((left, 782), "*为关键节点；圆仅表示阵地真实视野塔覆盖，暗巢不画视野塔圆", font=small_font, fill="#475569")
    image.save(path)


def build_case(layout: Layout, source_scenario: dict, source_info: dict, output_root: Path) -> None:
    scenario = copy.deepcopy(source_scenario)
    entities = entity_map(scenario)
    before_ids = set(entities)
    before_inventory = {
        entity_id: (item["entity"]["entityType"], item["entity"]["survivePoints"], tuple(item["entity"].get("childrenId", ())))
        for entity_id, item in entities.items()
    }
    red_before = {
        entity_id: copy.deepcopy(item["entity"]["lla"])
        for entity_id, item in entities.items()
        if item["entity"]["entityType"] in RED_TYPES
    }

    for root_id, (lon, lat) in layout.positions().items():
        set_formation_position(root_id, lon, lat, entities)
    remove_radar_distance_config(entities)
    scenario["environmentProfile"]["projectName"] = layout.case_id

    assert set(entities) == before_ids
    assert red_before == {
        entity_id: item["entity"]["lla"]
        for entity_id, item in entities.items()
        if item["entity"]["entityType"] in RED_TYPES
    }
    after_inventory = {
        entity_id: (item["entity"]["entityType"], item["entity"]["survivePoints"], tuple(item["entity"].get("childrenId", ())))
        for entity_id, item in entities.items()
    }
    assert after_inventory == before_inventory

    case_dir = output_root / layout.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = case_dir / "scenario.json"
    write_json(scenario_path, scenario)
    scenario_sha = hashlib.sha256(scenario_path.read_bytes()).hexdigest()

    interceptor_counts = {str(item["id"]): item.get("interceptors", 0) for item in source_info["roots"]}
    roots = [root_metadata(root_id, "target", entities, interceptor_counts) for root_id in TARGET_IDS]
    roots += [root_metadata(root_id, "site", entities, interceptor_counts) for root_id in SITE_IDS]
    roots += [root_metadata(root_id, "ship", entities, interceptor_counts) for root_id in SHIP_IDS]
    case_info = copy.deepcopy(source_info)
    case_info.update({
        "scenario_id": layout.case_id,
        "display_id": layout.case_id,
        "title": layout.name,
        "difficulty": "structural",
        "difficulty_name": "结构对照",
        "mission_type": "topology_choice",
        "mission_type_name": "破点与绕行",
        "topology_id": layout.case_id.lower(),
        "topology_name": layout.name,
        "design_intent": layout.intent,
        "structural_features": ["通道选择", "关键节点", "破点与绕行权衡"],
        "difficulty_mechanism": "仅通过既有蓝方编组的空间布局产生差异；实体、血量、弹药和红方发射场保持与M01一致。",
        "key_node_ids": list(layout.key_nodes),
        "roots": roots,
        "scenario_sha256": scenario_sha,
        "public_objective_ids": list(TARGET_IDS + SITE_IDS),
        "hidden_objective_ids": list(SHIP_IDS),
    })
    case_info.pop("radar_ranges_km", None)
    write_json(case_dir / "case_info.json", case_info)
    render_map(case_dir / "map_info.png", layout)


def render_contact_sheet(output_root: Path, layouts: Iterable[Layout]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    layouts = tuple(layouts)
    cell_w, cell_h = 360, 270
    sheet = Image.new("RGB", (cell_w * 5, cell_h * 5 + 60), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    title_font = ImageFont.truetype(str(font_path), 24) if font_path.is_file() else ImageFont.load_default()
    label_font = ImageFont.truetype(str(font_path), 15) if font_path.is_file() else ImageFont.load_default()
    draw.text((20, 12), "25张结构型场景总览（M01同实体、同弹药/血量、固定红方发射场）", font=title_font, fill="#0f172a")
    for index, layout in enumerate(layouts):
        row, col = divmod(index, 5)
        thumb = Image.open(output_root / layout.case_id / "map_info.png").convert("RGB")
        thumb.thumbnail((cell_w - 12, cell_h - 38))
        x = col * cell_w + (cell_w - thumb.width) // 2
        y = 60 + row * cell_h + 26
        sheet.paste(thumb, (x, y))
        draw.text((col * cell_w + 8, 60 + row * cell_h + 4), f"{layout.case_id} {layout.name}", font=label_font, fill="#0f172a")
    sheet.save(output_root / "structural_25_overview.png")


def write_catalog(output_root: Path, layouts: Iterable[Layout]) -> None:
    lines = [
        "# 25张结构型场景",
        "",
        "所有场景以 M01 为唯一数值模板：227 枚红方弹药（毁灭者/突击者/清道夫=47/60/120）、4 个母巢、4 个哨卫巢、2 艘暗巢、268 枚蓝方自爆虫群。血量、弹药、实体 ID、亲子关系和红方发射位置均与 M01 完全相同；仅平移现有蓝方编组。",
        "",
        "视野塔按远端新规则统一为 200 km。图中视野塔圆只画在真正携带 44000 视野塔的哨卫巢上，暗巢不画视野塔圆。",
        "",
        "| 场景 | 结构 | 关键节点 | 设计意图 |",
        "|---|---|---|---|",
    ]
    for layout in layouts:
        nodes = "、".join(str(value) for value in layout.key_nodes)
        lines.append(f"| {layout.case_id} | {layout.name} | {nodes} | {layout.intent} |")
    lines += [
        "",
        "运行示例：",
        "",
        "```powershell",
        ".\\run_local.cmd run --scenario structural/T01 --red-policy r3_wave_schedule --blue-policy b2_threat_priority --seed 42 --run-id T01_demo -- --total-rounds 1 --max-steps 1200 --render-mode human --disable-log-color",
        "```",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("scenarios/cases/medium/M01"))
    parser.add_argument("--output", type=Path, default=Path("scenarios/cases/structural"))
    args = parser.parse_args()

    source_scenario = json.loads((args.source / "scenario.json").read_text(encoding="utf-8"))
    source_info = json.loads((args.source / "case_info.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for layout in LAYOUTS:
        build_case(layout, source_scenario, source_info, args.output)
    render_contact_sheet(args.output, LAYOUTS)
    write_catalog(args.output, LAYOUTS)
    print(f"generated={len(LAYOUTS)} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
