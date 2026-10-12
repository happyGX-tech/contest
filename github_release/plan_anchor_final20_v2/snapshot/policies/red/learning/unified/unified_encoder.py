"""统一 POMDP 观测编码器。

局部观测布局（Actor 可见，全部来自隔离观测 + InfoCenter 合法共享）：
  o_i^t = [phase, self, comm/context, goal_slots × 5, id]

- phase: [waiting, flying, dead] one-hot（dead 智能体不产生观测）
- self:  步数/经纬高/阶段/血量/弹型 one-hot/机动状态/卫星已用/
         发射步/【速度估计】/【sin·cos 航向估计】
         （观测无速度字段，速度由智能体用相邻两步 pos_ecf 差分得到）
- comm:  压力/发射比例/拦截弹观测/通信群/卫星预算(含上步请求结算反馈)/剩余时间
- goal_slots: 官方规定的 5 个航迹槽。先按弹种过滤命中效能<=0 的目标，
  再按【与时间无关】的固定键（价值×效能↓, id↑）排序——槽位身份跨时间
  稳定，距离仅作为槽位特征；"是否当前目标"作为特征。
- id:    智能体序号归一化

部署期观测：同样维度的向量（episode 的第一个决策 o_0^deploy），
self 填弹型/部署区质心/指向目标质心的航向，槽位填初始目录。
"""

from __future__ import annotations

import numpy as np

from .info_center import InfoCenter

# 地图归一化常数（九图通用近似范围）
LON_MIN, LON_MAX = 115.0, 125.0
LAT_MIN, LAT_MAX = 20.0, 28.0
ALT_MAX = 40000.0
DIST_MAX_KM = 900.0

GOAL_SLOTS = 7                 # 5 战前目录 + 2 发现的 9500 槽位
GOAL_FEATURES = 7
SECTOR_ACTIONS = 2             # 搜索扇区动作数（仅 L 合法）
SECTOR_FEATURES = 4            # 每扇区 [valid, dlon, dlat, dist]

PHASE_FEATURES = 3             # waiting / flying / dead
SELF_FEATURES = 15             # 见 encode_self（含速度估计 3 维）
COMM_FEATURES = 7
ID_FEATURES = 1
UNIFIED_OBS_DIM = (
    PHASE_FEATURES + SELF_FEATURES + COMM_FEATURES
    + GOAL_SLOTS * GOAL_FEATURES
    + SECTOR_ACTIONS * SECTOR_FEATURES
    + ID_FEATURES
)

DEPLOY_FEATURE_DIM = 10

# 目标类型编码
TYPE_CODE = {9400: 0.6, 9500: 0.2, 9600: 0.4}

# 弹种×目标类型 命中效能（与引擎毁伤表一致）：用于弹种感知的槽位排序，
# 保证 5 槽内各类弹都能看到自己"打得动"的目标（H/M 看不到船是因为命中率为 0，
# L 的槽位里船优先——这是 9500 在 5 槽约束下可达的关键）
ENGAGEMENT_EFFICIENCY = {
    21000: {9400: 0.8, 9600: 0.6, 9500: 0.0},
    21001: {9400: 0.8, 9600: 0.6, 9500: 0.0},
    21002: {9400: 0.05, 9600: 0.05, 9500: 0.8},
}


def _norm(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), -1.0, 1.0))


def engagement_efficiency(missile_type: int, entity_type: int) -> float:
    """弹种×目标类型 命中效能（合法领域先验，与引擎毁伤表一致）。"""
    return ENGAGEMENT_EFFICIENCY.get(
        int(missile_type), ENGAGEMENT_EFFICIENCY[21000]
    ).get(int(entity_type), 0.0)


def eligible_targets(targets, missile_type: int) -> list:
    """过滤命中效能<=0 的目标：H/M 永不把 9500 补入槽位（打不动且挤占陆上目标），
    L 的槽位由效能排序自然以 9500 优先。"""
    return [
        t for t in targets
        if engagement_efficiency(missile_type, t.entity_type) > 0.0
    ]


def target_sort_key(target, missile_type: int = 21000):
    """持久稳定排序键：(-价值×命中效能, entity_id)。

    距离只作为槽位特征不进排序键——键中任何随时间变化的量都会破坏
    槽位身份的跨时间稳定性。槽位集合只因合法事件（发现/判死）变化。
    """
    efficiency = engagement_efficiency(missile_type, target.entity_type)
    return (-float(target.value) * efficiency, target.entity_id)


class UnifiedObsEncoder:
    """把隔离观测 + 共享信息编码成固定维向量（参数免费，纯特征工程）。"""

    def __init__(self, max_steps: int = 1200, agent_index: int = 0, missile_type: int = 21000):
        self.max_steps = max(int(max_steps), 1)
        self.agent_index = float(agent_index)
        self.missile_type = int(missile_type)

    # ---------- 主编码 ----------

    def encode(
        self,
        observation: dict,
        *,
        flying: bool,
        launch_step: int,
        maneuver_state: int,
        sat_used: bool,
        current_goal_id: int | None,
        info_center: InfoCenter,
        speed_estimate: float = 0.0,
        heading_sin: float = 0.0,
        heading_cos: float = 0.0,
    ) -> np.ndarray:
        features = np.zeros(UNIFIED_OBS_DIM, dtype=np.float32)
        entity = observation.get("self") or {}
        step = float(observation.get("step", 0))

        # phase one-hot
        features[1 if flying else 0] = 1.0

        # self
        base = PHASE_FEATURES
        position = entity.get("position") or {}
        features[base + 0] = _norm(step, 0.0, self.max_steps)
        features[base + 1] = _norm(float(position.get("lon", LON_MIN)), LON_MIN, LON_MAX)
        features[base + 2] = _norm(float(position.get("lat", LAT_MIN)), LAT_MIN, LAT_MAX)
        features[base + 3] = _norm(float(position.get("alt", 0.0)), 0.0, ALT_MAX)
        features[base + 4] = float(entity.get("stage", 0)) / 5.0
        missile_type = int(entity.get("type", self.missile_type))
        features[base + 6 + (0 if missile_type == 21000 else 1 if missile_type == 21001 else 2)] = 1.0
        features[base + 9] = float(maneuver_state)
        features[base + 10] = 1.0 if sat_used else 0.0
        features[base + 11] = _norm(float(launch_step if launch_step >= 0 else 0), 0.0, self.max_steps)
        # 速度估计（观测无速度字段，由智能体 pos_ecf 差分提供）
        features[base + 12] = float(np.clip(speed_estimate / 2500.0, 0.0, 1.0))
        features[base + 13] = float(heading_sin)
        features[base + 14] = float(heading_cos)
        # 血量按弹种最大生命归一（21001 原始值 1500 会饱和 Tanh）：
        # 红弹实际近似二元存活（血量 1、拦截伤害 5000），故归一为存活标志
        features[base + 5] = 1.0 if float(entity.get("health", 1.0)) > 0.0 else 0.0

        # comm / context
        base = PHASE_FEATURES + SELF_FEATURES
        features[base + 0] = float(np.clip(info_center.pressure, 0.0, 1.0))
        features[base + 1] = float(np.clip(info_center.launched_fraction, 0.0, 1.0))
        features[base + 2] = float(np.clip(info_center.interceptors_observed / 20.0, 0.0, 1.0))
        features[base + 3] = float(np.clip(info_center.comm_group_total / 50.0, 0.0, 1.0))
        # 卫星预算（含上一步请求被拒/批准的隐式反馈：预算变化即可感知）
        features[base + 4] = info_center.sat_budget_remaining / float(InfoCenter.SAT_BUDGET_TOTAL)
        features[base + 5] = 1.0 - _norm(step, 0.0, self.max_steps)
        features[base + 6] = float(len(info_center.known_ships)) / 6.0

        # goal slots（持久稳定排序：效能过滤 + 固定键，槽位身份跨时间稳定）
        base = PHASE_FEATURES + SELF_FEATURES + COMM_FEATURES
        own_lon = float(position.get("lon", LON_MIN))
        own_lat = float(position.get("lat", LAT_MIN))
        targets = sorted(
            eligible_targets(info_center.targets(), missile_type),
            key=lambda t: target_sort_key(t, missile_type),
        )
        for slot, target in enumerate(targets[:GOAL_SLOTS]):
            offset = base + slot * GOAL_FEATURES
            features[offset + 0] = _norm(target.position.lon - own_lon, -5.0, 5.0)
            features[offset + 1] = _norm(target.position.lat - own_lat, -4.0, 4.0)
            dist = np.hypot(
                (target.position.lon - own_lon) * 96.0, (target.position.lat - own_lat) * 110.0
            )
            features[offset + 2] = float(np.clip(dist / DIST_MAX_KM, 0.0, 1.0))
            features[offset + 3] = TYPE_CODE.get(target.entity_type, 0.0)
            features[offset + 4] = 1.0 if target.entity_id == current_goal_id else 0.0
            features[offset + 5] = (
                1.0 if target.entity_type == 9500 and target.entity_id in info_center.known_ships
                else 0.0
            )
            features[offset + 6] = float(np.clip(target.value / 10.0, 0.0, 1.0))

        # search sectors（发现 9500 前的合法搜索区域，坐标由 InfoCenter 派生）
        base = (
            PHASE_FEATURES + SELF_FEATURES + COMM_FEATURES
            + GOAL_SLOTS * GOAL_FEATURES
        )
        sectors = info_center.search_sectors()
        for s, (sec_lon, sec_lat) in enumerate(sectors[:SECTOR_ACTIONS]):
            offset = base + s * SECTOR_FEATURES
            features[offset + 0] = 1.0
            features[offset + 1] = _norm(sec_lon - own_lon, -5.0, 5.0)
            features[offset + 2] = _norm(sec_lat - own_lat, -4.0, 4.0)
            dist = np.hypot(
                (sec_lon - own_lon) * 96.0, (sec_lat - own_lat) * 110.0
            )
            features[offset + 3] = float(np.clip(dist / DIST_MAX_KM, 0.0, 1.0))

        # id
        features[-1] = float(np.clip(self.agent_index / 200.0, 0.0, 1.0))
        return features

    # ---------- 部署期观测（episode 首决策 o_0^deploy） ----------

    def encode_deploy(
        self,
        missile_type: int,
        agent_index: int,
        zone_bbox: tuple[float, float, float, float],
        initial_targets,
    ) -> np.ndarray:
        min_lon, max_lon, min_lat, max_lat = zone_bbox
        zone_lon = 0.5 * (min_lon + max_lon)
        zone_lat = 0.5 * (min_lat + max_lat)
        features = np.zeros(UNIFIED_OBS_DIM, dtype=np.float32)
        features[0] = 1.0  # phase = waiting

        base = PHASE_FEATURES
        features[base + 1] = _norm(zone_lon, LON_MIN, LON_MAX)
        features[base + 2] = _norm(zone_lat, LAT_MIN, LAT_MAX)
        features[base + 5] = 1.0  # health
        features[base + 6 + (0 if missile_type == 21000 else 1 if missile_type == 21001 else 2)] = 1.0

        base = PHASE_FEATURES + SELF_FEATURES
        features[base + 4] = 1.0  # 卫星预算满
        features[base + 5] = 1.0  # 剩余时间满

        base = PHASE_FEATURES + SELF_FEATURES + COMM_FEATURES
        targets = sorted(
            eligible_targets(initial_targets, missile_type),
            key=lambda t: target_sort_key(t, missile_type),
        )
        for slot, target in enumerate(list(targets)[:GOAL_SLOTS]):
            offset = base + slot * GOAL_FEATURES
            features[offset + 0] = _norm(target.position.lon - zone_lon, -5.0, 5.0)
            features[offset + 1] = _norm(target.position.lat - zone_lat, -4.0, 4.0)
            dist = np.hypot(
                (target.position.lon - zone_lon) * 96.0,
                (target.position.lat - zone_lat) * 110.0,
            )
            features[offset + 2] = float(np.clip(dist / DIST_MAX_KM, 0.0, 1.0))
            features[offset + 3] = TYPE_CODE.get(target.entity_type, 0.0)
            features[offset + 6] = float(np.clip(target.value / 10.0, 0.0, 1.0))

        features[-1] = float(np.clip(agent_index / 200.0, 0.0, 1.0))
        return features

    # ---------- 部署头特征（部署网络的输入） ----------

    @staticmethod
    def build_deploy_features(
        entity_index: int,
        total_entities: int,
        missile_type: int,
        zone_bbox: tuple[float, float, float, float],
        centroid_lon: float,
        centroid_lat: float,
    ) -> np.ndarray:
        """部署期特征：弹型 / 序号 / 部署区 bbox / 目标质心方向。

        zone_bbox: (min_lon, max_lon, min_lat, max_lat)
        """
        min_lon, max_lon, min_lat, max_lat = zone_bbox
        zone_lon = 0.5 * (min_lon + max_lon)
        zone_lat = 0.5 * (min_lat + max_lat)
        features = np.zeros(DEPLOY_FEATURE_DIM, dtype=np.float32)
        features[0 if missile_type == 21000 else 1 if missile_type == 21001 else 2] = 1.0
        features[3] = float(np.clip(entity_index / max(total_entities, 1), 0.0, 1.0))
        features[4] = _norm(min_lon, LON_MIN, LON_MAX)
        features[5] = _norm(max_lon, LON_MIN, LON_MAX)
        features[6] = _norm(min_lat, LAT_MIN, LAT_MAX)
        features[7] = _norm(max_lat, LAT_MIN, LAT_MAX)
        features[8] = _norm(centroid_lon - zone_lon, -8.0, 8.0)
        features[9] = _norm(centroid_lat - zone_lat, -8.0, 8.0)
        return features
