"""红方合法信息中心：共享航迹表 + 全局战术统计。

只从两类合法来源构建信息（执行期 Actor 可见的 c_i^t）：
1. 战前初始目录（init_observation_ship 中的 9400/9600）；
2. 各平台隔离观测 detectInfo 的航迹上报（含 9500 无人船的合法发现）。

引擎全局真值（双方血量等）只允许进入集中式 Critic 的全局态编码，
不得进入本中心，保证 POMDP 部分可观测语义。
"""

from __future__ import annotations

import math
import os

from policies.red.baselines import TargetPrior
from policies.red.tracks import InitialCatalogueTrackFusion


def infer_coordinate_scale(initial_targets: tuple[TargetPrior, ...]) -> float:
    """Return the simulator's physical-coordinate multiplier.

    The native models expose final20 longitude/latitude divided by
    ``ISimulator.MODEL_SCALE`` (10), while older scenarios use ordinary degree
    coordinates.  Detecting the representation once keeps policy geometry in
    the canonical 115--129 / 15--29 degree frame.
    """

    longitudes = [abs(float(target.position.lon)) for target in initial_targets]

    return 10.0 if longitudes and max(longitudes) < 30.0 else 1.0


class InfoCenter:
    """全体红弹共享的航迹融合与战术态势（每个回合重建）。"""

    SAT_BUDGET_TOTAL = 100
    THREAT_FRESH_STEPS = 30
    # 与 TopFormer 归一化区间一致的公开固定边界。仿真器若使用
    # MODEL_SCALE=10，则通过 coordinate_scale 还原到当前观测的坐标表示。
    SEARCH_FALLBACK_CANONICAL_BOUNDS = (115.0, 129.0, 15.0, 29.0)
    SEARCH_FALLBACK_PADDING_DEG = 1.5
    L_SCOUT_DISABLED = "disabled"
    L_SCOUT_SCOUTING = "scouting"
    L_SCOUT_COMPLETE = "complete"
    L_SCOUT_CONFIDENT_STOP = "confident_stop"
    L_SCOUT_DEADLINE_STOP = "deadline_stop"
    # L 的三维探测半径约 3km；典型 2km 高度下理论水平半径约
    # sqrt(3^2-2^2)=2.24km。这里用 2km 的保守半径生成确定性扫掠线，
    # 不读取任何隐藏目标位置。
    L_SCOUT_HORIZONTAL_COVERAGE_KM = 2.0
    L_SCOUT_WAYPOINT_SWITCH_KM = 1.0

    # 取赤道每经纬度上界来布线，避免用经验 96/110km 换算低估实际
    # 物理间距而在扫掠线之间留下窄缝。
    DEGREE_KM_UPPER_BOUND = 111.32

    def __init__(
        self,
        initial_targets: tuple[TargetPrior, ...],
        total_platforms: int,
        coordinate_scale: float | None = None,
        search_bounds: tuple[float, float, float, float] | None = None,
        search_rows: int = 4,
        search_cols: int = 8,
        l_scout_enabled: bool = False,
        l_scout_count: int = 0,
        l_scout_expected_targets: int = 0,
        l_scout_coverage_threshold: float = 1.0,
        l_scout_quiet_steps: int = 0,
        l_scout_deadline: int = 0,
    ):
        self.fusion = InitialCatalogueTrackFusion(initial_targets)
        self.coordinate_scale = float(
            coordinate_scale
            if coordinate_scale is not None
            else infer_coordinate_scale(initial_targets)
        )
        self.total_platforms = max(int(total_platforms), 1)
        self.search_rows = int(search_rows)
        self.search_cols = int(search_cols)
        if self.search_rows <= 0 or self.search_cols <= 0:
            raise ValueError("search_rows and search_cols must be positive")
        self.search_bounds = self._resolve_search_bounds(
            tuple(initial_targets), search_bounds
        )

        self._search_sector_centers = self._build_search_sector_centers()
        # 搜索覆盖只由已发射 L 平台的合法 self.position 轨迹生成。
        self.covered_search_cells: set[int] = set()
        self.search_cell_visits: dict[int, int] = {}
        self.search_cell_last_step: dict[int, int] = {}
        self.step_coverage_by_entity: dict[int, set[int]] = {}
        # P1 少量 L 侦察状态机。默认关闭，因此旧策略不会受到搜索
        # 格排他或截止条件的影响。侦察 ID 在 configure_l_scouts() 首次
        # 调用时冻结，避免 MAT 决策顺序改变侦察者集合。
        self.l_scout_enabled = bool(l_scout_enabled)
        self.l_scout_count = int(l_scout_count)
        self.l_scout_expected_targets = int(l_scout_expected_targets)
        self.l_scout_coverage_threshold = float(l_scout_coverage_threshold)
        self.l_scout_quiet_steps = int(l_scout_quiet_steps)
        self.l_scout_deadline = int(l_scout_deadline)
        self._validate_l_scout_config()
        self.l_scout_state = (
            self.L_SCOUT_SCOUTING
            if self.l_scout_enabled
            else self.L_SCOUT_DISABLED
        )
        self.l_scout_ids: tuple[int, ...] = ()
        self._l_scout_pool_ids: tuple[int, ...] = ()
        self._l_scout_history: list[int] = []
        self._l_scout_replacement_events: list[dict] = []
        self._l_scout_replacement_attempts = 0
        self._l_scout_replacement_count = 0
        self._l_scout_replacement_exhausted = 0
        self._l_scouts_configured = False
        self._l_scout_assignments: dict[int, int] = {}
        self._l_scout_cell_owners: dict[int, int] = {}
        self._l_scout_last_discovery_step: int | None = None
        self._l_scout_stop_step: int | None = None
        self._l_scout_assignments_total = 0
        self._l_scout_assignment_conflicts = 0
        self._l_scout_covered_rejections = 0
        # 目标航迹时间与威胁航迹同样只来自隔离 detectInfo。
        self.target_last_seen: dict[int, int] = {
            int(target.entity_id): 0 for target in initial_targets
        }
        self._target_track_times: dict[int, int] = {}
        self.threat_tracks: dict[int, dict] = {}
        self._latest_report_step = 0
        # 红方统计（由各智能体上报聚合）
        self.launched_ids: set[int] = set()
        self._pending_launched: set[int] = set()
        self.alive_ids: set[int] = set()
        self.lost_ids: set[int] = set()
        self.interceptors_observed: int = 0
        self.comm_group_total: int = 0
        self.sat_used_count: int = 0
        self.satellite_active: bool = False
        # 本步事件
        self.step_discoveries: list[int] = []
        self.step_discoverers: set[int] = set()   # 本步发现 9500 的实体（发现者专属奖励）
        # Episode-persistent legal discovery ledger for P0 evaluation.  Rows
        # are appended only after fusion accepts a 9500 track, so this never
        # exposes hidden truth to the actor or to telemetry before discovery.
        self.discovery_events: list[dict] = []
        self.step_destroyed: list[int] = []   # 本步判死事件（belief 或 oracle）
        # 饱和掩码记账：target_id -> {entity_id: 期望毁伤承诺}。
        # 发射/换靶时登记、死亡/离开时释放——拦截损失自然回补；
        # 同一决策步内同步生效（先到先得），形成跨弹的均衡分配。
        self.inflight_commit: dict[int, dict[int, float]] = {}
        # BDA 状态：合法 belief（投送推算）+ oracle（全局真值，默认关闭）
        self.expected_damage: dict[int, float] = {
            t.entity_id: 0.0 for t in initial_targets
        }
        self.initial_health: dict[int, float] = {
            t.entity_id: float(t.initial_health) for t in initial_targets
        }
        self._belief_dead: set[int] = set()
        self._oracle_dead: set[int] = set()
        self._known_ships: set[int] = {
            t.entity_id for t in initial_targets if t.entity_type == 9500
        }

    # ---------- 回合内信息流 ----------

    def ingest_report(self, observation: dict) -> None:
        """吸收单枚导弹的隔离观测：航迹、搜索覆盖与威胁融合。"""
        entity = observation.get("self") or {}
        step = int(observation.get("step", self._latest_report_step))
        self._latest_report_step = max(self._latest_report_step, step)
        entity_id = int(observation.get("entity_id", -1))
        self.satellite_active = bool(
            self.satellite_active or observation.get("is_using_satellite", False)
        )
        detect = entity.get("detectInfo") or {}
        detect_track_by_id = {}
        for raw_id, track in detect.items():
            entity_type = int(self._field(track, "entity_type", -1))
            track_id = int(self._field(track, "entity_id", raw_id))
            detect_track_by_id[track_id] = track
            if entity_type == 24000:
                self._ingest_threat_track(track_id, track, step)
        self._prune_threat_tracks(step)
        self.interceptors_observed = len(self.threat_tracks)
        comm = entity.get("commRangeInfo") or {}
        self.comm_group_total = max(self.comm_group_total, len(comm))
        before = {t.entity_id for t in self.fusion.targets}
        changed = self.fusion.ingest(observation)
        known_after = {t.entity_id for t in self.fusion.targets}
        for raw_id, track in detect.items():
            entity_type = int(self._field(track, "entity_type", -1))
            if entity_type not in (9400, 9500, 9600):
                continue
            track_id = int(self._field(track, "entity_id", raw_id))
            # 无效的隐藏航迹不得单独把 ID 注入可见状态。
            if track_id not in known_after:
                continue
            raw_time = self._int_field(track, "time", step)
            previous_time = self._target_track_times.get(track_id)
            if previous_time is not None and raw_time <= previous_time:
                continue
            self._target_track_times[track_id] = raw_time
            seen_step = self._track_step(raw_time, step)
            self.target_last_seen[track_id] = max(
                self.target_last_seen.get(track_id, 0), seen_step
            )
        if changed:
            for target in self.fusion.targets:
                if target.entity_id not in before and target.entity_type == 9500:
                    self._known_ships.add(target.entity_id)
                    self.step_discoveries.append(target.entity_id)
                    self._l_scout_last_discovery_step = step
                    # 发现归因必须来自原始传感器字段 detect_from；共享航迹
                    # 可能由另一枚弹转发，因此不能把当前 reporter 当发现者。
                    # 老场景/合成测试缺字段时才回退 reporter。
                    discovery_track = detect_track_by_id.get(target.entity_id)
                    discoverer_id = self._int_field(
                        discovery_track, "detect_from", -1
                    )
                    if discoverer_id < 0:
                        discoverer_id = int(observation.get("entity_id", -1))
                    self.step_discoverers.add(discoverer_id)
                    self.discovery_events.append({
                        "step": step,
                        "target_id": int(target.entity_id),
                        "discoverer_entity_id": discoverer_id,
                        # 转发航迹 reporter 可能远离原始传感器，且本报告中
                        # 没有 detect_from 的当前位置。使用刚合法进入融合表的
                        # 目标坐标定位发现宏格，绝不使用 reporter 航迹。
                        "search_cell": self._search_cell_index(
                            target.position.lon, target.position.lat
                        ),
                        "target_lon": float(target.position.lon),
                        "target_lat": float(target.position.lat),
                    })
                    # 发现即初始化 BDA 账本：血量先验 + 期望毁伤从 0 起算，
                    # 否则发现的船永远无法 belief 判死（expected_damage 查无键）
                    self.initial_health.setdefault(
                        target.entity_id, self.SHIP_HEALTH_PRIOR
                    )
                    self.expected_damage.setdefault(target.entity_id, 0.0)
            self.update_l_scout_phase(step)
        if (
            entity_id >= 0
            and entity_id in self.launched_ids
            and int(entity.get("type", -1)) == 21002
        ):
            position = entity.get("position") or {}
            self.record_search_visit(
                entity_id,
                position.get("lon"),
                position.get("lat"),
                step,
            )

    def mark_launched(self, entity_id: int) -> None:
        """登记发射（延迟应用：决策期内不改共享统计，避免智能体顺序依赖）。"""
        self._pending_launched.add(int(entity_id))

    def apply_pending(self) -> None:
        """环境步开始时应用延迟的发射登记。"""
        self.launched_ids.update(self._pending_launched)
        self._pending_launched.clear()

    def mark_alive(self, entity_id: int) -> None:
        self.alive_ids.add(int(entity_id))
        self.lost_ids.discard(int(entity_id))

    def mark_lost(self, entity_id: int) -> None:
        entity_id = int(entity_id)
        self.alive_ids.discard(entity_id)
        self.lost_ids.add(entity_id)
        self.release_search_assignment(entity_id)
        if entity_id in self.l_scout_ids:
            self._replace_lost_scout(entity_id, self._latest_report_step)

    def mark_satellite_used(self) -> None:
        self.sat_used_count += 1

    @property
    def sat_budget_remaining(self) -> int:
        return max(0, self.SAT_BUDGET_TOTAL - self.sat_used_count)

    def begin_step(self, step: int | None = None) -> None:
        """每仿真步开始：应用延迟登记、更新 P1 状态并清瞬时统计。

        ``step`` 为可选参数，以保持旧调用方式 ``begin_step()`` 兼容。
        未传入时使用最近一次合法观测的回合内步数。
        """
        if step is not None:
            self._latest_report_step = max(self._latest_report_step, int(step))
        self.update_l_scout_phase(self._latest_report_step)
        self.apply_pending()
        self.step_discoveries.clear()
        self.step_discoverers.clear()
        self.step_destroyed.clear()
        self.interceptors_observed = 0
        self.comm_group_total = 0
        self.satellite_active = False
        self.step_coverage_by_entity.clear()

    # ---------- 饱和掩码（在飞期望毁伤承诺记账） ----------

    # 饱和判定边际：累计在飞承诺 ≥ 血量 × 该值 → 对后到者掩码。
    # 边际即"预备队倍率"：b0 拦截损耗实测 60-70%，边际必须显著 >1
    # （边际 1.0 时每目标只放行最小杀伤链，承诺弹被拦截后无人生还到达，
    # 阵地全程零伤害）。可用环境变量 SATURATION_MARGIN 按图调整。
    SATURATION_MARGIN = float(os.getenv("SATURATION_MARGIN", "3.0"))
    # 分类型边际（final20 兵力算术校准）：9600 巢 hp18 两发 H 即毁，
    # 旧全局 3.0 需 4.5H 等效承诺才隐身 → 全图承诺需求超出现有火力，
    # 结构性"单点过杀 + 其他饥饿"（20 回合实测过杀指派与分数相关 -0.54）。
    # 9400 母巢 hp24 用 2.0（3H 隐身），9600 用 1.6（2H+2M 隐身），
    # 9500 船 hp1.2 用 2.0（3L 隐身）；未列出类型回落到全局 SATURATION_MARGIN。
    SAT_MARGIN_BY_TYPE = {
        9400: float(os.getenv("SAT_MARGIN_9400", "2.0")),
        9600: float(os.getenv("SAT_MARGIN_9600", "1.6")),
        9500: float(os.getenv("SAT_MARGIN_9500", "2.0")),
    }

    def _margin_for(self, target_id: int) -> float:
        target = self.target_by_id(int(target_id))
        if target is not None:
            return self.SAT_MARGIN_BY_TYPE.get(
                int(target.entity_type), self.SATURATION_MARGIN)
        return self.SATURATION_MARGIN

    def commit_expected_damage(self, entity_id: int, target_id: int | None,
                               missile_type: int) -> None:
        """登记一枚弹对目标的期望毁伤承诺（发射/换靶时调用，同步生效）。"""
        if target_id is None:
            return
        target = self.target_by_id(int(target_id))
        if target is None:
            return
        damage = self.EXPECTED_DAMAGE.get(int(missile_type), {}).get(
            target.entity_type, 0.0
        )
        if damage <= 0.0:
            return
        self.inflight_commit.setdefault(int(target_id), {})[int(entity_id)] = damage

    def release_commit(self, entity_id: int) -> None:
        """释放一枚弹的全部承诺（换靶离开原目标 / 死亡时调用）。"""
        entity_id = int(entity_id)
        empty = []
        for target_id, holders in self.inflight_commit.items():
            if entity_id in holders:
                del holders[entity_id]
                if not holders:
                    empty.append(target_id)
        for target_id in empty:
            del self.inflight_commit[target_id]

    def is_saturated(self, target_id: int) -> bool:
        """目标是否已被在飞承诺填满（对后到者应掩码）。"""
        holders = self.inflight_commit.get(int(target_id))
        if not holders:
            return False
        health = self.initial_health.get(int(target_id), 0.0)
        if health <= 0.0:
            return False
        return sum(holders.values()) >= health * self._margin_for(target_id)

    def is_redundant(self, entity_id: int, target_id: int) -> bool:
        """该弹的承诺是否冗余：其余在飞承诺已足以填满目标。

        KEEP 掩码（抽走冗余承诺改向未饱和目标）只对冗余弹生效——
        首批必要承诺者永不被抽走，保证已获分配的目标必然有弹进近。
        """
        holders = self.inflight_commit.get(int(target_id))
        if not holders or int(entity_id) not in holders:
            return False
        health = self.initial_health.get(int(target_id), 0.0)
        if health <= 0.0:
            return False
        others = sum(v for k, v in holders.items() if k != int(entity_id))
        return others >= health * self._margin_for(target_id)
    # ---------- 观测派生量 ----------

    @property
    def pressure(self) -> float:
        """火力暴露压力：0.65×拦截弹观测比 + 0.35×损失比。"""
        launched = max(len(self.launched_ids), 1)
        ratio = min(1.0, self.interceptors_observed / launched)
        loss_ratio = len(self.lost_ids) / self.total_platforms
        return 0.65 * ratio + 0.35 * loss_ratio

    @property
    def launched_fraction(self) -> float:
        return len(self.launched_ids) / self.total_platforms

    # ---------- 目标毁伤评估（BDA） ----------

    # 9500 战前不在目录，首次发现时的血量先验（合法领域先验；随地图血量
    # 设置：E01=1.6，final20 新图=0.8，可用环境变量 SHIP_HEALTH_PRIOR 覆盖）
    SHIP_HEALTH_PRIOR = float(os.getenv("SHIP_HEALTH_PRIOR", "1.6"))
    # belief 判死超杀边际：期望毁伤需超过血量 × 该系数（交付判定存在假阳性，
    # 加边际避免把存活目标硬屏蔽导致全队改向）
    BELIEF_OVERKILL_MARGIN = 1.25

    # 我方弹种对目标类型的期望毁伤（毁伤点×命中率，合法领域先验）
    EXPECTED_DAMAGE = {
        21000: {9400: 20 * 0.8, 9600: 20 * 0.6, 9500: 0.0},
        21001: {9400: 5 * 0.8, 9600: 5 * 0.6, 9500: 0.0},
        21002: {9400: 1 * 0.05, 9600: 1 * 0.05, 9500: 1 * 0.8},
    }

    def report_delivery(self, target_id: int | None, missile_type: int) -> None:
        """合法 BDA：我方一枚导弹末段交付（stage>=4 后消失/抵近自爆），按弹种×
        目标类型累积期望毁伤；累积期望毁伤 >= 战前血量 → 判定 belief-dead。

        全程只使用我方自身信息（弹种、投送事件）与战前目录，不触碰敌方真值。
        引擎毁伤为确定性乘法（毁伤点×命中率），该推算仅受卫星窗口与
        入射夹角修正的偏差影响。
        """
        if target_id is None or int(target_id) not in self.expected_damage:
            return
        target = self.target_by_id(int(target_id))
        target_type = target.entity_type if target is not None else -1
        damage = self.EXPECTED_DAMAGE.get(int(missile_type), {}).get(target_type, 0.0)
        self.expected_damage[int(target_id)] += damage
        health = self.initial_health.get(int(target_id), 0.0)
        if (
            int(target_id) not in self._belief_dead
            and health > 0.0
            and self.expected_damage[int(target_id)]
            >= health * self.BELIEF_OVERKILL_MARGIN
        ):
            self._belief_dead.add(int(target_id))
            self.step_destroyed.append(int(target_id))

    def oracle_sync_target_status(self, full_observation: dict) -> list[int]:
        """Oracle BDA（性能上界实验专用）：读引擎全局真值标记目标摧毁。

        违反 CTDE 边界（Critic 专属真值进入 Actor 观测），默认关闭；
        与合法模式不得共用 checkpoint（观测分布不同）。
        """
        entities = (full_observation or {}).get("entities") or {}
        newly_dead: list[int] = []
        known = {t.entity_id for t in self.fusion.targets}
        for raw_id, entity in entities.items():
            entity_id = int(raw_id)
            if entity_id not in known or entity_id in self._oracle_dead:
                continue
            if float(entity.get("health", 1.0)) <= 0.0:
                self._oracle_dead.add(entity_id)
                newly_dead.append(entity_id)
        return newly_dead

    @property
    def dead_target_ids(self) -> set[int]:
        return self._belief_dead | self._oracle_dead

    def targets(self) -> tuple[TargetPrior, ...]:
        """可选目标 = 共享航迹表 - 判死目标（belief/oracle，槽位实时更新）。"""
        dead = self.dead_target_ids
        return tuple(
            target for target in self.fusion.targets
            if target.entity_id not in dead
        )

    @property
    def known_ships(self) -> set[int]:
        """已被合法发现并进入共享航迹表的无人船集合（发现事件只结算一次）。"""
        return self._known_ships

    # ---------- P1 少量 L 侦察状态机 ----------

    def _validate_l_scout_config(self) -> None:
        """验证 P1 侦察配置；关闭时仍拒绝明显非法的数值。"""

        if self.l_scout_count < 0:
            raise ValueError("l_scout_count must be non-negative")
        if self.l_scout_enabled and self.l_scout_count <= 0:
            raise ValueError("l_scout_count must be positive when L scouting is enabled")
        if self.l_scout_expected_targets < 0:
            raise ValueError("l_scout_expected_targets must be non-negative")
        if not 0.0 <= self.l_scout_coverage_threshold <= 1.0:
            raise ValueError("l_scout_coverage_threshold must be in [0, 1]")
        if self.l_scout_quiet_steps < 0:
            raise ValueError("l_scout_quiet_steps must be non-negative")
        if self.l_scout_deadline < 0:
            raise ValueError("l_scout_deadline must be non-negative")

    def configure_l_scouts(self, l_entity_ids) -> tuple[int, ...]:
        """冻结全体 L 候选池，并激活按 ID 排序的前 K 枚侦察弹。

        全池顺序在回合内锁定；活跃 scout 损失时只按该顺序补下一枚
        reserve，因而 MAT 的随机/动态决策顺序不会暗中改变补位身份。
        """

        pool = (
            tuple(sorted({int(entity_id) for entity_id in l_entity_ids}))
            if self.l_scout_enabled else ()
        )
        if self._l_scouts_configured:
            if pool != self._l_scout_pool_ids:
                raise RuntimeError("L scout pool is already frozen for this episode")
            return self.l_scout_ids
        self._l_scout_pool_ids = pool
        self.l_scout_ids = pool[:self.l_scout_count]
        self._l_scout_history = list(self.l_scout_ids)
        self._l_scouts_configured = True
        self.update_l_scout_phase(self._latest_report_step)
        return self.l_scout_ids

    def _replace_lost_scout(self, lost_entity_id: int, step: int) -> None:
        """移除损失 scout，并在侦察仍 active 时顺序补一个 reserve。"""

        active = [
            entity_id for entity_id in self.l_scout_ids
            if entity_id != int(lost_entity_id)
        ]
        replacement = None
        replacement_allowed = self.l_scouting_active
        if replacement_allowed:
            self._l_scout_replacement_attempts += 1
            already_used = set(self._l_scout_history)
            replacement = next(
                (
                    entity_id for entity_id in self._l_scout_pool_ids
                    if entity_id not in already_used
                    and entity_id not in self.lost_ids
                ),
                None,
            )
            if replacement is None:
                self._l_scout_replacement_exhausted += 1
            else:
                active.append(replacement)
                active.sort()
                self._l_scout_history.append(replacement)
                self._l_scout_replacement_count += 1
        self._l_scout_replacement_events.append({
            "step": max(0, int(step)),
            "lost_entity_id": int(lost_entity_id),
            "replacement_entity_id": replacement,
            "replacement_allowed": replacement_allowed,
            "active_scout_ids": list(active),
        })
        # Stop 后同样移除死亡成员，但绝不再补位。
        self.l_scout_ids = tuple(active)

    def is_l_scout(self, entity_id: int) -> bool:
        """返回实体是否属于本回合已冻结的 L 侦察集合。"""

        return int(entity_id) in self.l_scout_ids

    @property
    def l_scouting_active(self) -> bool:
        """侦察状态机是否仍允许 scout 选择新搜索格。"""

        return self.l_scout_enabled and self.l_scout_state == self.L_SCOUT_SCOUTING

    @property
    def l_scout_stopped(self) -> bool:
        """侦察是否已通过完整发现、置信或截止时间锁存停止。"""

        return self.l_scout_enabled and self.l_scout_state in {
            self.L_SCOUT_COMPLETE,
            self.L_SCOUT_CONFIDENT_STOP,
            self.L_SCOUT_DEADLINE_STOP,
        }

    @property
    def l_scout_stop_reason(self) -> str | None:
        """返回锁存的停止原因，尚在侦察或功能关闭时为 None。"""

        return self.l_scout_state if self.l_scout_stopped else None

    def update_l_scout_phase(self, step: int) -> str:
        """根据合法发现、实际覆盖和时间更新并锁存侦察阶段。

        终止优先级是 ``complete`` > ``confident_stop`` >
        ``deadline_stop``。一旦离开 ``scouting`` 就不再回退，避免
        已知目标被判死或统计波动导致全队重新搜索。
        """

        step = max(0, int(step))
        self._latest_report_step = max(self._latest_report_step, step)
        if not self.l_scouting_active:
            return self.l_scout_state
        if (
            self.l_scout_expected_targets > 0
            and len(self._known_ships) >= self.l_scout_expected_targets
        ):
            return self._stop_l_scouting(self.L_SCOUT_COMPLETE, step)
        quiet_anchor = self._l_scout_last_discovery_step
        if quiet_anchor is None:
            quiet_anchor = 0
        if (
            self.l_scout_expected_targets == 0
            and self.coverage_fraction >= self.l_scout_coverage_threshold
            and step - quiet_anchor >= self.l_scout_quiet_steps
        ):
            return self._stop_l_scouting(self.L_SCOUT_CONFIDENT_STOP, step)
        if step >= self.l_scout_deadline:
            return self._stop_l_scouting(self.L_SCOUT_DEADLINE_STOP, step)
        return self.l_scout_state

    def _stop_l_scouting(self, state: str, step: int) -> str:
        """锁存终止状态并释放所有搜索格占用。"""

        if state not in {
            self.L_SCOUT_COMPLETE,
            self.L_SCOUT_CONFIDENT_STOP,
            self.L_SCOUT_DEADLINE_STOP,
        }:
            raise ValueError(f"invalid L scout terminal state: {state}")
        if not self.l_scouting_active:
            return self.l_scout_state
        self.l_scout_state = state
        self._l_scout_stop_step = max(0, int(step))
        self._l_scout_assignments.clear()
        self._l_scout_cell_owners.clear()
        return self.l_scout_state

    def available_search_cells(self, entity_id: int | None = None) -> tuple[int, ...]:
        """返回未覆盖且未被其他 scout 占用的搜索格。

        传入 scout ID 时，该 scout 自己已占用且尚未覆盖的格仍保持
        可用，以允许策略继续当前任务。非 scout 或终止后返回空集。
        """

        if not self.l_scouting_active:
            return ()
        requester = None if entity_id is None else int(entity_id)
        if requester is not None and not self.is_l_scout(requester):
            return ()
        return tuple(
            cell
            for cell in range(self.search_rows * self.search_cols)
            if cell not in self.covered_search_cells
            and (
                cell not in self._l_scout_cell_owners
                or self._l_scout_cell_owners[cell] == requester
            )
        )

    def assign_search_cell(self, entity_id: int, cell: int) -> bool:
        """将一个未覆盖搜索格原子地分配给指定 scout。

        若搜索格已被覆盖、被其他 scout 占用，或实体不是已冻结
        scout，分配失败且保留该实体的旧分配。
        """

        entity_id = int(entity_id)
        cell = int(cell)
        if not self.l_scouting_active or not self.is_l_scout(entity_id):
            return False
        if not 0 <= cell < self.search_rows * self.search_cols:
            return False
        if cell in self.covered_search_cells:
            self._l_scout_covered_rejections += 1
            return False
        owner = self._l_scout_cell_owners.get(cell)
        if owner is not None and owner != entity_id:
            self._l_scout_assignment_conflicts += 1
            return False
        previous = self._l_scout_assignments.get(entity_id)
        if previous == cell:
            return True
        if previous is not None:
            self._l_scout_cell_owners.pop(previous, None)
        self._l_scout_assignments[entity_id] = cell
        self._l_scout_cell_owners[cell] = entity_id
        self._l_scout_assignments_total += 1
        return True

    def release_search_assignment(self, entity_id: int) -> int | None:
        """释放 scout 的搜索格占用，并返回原搜索格索引。"""

        entity_id = int(entity_id)
        cell = self._l_scout_assignments.pop(entity_id, None)
        if cell is not None and self._l_scout_cell_owners.get(cell) == entity_id:
            del self._l_scout_cell_owners[cell]
        return cell

    def assigned_search_cell(self, entity_id: int) -> int | None:
        """返回 scout 当前占用的搜索格，无分配时为 None。"""

        return self._l_scout_assignments.get(int(entity_id))

    @property
    def l_scout_metrics(self) -> dict:
        """返回可直接写入 episode summary 的 P1 侦察诊断指标。"""

        last_discovery = self._l_scout_last_discovery_step
        quiet_anchor = 0 if last_discovery is None else last_discovery
        remaining = sum(
            1
            for cell in range(self.search_rows * self.search_cols)
            if cell not in self.covered_search_cells
            and cell not in self._l_scout_cell_owners
        )
        return {
            "enabled": self.l_scout_enabled,
            "state": self.l_scout_state,
            "stop_reason": self.l_scout_stop_reason,
            "stop_step": self._l_scout_stop_step,
            "configured": self._l_scouts_configured,
            "configured_count": min(
                self.l_scout_count, len(self._l_scout_pool_ids)
            ),
            "pool_count": len(self._l_scout_pool_ids),
            "pool_ids": list(self._l_scout_pool_ids),
            "active_count": len(self.l_scout_ids),
            "scout_ids": list(self.l_scout_ids),
            "scout_history": list(self._l_scout_history),
            "replacement_attempts": self._l_scout_replacement_attempts,
            "replacement_count": self._l_scout_replacement_count,
            "replacement_exhausted": self._l_scout_replacement_exhausted,
            "replacement_events": list(self._l_scout_replacement_events),
            "expected_targets": self.l_scout_expected_targets,
            "known_targets": len(self._known_ships),
            "coverage_fraction": self.coverage_fraction,
            "covered_cells": len(self.covered_search_cells),
            "coverage_threshold": self.l_scout_coverage_threshold,
            "last_discovery_step": last_discovery,
            "quiet_steps": self.l_scout_quiet_steps,
            "quiet_elapsed": max(0, self._latest_report_step - quiet_anchor),
            "deadline": self.l_scout_deadline,
            "active_assignments": len(self._l_scout_assignments),
            "assignments": dict(self._l_scout_assignments),
            "assignments_total": self._l_scout_assignments_total,
            "assignment_conflicts": self._l_scout_assignment_conflicts,
            "covered_rejections": self._l_scout_covered_rejections,
            "remaining_unassigned_cells": remaining,
        }

    # ---------- 公共搜索区域（L 发现 9500 前的合法先验） ----------

    def _resolve_search_bounds(
        self,
        initial_targets: tuple[TargetPrior, ...],
        requested: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float]:
        """返回 (west, east, south, north)，全程不使用隐藏目标真值。"""

        if requested is not None:
            if len(requested) != 4:
                raise ValueError("search_bounds must be (west, east, south, north)")
            west, east, south, north = (float(value) for value in requested)
            if not all(math.isfinite(value) for value in (west, east, south, north)):
                raise ValueError("search_bounds must contain finite values")
            if west >= east or south >= north:
                raise ValueError("search_bounds must be non-degenerate")
            return west, east, south, north

        scale = max(abs(float(self.coordinate_scale)), 1e-9)
        fallback = tuple(
            value / scale for value in self.SEARCH_FALLBACK_CANONICAL_BOUNDS
        )
        positions = [
            (float(target.position.lon), float(target.position.lat))
            for target in initial_targets
            if math.isfinite(float(target.position.lon))
            and math.isfinite(float(target.position.lat))
        ]
        if not positions:
            return fallback
        lons = [position[0] for position in positions]
        lats = [position[1] for position in positions]
        lon_span = max(lons) - min(lons)
        lat_span = max(lats) - min(lats)
        minimum_padding = self.SEARCH_FALLBACK_PADDING_DEG / scale
        lon_padding = max(minimum_padding, 0.2 * lon_span)
        lat_padding = max(minimum_padding, 0.2 * lat_span)
        bounds = (
            min(lons) - lon_padding,
            max(lons) + lon_padding,
            min(lats) - lat_padding,
            max(lats) + lat_padding,
        )
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            return fallback
        return bounds

    def _build_search_sector_centers(self) -> tuple[tuple[float, float], ...]:
        west, east, south, north = self.search_bounds
        cell_width = (east - west) / self.search_cols
        cell_height = (north - south) / self.search_rows
        # 与部署网格一致：row-major，由北向南。
        return tuple(
            (
                west + (col + 0.5) * cell_width,
                north - (row + 0.5) * cell_height,
            )
            for row in range(self.search_rows)
            for col in range(self.search_cols)
        )

    def search_sectors(self) -> tuple[tuple[float, float], ...]:
        """Map-bounded search-cell centers; hidden 9500 positions never participate."""

        return self._search_sector_centers

    def search_cell_bounds(self, cell: int) -> tuple[float, float, float, float]:
        """返回宏格 ``(west, east, south, north)``，不依赖隐藏真值。"""

        cell = int(cell)
        total = self.search_rows * self.search_cols
        if not 0 <= cell < total:
            raise IndexError(f"search cell out of range: {cell}")
        west, east, south, north = self.search_bounds
        cell_width = (east - west) / self.search_cols
        cell_height = (north - south) / self.search_rows
        row, col = divmod(cell, self.search_cols)
        cell_west = west + col * cell_width
        cell_east = cell_west + cell_width
        cell_north = north - row * cell_height
        cell_south = cell_north - cell_height
        return cell_west, cell_east, cell_south, cell_north

    def search_cell_waypoints(
        self,
        cell: int,
        horizontal_radius_km: float | None = None,
    ) -> tuple[tuple[float, float], ...]:
        """为一个顶层宏格生成确定性 boustrophedon 微观扫掠路径。

        相邻扫掠线的物理间距不超过 ``2 * horizontal_radius_km``，首尾
        扫掠线到边界不超过该半径。默认按 2km 保守水平覆盖半径设计；
        路径只使用公开搜索边界和坐标尺度，不读取隐藏目标。
        """

        radius = float(
            self.L_SCOUT_HORIZONTAL_COVERAGE_KM
            if horizontal_radius_km is None else horizontal_radius_km
        )
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("horizontal_radius_km must be positive and finite")
        west, east, south, north = self.search_cell_bounds(cell)
        width_km = (
            (east - west) * self.DEGREE_KM_UPPER_BOUND
        )
        height_km = (
            (north - south) * self.DEGREE_KM_UPPER_BOUND
        )

        # 航点会提前 switch_km 切换，因此横向半间距也要给纵向提前量
        # 留出圆形探测足迹：sqrt(radius^2-switch^2)。这样即使在端点
        # 提前转弯，宏格四角也仍处于保守 2km 水平覆盖圆内。
        switch_km = min(self.L_SCOUT_WAYPOINT_SWITCH_KM, radius * 0.99)
        half_lane_span = math.sqrt(max(radius * radius - switch_km * switch_km, 1e-9))
        max_lane_spacing = 2.0 * half_lane_span
        # 沿较长边飞、跨较短边布线，减少转弯点；每条线两个端点。
        if width_km >= height_km:
            lane_count = max(1, int(math.ceil(height_km / max_lane_spacing)))
            lat_step = (north - south) / lane_count
            waypoints = []
            for lane in range(lane_count):
                lat = south + (lane + 0.5) * lat_step
                endpoints = ((west, lat), (east, lat))
                if lane % 2:
                    endpoints = tuple(reversed(endpoints))
                waypoints.extend(endpoints)
        else:
            lane_count = max(1, int(math.ceil(width_km / max_lane_spacing)))
            lon_step = (east - west) / lane_count
            waypoints = []
            for lane in range(lane_count):
                lon = west + (lane + 0.5) * lon_step
                endpoints = ((lon, south), (lon, north))
                if lane % 2:
                    endpoints = tuple(reversed(endpoints))
                waypoints.extend(endpoints)
        return tuple((float(lon), float(lat)) for lon, lat in waypoints)


    def _search_cell_index(self, lon: float, lat: float) -> int | None:
        try:
            lon = float(lon)
            lat = float(lat)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(lon) or not math.isfinite(lat):
            return None
        west, east, south, north = self.search_bounds
        if lon < west or lon > east or lat < south or lat > north:
            return None
        col = min(
            self.search_cols - 1,
            int((lon - west) / (east - west) * self.search_cols),
        )
        row = min(
            self.search_rows - 1,
            int((north - lat) / (north - south) * self.search_rows),
        )
        return row * self.search_cols + col

    def record_search_visit(
        self, entity_id: int, lon: float, lat: float, step: int
    ) -> bool:
        """记录已发射 L 的实际路径；仅旧模式以“进入宏格”算覆盖。

        P1 启用时这里只记路径访问，不产生 coverage。P1 coverage 必须由
        底层完整执行 ``search_cell_waypoints`` 后调用
        :meth:`complete_search_cell` 提交，避免把一次穿格误当整格扫清。
        """

        cell = self._search_cell_index(lon, lat)
        if cell is None:
            return False
        entity_id = int(entity_id)
        step = int(step)
        self.search_cell_visits[cell] = self.search_cell_visits.get(cell, 0) + 1
        self.search_cell_last_step[cell] = step
        if self.l_scout_enabled:
            return False
        return self._mark_search_cell_covered(entity_id, cell, step)

    def complete_search_cell(self, entity_id: int, cell: int, step: int) -> bool:
        """提交 P1 完整微观扫掠；仅当前 owner 可完成其分配的宏格。"""

        entity_id = int(entity_id)
        cell = int(cell)
        step = int(step)
        if (
            not self.l_scouting_active
            or not self.is_l_scout(entity_id)
            or self._l_scout_assignments.get(entity_id) != cell
        ):
            return False
        return self._mark_search_cell_covered(entity_id, cell, step)

    def _mark_search_cell_covered(
        self, entity_id: int, cell: int, step: int
    ) -> bool:
        """覆盖账本的唯一写入口。"""

        if cell in self.covered_search_cells:
            return False
        self.covered_search_cells.add(cell)
        self.step_coverage_by_entity.setdefault(entity_id, set()).add(cell)
        # 完成后释放 owner，使其在下一个顶层事件选择新的未覆盖格。
        owner = self._l_scout_cell_owners.get(cell)
        if owner is not None:
            self.release_search_assignment(owner)
        self.update_l_scout_phase(step)
        return True

    @property
    def coverage_fraction(self) -> float:
        total = self.search_rows * self.search_cols
        return len(self.covered_search_cells) / max(total, 1)


    @classmethod
    def _track_step(cls, raw_time, report_step: int) -> int:
        """Map a track timestamp onto the episode step domain.

        Native ``DetectInfo.time`` is an absolute millisecond clock and cannot
        be subtracted from the episode-local step.  A newly received native
        timestamp is therefore fresh at ``report_step``; small synthetic/test
        timestamps that already lie in the step domain retain their age.
        """

        try:
            timestamp = int(raw_time)
        except (TypeError, ValueError):
            return int(report_step)
        report_step = max(0, int(report_step))
        if 0 <= timestamp <= report_step:
            return timestamp
        return report_step

    @classmethod
    def _vector_dict(cls, value) -> dict[str, float]:
        return {
            axis: float(cls._field(value, axis, 0.0))
            for axis in ("x", "y", "z")
        }

    def _ingest_threat_track(self, track_id: int, track, report_step: int) -> None:
        raw_time = self._int_field(track, "time", report_step)
        seen_step = self._track_step(raw_time, report_step)
        if report_step - seen_step > self.THREAT_FRESH_STEPS:
            return
        current = self.threat_tracks.get(int(track_id))
        if current is not None and raw_time <= int(current.get("time", -1)):
            return
        self.threat_tracks[int(track_id)] = {
            "entity_id": int(track_id),
            "entity_type": 24000,
            "detect_from": int(self._field(track, "detect_from", -1)),
            "time": raw_time,
            "seen_step": seen_step,
            "lla": self._vector_dict(self._field(track, "lla", {})),
            "pos_ecf": self._vector_dict(self._field(track, "pos_ecf", {})),
            "vel_ecf": self._vector_dict(self._field(track, "vel_ecf", {})),
        }

    def _prune_threat_tracks(self, step: int) -> None:
        stale = [
            track_id
            for track_id, track in self.threat_tracks.items()
            if int(step) - int(track.get("seen_step", 0))
            > self.THREAT_FRESH_STEPS
        ]
        for track_id in stale:
            del self.threat_tracks[track_id]

    @property
    def fresh_threat_tracks(self) -> dict[int, dict]:
        self._prune_threat_tracks(self._latest_report_step)
        return dict(self.threat_tracks)

    def target_by_id(self, entity_id: int) -> TargetPrior | None:
        for target in self.targets():
            if target.entity_id == int(entity_id):
                return target
        return None

    @staticmethod
    def _field(value, name: str, default):
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _int_field(cls, value, name: str, default: int) -> int:
        try:
            return int(cls._field(value, name, default))
        except (TypeError, ValueError):
            return int(default)
