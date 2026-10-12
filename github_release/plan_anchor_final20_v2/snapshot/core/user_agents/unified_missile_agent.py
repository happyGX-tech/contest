# -*-coding:utf-8 -*-
"""统一策略下的红方进攻弹智能体（无规则上层指挥官）。

转移时序契约（A1/A2/B1）：本智能体只做"上报 + 决策 + 执行"，
不做转移提交——环境是奖励时间戳和 transition 边界的唯一权威：
    report()                     上报阶段（环境 collect_reports 调用）：
                                  航迹融合/速度估计/卫星领取
    get_action()                 决策阶段（环境 collect_decisions 调用）：
                                  事件门控 + select_decision（冻结上下文） + 执行
    encode_current_observation() 环境提交转移时编码 O_{t+1}（编码器纯函数，
                                  与下一决策的 obs_t 严格一致）
    finalize_death()             环境发现本步死亡时调用：belief-BDA 终局结算

三决策时钟的事件门控（动作由共享统一策略采样）：
- 宏时钟·发射：未发射且无进行中的等待时，发射头激活
  （NOW/W10/W25/W50）。选 WAIT 后持续执行等待、不重复记录 log-prob，
  等待到期后发射头重新激活。launch=NOW 时目标头联合采样（发射即选靶）。
- 宏时钟·选靶：已发射后的重瞄准事件（当前目标判死[立即] / 新发现[L] /
  进度停滞 / 周期重评估），最小驻留 GOAL_MIN_DWELL=60 步；
  目标集合 = 实体槽 + 搜索扇区(仅L) + KEEP。
- 微时钟·机动：转弯以脉冲执行（MOTION_PULSE_STEPS=10 步后自动回中），
  脉冲期间不重复采样（P1#10：防持续 15-20g 单侧转弯坍缩）。
  机动标签遵循仓库约定：0=左(-1) 1=保持(0) 2=右(+1)。
- 微时钟·卫星：高性能弹且预算充足且未使用时激活（集中结算）。

飞行阶段状态机（FlightPhase，B1 契约）：
    PRELAUNCH → FLYING_AUTOGUIDE ⇄ EVASION_PULSE → TERMINAL
规避脉冲为自限时动作（到期自动回中/清过载），TERMINAL 后不再机动。

航向估计：ECEF 速度差分 → 当前经纬度的 ENU 局部切平面（P1#9），
与目标方向的经纬局部平面表示（dlon×96, dlat×110）同坐标系。

交付判定（belief-BDA 输入）：stage>=4（确认末段自爆）或历史最近距
目标 <1.2km（真实毁伤半径约 1km，原 3km 假阳性过高，P1#6）。
"""

import logging
import math
import os

import numpy as np

from user_agents.base_agent import BaseAgent, AgentType
from policies.red.learning.unified.action_spec import FlightPhase
from policies.red.learning.unified.unified_encoder import (
    GOAL_SLOTS,
    UnifiedObsEncoder,
    eligible_targets,
    target_sort_key,
)
from policies.red.learning.unified.unified_networks import (
    KEEP_INDEX,
    LAUNCH_WAIT_STEPS,
    SECTOR_INDEX_0,
    TARGET_DIM,
)

ACTION_SET_ACC_Z = 0
ACTION_LAUNCH = 1
ACTION_CHANGE_TARGET = 2
ACTION_USE_SAT = 3

# ---- 重评估触发窄化（P1#7：原 25 步周期重开 → 事件触发 + 最小驻留） ----
GOAL_MIN_DWELL = 60            # 两次目标决策之间的最小驻留（步）
GOAL_REASSESS_INTERVAL = 150   # 周期性重评估上限（步）——兜底而非主触发
STALL_STEPS = 40               # 距离超过历史最优 ×1.05 连续这么多步 → 停滞
STALL_RATIO = 1.05
# KEEP 掩码（冗余抽走）的最小改向距离：距当前目标近于该值时不抽走（临门保护）
GOAL_REDIRECT_MIN_DIST_KM = float(os.getenv("GOAL_REDIRECT_MIN_DIST_KM", "15.0"))
# ---- 机动脉冲（P1#10）----
MOTION_PULSE_STEPS = 10        # 单次转弯最长持续步数，到期自动回中
SATELLITE_TYPE = 21000
SAT_REASSESS_INTERVAL = max(1, int(os.getenv("SAT_REASSESS_INTERVAL", "30")))
DELIVERY_STAGE = 4             # 末段及以后消失 = 交付（确认自爆）
DELIVERY_PROXIMITY_KM = 1.2    # 或历史最近距目标 <1.2km 后消失 = 交付（兜底）
L_TYPE = 21002
# P1 微观扫掠在距离航点 1km 时即切下一个点，低于保守 2km 水平探测
# 半径，既不留覆盖空洞，也避免把无实体航点飞成“撞点”。
SCOUT_WAYPOINT_SWITCH_KM = 1.0
SCOUT_EGRESS_KM = 5.0

logger = logging.getLogger(__name__)


class UnifiedAttackMissileAgent(BaseAgent):
    def __init__(self, agent_id: int, entity_id: int, init_observation: dict,
                 unified_policy, max_steps: int = 1200, agent_index: int = 0,
                 evasion_skill=None):
        super().__init__(agent_id, entity_id, AgentType.AIRCRAFT, init_observation)
        self.policy = unified_policy
        self.agent_index = int(agent_index)
        self.missile_type = 21000
        # Explicit evaluation-only intervention for the paired P0 causal arm.
        # It is applied at the event gate (before MAT decoding), so the prefix
        # seen by later agents matches the action actually executed.
        self._p0_hold_all_l = os.getenv("P0_L_HOLD_ALL", "0") == "1"
        # 底层规避技能（可选）：env 仓库移植的 ppo_evasion_v4（51 维局部
        # 观测 → 左/保持/右）。顶层 motion 头 scripted 时由它接管威胁规避，
        # 无威胁步直通自动制导（不前向，避免每步×每弹的推理开销）
        self.evasion_skill = evasion_skill
        self._evasion_error_logged = False
        self.encoder = UnifiedObsEncoder(
            max_steps=max_steps, agent_index=agent_index, missile_type=self.missile_type
        )
        self.launch_step = -1
        # 脚本化波次（对照实验用）：WAVE_SCRIPT="0,300,600" → 按序号
        # 均匀分到各波，发射头被脚本接管（launch_active 只在波时刻开放）
        self._wave_time = None
        waves = os.getenv("WAVE_SCRIPT", "")
        if waves:
            times = sorted(int(x) for x in waves.split(",") if x.strip())
            if times:
                self._wave_time = times[self.agent_index % len(times)]
        self.maneuver_state = 0
        self.sat_used = False
        self.current_goal_id: int | None = None
        self._goal_position: tuple[float, float] | None = None  # 实体或扇区质心
        self._current_sector_index: int | None = None           # 当前搜索扇区编号
        self._sector_completed_step = -1    # 当前格首次确认覆盖的时刻
        self._search_waypoints: tuple[tuple[float, float], ...] = ()
        self._search_waypoint_index = -1
        self.last_goal_step = -1          # 上次目标头决策（发射或换靶）时刻
        self._task_started_step = -1      # 当前实际目标/搜索任务开始时刻
        self._last_macro_step = -1        # 最近一次实际宏动作采样时刻
        self._last_launch_action: int | None = None
        self._last_goal_kind = 0          # 0 none / 1 target / 2 search / 3 keep
        self._last_resource_action: int | None = None
        self._launch_wait_until = -1      # 发射头等待到期时刻（-1=无等待）
        self._pulse_steps_left = 0        # 机动脉冲剩余步数
        self._pulse_start = -1            # 脉冲起始步（驻留统计）
        self.latest_observation: dict | None = None
        # 死亡终局原因（finalize_death 记录：delivery/intercept/timeout）
        self.death_reason: str | None = None
        # 速度估计（ECEF 差分 → ENU 局部平面航向）
        self._prev_ecf: tuple[float, float, float] | None = None
        self._prev_pos: tuple[float, float] | None = None
        self._speed_estimate = 0.0
        self._heading_sin = 0.0
        self._heading_cos = 0.0
        self._sorted_targets = []
        # 交付判定缓存：最后观测阶段 / 距当前目标历史最近距离
        self._last_stage = 0
        self._min_goal_dist: float | None = None
        # 进度停滞检测：历史最优距离 / 连续停滞步数
        self._best_goal_dist: float | None = None
        self._stall_steps = 0
        # 当前距目标距离（KEEP 掩码的临门保护用；None=未测得，按很远处理）
        self._current_goal_dist_km: float | None = None

    def set_missile_type(self, missile_type: int) -> None:
        self.missile_type = int(missile_type)
        self.encoder.missile_type = int(missile_type)
        if (
            self._p0_hold_all_l
            and self.missile_type == L_TYPE
            and bool(getattr(self.policy, "training", False))
        ):
            raise RuntimeError(
                "P0_L_HOLD_ALL is evaluation-only and cannot be used in training"
            )

    def set_observation(self, observation: dict) -> None:
        self.latest_observation = observation

    # ---------------- 速度/航向估计（ENU 局部切平面） ----------------

    def _update_velocity_estimate(self, entity: dict) -> None:
        """ECEF 位置差分 → 速度大小 + ENU 局部平面航向（P1#9）。

        目标方向在编码器中以经纬局部平面表示（dlon×96, dlat×110 ≈ 东/北），
        航向必须同坐标系：ECEF 速度旋转到当前经纬点的东-北切平面。
        """
        ecf = entity.get("pos_ecf") or {}
        position = entity.get("position") or {}
        ecf_xyz = (
            float(ecf.get("x", 0.0)), float(ecf.get("y", 0.0)), float(ecf.get("z", 0.0))
        )
        lon_lat = (
            float(position.get("lon", 0.0)), float(position.get("lat", 0.0))
        )
        previous = self._prev_ecf
        # 幂等：同一帧只算一次（report 与 record_step 对同一位置不重复差分）
        if previous is None or ecf_xyz == previous:
            self._prev_ecf = ecf_xyz
            self._prev_pos = lon_lat
            return
        dx = ecf_xyz[0] - previous[0]
        dy = ecf_xyz[1] - previous[1]
        dz = ecf_xyz[2] - previous[2]
        self._prev_ecf = ecf_xyz
        self._prev_pos = lon_lat
        self._speed_estimate = math.sqrt(dx * dx + dy * dy + dz * dz)  # 每步 1 秒
        # ECEF → ENU（在当前经纬点的切平面；用差分起点/终点经纬均可，1 步内差异可忽略）
        lon, lat = math.radians(lon_lat[0]), math.radians(lon_lat[1])
        east = -math.sin(lon) * dx + math.cos(lon) * dy
        north = (
            -math.sin(lat) * math.cos(lon) * dx
            - math.sin(lat) * math.sin(lon) * dy
            + math.cos(lat) * dz
        )
        horizontal = math.hypot(east, north)
        if horizontal > 1e-6:
            self._heading_sin = east / horizontal    # sin(方位角)，方位角自北顺时针
            self._heading_cos = north / horizontal

    # ---------------- 上报阶段（全体先上报，再统一决策） ----------------

    def report(self) -> None:
        """上报阶段：融合航迹/更新统计/速度估计/领取卫星批准。"""
        observation = self.latest_observation
        if not observation or not observation.get("self"):
            return
        info_center = self.policy.info_center
        info_center.mark_alive(self.entity_id)
        info_center.ingest_report(observation)
        entity = observation["self"]
        self._update_velocity_estimate(entity)
        # 交付判定缓存
        self._last_stage = max(self._last_stage, int(entity.get("stage", 0)))
        goal_position = self._current_goal_position()
        if goal_position is not None:
            position = entity.get("position") or {}
            coordinate_scale = float(
                getattr(info_center, "coordinate_scale", 1.0)
            )
            dist_km = math.hypot(
                (float(position.get("lon", 0.0)) - goal_position[0])
                * 96.0 * coordinate_scale,
                (float(position.get("lat", 0.0)) - goal_position[1])
                * 110.0 * coordinate_scale,
            )
            self._current_goal_dist_km = dist_km
            if self._min_goal_dist is None or dist_km < self._min_goal_dist:
                self._min_goal_dist = dist_km
            # 进度停滞检测：距历史最优劣化超过 STALL_RATIO 连续 STALL_STEPS 步
            if self._best_goal_dist is None or dist_km < self._best_goal_dist:
                self._best_goal_dist = dist_km
                self._stall_steps = 0
            elif dist_km > self._best_goal_dist * STALL_RATIO:
                self._stall_steps += 1
            else:
                self._stall_steps = 0
        # 领取上一步获批的卫星（引擎指令已由策略结算统一下发）
        if self.entity_id in self.policy.satellite_granted:
            self.sat_used = True

    def _current_goal_position(self) -> tuple[float, float] | None:
        """当前目标位置：实体目标从航迹表实时读取，扇区目标用扇区质心。"""
        if self.current_goal_id is not None:
            target = self.policy.info_center.target_by_id(self.current_goal_id)
            if target is not None:
                return (target.position.lon, target.position.lat)
            return None
        return self._goal_position

    @property
    def _goal_slots(self) -> int:
        return int(getattr(self.policy, "goal_slots", GOAL_SLOTS))

    @property
    def _sector_index_0(self) -> int:
        return int(getattr(self.policy, "sector_index_0", SECTOR_INDEX_0))

    @property
    def _keep_index(self) -> int:
        return int(getattr(self.policy, "keep_index", KEEP_INDEX))

    @property
    def _target_dim(self) -> int:
        return int(getattr(self.policy, "target_dim", TARGET_DIM))

    # ---------------- 飞行阶段状态机（B1） ----------------

    @property
    def flight_phase(self) -> FlightPhase:
        """当前飞行阶段（由发射/脉冲/末段状态派生，只读）。"""
        if self.launch_step < 0:
            return FlightPhase.PRELAUNCH
        if self._last_stage >= DELIVERY_STAGE:
            return FlightPhase.TERMINAL
        if self._pulse_steps_left > 0:
            return FlightPhase.EVASION_PULSE
        return FlightPhase.FLYING_AUTOGUIDE

    # ---------------- 每步决策 ----------------

    def get_action(self) -> np.ndarray:
        observation = self.latest_observation
        actions: list[list[float]] = []
        if not observation or not observation.get("self"):
            return np.array(actions, dtype=np.float64)
        step = int(observation.get("step", 0))
        info_center = self.policy.info_center
        p1_l_scout = (
            self.missile_type == L_TYPE
            and bool(getattr(self.policy.config, "l_scout_enabled", False))
        )
        p1_scouting = p1_l_scout and info_center.l_scouting_active
        p1_scout_stopped = p1_l_scout and info_center.l_scout_stopped
        p1_is_scout = p1_l_scout and info_center.is_l_scout(self.entity_id)
        p1_hm_scripted_hold = (
            self.missile_type != L_TYPE
            and getattr(self.policy.config, "train_stage", "") == "p1_scout"
            and bool(getattr(self.policy.config, "l_scout_enabled", False))
            and info_center.l_scouting_active
        )
        p1_attack_ready = (
            p1_scout_stopped
            and len(info_center.known_ships) >= int(getattr(
                self.policy.config, "l_attack_min_discovered", 0
            ))
        )
        fixed_plan_search = bool(getattr(
            self.policy,
            "is_planned_search_entity",
            lambda _entity_id: False,
        )(self.entity_id))

        # ① 目标槽（持久稳定排序 + 效能过滤，与编码器完全一致 → 动作索引即槽位）
        self._sorted_targets = sorted(
            eligible_targets(info_center.targets(), self.missile_type),
            key=lambda t: target_sort_key(t, self.missile_type),
        )[:self._goal_slots]

        # ② 事件门控
        # 发射头：未发射且无进行中的等待（P0#4：选定 WAIT 后持续执行，
        # 不重复记录 log-prob，等待到期才重新激活）
        launch_active = (
            self.launch_step < 0 and step >= self._launch_wait_until
            and (self._wave_time is None or step >= self._wave_time)
            and not (self._p0_hold_all_l and self.missile_type == L_TYPE)
            and not p1_hm_scripted_hold
        )
        if p1_l_scout:
            # 侦察期严格只有冻结的 K 枚 scout 可发射；停止但发现不足时
            # reserve 继续待命，达到攻击门槛后才统一开放已知目标攻击。
            launch_active = launch_active and (
                (p1_scouting and p1_is_scout) or p1_attack_ready
            )

        # 一个顶层宏格动作在底层展开为确定性 lawnmower 微航点。这里只
        # 下发航点切换，不产生新的策略样本；全路径完成后 InfoCenter 才
        # 提交 coverage，随后顶层在下一事件选择新宏格。
        if (p1_scouting and p1_is_scout) or fixed_plan_search:
            sweep_action = self._advance_search_sweep(observation, step, info_center)
            if sweep_action is not None:
                actions.append(sweep_action)
        # 目标头：已发射后的重瞄准事件
        goal_event = False
        if self.launch_step >= 0:
            if self.current_goal_id is not None and info_center.target_by_id(
                self.current_goal_id
            ) is None:
                # 当前目标被判死（belief/oracle BDA）：立即重瞄准，不受驻留限制
                goal_event = True
            elif self.missile_type == L_TYPE:
                if p1_l_scout:
                    if p1_scouting:
                        # 当前搜索格被实际覆盖（InfoCenter 会释放占用）后，
                        # 至少驻留 sector_dwell 才选择下一个未覆盖格。
                        assigned = info_center.assigned_search_cell(self.entity_id)
                        dwell = int(getattr(
                            self.policy.config, "l_scout_sector_dwell", 1
                        ))
                        assignment_finished = (
                            self._current_sector_index is not None
                            and assigned != self._current_sector_index
                        )
                        if assignment_finished and self._sector_completed_step < 0:
                            self._sector_completed_step = step
                        elif not assignment_finished:
                            self._sector_completed_step = -1
                        if (
                            p1_is_scout
                            and self.current_goal_id is None
                            and assignment_finished
                            and step - self._sector_completed_step >= dwell
                        ):
                            goal_event = True
                    elif p1_attack_ready:
                        # 停止侦察后，仍在搜索的存活 scout 必须转入攻击；
                        # 已有实体目标的 L 只在目标失效时才由上面的分支换靶。
                        if self.current_goal_id is None:
                            goal_event = True
                else:
                    # L 承诺制：扇区航点期间锁定（不参与停滞/周期重评估，消除
                    # 扇区↔船槽抖动）；本步发现新船时立即开放换靶——饱和掩码
                    # 会把已填满的船槽对后到者隐藏，L 自然分散到各艘船
                    if info_center.step_discoveries:
                        goal_event = True
            else:
                # H/M：驻留 + 停滞/周期重评估
                dwell_ok = step - self.last_goal_step >= GOAL_MIN_DWELL
                if dwell_ok and (
                    self._stall_steps >= STALL_STEPS
                    or step - self.last_goal_step >= GOAL_REASSESS_INTERVAL
                ):
                    goal_event = True
        motion_active = (
            self.launch_step >= 0 and self._pulse_steps_left == 0
        )
        resource_active = (
            self.launch_step >= 0
            and self.missile_type == SATELLITE_TYPE
            and not self.sat_used
            and not bool(observation.get("is_using_satellite", False))
            and info_center.sat_budget_remaining > 0
            and (step - self.launch_step) % SAT_REASSESS_INTERVAL == 0
            and bool(getattr(
                self.policy,
                "is_satellite_decider",
                lambda _entity_id: True,
            )(self.entity_id))
        )

        # ③ 目标头动作掩码：实体槽 + 搜索扇区(仅L) + KEEP；
        #    L 打陆上目标受策略配置控制；饱和掩码——在飞期望毁伤承诺
        #    ≥ 血量 的实体槽对后到者不可见（发射/换靶即时登记、死亡/离开
        #    即时释放，拦截损失自然回补；同一决策步内先到先得）
        l_land_allowed = bool(getattr(self.policy.config, "l_land_targets", False))

        def build_entity_mask(
            apply_saturation: bool, *, known_ships_only: bool = False
        ) -> np.ndarray:
            mask = np.zeros(self._target_dim, dtype=bool)
            for slot, target in enumerate(self._sorted_targets):
                if known_ships_only and (
                    target.entity_type != 9500
                    or target.entity_id not in info_center.known_ships
                ):
                    continue
                if (
                    self.missile_type == L_TYPE
                    and not l_land_allowed
                    and target.entity_type in (9400, 9600)
                ):
                    continue
                if apply_saturation and info_center.is_saturated(target.entity_id):
                    continue
                mask[slot] = True
            return mask

        if p1_l_scout:
            goal_mask = np.zeros(self._target_dim, dtype=bool)
            if p1_scouting and p1_is_scout:
                # 扇区动作索引与 4x8 cell 索引一一对应；只开放未覆盖且
                # 未被其他 scout 占用的格，自己的未覆盖任务仍合法。
                for cell in info_center.available_search_cells(self.entity_id):
                    action_index = self._sector_index_0 + int(cell)
                    if action_index < self._keep_index:
                        goal_mask[action_index] = True
            elif p1_attack_ready:
                # P1 不做“全部表面饱和后重新放开”：没有概率容量的目标
                # 就等待承诺释放，避免全队同步过杀和高频换靶。
                goal_mask = build_entity_mask(True, known_ships_only=True)
            has_unsaturated = bool(
                goal_mask[:self._sector_index_0].any()
            )
        else:
            goal_mask = build_entity_mask(True)
            has_unsaturated = bool(goal_mask[:self._sector_index_0].any())
            if not has_unsaturated:
                # 全部实体槽饱和 → 回退为不筛饱和（宁可冗余也不无靶可选/禁发射）
                goal_mask = build_entity_mask(False)
            if self.missile_type == L_TYPE:
                search_stop = min(
                    self._sector_index_0 + len(info_center.search_sectors()),
                    self._keep_index,
                )
                goal_mask[self._sector_index_0:search_stop] = True
        goal_mask[self._keep_index] = True
        # 当前目标已失效（belief 判死）→ 掩掉 KEEP：判死分支不受驻留限制、
        # 每步都会重开目标头，若允许 KEEP 则 KEEP 解析为 None、换靶不生效，
        # 形成"判死→选KEEP→无效→再判死"的事件风暴（每弹每 2-3 步一次）
        current_goal_alive = (
            self.current_goal_id is None  # 扇区目标：KEEP 有效
            or info_center.target_by_id(self.current_goal_id) is not None
        )
        if not current_goal_alive:
            goal_mask[self._keep_index] = False
        elif (
            p1_l_scout
            and goal_event
            and self.launch_step >= 0
            and self.current_goal_id is None
        ):
            # 覆盖格重分配或侦察→攻击转换时不允许 KEEP；若没有新的
            # 合法格/未饱和目标，空掩码会暂时关闭事件，下一步再检查。
            goal_mask[self._keep_index] = False
        # KEEP 掩码（冗余抽走）：存在未饱和实体槽、自身承诺冗余（其余在飞
        # 承诺已填满当前目标）、且距当前目标仍远（>150km，改向代价低）时，
        # 掩掉 KEEP 让周期重评估改向未饱和目标。近距离保护防止临门弹被
        # 抽走（否则承诺弹永远无法完成 175-400 步的进近）
        elif (
            not p1_l_scout
            and has_unsaturated
            and self.current_goal_id is not None
            and info_center.is_redundant(self.entity_id, self.current_goal_id)
            and (self._current_goal_dist_km or 1e9) > GOAL_REDIRECT_MIN_DIST_KM
        ):
            goal_mask[self._keep_index] = False
        # 发射即选靶时无可选目标 → 禁 NOW 只允许等待。合法可选 = 实体槽
        # 或（仅 L）搜索扇区——L 战前无船槽，扇区是其 t=0 的唯一合法发射
        # 目标，必须计入（否则 L 机群在首次发现船前被整体禁射）
        launch_allowed = bool(goal_mask[:self._keep_index].any())
        # H/M can outlive the final compatible objective.  Suppress an
        # otherwise permanent retarget event when every semantic action,
        # including KEEP, is illegal; the policy retains a defensive guard too.
        if goal_event and not bool(goal_mask.any()):
            goal_event = False

        # ④ 编码 + 多头采样（select_decision 冻结 PendingDecision，
        #    转移提交由环境在引擎执行后统一完成——本智能体不再持有转移）
        obs_vector = self._encode_obs(observation, step, info_center)
        current_goal_slot = None
        if self.current_goal_id is not None:
            for slot, target in enumerate(self._sorted_targets):
                if target.entity_id == self.current_goal_id:
                    current_goal_slot = slot
                    break
        decision = self.policy.select_decision(
            agent_id=self.agent_id,
            step=step,
            obs=obs_vector,
            missile_type=self.missile_type,
            launch_active=launch_active and launch_allowed,
            goal_mask=goal_mask,
            goal_event=goal_event,
            motion_active=motion_active,
            resource_active=resource_active,
            current_goal_slot=current_goal_slot,
        )

        # 只记录本次真正激活并返回的宏动作字段；普通机动步的固定 OFF
        # 不能覆盖上一次卫星决策。卫星事件中的 None 同样按实际“不请求”记 0。
        launch_choice = decision.get("launch")
        goal_choice = decision.get("goal")
        resource_choice = decision.get("resource")
        sampled_macro = False
        if launch_choice is not None:
            self._last_launch_action = int(launch_choice)
            sampled_macro = True
        if goal_choice is not None:
            slot = int(goal_choice)
            if 0 <= slot < self._sector_index_0:
                self._last_goal_kind = 1
            elif self._sector_index_0 <= slot < self._keep_index:
                self._last_goal_kind = 2
            elif slot == self._keep_index:
                self._last_goal_kind = 3
            else:
                self._last_goal_kind = 0
            sampled_macro = True
        if resource_active or resource_choice == 1:
            self._last_resource_action = 1 if resource_choice == 1 else 0
            sampled_macro = True
        if sampled_macro:
            self._last_macro_step = step

        # ⑤ 发射头执行（宏时钟）
        if launch_choice is not None:
            if launch_choice == 0:
                pass  # NOW：目标头联合采样已在 decision["goal"] 中，下方执行
            else:
                # WAIT_k：设定等待期，期间发射头不再激活（不重复 log-prob）
                self._launch_wait_until = step + LAUNCH_WAIT_STEPS.get(
                    int(launch_choice), 10
                )

        # ⑥ 目标头执行（宏时钟：发射即选靶 或 重瞄准）
        if goal_choice is not None:
            if self.launch_step < 0:
                # 发射即选靶：goal 头仅在 launch=NOW 时激活
                target_position = self._resolve_goal_position(
                    goal_choice, info_center
                )
                if target_position is not None:
                    actions.append([
                        ACTION_LAUNCH, self.entity_id,
                        float(target_position[0]), float(target_position[1]),
                    ])
                    self.launch_step = step
                    self._task_started_step = step
                    self._launch_wait_until = -1
                    self.last_goal_step = step
                    self._reset_goal_tracking(target_position)
                    info_center.mark_launched(self.entity_id)
                    info_center.commit_expected_damage(
                        self.entity_id, self.current_goal_id, self.missile_type
                    )
                    self.policy.record_launch(self.missile_type, step,
                                              agent_id=self.agent_id)
                    if self.current_goal_id is not None:
                        self.policy.record_goal_choice(
                            self.current_goal_id, self.missile_type
                        )
                    else:
                        self.policy.record_search_action()
            elif goal_choice != self._keep_index:
                # 已发射：实体/扇区选择 → 判定是否真换靶（旧目标先留存，
                # _resolve_goal_position 会即时更新 current_goal_id）
                old_goal_id = self.current_goal_id
                old_sector = self._current_sector_index
                target_position = self._resolve_goal_position(
                    goal_choice, info_center
                )
                is_sector = self._is_sector_choice(goal_choice)
                changed = target_position is not None and (
                    (not is_sector and self.current_goal_id != old_goal_id)
                    or (is_sector and old_goal_id is not None)
                    or (is_sector and old_sector is not None
                        and old_sector != self._current_sector_index)
                )
                if changed:
                    self._task_started_step = step
                    actions.append([
                        ACTION_CHANGE_TARGET, self.entity_id,
                        float(target_position[0]), float(target_position[1]),
                    ])
                    # 饱和记账：离开旧目标释放承诺、对新目标登记承诺
                    info_center.release_commit(self.entity_id)
                    info_center.commit_expected_damage(
                        self.entity_id, self.current_goal_id, self.missile_type
                    )
                    self.policy.record_retarget(
                        step, self.launch_step, agent_id=self.agent_id)
                    if self.current_goal_id is not None:
                        self.policy.record_goal_choice(
                            self.current_goal_id, self.missile_type
                        )
                    else:
                        self.policy.record_search_action()
                    self.last_goal_step = step
                    self._reset_goal_tracking(target_position)
        if goal_choice is not None and self.launch_step >= 0:
            # 目标头激活即完成一次重评估（含 KEEP），刷新驻留计时
            self.last_goal_step = step

        # ⑦ 能力头执行：输出资源请求 → 全体采样后由策略统一集中结算
        if resource_choice == 1:
            self.policy.register_satellite_request(self.entity_id)

        # ⑧ 机动头执行（微时钟；仓库约定 0=左(-1) 1=保持(0) 2=右(+1)）
        motion = decision.get("motion")
        if (self.evasion_skill is not None and motion is not None
                and self.launch_step >= 0
                and self._last_stage < DELIVERY_STAGE):
            # 顶层机动 scripted 时由规避技能接管（无威胁步直通，不前向）
            motion = self._evasion_motion(observation, step)
        if motion is not None:
            acc = -1 if motion == 0 else (0 if motion == 1 else 1)
            if acc != 0:
                # 转弯脉冲：最长 MOTION_PULSE_STEPS 步后自动回中（P1#10）
                self._pulse_steps_left = MOTION_PULSE_STEPS
                self._pulse_start = step
                if acc != self.maneuver_state:
                    actions.append([ACTION_SET_ACC_Z, self.entity_id, float(acc), 0])
                self.maneuver_state = acc
            else:
                self._pulse_steps_left = 0
                if self.maneuver_state != 0:
                    actions.append([ACTION_SET_ACC_Z, self.entity_id, 0.0, 0])
                self.maneuver_state = 0
        elif self._pulse_steps_left > 0:
            # 脉冲进行中：不重复采样；到期自动回中
            self._pulse_steps_left -= 1
            if self._pulse_steps_left == 0:
                if self.maneuver_state != 0:
                    actions.append([ACTION_SET_ACC_Z, self.entity_id, 0.0, 0])
                self.policy.record_motion_dwell(step - self._pulse_start)
                self.maneuver_state = 0

        return np.array(actions, dtype=np.float64)

    # ---------------- 底层规避技能 ----------------

    def _evasion_motion(self, observation: dict, step: int) -> int:
        """规避技能动作（0=左 1=保持 2=右）；无威胁/异常回退自动制导。

        门控策略无关：prepare_action_mask 在无威胁时只放行"保持"——
        据此跳过网络前向（兼容 v4 的 51 维与 local_avoid_s3 的 33 维观测）。
        """
        skill = self.evasion_skill
        try:
            context = {
                "agent_id": self.agent_id,
                "entity_id": self.entity_id,
                "launch_step": self.launch_step,
                "maneuver_state": self.maneuver_state,
            }
            if getattr(skill, "requires_target_context", False):
                goal_position = self._current_goal_position()
                if goal_position is None:
                    return 1
                context["goal_position"] = {
                    "lon": goal_position[0],
                    "lat": goal_position[1],
                }
            encoded = skill.encode_motion_observation(observation, **context)
            mask = skill.prepare_action_mask(encoded, np.ones(3, dtype=np.bool_))
            if bool(mask[1]) and int(mask.sum()) == 1:
                return 1  # 无威胁：强制直行，不做前向
            return int(skill.select_action(encoded, mask))
        except Exception as error:
            if not self._evasion_error_logged:
                logger.exception(
                    "底层规避技能失败，agent=%s entity=%s；本回合回退自动制导: %s",
                    self.agent_id,
                    self.entity_id,
                    error,
                )
                self._evasion_error_logged = True
            return 1

    # ---------------- 目标解析辅助 ----------------

    def _is_sector_choice(self, choice: int) -> bool:
        return choice >= self._sector_index_0 and choice < self._keep_index

    def _slot_entity_id(self, choice: int) -> int | None:
        if 0 <= choice < len(self._sorted_targets):
            return self._sorted_targets[choice].entity_id
        return None

    def _resolve_goal_position(
        self, choice: int, info_center
    ) -> tuple[float, float] | None:
        """目标头动作 → (lon, lat)。实体槽坐标由最新合法航迹读取。"""
        p1_l_scout = (
            self.missile_type == L_TYPE
            and bool(getattr(self.policy.config, "l_scout_enabled", False))
        )
        fixed_plan_search = bool(getattr(
            self.policy,
            "is_planned_search_entity",
            lambda _entity_id: False,
        )(self.entity_id))
        if self._is_sector_choice(choice):
            sectors = info_center.search_sectors()
            index = choice - self._sector_index_0
            if index < len(sectors):
                if p1_l_scout and not info_center.assign_search_cell(
                    self.entity_id, index
                ):
                    return None
                self.current_goal_id = None
                self._current_sector_index = index
                self._sector_completed_step = -1
                if p1_l_scout or fixed_plan_search:
                    self._search_waypoints = info_center.search_cell_waypoints(index)
                    self._search_waypoint_index = 0
                    self._goal_position = self._search_waypoints[0]
                else:
                    self._search_waypoints = ()
                    self._search_waypoint_index = -1
                    self._goal_position = (
                        float(sectors[index][0]), float(sectors[index][1])
                    )
                return self._goal_position
            return None
        if 0 <= choice < len(self._sorted_targets):
            target = self._sorted_targets[choice]
            if p1_l_scout:
                if (
                    target.entity_type != 9500
                    or target.entity_id not in info_center.known_ships
                ):
                    return None
                info_center.release_search_assignment(self.entity_id)
            self.current_goal_id = target.entity_id
            self._current_sector_index = None
            self._sector_completed_step = -1
            self._search_waypoints = ()
            self._search_waypoint_index = -1
            self._goal_position = (
                float(target.position.lon), float(target.position.lat)
            )
            return self._goal_position
        return None

    def _advance_search_sweep(
        self, observation: dict, step: int, info_center
    ) -> list[float] | None:
        """临近微航点时提前切换；完整路径后原子提交宏格覆盖。"""

        cell = self._current_sector_index
        if (
            self.launch_step < 0
            or self.current_goal_id is not None
            or cell is None
            or not self._search_waypoints
            or not 0 <= self._search_waypoint_index < len(self._search_waypoints)
        ):
            return None
        fixed_plan_search = bool(getattr(
            self.policy,
            "is_planned_search_entity",
            lambda _entity_id: False,
        )(self.entity_id))
        if (
            not fixed_plan_search
            and info_center.assigned_search_cell(self.entity_id) != cell
        ):
            return None
        position = (observation.get("self") or {}).get("position") or {}
        try:
            lon = float(position["lon"])
            lat = float(position["lat"])
        except (KeyError, TypeError, ValueError):
            return None
        waypoint = self._search_waypoints[self._search_waypoint_index]
        distance_km = math.hypot(
            (lon - waypoint[0]) * 96.0,
            (lat - waypoint[1]) * 110.0,
        )
        switch_km = float(getattr(
            info_center, "L_SCOUT_WAYPOINT_SWITCH_KM",
            SCOUT_WAYPOINT_SWITCH_KM,
        ))
        if distance_km > switch_km:
            return None

        next_index = self._search_waypoint_index + 1
        if next_index < len(self._search_waypoints):
            self._search_waypoint_index = next_index
            self._goal_position = self._search_waypoints[next_index]
            return [
                ACTION_CHANGE_TARGET,
                self.entity_id,
                float(self._goal_position[0]),
                float(self._goal_position[1]),
            ]

        if fixed_plan_search:
            complete = getattr(self.policy, "record_planned_search_complete", None)
            if complete is not None:
                complete(self.entity_id, cell, step)
        elif not info_center.complete_search_cell(self.entity_id, cell, step):
            return None
        # 最后一点同样提前切走：沿末段方向延伸一个短 egress，给下一步
        # 顶层选格留出安全余量，避免在最后一个无实体航点处自爆。
        self._search_waypoint_index = len(self._search_waypoints)
        self._goal_position = self._search_egress_position(info_center)
        return [
            ACTION_CHANGE_TARGET,
            self.entity_id,
            float(self._goal_position[0]),
            float(self._goal_position[1]),
        ]

    def _search_egress_position(self, info_center) -> tuple[float, float]:
        """按末段切向外推固定安全距离；完全不依赖隐藏目标。"""

        end = self._search_waypoints[-1]
        if len(self._search_waypoints) >= 2:
            start = self._search_waypoints[-2]
        else:
            start = (end[0], end[1] - 1.0)
        east_km = (end[0] - start[0]) * 96.0
        north_km = (end[1] - start[1]) * 110.0
        norm = math.hypot(east_km, north_km)
        if norm <= 1e-9:
            east_km, north_km, norm = 0.0, 1.0, 1.0
        return (
            end[0] + (east_km / norm) * SCOUT_EGRESS_KM / 96.0,
            end[1] + (north_km / norm) * SCOUT_EGRESS_KM / 110.0,
        )

    def _reset_goal_tracking(self, position: tuple[float, float]) -> None:
        """换靶/发射后重置交付与停滞追踪（距离基线由环境层势函数管理）。"""
        self._min_goal_dist = None
        self._best_goal_dist = None
        self._stall_steps = 0

    def _encode_obs(self, observation: dict, step: int, info_center) -> np.ndarray:
        return self.encoder.encode(
            observation,
            flying=self.launch_step >= 0,
            launch_step=self.launch_step,
            maneuver_state=self.maneuver_state,
            sat_used=self.sat_used,
            current_goal_id=self.current_goal_id,
            info_center=info_center,
            speed_estimate=self._speed_estimate,
            heading_sin=self._heading_sin,
            heading_cos=self._heading_cos,
        )

    # ---------------- 转移事务的环境侧接口（A2 提交阶段调用） ----------------

    def encode_current_observation(self) -> np.ndarray | None:
        """上报后编码当前观测 O_{t+1}（环境提交转移时调用）。

        编码器为纯函数且输入（latest_observation / 内部执行状态 / 冻结后的
        info_center）在此时刻与下一步决策编码完全一致——GAE bootstrap 的
        next_obs 与下一决策的 obs_t 严格相同。仅对存活智能体调用。
        """
        observation = self.latest_observation
        if not observation or not observation.get("self"):
            return None
        step = int(observation.get("step", 0))
        return self._encode_obs(observation, step, self.policy.info_center)

    def finalize_death(self) -> None:
        """死亡终局结算（环境提交 MISSILE_DEATH 转移前调用，恰好一次）。

        记录明确死亡原因并结算 belief-BDA：
        - delivery：末段交付（stage>=4 或距目标历史最近 <1.2km 后消失）；
        - intercept：飞行中被击落（stage<4 且无抵近记录）；
        - timeout：未发射即消失（不应发生，防御性记录）。
        """
        delivered = (
            self._last_stage >= DELIVERY_STAGE
            or (
                self._min_goal_dist is not None
                and self._min_goal_dist < DELIVERY_PROXIMITY_KM
            )
        )
        if self.launch_step < 0:
            self.death_reason = "timeout"
        elif delivered:
            self.death_reason = "delivery"
        else:
            self.death_reason = "intercept"
        self.policy.info_center.mark_lost(self.entity_id)
        # 死亡释放承诺：拦截损失立即回补饱和掩码，后到者可重新入场
        self.policy.info_center.release_commit(self.entity_id)
        if delivered:
            self.policy.info_center.report_delivery(
                self.current_goal_id, self.missile_type
            )

    def reset(self):
        super().reset()
        self.launch_step = -1
        # 脚本化波次（对照实验用）：WAVE_SCRIPT="0,300,600" → 按序号
        # 均匀分到各波，发射头被脚本接管（launch_active 只在波时刻开放）
        self._wave_time = None
        waves = os.getenv("WAVE_SCRIPT", "")
        if waves:
            times = sorted(int(x) for x in waves.split(",") if x.strip())
            if times:
                self._wave_time = times[self.agent_index % len(times)]
        self.maneuver_state = 0
        self.sat_used = False
        self.current_goal_id = None
        self._goal_position = None
        self._current_sector_index = None
        self._sector_completed_step = -1
        self._search_waypoints = ()
        self._search_waypoint_index = -1
        self.last_goal_step = -1
        self._task_started_step = -1
        self._last_macro_step = -1
        self._last_launch_action = None
        self._last_goal_kind = 0
        self._last_resource_action = None
        self._launch_wait_until = -1
        self._pulse_steps_left = 0
        self._pulse_start = -1
        self._sorted_targets = []
        self._prev_ecf = None
        self._prev_pos = None
        self._speed_estimate = 0.0
        self._heading_sin = 0.0
        self._heading_cos = 0.0
        self._last_stage = 0
        self._min_goal_dist = None
        self._best_goal_dist = None
        self._stall_steps = 0
        self._current_goal_dist_km = None
        self.death_reason = None
        self._evasion_error_logged = False
