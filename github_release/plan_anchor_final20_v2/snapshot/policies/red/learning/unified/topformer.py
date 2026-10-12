"""TopFormer-v3：事件驱动、四字段、顺序条件化的顶层联合策略。

部署格、FIRE/HOLD、目的地和天眼请求共享同一套场景编码器、MAT 式动作
前缀解码器与 Actor 优化器。目的地是变长的“合法已知目标 + 4x8 搜索格 +
KEEP”指针动作；后决策单位显式读取本事件中前序单位已经发出的结构化动作。
状态同时编码当前任务、上一宏动作、持续时间、真实路径覆盖、目标承诺和合法
观测到的威胁。底层机动/规避策略保持冻结。

每条异步宏决策记录完整场景、合法掩码、动作前缀和有效字段的联合 log-prob，
使用 duration-aware SMDP-GAE 与 conditional PPO 重放。官方得分增量是共享团队
主回报；实际覆盖和任务进度仅以可回收的势函数差提供稠密反馈；真实 DAMAGE
命中源只作为 policy advantage 的因果信用通道，不重复进入 critic 目标。
"""

from __future__ import annotations

import copy
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from policies.red.baselines import TargetPrior
from .info_center import InfoCenter, infer_coordinate_scale
from .unified_encoder import (
    eligible_targets,
    target_sort_key,
)

logger = logging.getLogger(__name__)

TOPFORMER_SCHEMA_VERSION = 3
WAVE_WINDOW = 30          # 波次对齐窗口（步）
EXPERT_WAVES = {0: 300, 1: 350, 2: 0}   # 各型释放波次（步）：H/M 主力波、L 先行
EXPERT_SAT_LEAD = 170.0                  # H 临空前多少步请求窗口（窗口 180 步）
LON_MIN, LON_MAX = 115.0, 129.0
LAT_MIN, LAT_MAX = 15.0, 29.0
DIST_MAX_KM = 1200.0

# ---- 动作索引（与 unified 动作协议一致） ----
LAUNCH_NOW = 0
MOTION_AUTOGUIDE = 1
RESOURCE_OFF = 0


def _norm(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), -1.0, 1.0))


def _remap_candidate_rows(
    rows: np.ndarray, local_candidate_count: int, batch_candidate_count: int
) -> np.ndarray:
    """Move a sample-local KEEP sentinel to the padded batch KEEP row."""

    result = np.asarray(rows, dtype=np.int64).copy()
    if np.any(result < 0) or np.any(result > local_candidate_count):
        raise ValueError("candidate row is outside the sample-local candidate set")
    result[result == local_candidate_count] = batch_candidate_count
    return result


def _record_p1_search_phase(record) -> bool:
    """Read a serialized phase bit while supporting pre-P1.1 buffers."""

    # A newly added dataclass field has a class-level default. ``getattr``
    # therefore cannot tell an old unpickled instance from a new record that
    # explicitly stored False; inspect the serialized instance dictionary.
    values = getattr(record, "__dict__", {})
    if "p1_search_phase" in values:
        return bool(values["p1_search_phase"])
    return int(record.goal_kind) == 2


@dataclass
class TopFormerConfig:
    # v3 keeps the original 24 features as a stable prefix and appends explicit
    # task/action/threat history.  This makes old feature semantics inspectable
    # while preventing a slot id (whose meaning changes as targets disappear)
    # from being used as "history".
    agent_dim: int = 54
    cand_dim: int = 15
    global_dim: int = 24
    prefix_dim: int = 31
    activation_dim: int = 2       # FIRE / HOLD; HOLD 10 步后再触发
    d_model: int = 128
    n_head: int = 4
    enc_layers: int = 3
    ff_dim: int = 512
    learning_rate: float = 1e-4
    critic_lr: float = 3e-4
    clip_ratio: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    epochs: int = 2
    target_kl: float = 0.05
    entropy_floor: float = 0.3    # 指针熵低于该值时熵系数×10（防首轮坍缩）
    minibatch_records: int = 256  # 更新分块大小（场景快照大，整批前向会 OOM）
    update_every_rounds: int = 4
    min_update_records: int = 800
    gamma: float = 1.0            # 顶层段内不折扣（比分可加、回合有限）
    gae_lambda: float = float(os.getenv("GAE_LAMBDA", "0.995"))
    reward_scale: float = 0.01    # official 0--100 -> critic-friendly 0--1
    progress_shaping_coef: float = 0.05
    coverage_shaping_coef: float = 0.05
    causal_credit_coef: float = 0.20
    # Actor-only enabling credit for the exact L search action that first
    # discovered a hidden target.  The credited amount is not a fixed bounty:
    # it is the target's *subsequent realised official score increment*.  It
    # never enters the shared return or critic target.
    discovery_credit_coef: float = float(os.getenv(
        "DISCOVERY_CREDIT_COEF", "0.0"))
    # P1.1 is opt-in so legacy P1/joint checkpoints retain their exact update
    # semantics.  Main PPO is restricted to search records, while realised
    # discovery-enabling credit gets a separately balanced actor-only loss.
    p1_search_only_actor: bool = bool(os.getenv(
        "P1_SEARCH_ONLY_ACTOR", "0") == "1")
    p1_split_search_query: bool = bool(os.getenv(
        "P1_SPLIT_SEARCH_QUERY", "0") == "1")
    p1_discovery_aux_coef: float = float(os.getenv(
        "P1_DISCOVERY_AUX_COEF", "0.0"))
    discovery_bonus: float = 0.0  # v3: discovery is represented by coverage PBRS,
                                  # never as a non-potential team bonus.
    retarget_penalty: float = float(os.getenv("RETARGET_PENALTY", "0.0"))
    # 换靶代价（计入该弹当前未闭合段）：默认关闭，避免非势函数奖励改变
    # 官方目标；仅用于显式消融时由 RETARGET_PENALTY 开启。
    max_steps: int = 1200
    seed: int = 0
    device: str = "auto"
    upper_phase: str = "joint"    # goal: 发射 scripted=FIRE；joint: FIRE/HOLD 可训
    expert_rounds: int = int(os.getenv("EXPERT_ROUNDS", "0"))
    # 专家先验回合数：前 N 回合由专家规则驱动并记录（场景,专家动作），
    # 更新时做行为克隆（监督 CE）；之后自动切换 PPO。E01 级知识：
    # L 型先行搜索吸防（续航 1800）→ H/M 主力波次（300/350 步）→
    # H 临空前 ~170 步开全局窗口（180 步覆盖到达段）
    sat_trainable: bool = bool(os.getenv("SAT_LEARN", "1") == "1")
    # 卫星头：H 弹临空时请求天眼窗口（引擎预算 100 次 × 3 分钟；窗口内
    # H 对 9400/9600 命中率→1.0，哨卫巢 hp18 一发即杀）
    eval_sample: bool = bool(os.getenv("EVAL_SAMPLE", "0") == "1")
    # 评估期采样：argmax 会让全舰队对同一槽位共振（同型弹 query 相同 →
    # argmax 一致 → 同频换靶、8 目标全半血零击杀，实测 30.6 分）；
    # 采样才与训练期行为同分布。仅影响采样温度，不记录/不更新
    l_land_targets: bool = False
    deploy_trainable: bool = bool(os.getenv("DEPLOY_LEARN", "1") == "1")
    satellite_cost: float = 0.0
    # P1 is an intentionally narrow learning stage: the environment/agent
    # applies the few-L scout state machine, while TopFormer updates only the
    # L goal query.  The switch is off by default so existing checkpoints and
    # joint runs retain their exact behaviour.
    train_stage: str = os.getenv("TOPFORMER_TRAIN_STAGE", "joint")
    l_scout_enabled: bool = bool(os.getenv("L_SCOUT_ENABLED", "0") == "1")
    l_scout_count: int = int(os.getenv("L_SCOUT_COUNT", "8"))
    l_scout_expected_targets: int = int(
        os.getenv("L_SCOUT_EXPECTED_TARGETS", "4"))
    l_scout_coverage_threshold: float = float(
        os.getenv("L_SCOUT_COVERAGE_THRESHOLD", "0.75"))
    l_scout_quiet_steps: int = int(os.getenv("L_SCOUT_QUIET_STEPS", "180"))
    l_scout_deadline: int = int(os.getenv("L_SCOUT_DEADLINE", "600"))
    l_attack_min_discovered: int = int(
        os.getenv("L_ATTACK_MIN_DISCOVERED", "4"))
    l_scout_sector_dwell: int = int(os.getenv("L_SCOUT_SECTOR_DWELL", "10"))
    max_goal_slots: int = 64
    search_rows: int = 4
    search_cols: int = 8
    mat_decoder_layers: int = 2


# ================= token 构建 =================

class SceneTokenizer:
    """从智能体对象集合 + InfoCenter 构造变长 token（纯特征工程）。"""

    def __init__(self, max_steps: int, coordinate_scale: float = 1.0,
                 agent_dim: int = 54, global_dim: int = 24):
        self.max_steps = max_steps
        self.coordinate_scale = float(coordinate_scale)
        self.agent_dim = int(agent_dim)
        self.global_dim = int(global_dim)

    def _canonical(self, value: float) -> float:
        return float(value) * self.coordinate_scale

    @staticmethod
    def _field(value, name: str, default):
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _vector(cls, value) -> np.ndarray:
        return np.asarray([
            float(cls._field(value, "x", 0.0)),
            float(cls._field(value, "y", 0.0)),
            float(cls._field(value, "z", 0.0)),
        ], dtype=np.float32)

    def _threat_features(self, observation: dict, step: int,
                         info_center: InfoCenter) -> np.ndarray:
        """Nearest *legally observed* fresh interceptor track.

        DetectInfo is cached by the simulator, so a raw count silently treats
        stale tracks as current threats.  v3 keeps unique tracks no older than
        30 simulation steps and exposes relative position, velocity and age.
        """
        own = (observation or {}).get("self") or {}
        own_pos = self._vector(own.get("pos_ecf") or {})
        tracks = []
        seen: set[int] = set()
        fresh = getattr(info_center, "fresh_threat_tracks", {})
        for track_id, fused in fresh.items():
            age = max(0, int(step) - int(fused.get("seen_step", step)))
            if track_id in seen or age > info_center.THREAT_FRESH_STEPS:
                continue
            seen.add(track_id)
            rel = self._vector(fused.get("pos_ecf") or {}) - own_pos
            vel = self._vector(fused.get("vel_ecf") or {})
            distance = float(np.linalg.norm(rel))
            tracks.append((distance, rel, vel, age))
        out = np.zeros(9, dtype=np.float32)
        out[0] = float(np.clip(len(tracks) / 8.0, 0.0, 1.0))
        if not tracks:
            return out
        distance, rel, vel, age = min(tracks, key=lambda item: item[0])
        out[1:4] = np.clip(rel / 100_000.0, -1.0, 1.0)
        out[4:7] = np.clip(vel / 2_000.0, -1.0, 1.0)
        out[7] = float(np.clip(age / 30.0, 0.0, 1.0))
        out[8] = float(np.clip(distance / 100_000.0, 0.0, 1.0))
        return out

    def build(self, agents: list, info_center: InfoCenter,
              step: int, deciding: int | None, history: dict | None = None,
              prefix_len: int = 0) -> dict:
        alive = [a for a in agents
                 if a.latest_observation and a.latest_observation.get("self")
                 and int(getattr(a, "entity_id", -1))
                 not in info_center.lost_ids]
        lons, lats = [], []
        for a in alive:
            pos = a.latest_observation["self"].get("position") or {}
            lons.append(float(pos.get("lon", 118.5)))
            lats.append(float(pos.get("lat", 23.0)))
        cx = float(np.mean(lons)) if lons else 118.5 / self.coordinate_scale
        cy = float(np.mean(lats)) if lats else 23.0 / self.coordinate_scale

        agent_tokens = np.zeros((len(alive), self.agent_dim), dtype=np.float32)
        agent_index: dict[int, int] = {}
        type_counts = {0: 0, 1: 0, 2: 0}
        type_launched = {0: 0, 1: 0, 2: 0}
        for i, a in enumerate(alive):
            agent_index[a.agent_id] = i
            obs_self = a.latest_observation["self"]
            pos = obs_self.get("position") or {}
            lon, lat = float(pos.get("lon", 118.5)), float(pos.get("lat", 23.0))
            ti = 0 if a.missile_type == 21000 else (1 if a.missile_type == 21001 else 2)
            launched = 1.0 if a.launch_step >= 0 else 0.0
            type_counts[ti] += 1
            if launched:
                type_launched[ti] += 1
            t = agent_tokens[i]
            t[ti] = 1.0
            t[3] = launched
            t[4] = 1.0 if a.current_goal_id is not None else 0.0
            t[5] = 1.0 if a._current_sector_index is not None else 0.0
            t[6] = ((a._current_goal_dist_km or 0.0) / DIST_MAX_KM)
            t[7] = _norm(self._canonical(lon), LON_MIN, LON_MAX)
            t[8] = _norm(self._canonical(lat), LAT_MIN, LAT_MAX)
            t[9] = _norm(self._canonical(lon - cx), -5.0, 5.0)
            t[10] = _norm(self._canonical(lat - cy), -4.0, 4.0)
            t[11] = float(a._heading_sin)
            t[12] = float(a._heading_cos)
            t[13] = float(np.clip((a._speed_estimate or 0.0) / 2500.0, 0.0, 1.0))
            t[14] = 0.0 if launched else float(np.clip(step / 60.0, 0.0, 1.0))
            t[15] = 1.0 if a.sat_used else 0.0
            t[22] = float(obs_self.get("stage", 0)) / 5.0
            t[23] = 1.0 if a.agent_id == deciding else 0.0

            # Explicit current task and temporal action history (v3).
            current_goal = getattr(a, "current_goal_id", None)
            current_sector = getattr(a, "_current_sector_index", None)
            if current_goal is not None:
                t[24] = 1.0
                target = info_center.target_by_id(int(current_goal))
                if target is not None:
                    t[27] = _norm(
                        self._canonical(target.position.lon - lon), -5.0, 5.0)
                    t[28] = _norm(
                        self._canonical(target.position.lat - lat), -4.0, 4.0)
                    holders = info_center.inflight_commit.get(int(current_goal), {})
                    t[44] = float(np.clip(len(holders) / 8.0, 0.0, 1.0))
            elif current_sector is not None:
                t[25] = 1.0
                sectors_ = info_center.search_sectors()
                if 0 <= int(current_sector) < len(sectors_):
                    slon, slat = sectors_[int(current_sector)]
                    t[27] = _norm(self._canonical(slon - lon), -5.0, 5.0)
                    t[28] = _norm(self._canonical(slat - lat), -4.0, 4.0)
                    covered = getattr(info_center, "covered_search_cells", set())
                    t[43] = 1.0 if int(current_sector) in covered else 0.0
            else:
                t[26] = 1.0
            task_started = int(getattr(a, "_task_started_step", -1))
            t[29] = float(np.clip(
                (step - task_started) / self.max_steps if task_started >= 0 else 0.0,
                0.0, 1.0))
            wait_until = int(getattr(a, "_launch_wait_until", -1))
            t[30] = float(np.clip(max(0, wait_until - step) / 50.0, 0.0, 1.0))
            last_macro = int(getattr(a, "_last_macro_step", -1))
            t[31] = float(np.clip(
                (step - last_macro) / self.max_steps if last_macro >= 0 else 1.0,
                0.0, 1.0))
            last_launch = getattr(a, "_last_launch_action", None)
            if last_launch is not None and 0 <= int(last_launch) < 4:
                t[32 + int(last_launch)] = 1.0
            last_goal_kind = int(getattr(a, "_last_goal_kind", 0))
            if 0 <= last_goal_kind < 4:
                t[36 + last_goal_kind] = 1.0
            last_resource = getattr(a, "_last_resource_action", None)
            if last_resource is not None and 0 <= int(last_resource) < 2:
                t[40 + int(last_resource)] = 1.0
            t[42] = float(getattr(info_center, "coverage_fraction", 0.0))
            t[45:54] = self._threat_features(
                a.latest_observation, step, info_center)

        total = max(info_center.total_platforms, 1)
        gl = len(info_center.launched_ids) / total
        for i, a in enumerate(alive):
            ti = 0 if a.missile_type == 21000 else (1 if a.missile_type == 21001 else 2)
            frac = (type_launched[ti] / max(type_counts[ti], 1))
            agent_tokens[i, 16] = frac
            agent_tokens[i, 17] = gl
            agent_tokens[i, 18] = float(np.clip(info_center.interceptors_observed / 20.0, 0.0, 1.0))
            agent_tokens[i, 19] = float(np.clip(info_center.pressure, 0.0, 1.0))
            agent_tokens[i, 20] = float(np.clip(info_center.comm_group_total / 50.0, 0.0, 1.0))
            agent_tokens[i, 21] = 1.0  # alive

        # 候选并集：全部合法目标（未按弹型过滤）+ 扇区
        union: list = sorted(info_center.targets(), key=lambda x: target_sort_key(x))
        sectors = info_center.search_sectors()
        cand_tokens = np.zeros((len(union) + len(sectors), 15), dtype=np.float32)
        union_ids: list[int] = []
        for j, tgt in enumerate(union):
            union_ids.append(tgt.entity_id)
            hp0 = info_center.initial_health.get(tgt.entity_id, 0.0) or 1.0
            exp_dmg = info_center.expected_damage.get(tgt.entity_id, 0.0)
            holders = info_center.inflight_commit.get(tgt.entity_id, {})
            commit = sum(holders.values())
            t = cand_tokens[j]
            t[0 if tgt.entity_type == 9400 else 1 if tgt.entity_type == 9600 else 2] = 1.0
            t[4] = float(np.clip(tgt.value / 5.0, 0.0, 1.0))
            t[5] = float(np.clip(1.0 - exp_dmg / hp0, 0.0, 1.0))
            t[6] = _norm(self._canonical(tgt.position.lon), LON_MIN, LON_MAX)
            t[7] = _norm(self._canonical(tgt.position.lat), LAT_MIN, LAT_MAX)
            t[8] = float(np.clip(np.hypot(
                self._canonical(tgt.position.lon - cx) * 96.0,
                self._canonical(tgt.position.lat - cy) * 110.0,
            ) / DIST_MAX_KM, 0.0, 1.0))
            t[9] = 1.0 if (tgt.entity_type == 9500
                           and tgt.entity_id in info_center.known_ships) else 0.0
            t[10] = float(np.clip(len(holders) / 8.0, 0.0, 1.0))
            t[11] = float(np.clip(commit / max(hp0, 0.1), 0.0, 2.0))
            etas = []
            for agent in alive:
                if (getattr(agent, "current_goal_id", None) == tgt.entity_id
                        and getattr(agent, "_current_goal_dist_km", None) is not None):
                    speed = max(float(getattr(agent, "_speed_estimate", 0.0)), 1.0)
                    etas.append(float(agent._current_goal_dist_km) * 1000.0 / speed)
            t[12] = float(np.clip(min(etas) / self.max_steps, 0.0, 1.0)) if etas else 1.0
            last_seen = getattr(info_center, "target_last_seen", {}).get(tgt.entity_id)
            t[13] = float(np.clip((step - last_seen) / self.max_steps, 0.0, 1.0)) \
                if last_seen is not None else 1.0
            t[14] = float(np.clip(exp_dmg / max(hp0, 0.1), 0.0, 2.0))
        sector_assignments: dict[int, list] = {}
        for agent in alive:
            sector = getattr(agent, "_current_sector_index", None)
            if sector is not None:
                sector_assignments.setdefault(int(sector), []).append(agent)
        for s, (slon, slat) in enumerate(sectors):
            j = len(union) + s
            union_ids.append(-(s + 1))  # 扇区用负索引标记
            t = cand_tokens[j]
            t[3] = 1.0
            t[6] = _norm(self._canonical(slon), LON_MIN, LON_MAX)
            t[7] = _norm(self._canonical(slat), LAT_MIN, LAT_MAX)
            t[8] = float(np.clip(np.hypot(
                self._canonical(slon - cx) * 96.0,
                self._canonical(slat - cy) * 110.0,
            ) / DIST_MAX_KM, 0.0, 1.0))
            covered = getattr(info_center, "covered_search_cells", set())
            visits = getattr(info_center, "search_cell_visits", {})
            last_visit = getattr(info_center, "search_cell_last_step", {}).get(s)
            assigned = sector_assignments.get(s, [])
            t[9] = 1.0 if s in covered else 0.0
            t[10] = float(np.clip(len(assigned) / 8.0, 0.0, 1.0))
            t[11] = float(np.clip(visits.get(s, 0) / 4.0, 0.0, 1.0))
            t[12] = float(np.clip(
                (step - last_visit) / self.max_steps, 0.0, 1.0)) \
                if last_visit is not None else 1.0
            t[13] = 0.0 if s in covered else 1.0
            if assigned:
                eta = []
                for agent in assigned:
                    pos = (agent.latest_observation.get("self") or {}).get("position") or {}
                    dist = np.hypot(
                        self._canonical(float(pos.get("lon", slon)) - slon) * 96.0,
                        self._canonical(float(pos.get("lat", slat)) - slat) * 110.0,
                    )
                    speed = max(float(getattr(agent, "_speed_estimate", 0.0)), 1.0)
                    eta.append(dist * 1000.0 / speed)
                t[14] = float(np.clip(min(eta) / self.max_steps, 0.0, 1.0))
            else:
                t[14] = 1.0

        g = np.zeros(self.global_dim, dtype=np.float32)
        g[0] = step / self.max_steps
        g[1] = 1.0 - step / self.max_steps
        g[2] = info_center.sat_budget_remaining / float(InfoCenter.SAT_BUDGET_TOTAL)
        g[3] = gl
        for ti in range(3):
            g[4 + ti] = type_launched[ti] / max(type_counts[ti], 1)
        g[7] = len(info_center.known_ships) / 10.0
        g[8] = float(np.clip(info_center.pressure, 0.0, 1.0))
        g[9] = len(info_center.lost_ids) / total
        g[10] = len(alive) / total
        g[11] = 1.0 if info_center.satellite_active else 0.0
        if self.global_dim >= 24:
            g[12] = float(getattr(info_center, "coverage_fraction", 0.0))
            g[13] = float(np.clip(
                sum(len(v) for v in info_center.inflight_commit.values()) / total,
                0.0, 2.0))
            g[14] = float(np.clip(info_center.interceptors_observed / 20.0, 0.0, 1.0))
            g[15] = float(np.clip(len(info_center.step_discoveries) / 4.0, 0.0, 1.0))
            history = history or {}
            g[16] = float(np.clip(history.get("official_delta", 0.0) / 100.0, 0.0, 1.0))
            g[17] = float(np.clip(history.get("event_age", self.max_steps)
                                  / self.max_steps, 0.0, 1.0))
            g[18] = float(np.clip(history.get("launches", 0) / total, 0.0, 1.0))
            g[19] = float(np.clip(history.get("targets", 0) / total, 0.0, 1.0))
            g[20] = float(np.clip(history.get("searches", 0) / total, 0.0, 1.0))
            g[21] = 1.0 if history.get("satellite", 0) else 0.0
            g[22] = float(np.clip(prefix_len / total, 0.0, 1.0))
            g[23] = float(np.clip(len(sectors) / 64.0, 0.0, 1.0))

        return {
            "agent_tokens": agent_tokens, "agent_types": np.array(
                [0 if a.missile_type == 21000 else (1 if a.missile_type == 21001 else 2)
                 for a in alive], dtype=np.int64),
            "cand_tokens": cand_tokens, "union_ids": union_ids,
            "global": g, "agent_index": agent_index,
        }


# ================= 网络 =================

class TopFormerActor(nn.Module):
    def __init__(self, cfg: TopFormerConfig):
        super().__init__()
        d = cfg.d_model
        self.agent_proj = nn.Sequential(
            nn.Linear(cfg.agent_dim, d), nn.GELU(), nn.Linear(d, d))
        self.cand_proj = nn.Sequential(
            nn.Linear(cfg.cand_dim, d), nn.GELU(), nn.Linear(d, d))
        self.deploy_cand_proj = nn.Sequential(
            nn.Linear(cfg.cand_dim, d), nn.GELU(), nn.Linear(d, d))
        self.global_proj = nn.Sequential(
            nn.Linear(cfg.global_dim, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Embedding(3, d)
        self.keep_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.keep_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_head, dim_feedforward=cfg.ff_dim,
            activation="gelu", batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.enc_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg.n_head, dim_feedforward=cfg.ff_dim,
            activation="gelu", batch_first=True, norm_first=True, dropout=0.0)
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=cfg.mat_decoder_layers)
        self.action_prefix_proj = nn.Sequential(
            nn.Linear(cfg.prefix_dim, d), nn.GELU(), nn.Linear(d, d))
        self.action_pos_emb = nn.Embedding(256, d)
        self.decision_query = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.decision_query, std=0.02)
        self.launch_head = nn.Linear(d, cfg.activation_dim)
        self.sat_head = nn.Linear(d, 2)
        self.launch_action_emb = nn.Embedding(cfg.activation_dim + 1, d)
        self.goal_kind_emb = nn.Embedding(5, d)      # none/target/search/keep/invalid
        self.goal_arg_proj = nn.Linear(d, d)
        self.goal_query = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])
        self.goal_key = nn.Linear(d, d)
        self.deploy_query = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])
        self.deploy_key = nn.Linear(d, d)
        # Appended last to keep every legacy actor parameter index stable.
        # Exact cloning guarantees the opt-in split has zero behavioural jump.
        # Restore the RNG after constructing the module so adding this dormant
        # default-off branch cannot perturb the legacy critic initialization.
        rng_state = torch.get_rng_state()
        self.l_search_query = nn.Linear(d, d)
        torch.set_rng_state(rng_state)
        self.l_search_query.load_state_dict(self.goal_query[2].state_dict())

    def encode(self, agent_tokens, agent_types, cand_tokens, glob,
               agent_pad, cand_pad, *, candidate_kind: str = "goal"):
        """返回 (agent_emb [B,N,d], cand_emb 含 KEEP 拼接 [B,M+1,d])。"""
        a = self.agent_proj(agent_tokens) + self.type_emb(agent_types)
        c = (self.deploy_cand_proj(cand_tokens)
             if candidate_kind == "deploy" else self.cand_proj(cand_tokens))
        g = self.global_proj(glob).unsqueeze(1)
        tokens = torch.cat([g, a, c], dim=1)
        pad = torch.cat([
            torch.zeros(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device),
            agent_pad, cand_pad], dim=1)
        out = self.encoder(tokens, src_key_padding_mask=pad)
        ga = out[:, 1:1 + agent_tokens.shape[1]]
        gc = out[:, 1 + agent_tokens.shape[1]:]
        keep = self.keep_token.view(1, 1, -1).expand(gc.shape[0], -1, -1)
        return ga, torch.cat([gc, keep], dim=1)

    def condition_on_prefix(self, h, memory, memory_pad, prefix_actions,
                            prefix_pad):
        """MAT-style causal decoder for the current conditional action token.

        ``prefix_actions`` contains only actions already emitted in this event.
        The current query is appended on the right, and a strict causal mask
        prevents any token from reading a future action during teacher replay.
        """
        batch, prefix_len = prefix_actions.shape[:2]
        if prefix_len:
            prefix = self.action_prefix_proj(prefix_actions)
            positions = torch.arange(
                prefix_len, device=h.device).clamp(max=255)
            prefix = prefix + self.action_pos_emb(positions).unsqueeze(0)
        else:
            prefix = h.new_zeros((batch, 0, h.shape[-1]))
        # Padded minibatches can contain different prefix lengths.  The rollout
        # query used its own prefix length, so replay must not use the batch-wide
        # maximum position for every sample.
        query_pos = (~prefix_pad).sum(dim=1).clamp(max=255).to(torch.long)
        query = h + self.decision_query + self.action_pos_emb(query_pos)
        target = torch.cat([prefix, query.unsqueeze(1)], dim=1)
        target_pad = torch.cat([
            prefix_pad,
            torch.zeros(batch, 1, dtype=torch.bool, device=h.device),
        ], dim=1)
        causal = torch.triu(torch.ones(
            target.shape[1], target.shape[1], dtype=torch.bool, device=h.device
        ), diagonal=1)
        decoded = self.action_decoder(
            target, memory, tgt_mask=causal,
            tgt_key_padding_mask=target_pad,
            memory_key_padding_mask=memory_pad,
        )
        return decoded[:, -1]

    def after_launch(self, h, launch_action, launch_valid):
        index = torch.where(
            launch_valid,
            launch_action.clamp(min=0, max=self.launch_head.out_features - 1),
            torch.full_like(launch_action, self.launch_head.out_features),
        )
        return h + self.launch_action_emb(index)

    def after_goal(self, h, goal_kind, goal_arg_emb, goal_valid):
        kind = torch.where(
            goal_valid,
            goal_kind.clamp(min=0, max=3),
            torch.zeros_like(goal_kind),
        )
        arg = self.goal_arg_proj(goal_arg_emb) * goal_valid.float().unsqueeze(-1)
        return h + self.goal_kind_emb(kind) + arg


class TopFormerCritic(nn.Module):
    """条件价值基线：场景池化特征 → V（候选饱和等前缀信息已在 token 内）。"""

    def __init__(self, cfg: TopFormerConfig):
        super().__init__()
        d = cfg.d_model
        self.agent_proj = nn.Sequential(nn.Linear(cfg.agent_dim, d), nn.GELU())
        self.cand_proj = nn.Sequential(nn.Linear(cfg.cand_dim, d), nn.GELU())
        self.prefix_proj = nn.Sequential(nn.Linear(cfg.prefix_dim, d), nn.GELU())
        self.trunk = nn.Sequential(
            nn.Linear(3 * d + cfg.global_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, agent_tokens, cand_tokens, glob, agent_pad, cand_pad,
                prefix_actions=None, prefix_pad=None):
        inv_a = (~agent_pad).float().unsqueeze(-1)
        inv_c = (~cand_pad).float().unsqueeze(-1)
        pa = (self.agent_proj(agent_tokens) * inv_a).sum(1) / inv_a.sum(1).clamp(min=1.0)
        pc = (self.cand_proj(cand_tokens) * inv_c).sum(1) / inv_c.sum(1).clamp(min=1.0)
        if prefix_actions is None or prefix_actions.shape[1] == 0:
            pp = pa.new_zeros(pa.shape)
        else:
            if prefix_pad is None:
                prefix_pad = torch.zeros(
                    prefix_actions.shape[:2], dtype=torch.bool,
                    device=prefix_actions.device)
            inv_p = (~prefix_pad).float().unsqueeze(-1)
            pp = (self.prefix_proj(prefix_actions) * inv_p).sum(1) \
                / inv_p.sum(1).clamp(min=1.0)
        return self.trunk(torch.cat([pa, pc, pp, glob], dim=-1)).squeeze(-1)


# ================= 决策记录 =================

@dataclass
class MacroRecord:
    step: int
    agent_id: int
    agent_row: int                    # scene 中该弹的行号（重放时取 embedding）
    missile_type: int                 # 0/1/2
    agent_tokens: np.ndarray          # [N,24] 快照
    agent_types: np.ndarray           # [N]
    cand_tokens: np.ndarray           # [M,15] 快照（并集）
    glob: np.ndarray                  # [global_dim]
    prefix_actions: np.ndarray        # [P,prefix_dim], rollout 时的因果前缀
    cand_rows: np.ndarray             # 该弹合法候选在并集中的行号 + KEEP(-1→末位)
    goal_mask: np.ndarray             # [10] 与智能体槽位协议对齐
    launch_mask: np.ndarray           # [activation_dim]
    launch_context: int               # 实际激活语义；-1=本事件无激活字段
    goal_kind: int                    # 0 none / 1 target / 2 search / 3 keep
    goal_arg_row: int                 # scene candidate row; KEEP = local M
    launch_action: int | None
    goal_action: int | None           # 槽位索引（10 内）；KEEP=9
    launch_old_lp: float
    goal_old_lp: float
    value: float
    ret: float = 0.0                  # SMDP 段回报（official + PBRS）
    duration: int = 1                 # 与下一事件的真实仿真步数
    next_value: float = 0.0
    terminal: bool = False
    local_credit: float = 0.0         # 真实命中源局部信用，不改写 team reward
    discovery_credit: float = 0.0     # 该搜索动作解锁的后续真实官方分（已缩放）
    enabled_target_ids: tuple[int, ...] = ()
    discovery_credit_by_target: tuple[tuple[int, float], ...] = ()
    p1_search_phase: bool = False      # 采样前已确定；不由结果反推
    event_id: int = 0
    order_index: int = 0
    sat_action: int | None = None     # 卫星头动作（1=请求窗口；None=未激活）
    sat_old_lp: float = 0.0
    episode: int = 0                  # 回合标识（轨迹链分组）
    mc: float = 0.0                   # return-to-go（更新时由 ret 幂等计算）


# ================= 策略主体 =================

class TopFormerMAPPO:
    """顶层 Transformer 策略（兼容 UnifiedAttackMissileAgent / TrainingEnv 生命周期）。"""

    def __init__(self, config: TopFormerConfig | None = None,
                 initial_targets: tuple[TargetPrior, ...] = (),
                 total_platforms: int = 1,
                 search_bounds: tuple[float, float, float, float] | None = None):
        self.config = config or TopFormerConfig()
        if self.config.agent_dim < 54 or self.config.global_dim < 24:
            raise ValueError("TopFormer v3 requires agent_dim>=54 and global_dim>=24")
        if self.config.prefix_dim != 31:
            raise ValueError("TopFormer v3 prefix_dim must be 31")
        if self.config.activation_dim != 2:
            raise ValueError("TopFormer v3 activation_dim must be FIRE/HOLD (2)")
        if self.config.update_every_rounds <= 0:
            raise ValueError("update_every_rounds must be positive")
        if self.config.min_update_records <= 0:
            raise ValueError("min_update_records must be positive")
        self.config.train_stage = str(self.config.train_stage).strip().lower()
        if self.config.train_stage not in {"joint", "p1_scout"}:
            raise ValueError(
                "TOPFORMER_TRAIN_STAGE must be 'joint' or 'p1_scout'")
        if not (0.0 <= float(self.config.gae_lambda) <= 1.0):
            raise ValueError("GAE_LAMBDA must be in [0, 1]")
        if float(self.config.discovery_credit_coef) < 0.0:
            raise ValueError("DISCOVERY_CREDIT_COEF must be non-negative")
        if float(self.config.p1_discovery_aux_coef) < 0.0:
            raise ValueError("P1_DISCOVERY_AUX_COEF must be non-negative")
        if self.config.p1_split_search_query and not self.config.p1_search_only_actor:
            raise ValueError(
                "P1_SPLIT_SEARCH_QUERY requires P1_SEARCH_ONLY_ACTOR=1")
        if (
            (self.config.p1_search_only_actor
             or self.config.p1_split_search_query
             or self.config.p1_discovery_aux_coef > 0.0)
            and self.config.train_stage != "p1_scout"
        ):
            raise ValueError("P1.1 switches require TOPFORMER_TRAIN_STAGE=p1_scout")
        if self.config.l_scout_count < 0:
            raise ValueError("L_SCOUT_COUNT must be non-negative")
        if self.config.l_scout_expected_targets < 0:
            raise ValueError("L_SCOUT_EXPECTED_TARGETS must be non-negative")
        if not (0.0 <= self.config.l_scout_coverage_threshold <= 1.0):
            raise ValueError("L_SCOUT_COVERAGE_THRESHOLD must be in [0, 1]")
        if self.config.l_scout_quiet_steps < 0:
            raise ValueError("L_SCOUT_QUIET_STEPS must be non-negative")
        if self.config.l_scout_deadline < 0:
            raise ValueError("L_SCOUT_DEADLINE must be non-negative")
        if self.config.l_attack_min_discovered < 0:
            raise ValueError("L_ATTACK_MIN_DISCOVERED must be non-negative")
        if self.config.l_scout_sector_dwell <= 0:
            raise ValueError("L_SCOUT_SECTOR_DWELL must be positive")
        if self.config.train_stage == "p1_scout":
            if not self.config.l_scout_enabled:
                raise ValueError(
                    "p1_scout requires L_SCOUT_ENABLED=1")
            if self.config.l_scout_count <= 0:
                raise ValueError(
                    "p1_scout requires L_SCOUT_COUNT > 0")
            if self.config.expert_rounds:
                raise ValueError(
                    "p1_scout requires EXPERT_ROUNDS=0; P1 records PPO L-goal "
                    "actions only")
        if (
            self.config.expert_rounds < 0
            or (
                self.config.expert_rounds
                and self.config.expert_rounds
                % self.config.update_every_rounds != 0
            )
        ):
            raise ValueError(
                "expert_rounds must be 0 or a positive multiple of "
                "update_every_rounds"
            )
        self.initial_targets = tuple(initial_targets)
        self.total_platforms = max(int(total_platforms), 1)
        self.coordinate_scale = infer_coordinate_scale(self.initial_targets)
        self.goal_slots = max(1, int(self.config.max_goal_slots))
        self.search_count = max(
            1, int(self.config.search_rows) * int(self.config.search_cols))
        self.sector_index_0 = self.goal_slots
        self.keep_index = self.goal_slots + self.search_count
        self.target_dim = self.keep_index + 1
        self.search_bounds = search_bounds
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        device = self.config.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cpu":
            # Large host defaults (44 threads here) make these small,
            # sequential Transformer calls over 100x slower from scheduling
            # overhead.  Respect an explicit setting; otherwise cap only
            # PyTorch's intra-op pool without changing model semantics.
            requested_threads = os.getenv("TOPFORMER_CPU_THREADS")
            if requested_threads is not None:
                cpu_threads = max(1, int(requested_threads))
                torch.set_num_threads(cpu_threads)
            elif "OMP_NUM_THREADS" not in os.environ:
                cpu_threads = min(torch.get_num_threads(), 8)
                torch.set_num_threads(cpu_threads)
            logger.info(
                "[TopFormer] device=%s | CPU intra-op=%d inter-op=%d",
                self.device, torch.get_num_threads(),
                torch.get_num_interop_threads(),
            )
        self.actor = TopFormerActor(self.config).to(self.device)
        self.critic = TopFormerCritic(self.config).to(self.device)
        self._configure_actor_trainability()
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), self.config.learning_rate)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), self.config.critic_lr)

        self.info_center = InfoCenter(
            self.initial_targets,
            self.total_platforms,
            coordinate_scale=self.coordinate_scale,
            search_bounds=self.search_bounds,
            search_rows=self.config.search_rows,
            search_cols=self.config.search_cols,
            l_scout_enabled=self.config.l_scout_enabled,
            l_scout_count=self.config.l_scout_count,
            l_scout_expected_targets=self.config.l_scout_expected_targets,
            l_scout_coverage_threshold=(
                self.config.l_scout_coverage_threshold),
            l_scout_quiet_steps=self.config.l_scout_quiet_steps,
            l_scout_deadline=self.config.l_scout_deadline,
        )
        self.tokenizer = SceneTokenizer(
            self.config.max_steps,
            coordinate_scale=self.coordinate_scale,
            agent_dim=self.config.agent_dim,
            global_dim=self.config.global_dim,
        )
        self._agents: list = []                 # 智能体对象注册（token 构建用）
        self._entity_to_agent: dict[int, int] = {}
        self._pending: dict[int, dict] = {}     # agent_id → 本次决策上下文
        self._sat_granted: set[int] = set()
        self._sat_requests: list[int] = []   # 当步待结算的天眼请求
        self._deploy_records: list[dict] = []  # 部署决策（episode 级回报）
        self._wave_pending: list[tuple[int, int]] = []  # (agent_id, step) 待结算发射
        # 奖励分量记账：raw(击杀+削血=官方计分的奖励形态) / 启发式塑形各项
        self._comp_raw = 0.0; self._comp_discovery = 0.0
        self._comp_retarget = 0.0; self._comp_wave = 0.0
        self._comp_progress = 0.0; self._comp_official = 0.0
        self._comp_causal_credit = 0.0
        self._comp_discovery_credit_raw = 0.0
        self._episode_official = 0.0
        self._episode_team_total = 0.0       # 回合团队标量累计（部署信用）
        self._records: list[MacroRecord] = []
        self._open_ret: dict[int, float] = {}   # agent_id → 段内已累积团队标量
        self._open_credit: dict[int, float] = {}
        self._decision_prefix: list[np.ndarray] = []
        self._decision_step = -1
        self._decision_order: tuple[int, ...] = ()
        self._decision_order_index: dict[int, int] = {}
        self._current_step = 0
        self._event_id = 0
        self._event_history = {
            "official_delta": 0.0, "event_age": self.config.max_steps,
            "launches": 0, "targets": 0, "searches": 0, "satellite": 0,
        }
        self._current_event_summary = {
            "launches": 0, "targets": 0, "searches": 0, "satellite": 0,
        }
        self._progress_balance: dict[int, float] = {}
        self._coverage_potential: dict[int, float] = {}
        self._reward_ledger = {
            "official": 0.0, "progress_pbrs": 0.0,
            "coverage_pbrs": 0.0, "causal_credit": 0.0,
            "discovery_enable_credit": 0.0,
        }
        # target_id -> (the exact L-search record that enabled it,
        #               legal discovery step, discoverer entity id).
        # The MacroRecord reference deliberately survives segment closure, so
        # later target damage is written back to the action that discovered it
        # instead of whichever action the scout happens to execute later.
        self._discovery_credit_links: dict[
            int, tuple[MacroRecord, int, int]
        ] = {}
        self._discovery_event_cursor = 0
        self._ep_discovery_links = 0
        self._ep_discovery_unbound = 0
        self._ep_discovery_credited_targets: set[int] = set()
        self._ep_discovery_credit_raw = 0.0
        self._window_discovery_links = 0
        self._window_discovery_unbound = 0
        self._window_discovery_credited_targets: set[tuple[int, int]] = set()
        self._rounds_done = 0
        self._expert_bc_count = 0        # 已完成的 BC 更新次数
        self._g_run_mu = 0.0; self._g_run_n = 0; self._g_run_m2 = 0.0  # G 运行统计
        self._updates_window_rounds = 0   # 当前更新窗口内回合数（分量日志分母）
        self.training = True
        self.update_count = 0
        self.best_score = float("-inf")
        self.last_metrics: dict[str, float] = {}
        # 回合执行指标（episode_metrics 用，字段对齐 unified_metrics）
        self._ep_launch_steps: list[int] = []
        self._ep_launch_by_type: dict[int, int] = {}
        self._ep_goal_choices: dict[int, int] = {}
        self._ep_assigned_damage: dict[int, float] = {}
        self._ep_overkill: dict[int, int] = {}
        self._ep_retarget_delays: list[int] = []
        self._ep_search_actions = 0
        # 宏事件诊断：记录爆炸/熵坍缩排查（launch vs goal 触发源计数）
        self._ep_events_launch = 0
        self._ep_events_goal = 0
        self._ep_events_by_agent: dict[int, int] = {}
        self._ep_p1_goal_records = 0
        self._ep_p1_filtered_events = 0
        self._p1_window_goal_records = 0
        self._p1_window_filtered_events = 0

    # ---------- 注册与生命周期 ----------

    def register_entity_agent(self, entity_id, agent_id):
        self._entity_to_agent[int(entity_id)] = int(agent_id)

    def register_agent_object(self, agent):
        """main.py 在创建智能体后注册（token 构建需要智能体状态）。"""
        self._agents.append(agent)

    def set_training(self, training: bool):
        self.training = bool(training)
        self.actor.train(self.training)
        self.critic.train(self.training)
        if not training:
            self._records.clear()
            self._pending.clear()

    def _configure_actor_trainability(self) -> None:
        """Apply the stage-specific actor freeze without freezing the critic."""

        p1_scout = self.config.train_stage == "p1_scout"
        for parameter in self.actor.parameters():
            parameter.requires_grad_(not p1_scout)
        if p1_scout:
            # Type index 2 is L throughout TopFormer.  Keeping the shared
            # encoder/key/decoder frozen makes this a controlled attribution
            # experiment: only the L destination query can change.
            query = (
                self.actor.l_search_query
                if self.config.p1_split_search_query
                else self.actor.goal_query[2]
            )
            for parameter in query.parameters():
                parameter.requires_grad_(True)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        self._trainable_actor_parameter_names = tuple(
            name for name, parameter in self.actor.named_parameters()
            if parameter.requires_grad
        )
        if p1_scout:
            logger.info(
                "[TopFormer] P1 scout freeze: actor trainable=%s; critic=all",
                ",".join(self._trainable_actor_parameter_names),
            )

    @property
    def p1_scout_settings(self) -> dict[str, object]:
        """Stable, InfoCenter-independent view of the P1 runtime contract."""

        return {
            "enabled": bool(self.config.l_scout_enabled),
            "count": int(self.config.l_scout_count),
            "expected_targets": int(self.config.l_scout_expected_targets),
            "coverage_threshold": float(
                self.config.l_scout_coverage_threshold),
            "quiet_steps": int(self.config.l_scout_quiet_steps),
            "deadline": int(self.config.l_scout_deadline),
            "attack_min_discovered": int(
                self.config.l_attack_min_discovered),
            "search_only_actor": bool(
                self.config.p1_search_only_actor),
            "split_search_query": bool(
                self.config.p1_split_search_query),
            "discovery_aux_coef": float(
                self.config.p1_discovery_aux_coef),
            "sector_dwell": int(self.config.l_scout_sector_dwell),
        }

    @property
    def mat_sequential(self) -> bool:
        return True

    def begin_decision_phase(self, step: int,
                             agent_ids: tuple[int, ...]) -> tuple[int, ...]:
        """Start one causal action sequence and return its replayable order.

        Training randomizes both type blocks and agents inside a block, avoiding
        permanent first/last-agent roles.  Evaluation uses a deterministic
        type/id order.  Only macro-active agents append a prefix token, so an
        ordinary motion-only step remains cheap.
        """
        self._decision_step = int(step)
        self._current_step = int(step)
        self._event_id += 1
        self._decision_prefix = []
        self._current_event_summary = {
            "launches": 0, "targets": 0, "searches": 0, "satellite": 0,
        }
        ids = [int(value) for value in agent_ids]
        by_type: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
        objects = {int(a.agent_id): a for a in self._agents}
        for aid in ids:
            missile_type = int(getattr(objects.get(aid), "missile_type", -1))
            ti = 0 if missile_type == 21000 else (
                1 if missile_type == 21001 else 2 if missile_type == 21002 else 3)
            by_type[ti].append(aid)
        if self.training:
            blocks = [key for key, values in by_type.items() if values]
            np.random.shuffle(blocks)
            order: list[int] = []
            for key in blocks:
                values = np.asarray(by_type[key], dtype=np.int64)
                np.random.shuffle(values)
                order.extend(int(value) for value in values)
        else:
            order = [aid for key in range(4) for aid in sorted(by_type[key])]
        # The satellite request is a team action.  Its sole eligible H decider
        # is decoded last so it can condition on the complete per-agent prefix.
        satellite_deciders = [
            aid for aid in order
            if self.is_satellite_decider(int(getattr(objects.get(aid), "entity_id", -1)))
        ]
        if satellite_deciders:
            decider = satellite_deciders[0]
            order = [aid for aid in order if aid != decider] + [decider]
        self._decision_order = tuple(order)
        self._decision_order_index = {
            aid: index for index, aid in enumerate(self._decision_order)
        }
        return self._decision_order

    def end_decision_phase(self, step: int) -> None:
        if int(step) != self._decision_step:
            logger.warning(
                "[TopFormer] decision phase step mismatch: begin=%d end=%d",
                self._decision_step, int(step))
        self._decision_prefix = []
        self._decision_step = -1
        self._event_history.update(self._current_event_summary)

    def _prefix_snapshot(self) -> np.ndarray:
        if not self._decision_prefix:
            return np.zeros((0, self.config.prefix_dim), dtype=np.float32)
        return np.stack(self._decision_prefix).astype(np.float32, copy=True)

    def _action_prefix_token(self, *, missile_type: int,
                             launch_action: int | None,
                             goal_action: int | None,
                             cand_rows: np.ndarray | None,
                             scene: dict, sat_action: int | None,
                             agent_id: int) -> np.ndarray:
        token = np.zeros(self.config.prefix_dim, dtype=np.float32)
        ti = 0 if missile_type == 21000 else (1 if missile_type == 21001 else 2)
        token[ti] = 1.0
        if (launch_action is not None
                and 0 <= int(launch_action) < self.config.activation_dim):
            token[3 + int(launch_action)] = 1.0
            token[7] = 1.0
        goal_kind = 0
        if goal_action is not None:
            slot = int(goal_action)
            if slot < self.goal_slots:
                goal_kind = 1
            elif slot < self.keep_index:
                goal_kind = 2
            else:
                goal_kind = 3
            token[8 + goal_kind] = 1.0
            if cand_rows is not None and 0 <= slot < len(cand_rows):
                row = int(cand_rows[slot])
                candidates = scene["cand_tokens"]
                if 0 <= row < len(candidates):
                    token[15:30] = candidates[row]
        if sat_action is not None and 0 <= int(sat_action) < 2:
            token[12 + int(sat_action)] = 1.0
            token[14] = 1.0
        denominator = max(len(self._decision_order) - 1, 1)
        token[30] = float(
            self._decision_order_index.get(int(agent_id), 0) / denominator)
        return token

    # ---------- 部署头（3×8×4 计数矩阵的逐弹格子采样） ----------

    def _deploy_scene(self, coordinates, init_observation,
                      occupancy: np.ndarray | None = None):
        """部署期场景：3 个类型 token + 32 个格子候选 token（15 维）。

        部署候选使用独立投影；信息源仅为蓝方战前目录、部署区几何和本事件
        已占用格子。
        """
        import math as _math
        pts = coordinates[0]
        west, east = min(p[0] for p in pts), max(p[0] for p in pts)
        south, north = min(p[1] for p in pts), max(p[1] for p in pts)
        # 蓝方目标聚类（战前目录）
        clusters = {9400: [0.0, 0.0, 0], 9600: [0.0, 0.0, 0], 9500: [0.0, 0.0, 0]}
        for tgt in self.initial_targets:
            c = clusters.get(tgt.entity_type)
            if c:
                c[0] += tgt.position.lon; c[1] += tgt.position.lat; c[2] += 1
        centers = {t: ((c[0]/c[2], c[1]/c[2]) if c[2] else
                       ((west+east)/2, (south+north)/2))
                   for t, c in clusters.items()}
        cells = np.zeros((32, 15), dtype=np.float32)
        for row in range(8):
            for col in range(4):
                lon = west + (col + 0.5) * (east - west) / 4.0
                # compile_actions uses row 0 at the north edge.  Encoding the
                # same convention is essential: otherwise deployment learning
                # observes a vertically mirrored action outcome.
                lat = north - (row + 0.5) * (north - south) / 8.0
                t = cells[row * 4 + col]
                t[3] = 1.0                              # 格子标记（复用扇区位）
                t[6] = _norm(lon * self.coordinate_scale, LON_MIN, LON_MAX)
                t[7] = _norm(lat * self.coordinate_scale, LAT_MIN, LAT_MAX)
                t[4] = float(np.clip(row / 7.0, 0, 1))  # 位置特征复用
                for k, (ttype, (clon, clat)) in enumerate(centers.items()):
                    d = _math.hypot(
                        (clon - lon) * 96.0 * self.coordinate_scale,
                        (clat - lat) * 110.0 * self.coordinate_scale,
                    )
                    t[8 + k] = float(np.clip(d / DIST_MAX_KM, 0.0, 1.0))
                t[13] = float(np.clip(
                    (east - west) * 96.0 * self.coordinate_scale / 50.0,
                    0,
                    1,
                ))
                if occupancy is not None:
                    total = max(float(np.asarray(occupancy).sum()), 1.0)
                    t[11] = float(np.clip(occupancy[row * 4 + col] / 8.0, 0.0, 1.0))
                    t[12] = float(np.clip(occupancy[row * 4 + col] / total, 0.0, 1.0))
        agent_tokens = np.zeros((3, self.config.agent_dim), dtype=np.float32)
        agent_types = np.arange(3, dtype=np.int64)
        for ti in range(3):
            agent_tokens[ti, ti] = 1.0
        g = np.zeros(self.config.global_dim, dtype=np.float32)
        g[10] = 1.0
        if self.config.global_dim >= 24 and occupancy is not None:
            g[12] = float(np.clip(np.count_nonzero(occupancy) / 32.0, 0.0, 1.0))
            g[13] = float(np.clip(np.max(occupancy) / 8.0, 0.0, 1.0))
        return {"agent_tokens": agent_tokens, "agent_types": agent_types,
                "cand_tokens": cells, "glob": g}

    def build_deploy_plan(self, *, coordinates, coordinates_hm,
                          expected_counts, init_observation):
        """逐弹采样格子 → 3×8×4 计数矩阵（合法性由构造保证：每弹恰选一格）。

        训练态记录每条采样（episode 级回报在终局回填）；评估态取每型
        分布的前 N 大格子贪心填充（去采样噪声）。"""
        occupancy = np.zeros(32, dtype=np.float32)
        # 专家部署：H 靠母巢质心、M 靠哨卫质心、L 均匀铺开（E01 先验：
        # 高价值目标近距进近缩短暴露；L 需扇区覆盖）
        if self.training and self._rounds_done < self.config.expert_rounds:
            import math as _m
            pts = coordinates[0]
            west, east = min(p[0] for p in pts), max(p[0] for p in pts)
            south, north = min(p[1] for p in pts), max(p[1] for p in pts)
            pref = {0: 9400, 1: 9600, 2: 9500}
            counts = {}
            expert_placements: list[tuple[int, int, int]] = []
            for type_code, missile_type in enumerate((21000, 21001, 21002)):
                n = int(expected_counts[missile_type])
                if n <= 0:
                    counts[str(missile_type)] = [[0]*4 for _ in range(8)]
                    continue
                cl = [(t.position.lon, t.position.lat) for t in self.initial_targets
                      if t.entity_type == pref[type_code]]
                if not cl:
                    cl = [((west+east)/2, (south+north)/2)]
                w = []
                for row in range(8):
                    for col in range(4):
                        lon = west + (col + .5) * (east - west) / 4
                        lat = north - (row + .5) * (north - south) / 8
                        d = min(_m.hypot(
                            (cx_ - lon) * 96 * self.coordinate_scale,
                            (cy_ - lat) * 110 * self.coordinate_scale,
                        )
                                for cx_, cy_ in cl)
                        w.append(1.0 / (1.0 + d / 10.0))     # 逆距离权重
                w = np.array(w, dtype=np.float64)
                if type_code == 2:
                    w = np.ones(32)                            # L 均匀覆盖搜索区
                w = w / w.sum()
                picks = np.random.choice(32, size=n, p=w)
                matrix = [[0]*4 for _ in range(8)]
                for cell in picks:
                    cell = int(cell)
                    matrix[cell // 4][cell % 4] += 1
                    expert_placements.append((type_code, missile_type, cell))
                counts[str(missile_type)] = matrix
            # Preserve the scripted expert plan while storing the exact
            # sequential scenes/actions needed to behaviour-clone PLACE.
            np.random.shuffle(expert_placements)
            prefix: list[np.ndarray] = []
            for order_index, (type_code, _missile_type, cell) in enumerate(
                    expert_placements):
                scene = self._deploy_scene(
                    coordinates, init_observation, occupancy=occupancy)
                prefix_np = (
                    np.stack(prefix).astype(np.float32)
                    if prefix
                    else np.zeros((0, self.config.prefix_dim), np.float32)
                )
                self._deploy_records.append({
                    "episode": self._rounds_done,
                    "scene": scene,
                    "prefix_actions": prefix_np,
                    "type": type_code,
                    "cell": cell,
                    "expert": True,
                    "order_index": order_index,
                })
                action_token = np.zeros(self.config.prefix_dim, np.float32)
                action_token[type_code] = 1.0
                action_token[8] = 1.0
                action_token[15:30] = scene["cand_tokens"][cell]
                action_token[30] = order_index / max(
                    len(expert_placements) - 1, 1)
                prefix.append(action_token)
                occupancy[cell] += 1.0
            self._last_deploy_learned = False
            return {"version": 1, "counts": counts}
        counts = {
            str(missile_type): [[0] * 4 for _ in range(8)]
            for missile_type in (21000, 21001, 21002)
        }
        placements = [
            (type_code, missile_type)
            for type_code, missile_type in enumerate((21000, 21001, 21002))
            for _ in range(int(expected_counts[missile_type]))
        ]
        if self.training:
            np.random.shuffle(placements)
        prefix: list[np.ndarray] = []
        with torch.no_grad():
            for order_index, (type_code, missile_type) in enumerate(placements):
                scene = self._deploy_scene(
                    coordinates, init_observation, occupancy=occupancy)
                at = torch.as_tensor(
                    scene["agent_tokens"], device=self.device).unsqueeze(0)
                atty = torch.as_tensor(
                    scene["agent_types"], device=self.device).unsqueeze(0)
                ct = torch.as_tensor(
                    scene["cand_tokens"], device=self.device).unsqueeze(0)
                gl = torch.as_tensor(scene["glob"], device=self.device).unsqueeze(0)
                apad = torch.zeros_like(at[..., 0], dtype=torch.bool)
                cpad = torch.zeros_like(ct[..., 0], dtype=torch.bool)
                ga, gc = self.actor.encode(
                    at, atty, ct, gl, apad, cpad, candidate_kind="deploy")
                prefix_np = (np.stack(prefix).astype(np.float32)
                             if prefix else np.zeros(
                                 (0, self.config.prefix_dim), np.float32))
                prefix_t = torch.as_tensor(
                    prefix_np, device=self.device).unsqueeze(0)
                prefix_pad = torch.zeros(
                    (1, len(prefix_np)), dtype=torch.bool, device=self.device)
                memory = torch.cat([ga, gc], dim=1)
                memory_pad = torch.cat([
                    apad, cpad,
                    torch.zeros((1, 1), dtype=torch.bool, device=self.device),
                ], dim=1)
                h = self.actor.condition_on_prefix(
                    ga[:, type_code], memory, memory_pad, prefix_t, prefix_pad)[0]
                keys = self.actor.deploy_key(gc[0])[:32]
                q = self.actor.deploy_query[type_code](h)
                logits = (q @ keys.T) / (self.config.d_model ** 0.5)
                dist = torch.distributions.Categorical(logits=logits)
                cell = int(dist.sample().item()) if (
                    self.training or self.config.eval_sample
                ) else int(torch.argmax(logits).item())
                old_lp = float(dist.log_prob(torch.as_tensor(
                    cell, device=self.device)).item())
                if self.training:
                    self._deploy_records.append({
                        "episode": self._rounds_done,
                        "scene": scene,
                        "prefix_actions": prefix_np,
                        "type": type_code,
                        "cell": cell,
                        "old_lp": old_lp,
                        "order_index": order_index,
                    })
                counts[str(missile_type)][cell // 4][cell % 4] += 1
                action_token = np.zeros(self.config.prefix_dim, np.float32)
                action_token[type_code] = 1.0
                action_token[8] = 1.0                 # PLACE operation
                action_token[15:30] = scene["cand_tokens"][cell]
                action_token[30] = order_index / max(len(placements) - 1, 1)
                prefix.append(action_token)
                occupancy[cell] += 1.0
        # validate_plan 严格校验键集：标记通过实例属性传递
        self._last_deploy_learned = bool(self.training)
        return {"version": 1, "counts": counts}

    def _deploy_logits_for_records(self, records: list[dict]) -> torch.Tensor:
        """Replay stored placements as one padded batch.

        Deployment is autoregressive only while sampling.  During PPO/BC the
        complete scene and action prefix are already frozen in every record,
        so encoding records one-by-one is mathematically unnecessary; batching
        preserves every prefix while avoiding repeated encoder launch overhead.
        """

        if not records:
            return torch.empty((0, 32), device=self.device)
        at = torch.as_tensor(np.stack([
            record["scene"]["agent_tokens"] for record in records
        ]), device=self.device)
        atty = torch.as_tensor(np.stack([
            record["scene"]["agent_types"] for record in records
        ]), dtype=torch.long, device=self.device)
        ct = torch.as_tensor(np.stack([
            record["scene"]["cand_tokens"] for record in records
        ]), device=self.device)
        gl = torch.as_tensor(np.stack([
            record["scene"]["glob"] for record in records
        ]), device=self.device)
        apad = torch.zeros_like(at[..., 0], dtype=torch.bool)
        cpad = torch.zeros_like(ct[..., 0], dtype=torch.bool)
        ga, gc = self.actor.encode(
            at, atty, ct, gl,
            agent_pad=apad, cand_pad=cpad, candidate_kind="deploy",
        )
        prefixes = []
        for record in records:
            stored = record.get("prefix_actions")
            if stored is None:
                stored = np.zeros((0, self.config.prefix_dim), np.float32)
            prefixes.append(np.asarray(stored, dtype=np.float32))
        max_prefix = max(len(prefix) for prefix in prefixes)
        prefix_np = np.zeros(
            (len(records), max_prefix, self.config.prefix_dim), np.float32)
        prefix_pad_np = np.ones((len(records), max_prefix), dtype=bool)
        for index, prefix in enumerate(prefixes):
            prefix_np[index, :len(prefix)] = prefix
            prefix_pad_np[index, :len(prefix)] = False
        prefix_t = torch.as_tensor(prefix_np, device=self.device)
        prefix_pad = torch.as_tensor(
            prefix_pad_np, dtype=torch.bool, device=self.device)
        memory = torch.cat([ga, gc], dim=1)
        memory_pad = torch.cat([
            apad, cpad,
            torch.zeros(
                (len(records), 1), dtype=torch.bool, device=self.device),
        ], dim=1)
        ti = torch.as_tensor(
            [int(record["type"]) for record in records],
            dtype=torch.long, device=self.device,
        )
        batch_index = torch.arange(len(records), device=self.device)
        h = self.actor.condition_on_prefix(
            ga[batch_index, ti], memory, memory_pad, prefix_t, prefix_pad)
        keys = self.actor.deploy_key(gc[:, :32])
        all_queries = torch.stack([
            head(h) for head in self.actor.deploy_query
        ], dim=0)
        q = all_queries[ti, batch_index]
        return (q.unsqueeze(1) * keys).sum(-1) / (
            self.config.d_model ** 0.5)

    def _deploy_logits_for_record(self, record: dict) -> torch.Tensor:
        """Compatibility wrapper for focused tools/tests."""

        return self._deploy_logits_for_records([record])[0]

    def _deploy_policy_loss(self):
        """Replay sequential placements without changing the actor snapshot."""
        recs = self._deploy_records
        if not recs:
            return None, None
        complete = [r for r in recs if "ret" in r]
        if not complete:
            return None, None
        episode_returns = {
            int(r["episode"]): float(r["ret"])
            for r in complete
        }
        values = np.asarray(list(episode_returns.values()), dtype=np.float32)
        mean = float(values.mean())
        std = float(values.std())
        advantages = {
            episode: (value - mean) / max(std, 1.0)
            for episode, value in episode_returns.items()
        }
        logits = self._deploy_logits_for_records(complete)
        lsm = torch.log_softmax(logits, dim=-1)
        cells = torch.as_tensor(
            [int(record["cell"]) for record in complete],
            dtype=torch.long, device=self.device,
        )
        lp_new = lsm.gather(1, cells.unsqueeze(1)).squeeze(1)
        old_lp = torch.as_tensor(
            [float(record["old_lp"]) for record in complete],
            dtype=torch.float32, device=self.device,
        )
        adv = torch.as_tensor([
            advantages[int(record["episode"])] for record in complete
        ], dtype=torch.float32, device=self.device)
        ratio = torch.exp(lp_new - old_lp)
        policy_losses = -torch.min(
            ratio * adv,
            torch.clamp(
                ratio,
                1 - self.config.clip_ratio,
                1 + self.config.clip_ratio,
            ) * adv,
        )
        entropies = -(lsm.exp() * lsm).sum(-1) / np.log(32.0)
        loss = policy_losses.mean()
        loss = loss - self.config.entropy_coef * entropies.mean()
        replay_gap = float((lp_new.detach() - old_lp).abs().max().item())
        metrics = {
            "deploy_records": float(len(complete)),
            "deploy_episodes": float(len(episode_returns)),
            "deploy_return_mean": mean,
            "deploy_entropy": float(entropies.detach().mean().item()),
            "deploy_replay_gap": float(replay_gap),
        }
        return loss, metrics

    def _update_deploy(self):
        """Standalone deployment update retained for focused tests/tools."""
        loss, metrics = self._deploy_policy_loss()
        if loss is None:
            return None
        self.opt_actor.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.opt_actor.step()
        return metrics

    def begin_deploy_phase(self, full_observation):
        pass  # GridDeployAgent 在 reset 后通过 build_deploy_plan() 取样。

    def select_deploy(self, **kwargs):
        return None  # 旧 DeployAgent 接口保持规则回退；学习部署走网格适配器。

    def begin_observation_phase(self, full_observation):
        self._current_step = int((full_observation or {}).get(
            "step", self._current_step))
        self.info_center.begin_step(self._current_step)
        return None

    # ---------- 团队标量（SMDP 段回报的输入） ----------

    def note_team_scalar(self, team_reward: float, per_target: dict | None = None,
                         official_delta: float = 0.0,
                         source_credit: dict[int, float] | None = None):
        """Consume each reward source exactly once.

        Every open macro segment receives the complete official increment.
        Exact simulator hit sources are stored in a separate causal-credit
        channel and only sharpen the advantage; they never replace or duplicate
        the cooperative team objective.  Search feedback is a per-discoverer
        coverage potential difference, not a hidden-target discovery bounty.
        """
        # ``collect_reports`` has already appended any newly legal discoveries
        # for S_{t+1}.  Bind each one exactly once to the L search record that
        # was active before that discovery.  A shared/relayed track whose true
        # sensor was H/M intentionally has no L-search link.
        discovery_events = getattr(self.info_center, "discovery_events", ())
        new_events = discovery_events[self._discovery_event_cursor:]
        self._discovery_event_cursor = len(discovery_events)
        if self.config.train_stage == "p1_scout":
            for event in new_events:
                try:
                    target_id = int(event["target_id"])
                    discoverer_entity_id = int(
                        event["discoverer_entity_id"])
                    discovery_step = int(event["step"])
                except (KeyError, TypeError, ValueError):
                    self._ep_discovery_unbound += 1
                    self._window_discovery_unbound += 1
                    continue
                aid = self._entity_to_agent.get(discoverer_entity_id)
                record = None
                if aid is not None and aid in self._open_ret:
                    record = next((
                        candidate for candidate in reversed(self._records)
                        if int(candidate.agent_id) == int(aid)
                        and int(candidate.episode) == int(self._rounds_done)
                    ), None)
                if not (
                    record is not None
                    and int(record.missile_type) == 2
                    and int(record.goal_kind) == 2
                    # Prevent reverse causality when a target is already in
                    # S_t and a search action is sampled later in the same
                    # simulator step.
                    and int(record.step) < discovery_step
                ):
                    self._ep_discovery_unbound += 1
                    self._window_discovery_unbound += 1
                    continue
                if target_id not in self._discovery_credit_links:
                    self._discovery_credit_links[target_id] = (
                        record, discovery_step, discoverer_entity_id)
                    enabled = tuple(getattr(
                        record, "enabled_target_ids", ()))
                    if target_id not in enabled:
                        record.enabled_target_ids = enabled + (target_id,)
                    self._ep_discovery_links += 1
                    self._window_discovery_links += 1

        # Attribute only realised score earned *after* legal discovery.  This
        # channel is attached directly to the one enabling record; it does not
        # enter _open_ret, the critic target, or the recursive source-credit
        # GAE, and therefore cannot become a disguised discovery bounty.
        for raw_target_id, amount in (per_target or {}).items():
            try:
                target_id = int(raw_target_id)
                raw_amount = float(amount)
            except (TypeError, ValueError):
                continue
            link = self._discovery_credit_links.get(target_id)
            if link is None:
                continue
            record, discovery_step, _discoverer_entity_id = link
            if self._current_step <= discovery_step or raw_amount <= 0.0:
                continue
            scaled = raw_amount * self.config.reward_scale
            record.discovery_credit = float(getattr(
                record, "discovery_credit", 0.0)) + scaled
            by_target = dict(getattr(
                record, "discovery_credit_by_target", ()))
            by_target[target_id] = float(by_target.get(target_id, 0.0)) + scaled
            record.discovery_credit_by_target = tuple(sorted(
                (int(key), float(value))
                for key, value in by_target.items()
            ))
            self._ep_discovery_credited_targets.add(target_id)
            self._window_discovery_credited_targets.add((
                int(record.episode), target_id))
            self._ep_discovery_credit_raw += raw_amount
            self._comp_discovery_credit_raw += raw_amount
            self._reward_ledger["discovery_enable_credit"] += scaled

        self._comp_raw += float(team_reward)
        self._comp_official += float(official_delta)
        self._episode_official += float(official_delta)
        self._episode_team_total += float(official_delta)
        self._reward_ledger["official"] += float(official_delta)
        team_value = float(official_delta) * self.config.reward_scale
        if team_value:
            for aid in self._open_ret:
                self._open_ret[aid] += team_value
        for entity_id, amount in (source_credit or {}).items():
            aid = self._entity_to_agent.get(int(entity_id))
            if aid is None or aid not in self._open_credit:
                continue
            value = float(amount) * self.config.reward_scale
            self._open_credit[aid] += value
            self._reward_ledger["causal_credit"] += float(amount)
            self._comp_causal_credit += float(amount)

        for entity_id, cells in getattr(
                self.info_center, "step_coverage_by_entity", {}).items():
            aid = self._entity_to_agent.get(int(entity_id))
            if aid is None or aid not in self._open_ret:
                continue
            previous = self._coverage_potential.get(aid, 0.0)
            current = min(1.0, previous + len(cells) / max(self.search_count, 1))
            shaped = self.config.coverage_shaping_coef * (current - previous)
            self._coverage_potential[aid] = current
            if shaped:
                self._open_ret[aid] += shaped
                self._comp_progress += shaped
                self._reward_ledger["coverage_pbrs"] += shaped

        if official_delta or getattr(self.info_center, "step_discoveries", None):
            self._event_history["official_delta"] = float(official_delta)
            self._event_history["event_age"] = 0
        else:
            self._event_history["event_age"] = min(
                self.config.max_steps,
                int(self._event_history.get("event_age", 0)) + 1,
            )

    def note_agent_shaping(self, agent_id: int, potential_delta: float) -> None:
        """Add only policy-invariant, gamma=1 progress shaping to one segment."""

        aid = int(agent_id)
        raw = float(potential_delta)
        value = self.config.progress_shaping_coef * raw
        if self.training and aid in self._open_ret and value:
            self._open_ret[aid] += value
            self._comp_progress += value
            self._progress_balance[aid] = self._progress_balance.get(aid, 0.0) + raw
            self._reward_ledger["progress_pbrs"] += value

    # ---------- 执行期决策 ----------

    def select_decision(self, agent_id, step, obs, missile_type, *,
                        launch_active, goal_mask, goal_event=False,
                        motion_active=False, resource_active=False,
                        current_goal_slot=None):
        # 底层固定：纯机动步直接走 scripted 路径；卫星按稀疏资源时钟决策。
        # （宏决策才触发 token 化与编码——事件门控由智能体侧保证）
        resource_trainable = bool(resource_active and self.config.sat_trainable)
        if not (launch_active or goal_event or resource_trainable):
            return {"launch": None, "goal": None,
                    "motion": MOTION_AUTOGUIDE, "resource": RESOURCE_OFF}

        # ── 专家先验阶段：专家规则替代网络采样，记录 (场景, 专家动作) 供 BC ──
        if (self.training and self._rounds_done < self.config.expert_rounds):
            ti_e = 0 if missile_type == 21000 else (1 if missile_type == 21001 else 2)
            return self._expert_decision(
                agent_id, step, missile_type, ti_e, launch_active,
                goal_mask, goal_event, current_goal_slot,
                resource_active=resource_trainable)

        if not self._agents:
            self._agents.append(_AgentStub.from_obs(agent_id, obs, missile_type))
        ti = 0 if missile_type == 21000 else (1 if missile_type == 21001 else 2)
        launch_trainable = self.config.upper_phase == "joint"
        prefix_np = self._prefix_snapshot()
        scene = self.tokenizer.build(
            self._agents, self.info_center, step, deciding=agent_id,
            history=self._event_history, prefix_len=len(prefix_np))
        with torch.no_grad():
            at = torch.as_tensor(scene["agent_tokens"], device=self.device).unsqueeze(0)
            atty = torch.as_tensor(scene["agent_types"], device=self.device).unsqueeze(0)
            ct = torch.as_tensor(scene["cand_tokens"], device=self.device).unsqueeze(0)
            gl = torch.as_tensor(scene["global"], device=self.device).unsqueeze(0)
            ga, gc = self.actor.encode(at, atty, ct, gl,
                                       agent_pad=torch.zeros_like(at[..., 0], dtype=torch.bool),
                                       cand_pad=torch.zeros_like(ct[..., 0], dtype=torch.bool))
            row = scene["agent_index"].get(int(agent_id))
            h = ga[0, row] if row is not None else ga[0].mean(dim=0)
            apad = torch.zeros_like(at[..., 0], dtype=torch.bool)
            cpad = torch.zeros_like(ct[..., 0], dtype=torch.bool)
            memory = torch.cat([ga, gc], dim=1)
            memory_pad = torch.cat([
                apad, cpad,
                torch.zeros((1, 1), dtype=torch.bool, device=self.device),
            ], dim=1)
            prefix_t = torch.as_tensor(
                prefix_np, device=self.device).unsqueeze(0)
            prefix_pad = torch.zeros(
                (1, len(prefix_np)), dtype=torch.bool, device=self.device)
            h = self.actor.condition_on_prefix(
                h.unsqueeze(0), memory, memory_pad, prefix_t, prefix_pad)[0]

            launch_action, launch_lp = None, 0.0
            if launch_active:
                if launch_trainable:
                    # 掩码恒全 1：launch_active=True 本身已蕴含"有合法发射目标"
                    # （智能体侧 launch_allowed 门控）；三个等待选项恒合法
                    lm = torch.ones(
                        self.config.activation_dim,
                        dtype=torch.bool, device=self.device)
                    logits = self.actor.launch_head(h)
                    logits = logits.masked_fill(~lm, -1e9)
                    dist = torch.distributions.Categorical(logits=logits)
                    idx = int(dist.sample().item()) if (
                        self.training or self.config.eval_sample) else int(
                        torch.argmax(logits).item())
                    launch_action, launch_lp = idx, float(dist.log_prob(
                        torch.tensor(idx, device=self.device)).item())
                else:
                    launch_action = LAUNCH_NOW

            launch_context = int(launch_action) if launch_action is not None else -1
            launch_context_t = torch.as_tensor(
                [max(launch_context, 0)], dtype=torch.long, device=self.device)
            goal_h = self.actor.after_launch(
                h.unsqueeze(0), launch_context_t,
                torch.as_tensor([launch_context >= 0], device=self.device))[0]

            cand_rows, goal_action, goal_lp = None, None, 0.0
            goal_kind, goal_arg_row = 0, len(scene["cand_tokens"])
            eff_goal_mask = np.asarray(goal_mask, dtype=bool).copy()
            # Query routing is a pre-action phase fact, never inferred from
            # the sampled slot.  Once the scout state machine has latched a
            # stop, post-deadline target choices stay on the frozen legacy L
            # target query even if an upstream mask accidentally retains a
            # stale search bit.
            p1_search_phase = bool(
                self.config.train_stage == "p1_scout"
                and ti == 2
                and self.info_center.l_scouting_active
            )
            goal_active = goal_event or (launch_action == LAUNCH_NOW)
            if goal_active:
                # 候选行号（实体槽 + 4x8 搜索格 + KEEP）；KEEP 在每条场景
                # 本地候选数组末尾，组 batch 时会重映射到该 batch 的末尾。
                own = sorted(eligible_targets(self.info_center.targets(), missile_type),
                             key=lambda x: target_sort_key(x, missile_type))[:self.goal_slots]
                union_ids = scene["union_ids"]
                n_union = len(union_ids)
                rows = np.full(self.target_dim, n_union, dtype=np.int64)
                for i, tgt in enumerate(own):
                    try:
                        rows[i] = union_ids.index(tgt.entity_id)
                    except ValueError:
                        pass  # 槽位无对应实体 → 指向 KEEP 位，mask 已屏蔽
                for s in range(len(self.info_center.search_sectors())):
                    try:
                        rows[self.goal_slots + s] = union_ids.index(-(s + 1))
                    except ValueError:
                        pass
                cand_rows = rows
                q = self.actor.goal_query[ti](goal_h)
                keys = self.actor.goal_key(gc[0])          # [M+1, d]
                scores = (q @ keys.T) / (self.config.d_model ** 0.5)
                # 槽位空间 = TARGET_DIM：按 rows 从并集分数中取出各槽位的分数
                # （无效槽位指向 KEEP 位，随后被 mask 屏蔽）
                logits = scores[torch.as_tensor(rows, dtype=torch.long,
                                                device=self.device)]
                gm = torch.as_tensor(np.asarray(goal_mask, dtype=bool),
                                     device=self.device)
                if launch_action == LAUNCH_NOW:
                    # 发射选靶：KEEP 无语义（尚无目标可保持）——掩掉。
                    # 否则 argmax 态（评估模式）会死锁在 KEEP 上永不发射，
                    # 且每步重试整场景前向（实测一回合 1195s、24 枚 M 全趴窝）
                    gm = gm.clone()
                    gm[self.keep_index] = False
                    eff_goal_mask[self.keep_index] = False
                    if not bool(gm.any()):
                        return {"launch": None, "goal": None,
                                "motion": MOTION_AUTOGUIDE,
                                "resource": RESOURCE_OFF}
                if bool(gm.any()):
                    logits = logits.masked_fill(~gm, -1e9)
                    if (
                        ti == 2
                        and self.config.p1_split_search_query
                        and p1_search_phase
                    ):
                        q = self.actor.l_search_query(goal_h)
                        scores = (q @ keys.T) / (
                            self.config.d_model ** 0.5)
                        logits = scores[torch.as_tensor(
                            rows, dtype=torch.long, device=self.device)]
                        logits = logits.masked_fill(~gm, -1e9)
                    dist = torch.distributions.Categorical(logits=logits)
                    slot = int(dist.sample().item()) if (
                        self.training or self.config.eval_sample) else int(
                        torch.argmax(logits).item())
                    goal_action, goal_lp = slot, float(dist.log_prob(
                        torch.tensor(slot, device=self.device)).item())
                    goal_arg_row = int(rows[slot])
                    goal_kind = (1 if slot < self.goal_slots else
                                 2 if slot < self.keep_index else 3)
                else:
                    # A launched H/M may outlive every compatible objective.
                    # Do not sample a uniform action from an all-false mask or
                    # create an actor record with no legal semantic action.
                    cand_rows = None

            # 卫星头（resource_active：已发射的 H ∧ 未用过 ∧ 预算剩余）：
            # 请求 → 当步集中结算 → 引擎开 3 分钟全局窗口，窗口内 H 对
            # 9400/9600 命中率→1.0（哨卫巢 hp18 一发即杀）。时机可学：
            # 距目标距离已在 agent token 中
            goal_arg_emb = gc[:, min(goal_arg_row, gc.shape[1] - 1)]
            sat_h = self.actor.after_goal(
                goal_h.unsqueeze(0),
                torch.as_tensor([goal_kind], dtype=torch.long, device=self.device),
                goal_arg_emb,
                torch.as_tensor([goal_action is not None], device=self.device),
            )[0]
            sat_action, sat_lp = None, 0.0
            if resource_trainable:
                slogits = self.actor.sat_head(sat_h)
                sdist = torch.distributions.Categorical(logits=slogits)
                sidx = int(sdist.sample().item()) if (
                    self.training or self.config.eval_sample) else int(
                    torch.argmax(slogits).item())
                sat_action, sat_lp = sidx, float(sdist.log_prob(
                    torch.tensor(sidx, device=self.device)).item())

            has_trainable_action = bool(
                (launch_trainable and launch_action is not None)
                or goal_action is not None
                or resource_trainable
            )
            p1_goal_record = bool(
                self.config.train_stage == "p1_scout"
                and ti == 2
                and goal_action is not None
            )
            if self.config.train_stage == "p1_scout":
                # Filter at collection time, before segment construction and
                # GAE.  A P1 record therefore contains exactly one valid
                # likelihood term: the L goal choice.  H/M actions still enter
                # the MAT prefix below, but never become optimization samples.
                if self.training and has_trainable_action and not p1_goal_record:
                    self._ep_p1_filtered_events += 1
                    self._p1_window_filtered_events += 1
                has_trainable_action = p1_goal_record
            if self.training and has_trainable_action:
                with torch.no_grad():
                    value = float(self.critic(
                        at, ct, gl, agent_pad=apad, cand_pad=cpad,
                        prefix_actions=prefix_t,
                        prefix_pad=prefix_pad)[0].item())
                # 闭合该弹上一段并开新段（SMDP）
                if int(agent_id) in self._open_ret:
                    self._close_record(
                        int(agent_id), end_step=int(step),
                        next_value=value, terminal=False)
                self._open_ret[int(agent_id)] = 0.0
                self._open_credit[int(agent_id)] = 0.0
                rec = MacroRecord(
                    episode=self._rounds_done,
                    step=int(step), agent_id=int(agent_id), agent_row=row if row is not None else 0,
                    missile_type=ti,
                    agent_tokens=scene["agent_tokens"], agent_types=scene["agent_types"],
                    cand_tokens=scene["cand_tokens"], glob=scene["global"],
                    prefix_actions=prefix_np,
                    cand_rows=cand_rows if cand_rows is not None else np.zeros(0, np.int64),
                    goal_mask=eff_goal_mask,
                    launch_mask=(
                        np.zeros(self.config.activation_dim, dtype=bool)
                        if p1_goal_record
                        else (
                            np.ones(self.config.activation_dim, dtype=bool)
                            if launch_active else np.zeros(
                                self.config.activation_dim, bool)
                        )
                    ),
                    launch_context=launch_context,
                    goal_kind=goal_kind,
                    goal_arg_row=goal_arg_row,
                    launch_action=(
                        None if p1_goal_record else (
                            launch_action if launch_trainable else None)),
                    goal_action=goal_action,
                    launch_old_lp=(
                        0.0 if p1_goal_record else (
                            launch_lp if launch_trainable else 0.0)),
                    goal_old_lp=goal_lp,
                    value=value,
                    sat_action=(
                        None if p1_goal_record else (
                            sat_action if resource_trainable else None)),
                    sat_old_lp=(
                        0.0 if p1_goal_record else (
                            sat_lp if resource_trainable else 0.0)),
                    event_id=self._event_id,
                    order_index=self._decision_order_index.get(int(agent_id), 0),
                    p1_search_phase=bool(p1_search_phase))
                self._records.append(rec)
                if p1_goal_record:
                    self._ep_p1_goal_records += 1
                    self._p1_window_goal_records += 1
                if launch_active:
                    self._ep_events_launch += 1
                else:
                    self._ep_events_goal += 1
                self._ep_events_by_agent[int(agent_id)] = \
                    self._ep_events_by_agent.get(int(agent_id), 0) + 1

            if launch_action is not None or goal_action is not None or sat_action is not None:
                self._decision_prefix.append(self._action_prefix_token(
                    missile_type=missile_type,
                    launch_action=launch_action,
                    goal_action=goal_action,
                    cand_rows=cand_rows,
                    scene=scene,
                    sat_action=sat_action,
                    agent_id=int(agent_id),
                ))
                if launch_action == LAUNCH_NOW:
                    self._current_event_summary["launches"] += 1
                if goal_action is not None:
                    if goal_action < self.goal_slots:
                        self._current_event_summary["targets"] += 1
                    elif goal_action < self.keep_index:
                        self._current_event_summary["searches"] += 1
                if sat_action == 1:
                    self._current_event_summary["satellite"] = 1

        return {"launch": launch_action, "goal": goal_action,
                "motion": MOTION_AUTOGUIDE, "resource": sat_action}

    # ---------- 卫星（可训练头 + 集中结算） ----------

    def register_satellite_request(self, entity_id):
        self._sat_requests.append(int(entity_id))

    def is_satellite_decider(self, entity_id: int) -> bool:
        """Choose one live launched H missile to make the global window decision."""

        candidates = [
            int(agent.entity_id)
            for agent in self._agents
            if int(agent.missile_type) == 21000
            and int(agent.launch_step) >= 0
            and not bool(agent.sat_used)
            and int(agent.entity_id) in self.info_center.alive_ids
            and int(agent.entity_id) not in self.info_center.lost_ids
            and agent.latest_observation
            and agent.latest_observation.get("self")
        ]
        return bool(candidates) and int(entity_id) == min(candidates)

    def settle_satellite_requests(self):
        """当步集中结算：预算内全部批准 → 引擎 UseSatellite 命令。

        引擎预算（satelliteMaxUseCount=100 次 × 3 分钟窗口）与
        InfoCenter 记账对齐；每弹至多请求一次（agent 侧 sat_used 门控）。
        """
        requests, self._sat_requests = self._sat_requests, []
        granted = []
        # One request opens a global window.  Approving every simultaneous H
        # request consumed multiple uses for exactly the same observation window.
        if requests and self.info_center.sat_budget_remaining > 0:
            entity_id = requests[0]
            self.info_center.sat_used_count += 1
            self._sat_granted.add(entity_id)
            granted.append([3, entity_id, 0.0, 0.0])   # ACTION_USE_SAT
        return granted, {"granted_entity_ids": tuple(self._sat_granted),
                         "budget_remaining": self.info_center.sat_budget_remaining}

    # ---------- 专家先验（E01 级知识；动作始终在 goal_mask 合法槽内） ----------

    def _agent_pos(self, agent_id: int):
        for agent in self._agents:
            if int(agent.agent_id) == int(agent_id):
                pos = (agent.latest_observation or {}).get("self", {}).get("position") or {}
                return float(pos.get("lon", 118.5)), float(pos.get("lat", 23.0))
        return None

    def _expert_decision(self, agent_id, step, missile_type, ti, launch_active,
                         goal_mask, goal_event, current_goal_slot,
                         resource_active=False):
        """专家规则：波次释放 + 类型偏好近距优先选靶 + 临空开窗。

        决策语义与网络完全同构（同一掩码、同一槽位协议），动作永远合法；
        记录场景快照与专家动作（old_lp=0，BC 不需要采样概率）。
        """
        decision = {"launch": None, "goal": None,
                    "motion": MOTION_AUTOGUIDE, "resource": None}
        launch_action = None
        if launch_active:
            launch_action = 0 if step >= EXPERT_WAVES.get(ti, 0) else 1  # FIRE / HOLD
            decision["launch"] = launch_action
        effective_goal_mask = np.asarray(goal_mask, dtype=bool).copy()
        if launch_action == LAUNCH_NOW:
            effective_goal_mask[self.keep_index] = False
            if not bool(effective_goal_mask.any()):
                # FIRE without a destination is not executable.  Preserve a
                # legal expert sample and retry after one HOLD interval.
                launch_action = 1
                decision["launch"] = launch_action
        goal_active = goal_event or (launch_action == 0)

        goal_action = None
        if goal_active:
            own = sorted(eligible_targets(self.info_center.targets(), missile_type),
                         key=lambda x: target_sort_key(x, missile_type))[:self.goal_slots]
            pos = self._agent_pos(agent_id)
            pref = {0: {9400: 0, 9600: 1, 9500: 9},        # H：母巢优先（w5 且 2 发杀）
                    1: {9600: 0, 9400: 1, 9500: 9},        # M：哨卫优先
                    2: {9500: 0, 9400: 9, 9600: 9}}[ti]     # L：已发现船优先，否则扇区
            best, best_key = None, None
            gm = effective_goal_mask
            for slot in range(self.target_dim):
                if not gm[slot]:
                    continue
                if slot < len(own):
                    tgt = own[slot]
                    rank = pref.get(tgt.entity_type, 5)
                    d = (abs(tgt.position.lon - (pos[0] if pos else 118.5)) * 96.0
                         + abs(tgt.position.lat - (pos[1] if pos else 23.0)) * 110.0)
                elif slot >= self.sector_index_0 and slot < self.keep_index:
                    rank, d = 8 if ti == 2 else 99, 500.0   # 扇区仅 L 可选
                else:
                    rank, d = 7, 0.0                          # KEEP（有目标时保持）
                key = (rank, d, slot)
                if best_key is None or key < best_key:
                    best, best_key = slot, key
            goal_action = best
            decision["goal"] = goal_action

        sat_action = 0 if resource_active else None
        if resource_active and ti == 0:
            agent = next(
                (item for item in self._agents
                 if int(item.agent_id) == int(agent_id)),
                None,
            )
            d_km = getattr(agent, "_current_goal_dist_km", None)
            if d_km is not None and float(d_km) <= EXPERT_SAT_LEAD / 10.0:
                sat_action = 1
            decision["resource"] = sat_action

        # 记录（BC 用）：场景快照 + 专家动作
        prefix_np = self._prefix_snapshot()
        scene = self.tokenizer.build(
            self._agents, self.info_center, step, deciding=agent_id,
            history=self._event_history, prefix_len=len(prefix_np))
        with torch.no_grad():
            at = torch.as_tensor(scene["agent_tokens"], device=self.device).unsqueeze(0)
            atty = torch.as_tensor(scene["agent_types"], device=self.device).unsqueeze(0)
            ct = torch.as_tensor(scene["cand_tokens"], device=self.device).unsqueeze(0)
            gl = torch.as_tensor(scene["global"], device=self.device).unsqueeze(0)
            prefix_t = torch.as_tensor(prefix_np, device=self.device).unsqueeze(0)
            prefix_pad = torch.zeros(
                (1, len(prefix_np)), dtype=torch.bool, device=self.device)
            value = float(self.critic(
                at, ct, gl,
                agent_pad=torch.zeros_like(at[..., 0], dtype=torch.bool),
                cand_pad=torch.zeros_like(ct[..., 0], dtype=torch.bool),
                prefix_actions=prefix_t, prefix_pad=prefix_pad)[0].item())
        if int(agent_id) in self._open_ret:
            self._close_record(
                int(agent_id), end_step=int(step), next_value=value,
                terminal=False)
        self._open_ret[int(agent_id)] = 0.0
        self._open_credit[int(agent_id)] = 0.0
        own_full = sorted(eligible_targets(self.info_center.targets(), missile_type),
                          key=lambda x: target_sort_key(x, missile_type))[:self.goal_slots]
        union_ids = scene["union_ids"]
        rows = np.full(self.target_dim, len(union_ids), dtype=np.int64)
        for i, tgt in enumerate(own_full):
            try:
                rows[i] = union_ids.index(tgt.entity_id)
            except ValueError:
                pass
        for s in range(len(self.info_center.search_sectors())):
            try:
                rows[self.goal_slots + s] = union_ids.index(-(s + 1))
            except ValueError:
                pass
        goal_kind = (0 if goal_action is None else
                     1 if goal_action < self.goal_slots else
                     2 if goal_action < self.keep_index else 3)
        goal_arg_row = (
            int(rows[goal_action])
            if goal_action is not None and goal_action < len(rows)
            else len(scene["cand_tokens"])
        )
        self._records.append(MacroRecord(
            episode=self._rounds_done, step=int(step), agent_id=int(agent_id),
            agent_row=scene["agent_index"].get(int(agent_id), 0), missile_type=ti,
            agent_tokens=scene["agent_tokens"], agent_types=scene["agent_types"],
            cand_tokens=scene["cand_tokens"], glob=scene["global"],
            prefix_actions=prefix_np,
            cand_rows=rows, goal_mask=effective_goal_mask,
            launch_mask=(np.ones(self.config.activation_dim, dtype=bool)
                         if launch_active else np.zeros(
                             self.config.activation_dim, bool)),
            launch_context=int(launch_action) if launch_action is not None else -1,
            goal_kind=goal_kind,
            goal_arg_row=goal_arg_row,
            launch_action=launch_action, goal_action=goal_action,
            launch_old_lp=0.0, goal_old_lp=0.0, value=value,
            sat_action=sat_action, sat_old_lp=0.0,
            event_id=self._event_id,
            order_index=self._decision_order_index.get(int(agent_id), 0)))
        if launch_action is not None or goal_action is not None or sat_action is not None:
            self._decision_prefix.append(self._action_prefix_token(
                missile_type=missile_type, launch_action=launch_action,
                goal_action=goal_action, cand_rows=rows, scene=scene,
                sat_action=sat_action, agent_id=int(agent_id)))
            if launch_action == LAUNCH_NOW:
                self._current_event_summary["launches"] += 1
            if goal_action is not None:
                if goal_action < self.goal_slots:
                    self._current_event_summary["targets"] += 1
                elif goal_action < self.keep_index:
                    self._current_event_summary["searches"] += 1
            if sat_action == 1:
                self._current_event_summary["satellite"] = 1
        return decision

    def _close_record(self, agent_id: int, *, end_step: int | None = None,
                      next_value: float = 0.0, terminal: bool = False):
        ret = self._open_ret.pop(agent_id, 0.0)
        credit = self._open_credit.pop(agent_id, 0.0)
        if ret < -500:
            logger.warning("[TopFormer] 异常负段 ret=%.0f agent=%d", ret, agent_id)
        for rec in reversed(self._records):
            if rec.agent_id == agent_id:
                rec.ret = ret
                rec.local_credit = credit
                rec.duration = max(1, int(
                    (self._current_step if end_step is None else end_step)
                    - rec.step))
                rec.next_value = float(next_value)
                rec.terminal = bool(terminal)
                return

    @property
    def satellite_granted(self):
        return self._sat_granted

    # ---------- 团队标量注入（发现奖在顶层按团队计入） ----------

    def _discovery_scalar(self) -> float:
        return self.config.discovery_bonus * len(self.info_center.step_discoveries)

    # ---------- 回合执行指标（对齐 unified_metrics） ----------

    def record_launch(self, missile_type, step, agent_id=None):
        self._ep_launch_steps.append(int(step))
        self._ep_launch_by_type[int(missile_type)] = \
            self._ep_launch_by_type.get(int(missile_type), 0) + 1
        if agent_id is not None:
            self._wave_pending.append((int(agent_id), int(step)))
        self._flush_wave_bonuses(current_step=int(step))

    def _flush_wave_bonuses(self, current_step: int, final: bool = False):
        """波次奖励（倒 U 型，延迟结算）。

        形状 4·frac·(1−frac)（frac = 同窗批次数/总兵力）：
        - 半兵力左右的中型波次拿峰值（真波次）；
        - 全军齐射 frac→1 → 0（E 线教训：旧形状 min(cohort/20,1) 把
          126 枚齐射的每个成员都奖满，反而把发射分布推回 0 步齐射）；
        - 孤立发射 frac→0 → 0。
        延迟 ±WAVE_WINDOW 步结算：发射者决策时看不到未来，等窗口内
        的发射全部落盘后按完整批次数统一计奖（批内公平）。
        """
        coef = float(os.getenv("WAVE_BONUS", "0.0"))
        if coef <= 0 or not self._wave_pending:
            return
        done, remain = [], self._wave_pending
        if final:
            done, remain = remain, []
        else:
            done = [w for w in remain if w[1] + WAVE_WINDOW <= current_step]
            if done:
                remain = [w for w in remain if w[1] + WAVE_WINDOW > current_step]
        if not done:
            self._wave_pending = remain
            return
        total = max(self.total_platforms, 1)
        for aid, t in done:
            cohort = sum(1 for s in self._ep_launch_steps
                         if t - WAVE_WINDOW <= s <= t + WAVE_WINDOW)
            frac = cohort / total
            bonus = coef * 4.0 * frac * (1.0 - frac)
            if bonus > 0 and self.training and aid in self._open_ret:
                self._open_ret[aid] += bonus
                self._comp_wave += bonus
        self._wave_pending = remain

    def record_goal_choice(self, target_id, missile_type):
        tid = int(target_id)
        self._ep_goal_choices[tid] = self._ep_goal_choices.get(tid, 0) + 1
        health = self.info_center.initial_health.get(tid, 0.0)
        assigned = self._ep_assigned_damage.get(tid, 0.0)
        if health > 0.0 and assigned >= health:
            self._ep_overkill[tid] = self._ep_overkill.get(tid, 0) + 1
        target_type = -1
        for target in self.info_center.fusion.targets:
            if target.entity_id == tid:
                target_type = target.entity_type
                break
        dmg = self.info_center.EXPECTED_DAMAGE.get(int(missile_type), {}).get(target_type, 0.0)
        self._ep_assigned_damage[tid] = assigned + dmg

    def record_retarget(self, step, launch_step, agent_id=None):
        self._ep_retarget_delays.append(max(0, int(step) - int(launch_step)))
        # 换靶代价：记入该弹当前未闭合段（作废进近进度的显式化）
        penalty = float(self.config.retarget_penalty)
        if self.training and penalty and agent_id is not None:
            aid = int(agent_id)
            if aid in self._open_ret:
                self._open_ret[aid] -= penalty
                self._comp_retarget -= penalty

    def record_search_action(self):
        self._ep_search_actions += 1

    def record_motion_dwell(self, dwell):
        pass

    def episode_metrics(self, ships_total=None):
        launches = sorted(self._ep_launch_steps)
        hist: dict[str, int] = {}
        for s in launches:
            b = (s // 50) * 50
            hist[str(b)] = hist.get(str(b), 0) + 1
        delays = sorted(self._ep_retarget_delays)
        if ships_total is None:
            ships_total = sum(1 for t in self.initial_targets if t.entity_type == 9500)
        ships_total = max(int(ships_total), 0)
        found = len(self.info_center.known_ships)
        trainable_actor_numel = sum(
            parameter.numel() for parameter in self.actor.parameters()
            if parameter.requires_grad
        )
        l_scout_runtime = self.info_center.l_scout_metrics
        return {
            "launch": {"count": len(launches),
                       "by_missile_type": {str(k): v for k, v in sorted(self._ep_launch_by_type.items())},
                       "time_histogram_50step": hist,
                       "mean_step": float(np.mean(launches)) if launches else 0.0,
                       "p50_step": float(np.percentile(launches, 50)) if launches else 0.0,
                       "min_step": float(launches[0]) if launches else -1.0,
                       "max_step": float(launches[-1]) if launches else -1.0},
            "target_assignment": {str(k): v for k, v in sorted(self._ep_goal_choices.items())},
            "repeated_focus": {"total": sum(self._ep_overkill.values()),
                               "by_target": {str(k): v for k, v in sorted(self._ep_overkill.items())}},
            "discovery": {"ships_total": ships_total, "ships_found": found,
                          "rate": (found / ships_total) if ships_total > 0 else 1.0,
                          "causal_links": int(self._ep_discovery_links),
                          "unbound_events": int(self._ep_discovery_unbound),
                          "credited_targets": len(
                              self._ep_discovery_credited_targets),
                          "credited_official_points": float(
                              self._ep_discovery_credit_raw)},
            "retarget": {"count": len(delays),
                         "mean_delay": float(np.mean(delays)) if delays else 0.0,
                         "max_delay": float(delays[-1]) if delays else 0.0},
            "search_actions": int(self._ep_search_actions),
            "motion_dwell": {"count": 0, "mean": 0.0, "max": 0.0},
            "satellite_usage": int(self.info_center.sat_used_count),
            "train_phase": f"topformer-{self.config.upper_phase}",
            "train_stage": self.config.train_stage,
            "p1_scout": {
                **self.p1_scout_settings,
                **l_scout_runtime,
                "goal_records": int(self._ep_p1_goal_records),
                "filtered_events": int(self._ep_p1_filtered_events),
                "actor_trainable_numel": int(trainable_actor_numel),
                "actor_trainable_names": list(
                    self._trainable_actor_parameter_names),
            },
            "macro_events": {"launch": self._ep_events_launch,
                             "goal": self._ep_events_goal,
                             "top_agents": sorted(
                                 self._ep_events_by_agent.values(), reverse=True)[:5]},
        }

    # ---------- 转移提交 ----------

    def commit_transition(self, *, agent_id, entity_id, next_obs,
                          next_global_state, env_reward, terminated):
        # 个体退出 ≠ 环境终止：保留完整团队 return-to-go——on-policy 似然比
        # 推导下的正确目标，避免无依据的截断偏差。身后与该动作无关的奖励
        # 期望梯度为零、只增加方差（方差由基线处理）；这不代表它们是该
        # 动作的真实因果贡献。
        return self._pending.pop(int(agent_id), None) is not None

    def finish_commit_phase(self):
        pass

    # ---------- 回合边界 ----------

    def finish_episode(self, score: float):
        if self.training:
            if abs(self._episode_official - float(score)) > 0.5:
                logger.warning("[TopFormer] 官方计分不变量破坏: Σ增量=%.2f vs 回合分=%.2f",
                               self._episode_official, float(score))
            self._flush_wave_bonuses(current_step=10**9, final=True)
            # PBRS terminal convention Phi(terminal)=0.  These corrections
            # make every shaping channel telescope over the full episode while
            # still providing dense, local TD/GAE feedback between events.
            for aid in list(self._open_ret):
                progress = self._progress_balance.get(aid, 0.0)
                if progress:
                    correction = -self.config.progress_shaping_coef * progress
                    self._open_ret[aid] += correction
                    self._reward_ledger["progress_pbrs"] += correction
                    self._comp_progress += correction
                coverage = self._coverage_potential.get(aid, 0.0)
                if coverage:
                    correction = -self.config.coverage_shaping_coef * coverage
                    self._open_ret[aid] += correction
                    self._reward_ledger["coverage_pbrs"] += correction
                    self._comp_progress += correction
            # Final-window launch bonuses must be applied before closing the
            # episode's open SMDP segments.
            for aid in list(self._open_ret):
                self._close_record(
                    aid, end_step=self._current_step,
                    next_value=0.0, terminal=True)
            # 部署信用：直接使用整局官方比分，并保留多回合记录到更新边界。
            if self._deploy_records:
                for r in self._deploy_records:
                    if int(r["episode"]) == self._rounds_done:
                        r["ret"] = float(score) * self.config.reward_scale
            self._rounds_done += 1
            self._updates_window_rounds += 1
            update_boundary = (
                self._rounds_done % self.config.update_every_rounds == 0
            )
            expert_boundary = (
                self.config.expert_rounds > 0
                and self._rounds_done == self.config.expert_rounds
            )
            if (
                update_boundary
                and self._records
                and (
                    len(self._records) >= self.config.min_update_records
                    or expert_boundary
                )
            ):
                self.update()

    def reset_episode(self):
        self.info_center = InfoCenter(
            self.initial_targets,
            self.total_platforms,
            coordinate_scale=self.coordinate_scale,
            search_bounds=self.search_bounds,
            search_rows=self.config.search_rows,
            search_cols=self.config.search_cols,
            l_scout_enabled=self.config.l_scout_enabled,
            l_scout_count=self.config.l_scout_count,
            l_scout_expected_targets=self.config.l_scout_expected_targets,
            l_scout_coverage_threshold=(
                self.config.l_scout_coverage_threshold),
            l_scout_quiet_steps=self.config.l_scout_quiet_steps,
            l_scout_deadline=self.config.l_scout_deadline,
        )
        self.info_center.configure_l_scouts(
            getattr(agent, "entity_id")
            for agent in self._agents
            if getattr(agent, "missile_type", None) == 21002
            and getattr(agent, "entity_id", None) is not None
        )
        self._pending.clear()
        self._sat_granted.clear()
        self._sat_requests.clear()
        self._episode_team_total = 0.0
        self._wave_pending.clear()
        self._episode_official = 0.0
        self._open_ret.clear()
        self._open_credit.clear()
        self._decision_prefix.clear()
        self._decision_order = ()
        self._decision_order_index.clear()
        self._current_step = 0
        self._event_history = {
            "official_delta": 0.0, "event_age": self.config.max_steps,
            "launches": 0, "targets": 0, "searches": 0, "satellite": 0,
        }
        self._current_event_summary = {
            "launches": 0, "targets": 0, "searches": 0, "satellite": 0,
        }
        self._progress_balance.clear()
        self._coverage_potential.clear()
        self._reward_ledger = {
            "official": 0.0, "progress_pbrs": 0.0,
            "coverage_pbrs": 0.0, "causal_credit": 0.0,
            "discovery_enable_credit": 0.0,
        }
        self._discovery_credit_links.clear()
        self._discovery_event_cursor = 0
        self._ep_discovery_links = 0
        self._ep_discovery_unbound = 0
        self._ep_discovery_credited_targets.clear()
        self._ep_discovery_credit_raw = 0.0
        self._ep_launch_steps.clear()
        self._ep_launch_by_type.clear()
        self._ep_goal_choices.clear()
        self._ep_assigned_damage.clear()
        self._ep_overkill.clear()
        self._ep_retarget_delays.clear()
        self._ep_search_actions = 0
        self._ep_events_launch = 0
        self._ep_events_goal = 0
        self._ep_events_by_agent.clear()
        self._ep_p1_goal_records = 0
        self._ep_p1_filtered_events = 0

    def _expert_bc_pending(self) -> bool:
        return (self.config.expert_rounds > 0 and self._records
                and self._records[0].episode < self.config.expert_rounds
                and self._expert_bc_count < (
                    self.config.expert_rounds
                    // self.config.update_every_rounds
                ))

    def _apply_lr_schedule(self) -> tuple[float, float]:
        """Apply a monotone schedule that never exceeds configured base LRs."""

        # ``update_count`` is the number of already completed updates because
        # this is called immediately before the next optimizer step.  Thus
        # updates 1--8 use the base LR, 9--16 use half, and update 17 onward
        # uses the second decay (subject to the configured floor).
        decay = 0.5 ** max(0, self.update_count // 8)
        values = []
        for opt, base_lr, floor in (
            (self.opt_actor, self.config.learning_rate, 2.5e-5),
            (self.opt_critic, self.config.critic_lr, 1.5e-4),
        ):
            lr = min(base_lr, max(base_lr * decay, floor))
            for group in opt.param_groups:
                group["lr"] = lr
            values.append(float(lr))
        return values[0], values[1]

    # ---------- PPO 更新 ----------

    @staticmethod
    def compute_smdp_gae(records: list[MacroRecord], gamma: float,
                         gae_lambda: float,
                         causal_credit_coef: float = 0.0,
                         discovery_credit_coef: float = 0.0,
                         search_only_actor: bool = False,
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Return token advantages and team-value targets for async events.

        Durations are measured in simulator steps.  The exact-source credit
        channel sharpens only the policy advantage; the critic target remains
        the cooperative official/PBRS return so it cannot learn a duplicated
        reward definition.  Discovery enabling credit is deliberately *not*
        recursive: it belongs only to the exact search action that produced
        the first legal target track.
        """
        advantages = np.zeros(len(records), dtype=np.float32)
        value_targets = np.zeros(len(records), dtype=np.float32)
        chains: dict[tuple[int, int], list[int]] = {}
        for index, record in enumerate(records):
            chains.setdefault(
                (int(record.episode), int(record.agent_id)), []).append(index)
        for indices in chains.values():
            indices.sort(key=lambda i: (
                int(records[i].step), int(records[i].order_index)))
            team_gae = 0.0
            credit_gae = 0.0
            for index in reversed(indices):
                record = records[index]
                duration = max(1, int(record.duration))
                continuation = 0.0 if record.terminal else 1.0
                discount = float(gamma) ** duration
                trace = float(gamma * gae_lambda) ** duration
                delta = (
                    float(record.ret)
                    + discount * continuation * float(record.next_value)
                    - float(record.value)
                )
                team_gae = delta + trace * continuation * team_gae
                credit_gae = (
                    float(record.local_credit)
                    + trace * continuation * credit_gae
                )
                value_targets[index] = team_gae + float(record.value)
                if search_only_actor:
                    # P1.1 main PPO is cooperative search-only.  Recursive
                    # hit-source credit belongs to attackers and must not leak
                    # backwards into a scout's search action.  Discovery
                    # enabling credit has its own balanced auxiliary PPO loss.
                    advantages[index] = (
                        team_gae if int(record.goal_kind) == 2 else 0.0
                    )
                else:
                    advantages[index] = (
                        team_gae
                        + float(causal_credit_coef) * credit_gae
                        + float(discovery_credit_coef) * float(getattr(
                            record, "discovery_credit", 0.0))
                    )
        return advantages, value_targets

    @staticmethod
    def discovery_aux_weights(
        records: list[MacroRecord],
    ) -> tuple[np.ndarray, int, int]:
        """Return per-record weights for target-balanced discovery PPO.

        A target contributes total weight ``1 / number_of_targets`` no matter
        whether the same entity id was discovered in one episode or six.
        Within a target group, realised positive official credit determines
        the relative split.  Legacy records without the per-target ledger are
        migrated from ``enabled_target_ids`` and their aggregate credit.
        """

        groups: dict[int, list[tuple[int, float]]] = {}
        for index, record in enumerate(records):
            if int(record.goal_kind) != 2:
                continue
            by_target = tuple(getattr(
                record, "discovery_credit_by_target", ()))
            if by_target:
                credits = [
                    (int(target_id), float(value))
                    for target_id, value in by_target if float(value) > 0.0
                ]
            else:
                target_ids = tuple(dict.fromkeys(int(target_id) for target_id in
                    getattr(record, "enabled_target_ids", ())))
                total = float(getattr(record, "discovery_credit", 0.0))
                share = total / max(len(target_ids), 1)
                credits = [
                    (target_id, share) for target_id in target_ids
                    if share > 0.0
                ]
            for target_id, value in credits:
                groups.setdefault(target_id, []).append((index, value))
        weights = np.zeros(len(records), dtype=np.float32)
        if not groups:
            return weights, 0, 0
        target_weight = 1.0 / len(groups)
        linked: set[int] = set()
        for entries in groups.values():
            total = sum(value for _index, value in entries)
            if total <= 0.0:
                continue
            for index, value in entries:
                weights[index] += target_weight * value / total
                linked.add(index)
        return weights, len(groups), len(linked)

    @torch.no_grad()
    def replay_goal_logits(
        self, records: list[MacroRecord], *, batch_size: int | None = None,
    ) -> np.ndarray:
        """Recompute masked goal logits for offline on-policy audits.

        This follows the same padded-candidate and MAT-prefix path as PPO.
        Keeping it on the policy makes P1.1 acceptance checks test the actual
        network rather than an approximate analysis-only implementation.
        """

        if not records:
            return np.zeros((0, self.target_dim), dtype=np.float32)
        mb = max(1, int(batch_size or self.config.minibatch_records))
        result = np.full(
            (len(records), self.target_dim), -1e9, dtype=np.float32)
        for start in range(0, len(records), mb):
            batch = records[start:start + mb]
            b = len(batch)
            max_n = max(record.agent_tokens.shape[0] for record in batch)
            max_m = max(record.cand_tokens.shape[0] for record in batch)
            max_p = max(record.prefix_actions.shape[0] for record in batch)
            agent = np.zeros(
                (b, max_n, self.config.agent_dim), dtype=np.float32)
            types = np.zeros((b, max_n), dtype=np.int64)
            agent_pad = np.ones((b, max_n), dtype=bool)
            candidate = np.zeros(
                (b, max_m, self.config.cand_dim), dtype=np.float32)
            candidate_pad = np.ones((b, max_m), dtype=bool)
            glob = np.zeros((b, self.config.global_dim), dtype=np.float32)
            prefix = np.zeros(
                (b, max_p, self.config.prefix_dim), dtype=np.float32)
            prefix_pad = np.ones((b, max_p), dtype=bool)
            row = np.zeros(b, dtype=np.int64)
            missile_type = np.zeros(b, dtype=np.int64)
            launch_context = np.full(b, -1, dtype=np.int64)
            candidate_rows = np.zeros(
                (b, self.target_dim), dtype=np.int64)
            goal_mask = np.zeros((b, self.target_dim), dtype=bool)
            search_phase = np.zeros(b, dtype=bool)
            for index, record in enumerate(batch):
                n_agents = record.agent_tokens.shape[0]
                n_candidates = record.cand_tokens.shape[0]
                n_prefix = record.prefix_actions.shape[0]
                agent[index, :n_agents] = record.agent_tokens
                types[index, :n_agents] = record.agent_types
                agent_pad[index, :n_agents] = False
                candidate[index, :n_candidates] = record.cand_tokens
                candidate_pad[index, :n_candidates] = False
                glob[index] = record.glob
                prefix[index, :n_prefix] = record.prefix_actions
                prefix_pad[index, :n_prefix] = False
                row[index] = min(int(record.agent_row), max_n - 1)
                missile_type[index] = int(record.missile_type)
                launch_context[index] = int(record.launch_context)
                count = len(record.cand_rows)
                if count:
                    candidate_rows[index, :count] = _remap_candidate_rows(
                        record.cand_rows, n_candidates, max_m)
                    goal_mask[index, :count] = record.goal_mask
                search_phase[index] = _record_p1_search_phase(record)

            tensor = lambda value, dtype: torch.as_tensor(
                value, dtype=dtype, device=self.device)
            agent_t = tensor(agent, torch.float32)
            types_t = tensor(types, torch.long)
            agent_pad_t = tensor(agent_pad, torch.bool)
            candidate_t = tensor(candidate, torch.float32)
            candidate_pad_t = tensor(candidate_pad, torch.bool)
            glob_t = tensor(glob, torch.float32)
            prefix_t = tensor(prefix, torch.float32)
            prefix_pad_t = tensor(prefix_pad, torch.bool)
            row_t = tensor(row, torch.long)
            missile_type_t = tensor(missile_type, torch.long)
            launch_context_t = tensor(launch_context, torch.long)
            candidate_rows_t = tensor(candidate_rows, torch.long)
            goal_mask_t = tensor(goal_mask, torch.bool)
            search_phase_t = tensor(search_phase, torch.bool)

            agent_h, candidate_h = self.actor.encode(
                agent_t, types_t, candidate_t, glob_t,
                agent_pad_t, candidate_pad_t)
            base_h = agent_h[
                torch.arange(b, device=self.device), row_t]
            memory = torch.cat([agent_h, candidate_h], dim=1)
            memory_pad = torch.cat([
                agent_pad_t, candidate_pad_t,
                torch.zeros(
                    (b, 1), dtype=torch.bool, device=self.device),
            ], dim=1)
            decision_h = self.actor.condition_on_prefix(
                base_h, memory, memory_pad, prefix_t, prefix_pad_t)
            goal_h = self.actor.after_launch(
                decision_h, launch_context_t.clamp(min=0),
                launch_context_t >= 0)
            keys = self.actor.goal_key(candidate_h)
            queries = torch.stack([
                self.actor.goal_query[k](goal_h) for k in range(3)
            ], dim=0)
            query = queries[
                missile_type_t, torch.arange(b, device=self.device)]
            if self.config.p1_split_search_query:
                split_query = self.actor.l_search_query(goal_h)
                use_split = (
                    (missile_type_t == 2) & search_phase_t
                ).unsqueeze(-1)
                query = torch.where(use_split, split_query, query)
            gathered = keys.gather(
                1, candidate_rows_t.unsqueeze(-1).expand(
                    -1, -1, keys.shape[-1]))
            logits = (query.unsqueeze(1) * gathered).sum(-1) / (
                self.config.d_model ** 0.5)
            logits = logits.masked_fill(~goal_mask_t, -1e9)
            result[start:start + b] = logits.detach().cpu().numpy()
        return result

    def _bc_update(self, recs, n):
        """专家数据的行为克隆：条件字段 CE（同 minibatch 机制）。"""
        self._expert_bc_count += 1
        mb = max(8, int(self.config.minibatch_records))
        order = np.random.permutation(n)
        to = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.device)

        def chunk(idx):
            b = len(idx)
            max_n = max(recs[i].agent_tokens.shape[0] for i in idx)
            max_m = max(recs[i].cand_tokens.shape[0] for i in idx)
            max_p = max(recs[i].prefix_actions.shape[0] for i in idx)
            a_b = np.zeros((b, max_n, self.config.agent_dim), np.float32)
            t_b = np.zeros((b, max_n), np.int64)
            ap = np.ones((b, max_n), bool)
            c_b = np.zeros((b, max_m, self.config.cand_dim), np.float32)
            cp = np.ones((b, max_m), bool)
            g_b = np.zeros((b, self.config.global_dim), np.float32)
            p_b = np.zeros((b, max_p, self.config.prefix_dim), np.float32)
            pp = np.ones((b, max_p), bool)
            rw = np.zeros(b, np.int64); ty = np.zeros(b, np.int64)
            lc = np.full(b, -1, np.int64); gk = np.zeros(b, np.int64)
            gr = np.zeros(b, np.int64)
            cr = np.zeros((b, self.target_dim), np.int64)
            gm = np.zeros((b, self.target_dim), bool)
            ga = np.zeros(b, np.int64); gv = np.zeros(b, np.float32)
            lm = np.zeros((b, self.config.activation_dim), bool)
            la = np.zeros(b, np.int64); lv = np.zeros(b, np.float32)
            sa = np.zeros(b, np.int64); sv = np.zeros(b, np.float32)
            for j, i in enumerate(idx):
                r = recs[i]
                a_b[j, :r.agent_tokens.shape[0]] = r.agent_tokens
                t_b[j, :r.agent_types.shape[0]] = r.agent_types
                ap[j, :r.agent_tokens.shape[0]] = False
                c_b[j, :r.cand_tokens.shape[0]] = r.cand_tokens
                cp[j, :r.cand_tokens.shape[0]] = False
                g_b[j] = r.glob
                p_b[j, :r.prefix_actions.shape[0]] = r.prefix_actions
                pp[j, :r.prefix_actions.shape[0]] = False
                rw[j] = min(r.agent_row, max_n - 1); ty[j] = r.missile_type
                lc[j] = int(r.launch_context); gk[j] = int(r.goal_kind)
                gr[j] = (max_m if int(r.goal_arg_row) == r.cand_tokens.shape[0]
                         else min(int(r.goal_arg_row), max_m))
                k = len(r.cand_rows)
                if k:
                    cr[j, :k] = _remap_candidate_rows(
                        r.cand_rows,
                        r.cand_tokens.shape[0],
                        max_m,
                    )
                    gm[j, :k] = r.goal_mask
                ga[j] = r.goal_action if r.goal_action is not None else 0
                gv[j] = 1.0 if r.goal_action is not None else 0.0
                lm[j] = r.launch_mask; la[j] = r.launch_action or 0
                lv[j] = 1.0 if r.launch_action is not None else 0.0
                sa[j] = r.sat_action or 0; sv[j] = 1.0 if r.sat_action is not None else 0.0
            return (to(a_b), to(t_b, torch.long), to(ap, torch.bool), to(c_b),
                    to(cp, torch.bool), to(g_b), to(p_b), to(pp, torch.bool),
                    to(rw, torch.long), to(ty, torch.long),
                    to(lc, torch.long), to(gk, torch.long), to(gr, torch.long),
                    to(cr, torch.long), to(gm, torch.bool), to(ga, torch.long), to(gv),
                    to(lm, torch.bool), to(la, torch.long), to(lv),
                    to(sa, torch.long), to(sv))

        totals = {
            "bc_loss": 0.0,
            "goal_ce": 0.0,
            "launch_ce": 0.0,
            "sat_ce": 0.0,
            "deploy_ce": 0.0,
        }
        steps = 0
        flight_steps = 0
        for cs in range(0, n, mb):
            idx = order[cs:cs + mb]
            (a_t, t_t, ap_t, c_t, cp_t, g_t, p_t, pp_t, rw_t, ty_t,
             lc_t, gk_t, gr_t, cr_t, gm_t, ga_t, gv_t,
             lm_t, la_t, lv_t, sa_t, sv_t) = chunk(idx)
            b = a_t.shape[0]
            ga_, gc_ = self.actor.encode(a_t, t_t, c_t, g_t, ap_t, cp_t)
            base_h = ga_[torch.arange(b, device=self.device), rw_t]
            memory = torch.cat([ga_, gc_], dim=1)
            memory_pad = torch.cat([
                ap_t, cp_t,
                torch.zeros((b, 1), dtype=torch.bool, device=self.device),
            ], dim=1)
            h = self.actor.condition_on_prefix(
                base_h, memory, memory_pad, p_t, pp_t)
            goal_h = self.actor.after_launch(h, lc_t.clamp(min=0), lc_t >= 0)
            keys = self.actor.goal_key(gc_)
            q = torch.stack([
                self.actor.goal_query[k](goal_h) for k in range(3)], 0)
            q = q[ty_t, torch.arange(b, device=self.device)]
            gathered = keys.gather(1, cr_t.unsqueeze(-1).expand(-1, -1, keys.shape[-1]))
            glogits = (q.unsqueeze(1) * gathered).sum(-1) / (self.config.d_model ** 0.5)
            glogits = glogits.masked_fill(~gm_t, -1e9)
            goal_ce = (torch.nn.functional.cross_entropy(
                glogits, ga_t, reduction="none") * gv_t).sum() / max(gv_t.sum(), 1)
            llogits = self.actor.launch_head(h).masked_fill(~lm_t, -1e9)
            launch_ce = (torch.nn.functional.cross_entropy(
                llogits, la_t, reduction="none") * lv_t).sum() / max(lv_t.sum(), 1)
            goal_arg = gc_.gather(
                1, gr_t.view(-1, 1, 1).expand(-1, 1, gc_.shape[-1])
            ).squeeze(1)
            sat_h = self.actor.after_goal(
                goal_h, gk_t, goal_arg, gv_t.bool())
            slogits = self.actor.sat_head(sat_h)
            sat_ce = (torch.nn.functional.cross_entropy(
                slogits, sa_t, reduction="none") * sv_t).sum() / max(sv_t.sum(), 1)
            loss = goal_ce + launch_ce + sat_ce
            self.opt_actor.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.opt_actor.step()
            totals["bc_loss"] += float(loss.item())
            totals["goal_ce"] += float(goal_ce.item())
            totals["launch_ce"] += float(launch_ce.item())
            totals["sat_ce"] += float(sat_ce.item())
            steps += 1
            flight_steps += 1

        expert_deploy = [
            record for record in self._deploy_records
            if bool(record.get("expert", False))
        ]
        deploy_steps = 0
        deploy_mb = max(1, min(256, mb))
        for start in range(0, len(expert_deploy), deploy_mb):
            records = expert_deploy[start:start + deploy_mb]
            logits = self._deploy_logits_for_records(records)
            targets = torch.as_tensor(
                [int(record["cell"]) for record in records],
                dtype=torch.long, device=self.device,
            )
            deploy_ce = torch.nn.functional.cross_entropy(logits, targets)
            self.opt_actor.zero_grad()
            deploy_ce.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.max_grad_norm)
            self.opt_actor.step()
            value = float(deploy_ce.item())
            totals["bc_loss"] += value
            totals["deploy_ce"] += value
            steps += 1
            deploy_steps += 1

        out = {
            "bc_loss": totals["bc_loss"] / max(steps, 1),
            "goal_ce": totals["goal_ce"] / max(flight_steps, 1),
            "launch_ce": totals["launch_ce"] / max(flight_steps, 1),
            "sat_ce": totals["sat_ce"] / max(flight_steps, 1),
            "deploy_ce": totals["deploy_ce"] / max(deploy_steps, 1),
            "records": float(n),
            "valid_activate": float(sum(
                record.launch_action is not None for record in recs)),
            "valid_goal": float(sum(
                record.goal_action is not None for record in recs)),
            "valid_satellite": float(sum(
                record.sat_action is not None for record in recs)),
            "valid_deploy": float(len(expert_deploy)),
        }
        self.last_metrics = out
        self._records.clear()
        self._deploy_records.clear()
        self._updates_window_rounds = 0
        self.update_count += 1
        self._comp_raw = 0.0; self._comp_discovery = 0.0
        self._comp_retarget = 0.0; self._comp_wave = 0.0
        self._comp_progress = 0.0; self._comp_official = 0.0
        self._comp_causal_credit = 0.0
        self._comp_discovery_credit_raw = 0.0
        self._window_discovery_links = 0
        self._window_discovery_unbound = 0
        self._window_discovery_credited_targets.clear()
        logger.info("[TopFormer] BC #%d | 记录=%d 部署=%d | loss=%.4f "
                    "(goal=%.3f launch=%.3f sat=%.3f deploy=%.3f)",
                    self.update_count, n, len(expert_deploy),
                    out["bc_loss"], out["goal_ce"], out["launch_ce"],
                    out["sat_ce"], out["deploy_ce"])
        return out

    def update(self):
        """条件 PPO 更新（minibatch 分块：场景快照大，整批前向会 OOM）。"""
        n = len(self._records)
        if n == 0:
            return self.last_metrics
        recs = self._records
        p1_search_only = bool(
            self.config.train_stage == "p1_scout"
            and self.config.p1_search_only_actor
        )
        adv_raw, rets = self.compute_smdp_gae(
            recs, gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            causal_credit_coef=self.config.causal_credit_coef,
            discovery_credit_coef=self.config.discovery_credit_coef,
            search_only_actor=p1_search_only)
        for record, value_target in zip(recs, rets):
            record.mc = float(value_target)
        actor_main_mask = np.asarray([
            (int(record.goal_kind) == 2) if p1_search_only else True
            for record in recs
        ], dtype=bool)
        if not bool(actor_main_mask.any()):
            raise ValueError("P1.1 update has no search actor records")
        actor_adv = adv_raw[actor_main_mask]
        adv_mu = float(actor_adv.mean())
        adv_sd = float(actor_adv.std() + 1e-8)
        adv = np.zeros_like(adv_raw, dtype=np.float32)
        adv[actor_main_mask] = (
            (actor_adv - adv_mu) / adv_sd
        ).astype(np.float32)
        discovery_aux_weights, discovery_aux_targets, discovery_aux_records = (
            self.discovery_aux_weights(recs) if p1_search_only else (
                np.zeros(n, dtype=np.float32), 0, 0)
        )
        # G 分布诊断：分位数 + 正负占比（定位异常来源）
        qs = np.percentile(rets, [0, 5, 50, 95, 100])
        neg_frac = float((rets < 0).mean())
        logger.info("[TopFormer] G分布: min=%.0f p5=%.0f p50=%.0f p95=%.0f max=%.0f 负值占比=%.0f%%",
                    qs[0], qs[1], qs[2], qs[3], qs[4], neg_frac * 100)
        # 最负 3 条记录的身份（定位异常来源）
        order_desc = np.argsort(rets)[:3]
        for j in order_desc:
            r = recs[j]
            logger.info("[TopFormer] 最负记录: ret=%.0f ep=%d agent=%d step=%d 弹型=%d "
                        "launch=%s goal=%s sat=%s",
                        rets[j], r.episode, r.agent_id, r.step, r.missile_type,
                        r.launch_action, r.goal_action, r.sat_action)
        ret_norm = rets.astype(np.float32)

        mb = max(8, int(self.config.minibatch_records))
        order = np.random.permutation(n)
        to = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.device)
        # 熵闸门：上轮实测熵过低 → 熵系数放大 10 倍（防止首轮即坍缩死锁）
        ent_coef = self.config.entropy_coef
        if (self.update_count > 0
                and float(self.last_metrics.get("entropy", 1.0))
                < self.config.entropy_floor):
            ent_coef = self.config.entropy_coef * 10.0
        # ── 专家窗口：行为克隆（监督 CE）替代 PPO ──
        if self._expert_bc_pending():
            return self._bc_update(recs, n)

        replay_gap = 0.0
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                  "approx_kl": 0.0, "ret": float(rets.mean()),
                  "replay_gap": replay_gap, "discovery_aux_loss": 0.0}
        updates = 0
        actor_updates = 0
        stop = False

        def chunk_tensors(idx):
            cr = [recs[i] for i in idx]
            b = len(cr)
            max_n = max(r.agent_tokens.shape[0] for r in cr)
            max_m = max(r.cand_tokens.shape[0] for r in cr)
            max_p = max(r.prefix_actions.shape[0] for r in cr)
            agent_b = np.zeros(
                (b, max_n, self.config.agent_dim), dtype=np.float32)
            types_b = np.zeros((b, max_n), dtype=np.int64)
            apad_b = np.ones((b, max_n), dtype=bool)
            cand_b = np.zeros(
                (b, max_m, self.config.cand_dim), dtype=np.float32)
            cpad_b = np.ones((b, max_m), dtype=bool)
            glob_b = np.zeros((b, self.config.global_dim), dtype=np.float32)
            prefix_b = np.zeros(
                (b, max_p, self.config.prefix_dim), dtype=np.float32)
            ppad_b = np.ones((b, max_p), dtype=bool)
            row_b = np.zeros(b, dtype=np.int64)
            type_b = np.zeros(b, dtype=np.int64)
            lctx_b = np.full(b, -1, dtype=np.int64)
            gkind_b = np.zeros(b, dtype=np.int64)
            gargrow_b = np.zeros(b, dtype=np.int64)
            crow_b = np.full((b, self.target_dim), 0, dtype=np.int64)
            gm_b = np.zeros((b, self.target_dim), dtype=bool)
            gact_b = np.zeros(b, dtype=np.int64)
            gvalid_b = np.zeros(b, dtype=np.float32)
            lmask_b = np.zeros(
                (b, self.config.activation_dim), dtype=bool)
            lact_b = np.zeros(b, dtype=np.int64)
            lvalid_b = np.zeros(b, dtype=np.float32)
            lold_b = np.zeros(b, dtype=np.float32)
            gold_b = np.zeros(b, dtype=np.float32)
            satact_b = np.zeros(b, dtype=np.int64)
            satvalid_b = np.zeros(b, dtype=np.float32)
            satold_b = np.zeros(b, dtype=np.float32)
            adv_b = np.zeros(b, dtype=np.float32)
            actor_main_b = np.zeros(b, dtype=np.float32)
            discovery_aux_b = np.zeros(b, dtype=np.float32)
            p1_search_phase_b = np.zeros(b, dtype=bool)
            ret_b = np.zeros(b, dtype=np.float32)
            ep_b = np.zeros(b, dtype=np.int64)
            vold_b = np.zeros(b, dtype=np.float32)
            for j, r in enumerate(cr):
                agent_b[j, :r.agent_tokens.shape[0]] = r.agent_tokens
                types_b[j, :r.agent_types.shape[0]] = r.agent_types
                apad_b[j, :r.agent_tokens.shape[0]] = False
                cand_b[j, :r.cand_tokens.shape[0]] = r.cand_tokens
                cpad_b[j, :r.cand_tokens.shape[0]] = False
                glob_b[j] = r.glob
                prefix_b[j, :r.prefix_actions.shape[0]] = r.prefix_actions
                ppad_b[j, :r.prefix_actions.shape[0]] = False
                row_b[j] = min(r.agent_row, max_n - 1)
                type_b[j] = r.missile_type
                lctx_b[j] = int(r.launch_context)
                gkind_b[j] = int(r.goal_kind)
                gargrow_b[j] = (
                    max_m if int(r.goal_arg_row) == r.cand_tokens.shape[0]
                    else min(int(r.goal_arg_row), max_m)
                )
                k = len(r.cand_rows)
                if k:
                    crow_b[j, :k] = _remap_candidate_rows(
                        r.cand_rows,
                        r.cand_tokens.shape[0],
                        max_m,
                    )
                    gm_b[j, :k] = r.goal_mask
                gact_b[j] = r.goal_action if r.goal_action is not None else 0
                gvalid_b[j] = 1.0 if r.goal_action is not None else 0.0
                lmask_b[j] = r.launch_mask
                lact_b[j] = r.launch_action if r.launch_action is not None else 0
                lvalid_b[j] = 1.0 if r.launch_action is not None else 0.0
                lold_b[j] = r.launch_old_lp
                gold_b[j] = r.goal_old_lp
                satact_b[j] = r.sat_action if r.sat_action is not None else 0
                satvalid_b[j] = 1.0 if r.sat_action is not None else 0.0
                satold_b[j] = r.sat_old_lp
                adv_b[j] = adv[idx[j]]
                actor_main_b[j] = float(actor_main_mask[idx[j]])
                discovery_aux_b[j] = discovery_aux_weights[idx[j]]
                # Buffers collected before P1.1 have no explicit phase bit.
                # Their search and target phases were disjoint, making the
                # recorded goal kind the exact compatible fallback.
                p1_search_phase_b[j] = _record_p1_search_phase(r)
                ret_b[j] = ret_norm[idx[j]]
                ep_b[j] = r.episode
                vold_b[j] = float(r.value)
            return (to(agent_b), to(types_b, torch.long), to(apad_b, torch.bool),
                    to(cand_b), to(cpad_b, torch.bool), to(glob_b),
                    to(prefix_b), to(ppad_b, torch.bool),
                    to(row_b, torch.long), to(type_b, torch.long),
                    to(lctx_b, torch.long), to(gkind_b, torch.long),
                    to(gargrow_b, torch.long),
                    to(crow_b, torch.long), to(gm_b, torch.bool),
                    to(gact_b, torch.long), to(gvalid_b),
                    to(lmask_b, torch.bool), to(lact_b, torch.long), to(lvalid_b),
                    to(lold_b), to(gold_b), to(adv_b),
                    to(actor_main_b), to(discovery_aux_b),
                    to(p1_search_phase_b, torch.bool),
                    to(ret_b), to(vold_b),
                    to(satact_b, torch.long), to(satvalid_b), to(satold_b),
                    to(ep_b, torch.long))

        def policy_forward(agent_t, types_t, apad_t, cand_t, cpad_t, glob_t,
                           prefix_t, ppad_t, row_t, type_t, lctx_t,
                           gkind_t, p1_search_phase_t, gargrow_t, crow_t,
                           gm_t, gact_t,
                           gvalid_t, lmask_t, lact_t, satact_t, satvalid_t):
            b = agent_t.shape[0]
            ga, gc = self.actor.encode(
                agent_t, types_t, cand_t, glob_t, apad_t, cpad_t)
            base_h = ga[torch.arange(b, device=self.device), row_t]
            memory = torch.cat([ga, gc], dim=1)
            memory_pad = torch.cat([
                apad_t, cpad_t,
                torch.zeros((b, 1), dtype=torch.bool, device=self.device),
            ], dim=1)
            h = self.actor.condition_on_prefix(
                base_h, memory, memory_pad, prefix_t, ppad_t)

            launch_logits = self.actor.launch_head(h).masked_fill(~lmask_t, -1e9)
            launch_lsm = torch.log_softmax(launch_logits, dim=-1)
            launch_lp = launch_lsm.gather(1, lact_t.unsqueeze(1)).squeeze(1)
            launch_legal = lmask_t.sum(-1).clamp(min=2).float()
            launch_ent = (-(launch_lsm.exp() * launch_lsm).sum(-1)
                          / torch.log(launch_legal))

            goal_h = self.actor.after_launch(
                h, lctx_t.clamp(min=0), lctx_t >= 0)
            keys = self.actor.goal_key(gc)
            q = torch.stack([
                self.actor.goal_query[k](goal_h) for k in range(3)
            ], 0)
            q = q[type_t, torch.arange(b, device=self.device)]
            if self.config.p1_split_search_query:
                search_query = self.actor.l_search_query(goal_h)
                use_search_query = (
                    (type_t == 2) & p1_search_phase_t
                ).unsqueeze(-1)
                q = torch.where(use_search_query, search_query, q)
            gathered = keys.gather(
                1, crow_t.unsqueeze(-1).expand(-1, -1, keys.shape[-1]))
            goal_logits = (q.unsqueeze(1) * gathered).sum(-1) / (
                self.config.d_model ** 0.5)
            goal_logits = goal_logits.masked_fill(~gm_t, -1e9)
            goal_lsm = torch.log_softmax(goal_logits, dim=-1)
            goal_lp = goal_lsm.gather(1, gact_t.unsqueeze(1)).squeeze(1)
            goal_legal = gm_t.sum(-1).clamp(min=2).float()
            goal_ent = (-(goal_lsm.exp() * goal_lsm).sum(-1)
                        / torch.log(goal_legal))

            goal_arg = gc.gather(
                1, gargrow_t.view(-1, 1, 1).expand(-1, 1, gc.shape[-1])
            ).squeeze(1)
            sat_h = self.actor.after_goal(
                goal_h, gkind_t, goal_arg, gvalid_t.bool())
            sat_lsm = torch.log_softmax(self.actor.sat_head(sat_h), dim=-1)
            sat_lp = sat_lsm.gather(1, satact_t.unsqueeze(1)).squeeze(1)
            sat_ent = -(sat_lsm.exp() * sat_lsm).sum(-1) / np.log(2.0)
            values = self.critic(
                agent_t, cand_t, glob_t, apad_t, cpad_t,
                prefix_actions=prefix_t, prefix_pad=ppad_t)
            return (launch_lp, goal_lp, sat_lp,
                    launch_ent, goal_ent, sat_ent, values)

        # 重放一致性诊断：用未更新的当前参数重算全部记录的联合 log-prob，
        # 与采样时记录的 old_lp 之比应为 1（文档 §10.3：ratio≠1 说明
        # dropout/mask/快照/重算路径不一致——本网络无 dropout，主要
        # 检查掩码与场景快照的重放保真）。逐 minibatch 扫描避免一次性
        # 搬运全部场景快照，同时让 replay_gap 真正覆盖合并后的完整窗口。
        replay_gap = 0.0
        with torch.no_grad():
            for replay_start in range(0, n, mb):
                idx0 = order[replay_start:replay_start + mb]
                (agent_t, types_t, apad_t, cand_t, cpad_t, glob_t,
                 prefix_t, ppad_t, row_t, type_t, lctx_t, gkind_t,
                 gargrow_t, crow_t, gm_t, gact_t, gvalid_t, lmask_t,
                 lact_t, lvalid_t, lold_t, gold_t, _adv_t,
                 _actor_main_t, _discovery_aux_t, _p1_search_phase_t,
                 _ret_t, _vold_t, satact_t, satvalid_t, satold_t,
                 _ep_t) = chunk_tensors(idx0)
                llp, glp, slp, *_ = policy_forward(
                    agent_t, types_t, apad_t, cand_t, cpad_t, glob_t,
                    prefix_t, ppad_t, row_t, type_t, lctx_t, gkind_t,
                    _p1_search_phase_t, gargrow_t, crow_t, gm_t, gact_t,
                    gvalid_t, lmask_t,
                    lact_t, satact_t, satvalid_t)
                replay_joint_new = (
                    llp * lvalid_t + glp * gvalid_t + slp * satvalid_t
                )
                replay_diff = (
                    replay_joint_new
                    - (lold_t * lvalid_t + gold_t * gvalid_t
                       + satold_t * satvalid_t)
                ).abs().max()
                replay_gap = max(replay_gap, float(replay_diff.item()))
        totals["replay_gap"] = replay_gap
        # 后期学习率衰减：高方差段更新失稳的对策（D2 观测：中段冲高后回落）
        # 第 8 次更新后减半，第 16 次后再减半，下限 2.5e-5
        self._apply_lr_schedule()

        # Deployment is replayed against the very same pre-update actor
        # snapshot and contributes to the first optimizer step.  This avoids
        # comparing old placement log-probs with an actor already changed by
        # flight PPO.
        deploy_loss, dep_out = self._deploy_policy_loss()
        deploy_pending = deploy_loss is not None
        tracked_groups = {
            "shared": list(self.actor.agent_proj.parameters())
                      + list(self.actor.cand_proj.parameters())
                      + list(self.actor.global_proj.parameters())
                      + list(self.actor.type_emb.parameters())
                      + list(self.actor.encoder.parameters())
                      + list(self.actor.action_decoder.parameters())
                      + list(self.actor.action_prefix_proj.parameters())
                      + list(self.actor.action_pos_emb.parameters())
                      + [self.actor.decision_query, self.actor.keep_token],
            "activate": list(self.actor.launch_head.parameters())
                        + list(self.actor.launch_action_emb.parameters()),
            "goal": list(self.actor.goal_query.parameters())
                    + list(self.actor.l_search_query.parameters())
                    + list(self.actor.goal_key.parameters())
                    + list(self.actor.goal_kind_emb.parameters())
                    + list(self.actor.goal_arg_proj.parameters()),
            "satellite": list(self.actor.sat_head.parameters()),
            "deploy": list(self.actor.deploy_cand_proj.parameters())
                      + list(self.actor.deploy_query.parameters())
                      + list(self.actor.deploy_key.parameters()),
        }
        parameter_before = {
            name: [parameter.detach().clone() for parameter in parameters]
            for name, parameters in tracked_groups.items()
        }
        grad_norms = {name: 0.0 for name in tracked_groups}

        for _ in range(self.config.epochs):
            if stop:
                break
            for cs in range(0, n, mb):
                idx = order[cs:cs + mb]
                (agent_t, types_t, apad_t, cand_t, cpad_t, glob_t,
                 prefix_t, ppad_t, row_t, type_t, lctx_t, gkind_t, gargrow_t,
                 crow_t, gm_t, gact_t, gvalid_t,
                 lmask_t, lact_t, lvalid_t, lold_t, gold_t,
                 adv_t, actor_main_t, discovery_aux_t, p1_search_phase_t,
                 ret_t, vold_t,
                 satact_t, satvalid_t, satold_t, ep_t) = chunk_tensors(idx)
                (launch_lp_new, goal_lp_new, sat_lp_new,
                 launch_ent, goal_ent, sat_ent, v_pred) = policy_forward(
                    agent_t, types_t, apad_t, cand_t, cpad_t, glob_t,
                    prefix_t, ppad_t, row_t, type_t, lctx_t, gkind_t,
                    p1_search_phase_t, gargrow_t, crow_t, gm_t, gact_t,
                    gvalid_t, lmask_t,
                    lact_t, satact_t, satvalid_t)

                old_joint = (lold_t * lvalid_t + gold_t * gvalid_t
                             + satold_t * satvalid_t)
                new_joint = (launch_lp_new * lvalid_t + goal_lp_new * gvalid_t
                             + sat_lp_new * satvalid_t)
                ratio = torch.exp(new_joint - old_joint)
                per_rec = -torch.min(
                    ratio * adv_t,
                    torch.clamp(ratio, 1 - self.config.clip_ratio,
                                1 + self.config.clip_ratio) * adv_t)
                # 近似正则（导师已确认保留近似但需标注）：对激活头条件熵
                # 直接相加回传，未含分支概率项梯度
                field_count = (lvalid_t + gvalid_t + satvalid_t).clamp(min=1.0)
                per_entropy = ((launch_ent * lvalid_t + goal_ent * gvalid_t
                                + sat_ent * satvalid_t) / field_count)
                if p1_search_only:
                    actor_count = actor_main_t.sum()
                    pl = (per_rec * actor_main_t).sum() / actor_count.clamp(min=1.0)
                    ent = (per_entropy * actor_main_t).sum() / actor_count.clamp(min=1.0)
                    discovery_aux_per = -torch.min(
                        ratio,
                        torch.clamp(
                            ratio, 1 - self.config.clip_ratio,
                            1 + self.config.clip_ratio),
                    )
                    discovery_aux_loss = (
                        discovery_aux_per * discovery_aux_t
                    ).sum()
                    actor_active = bool(actor_count.item() > 0.0)
                else:
                    pl = per_rec.mean()
                    ent = per_entropy.mean()
                    discovery_aux_loss = pl.new_zeros(())
                    actor_active = True
                vl = 0.5 * (v_pred - ret_t).pow(2).mean()

                if actor_active or deploy_pending:
                    self.opt_actor.zero_grad()
                    actor_loss = (
                        pl - ent_coef * ent
                        + float(self.config.p1_discovery_aux_coef)
                        * discovery_aux_loss
                    )
                    if deploy_pending:
                        actor_loss = actor_loss + deploy_loss
                    actor_loss.backward()
                    for name, parameters in tracked_groups.items():
                        squared = sum(
                            float(parameter.grad.detach().pow(2).sum().item())
                            for parameter in parameters
                            if parameter.grad is not None)
                        grad_norms[name] = max(
                            grad_norms[name], squared ** 0.5)
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.config.max_grad_norm)
                    self.opt_actor.step()
                    deploy_pending = False
                    actor_updates += 1
                self.opt_critic.zero_grad()
                (self.config.value_coef * vl).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.opt_critic.step()

                with torch.no_grad():
                    # k3 对称估计（非负）：mean log-ratio 只能捕捉正向漂移，
                    # 熵项导致的负向漂移（系统性降低采样动作概率）会漏检
                    ratio_val = torch.exp(new_joint - old_joint)
                    per_kl = ((ratio_val - 1.0) - (new_joint - old_joint))
                    if p1_search_only:
                        kl = float((per_kl * actor_main_t).sum().item()
                                   / max(float(actor_main_t.sum().item()), 1.0))
                    else:
                        kl = float(per_kl.mean().item())
                if actor_active:
                    totals["policy_loss"] += float(pl.item())
                    totals["entropy"] += float(ent.item())
                    totals["approx_kl"] += kl
                    totals["discovery_aux_loss"] += float(
                        discovery_aux_loss.item())
                totals["value_loss"] += float(vl.item())
                updates += 1
                if actor_active and kl > 1.5 * self.config.target_kl:
                    stop = True
                    break

        if self._deploy_records:
            self._deploy_records.clear()
        self.update_count += 1
        out = {
            "policy_loss": totals["policy_loss"] / max(actor_updates, 1),
            "value_loss": totals["value_loss"] / max(updates, 1),
            "entropy": totals["entropy"] / max(actor_updates, 1),
            "approx_kl": totals["approx_kl"] / max(actor_updates, 1),
            "discovery_aux_loss": (
                totals["discovery_aux_loss"] / max(actor_updates, 1)
            ),
        }
        out["ret"] = float(rets.mean())
        out["replay_gap"] = replay_gap
        if dep_out:
            out.update(dep_out)
        out["records"] = float(n)
        rounds_in_batch = max(self._updates_window_rounds, 1)
        out.update({
            "valid_activate": float(sum(
                record.launch_action is not None for record in recs)),
            "valid_goal": float(sum(
                record.goal_action is not None for record in recs)),
            "valid_satellite": float(sum(
                record.sat_action is not None for record in recs)),
            "valid_deploy": float(
                dep_out.get("deploy_records", 0.0) if dep_out else 0.0),
            "reward_official_per_round": float(
                self._comp_official / rounds_in_batch),
            "reward_dense_per_round": float(
                self._comp_progress / rounds_in_batch),
            "causal_credit_per_round": float(
                self._comp_causal_credit / rounds_in_batch),
            "discovery_credit_raw_per_round": float(
                self._comp_discovery_credit_raw / rounds_in_batch),
            "discovery_credit_links": float(
                self._window_discovery_links),
            "discovery_credit_unbound": float(
                self._window_discovery_unbound),
            "discovery_credit_targets": float(len(
                self._window_discovery_credited_targets)),
            "discovery_credit_coef": float(
                self.config.discovery_credit_coef),
            "p1_search_actor_records": float(actor_main_mask.sum()),
            "p1_target_critic_only_records": float(
                n - int(actor_main_mask.sum()) if p1_search_only else 0),
            "p1_discovery_aux_targets": float(discovery_aux_targets),
            "p1_discovery_aux_records": float(discovery_aux_records),
            "p1_discovery_aux_weight_sum": float(
                discovery_aux_weights.sum()),
            "p1_discovery_aux_coef": float(
                self.config.p1_discovery_aux_coef),
            "p1_actor_adv_mean": adv_mu,
            "p1_actor_adv_std": adv_sd,
            "p1_actor_minibatch_updates": float(actor_updates),
            "p1_goal_records": float(self._p1_window_goal_records),
            "p1_filtered_events": float(self._p1_window_filtered_events),
            "p1_actor_trainable_numel": float(sum(
                parameter.numel() for parameter in self.actor.parameters()
                if parameter.requires_grad
            )),
        })
        for name, parameters in tracked_groups.items():
            squared_delta = 0.0
            for before, after in zip(parameter_before[name], parameters):
                squared_delta += float(
                    (after.detach() - before).pow(2).sum().item())
            out[f"grad_norm_{name}"] = float(grad_norms[name])
            out[f"param_delta_{name}"] = float(squared_delta ** 0.5)
        self.last_metrics = out
        self._records.clear()
        # 分量记账：有符号与绝对值同时报告（正负塑形抵消会掩盖实际强度，
        # 导师复审#3），official 为引擎计分的逐步增量（不变量已校验）
        logger.info(
            "[TopFormer] 更新 #%d | 记录=%d 回合均G=%.3f | policy=%.4f value=%.4f "
            "kl=%.4f 熵=%.3f replay_gap=%.1e | 分量/回合(符号|绝对): official=%.1f "
            "raw=%.1f 进度=%.1f 发现=%.1f|%.1f 换靶=%.1f|%.1f 波次=%.1f",
            self.update_count, n, float(rets.mean()),
            out["policy_loss"], out["value_loss"], out["approx_kl"], out["entropy"],
            out.get("replay_gap", 0.0),
            self._comp_official / rounds_in_batch,
            self._comp_raw / rounds_in_batch,
            self._comp_progress / rounds_in_batch,
            self._comp_discovery / rounds_in_batch, abs(self._comp_discovery) / rounds_in_batch,
            self._comp_retarget / rounds_in_batch, abs(self._comp_retarget) / rounds_in_batch,
            self._comp_wave / rounds_in_batch)
        logger.info(
            "[TopFormer] 四字段有效样本 activate=%d goal=%d satellite=%d deploy=%d "
            "| replay flight=%.1e deploy=%.1e | grad(shared/a/g/s/d)="
            "%.2e/%.2e/%.2e/%.2e/%.2e | delta=%.2e/%.2e/%.2e/%.2e/%.2e "
            "| causal/回合=%.2f",
            int(out["valid_activate"]), int(out["valid_goal"]),
            int(out["valid_satellite"]), int(out["valid_deploy"]),
            out.get("replay_gap", 0.0), out.get("deploy_replay_gap", 0.0),
            out["grad_norm_shared"], out["grad_norm_activate"],
            out["grad_norm_goal"], out["grad_norm_satellite"],
            out["grad_norm_deploy"], out["param_delta_shared"],
            out["param_delta_activate"], out["param_delta_goal"],
            out["param_delta_satellite"], out["param_delta_deploy"],
            out["causal_credit_per_round"],
        )
        logger.info(
            "[TopFormer] 发现因果信用 links=%d unbound=%d targets=%d "
            "|后续真实官方分/回合=%.2f actor_coef=%.3f "
            "| P1.1 search/target=%d/%d aux_targets/records=%d/%d "
            "aux_weight=%.3f aux_coef=%.3f aux_loss=%.4f",
            int(out["discovery_credit_links"]),
            int(out["discovery_credit_unbound"]),
            int(out["discovery_credit_targets"]),
            out["discovery_credit_raw_per_round"],
            out["discovery_credit_coef"],
            int(out["p1_search_actor_records"]),
            int(out["p1_target_critic_only_records"]),
            int(out["p1_discovery_aux_targets"]),
            int(out["p1_discovery_aux_records"]),
            out["p1_discovery_aux_weight_sum"],
            out["p1_discovery_aux_coef"],
            out["discovery_aux_loss"],
        )
        self._updates_window_rounds = 0
        self._comp_raw = 0.0; self._comp_discovery = 0.0
        self._comp_retarget = 0.0; self._comp_wave = 0.0
        self._comp_progress = 0.0; self._comp_official = 0.0
        self._comp_causal_credit = 0.0
        self._comp_discovery_credit_raw = 0.0
        self._p1_window_goal_records = 0
        self._p1_window_filtered_events = 0
        self._window_discovery_links = 0
        self._window_discovery_unbound = 0
        self._window_discovery_credited_targets.clear()
        return out


    # ---------- 持久化 ----------

    def save(self, path: str, *, include_training_state: bool = True):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp")
        training_state = {
            "best_score": self.best_score,
            "evaluated_only": not include_training_state,
        }
        if include_training_state:
            training_state.update({
                "rounds_done": self._rounds_done,
                "expert_bc_count": self._expert_bc_count,
                "g_run_mu": self._g_run_mu,
                "g_run_n": self._g_run_n,
                "g_run_m2": self._g_run_m2,
                "updates_window_rounds": self._updates_window_rounds,
                "p1_window_goal_records": self._p1_window_goal_records,
                "p1_window_filtered_events": self._p1_window_filtered_events,
                "window_discovery_links": self._window_discovery_links,
                "window_discovery_unbound": self._window_discovery_unbound,
                "window_discovery_credited_targets": list(
                    self._window_discovery_credited_targets),
                "last_metrics": self.last_metrics,
                "records": self._records,
                "deploy_records": self._deploy_records,
                "components": {
                    "raw": self._comp_raw,
                    "discovery": self._comp_discovery,
                    "retarget": self._comp_retarget,
                    "wave": self._comp_wave,
                    "progress": self._comp_progress,
                    "official": self._comp_official,
                    "causal_credit": self._comp_causal_credit,
                    "discovery_credit_raw": self._comp_discovery_credit_raw,
                },
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
            })
        torch.save({
            "algorithm": "topformer_mappo",
            "schema_version": TOPFORMER_SCHEMA_VERSION,
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "update_count": self.update_count,
            "training_state": training_state,
        }, tmp)
        tmp.replace(target)

    def save_evaluated(self, path: str):
        """Save the exact evaluated weights without an incomplete rollout buffer."""

        self.save(path, include_training_state=False)

    @staticmethod
    def _load_shape_compatible(module: nn.Module, saved: dict) -> tuple[int, int]:
        """Load matching tensors and copy overlapping axes for expanded inputs."""
        current = module.state_dict()
        loaded = 0
        partial = 0
        for name, value in saved.items():
            if name not in current:
                continue
            target = current[name]
            source = value.to(device=target.device, dtype=target.dtype)
            if source.shape == target.shape:
                current[name] = source
                loaded += 1
                continue
            if source.ndim != target.ndim:
                continue
            slices = tuple(
                slice(0, min(int(a), int(b)))
                for a, b in zip(source.shape, target.shape)
            )
            copied = target.clone()
            copied[slices] = source[slices]
            current[name] = copied
            partial += 1
        module.load_state_dict(current, strict=True)
        return loaded, partial

    @staticmethod
    def _canonical_actor_state(saved: dict) -> tuple[dict, bool]:
        """Add the P1.1 search query to pre-P1.1 actor snapshots.

        The new module was appended after every legacy actor parameter.  Its
        exact initial contract is a clone of the existing L query, so this is
        a behaviour-preserving migration rather than a random initialization.
        """

        result = saved.copy()
        weight = "l_search_query.weight"
        bias = "l_search_query.bias"
        present = (weight in result, bias in result)
        if all(present):
            return result, False
        if any(present):
            raise ValueError(
                "checkpoint has an incomplete l_search_query state")
        source_weight = result.get("goal_query.2.weight")
        source_bias = result.get("goal_query.2.bias")
        if source_weight is None or source_bias is None:
            raise ValueError(
                "checkpoint cannot initialize l_search_query: "
                "goal_query.2 is missing")
        result[weight] = source_weight.detach().clone()
        result[bias] = source_bias.detach().clone()
        return result, True

    def _canonical_actor_optimizer_state(
        self, saved: dict, *, actor_state_migrated: bool,
    ) -> dict:
        """Extend legacy Adam parameter ids without changing old moments."""

        result = copy.deepcopy(saved)
        if not actor_state_migrated:
            return result
        expected = self.opt_actor.state_dict()
        saved_groups = result.get("param_groups") or []
        expected_groups = expected.get("param_groups") or []
        if len(saved_groups) != len(expected_groups):
            raise ValueError(
                "legacy actor Adam param-group count cannot be migrated")
        added = 0
        for old_group, new_group in zip(saved_groups, expected_groups):
            old_params = list(old_group.get("params") or [])
            new_params = list(new_group.get("params") or [])
            if old_params != new_params[:len(old_params)]:
                raise ValueError(
                    "legacy actor Adam parameter order cannot be migrated")
            added += len(new_params) - len(old_params)
            old_group["params"] = new_params
        if added != 2:
            raise ValueError(
                "legacy actor Adam migration expected exactly two appended "
                f"parameters, found {added}")
        # No state is fabricated for the cloned query. Adam initializes its
        # moments lazily on the first genuine P1.1 gradient.
        return result

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("algorithm") != "topformer_mappo":
            raise ValueError("该 checkpoint 不是 TopFormer 模型")
        version = int(ckpt.get("schema_version", 0))
        # Immutable source identity for P0/P1 evaluation summaries.  This is
        # distinct from the in-memory counters, which intentionally reset to
        # zero for UPDATE_RESET=1 weights-only experiments.
        source_state = ckpt.get("training_state") or {}
        self._source_checkpoint_schema_version = version
        self._source_checkpoint_update_count = int(ckpt.get("update_count", 0))
        self._source_checkpoint_rounds_done = int(
            source_state.get("rounds_done", 0)
        )
        reset_requested = os.getenv("UPDATE_RESET", "0") == "1"
        legacy_weights_only = version in (1, 2) and reset_requested
        saved_config = ckpt.get("config") or {}
        state = ckpt.get("training_state") or {}
        actor_state, actor_state_migrated = self._canonical_actor_state(
            ckpt["actor"])
        evaluated_only = bool(state.get("evaluated_only", False))
        # Behavioural phase flags are checkpoint semantics.  When the caller
        # did not explicitly provide an override, restore them so evaluating a
        # goal-only/sat-disabled/deploy-disabled best snapshot reproduces the
        # policy that actually earned that score.
        for config_key, environment_key in (
            ("upper_phase", "UPPER_PHASE"),
            ("sat_trainable", "SAT_LEARN"),
            ("deploy_trainable", "DEPLOY_LEARN"),
            ("eval_sample", "EVAL_SAMPLE"),
            ("train_stage", "TOPFORMER_TRAIN_STAGE"),
            ("l_scout_enabled", "L_SCOUT_ENABLED"),
            ("l_scout_count", "L_SCOUT_COUNT"),
            ("l_scout_expected_targets", "L_SCOUT_EXPECTED_TARGETS"),
            ("l_scout_coverage_threshold", "L_SCOUT_COVERAGE_THRESHOLD"),
            ("l_scout_quiet_steps", "L_SCOUT_QUIET_STEPS"),
            ("l_scout_deadline", "L_SCOUT_DEADLINE"),
            ("l_attack_min_discovered", "L_ATTACK_MIN_DISCOVERED"),
            ("l_scout_sector_dwell", "L_SCOUT_SECTOR_DWELL"),
            ("p1_search_only_actor", "P1_SEARCH_ONLY_ACTOR"),
            ("p1_split_search_query", "P1_SPLIT_SEARCH_QUERY"),
            ("p1_discovery_aux_coef", "P1_DISCOVERY_AUX_COEF"),
        ):
            if environment_key not in os.environ and config_key in saved_config:
                setattr(self.config, config_key, saved_config[config_key])
        if "l_land_targets" in saved_config:
            self.config.l_land_targets = bool(saved_config["l_land_targets"])
        self.config.train_stage = str(self.config.train_stage).strip().lower()
        self._configure_actor_trainability()
        if version != TOPFORMER_SCHEMA_VERSION and not legacy_weights_only:
            raise ValueError(
                f"TopFormer 协议版本 {version} != {TOPFORMER_SCHEMA_VERSION}；"
                "坐标与动作槽语义已修复；旧 checkpoint 仅允许 UPDATE_RESET=1 "
                "作为 weights-only 初始化，不能直接续训"
            )
        # Exact resume restores rollout records, optimizer state and reward
        # statistics.  State-dict shapes alone cannot detect a changed search
        # slot protocol or different SMDP/reward semantics, so reject those
        # mismatches before mutating this policy object.
        if version == TOPFORMER_SCHEMA_VERSION and not reset_requested:
            protocol_keys = (
                "agent_dim", "cand_dim", "global_dim", "prefix_dim",
                "activation_dim", "d_model", "n_head", "enc_layers",
                "ff_dim", "mat_decoder_layers", "max_goal_slots",
                "search_rows", "search_cols", "max_steps",
                "upper_phase", "sat_trainable", "deploy_trainable",
                "eval_sample", "l_land_targets",
                "train_stage", "l_scout_enabled", "l_scout_count",
                "l_scout_expected_targets", "l_scout_coverage_threshold",
                "l_scout_quiet_steps", "l_scout_deadline",
                "l_attack_min_discovered", "l_scout_sector_dwell",
                "p1_search_only_actor", "p1_split_search_query",
            )
            training_keys = (
                "learning_rate", "critic_lr", "clip_ratio", "value_coef",
                "entropy_coef", "max_grad_norm", "epochs", "target_kl",
                "entropy_floor", "minibatch_records", "update_every_rounds",
                "min_update_records", "gamma",
                "gae_lambda", "reward_scale", "progress_shaping_coef",
                "coverage_shaping_coef", "causal_credit_coef",
                "discovery_credit_coef", "p1_discovery_aux_coef",
                "discovery_bonus", "retarget_penalty", "expert_rounds",
            )
            resume_keys = protocol_keys + (() if evaluated_only else training_keys)
            current_config = asdict(self.config)
            mismatches = []
            phase_override = os.getenv(
                "TOPFORMER_ALLOW_PHASE_OVERRIDE", "0"
            ) == "1"
            phase_keys = {"upper_phase", "sat_trainable", "deploy_trainable"}
            eval_override = os.getenv(
                "TOPFORMER_ALLOW_EVAL_OVERRIDE", "0"
            ) == "1"
            implied_legacy_defaults = {
                "train_stage": "joint",
                "l_scout_enabled": False,
                "l_scout_count": 8,
                "l_scout_expected_targets": 4,
                "l_scout_coverage_threshold": 0.75,
                "l_scout_quiet_steps": 180,
                "l_scout_deadline": 600,
                "l_attack_min_discovered": 4,
                "l_scout_sector_dwell": 10,
                "discovery_credit_coef": 0.0,
                "p1_search_only_actor": False,
                "p1_split_search_query": False,
                "p1_discovery_aux_coef": 0.0,
            }
            for key in resume_keys:
                if key not in saved_config:
                    if key not in implied_legacy_defaults:
                        mismatches.append(
                            f"{key}=<missing>->{current_config[key]!r}")
                        continue
                    old = implied_legacy_defaults[key]
                else:
                    old = saved_config[key]
                new = current_config[key]
                equal = (
                    bool(np.isclose(old, new, rtol=0.0, atol=1e-12))
                    if isinstance(old, (int, float))
                    and isinstance(new, (int, float))
                    else old == new
                )
                if not equal:
                    if phase_override and key in phase_keys:
                        logger.warning(
                            "[TopFormer] 显式阶段覆盖 %s: checkpoint=%r current=%r",
                            key, old, new,
                        )
                        continue
                    if eval_override and key == "eval_sample":
                        logger.warning(
                            "[TopFormer] 显式评估行为覆盖 %s: "
                            "checkpoint=%r current=%r",
                            key, old, new,
                        )
                        continue
                    mismatches.append(f"{key}={old!r}->{new!r}")
            if mismatches:
                raise ValueError(
                    "TopFormer v3 精确续训配置不一致: "
                    + ", ".join(mismatches)
                    + "；如需仅迁移权重请显式设置 UPDATE_RESET=1"
                )
        if legacy_weights_only:
            logger.warning(
                "[TopFormer] 将 v%d checkpoint 迁移为 v%d weights-only 初始化；"
                "旧优化器、回报统计和 rollout 均不会恢复"
                , version, TOPFORMER_SCHEMA_VERSION
            )
            old_d_model = int(saved_config.get("d_model", self.config.d_model))
            old_critic_input = ckpt["critic"].get("trunk.0.weight")
            if (
                old_d_model != self.config.d_model
                or old_critic_input is None
                or old_critic_input.ndim != 2
                or old_critic_input.shape[1] < 2 * old_d_model
            ):
                raise ValueError(
                    "legacy TopFormer critic layout cannot be safely migrated; "
                    "d_model and trunk layout must match v1/v2"
                )
            actor_loaded = self._load_shape_compatible(
                self.actor, actor_state)
            critic_input_initial = self.critic.trunk[0].weight.detach().clone()
            critic_loaded = self._load_shape_compatible(
                self.critic, ckpt["critic"])
            # v2 critic layout: [agent, candidate, global16]
            # v3 critic layout: [agent, candidate, prefix, global24].
            # Generic left-overlap would put old global weights into the new
            # prefix block.  Preserve the fresh prefix initialization and move
            # the old global block to the new global block.
            remapped = critic_input_initial
            source = old_critic_input.to(
                device=remapped.device, dtype=remapped.dtype)
            rows = min(remapped.shape[0], source.shape[0])
            shared = 2 * self.config.d_model
            remapped[:rows, :shared] = source[:rows, :shared]
            old_global = source[:rows, shared:]
            global_width = min(old_global.shape[1], self.config.global_dim)
            new_global_start = 3 * self.config.d_model
            remapped[
                :rows,
                new_global_start:new_global_start + global_width,
            ] = old_global[:, :global_width]
            with torch.no_grad():
                self.critic.trunk[0].weight.copy_(remapped)
            # v1/v2 used four launch actions plus a final "field inactive"
            # sentinel.  A rectangular overlap copy would incorrectly place
            # old WAIT_25 into v3's sentinel row (index 2).
            old_launch_emb = ckpt["actor"].get("launch_action_emb.weight")
            if old_launch_emb is not None and old_launch_emb.ndim == 2:
                with torch.no_grad():
                    source = old_launch_emb[-1].to(
                        device=self.actor.launch_action_emb.weight.device,
                        dtype=self.actor.launch_action_emb.weight.dtype,
                    )
                    if source.shape == self.actor.launch_action_emb.weight[-1].shape:
                        self.actor.launch_action_emb.weight[-1].copy_(source)
            logger.info(
                "[TopFormer] legacy weights: actor exact/partial=%s, "
                "critic exact/partial=%s", actor_loaded, critic_loaded)
        else:
            self.actor.load_state_dict(actor_state, strict=True)
            self.critic.load_state_dict(ckpt["critic"], strict=True)
        reset = reset_requested or legacy_weights_only or evaluated_only
        if reset:
            # Explicit weights-only warm start: old Adam moments must not be
            # combined with a reset scheduler/full learning rate.
            self.opt_actor = torch.optim.Adam(
                self.actor.parameters(), lr=self.config.learning_rate
            )
            self.opt_critic = torch.optim.Adam(
                self.critic.parameters(), lr=self.config.critic_lr
            )
            self.update_count = 0
            self._rounds_done = 0
            self._expert_bc_count = 0
            self._g_run_mu = 0.0; self._g_run_n = 0; self._g_run_m2 = 0.0
            self._updates_window_rounds = 0
            self._p1_window_goal_records = 0
            self._p1_window_filtered_events = 0
            self._window_discovery_links = 0
            self._window_discovery_unbound = 0
            self._window_discovery_credited_targets.clear()
            self.last_metrics = {}
            self._records.clear()
            self._deploy_records.clear()
            self._comp_raw = 0.0; self._comp_discovery = 0.0
            self._comp_retarget = 0.0; self._comp_wave = 0.0
            self._comp_progress = 0.0; self._comp_official = 0.0
            self._comp_causal_credit = 0.0
            self._comp_discovery_credit_raw = 0.0
            # Loading an evaluated snapshot without an explicit reset should
            # retain its selection threshold.  An explicit UPDATE_RESET starts
            # a new experiment and deliberately resets that threshold.
            self.best_score = (
                float(state.get("best_score", float("-inf")))
                if evaluated_only and not reset_requested
                else float("-inf")
            )
            reason = (
                "评估快照" if evaluated_only and not reset_requested
                else "UPDATE_RESET=1"
            )
            logger.info(
                "[TopFormer] %s：仅加载权重，Adam/调度/统计已重置",
                reason,
            )
            return

        self.opt_actor.load_state_dict(self._canonical_actor_optimizer_state(
            ckpt["opt_actor"],
            actor_state_migrated=actor_state_migrated,
        ))
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
        self.update_count = int(ckpt.get("update_count", 0))
        self._rounds_done = int(state.get("rounds_done", 0))
        self._expert_bc_count = int(state.get("expert_bc_count", 0))
        self._g_run_mu = float(state.get("g_run_mu", 0.0))
        self._g_run_n = int(state.get("g_run_n", 0))
        self._g_run_m2 = float(state.get("g_run_m2", 0.0))
        self._updates_window_rounds = int(state.get("updates_window_rounds", 0))
        self._p1_window_goal_records = int(
            state.get("p1_window_goal_records", 0))
        self._p1_window_filtered_events = int(
            state.get("p1_window_filtered_events", 0))
        self._window_discovery_links = int(
            state.get("window_discovery_links", 0))
        self._window_discovery_unbound = int(
            state.get("window_discovery_unbound", 0))
        self._window_discovery_credited_targets = {
            (int(episode), int(target_id))
            for episode, target_id in state.get(
                "window_discovery_credited_targets", ())
        }
        self.last_metrics = dict(state.get("last_metrics") or {})
        self.best_score = float(state.get("best_score", float("-inf")))
        self._records = list(state.get("records") or [])
        self._deploy_records = list(state.get("deploy_records") or [])
        components = state.get("components") or {}
        self._comp_raw = float(components.get("raw", 0.0))
        self._comp_discovery = float(components.get("discovery", 0.0))
        self._comp_retarget = float(components.get("retarget", 0.0))
        self._comp_wave = float(components.get("wave", 0.0))
        self._comp_progress = float(components.get("progress", 0.0))
        self._comp_official = float(components.get("official", 0.0))
        self._comp_causal_credit = float(components.get("causal_credit", 0.0))
        self._comp_discovery_credit_raw = float(
            components.get("discovery_credit_raw", 0.0))
        if state.get("torch_rng_state") is not None:
            # torch.load(map_location=cuda) also maps the CPU RNG tensor; move
            # it back because torch.set_rng_state accepts CPU ByteTensor only.
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            saved_cuda_rng = [rng.cpu() for rng in state["cuda_rng_state"]]
            device_count = torch.cuda.device_count()
            if len(saved_cuda_rng) != device_count:
                logger.warning(
                    "[TopFormer] checkpoint CUDA RNG 数量=%d，当前可见 GPU=%d；"
                    "按当前设备数截断/补齐",
                    len(saved_cuda_rng),
                    device_count,
                )
            current_cuda_rng = torch.cuda.get_rng_state_all()
            restored_cuda_rng = [
                saved_cuda_rng[index]
                if index < len(saved_cuda_rng)
                else current_cuda_rng[index]
                for index in range(device_count)
            ]
            torch.cuda.set_rng_state_all(restored_cuda_rng)
        if state.get("numpy_rng_state") is not None:
            np.random.set_state(state["numpy_rng_state"])
        if state.get("python_rng_state") is not None:
            random.setstate(state["python_rng_state"])


class _AgentStub:
    """select_decision 早于智能体注册时的兜底（正常流程不触发）。"""

    def __init__(self, agent_id, missile_type):
        self.agent_id = agent_id
        self.missile_type = missile_type
        self.latest_observation = None
        self.launch_step = -1
        self.current_goal_id = None
        self._current_sector_index = None
        self._current_goal_dist_km = None
        self._heading_sin = 0.0
        self._heading_cos = 0.0
        self._speed_estimate = 0.0
        self.sat_used = False

    @classmethod
    def from_obs(cls, agent_id, obs, missile_type):
        return cls(agent_id, missile_type)
