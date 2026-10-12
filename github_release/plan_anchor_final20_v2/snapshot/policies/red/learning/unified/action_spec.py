"""统一策略动作协议：飞行阶段、枚举动作索引、三态行为模式。

ACTION_SCHEMA_VERSION 随动作/观测/动力学契约变化递增，checkpoint 携带
该版本；加载时版本不匹配直接拒绝（防 61 维旧观测混入新协议）。

三态行为模式（每个决策头独立配置，由 TRAIN_PHASE 预设驱动）：
  scripted   规则执行：不采样、不记录 log-prob、不进 loss。
             b0 评测下发射 scripted=NOW 是合理策略（A/B 实验证实
             三种时序得分完全一致），不属于训练失败。
  snapshot   只读执行：网络采样但不记录（冻结模块按当前权重执行）。
  trainable  采样 + 记录 + 进入 PPO 更新。

每个 loss 的有效样本掩码：
  effective = active_mask * (behavior_source == "policy")
（active=False 的头即使有动作也不进 loss；本实现中 active 即
"从策略采样并记录"，scripted/snapshot 头一律 active=False。）

每个头必须存在合法 no-op（KEEP/AUTOGUIDE/OFF/WAIT），masked_categorical
在全 False 时直接报错（不再静默放开全部动作）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

ACTION_SCHEMA_VERSION = 2


class FlightPhase(int, Enum):
    """飞行阶段状态机（B1 契约）。"""

    PRELAUNCH = 0          # 未发射：launch 头激活
    FLYING_AUTOGUIDE = 1   # 在飞·自动制导：motion 头可激活
    EVASION_PULSE = 2      # 在飞·规避脉冲：脉冲期间 motion 头关闭
    TERMINAL = 3           # 末段（stage>=4）：不再机动


class BehaviorMode(str, Enum):
    """决策头三态控制。"""

    SCRIPTED = "scripted"
    SNAPSHOT = "snapshot"
    TRAINABLE = "trainable"


# ---- 发射头动作索引（LAUNCH_DIM=4）----
LAUNCH_NOW = 0             # 立即发射（目标头联合采样）
LAUNCH_WAIT_10 = 1         # 等待 10 步后重新激活
LAUNCH_WAIT_25 = 2
LAUNCH_WAIT_50 = 3
# WAIT 持续期间不重复采样（状态机 wait_until_step 持有）
LAUNCH_WAIT_STEPS = {LAUNCH_WAIT_10: 10, LAUNCH_WAIT_25: 25, LAUNCH_WAIT_50: 50}

# ---- 机动头动作索引（MOTION_DIM=3，仓库约定）----
MOTION_LEFT_PULSE = 0      # 左转脉冲（-1，持续若干步后自动回中）
MOTION_AUTOGUIDE = 1       # 保持自动制导（清零横向过载）
MOTION_RIGHT_PULSE = 2     # 右转脉冲（+1）

# ---- 卫星头动作索引（RESOURCE_DIM=2）----
RESOURCE_OFF = 0
RESOURCE_ON = 1

# ---- 训练阶段 → 各头行为模式预设 ----
# macro（默认主线）：先学目标分配与搜索。发射 scripted=NOW（b0 下立即
#   发射是合理策略，且避免 NOW=-3.5 偏置饿死 goal/motion 训练数据）；
#   机动 scripted=AUTOGUIDE（AUTOGUIDE / 规则脉冲 / 学习策略的对照臂）；
#   卫星 scripted=OFF（预算不紧张时无需训练资源头）。
# motion：固定宏决策，单独训练机动与卫星。
# search：L 的目标头 trainable（搜索扇区/选船/重瞄准），H/M 目标 scripted。
# deploy：固定下游，训练部署（编队参数化后再启用）。
# joint：全部解冻（发射头此时 NOW=-3.5 应重置为中性或教师初始化）。
TRAIN_PHASE_PRESETS: dict[str, dict[str, BehaviorMode]] = {
    "macro": {
        "launch": BehaviorMode.SCRIPTED,
        "goal": BehaviorMode.TRAINABLE,
        "motion": BehaviorMode.SCRIPTED,
        "resource": BehaviorMode.SCRIPTED,
        "deploy": BehaviorMode.SCRIPTED,
    },
    "motion": {
        "launch": BehaviorMode.SCRIPTED,
        "goal": BehaviorMode.SCRIPTED,
        "motion": BehaviorMode.TRAINABLE,
        "resource": BehaviorMode.TRAINABLE,
        "deploy": BehaviorMode.SCRIPTED,
    },
    "search": {
        "launch": BehaviorMode.SCRIPTED,
        "goal": BehaviorMode.TRAINABLE,   # L 聚焦搜索；H/M 由 scripted 覆盖
        "motion": BehaviorMode.SCRIPTED,
        "resource": BehaviorMode.SCRIPTED,
        "deploy": BehaviorMode.SCRIPTED,
    },
    "deploy": {
        "launch": BehaviorMode.SCRIPTED,
        "goal": BehaviorMode.SNAPSHOT,
        "motion": BehaviorMode.SNAPSHOT,
        "resource": BehaviorMode.SNAPSHOT,
        "deploy": BehaviorMode.TRAINABLE,
    },
    "joint": {
        "launch": BehaviorMode.TRAINABLE,
        "goal": BehaviorMode.TRAINABLE,
        "motion": BehaviorMode.TRAINABLE,
        "resource": BehaviorMode.TRAINABLE,
        "deploy": BehaviorMode.TRAINABLE,
    },
}

# search 阶段 H/M 的目标头降级为 scripted（L 保持 trainable）
SEARCH_HM_GOAL_MODE = BehaviorMode.SCRIPTED


@dataclass
class DecisionToken:
    """一次决策的唯一标识与溯源（回合内 (step, agent_id) 即唯一）。"""

    step: int
    agent_id: int
    episode_id: int = -1
    policy_version: int = -1


@dataclass
class ScriptedPolicy:
    """scripted 模式的规则参数（当前固定，后续可扩展为规则集）。"""

    launch_action: int = LAUNCH_NOW          # 立即发射
    motion_action: int = MOTION_AUTOGUIDE    # 纯自动制导
    resource_action: int = RESOURCE_OFF      # 不请求卫星
