"""统一 MAPPO 策略：共享循环表征 + 三决策时钟 + CTDE 分头价值 + 三态行为控制。

与 TrainingEnv 的事务化生命周期（环境是转移边界的唯一权威）：
    begin_deploy_phase(full_obs)                     部署期 Critic 基准
    begin_observation_phase(full_obs) -> G_t         上报阶段开始（编码 G_t、
                                                     清步事件、应用延迟登记）
    select_decision(...)                             冻结 PendingDecision
    settle_satellite_requests() -> (commands, receipt) 集中结算
    commit_transition(...)                           引擎执行后补事实
    finish_commit_phase()                            步边界（rollout 硬上限安全阀）
    finish_episode(score) / reset_episode()          回合边界（唯一更新入口）

一步转移事务（A1）：
  动作采样时冻结 O_t / G_t / h_in / heads 进 PendingDecision（DecisionKey 溯源）；
  引擎执行后只补 reward_t / O_{t+1} / G_{t+1} / terminated。提交时不再读取
  任何可变全局态（旧隐式依赖全部删除）：
    - next_global_state 由环境显式传入（G_{t+1}，上报阶段编码）；
    - 发现奖励按提交时刻的 step_discoveries 结算（每步恰好一次，
      死亡智能体的终局转移不再重复消费）；
    - 卫星成本 _sat_charge 跨步记账、提交步恰好扣一次；
    - save() 不再隐式触发训练更新；更新只发生在回合边界
      （reset_episode 门槛判定）或步边界硬上限（内存安全阀）。

三决策时钟（一个 checkpoint、共享 GRU 表征，优势按时间尺度分离）：
  部署时钟（每局一次）：DeployHead + V_D，终局整局官方比分结算。
  宏时钟（发射/选靶事件）：launch(NOW/W10/W25/W50) + goal 头联合 log-prob、
      单次 clip；SMDP 段回报优势（见 unified_buffer）。
  微时钟（每步）：motion/resource 头联合 log-prob；逐步 GAE。

三态行为控制（T1，TRAIN_PHASE 预设驱动，见 action_spec）：
  scripted   规则执行：不采样、不记录 log-prob、不进 loss。
             b0 评测下发射 scripted=NOW 是合理策略（A/B 实验证实
             immediate/waves3/r9_timing 得分完全一致）。
  snapshot   只读执行：网络采样但不记录（冻结模块按当前权重执行）。
  trainable  采样 + 记录 + 进入 PPO 更新。
  头未激活（active=False）一律不进 loss，即使携带 scripted/snapshot 动作。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from policies.red.baselines import TargetPrior
from policies.red.learning.red_policy import GlobalStateEncoder
from .action_spec import (
    ACTION_SCHEMA_VERSION,
    LAUNCH_NOW,
    MOTION_AUTOGUIDE,
    RESOURCE_OFF,
    BehaviorMode,
    TRAIN_PHASE_PRESETS,
)
from .info_center import InfoCenter
from .unified_buffer import HeadAction, MultiHeadRollout, MultiHeadTransition
from .unified_encoder import UNIFIED_OBS_DIM, UnifiedObsEncoder
from .unified_networks import (
    KEEP_INDEX,
    TYPE_INDEX,
    CentralCritic,
    DeployHead,
    SquashedGaussian,
    UnifiedActor,
    masked_categorical,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionKey:
    """一次决策的唯一标识与溯源（回合内 (step, agent_id) 即唯一）。

    policy_version 保证 batch 不混版本；decision_seq 为回合内单调序号。
    """

    episode_id: int
    policy_version: int
    step: int
    agent_id: int
    decision_seq: int


@dataclass
class PendingDecision:
    """动作采样时刻冻结的全部决策上下文（不可变事实）。

    引擎执行后只允许补：reward_t / O_{t+1} / G_{t+1} / terminated。
    """

    key: DecisionKey
    obs_t: np.ndarray
    global_state_t: np.ndarray | None      # G_t 快照（Critic 输入基准）
    h_in: np.ndarray                        # GRU 输入隐状态（BPTT 块起点）
    heads: dict[str, HeadAction]
    micro_value: float
    macro_value: float
    sequence_start: bool
    missile_type: int                       # TYPE_INDEX: 0=H 1=M 2=L（头路由）


@dataclass
class UnifiedConfig:
    obs_dim: int = UNIFIED_OBS_DIM
    global_state_dim: int = 70
    hidden_dim: int = 128
    critic_hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    target_kl: float = 0.03
    # rollout 硬上限（内存安全阀）：步边界检查，达到即更新（人工切段，
    # buffer 对非 done 段自动 bootstrap）。回合边界门槛见 min_update_episodes。
    rollout_size: int = 200000
    min_update_size: int = 4096
    # 有效事件量门槛（T3：按事件量而非总 transition 数；回合边界判定）
    min_update_episodes: int = 32          # 完整回合数
    min_nonzero_episodes: int = 16         # 非零事件回合数
    min_goal_events: int = 1000            # goal 头 trainable 时生效
    min_motion_events: int = 16000         # motion 头 trainable 时生效
    bptt_chunk_len: int = 16      # truncated BPTT 序列块长度
    chunk_batch_size: int = 32    # 每批并行处理的序列块数
    discovery_bonus: float = 1.0  # 唯一实体首次进入合法航迹（9500 发现，发现者专属）
    goal_entropy_floor: float = 0.25  # goal 头熵下限：低于则停更该头（熵正则保留以恢复）
    satellite_cost: float = 0.2   # 每次获批卫星的资源成本（记到请求者）
    max_steps: int = 1200
    seed: int = 0
    device: str = "auto"
    # 训练阶段（T1 三态控制预设：macro 默认主线——先学目标分配与搜索）
    train_phase: str = "macro"
    # L 打陆上目标（效能 0.05，合法但低效）：训练前期屏蔽降低探索难度，
    # 最终阶段置 True 恢复为合法动作
    l_land_targets: bool = False
    # Oracle BDA（读引擎全局真值判死）：违反 CTDE 边界，仅作性能上界实验，
    # 默认关闭；与合法模式（belief-BDA）不得共用 checkpoint（观测分布不同）。
    oracle_bda: bool = False


class UnifiedMAPPO:
    """所有红弹共享的统一策略（三决策时钟 + 部署头 + 三态行为控制）。"""

    def __init__(self, config: UnifiedConfig | None = None,
                 initial_targets: tuple[TargetPrior, ...] = (),
                 total_platforms: int = 1):
        self.config = config or UnifiedConfig()
        self.initial_targets = tuple(initial_targets)
        self.total_platforms = max(int(total_platforms), 1)
        self.global_encoder = GlobalStateEncoder(max_steps=self.config.max_steps)
        if self.global_encoder.state_dim != self.config.global_state_dim:
            raise ValueError(
                f"global_state_dim={self.config.global_state_dim}，"
                f"编码器输出 {self.global_encoder.state_dim}"
            )
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        device_name = self.config.device
        if device_name == "auto":
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.actor = UnifiedActor(self.config.obs_dim, self.config.hidden_dim).to(self.device)
        self.deploy_head = DeployHead().to(self.device)
        self.critic = CentralCritic(
            self.config.global_state_dim, self.config.obs_dim, self.config.critic_hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters())
            + list(self.deploy_head.parameters())
            + list(self.critic.parameters()),
            lr=self.config.learning_rate,
        )

        self.info_center = InfoCenter(self.initial_targets, self.total_platforms)
        # 微/宏时钟缓冲（部署时钟独立见 _deploy_buffer）
        self.rollout = MultiHeadRollout()
        # 部署时钟缓冲：每局 164 条部署转移，终局整局比分结算
        self._deploy_buffer: list[dict] = []
        self._hidden: dict[int, np.ndarray] = {}
        self._pending: dict[int, PendingDecision] = {}
        self._entity_to_agent: dict[int, int] = {}
        self._deploy_encoder = UnifiedObsEncoder(max_steps=self.config.max_steps)
        # 卫星集中结算（_sat_charge 跨步存活：结算步记账、转移提交步扣费，恰好一次）
        self._sat_requests: list[int] = []
        self._sat_granted: set[int] = set()
        self._sat_charge: dict[int, float] = {}
        # 策略版本与回合计数（batch 不混版本；更新只发生在回合/步边界）
        self._policy_version: int = 0
        self._episode_id: int = 0
        self._decision_seq: int = 0
        self._rollout_episodes: int = 0
        self._rollout_nonzero_episodes: int = 0
        self._rollout_episode_nonzero: bool = False
        self._global_state: np.ndarray | None = None
        self.training = True
        self.update_count = 0
        self.deploy_update_count = 0
        self.transition_count = 0
        self.last_metrics: dict[str, float] = {}
        # 三态行为控制（T1）
        self._train_phase: str = "macro"
        self._head_modes: dict[str, BehaviorMode] = dict(TRAIN_PHASE_PRESETS["macro"])
        self.set_train_phase(self.config.train_phase)
        # 回合执行指标（record_* 由智能体在执行期上报，reset_episode 清零）
        self._ep_launch_steps: list[int] = []
        self._ep_launch_by_type: dict[int, int] = {}
        self._ep_goal_choices: dict[int, int] = {}
        self._ep_assigned_damage: dict[int, float] = {}
        self._ep_overkill: dict[int, int] = {}
        self._ep_retarget_delays: list[int] = []
        self._ep_search_actions: int = 0
        self._ep_motion_dwell: list[int] = []

    # ================= 三态行为控制（T1） =================

    def set_train_phase(self, phase: str) -> None:
        """切换训练阶段预设（scripted/snapshot/trainable 逐头配置）。

        joint 阶段入口把未训练的 NOW=-3.5 初始偏置重置为中性
        （只重置未被训练过的偏置，不覆盖已训练值）。
        """
        if phase not in TRAIN_PHASE_PRESETS:
            raise ValueError(
                f"未知 TRAIN_PHASE '{phase}'；可选: {sorted(TRAIN_PHASE_PRESETS)}"
            )
        self._train_phase = phase
        self._head_modes = dict(TRAIN_PHASE_PRESETS[phase])
        self.config.train_phase = phase
        if phase == "joint":
            self._reset_launch_bias()
        logger.info(
            "[统一策略] TRAIN_PHASE=%s | launch=%s goal=%s motion=%s resource=%s deploy=%s",
            phase, self._head_modes["launch"].value, self._head_modes["goal"].value,
            self._head_modes["motion"].value, self._head_modes["resource"].value,
            self._head_modes["deploy"].value,
        )

    def _reset_launch_bias(self) -> None:
        """joint 阶段：NOW=-3.5 初始偏置重置为中性（发射头开始训练时）。"""
        with torch.no_grad():
            if abs(float(self.actor.launch_head.bias[0].item()) - (-3.5)) < 1e-6:
                self.actor.launch_head.bias[0] = 0.0

    def _effective_goal_mode(self, missile_type: int) -> BehaviorMode:
        """目标头有效模式：search 阶段 H/M 降级 scripted（L 保持 trainable）。"""
        if self._train_phase == "search" and int(missile_type) != 21002:
            return BehaviorMode.SCRIPTED
        return self._head_modes["goal"]

    def _any_goal_trainable(self) -> bool:
        """是否存在 trainable 的目标头（决定 goal_events 门槛是否生效）。"""
        if self._head_modes["goal"] is BehaviorMode.TRAINABLE:
            return True
        return self._train_phase == "search"

    # ================= 注册与生命周期 =================

    def register_entity_agent(self, entity_id: int, agent_id: int) -> None:
        self._entity_to_agent[int(entity_id)] = int(agent_id)

    def begin_deploy_phase(self, full_observation) -> None:
        """部署阶段开始：构建部署期全局态（部署决策的 Critic 输入基准）。"""
        self._global_state = self.global_encoder.encode(full_observation)

    def begin_observation_phase(self, full_observation) -> np.ndarray:
        """上报阶段开始：编码 G_t、清步事件、应用延迟发射登记。

        由环境在每个观测态（S_t，含回合初始 S_0）的全体上报前调用；
        返回 G_t 供环境持有并在提交转移时显式传回（不留隐式全局态依赖）。
        """
        self._global_state = self.global_encoder.encode(full_observation)
        self.info_center.begin_step()
        # Oracle BDA（默认关闭，性能上界实验专用）：读全局真值判死。
        # 合法模式只用 belief-BDA（InfoCenter.report_delivery 的投送推算）。
        if self.config.oracle_bda:
            self.info_center.step_destroyed = self.info_center.oracle_sync_target_status(
                full_observation
            )
        return self._global_state

    def finish_episode(self, score: float) -> None:
        """回合终局结算：部署转移以整局比分做 episode-level return。

        由 main.py 在 run_summary.build 之后、reset_episode 之前调用；
        score 用官方比分（0-100 量级）——部署信用不经过 γ^T 折扣链。
        同时累计 rollout 的回合统计（有效事件量门槛的分子）。
        """
        if self.training and self._deploy_buffer:
            self._update_deploy(float(score))
        self._deploy_buffer.clear()
        self._rollout_episodes += 1
        if self._rollout_episode_nonzero:
            self._rollout_nonzero_episodes += 1
        self._rollout_episode_nonzero = False

    def reset_episode(self) -> None:
        """回合边界：唯一的主更新入口（按有效事件量门槛判定）。"""
        if self._update_ready():
            self.update()
        self._episode_id += 1
        self._decision_seq = 0
        self.info_center = InfoCenter(self.initial_targets, self.total_platforms)
        self._hidden.clear()
        self._pending.clear()
        self._sat_requests.clear()
        self._sat_granted.clear()
        self._sat_charge.clear()
        self._deploy_buffer.clear()
        self._global_state = None
        # 回合指标清零（上一回合的值已由 main.py 在轮末合入 summary）
        self._ep_launch_steps.clear()
        self._ep_launch_by_type.clear()
        self._ep_goal_choices.clear()
        self._ep_assigned_damage.clear()
        self._ep_overkill.clear()
        self._ep_retarget_delays.clear()
        self._ep_search_actions = 0
        self._ep_motion_dwell.clear()

    def set_training(self, training: bool) -> None:
        self.training = bool(training)
        self.actor.train(self.training)
        self.deploy_head.train(self.training)
        self.critic.train(self.training)
        if not self.training:
            self.rollout.clear()
            self._pending.clear()
            self._deploy_buffer.clear()

    # ================= 部署时钟：episode 首决策 =================

    def select_deploy(
        self,
        entity_id: int,
        entity_index: int,
        total_entities: int,
        missile_type: int,
        zone_bbox: tuple[float, float, float, float],
    ) -> tuple[float, float] | None:
        """部署头采样，写入部署缓冲（终局整局比分结算，不进 rollout）。

        deploy 头 scripted 阶段返回 None——部署由调用方（DeployAgent）
        回退到规则部署。返回部署位置 (lon, lat)。
        """
        if self._head_modes["deploy"] is BehaviorMode.SCRIPTED:
            return None
        targets = self.info_center.targets()
        centroid_lon = float(np.mean([t.position.lon for t in targets])) if targets else 0.0
        centroid_lat = float(np.mean([t.position.lat for t in targets])) if targets else 0.0
        features = UnifiedObsEncoder.build_deploy_features(
            entity_index, total_entities, missile_type, zone_bbox, centroid_lon, centroid_lat
        )
        obs_vector = self._deploy_encoder.encode_deploy(
            missile_type, entity_index, zone_bbox, self.initial_targets
        )

        obs_tensor = torch.as_tensor(obs_vector, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
        features_tensor = torch.as_tensor(features, dtype=torch.float32,
                                          device=self.device).unsqueeze(0)
        h_prev = torch.zeros(1, self.config.hidden_dim, device=self.device)
        with torch.no_grad():
            hidden = self.actor.step_forward(obs_tensor, h_prev)
            value = 0.0
            if self.training:
                if self._global_state is None:
                    raise RuntimeError("部署阶段前未调用 begin_deploy_phase")
                state_tensor = torch.as_tensor(
                    self._global_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                # 部署价值：动作发生前的状态（G_deploy, o_deploy）→ V_D
                value = float(
                    self.critic.values(state_tensor, obs_tensor)["deploy"].item()
                )
            mean = self.deploy_head(features_tensor)
            squashed = SquashedGaussian(mean, self.deploy_head.log_std)
            # 直接保存真实 pre-tanh 样本（禁止从 float32 tanh 动作反算）
            noise_t = squashed.normal.sample() if self.training else squashed.normal.mean
            action_t = torch.tanh(noise_t)
            log_prob = squashed.log_prob(noise_t, action_t)

        eps = 1e-6
        noise = noise_t[0].cpu().numpy().astype(np.float32)
        clipped = np.clip(
            action_t[0].cpu().numpy().astype(np.float32), -1 + eps, 1 - eps
        )

        agent_id = self._entity_to_agent.get(int(entity_id))
        if agent_id is not None:
            # 隐状态链：训练与评测一致，部署观测始终推进 GRU
            self._hidden[agent_id] = hidden[0].cpu().numpy()
            if self.training and self._head_modes["deploy"] is BehaviorMode.TRAINABLE:
                # 部署转移进独立缓冲：终局整局比分结算（不进 rollout/GAE）
                self._deploy_buffer.append({
                    "agent_id": agent_id,
                    "features": features,
                    "noise": noise,
                    "action": clipped,
                    "log_prob": float(log_prob.item()),
                    "value": value,
                    "obs": obs_vector,
                    "global_state": self._global_state,
                })

        min_lon, max_lon, min_lat, max_lat = zone_bbox
        lon = float(min_lon + (clipped[0] + 1.0) * 0.5 * (max_lon - min_lon))
        lat = float(min_lat + (clipped[1] + 1.0) * 0.5 * (max_lat - min_lat))
        return lon, lat

    # ================= 执行期决策（宏 + 微时钟） =================

    def _hidden_of(self, agent_id: int) -> np.ndarray:
        if agent_id not in self._hidden:
            self._hidden[agent_id] = np.zeros(self.config.hidden_dim, dtype=np.float32)
        return self._hidden[agent_id]

    def select_decision(
        self,
        agent_id: int,
        step: int,
        obs: np.ndarray,
        missile_type: int,
        *,
        launch_active: bool,
        goal_mask: np.ndarray,
        goal_event: bool = False,
        motion_active: bool = False,
        resource_active: bool = False,
        current_goal_slot: int | None = None,
    ) -> dict:
        """事件门控多头采样，冻结 PendingDecision。

        返回 {launch, goal, motion, resource}（本步无该决策则为 None）。
        scripted 头返回规则动作（不采样、active=False、不进 loss）；
        snapshot 头采样但 active=False；trainable 头采样且 active=True。
        - launch 头（宏时钟）：NOW/W10/W25/W50；WAIT 持续期间由智能体
          状态机持有、不重复采样；
        - goal 头（宏时钟）：launch=NOW 联合采样（发射即选靶）或重瞄准事件；
        - motion/resource 头（微时钟）：逐步；
        - G_t 快照进 PendingDecision（提交时不读当步全局态）。
        """
        sequence_start = agent_id not in self._hidden
        h_prev_np = self._hidden_of(agent_id)
        type_index = TYPE_INDEX.get(int(missile_type), 0)
        launch_mode = self._head_modes["launch"]
        goal_mode = self._effective_goal_mode(int(missile_type))
        motion_mode = self._head_modes["motion"]
        resource_mode = self._head_modes["resource"]

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        h_prev = torch.as_tensor(h_prev_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            hidden = self.actor.step_forward(obs_tensor, h_prev)
            micro_value = 0.0
            macro_value = 0.0
            if self.training:
                if self._global_state is None:
                    raise RuntimeError("统一策略选动作前未设置全局状态")
                state_tensor = torch.as_tensor(
                    self._global_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                values = self.critic.values(state_tensor, obs_tensor)
                micro_value = float(values["micro"].item())
                if launch_active or goal_event:
                    macro_value = float(values["macro"].item())

            heads: dict[str, HeadAction] = {}

            # ---- 发射头（宏时钟）----
            launch_head = HeadAction(active=False)
            if launch_active:
                if launch_mode is BehaviorMode.SCRIPTED:
                    # b0 下立即发射是合理策略：绕过 NOW=-3.5 偏置
                    launch_head.action = np.array([LAUNCH_NOW], dtype=np.int64)
                else:
                    launch_mask = np.ones(4, dtype=bool)
                    launch_dist = masked_categorical(
                        self.actor.launch_logits(hidden)[0],
                        torch.as_tensor(launch_mask, dtype=torch.bool, device=self.device),
                    )
                    # 评测也用采样：argmax 会因偏置导致确定性评测永不发射
                    launch_index = int(launch_dist.sample().item())
                    launch_head.action = np.array([launch_index], dtype=np.int64)
                    if launch_mode is BehaviorMode.TRAINABLE:
                        launch_head.active = True
                        launch_head.log_prob = float(launch_dist.log_prob(
                            torch.as_tensor(launch_index, device=self.device)).item())
                    launch_head.mask = launch_mask
            heads["launch"] = launch_head

            # ---- 目标头（宏时钟）：NOW 联合采样 或 重瞄准事件 ----
            launch_now = launch_active and int(np.ravel(launch_head.action)[0]) == LAUNCH_NOW
            goal_active = goal_event or launch_now
            goal_head = HeadAction(active=False)
            if goal_active:
                mask = np.asarray(goal_mask, dtype=bool).copy()
                if launch_now:
                    # 发射即选靶：KEEP 不参与（必须选出实体槽或搜索扇区）；
                    # 可选性由智能体侧 launch_allowed 门控保证，全空即协议破坏
                    mask[KEEP_INDEX] = False
                    if not mask.any():
                        raise ValueError(
                            "发射即选靶但目标掩码全空：launch_active 必须由 "
                            "launch_allowed（存在合法实体槽/扇区）门控"
                        )
                if goal_mode is BehaviorMode.SCRIPTED:
                    goal_head.action = np.array(
                        [self._scripted_goal_action(mask, current_goal_slot, launch_now)],
                        dtype=np.int64,
                    )
                else:
                    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                    goal_logits = self.actor.goal_logits(hidden[0], type_index)
                    goal_dist = masked_categorical(goal_logits, mask_tensor)
                    # 评测也用采样（与发射头一致，避免确定性同步行为）
                    goal_index = int(goal_dist.sample().item())
                    goal_head.action = np.array([goal_index], dtype=np.int64)
                    if goal_mode is BehaviorMode.TRAINABLE:
                        goal_head.active = True
                        goal_head.log_prob = float(goal_dist.log_prob(
                            torch.as_tensor(goal_index, device=self.device)).item())
                    goal_head.mask = mask
            heads["goal"] = goal_head

            # ---- 机动头（微时钟，分型路由）----
            motion_head = HeadAction(active=False)
            if motion_active:
                if motion_mode is BehaviorMode.SCRIPTED:
                    # 纯自动制导（对照臂：AUTOGUIDE / 规则脉冲 / 学习策略）
                    motion_head.action = np.array([MOTION_AUTOGUIDE], dtype=np.int64)
                else:
                    motion_mask = np.ones(3, dtype=bool)
                    motion_dist = masked_categorical(
                        self.actor.motion_logits(hidden[0], type_index),
                        torch.as_tensor(motion_mask, dtype=torch.bool, device=self.device),
                    )
                    motion_index = int(motion_dist.sample().item())
                    motion_head.action = np.array([motion_index], dtype=np.int64)
                    if motion_mode is BehaviorMode.TRAINABLE:
                        motion_head.active = True
                        motion_head.log_prob = float(motion_dist.log_prob(
                            torch.as_tensor(motion_index, device=self.device)).item())
                    motion_head.mask = motion_mask
            heads["motion"] = motion_head

            # ---- 卫星头（微时钟，仅 H 激活——由智能体侧门控保证）----
            resource_head = HeadAction(active=False)
            if resource_active:
                if resource_mode is BehaviorMode.SCRIPTED:
                    # 预算不紧张时无需训练资源头
                    resource_head.action = np.array([RESOURCE_OFF], dtype=np.int64)
                else:
                    resource_mask = np.ones(2, dtype=bool)
                    resource_dist = masked_categorical(
                        self.actor.resource_logits(hidden[0]),
                        torch.as_tensor(resource_mask, dtype=torch.bool, device=self.device),
                    )
                    resource_index = int(resource_dist.sample().item())
                    resource_head.action = np.array([resource_index], dtype=np.int64)
                    if resource_mode is BehaviorMode.TRAINABLE:
                        resource_head.active = True
                        resource_head.log_prob = float(resource_dist.log_prob(
                            torch.as_tensor(resource_index, device=self.device)).item())
                    resource_head.mask = resource_mask
            heads["resource"] = resource_head

        self._hidden[agent_id] = hidden[0].cpu().numpy()
        self._decision_seq += 1
        key = DecisionKey(
            episode_id=self._episode_id,
            policy_version=self._policy_version,
            step=int(step),
            agent_id=int(agent_id),
            decision_seq=self._decision_seq,
        )
        self._pending[int(agent_id)] = PendingDecision(
            key=key,
            obs_t=np.array(obs, dtype=np.float32, copy=True),
            global_state_t=None if self._global_state is None
            else np.array(self._global_state, copy=True),  # G_t 快照
            h_in=h_prev_np,
            heads=heads,
            micro_value=micro_value,
            macro_value=macro_value,
            sequence_start=sequence_start,
            missile_type=type_index,
        )
        return {
            name: (int(np.ravel(heads[name].action)[0])
                   if heads[name].action is not None else None)
            for name in ("launch", "goal", "motion", "resource")
        }

    @staticmethod
    def _scripted_goal_action(
        goal_mask: np.ndarray, current_goal_slot: int | None, at_launch: bool
    ) -> int:
        """scripted 目标规则：有效目标保持（KEEP），否则首个合法实体槽/扇区。"""
        if not at_launch and current_goal_slot is not None and goal_mask[current_goal_slot]:
            return KEEP_INDEX
        candidates = np.flatnonzero(goal_mask[:KEEP_INDEX])
        if candidates.size > 0:
            return int(candidates[0])
        return KEEP_INDEX

    # ================= 卫星集中结算（全体采样后统一结算） =================

    def register_satellite_request(self, entity_id: int) -> None:
        """能力头输出"资源请求"：先登记，等本步所有智能体采样完再统一结算。"""
        self._sat_requests.append(int(entity_id))

    def settle_satellite_requests(self) -> tuple[list[list[float]], dict]:
        """集中结算：预算内按请求顺序（确定性）批准并生成引擎指令，超额拒绝。

        由 training_env 在收集完全部智能体动作后、指令转换前调用；
        返回 ([[3, entity_id, 0, 0], ...], ResourceReceipt 字段 dict)。
        成本记到 _sat_charge[entity_id]（跨步存活），该弹转移提交步恰好扣一次。
        """
        commands: list[list[float]] = []
        granted_ids: list[int] = []
        for entity_id in self._sat_requests:
            if self.info_center.sat_budget_remaining <= 0:
                break
            self.info_center.mark_satellite_used()
            self._sat_granted.add(entity_id)
            granted_ids.append(int(entity_id))
            self._sat_charge[entity_id] = (
                self._sat_charge.get(entity_id, 0.0) + self.config.satellite_cost
            )
            commands.append([3, entity_id, 0, 0])
        self._sat_requests.clear()
        receipt = {
            "granted_entity_ids": tuple(granted_ids),
            "budget_remaining": int(self.info_center.sat_budget_remaining),
        }
        return commands, receipt

    @property
    def satellite_granted(self) -> set[int]:
        """本回合已获批卫星的实体集合（智能体在上报阶段领取并更新自身状态）。"""
        return self._sat_granted

    # ================= 回合执行指标（多种子训练记录） =================

    def record_launch(self, missile_type: int, step: int) -> None:
        """发射事件：发射时间分布 + 分弹种发射数。"""
        self._ep_launch_steps.append(int(step))
        key = int(missile_type)
        self._ep_launch_by_type[key] = self._ep_launch_by_type.get(key, 0) + 1

    def record_goal_choice(self, target_id: int, missile_type: int) -> None:
        """目标分配事件：直方图 + 重复集火判定。

        重复集火 = 分配时该目标累计分配期望毁伤已 ≥ 战前血量仍继续分配
        （按弹种×目标类型效能推算，与 belief-BDA 同一合法先验）。
        """
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
        damage = self.info_center.EXPECTED_DAMAGE.get(
            int(missile_type), {}
        ).get(target_type, 0.0)
        self._ep_assigned_damage[tid] = assigned + damage

    def record_retarget(self, step: int, launch_step: int) -> None:
        """重瞄准事件：换靶时刻相对发射时刻的延迟（步）。"""
        self._ep_retarget_delays.append(max(0, int(step) - int(launch_step)))

    def record_search_action(self) -> None:
        """L 选择搜索扇区事件（课程指标：搜索行为频率）。"""
        self._ep_search_actions += 1

    def record_motion_dwell(self, dwell: int) -> None:
        """机动脉冲驻留统计（监控持续单侧转弯坍缩）。"""
        self._ep_motion_dwell.append(int(dwell))

    def episode_metrics(self, ships_total: int | None = None) -> dict:
        """回合执行指标汇总。main.py 在轮末（reset_episode 前）读取合入 summary。"""
        launches = sorted(self._ep_launch_steps)
        launch_histogram: dict[str, int] = {}
        for step in launches:
            bucket = (step // 50) * 50
            launch_histogram[str(bucket)] = launch_histogram.get(str(bucket), 0) + 1
        delays = sorted(self._ep_retarget_delays)
        dwell = sorted(self._ep_motion_dwell)
        if ships_total is None:
            ships_total = sum(
                1 for target in self.initial_targets if target.entity_type == 9500
            )
        ships_total = max(int(ships_total), 0)
        ships_found = len(self.info_center.known_ships)
        return {
            "launch": {
                "count": len(launches),
                "by_missile_type": {
                    str(key): count
                    for key, count in sorted(self._ep_launch_by_type.items())
                },
                "time_histogram_50step": launch_histogram,
                "mean_step": float(np.mean(launches)) if launches else 0.0,
                "p50_step": float(np.percentile(launches, 50)) if launches else 0.0,
                "min_step": float(launches[0]) if launches else -1.0,
                "max_step": float(launches[-1]) if launches else -1.0,
            },
            "target_assignment": {
                str(key): count
                for key, count in sorted(self._ep_goal_choices.items())
            },
            "repeated_focus": {
                "total": sum(self._ep_overkill.values()),
                "by_target": {
                    str(key): count
                    for key, count in sorted(self._ep_overkill.items())
                },
            },
            "discovery": {
                "ships_total": ships_total,
                "ships_found": ships_found,
                "rate": (ships_found / ships_total) if ships_total > 0 else 1.0,
            },
            "retarget": {
                "count": len(delays),
                "mean_delay": float(np.mean(delays)) if delays else 0.0,
                "max_delay": float(delays[-1]) if delays else 0.0,
            },
            "search_actions": int(self._ep_search_actions),
            "motion_dwell": {
                "count": len(dwell),
                "mean": float(np.mean(dwell)) if dwell else 0.0,
                "max": float(dwell[-1]) if dwell else 0.0,
            },
            "satellite_usage": int(self.info_center.sat_used_count),
            "train_phase": self._train_phase,
        }

    # ================= 经验回传（一步转移事务的提交半边） =================

    def commit_transition(
        self,
        *,
        agent_id: int,
        entity_id: int,
        next_obs: np.ndarray | None,
        next_global_state: np.ndarray | None,
        env_reward: float,
        terminated: bool,
    ) -> bool:
        """提交一步转移：引擎执行后只补事实（reward/O_{t+1}/G_{t+1}/terminated）。

        由环境在全体上报冻结 snapshot_{t+1} 之后调用（每智能体每步恰好一次）：
        - 发现奖励按当步 step_discoveries 结算（全体同值、恰好一次，
          死亡智能体的终局转移与存活智能体同等对待，不再重复消费）；
        - 卫星成本从 _sat_charge 弹出（跨步记账、提交步扣费恰好一次）；
        - next_obs=None 表示终态（MISSILE_DEATH）：取零向量，done 位屏蔽
          其 bootstrap。
        """
        pending = self._pending.pop(int(agent_id), None)
        if pending is None:
            return False
        if not self.training:
            return True  # 评测：决策上下文直接丢弃
        if next_obs is None:
            next_obs = np.zeros_like(pending.obs_t)
        reward = float(env_reward)
        if int(entity_id) in self.info_center.step_discoverers:
            # 发现者专属：只有实际探测到新船的弹获得发现奖励
            #（全体同值版本没有个体梯度，发现行为学不出来）
            reward += self.config.discovery_bonus
        reward -= self._sat_charge.pop(int(entity_id), 0.0)
        if reward != 0.0:
            self._rollout_episode_nonzero = True

        heads = pending.heads
        macro_old = sum(
            heads[name].log_prob for name in ("launch", "goal")
            if heads[name].active
        )
        micro_old = sum(
            heads[name].log_prob for name in ("motion", "resource")
            if heads[name].active
        )
        self.rollout.append(
            MultiHeadTransition(
                agent_id=int(agent_id),
                step=int(pending.key.step),
                obs=pending.obs_t,
                h_prev=pending.h_in,
                missile_type=int(pending.missile_type),
                launch=heads["launch"],
                goal=heads["goal"],
                motion=heads["motion"],
                resource=heads["resource"],
                value=pending.micro_value,
                macro_value=pending.macro_value,
                reward=reward,
                done=bool(terminated),
                sequence_start=bool(pending.sequence_start),
                # G_t 决策时刻快照；G_{t+1} 由环境显式传入（终局置 None）
                global_state=pending.global_state_t,
                next_global_state=None if terminated else next_global_state,
                next_obs=next_obs,
                joint_old_log_prob=float(macro_old + micro_old),
                macro_old_log_prob=float(macro_old),
                micro_old_log_prob=float(micro_old),
                policy_version=pending.key.policy_version,
            )
        )
        self.transition_count += 1
        return True

    def finish_commit_phase(self) -> None:
        """步边界：rollout 硬上限安全阀（内存保护；步内绝不更新）。

        达到上限时在本步全体转移提交完毕后更新——非 done 段由
        compute_advantages 以存储的 next_obs/next_global 自动 bootstrap。
        """
        if self.training and len(self.rollout) >= self.config.rollout_size:
            self.update()

    # ================= 更新与持久化 =================

    def _update_ready(self) -> bool:
        """回合边界更新门槛：有效事件量（而非总 transition 数）。"""
        if not self.training or len(self.rollout) < self.config.min_update_size:
            return False
        if self._rollout_episodes < self.config.min_update_episodes:
            return False
        if self._rollout_nonzero_episodes < self.config.min_nonzero_episodes:
            return False
        counts = self.rollout.event_counts()
        if self._any_goal_trainable() and counts["goal_events"] < self.config.min_goal_events:
            return False
        if (self._head_modes["motion"] is BehaviorMode.TRAINABLE
                and counts["motion_events"] < self.config.min_motion_events):
            return False
        return True

    def update(self) -> dict[str, float]:
        if len(self.rollout) < self.config.min_update_size:
            return self.last_metrics
        # goal 头熵闸门：上一轮实测熵低于下限 → 本轮停更该头
        #（跳过其优势策略梯度，熵正则保留——熵回升后自动解除）
        counts = self.rollout.event_counts()
        gate_goal = (
            self.update_count > 0
            and counts.get("goal_events", 0) > 0
            and float(self.last_metrics.get("entropy_goal", 1.0))
            < self.config.goal_entropy_floor
        )
        if gate_goal:
            logger.info(
                "[统一策略] goal 头熵 %.3f 低于下限 %.2f：本轮停更 goal 头（熵正则保留）",
                float(self.last_metrics.get("entropy_goal", 0.0)),
                self.config.goal_entropy_floor,
            )
        metrics = self.rollout.ppo_losses(
            self.actor, self.critic,
            self.optimizer, self.device, self.config,
            gate_goal=gate_goal,
        )
        self.update_count += 1
        metrics["samples"] = float(len(self.rollout))
        metrics["episodes"] = float(self._rollout_episodes)
        metrics["nonzero_episodes"] = float(self._rollout_nonzero_episodes)
        metrics["goal_gated"] = 1.0 if gate_goal else 0.0
        self.last_metrics = metrics
        self.rollout.clear()
        # 新策略版本：后续采样以此版本标记（batch 不混版本）
        self._policy_version += 1
        self._rollout_episodes = 0
        self._rollout_nonzero_episodes = 0
        self._rollout_episode_nonzero = False
        logger.info(
            "[统一策略] 更新 #%d | 样本=%d 回合=%d(非零%d) | policy=%.4f value=%.4f "
            "kl=%.4f | 熵(goal=%.3f launch=%.3f motion=%.3f resource=%.3f)",
            self.update_count, int(metrics.get("samples", 0)),
            int(metrics.get("episodes", 0)), int(metrics.get("nonzero_episodes", 0)),
            metrics.get("policy_loss", 0.0), metrics.get("value_loss", 0.0),
            metrics.get("approx_kl", 0.0),
            metrics.get("entropy_goal", 0.0), metrics.get("entropy_launch", 0.0),
            metrics.get("entropy_motion", 0.0), metrics.get("entropy_resource", 0.0),
        )
        return metrics

    def _update_deploy(self, episode_return: float) -> None:
        """部署时钟 PPO：整局比分作为所有部署转移的 episode-level return。

        优势 = return − V_D(o_0^deploy)（不经过 γ^T 折扣链），
        tanh 高斯 ratio 用存储的 noise 精确重算 log-prob。
        """
        records = self._deploy_buffer
        n = len(records)
        if n == 0:
            return
        features = torch.as_tensor(
            np.stack([r["features"] for r in records]),
            dtype=torch.float32, device=self.device,
        )
        noise = torch.as_tensor(
            np.stack([r["noise"] for r in records]),
            dtype=torch.float32, device=self.device,
        )
        old_lp = torch.as_tensor(
            np.array([r["log_prob"] for r in records], dtype=np.float32),
            device=self.device,
        )
        glob = torch.as_tensor(
            np.stack([
                r["global_state"] if r["global_state"] is not None
                else np.zeros(self.config.global_state_dim, dtype=np.float32)
                for r in records
            ]),
            dtype=torch.float32, device=self.device,
        )
        obs = torch.as_tensor(
            np.stack([r["obs"] for r in records]),
            dtype=torch.float32, device=self.device,
        )
        v_old_np = np.array([r["value"] for r in records], dtype=np.float32)
        v_old = torch.as_tensor(v_old_np, device=self.device)
        ret = torch.full((n,), float(episode_return), device=self.device)

        adv_np = ret.cpu().numpy() - v_old_np
        if adv_np.size > 1 and adv_np.std() > 1e-6:
            adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
        adv = torch.as_tensor(adv_np, device=self.device)

        target_kl = self.config.target_kl
        metrics = {"deploy_policy_loss": 0.0, "deploy_value_loss": 0.0,
                   "deploy_kl": 0.0, "deploy_return": float(episode_return)}
        updates = 0
        stop = False
        for _ in range(self.config.update_epochs):
            if stop:
                break
            mean = self.deploy_head(features)
            dist = SquashedGaussian(mean, self.deploy_head.log_std)
            action = torch.tanh(noise)
            new_lp = dist.log_prob(noise, action)
            ratio = torch.exp(new_lp - old_lp)
            policy_loss = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 1.0 - self.config.clip_ratio,
                            1.0 + self.config.clip_ratio) * adv,
            ).mean()
            v_pred = self.critic.values(glob, obs)["deploy"]
            v_clipped = v_old + torch.clamp(
                v_pred - v_old,
                -self.config.value_clip_ratio, self.config.value_clip_ratio,
            )
            value_loss = 0.5 * torch.maximum(
                (v_pred - ret).pow(2), (v_clipped - ret).pow(2)
            ).mean()
            entropy = dist.normal.entropy().sum(dim=-1).mean()
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.deploy_head.parameters()) + list(self.critic.parameters()),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            with torch.no_grad():
                approx_kl = float(((ratio - 1.0) - (new_lp - old_lp)).mean().item())
            metrics["deploy_policy_loss"] += float(policy_loss.item())
            metrics["deploy_value_loss"] += float(value_loss.item())
            metrics["deploy_kl"] += approx_kl
            updates += 1
            if approx_kl > 1.5 * target_kl:
                break
        self.deploy_update_count += 1
        for key in ("deploy_policy_loss", "deploy_value_loss", "deploy_kl"):
            metrics[key] /= max(updates, 1)
        logger.info(
            "[统一策略] 部署更新 #%d | 样本=%d | return=%.2f | policy=%.4f "
            "value=%.4f kl=%.4f",
            self.deploy_update_count, n, episode_return,
            metrics["deploy_policy_loss"], metrics["deploy_value_loss"],
            metrics["deploy_kl"],
        )
        return metrics

    def save(self, path: str) -> None:
        """持久化（不触发训练更新——更新只发生在回合/步边界）。"""
        payload = {
            "algorithm": "unified_mappo",
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "train_phase": self._train_phase,
            "policy_version": self._policy_version,
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "deploy_head": self.deploy_head.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "deploy_update_count": self.deploy_update_count,
            "transition_count": self.transition_count,
            "last_metrics": self.last_metrics,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm") != "unified_mappo":
            raise ValueError("该checkpoint不是统一MAPPO模型")
        version = int(checkpoint.get("action_schema_version", 1))
        if version != ACTION_SCHEMA_VERSION:
            raise ValueError(
                f"动作协议版本不匹配：checkpoint={version} 当前={ACTION_SCHEMA_VERSION}，"
                "拒绝加载（旧观测/动作协议的 checkpoint 不得混入）"
            )
        self.actor.load_state_dict(checkpoint["actor"])
        self.deploy_head.load_state_dict(checkpoint["deploy_head"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_count = int(checkpoint.get("update_count", 0))
        self.deploy_update_count = int(checkpoint.get("deploy_update_count", 0))
        self.transition_count = int(checkpoint.get("transition_count", 0))
        self.last_metrics = dict(checkpoint.get("last_metrics", {}))
        self._policy_version = int(checkpoint.get("policy_version", 0))
        self.set_train_phase(checkpoint.get("train_phase", self.config.train_phase))
