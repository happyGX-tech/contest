"""多头轨迹缓存：三时钟优势分离 + 分块 truncated BPTT + 分组 ratio PPO。

三决策时钟，优势不共用（不同时间尺度的动作不能用同一个 transition advantage）：
  宏时钟（launch/goal 联合事件）：SMDP 段回报
      R_k = Σ_{j=0..Δk-1} γ^j r_{tk+j} + γ^{Δk} V_M(s_{k+1})
      adv_k = R_k − V_M(s_k)
      段内 reward 从宏决策当步起累计到下一宏决策前一步；bootstrap 取下一宏
      决策转移上已存的 V_M(s_{k+1})（恰为该决策时刻的价值），段尾（done）
      不 bootstrap。
  微时钟（motion/resource 每步）：按 agent 序列标准 GAE（V_C bootstrap）
  部署时钟：不在本缓冲——见 UnifiedMAPPO._deploy_buffer（终局比分结算）

每头按自己的有效决策数量归一化；宏/微联合 log-prob 各自成组、单次 clip。
approx_kl 按有效样本数归一化，超过 1.5×target_kl 提前终止 epoch。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .unified_networks import (
    LAUNCH_DIM,
    MOTION_DIM,
    RESOURCE_DIM,
    TARGET_DIM,
    SquashedGaussian,
    masked_categorical,
)

_HEAD_DIMS = {
    "launch": LAUNCH_DIM,
    "goal": TARGET_DIM,
    "motion": MOTION_DIM,
    "resource": RESOURCE_DIM,
}
_ALL_HEADS = ("launch", "goal", "motion", "resource")


@dataclass
class HeadAction:
    active: bool = False
    action: np.ndarray | None = None     # 离散头: 标量
    log_prob: float = 0.0
    mask: np.ndarray | None = None       # 离散头合法掩码


@dataclass
class MultiHeadTransition:
    agent_id: int
    step: int
    obs: np.ndarray
    h_prev: np.ndarray
    missile_type: int = 0                # TYPE_INDEX: 0=H 1=M 2=L（头路由）
    launch: HeadAction = field(default_factory=HeadAction)
    goal: HeadAction = field(default_factory=HeadAction)
    motion: HeadAction = field(default_factory=HeadAction)
    resource: HeadAction = field(default_factory=HeadAction)
    value: float = 0.0                   # V_C（微价值）
    macro_value: float = 0.0             # V_M（宏决策时刻；非宏决策步为 0）
    reward: float = 0.0
    done: bool = False
    sequence_start: bool = False
    global_state: np.ndarray | None = None
    next_global_state: np.ndarray | None = None
    next_obs: np.ndarray | None = None
    joint_old_log_prob: float = 0.0      # Σ 全部激活头旧 log-prob（日志用）
    macro_old_log_prob: float = 0.0      # launch+goal 联合旧 log-prob
    micro_old_log_prob: float = 0.0      # motion+resource 联合旧 log-prob
    policy_version: int = 0              # 采样时的策略版本（batch 不得混版本）

    def macro_active(self) -> bool:
        return self.launch.active or self.goal.active

    def micro_active(self) -> bool:
        return self.motion.active or self.resource.active


class MultiHeadRollout:
    """纯存储 + 三时钟优势；PPO 更新由 UnifiedMAPPO.update() 驱动。"""

    def __init__(self):
        self.transitions: list[MultiHeadTransition] = []

    def __len__(self) -> int:
        return len(self.transitions)

    def clear(self) -> None:
        self.transitions.clear()

    def append(self, transition: MultiHeadTransition) -> None:
        """追加转移；同一 batch 内 policy_version 必须一致（不混版本更新）。"""
        if self.transitions:
            existing = self.transitions[0].policy_version
            if transition.policy_version != existing:
                raise ValueError(
                    f"rollout 内混入不同 policy_version "
                    f"({existing} vs {transition.policy_version})；"
                    "更新只允许发生在回合边界，开放式宏动作不得跨越策略更新"
                )
        self.transitions.append(transition)

    def event_counts(self) -> dict[str, int]:
        """有效事件量统计（训练门槛按事件量而非总 transition 数）。"""
        episodes = 0
        nonzero_episodes = 0
        current_nonzero = False
        for transition in self.transitions:
            if transition.sequence_start:
                if current_nonzero:
                    nonzero_episodes += 1
                current_nonzero = False
                episodes += 1
            if transition.reward != 0.0:
                current_nonzero = True
            if transition.done:
                if current_nonzero:
                    nonzero_episodes += 1
                current_nonzero = False
                episodes += 1
        return {
            "episodes": episodes,
            "nonzero_episodes": nonzero_episodes,
            "goal_events": sum(1 for t in self.transitions if t.goal.active),
            "launch_events": sum(1 for t in self.transitions if t.launch.active),
            "motion_events": sum(1 for t in self.transitions if t.motion.active),
            "resource_events": sum(1 for t in self.transitions if t.resource.active),
        }

    # ---------- 序列组织 ----------

    def episodes(self) -> list[list[MultiHeadTransition]]:
        """按智能体把缓冲切成时间连续的回合序列（跨 done/sequence_start 边界分割）。"""
        by_agent: dict[int, list[MultiHeadTransition]] = {}
        for transition in self.transitions:
            by_agent.setdefault(transition.agent_id, []).append(transition)
        episodes: list[list[MultiHeadTransition]] = []
        for sequence in by_agent.values():
            current: list[MultiHeadTransition] = []
            for transition in sequence:
                if transition.sequence_start and current:
                    episodes.append(current)
                    current = []
                current.append(transition)
                if transition.done:
                    episodes.append(current)
                    current = []
            if current:
                episodes.append(current)
        return episodes

    # ---------- 优势计算（微: GAE / 宏: SMDP 段回报） ----------

    def compute_advantages(self, critic, device, gamma: float, lam: float):
        """返回 (micro_adv, micro_ret, macro_adv, macro_ret)，与 transitions 等长。"""
        index_of = {id(t): i for i, t in enumerate(self.transitions)}
        rewards = np.array([t.reward for t in self.transitions], dtype=np.float32)
        dones = np.array([float(t.done) for t in self.transitions], dtype=np.float32)
        values = np.array([t.value for t in self.transitions], dtype=np.float32)
        macro_values = np.array(
            [t.macro_value for t in self.transitions], dtype=np.float32
        )

        # 统一预计算 s_{t+1} 的微/宏 bootstrap 价值
        # （next_obs 为 None 的终局转移用零观测；done 位会将其屏蔽）
        with torch.no_grad():
            glob_stack = torch.as_tensor(
                np.stack([
                    t.next_global_state
                    if (t.next_global_state is not None and not t.done)
                    else (t.global_state if t.global_state is not None
                          else np.zeros(1, dtype=np.float32))
                    for t in self.transitions
                ]),
                dtype=torch.float32, device=device,
            )
            obs_stack = torch.as_tensor(
                np.stack([
                    t.next_obs if (t.next_obs is not None and not t.done)
                    else np.zeros_like(t.obs)
                    for t in self.transitions
                ]),
                dtype=torch.float32, device=device,
            )
            next_values = critic.values(glob_stack, obs_stack)
            micro_next = next_values["micro"].cpu().numpy()
            macro_next = next_values["macro"].cpu().numpy()

        micro_adv = np.zeros_like(rewards)
        micro_ret = np.zeros_like(rewards)
        macro_adv = np.zeros_like(rewards)
        macro_ret = np.zeros_like(rewards)

        for episode in self.episodes():
            # ---- 微时钟：标准 GAE（按 agent 序列倒序回传） ----
            gae = 0.0
            for transition in reversed(episode):
                i = index_of[id(transition)]
                nonterminal = 1.0 - dones[i]
                delta = rewards[i] + gamma * micro_next[i] * nonterminal - values[i]
                gae = delta + gamma * lam * nonterminal * gae
                micro_adv[i] = gae
            for transition in episode:
                i = index_of[id(transition)]
                micro_ret[i] = micro_adv[i] + values[i]

            # ---- 宏时钟：SMDP 段回报（段 = 本宏决策步 → 下一宏决策前一步） ----
            macro_positions = [
                p for p, transition in enumerate(episode)
                if transition.macro_active()
            ]
            for k, p in enumerate(macro_positions):
                i = index_of[id(episode[p])]
                if k + 1 < len(macro_positions):
                    next_p = macro_positions[k + 1]
                    segment = episode[p:next_p]
                    # bootstrap = 下一宏决策时刻的价值（该转移已存 V_M）
                    boot = float(macro_values[index_of[id(episode[next_p])]])
                else:
                    segment = episode[p:]
                    last = index_of[id(episode[-1])]
                    # 段尾：done 不 bootstrap，否则用终态之后一步的 V_M
                    boot = 0.0 if dones[last] else float(macro_next[last])
                acc = 0.0
                for transition in reversed(segment):
                    acc = rewards[index_of[id(transition)]] + gamma * acc
                macro_ret[i] = acc + (gamma ** len(segment)) * boot
                macro_adv[i] = macro_ret[i] - macro_values[i]

        return micro_adv, micro_ret, macro_adv, macro_ret

    # ---------- 分块 truncated BPTT + 分组 ratio PPO ----------

    def ppo_losses(self, actor, critic, optimizer, device, config,
                   gate_goal: bool = False):
        """分头 PPO 更新。

        launch 与 goal 头分布可分解（两头各自独立 categorical），联合概率
        等于两者之积，故按头各自计算 ratio 并独立 clip（替代旧的联合
        单次 clip）。gate_goal=True 时 goal 头跳过优势策略梯度（熵闸门），
        但熵正则与 KL 观测保留——熵回升后闸门自动解除，不会永久冻死。
        """
        n = len(self.transitions)
        if n == 0:
            return {}

        micro_adv, micro_ret, macro_adv, macro_ret = self.compute_advantages(
            critic, device, config.gamma, config.gae_lambda
        )
        # 分头优势归一化：宏/微各自标准化；宏仅在有效样本上统计
        # （信号门限：宏优势方差过小——无事件 rollout——不归一化放大）
        macro_active_mask = np.array(
            [t.macro_active() for t in self.transitions], dtype=np.float32
        )
        if macro_adv.size > 1 and macro_active_mask.sum() > 1:
            sel = macro_active_mask > 0
            std = macro_adv[sel].std()
            if std > 1e-6:
                macro_adv = np.where(
                    sel, (macro_adv - macro_adv[sel].mean()) / (std + 1e-8), 0.0
                )
        if micro_adv.size > 1:
            micro_adv = (micro_adv - micro_adv.mean()) / (micro_adv.std() + 1e-8)

        chunk_len = config.bptt_chunk_len
        chunks: list[list[MultiHeadTransition]] = []
        for episode in self.episodes():
            for start in range(0, len(episode), chunk_len):
                chunks.append(episode[start:start + chunk_len])

        obs_dim = actor.embed[0].in_features
        hidden_dim = actor.hidden_dim
        global_dim = critic.trunk[0].in_features - obs_dim
        index_of = {id(t): i for i, t in enumerate(self.transitions)}

        def chunk_arrays(chunk):
            length = len(chunk)
            obs = np.zeros((length, obs_dim), dtype=np.float32)
            glob = np.zeros((length, global_dim), dtype=np.float32)
            head_active = {name: np.zeros(length) for name in _ALL_HEADS}
            head_action = {name: np.zeros(length, dtype=np.int64) for name in _ALL_HEADS}
            head_mask = {
                name: np.zeros((length, _HEAD_DIMS[name]), dtype=bool)
                for name in _ALL_HEADS
            }
            for i, t in enumerate(chunk):
                obs[i] = t.obs
                if t.global_state is not None:
                    glob[i] = t.global_state
                for name in _ALL_HEADS:
                    head = getattr(t, name)
                    if head.active and head.action is not None:
                        head_active[name][i] = 1.0
                        head_action[name][i] = int(np.ravel(head.action)[0])
                    head_mask[name][i] = (
                        head.mask.astype(bool)
                        if head.mask is not None
                        else np.ones(_HEAD_DIMS[name], dtype=bool)
                    )
            return obs, glob, head_active, head_action, head_mask, chunk[0].h_prev

        def batch_macros(batch):
            """从原始 chunk 对象读各头旧 log-prob（精确值）。"""
            b = len(batch)
            max_len = max(len(c) for c in batch)
            launch_lp = np.zeros((b, max_len), dtype=np.float32)
            goal_lp = np.zeros((b, max_len), dtype=np.float32)
            micro_lp = np.zeros((b, max_len), dtype=np.float32)
            for j, chunk in enumerate(batch):
                for i, tr in enumerate(chunk):
                    launch_lp[j, i] = tr.launch.log_prob if tr.launch.active else 0.0
                    goal_lp[j, i] = tr.goal.log_prob if tr.goal.active else 0.0
                    micro_lp[j, i] = tr.micro_old_log_prob
            return launch_lp, goal_lp, micro_lp

        batch_size = max(1, min(config.chunk_batch_size, len(chunks)))
        totals = {
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
            "entropy_goal": 0.0, "entropy_launch": 0.0,
            "entropy_motion": 0.0, "entropy_resource": 0.0,
            "macro_fraction": 0.0,
        }
        updates = 0
        stop = False
        target_kl = getattr(config, "target_kl", 0.03)

        for _ in range(config.update_epochs):
            if stop:
                break
            order = np.random.permutation(len(chunks))
            for bstart in range(0, len(chunks), batch_size):
                batch = [chunks[i] for i in order[bstart:bstart + batch_size]]
                arrays = [chunk_arrays(c) for c in batch]
                max_len = max(len(c) for c in batch)
                b = len(batch)

                obs_b = np.zeros((b, max_len, obs_dim), dtype=np.float32)
                glob_b = np.zeros((b, max_len, global_dim), dtype=np.float32)
                adv_b = np.zeros((b, max_len), dtype=np.float32)
                ret_b = np.zeros((b, max_len), dtype=np.float32)
                v_old_b = np.zeros((b, max_len), dtype=np.float32)
                m_adv_b = np.zeros((b, max_len), dtype=np.float32)
                m_ret_b = np.zeros((b, max_len), dtype=np.float32)
                mv_old_b = np.zeros((b, max_len), dtype=np.float32)
                m_active_b = np.zeros((b, max_len), dtype=np.float32)
                type_b = np.zeros((b, max_len), dtype=np.int64)
                valid_b = np.zeros((b, max_len), dtype=np.float32)
                h0_b = np.zeros((b, hidden_dim), dtype=np.float32)
                head_active_b = {k: np.zeros((b, max_len)) for k in _ALL_HEADS}
                head_action_b = {
                    k: np.zeros((b, max_len), dtype=np.int64) for k in _ALL_HEADS
                }
                head_mask_b = {
                    k: np.zeros((b, max_len, _HEAD_DIMS[k]), dtype=bool)
                    for k in _ALL_HEADS
                }
                for j, (obs, glob, ha, hact, hm, h0) in enumerate(arrays):
                    length = obs.shape[0]
                    obs_b[j, :length] = obs
                    glob_b[j, :length] = glob
                    adv_b[j, :length] = [
                        micro_adv[index_of[id(tr)]] for tr in batch[j]
                    ]
                    ret_b[j, :length] = [
                        micro_ret[index_of[id(tr)]] for tr in batch[j]
                    ]
                    v_old_b[j, :length] = [tr.value for tr in batch[j]]
                    m_adv_b[j, :length] = [
                        macro_adv[index_of[id(tr)]] for tr in batch[j]
                    ]
                    m_ret_b[j, :length] = [
                        macro_ret[index_of[id(tr)]] for tr in batch[j]
                    ]
                    mv_old_b[j, :length] = [tr.macro_value for tr in batch[j]]
                    m_active_b[j, :length] = [
                        1.0 if tr.macro_active() else 0.0 for tr in batch[j]
                    ]
                    type_b[j, :length] = [tr.missile_type for tr in batch[j]]
                    valid_b[j, :length] = 1.0
                    h0_b[j] = h0.astype(np.float32)
                    for k in _ALL_HEADS:
                        head_active_b[k][j, :length] = ha[k]
                        head_action_b[k][j, :length] = hact[k]
                        head_mask_b[k][j, :length] = hm[k]

                to = lambda x, dtype=torch.float32: torch.as_tensor(
                    x, dtype=dtype, device=device
                )
                obs_t = to(obs_b)
                glob_t = to(glob_b)
                adv_t = to(adv_b)
                ret_t = to(ret_b)
                v_old_t = to(v_old_b)
                m_adv_t = to(m_adv_b)
                m_ret_t = to(m_ret_b)
                mv_old_t = to(mv_old_b)
                m_active_t = to(m_active_b)
                type_t = to(type_b, torch.long)
                valid_t = to(valid_b)
                h0_t = to(h0_b).detach()
                head_active_t = {k: to(head_active_b[k]) for k in _ALL_HEADS}
                head_action_t = {k: to(head_action_b[k], torch.long) for k in _ALL_HEADS}
                head_mask_t = {k: to(head_mask_b[k], torch.bool) for k in _ALL_HEADS}

                launch_old_lps, goal_old_lps, micro_old_lps = batch_macros(batch)
                launch_old_t = to(launch_old_lps)
                goal_old_t = to(goal_old_lps)
                micro_old_t = to(micro_old_lps)

                hidden = h0_t
                launch_loss_steps = []
                goal_loss_steps = []
                micro_loss_steps = []
                launch_valid_steps = []
                goal_valid_steps = []
                goal_metric_steps = []
                micro_valid_steps = []
                ent_goal_steps, ent_launch_steps = [], []
                ent_motion_steps, ent_resource_steps = [], []
                kl_launch_sum = kl_goal_sum = kl_micro_sum = 0.0
                kl_launch_cnt = kl_goal_cnt = kl_micro_cnt = 0.0
                for t_step in range(max_len):
                    hidden = actor.step_forward(obs_t[:, t_step], hidden)
                    types = type_t[:, t_step]

                    # ---- 宏组：launch / goal 分头 ratio（分布可分解，各自 clip） ----
                    launch_dist = masked_categorical(
                        actor.launch_logits(hidden),
                        head_mask_t["launch"][:, t_step],
                    )
                    launch_lp = launch_dist.log_prob(
                        head_action_t["launch"][:, t_step]
                    )
                    launch_a = head_active_t["launch"][:, t_step]

                    goal_logits_hm = actor.goal_logits(hidden, 0)
                    goal_logits_l = actor.goal_logits(hidden, 2)
                    is_l = (types == 2).float().unsqueeze(-1)
                    goal_logits = goal_logits_hm * (1.0 - is_l) + goal_logits_l * is_l
                    goal_dist = masked_categorical(
                        goal_logits, head_mask_t["goal"][:, t_step]
                    )
                    goal_lp = goal_dist.log_prob(head_action_t["goal"][:, t_step])
                    goal_a = head_active_t["goal"][:, t_step]
                    # 熵闸门：goal 头的优势策略梯度置零（熵正则/KL 观测不受影响）
                    goal_a_eff = goal_a * 0.0 if gate_goal else goal_a

                    m_adv = m_adv_t[:, t_step]
                    launch_ratio = torch.exp(
                        launch_a * launch_lp - launch_old_t[:, t_step]
                    )
                    launch_step_loss = -torch.min(
                        launch_ratio * m_adv,
                        torch.clamp(
                            launch_ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                        ) * m_adv,
                    )
                    launch_loss_steps.append(launch_step_loss)
                    launch_valid_steps.append(launch_a)

                    goal_ratio = torch.exp(goal_a * goal_lp - goal_old_t[:, t_step])
                    goal_step_loss = -torch.min(
                        goal_ratio * m_adv,
                        torch.clamp(
                            goal_ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                        ) * m_adv,
                    )
                    goal_loss_steps.append(goal_step_loss)
                    goal_valid_steps.append(goal_a_eff)
                    goal_metric_steps.append(goal_a)

                    # ---- 微组：motion(分型) + resource（联合 log-prob，单次 clip） ----
                    motion_logits = torch.stack(
                        [actor.motion_logits(hidden, k) for k in range(3)], dim=1
                    )  # [B, 3_types, MOTION_DIM]
                    type_sel = motion_logits[torch.arange(b, device=device), types]
                    motion_dist = masked_categorical(
                        type_sel, head_mask_t["motion"][:, t_step]
                    )
                    motion_lp = motion_dist.log_prob(
                        head_action_t["motion"][:, t_step]
                    )
                    motion_a = head_active_t["motion"][:, t_step]

                    resource_dist = masked_categorical(
                        actor.resource_logits(hidden),
                        head_mask_t["resource"][:, t_step],
                    )
                    resource_lp = resource_dist.log_prob(
                        head_action_t["resource"][:, t_step]
                    )
                    resource_a = head_active_t["resource"][:, t_step]

                    micro_new = motion_a * motion_lp + resource_a * resource_lp
                    micro_ratio = torch.exp(micro_new - micro_old_t[:, t_step])
                    adv_step = adv_t[:, t_step]
                    micro_step_loss = -torch.min(
                        micro_ratio * adv_step,
                        torch.clamp(
                            micro_ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                        ) * adv_step,
                    )
                    micro_valid = valid_t[:, t_step] * (
                        motion_a + resource_a
                    ).clamp(max=1.0)
                    micro_loss_steps.append(micro_step_loss)
                    micro_valid_steps.append(micro_valid)

                    ent_goal_steps.append(goal_a * goal_dist.entropy())
                    ent_launch_steps.append(launch_a * launch_dist.entropy())
                    ent_motion_steps.append(motion_a * motion_dist.entropy())
                    ent_resource_steps.append(resource_a * resource_dist.entropy())

                    with torch.no_grad():
                        log_r_launch = launch_a * launch_lp - launch_old_t[:, t_step]
                        kl_launch_sum += float(
                            (((launch_ratio - 1.0) - log_r_launch) * launch_a).sum()
                        )
                        kl_launch_cnt += float(launch_a.sum())
                        log_r_goal = goal_a * goal_lp - goal_old_t[:, t_step]
                        kl_goal_sum += float(
                            (((goal_ratio - 1.0) - log_r_goal) * goal_a).sum()
                        )
                        kl_goal_cnt += float(goal_a.sum())
                        log_r_micro = micro_new - micro_old_t[:, t_step]
                        kl_micro_sum += float(
                            (((micro_ratio - 1.0) - log_r_micro) * micro_valid).sum()
                        )
                        kl_micro_cnt += float(micro_valid.sum())

                launch_stack = torch.stack(launch_loss_steps)
                goal_stack = torch.stack(goal_loss_steps)
                micro_stack = torch.stack(micro_loss_steps)
                launch_valid_stack = torch.stack(launch_valid_steps)
                goal_valid_stack = torch.stack(goal_valid_steps)
                goal_metric_stack = torch.stack(goal_metric_steps)
                micro_valid_stack = torch.stack(micro_valid_steps)
                launch_total = launch_valid_stack.sum() + 1e-6
                goal_total = goal_valid_stack.sum() + 1e-6
                micro_total = micro_valid_stack.sum() + 1e-6
                # 每头按有效决策数归一化（防逐步机动头淹没低频宏/部署头）
                policy_loss_sum = (
                    (launch_stack * launch_valid_stack).sum() / launch_total
                    + (goal_stack * goal_valid_stack).sum() / goal_total
                    + (micro_stack * micro_valid_stack).sum() / micro_total
                )

                goal_metric_total = goal_metric_stack.sum() + 1e-6
                ent_goal = (
                    (torch.stack(ent_goal_steps) * goal_metric_stack).sum()
                    / goal_metric_total
                )
                ent_launch = (
                    (torch.stack(ent_launch_steps) * launch_valid_stack).sum()
                    / launch_total
                )
                ent_motion = (
                    (torch.stack(ent_motion_steps) * micro_valid_stack).sum() / micro_total
                )
                ent_resource = (
                    (torch.stack(ent_resource_steps) * micro_valid_stack).sum()
                    / micro_total
                )
                entropy_sum = (
                    ent_goal + ent_launch + ent_motion + ent_resource
                ) * 0.5

                # ---- 价值损失：宏/微两个头（部署头由 _deploy_buffer 单独结算） ----
                values_all = critic.values(
                    glob_t.reshape(-1, global_dim), obs_t.reshape(-1, obs_dim)
                )
                micro_pred = values_all["micro"].reshape(b, max_len)
                macro_pred = values_all["macro"].reshape(b, max_len)

                micro_clipped_v = v_old_t + torch.clamp(
                    micro_pred - v_old_t,
                    -config.value_clip_ratio, config.value_clip_ratio,
                )
                micro_v_loss = 0.5 * torch.maximum(
                    (micro_pred - ret_t).pow(2), (micro_clipped_v - ret_t).pow(2)
                )
                micro_v_loss = (micro_v_loss * valid_t).sum() / (valid_t.sum() + 1e-6)

                macro_clipped_v = mv_old_t + torch.clamp(
                    macro_pred - mv_old_t,
                    -config.value_clip_ratio, config.value_clip_ratio,
                )
                macro_v_loss = 0.5 * torch.maximum(
                    (macro_pred - m_ret_t).pow(2), (macro_clipped_v - m_ret_t).pow(2)
                )
                macro_v_loss = (
                    macro_v_loss * m_active_t
                ).sum() / (m_active_t.sum() + 1e-6)

                value_loss = micro_v_loss + macro_v_loss
                loss = (
                    policy_loss_sum
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy_sum
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    config.max_grad_norm,
                )
                optimizer.step()

                totals["policy_loss"] += float(policy_loss_sum.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy_sum.item())
                totals["entropy_goal"] += float(ent_goal.item())
                totals["entropy_launch"] += float(ent_launch.item())
                totals["entropy_motion"] += float(ent_motion.item())
                totals["entropy_resource"] += float(ent_resource.item())
                totals["macro_fraction"] += float(m_active_t.mean().item())
                approx_kl = (
                    kl_launch_sum / max(kl_launch_cnt, 1.0)
                    + kl_goal_sum / max(kl_goal_cnt, 1.0)
                    + kl_micro_sum / max(kl_micro_cnt, 1.0)
                ) / 3.0
                totals["approx_kl"] += approx_kl
                updates += 1
                if approx_kl > 1.5 * target_kl:
                    stop = True
                    break

        return {key: value / max(updates, 1) for key, value in totals.items()}
