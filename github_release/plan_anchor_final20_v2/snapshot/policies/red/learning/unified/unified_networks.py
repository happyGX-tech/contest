"""统一策略网络：GRU 共享主干 + 分型动作头 + 三价值头集中 Critic。

三决策时钟（同一 checkpoint、共享表征，优势按时间尺度分离）：
  部署时钟（每局一次）：DeployHead + V_D（终局比分结算）
  宏时钟（发射/选靶事件）：launch/goal 头 + V_M（SMDP 段回报）
  微时钟（每步）：motion/resource 头 + V_C（逐步 GAE）

Actor（执行期只依赖局部观测与自身隐状态）:
    obs → embed → GRU(h_prev) → h
    h → 机动头  ×3 分型 Linear(3)   左(0)/持(1)/右(2)（仓库约定）
    h → 目标头  H/M 共享 + L 独立 adapter Linear(TARGET_DIM)
    h → 发射头  Linear(4)            NOW/W10/W25/W50（初始 P(NOW)≈3%）
    h → 能力头  Linear(2)            Bernoulli(用/不用卫星)
    部署头独立 MLP：DEPLOY_FEATURE_DIM → (x, y) tanh 高斯。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .unified_encoder import (
    DEPLOY_FEATURE_DIM,
    GOAL_SLOTS,
    SECTOR_ACTIONS,
    UNIFIED_OBS_DIM,
)

MOTION_DIM = 3
LAUNCH_DIM = 4                        # NOW / WAIT_10 / WAIT_25 / WAIT_50
TARGET_DIM = GOAL_SLOTS + SECTOR_ACTIONS + 1   # 实体槽 + 搜索扇区 + KEEP
RESOURCE_DIM = 2

# 目标头动作索引（发射前目标头仅在 launch=NOW 时联合采样，KEEP 不参与）
SECTOR_INDEX_0 = GOAL_SLOTS           # 7: SEARCH_SECTOR_0（仅 L 合法）
KEEP_INDEX = TARGET_DIM - 1           # 9: 保持当前目标
# 发射头 WAIT 选项 → 持续等待步数（选定后持续执行，不重复记录 log-prob）
LAUNCH_WAIT_STEPS = {1: 10, 2: 25, 3: 50}

TYPE_INDEX = {21000: 0, 21001: 1, 21002: 2}


class UnifiedActor(nn.Module):
    def __init__(self, obs_dim: int = UNIFIED_OBS_DIM, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        # 分型机动头（三类动力学/任务差异大，防梯度互扰）
        self.motion_heads = nn.ModuleList(
            nn.Linear(hidden_dim, MOTION_DIM) for _ in range(3)
        )
        # 目标头：H/M 共享，L 独立 adapter（搜索/打船是其主业）
        self.goal_head_hm = nn.Linear(hidden_dim, TARGET_DIM)
        self.goal_head_l = nn.Linear(hidden_dim, TARGET_DIM)
        # 发射头：NOW 初始 logit 压低，形成可探索的自然波次
        self.launch_head = nn.Linear(hidden_dim, LAUNCH_DIM)
        with torch.no_grad():
            self.launch_head.bias[0] = -3.5
            for i in range(1, LAUNCH_DIM):
                self.launch_head.bias[i] = 0.0
        self.resource_head = nn.Linear(hidden_dim, RESOURCE_DIM)

    def initial_hidden(self, batch: int = 1, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_dim, device=device)

    def step_forward(self, obs: torch.Tensor, hidden_prev: torch.Tensor) -> torch.Tensor:
        """单步 GRU 前向（hidden_prev 在更新期 detach，不做 BPTT）。"""
        embedded = self.embed(obs)
        return self.gru(embedded, hidden_prev)

    def motion_logits(self, hidden: torch.Tensor, type_index: int) -> torch.Tensor:
        return self.motion_heads[type_index](hidden)

    def goal_logits(self, hidden: torch.Tensor, type_index: int) -> torch.Tensor:
        return self.goal_head_l(hidden) if type_index == 2 else self.goal_head_hm(hidden)

    def launch_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.launch_head(hidden)

    def resource_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.resource_head(hidden)


class DeployHead(nn.Module):
    """部署头：输出高斯均值（未压缩）。tanh 压缩只由 SquashedGaussian 做一次。"""

    def __init__(self, feature_dim: int = DEPLOY_FEATURE_DIM, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SquashedGaussian:
    """tanh(Normal) 分布的采样与精确 log-prob。"""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.std = log_std.clamp(-3.0, 0.5).exp().expand_as(mean)
        self.normal = Normal(mean, self.std)

    def sample(self):
        noise = self.normal.sample()
        action = torch.tanh(noise)
        return action, self.log_prob(noise, action)

    def mode(self):
        noise = self.normal.mean
        action = torch.tanh(noise)
        return action, self.log_prob(noise, action)

    def log_prob(self, noise: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        base = self.normal.log_prob(noise).sum(dim=-1)
        correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return base - correction


class CentralCritic(nn.Module):
    """共享主干 + 三价值头（部署/宏/微），全部 CTDE 输入 [global, local]。"""

    def __init__(self, global_dim: int, obs_dim: int = UNIFIED_OBS_DIM, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(global_dim + obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.deploy_value = nn.Linear(hidden_dim, 1)
        self.macro_value = nn.Linear(hidden_dim, 1)
        self.micro_value = nn.Linear(hidden_dim, 1)

    def _cat(self, global_state: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat((global_state, obs), dim=-1))

    def values(self, global_state: torch.Tensor, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self._cat(global_state, obs)
        return {
            "deploy": self.deploy_value(h).squeeze(-1),
            "macro": self.macro_value(h).squeeze(-1),
            "micro": self.micro_value(h).squeeze(-1),
        }

    def forward(self, global_state: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """兼容旧接口：返回微价值。"""
        return self.micro_value(self._cat(global_state, obs)).squeeze(-1)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor):
    """按掩码构造 Categorical；全 False 直接报错（不再静默放开全部动作）。

    每个头必须存在合法 no-op（KEEP/AUTOGUIDE/OFF/WAIT），调用方保证
    掩码非空；全空掩码意味着协议被破坏，必须立刻暴露。
    """
    if not bool(mask.any()):
        raise ValueError(
            "动作掩码全为 False：每个头必须保留合法 no-op "
            "（KEEP/AUTOGUIDE/OFF/WAIT），请检查目标槽有效性与掩码构造"
        )
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return Categorical(logits=logits)
