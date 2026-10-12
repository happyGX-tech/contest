"""统一 POMDP 红方策略（多头 MAPPO）。

取消上下两层规则/学习分层，把部署、目标选择、机动、卫星全部建模为
单一共享循环策略的事件门控多头动作，配合集中式 Critic 的 MAPPO 训练。
"""

from .unified_policy import UnifiedConfig, UnifiedMAPPO
from .topformer import TopFormerConfig, TopFormerMAPPO
from .info_center import InfoCenter

__all__ = ["UnifiedConfig", "UnifiedMAPPO", "TopFormerConfig", "TopFormerMAPPO", "InfoCenter"]
