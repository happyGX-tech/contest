# -*- coding: utf-8 -*-#
from typing import Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from envengine.common import SimmerCommandType
from envengine.sdk.base_struct.Basic import Vector3d


@dataclass_json
@dataclass
class ChangeTargetAction(object):
    """
    飞行制导：按已知目标编号进攻，或前往公开地图中的搜索航点。

    Attributes:
        executor_id : 执行者id
        commandType_id : 指令类型
        intent : track 或 search
        target_id : track 指令的目标编号，由引擎解析合法目标位置
        target : search 指令的航点（沿用仿真坐标格式）
    """

    executor_id: Optional[int] = -1
    commandType_id: int = SimmerCommandType.CHANGE_MISSILE_TARGET
    intent: str = "search"
    target_id: Optional[int] = None
    target: Optional[Vector3d] = None
