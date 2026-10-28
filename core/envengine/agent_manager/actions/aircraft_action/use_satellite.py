# -*- coding: utf-8 -*-#
from typing import Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from envengine.common import SimmerCommandType


@dataclass_json
@dataclass
class UseSatelliteAction(object):
    """
    请求天眼支援，由引擎统一检查可用次数。

    Attributes:
        executor_id : 执行者id
        commandType_id : 指令类型
    """

    executor_id: Optional[int] = -1
    commandType_id: int = SimmerCommandType.EXECUTE_SATELLITE_DETECTION
    requested: bool = True
