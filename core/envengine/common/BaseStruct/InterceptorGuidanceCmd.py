# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2023/10/18 14:51:04.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .PosVelAcc import PosVelAcc


@dataclass_json
@dataclass
class InterceptorGuidanceCmd:
	# 自爆虫群id
	id: int = 0
	# 目标id
	targetId: int = 0
	# 目标预测速度位置
	targetPosLLA: PosVelAcc = field(default_factory=PosVelAcc)
