# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2023/10/18 14:53:20.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .LaunchIntructionIn import LaunchIntructionIn
from .PosVelAcc import PosVelAcc


@dataclass_json
@dataclass
class InterceptorLaunchCmd:
	# 自爆虫群id
	weaponId: int = 0
	# 目标id
	targetId: int = 0
	# 指控发射诸元输入
	launchPara: LaunchIntructionIn = field(default_factory=LaunchIntructionIn)
	# 目标预测位置速度
	targetPosLLA: PosVelAcc = field(default_factory=PosVelAcc)
