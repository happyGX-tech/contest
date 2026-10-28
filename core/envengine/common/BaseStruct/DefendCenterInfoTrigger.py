# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2023/10/18 17:28:43.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .ShipInfo import ShipInfo
from .IcptInfo import IcptInfo
from .MissileTargetInfo import MissileTargetInfo


@dataclass_json
@dataclass
class DefendCenterInfoTrigger:
	# 引擎时间
	engineTime: int = 0
	# 舰船信息
	ships: list[ShipInfo] = field(default_factory=list[ShipInfo])
	# 自爆虫群信息
	icpts: list[IcptInfo] = field(default_factory=list[IcptInfo])
	# 弹目标信息
	missileTargets: list[MissileTargetInfo] = field(default_factory=list[MissileTargetInfo])
