# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2023/10/19 17:17:50.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class RadarDetectRegionChangeCmd:
	# rcs开关状态
	status: bool = False
	# 新的探测范围
	maxDistance: float = 0.0
	# 修改当前视野塔扫描指向角度
	direct: Vector3d = field(default_factory=Vector3d)
