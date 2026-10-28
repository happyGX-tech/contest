# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/04 17:04:18.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class RaderExternal:
	# 张角限制（欧拉角）
	range: Vector3d = field(default_factory=Vector3d)
	# 视野塔探测距离(米)
	distance: float = 0.0
	# 视野塔扫描周期(度/秒)
	scanSpeed: float = 0.0
