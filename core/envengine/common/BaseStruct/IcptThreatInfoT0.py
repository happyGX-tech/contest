# -*- coding: utf-8 -*-#
# Version:4, Last Modified time: 2023/10/18 14:00:41.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class IcptThreatInfoT0:
	# id
	id: int = 0
	# 位置
	pos: Vector3d = field(default_factory=Vector3d)
	# 类型
	type: int = 0
	# 自爆虫群信息（自爆虫群型号，自爆虫群数量）
	interceptorInfo: dict[int, int] = field(default_factory=dict[int, int])
