# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2023/10/18 14:01:05.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d
from .EntityInfo import EntityInfo


@dataclass_json
@dataclass
class DetectedEntity:
	# 位置
	pos: Vector3d = field(default_factory=Vector3d)
	# 速度
	vel: Vector3d = field(default_factory=Vector3d)
	# 实体信息
	entityInfo: EntityInfo = field(default_factory=EntityInfo)
	# 地固系坐标
	ecf: Vector3d = field(default_factory=Vector3d)
	# 探测时刻
	time: int = 0
	# 是否可探测到
	visible: bool = False
	# 视野塔id
	radarId: int = 0
