# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2026/05/07 15:08:26.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .TargetState import TargetState


@dataclass_json
@dataclass
class SeekerOutput_17s:
	# 是否开机
	IsActive: bool = False
	# 目标状态
	TargetStates: list[TargetState] = field(default_factory=list[TargetState])
	# 方位角
	Azimuth: float = 0.0
	# 俯仰角
	Elevation: float = 0.0
	# 搜索状态
	SearchState: str = ""
	# 视野塔图
	RDImage: list[float] = field(default_factory=list[float])
	# BeamAzimuth
	BeamAzimuth: float = 0.0
	# BeamElevation
	BeamElevation: float = 0.0
	# BeamWidth
	BeamWidth: float = 0.0
	# 弹目距
	Distance: float = 0.0
