# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/22 01:31:42.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ZNY_GeneralType:
	# 质量（kg）
	quality: float = 0.0
	# 最大速度（km/h)
	maxSpeed: float = 0.0
	# 最大高度（m）
	maxHeight: float = 0.0
	# 航程（km）
	voyage: float = 0.0
	# 续航时间（min）
	enduranceTime: float = 0.0
	# 机长（车长）（m）
	length: float = 0.0
	# 机宽（车宽、弹直径）（m）
	width: float = 0.0
	# 机高（车高、弹翼展、人高）（m）
	height: float = 0.0
	# 最小视野塔特性（m^2）
	minRCS: float = 0.0
	# 最大视野塔特性（m^2）
	maxRCS: float = 0.0
	# 最小红外特征(w/sr)
	minInfrared: float = 0.0
	# 最大红外特征(w/sr)
	maxInfrared: float = 0.0
