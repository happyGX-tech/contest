# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/15 01:41:12.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class InterfereInfo:
	# 视野塔工作频率
	radarFrequency: float = 0.0
	# 信号带宽
	signalBandWidth: float = 0.0
	# 极化方式，0-H极化，1-V极化，2-L左旋极化，3-R右旋极化
	polarizationMode: float = 0.0
	# 干扰编号
	equipInterfereCode: float = 0.0
	# 干扰类型,1充气式角反,2箔条,3弦外有源干扰,4舰载有源干扰
	equipInterfereType: float = 0.0
	# 干扰数量
	equipInterfereNum: float = 0.0
	# 干扰位置X
	interferePositionX: float = 0.0
	# 干扰位置Y
	interferePositionY: float = 0.0
	# 干扰位置Z
	interferePositionZ: float = 0.0
	# 干扰中心点位置X
	centerPositionX: float = 0.0
	# 干扰中心点位置Y
	centerPositionY: float = 0.0
	# 干扰中心点位置Z
	centerPositionZ: float = 0.0
	# 方位角
	yawAngle: float = 0.0
	# 俯仰角
	pitchAngle: float = 0.0
	# 功率
	power: float = 0.0
	# 开机/释放时间,默认值为-1不开机，为正时相对发射T0的时间
	bootTime: float = 0.0
