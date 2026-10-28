# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/10/15 02:07:15.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .ImageElectromagneticInfo import ImageElectromagneticInfo
from .AircraftElectromagneticInfo import AircraftElectromagneticInfo
from .ShipElectromagneticInfo import ShipElectromagneticInfo
from .RadarElectromagneticInfo import RadarElectromagneticInfo
from .AirportElectromagneticInfo import AirportElectromagneticInfo
from .PortElectromagneticInfo import PortElectromagneticInfo
from .MissileDefenseElectromagneticInfo import MissileDefenseElectromagneticInfo


@dataclass_json
@dataclass
class TargetCharacterInfoSetCmd:
	# 视野塔工作频率
	radarFrequency: float = 0.0
	# 信号带宽
	signalBandWidth: float = 0.0
	# 极化方式
	polarizationMode: int = 0
	# 成像区匹配电磁特征
	imageElectromagneticInfo: list[ImageElectromagneticInfo] = field(default_factory=list[ImageElectromagneticInfo])
	# 飞机目标电磁特征
	aircraftElectromagneticInfo: list[AircraftElectromagneticInfo] = field(default_factory=list[AircraftElectromagneticInfo])
	# 舰船目标电磁特征
	shipElectromagneticInfo: list[ShipElectromagneticInfo] = field(default_factory=list[ShipElectromagneticInfo])
	# 时敏视野塔目标电磁特征
	radarElectromagneticInfo: list[RadarElectromagneticInfo] = field(default_factory=list[RadarElectromagneticInfo])
	# 机场环境电磁特征
	airportElectromagneticInfo: list[AirportElectromagneticInfo] = field(default_factory=list[AirportElectromagneticInfo])
	# 港口环境电磁特征
	portElectromagneticInfo: list[PortElectromagneticInfo] = field(default_factory=list[PortElectromagneticInfo])
	# 导弹防御阵地目标电磁特征
	missileDefenseElectromagneticInfo: list[MissileDefenseElectromagneticInfo] = field(default_factory=list[MissileDefenseElectromagneticInfo])
