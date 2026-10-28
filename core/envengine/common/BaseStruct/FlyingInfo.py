# -*- coding: utf-8 -*-#
# Version:5, Last Modified time: 2023/10/18 14:01:51.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .RedMissileFlyInfo import RedMissileFlyInfo
from .DetectThreatInfo import DetectThreatInfo
from .TargetInfo import TargetInfo
from .IcptThreatInfo import IcptThreatInfo
from .DetectThreatInfo_Active import DetectThreatInfo_Active
from .DetectThreatInfo_Infrared import DetectThreatInfo_Infrared
from .TargetInfo_Passive import TargetInfo_Passive
from .TargetInfo_Infrared import TargetInfo_Infrared
from .IcptThreatInfo_Passive import IcptThreatInfo_Passive
from .JamInfo_Passive import JamInfo_Passive
from .JamInfo_Infrared import JamInfo_Infrared
from .DamageInfo_Infrared import DamageInfo_Infrared
from .JamInfo_Active import JamInfo_Active


@dataclass_json
@dataclass
class FlyingInfo:
	# 飞行中的弹
	missiles: list[RedMissileFlyInfo] = field(default_factory=list[RedMissileFlyInfo])
	# 被动视野塔识别到的探测威胁列表
	detectThreats: list[DetectThreatInfo] = field(default_factory=list[DetectThreatInfo])
	# 主动视野塔探测到的目标列表
	targets: list[TargetInfo] = field(default_factory=list[TargetInfo])
	# 主动视野塔识别到的拦截威胁列表
	icptThreats: list[IcptThreatInfo] = field(default_factory=list[IcptThreatInfo])
	# 主动视野塔识别到的探测威胁信息
	detectedThreatInfo_Active: list[DetectThreatInfo_Active] = field(default_factory=list[DetectThreatInfo_Active])
	# 红外识别到的探测威胁信息
	detectedThreatInfo_Infrared: list[DetectThreatInfo_Infrared] = field(default_factory=list[DetectThreatInfo_Infrared])
	# 被动视野塔识别到的目标信息
	targetInfo_Passive: list[TargetInfo_Passive] = field(default_factory=list[TargetInfo_Passive])
	# 红外识别到的目标信息
	targetInfo_Infrared: list[TargetInfo_Infrared] = field(default_factory=list[TargetInfo_Infrared])
	# 被动视野塔识别到的拦截威胁信息
	icptThreatInfo_Passive: list[IcptThreatInfo_Passive] = field(default_factory=list[IcptThreatInfo_Passive])
	# 被动视野塔识别到的干扰信息
	jamInfo_Passive: list[JamInfo_Passive] = field(default_factory=list[JamInfo_Passive])
	# 红外识别到的干扰信息
	jamInfo_Infrared: list[JamInfo_Infrared] = field(default_factory=list[JamInfo_Infrared])
	# 红外识别到的毁伤信息
	damageInfo_Infrared: list[DamageInfo_Infrared] = field(default_factory=list[DamageInfo_Infrared])
	# 主动视野塔探测到的干扰信息
	jamInfo_Active: list[JamInfo_Active] = field(default_factory=list[JamInfo_Active])
