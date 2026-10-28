# -*- coding: utf-8 -*-#
# Version:7, Last Modified time: 2024/06/25 18:57:51.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class InterceptorLaunchC:
	# 自爆虫群发射指令
	cmd: InterceptorLaunchCmd = field(default_factory=InterceptorLaunchCmd)
