# -*-coding:utf-8 -*-
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from envengine.sdk.base_struct.Entity.EntityExt import EntityExt
from envengine.sdk.base_struct.Message import Task, Event, MapArea
from envengine.sdk.base_struct.Message.Area import Area


@dataclass_json
@dataclass
class Imagine(object):
    # *模拟步长
    simStep: int = 0

    # *DateTime
    simTime: float = 0

    # *模拟运行的总时长
    simEndLogicTime: float = 0

    # *模型列表
    entityList: list[EntityExt] = field(default_factory=list)

    # *事件列表
    eventList: list[Event] = field(default_factory=list)

    # *任务
    taskList: list[Task] = field(default_factory=list)

    # 红方区域
    redArea: Area = field(default=None)

    # 地图区域
    mapArea: MapArea = field(default=None)

    # 天眼最大使用次数
    satelliteMaxUseCount:int = 100

    # 天眼使用时间（分钟）
    satelliteUseMinutes:float = 3

    # 导弹命中率提升条件的判断时间间隔（分钟）
    missileRateIncreaseTimeIntervalMinutes:float = 2.0
    # 导弹命中率提升条件的最小角度
    missileRateIncreaseMinAngle:float = 30.0
    # 导弹命中率提升最大值
    missileRateIncreaseMaxValue:float = 0.2

    # 导弹命中率降低条件的判断时间间隔（分钟）
    missileRateDecreaseTimeIntervalMinutes:float= 2.0
    # 导弹命中率降低判断条件的最大角度
    missileRateDecreaseMaxAngle:float= 10.0
    # 导弹命中率降低最大值
    missileRateDecreaseMaxValue:float= 0.4
