from dataclasses import dataclass, field

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Area(object):
    # 区域类型
    type: str = ""
    # 区域坐标
    coordinates: list[list[list[float]]] = field(default_factory=list)

    # 毁灭者/突击者区域
    coordinatesHM:list[list[list[float]]] = field(default_factory=list)