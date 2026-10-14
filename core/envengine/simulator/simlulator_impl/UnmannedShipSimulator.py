from typing import Callable

from envengine.sdk.Util import UtilsPy
from envengine.sdk.base_struct.Basic import Vector3d
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command
from envengine.simulator.decorator import Simulator
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.simulator_factory import SimulatorFactory


@Simulator.register("UnmannedShipS")
class UnmannedShipSimulator(ISimulator):
    """
    9500 暗巢
    """

    def __init__(self, entity_ext: EntityExt,
                 send_commands: Callable[[list[Command]], None],
                 send_events: Callable[[dict], None],
                 simulator_factory: SimulatorFactory = None):
        super().__init__(entity_ext, send_commands, send_events, simulator_factory)
        self.pos_flag = -1
        self.entity_ext.entity.stage = 1

    @property
    def simulator_sim_step(self) -> float:
        """
        仿真器内部步长（毫秒）
        :return: 内部步长（毫秒），返回0表示使用引擎步长
        """
        return 0

    def update(self) -> None:
        """
        执行仿真步进
        """
        # self.entity_ext.entity.lla.x += 0.01
        # if self.entity_ext.entity.lla.x > 180:
        #     self.entity_ext.entity.lla.x = 0
        if self.pos_flag == -1:
            lla = self.entity_ext.entity.lla
            ecf: UtilsPy.Vector3D = UtilsPy.CoordinateHelper.llaToEcef_py(UtilsPy.Vector3D(lla.x, lla.y, lla.z))
            self.entity_ext.entity.posEcf = Vector3d(ecf.x(), ecf.y(), ecf.z())
            self.pos_flag = 1

    def set_lla(self, lla: Vector3d) -> None:
        """
        初始设置lla
        """
        pass

    def set_speed(self, speed: float) -> None:
        """
        初始设置速度
        """
        pass

    def reset(self) -> None:
        """重置到初始状态"""
        super().reset()
        self.pos_flag = -1

    def init_model(self) -> None:
        pass