from typing import Callable

from envengine.sdk.base_struct.Basic import Vector3d, DetectInfo
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command
from envengine.simulator.decorator import Simulator
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.simulator_factory import SimulatorFactory


@Simulator.register("TleSatelliteS")
class OrbitModelSimulator(ISimulator):
    """
    天眼
    """

    def __init__(self, entity_ext: EntityExt,
                 send_commands: Callable[[list[Command]], None],
                 send_events: Callable[[dict], None],
                 simulator_factory: SimulatorFactory = None):
        super().__init__(entity_ext, send_commands, send_events, simulator_factory)
        self.entity_ext.entity.stage = 1

    @property
    def simulator_sim_step(self) -> float:
        """
        仿真器内部步长（毫秒）
        :return: 内部步长（毫秒）
        """
        return 1000

    def update(self) -> None:
        """
        执行仿真步进
        """

        # 执行探测
        if self.sim_time % 1000 == 0:
            self.execute_detection()


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

    def execute_detection(self):
        """
        天眼启用期间，全局探测各类虫巢和已出巢虫群
        :return:
        """

        """
        9400:   母巢
        9500:   暗巢
        9600:   哨卫巢
        24000:  自爆虫群
        """

        if self._simulator_factory is None or not self._simulator_factory.is_using_satellite():
            return

        detected: list[ISimulator] = []
        for detect_t in [24000, 9400, 9500, 9600]:
            for sim in self._simulator_factory.get_simulators_by_type(detect_t):
                if sim.entity_ext.entity.isVisible and sim.entity_ext.entity.survivePoints > 0:
                    detected.append(sim)

        if not detected:
            return

        # 组装探测信息
        detect_info: dict[int, DetectInfo] = {
            target.entity_ext.entity.id: DetectInfo(
                detect_from=self.entity_ext.entity.id,
                time=int(self.sim_time),
                entity_id=target.entity_ext.entity.id,
                entity_type=target.entity_ext.entity.entityType,
                nameChn=target.entity_ext.entity.nameChn,
                lla=target.entity_ext.entity.lla,
                pos_ecf=target.entity_ext.entity.posEcf,
                vel_ecf=target.entity_ext.entity.velEcf
            ) for target in detected}
        # 更新自身探测信息
        self.handel_detect_info(detect_info)

    def reset(self) -> None:
        """重置到初始状态"""
        super().reset()

    def init_model(self) -> None:
        #                             "2 68310  97.2843 274.2645 0001211  70.9562 289.1834 15.75199990")
        pass
