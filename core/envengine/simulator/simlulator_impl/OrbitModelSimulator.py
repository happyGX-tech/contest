import datetime
import time
from typing import Callable

from envengine.common import SimmerCommandType
from envengine.sdk.base_struct.Basic import Vector3d, DetectInfo
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command
from envengine.sdk.Util import UtilsPy
from envengine.sdk.Util import State_py
from envengine.simulator.decorator import Simulator, timer_decorator
from envengine.simulator.interfaces import ISimulator
# from envengine.simulator.models.OrbitModel.OrbitModelPy import JTC_OrbitModel
from envengine.simulator.simulator_factory import SimulatorFactory

EARTH_RADIUS = 6371000.0


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

        # 天眼轨道解算
        # dt = datetime.datetime.fromtimestamp(self.sim_time / 1000)
        # self.model.update(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1000. / 1000.,
        #                   50 / 1000)
        # state: State_py = self.model.getState()
        #
        # pos = state.posEcf()
        # vel = state.velEcf()
        # lla = UtilsPy.CoordinateHelper.ecefToLla_py(pos)
        # entity = self.entity_ext.entity
        # entity.lla.x, entity.lla.y, entity.lla.z = lla.x(), lla.y(), lla.z()
        # entity.posEcf.x, entity.posEcf.y, entity.posEcf.z = pos.x(), pos.y(), pos.z()
        # entity.velEcf.x, entity.velEcf.y, entity.velEcf.z = vel.x(), vel.y(), vel.z()

        # 执行探测
        if self.sim_time % 1000 == 0:
            self.execute_detection()

        # 发送探测信息
        # self.send_detect_info()

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

    # @timer_decorator
    def send_detect_info(self):
        """
        发送探测信息
        """
        if self.entity_ext.entity.detectInfo:
            communication_candidate_simulators: list[ISimulator] = self._simulator_factory.get_simulators_by_side(
                self.entity_ext.entity.sideId)

            # 如果已经发送过同样的探测信息，则不再发送
            can_communication_send_simulators = []
            for i in communication_candidate_simulators:
                if i.entity_ext.entity.id not in self._delivered_to:
                    can_communication_send_simulators.append(i)
                    self._delivered_to.add(i.entity_ext.entity.id)
            # 构建探测信息指令
            command_list: list[Command] = [Command(
                executorId=i.entity_ext.entity.id,
                commandTypeId=SimmerCommandType.DETECT_STATUS_UPDATE,
                commandAttributes=self.entity_ext.entity.detectInfo
            ) for i in can_communication_send_simulators]

            # 指令转发
            self._send_commands(command_list)

    def execute_detection(self):
        """
        执行探测，调用使用天眼指令才开始探测，可以探测到全部自爆虫群
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
        # self.model = JTC_OrbitModel("1 68310U 26057S   26147.90258910  .00262022  00000-0  17035-2 0  9994",
        #                             "2 68310  97.2843 274.2645 0001211  70.9562 289.1834 15.75199990")
        pass
