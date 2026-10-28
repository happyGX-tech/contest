# -*-coding:utf-8 -*-
import logging
from typing import Callable

from envengine.common import SimmerCommandType, DamageC, DamageCmd
from envengine.sdk.Util.UtilsPy import Vector3D
from envengine.sdk.base_struct.Basic import Vector3d, DetectInfo
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command
from envengine.sdk.Util import UtilsPy
from envengine.sdk.Util import State_py
from envengine.simulator.decorator import Simulator
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.models.ACMMissileModel import Missile
from envengine.simulator.simulator_factory import SimulatorFactory

logger = logging.getLogger(__name__)


@Simulator.register("CruiseMissileLS")
class CompCruiseMissileLSimulator(ISimulator):
    """
    清道夫

    Attributes:
        launch: 发射时间
    """

    def __init__(self, entity_ext: EntityExt,
                 send_commands: Callable[[list[Command]], None],
                 send_events: Callable[[dict], None],
                 simulator_factory: SimulatorFactory = None):
        super().__init__(entity_ext, send_commands, send_events, simulator_factory)
        self.ret = -1
        self.launch = -1
        self.damage_point = 1

    @property
    def simulator_sim_step(self) -> float:
        """
        仿真器内部步长（毫秒）
        :return: 内部步长（毫秒）
        """
        return 50

    def update(self) -> None:
        """
        执行仿真步进
        """
        if self.launch < 0:
            return

        if self.ret < 0 and (self.sim_time - self.launch <= 1800_000):
            # 更新探测信息
            if self.sim_time % 1000 == 0:
                self.execute_detection()

            self.ret = self.model.Update()
            state: State_py = self.model.getState()
            self._update_state(state)
        else:
            if self.entity_ext.entity.isVisible:
                if self.ret < 0:
                    logger.warning(f"清道夫 '{self.entity_ext.entity.nameChn}' 飞行时间超时自爆")

                # 自爆
                self.entity_ext.entity.isVisible = False
                self.entity_ext.entity.survivePoints = 0
                # 构建自爆掉血指令
                self_command_list: list[Command] = [Command(
                    prevTriggerId=self.entity_ext.entity.id,
                    executorId=self.entity_ext.entity.id,
                    commandTypeId=SimmerCommandType.DAMAGE,
                    commandAttributes=DamageC(
                        DamageCmd(
                            damagePoint=self.damage_point
                        )
                    ).to_json()
                )]

                # 指令转发
                self._send_commands(self_command_list)

                # 弹爆炸了，计算与周围仿真器的距离，如果距离小于100米，则通知仿真器进行爆炸
                damage_candidate_simulators: list[ISimulator] = self._simulator_factory.get_simulators_by_side(
                    1 if self.entity_ext.entity.sideId == 0 else 0)
                can_damage_simulators: list[ISimulator] = self._get_targets_within_self_range(
                    damage_candidate_simulators,
                    max_range=100)
                if not can_damage_simulators:
                    return

                # 构建掉血指令
                command_list: list[Command] = [Command(
                    prevTriggerId=self.entity_ext.entity.id,
                    executorId=i.entity_ext.entity.id,
                    commandTypeId=SimmerCommandType.DAMAGE,
                    commandAttributes=DamageC(
                        DamageCmd(
                            damagePoint=self.damage_point
                        )
                    ).to_json()
                ) for i in can_damage_simulators]

                # 指令转发
                self._send_commands(command_list)


    def execute_detection(self):
        """
        清道夫探测三公里内的暗巢。
        :return:
        """

        """
        9500:  暗巢
        24000: 自爆虫群
        """



        own_pos = self.entity_ext.entity.posEcf
        detected: list[ISimulator] = []
        if self._simulator_factory is not None:
            ships: list[ISimulator] = self._simulator_factory.get_simulators_by_type(9500)
            for sim in ships:
                if (sim.entity_ext.entity.isVisible
                        and sim.entity_ext.entity.survivePoints > 0
                        and self._is_geometrically_visible(sim.entity_ext.entity.posEcf, own_pos, 3 * 1000)):
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

        self.model = None

        self.ret = -1
        self.launch = -1

    def init_model(self):
        self.model = Missile()
        self.set_lla(self.entity_ext.entity.lla)

    def set_lla(self, lla: Vector3d) -> None:
        """
        初始设置lla
        """
        self.entity_ext.entity.lla = lla
        self.entity_ext.entity.posEcf = self._get_ecf_position()
        missile_lla = UtilsPy.Vector3D(self.entity_ext.entity.lla.x, self.entity_ext.entity.lla.y,
                                       self.entity_ext.entity.lla.z + 0.1)

        self.model.Init(self.simulator_sim_step / 1000, missile_lla * ISimulator.MODEL_SCALE, 5)

        # Deployment may call set_lla again; mission start is deferred until
        # every platform has received its final position.
        self.launch = -1

        self.model.SetDesiredHeight(2_000 * ISimulator.MODEL_SCALE)
        self.model.Save(False, self.entity_ext.entity.id)
        self.set_speed(100 * ISimulator.MODEL_SCALE)

    def start_mission(self) -> None:
        """Start the initialized model once after deployment completes."""

        if self.launch >= 0:
            return

        # 发射
        if self._simulator_factory is None:
            return
        targets:list[ISimulator] = self._simulator_factory.get_simulators_by_type(9400) # 发射目标为母巢
        if targets:
            target_lla = targets[0].entity_ext.entity.lla
            self.model.Launch(Vector3D(target_lla.x, target_lla.y, target_lla.z) * ISimulator.MODEL_SCALE)

        self.launch = self.sim_time

    def set_speed(self, speed: float) -> None:
        """
        初始设置速度
        """
        self.model.SetDesiredSpeed(100 * ISimulator.MODEL_SCALE)

    def command_received(self, command: Command) -> None:
        """
        自定义指令接收
        :param command:
        :return:
        """
        if command.commandTypeId == SimmerCommandType.ATTACK_COMMANDER_START:
            pass
        elif command.commandTypeId == SimmerCommandType.SET_DESIRED_ACC_Z:
            if command.commandAttributes["acc_z"] == 0:
                self.model.ClearDesiredAccZ()
            else:
                self.model.SetDesiredAccZ(command.commandAttributes["acc_z"] * 20 * 9.8)
        elif command.commandTypeId == SimmerCommandType.SET_DESIRED_VEL_X:
            self.model.SetDesiredSpeed(command.commandAttributes["vel_x"] * ISimulator.MODEL_SCALE)
        elif command.commandTypeId == SimmerCommandType.CHANGE_MISSILE_TARGET:
            # Match Launch: scale LLA before converting to native ECEF.
            pos = UtilsPy.CoordinateHelper.llaToEcef_py(Vector3D(  # type:ignore
                command.commandAttributes["target"]["x"],
                command.commandAttributes["target"]["y"],
                command.commandAttributes["target"]["z"]
            ) * ISimulator.MODEL_SCALE)
            self.model.SetTargetEcf(pos, Vector3D(0, 0, 0))
        else:
            super().command_received(command)
