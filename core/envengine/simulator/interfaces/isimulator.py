# -*-coding:utf-8 -*-
import json
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Callable, Any, TYPE_CHECKING

from envengine.common import *
from envengine.common.BaseStruct import PosVelAcc
from envengine.common.Command import DamageC
from envengine.sdk.Util import UtilsPy
from envengine.sdk.base_struct.Basic import Vector3d, DetectInfo
from envengine.sdk.base_struct.Entity import EntityExt, Entity
from envengine.sdk.base_struct.Message import Command, Event
from envengine.simulator.util.scaled_coordinates import scaled_ecef_velocity

if TYPE_CHECKING:
    from envengine.simulator.simulator_factory import SimulatorFactory

logger = logging.getLogger(__name__)


class ISimulator(ABC):
    """
    仿真器接口

    Attributes:
        _entity_ext : 保存实体
    """

    MODEL_SCALE = 10

    def __init__(self,
                 entity_ext: EntityExt,
                 send_commands: Callable[[list[Command]], None],
                 send_events: Callable[[dict], None],
                 simulator_factory: "SimulatorFactory" = None):

        self._entity_ext = deepcopy(entity_ext)
        self._initial_entity_ext = deepcopy(entity_ext)
        self._initial_entity_ext.entity.posEcf = self._convert_lla_to_ecf(self._initial_entity_ext.entity.lla)
        self._last_lla = Vector3d(0, 0, 0)
        self._delivered_to = set()  # 已将探测信息送达的飞行器ID
        self._simulator_factory = simulator_factory
        self._send_commands = send_commands
        self._send_events = send_events
        self.model = None
        self._sim_time = self._simulator_factory.profile.imagineProfile.simTime if self._simulator_factory else 0

    @property
    def sim_time(self) -> float:
        """仿真时间（毫秒）"""
        return self._sim_time

    @sim_time.setter
    def sim_time(self, value):
        self._sim_time = value

    @property
    @abstractmethod
    def simulator_sim_step(self) -> float:
        """
        仿真器内部步长（毫秒）
        子类可以重写此属性来指定自己的内部步长
        :return: 内部步长（毫秒），返回0表示使用引擎步长
        """
        return 0.0

    @property
    def entity_ext(self) -> EntityExt:
        """实体扩展信息（包含所有Entity属性）"""
        return self._entity_ext

    @abstractmethod
    def update(self) -> None:
        """
        仿真步进
        """
        pass

    def _update_state(self, state):
        pos = state.posEcf()
        vel = state.velEcf()
        lla = UtilsPy.CoordinateHelper.ecefToLla_py(pos)

        # 更新实体数据
        entity = self.entity_ext.entity
        entity.lla.x, entity.lla.y, entity.lla.z = lla.x() / ISimulator.MODEL_SCALE, lla.y() / ISimulator.MODEL_SCALE, lla.z() / ISimulator.MODEL_SCALE
        entity.posEcf = self._get_ecf_position()  # posEcf 不能缩放，应该与 lla 对应
        mapped_velocity = scaled_ecef_velocity(lla, vel, ISimulator.MODEL_SCALE)
        entity.velEcf.x, entity.velEcf.y, entity.velEcf.z = mapped_velocity.x, mapped_velocity.y, mapped_velocity.z
        entity.stage = state.stage()

    @abstractmethod
    def set_lla(self, lla: Vector3d) -> None:
        """
        初始设置lla
        """
        pass

    @abstractmethod
    def set_speed(self, speed: float) -> None:
        """
        初始设置速度
        """
        pass

    def reset(self) -> None:
        """重置仿真器"""
        self._entity_ext = deepcopy(self._initial_entity_ext)
        self._delivered_to = set()  # 已将探测信息送达的飞行器ID

    @abstractmethod
    def init_model(self) -> None:
        """
        初始化模型
        :return:
        """
        pass

    def _damage(self, damage_point: float, source: EntityInfo) -> None:
        """
        造成伤害
        :param damage_point: 伤害量
        :param source: 伤害来源
        :return:
        """
        remain = self._entity_ext.entity.survivePoints - damage_point
        self._entity_ext.entity.survivePoints = remain if remain >= 0 else 0
        if remain > 0:
            self._consume_event(SimmerTriggerType.LIFE_POINT_CHANGE,
                                LifePointChangeT(
                                    LifePointChangeTrigger(self._entity_info(self._entity_ext.entity), source,
                                                           self._sim_time, damage_point)))
        else:
            self._entity_ext.entity.isVisible = False
            if self._entity_ext.entity.parentId == -1 or source.id == self._entity_ext.entity.parentId:
                self._consume_event(SimmerTriggerType.DESTROY,
                                    DestroyT(DestroyTrigger(self._entity_info(self._entity_ext.entity), source,
                                                            self._sim_time, damage_point)))
            self._destroy_children()

    def _destroy_children(self) -> None:
        """
        摧毁子节点
        :return:
        """
        if not self._entity_ext.entity.childrenId:
            return
        cmd_list = [
            self.assemble_command(
                children,
                SimmerCommandType.DESTROY,
                DestroyC(DestoryCmd(True)).to_json()
            )
            for children in self._entity_ext.entity.childrenId
        ]
        self._send_commands(cmd_list)

    def _consume_event(self, type: SimmerTriggerType, content: Any) -> None:
        """
        消费事件
        :param type: 类型
        :param content: 事件内容
        :return:
        """
        event = {
            "entityName": self._entity_ext.entity.nameChn,
            "entityId": self._entity_ext.entity.id,
            "parentId": self._entity_ext.entity.parentId,
            "childrenId": self._entity_ext.entity.childrenId,
            "logicTime": self._sim_time,
            "type": type,
            "content": content.to_dict()
        }
        self._send_events(event)

    def _retrieve_children(self, children_id: list[int]) -> None:
        """
        回收子物体
        :param children_id:
        :return:
        """
        # 父物体将数据同步到子物体
        cmd_list = [
            self.assemble_command(
                children,
                SimmerCommandType.CHILD_SYNC,
                ChildSyncC(PosVelAcc(self._entity_ext.entity.lla, self._entity_ext.entity.velNue,
                                     Vector3d(0, 0, 0))).to_json()
            )
            for children in children_id
        ]
        self._send_commands(cmd_list)

    def _release_children(self, children_id: list[int]) -> None:
        """
        释放子物体
        :param children_id:
        :return:
        """
        for children in children_id:
            if children in self._entity_ext.entity.childrenId:
                self._entity_ext.entity.childrenId.remove(children)

    def _entity_info(self, value: Entity) -> EntityInfo:
        """
        获取EntityInfo
        :param value: 实体
        :return: 实体信息
        """
        return EntityInfo(
            id=value.id,
            type=value.entityType,
            typeId=value.typeId,
            side=value.sideId,
        )

    def _target_entity_info_or_self(self, oid: Optional[int] = None) -> EntityInfo:
        """
        获取目标EntityInfo
        :param oid: id
        :return: 若oid不为空且oid指示的实体存在则返回id为oid的实体, 否则返回自身
        """
        if oid is None or oid == self.entity_ext.entity.id:
            return self._entity_info(self.entity_ext.entity)
        else:
            target = self._simulator_factory.get_simulator_by_id(oid)
            return self._entity_info(target.entity_ext.entity)

    def _set_position(self, other: PosVelAcc) -> None:
        """
        将当前仿真器位置速度设定为其他仿真器位置速度
        :param other: 其他仿真器
        """
        self._entity_ext.entity.lla = other.cPos
        self._entity_ext.entity.velNue = other.cVel

    def _health_available(self, health: bool) -> None:
        """
        设定健康可用性
        :param health: 健康值
        """
        self._entity_ext.entity.healthState = health
        if not self._entity_ext.entity.childrenId:
            return
        cmd_list = [
            self.assemble_command(
                children,
                SimmerCommandType.HEALTH_STATE_CHANGE,
                HealthStateChangeC(HealthStateChangeCmd(health)).to_json()
            )
            for children in self._entity_ext.entity.childrenId
        ]
        self._send_commands(cmd_list)

    def _power_available(self, power: bool) -> None:
        """
        设定电源可用性
        :param power: 电源值
        :return:
        """
        if power:
            self._entity_ext.entity.powerState = power
        if not self._entity_ext.entity.childrenId:
            return
        cmd_list = [
            self.assemble_command(
                children,
                SimmerCommandType.POWER_STATE_CHANGE,
                PowerStateChangeC(PowerStateChangeCmd(power)).to_json()
            )
            for children in self._entity_ext.entity.childrenId
        ]
        self._send_commands(cmd_list)

    def assemble_command(self, executor_id: int, command_type: SimmerCommandType, attributes: str,
                         trigger_type: Optional[int] = None) -> Command:
        """
        组装指令
        :param executor_id: 执行者ID
        :param command_type: 指令类型
        :param attributes: 指令内容
        :param trigger_type: 前序触发者类型
        :return: 指令
        """
        return Command(
            prevTriggerId=self.entity_ext.entity.id,
            prevTriggerTypeId=trigger_type,
            executorId=executor_id,
            commandTypeId=command_type,
            commandAttributes=attributes
        )

    def assemble_event(self, trigger_id: int, trigger_type: SimmerTriggerType, command_list: list[Command]) -> Event:
        """
        组装事件
        :param trigger_id: 触发id
        :param trigger_type: 触发器类型
        :param command_list: 指令列表
        :return:
        """
        return Event(
            prevTriggerId=self.entity_ext.entity.id,
            prevTriggerTypeId=-1,
            triggerId=trigger_id,
            triggerTypeId=trigger_type,
            commandList=command_list
        )

    def leave_parent(self) -> None:
        """
        离开父物体
        :return:
        """
        if self._entity_ext.entity.parentId == -1:
            return
        self._entity_ext.entity.isVisible = True
        self._entity_ext.entity.parentId = -1
        # 向父物体发送离开指令
        cmd_list = [
            self.assemble_command(
                self._entity_ext.entity.parentId,
                SimmerCommandType.RELEASE_CHILDREN,
                ReleaseChildrenC(
                    ReleaseChildrenCmd(
                        self._entity_ext.entity.id
                    )
                ).to_json()
            )
        ]
        self._send_commands(cmd_list)

    def return_parent(self, parent_id: int) -> None:
        """
        回归父物体
        :return:
        """
        self._entity_ext.entity.parentId = parent_id
        self._entity_ext.entity.isVisible = False
        # 给父物体发送指令, 让其回收
        cmd_list = [
            self.assemble_command(
                parent_id,
                SimmerCommandType.RETRIEVE_CHILDREN,
                RetrieveChildrenC(
                    RetrieveChildrenCmd(
                        self._entity_ext.entity.id
                    )
                ).to_json()
            )
        ]
        self._send_commands(cmd_list)

    def _get_targets_within_self_range(self, candidate_simulators: list["ISimulator"], max_range: float) -> list[
        "ISimulator"]:
        """
        获取自身范围内的目标
        :param max_range: 最大距离（米）
        :return: 目标列表
        """
        if max_range is None:
            return []

        detected = []
        self_pos = self._entity_ext.entity.posEcf

        for target in candidate_simulators:
            # 跳过自身
            if target.entity_ext.entity.id == self._entity_ext.entity.id:
                continue

            # 跳过已摧毁或不可见的实体
            if not target.entity_ext.entity.isVisible or target.entity_ext.entity.survivePoints <= 0:
                continue

            target_pos = target.entity_ext.entity.posEcf

            if self._is_geometrically_visible(self_pos, target_pos, max_range):
                detected.append(target)
        return detected

    def handel_detect_info(self, new_detect_info: dict[int, DetectInfo]) -> None:
        """
        处理探测信息, 主要做新数据和旧数据的融合
        :param new_detect_info: 新探测信息
        """
        data_change = False
        target_dict = self._entity_ext.entity.detectInfo
        for key, new_value in new_detect_info.items():
            old_value = target_dict.get(key)
            # 如果不存在 或 新数据时间更新，则更新
            if old_value is None or new_value.time > old_value.time:
                # Tracks are observations at a particular time. Copy the full
                # record so a live entity's mutable position cannot keep
                # updating it after detection or communication has stopped.
                target_dict[key] = deepcopy(new_value)
                data_change = True
        if data_change:
            self._delivered_to.clear()

    def detect_targets(self, candidate_simulators: list["ISimulator"], max_range: float = None,
                       detect_radius: float = None, detect_point: Vector3d = None) -> list["ISimulator"]:
        """
        探测范围内的目标，基于给出的探测点，判断距离是否满足（考虑高度），如果满足则以探测点为圆心，返回在探测半径内的实体(不考虑高度)
        :param candidate_simulators: 候选目标仿真器列表（通常来自工厂的按方获取）
        :param max_range: 最大探测距离（米）
        :param detect_radius: 探测半径（米）
        :param detect_point: 探测点（经纬高）
        :return: 被探测到的仿真器列表
        """
        if max_range is None or detect_radius is None or detect_point is None:
            return []

        detected = []
        sat_pos = self._entity_ext.entity.posEcf
        detect_point_pos = self._convert_lla_to_ecf(detect_point)

        if not self._is_geometrically_visible(sat_pos, detect_point_pos, max_range):
            return []
        for target in candidate_simulators:
            # 跳过自身
            if target.entity_ext.entity.id == self._entity_ext.entity.id:
                continue
            # 跳过已摧毁或不可见的实体
            if not target.entity_ext.entity.isVisible or target.entity_ext.entity.survivePoints <= 0:
                continue
            target_pos = target.entity_ext.entity.posEcf
            if self._is_geometrically_visible(detect_point_pos, target_pos, detect_radius, ignore_z=True):
                detected.append(target)
        return detected

    def _get_ecf_position(self) -> Vector3d:
        """
        获取实体当前的 ECF 坐标
        :return: Vector3d (x, y, z) in meters
        """
        lla = self._entity_ext.entity.lla
        return self._convert_lla_to_ecf(lla)

    @staticmethod
    def _convert_lla_to_ecf(lla) -> Vector3d:
        """
        将经纬高(LLA)转换为地心地固坐标(ECF)
        :param lla: Vector3d (lat, lon, alt)
        :return: Vector3d (x, y, z)
        """
        ecf: UtilsPy.Vector3D = UtilsPy.CoordinateHelper.llaToEcef_py(UtilsPy.Vector3D(lla.x, lla.y, lla.z))
        return Vector3d(ecf.x(), ecf.y(), ecf.z())

    @staticmethod
    def _is_geometrically_visible(a_pos: Vector3d, b_pos: Vector3d, max_range: float, ignore_z: bool = False) -> bool:
        """
        判断目标是否可见（距离约束）
        :return : True:在范围内；False：不在范围内
        """
        dx = a_pos.x - b_pos.x
        dy = a_pos.y - b_pos.y

        # 计算距离的平方，避免耗时的开平方运算
        distance_squared = dx * dx + dy * dy

        # 如果不忽略高度，则加上Z轴的差值平方
        if not ignore_z:
            dz = a_pos.z - b_pos.z
            distance_squared += dz * dz

        # 与最大距离的平方进行比较
        if distance_squared > max_range * max_range:
            return False

        return True

    def data_sync(self) -> None:
        """
        数据同步
        :return:
        """
        if self._last_lla != self._entity_ext.entity.lla:
            cmd_list = [
                self.assemble_command(
                    children,
                    SimmerCommandType.CHILD_SYNC,
                    ChildSyncC(PosVelAcc(self._entity_ext.entity.lla, self._entity_ext.entity.velNue,
                                         Vector3d(0, 0, 0))).to_json()
                )
                for children in self._entity_ext.entity.childrenId
            ]
            self._send_commands(cmd_list)
            self._last_lla = self._entity_ext.entity.lla

    def command_received(self, command: Command) -> None:
        """
        指令接收
        :param command: 指令
        """
        if command.commandTypeId == SimmerCommandType.CHILD_SYNC:
            pos_vel_acc = json.loads(command.commandAttributes)["cmd"]
            self._set_position(pos_vel_acc)
        elif command.commandTypeId == SimmerCommandType.DAMAGE:
            source = self._target_entity_info_or_self(command.prevTriggerId)
            damage_point = DamageC.from_json(command.commandAttributes).cmd.damagePoint
            self._damage(damage_point, source)
        elif command.commandTypeId == SimmerCommandType.DESTROY:
            source = self._target_entity_info_or_self(command.prevTriggerId)
            self._damage(self._entity_ext.entity.survivePoints, source)
        elif command.commandTypeId == SimmerCommandType.HEALTH_STATE_CHANGE:
            health_state = HealthStateChangeC.from_json(command.commandAttributes)["cmd"]["healthState"]
            self._health_available(health_state)
        elif command.commandTypeId == SimmerCommandType.POWER_STATE_CHANGE:
            power_state = PowerStateChangeC.from_json(command.commandAttributes)["cmd"]["powerState"]
            self._power_available(power_state)
        elif command.commandTypeId == SimmerCommandType.RETRIEVE_CHILDREN:
            children_id = RetrieveChildrenC.from_json(command.commandAttributes)["cmd"]["childrenId"]
            self._retrieve_children(children_id)
        elif command.commandTypeId == SimmerCommandType.RELEASE_CHILDREN:
            children_id = ReleaseChildrenC.from_json(command.commandAttributes)["cmd"]["childrenId"]
            self._release_children(children_id)
        elif command.commandTypeId == SimmerCommandType.DETECT_STATUS_UPDATE:
            detect_info = command.commandAttributes
            self.handel_detect_info(detect_info)
        else:
            logger.warning(f"未知指令：{command}, 系统将忽略该指令！")

    def __str__(self) -> str:
        if self.entity_ext and self.entity_ext.entity:
            return f"name={self.entity_ext.entity.nameChn}, type={self.entity_ext.entity.entityType}"
        else:
            return "未正确传入 entity_ext"
