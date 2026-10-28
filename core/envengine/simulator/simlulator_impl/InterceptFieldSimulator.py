from copy import deepcopy
from typing import Callable

from envengine.sdk.Util import UtilsPy
from envengine.sdk.base_struct.Basic import Vector3d
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command
from envengine.simulator.decorator import Simulator
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.simulator_factory import SimulatorFactory
from blue_defense.base import ThreatTrack, Vector3


@Simulator.register("InterceptFieldS")
class InterceptFieldSimulator(ISimulator):
    """
    9600 哨卫巢
    """

    # 速度
    CRUISE_SPEED = 12.0

    def __init__(self, entity_ext: EntityExt,
                 send_commands: Callable[[list[Command]], None],
                 send_events: Callable[[dict], None],
                 simulator_factory: SimulatorFactory = None):
        super().__init__(entity_ext, send_commands, send_events, simulator_factory)
        self.entity_ext.entity.stage = 1
        self._attacker: ThreatTrack | None = None

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
        有来袭目标时垂直于其速度方向躲避，无目标时返回
        """

        if self.entity_ext.entity.survivePoints <= 0:
            return

        # 降低探测频率
        if self.sim_time % 1000 == 0:
            self.detect_attacker()

        self._move()

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
        self.entity_ext.entity.posEcf = self._get_ecf_position()
        self._attacker = None

    def init_model(self) -> None:
        pass

    # ====================== 运动逻辑 ======================
    def detect_attacker(self):
        """依据信息库中距离哨卫巢20公里内的记录选择避让目标。"""
        own_pos = self.entity_ext.entity.posEcf
        self._attacker = self._simulator_factory.blue_information.nearest_track(
            Vector3(own_pos.x, own_pos.y, own_pos.z), 20_000, self.sim_time,
        )

    def _elapsed_seconds(self) -> float:
        """
        步长
        """

        if self.simulator_sim_step > 0:
            return self.simulator_sim_step / 1000.0
        elif self._simulator_factory is not None and self._simulator_factory.sim_step > 0:
            return self._simulator_factory.sim_step / 1000.0
        else:
            return 50 / 1000.0

    def _move(self) -> None:
        """
        推进一个运动步：先计算本步速度，再按速度推进位置
        """

        dt = self._elapsed_seconds()

        if dt <= 0:
            return

        # 计算速度
        self._calculate_velocity(dt)

        # 移动
        self._apply_velocity(dt)

    def _calculate_velocity(self, dt:float):
        """
        计算移动速度（方向和大小）
        1、如果有来袭单位，则躲避该单位
        2、如果没有来袭单位，
            a）如果已经离开初始位置，则回到初始位置
            b）如果在初始位置，则不再移动
        :return:
        """

        velocity_unit = None

        # 计算速度方向
        if self._attacker is not None:
            velocity_unit = self._perpendicular_velocity()

        if velocity_unit is None:
            velocity_unit = self._home_velocity(dt)

        if velocity_unit is None:
            self._set_velocity(Vector3d(0,0,0))
        else:
            self._set_velocity(velocity_unit * self.CRUISE_SPEED)

    def _perpendicular_velocity(self) -> Vector3d | None:
        """
        根据来袭目标的速度垂直方向躲避
        :param dt: 步长（秒）
        :return: 本步速度（米/秒），零向量表示保持阵位
        """

        observed_velocity = self._attacker.vel_ecf
        attacker_vel = Vector3d(observed_velocity.x, observed_velocity.y, observed_velocity.z)
        if attacker_vel.length <= 1e-6:
            # 目标速度为零，无法定义垂直方向，转为返航
            return None

        own_pos = self.entity_ext.entity.posEcf
        vel_unit = attacker_vel.normalized()

        # 天向与目标速度的叉乘方向：同时垂直于两者，因此严格垂直于目标速度且在水平面内
        up_unit = own_pos.normalized()
        direction = up_unit.cross(vel_unit)
        if direction.length <= 1e-6:
            return None

        direction = direction.normalized()

        return direction

    def _home_velocity(self, dt:float) -> Vector3d | None:
        """
        返航速度（米/秒）：指向初始位置
        :param dt: 步长（秒）
        """

        home_pos = self._initial_entity_ext.entity.posEcf
        own_pos = self.entity_ext.entity.posEcf
        to_home = home_pos - own_pos

        # 判断是否已经到家，考虑到仿真精度，需要计算在一步之内运动的长度，如果距离初始位置小于这个长度，则说明已经回到初始位置
        ds = dt * self.CRUISE_SPEED
        if to_home.length <= ds:
            return None

        return to_home.normalized()

    def _apply_velocity(self, dt: float) -> None:
        """
        按当前速度和时间间隔推进位置，同步 lla / posEcf
        :param dt: 步长（秒）
        """

        entity = self.entity_ext.entity
        if entity.velEcf.length <= 0.0:
            return

        new_pos = entity.posEcf + entity.velEcf * dt
        lla = UtilsPy.CoordinateHelper.ecefToLla_py(UtilsPy.Vector3D(new_pos.x, new_pos.y, new_pos.z))

        entity.lla.x, entity.lla.y = lla.x(), lla.y()
        entity.lla.z = self._initial_entity_ext.entity.lla.z  # 保持在初始高度
        entity.posEcf = self._get_ecf_position()

        self._move_children()

    def _set_velocity(self, velocity: Vector3d) -> None:
        """
        设置速度（ECF，米/秒）
        """
        entity = self.entity_ext.entity
        entity.velEcf.x, entity.velEcf.y, entity.velEcf.z = velocity.x, velocity.y, velocity.z

    def _move_children(self):
        """
        更新子物体的位置
        :return:
        """
        if not self.entity_ext.entity.childrenId or self._simulator_factory is None:
            return

        for childId in self.entity_ext.entity.childrenId:
            child = self._simulator_factory.get_simulator_by_id(childId)
            if child is None:
                continue
            child.set_lla(deepcopy(self.entity_ext.entity.lla))
