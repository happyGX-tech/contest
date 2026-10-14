# -*-coding:utf-8 -*-
import heapq
import logging
import time
from typing import Callable, Optional, Any

import fastdisjointset

from envengine.common import SimmerCommandType
from envengine.sdk.base_struct.Basic import Vector3d, DetectInfo
from envengine.sdk.base_struct.Message import Command
from envengine.sdk.base_struct.profile.profile import Profile
from envengine.simulator.simulator_factory import SimulatorFactory
from envengine.simulator.interfaces import ISimulator

logger = logging.getLogger(__name__)


MISSILE_COMMUNICATION_RANGE_M = {
    21000: 50_000.0,
    21001: 20_000.0,
    21002: 10_000.0,
}


class Engine:
    """
    Env引擎 - 驱动仿真的核心引擎
    """

    def __init__(self, profile: Profile, speed_multiplier: float = 1):
        """
        初始化Env引擎
        :param profile: 想定配置
        """
        self.profile = profile
        self.simulator_factory = SimulatorFactory(profile)
        self.current_step: int = 0
        self.speed_multiplier = speed_multiplier
        self.sim_step = profile.imagineProfile.simStep
        self.sim_end_time = profile.imagineProfile.simEndLogicTime
        self.sim_time = profile.imagineProfile.simTime
        # 计算总步数
        self.total_steps = int(self.sim_end_time // self.sim_step) if self.sim_step > 0 else 0
        # 计算每步需要的真实时间（秒）
        self.step_duration = (self.sim_step / 1000.0) / speed_multiplier if speed_multiplier > 0 else 0

    def step(self, ai_commands: list[Command] = None) -> None:
        """
        仿真推进一步
        :param ai_commands: AI输入的指令 {entity_id: command}
        :return: 当前态势
        """
        if ai_commands is None:
            ai_commands = []

        # 处理AI输入的指令
        self.simulator_factory.update_sim_time(self.sim_time)
        self.simulator_factory.process_ai_commands(ai_commands)

        start_time = time.time()  # 当前步开始时间

        # ===== 基于事件驱动的交替更新 =====
        event_heap = []  # 空队列
        for simulator in self.simulator_factory.get_all_lived_simulators():
            # 获取步长
            internal_step = simulator.simulator_sim_step or self.sim_step

            # 把每个仿真器的第一次更新事件放入队列
            # (时间, 唯一ID, 仿真器对象, 步长)
            # 用id()确保即使时间相同也能区分不同仿真器
            heapq.heappush(event_heap, (self.sim_time, id(simulator.entity_ext.entity), simulator, internal_step))

        # 结束时间
        end_time = self.sim_time + self.sim_step

        while event_heap:  # 队列不为空就继续
            # 取出队列中时间最早的事件
            event_time, _, simulator, step = heapq.heappop(event_heap)

            # 如果这个事件的时间已经超过了结束时间，停止
            if event_time >= end_time:
                break

            # 更新仿真器到事件时间
            simulator.sim_time = event_time
            simulator.update()

            # 计算下次更新时间
            next_time = event_time + step

            # 如果下次更新时间还没到结束时间，放回队列
            if next_time < end_time:
                heapq.heappush(event_heap, (next_time, id(simulator), simulator, step))

        # 弹间探测共享
        self.share_detect_between_missiles()

        # 处理指令
        self.simulator_factory.process_commands()

        self.current_step += 1
        self.sim_time += self.sim_step

        # 控制倍速
        if self.speed_multiplier > 0:
            elapsed = time.time() - start_time  # 计算当前步已经用了多久
            sleep_time = self.step_duration - elapsed  # 应该用多久 - 已经用了多久 = 还需要等多久
            if sleep_time > 0:  # 如果任务执行得快，就睡一会儿补齐时间
                time.sleep(sleep_time)

    def run(self, max_steps: int = None, step_callback: Callable[[int, dict], dict] = None) -> None:
        """
        运行仿真
        :param max_steps: 最大步数（None则使用想定配置）
        :param step_callback: 每步回调函数 (step, situation) -> ai_commands
        :return: 最终态势
        """
        if max_steps is None:
            max_steps = self.total_steps

        logger.info(f"[Env引擎] 开始仿真，总步数: {max_steps}")

        for step in range(max_steps):
            # 如果有回调，通过回调获取AI指令
            if step_callback:
                current_situation = self._get_current_situation()
                ai_commands = step_callback(step, current_situation)
            else:
                ai_commands = {}

            # 执行步进
            self.step(ai_commands)

            # 打印进度
            if (step + 1) % 10 == 0 or step == max_steps - 1:
                logger.info(f"[Env引擎] 仿真进度: {step + 1}/{max_steps}")

        logger.info("[Env引擎] 仿真结束")

    def _get_current_situation(self) -> dict:
        """获取当前态势摘要"""
        return {
            "step": self.current_step,
            "entity_count": self.simulator_factory.get_simulator_count(),
            "timestamp": self.current_step * self.profile.imagineProfile.simStep / 1000.0
        }

    def reset(self):
        """重置仿真到初始状态"""
        self.current_step = 0
        self.simulator_factory.reset_all()
        self.sim_step = self.profile.imagineProfile.simStep
        self.sim_end_time = self.profile.imagineProfile.simEndLogicTime
        self.sim_time = self.profile.imagineProfile.simTime
        # 计算总步数
        self.total_steps = int(self.sim_end_time // self.sim_step) if self.sim_step > 0 else 0
        # 计算每步需要的真实时间（秒）
        self.step_duration = (self.sim_step / 1000.0) / self.speed_multiplier if self.speed_multiplier > 0 else 0
        logger.info("[Env引擎] 仿真已重置")

    # 委托方法 - 方便直接访问工厂功能
    def get_simulator_by_id(self, entity_id: int) -> Optional[ISimulator]:
        """按ID获取仿真器"""
        return self.simulator_factory.get_simulator_by_id(entity_id)

    def get_simulators_by_type(self, entity_type: int) -> list[ISimulator]:
        """按类型获取仿真器"""
        return self.simulator_factory.get_simulators_by_type(entity_type)

    def get_simulators_by_side(self, side_id: int) -> list[ISimulator]:
        """按作战方获取仿真器"""
        return self.simulator_factory.get_simulators_by_side(side_id)

    ########################################## 弹间探测共享 ##################################################
    def share_detect_between_missiles(self):
        """弹间探测共享"""
        groups: Any = self.calculate_missile_clusters()
        fuse_result: dict[int, dict[int, DetectInfo]] = self.fuse_cluster_detection(groups)
        self.distribute_cluster_result(groups, fuse_result)

    # 1. 对所有弹进行弹群计算（并查集）
    def calculate_missile_clusters(self) -> Any:
        """使用并查集计算导弹分群"""
        # 获取所有弹
        hf_list: list[ISimulator] = self.simulator_factory.get_lived_simulators_by_type(21000)
        mf_list: list[ISimulator] = self.simulator_factory.get_lived_simulators_by_type(21001)
        lf_list: list[ISimulator] = self.simulator_factory.get_lived_simulators_by_type(21002)

        allf_list: list[ISimulator] = hf_list + mf_list + lf_list
        ds = fastdisjointset.DisjointSet()
        for missile in allf_list:
            missile_id = missile.entity_ext.entity.id
            ds.union(missile_id, missile_id)
        for i in range(len(allf_list)):
            for j in range(i + 1, len(allf_list)):
                # 获取两个弹
                f1: ISimulator = allf_list[i]
                f2: ISimulator = allf_list[j]
                # 获取两个弹的坐标
                f1_pos: Vector3d = f1.entity_ext.entity.posEcf
                f2_pos: Vector3d = f2.entity_ext.entity.posEcf
                # A link is available when either platform can reach the other.
                # entityType identifies the high/medium/low platform class;
                # typeId is only a scenario-specific model instance identifier.
                f1_type = f1.entity_ext.entity.entityType
                f2_type = f2.entity_ext.entity.entityType
                distance_limit = max(
                    MISSILE_COMMUNICATION_RANGE_M[f1_type],
                    MISSILE_COMMUNICATION_RANGE_M[f2_type],
                )
                if self.is_geometrically_visible(f1_pos, f2_pos, distance_limit):
                    ds.union(f1.entity_ext.entity.id, f2.entity_ext.entity.id)
        groups = ds.sets()
        # print(f"群的数量：{len(groups)}")
        # for idx, group in enumerate(groups):
        #     print(f"  群{idx + 1}: {sorted(group)} (共{len(group)}个飞行器)")
        # print()
        # 为每个弹赋值通信范围内实体id信息
        for idx, group in enumerate(groups):
            for i in group:
                self.simulator_factory.get_simulator_by_id(i).entity_ext.entity.commRangeInfo = list(group)
        return groups

    # 2. 计算弹群内的探测结果（融合）
    def fuse_cluster_detection(self, groups: Any) -> dict[int, dict[int, DetectInfo]]:
        """融合每个弹群内的探测数据"""
        fuse_result: dict[int, dict[int, DetectInfo]] = {}

        # 红方天眼探测到的东西全局共享
        sta_detect_info:dict[int, DetectInfo] = {}
        sta_sims = self.simulator_factory.get_simulators_by_type(9202)
        for sta in sta_sims:
            if sta.entity_ext.entity.sideId == 0 and sta.entity_ext.entity.detectInfo:
                sta_detect_info.update(sta.entity_ext.entity.detectInfo)

        for idx, group in enumerate(groups):
            if idx not in fuse_result:
                fuse_result[idx] = {}
            for i in group:
                # 获取弹
                f: ISimulator | None = self.simulator_factory.get_simulator_by_id(i)
                if not f:
                    continue

                # 获取弹的探测信息
                detect_info: dict[int, DetectInfo] = f.entity_ext.entity.detectInfo
                detect_info.update(sta_detect_info)
                for key, new_value in detect_info.items():
                    old_value = fuse_result.get(idx).get(key)
                    # 如果不存在 或 新数据时间更新，则更新
                    if old_value is None or new_value.time > old_value.time:
                        fuse_result[idx][key] = new_value
        return fuse_result

    # 3. 分发探测结果到单个弹
    def distribute_cluster_result(self, groups: Any, fuse_result: dict[int, dict[int, DetectInfo]]):
        """将弹群融合结果分发给群内各导弹"""
        for idx, group in enumerate(groups):
            # 构建探测信息指令
            command_list: list[Command] = [Command(
                executorId=i,
                commandTypeId=SimmerCommandType.DETECT_STATUS_UPDATE,
                commandAttributes=fuse_result[idx]
            ) for i in group]
            if command_list:
                # 指令转发
                self.simulator_factory.command_queue.extend(command_list)

    def is_geometrically_visible(self, a_pos: Vector3d, b_pos: Vector3d, max_range: float,
                                 ignore_z: bool = False) -> bool:
        """
        判断目标是否可见（距离约束）
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
