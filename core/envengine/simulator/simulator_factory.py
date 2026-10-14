# -*-coding:utf-8 -*-
import json
import logging
import math
from typing import Optional

from envengine.sdk.base_struct.Message import Command
from envengine.sdk.base_struct.profile.profile import Profile
from envengine.sdk.base_struct.Entity.EntityExt import EntityExt
from .decorator.simulator_decorator import Simulator
from .interfaces.isimulator import ISimulator
from ..common import SimmerCommandType
from ..sdk.base_struct.Basic import Vector3d
from ..sdk.writer import write_event

logger = logging.getLogger(__name__)

class SimulatorFactory:
    """
    仿真器工厂
    负责创建、管理和查询仿真器实例

    Attributes:
        red_sat_max_use_count: 天眼最大使用次数
        red_sat_use_count: 天眼已使用次数
        sat_use_minutes: 天眼使用时间（分钟）
        _satellite_use_end_time: 本次天眼使用结束时间
    """

    def __init__(self, profile: Profile):
        self._profile = profile
        self._simulators: dict[int, ISimulator] = {}  # 按ID索引
        self._simulators_by_type: dict[int, list[ISimulator]] = {}  # 按类型索引
        self._simulators_by_side: dict[int, list[ISimulator]] = {}  # 按作战方索引
        self._command_queue: list[Command] = []  # 存储待处理的指令
        self.sim_step = profile.imagineProfile.simStep  # 引擎仿真步长
        self.sim_time = 0  # 当前仿真时间

        # 初始化所有仿真器
        self._initialize_simulators()

        # 当前仿真轮次
        self.current_round = 0

        # 目标命中关系，记录击中每个目标的飞行器信息
        self.target_hit_relation:dict[int, list] = {}

        # 天眼使用情况
        self.red_sat_use_count = 0
        self._satellite_use_end_time = 0
        if self.profile and self.profile.imagineProfile:
            self.red_sat_max_use_count = max(0, self.profile.imagineProfile.satelliteMaxUseCount)
            self.sat_use_minutes = max(0, self.profile.imagineProfile.satelliteUseMinutes)

    @property
    def profile(self)->Profile:
        """当前想定"""
        return self._profile

    @property
    def command_queue(self):
        return self._command_queue

    def _initialize_simulators(self):
        """初始化所有仿真器"""
        for entity_ext in self._profile.imagineProfile.entityList:
            # 创建仿真器实例（可以根据entityType创建不同类型的仿真器）
            entity_type: int = entity_ext.entity.entityType
            class_name: str = self._get_simulator_class_name_by_entity_type(entity_type)
            try:
                simulator = self._create_simulator(class_name, entity_ext)
                # 注册到工厂
                self._register_simulator(simulator)
            except Exception as e:
                logging.error(f"[仿真器工厂] 创建仿真器 {class_name} 失败: {e}")

        logging.info(
            f"[仿真器工厂] 初始化完成: 共{len(self._simulators)}个实体, "
            f"分别是{[i.entity_ext.entity.nameChn for i in self._simulators.values()]}"
        )

    def _get_simulator_class_name_by_entity_type(self, entity_type: int) -> str:
        """
        根据实体类型获取对应的仿真器类名
        :param entity_type: 实体类型
        :return: 仿真器类名
        """
        environment_profile: Profile.environmentProfile = self._profile.environmentProfile
        dynamic_library_configs: list[Profile.environmentProfile.dynamicLibraryConfigs] = environment_profile.dynamicLibraryConfigs
        for dynamic_library_config in dynamic_library_configs:
            if dynamic_library_config.entityType == entity_type:
                class_name = dynamic_library_config.dynamicLibraryPath.rsplit("/")[-1].rsplit(".", 1)[0].rsplit("lib")[
                    -1]
                return class_name

    def _create_simulator(self, class_name: str, entity_ext: EntityExt) -> ISimulator:
        """
        根据实体类型创建对应的仿真器
        :param entity_ext: 实体扩展信息
        :return: 仿真器实例
        """
        return Simulator.create(class_name, entity_ext, self._send_command_callback, self._send_events_callback,
                                simulator_factory=self)

    def _register_simulator(self, simulator: ISimulator):
        """注册仿真器到索引"""
        id = simulator.entity_ext.entity.id
        entity_type = simulator.entity_ext.entity.entityType
        side_id = simulator.entity_ext.entity.sideId

        # 按ID存储
        self._simulators[id] = simulator

        # 按类型分组
        if entity_type not in self._simulators_by_type:
            self._simulators_by_type[entity_type] = []
        self._simulators_by_type[entity_type].append(simulator)

        # 按作战方分组
        if side_id not in self._simulators_by_side:
            self._simulators_by_side[side_id] = []
        self._simulators_by_side[side_id].append(simulator)

    def _send_command_callback(self, commands: list[Command]) -> None:
        """
        仿真器发送指令的回调，用于将指令加入待处理的指令队列
        :param commands:
        :return:
        """
        if commands:
            self._command_queue.extend(commands)

    def process_commands(self):
        """
        引擎调用： 分发并执行队列中的所有指令
        :return:
        """
        if not self._command_queue:
            return
        for command in self._command_queue:
            target_sim = self.get_simulator_by_id(command.executorId)
            if target_sim:
                if command.commandTypeId == SimmerCommandType.DAMAGE:
                    command = self.process_hit(command)

                    prev_trigger = self.get_simulator_by_id(command.prevTriggerId)
                    if prev_trigger:
                        self.target_hit_relation.setdefault(command.executorId,[]).append({
                            "id":prev_trigger.entity_ext.entity.id,
                            "entity_type":prev_trigger.entity_ext.entity.entityType,
                            "sim_time":prev_trigger.sim_time,
                            "velEcf":prev_trigger.entity_ext.entity.velEcf,
                            "posEcf":prev_trigger.entity_ext.entity.posEcf,
                        })
                target_sim.command_received(command)
                # if command.commandTypeId == SimmerCommandType.DAMAGE:
                #     print(command)
        self._command_queue.clear()

    def process_hit(self, command):
        """
        命中率映射表：后序需要提取成配置文件，新增不同entityType模型不同模型
        :return:
        """

        """
        21000 毁灭者
        21001 清道夫
        21002 清道夫
        
        9400 目标
        9500 暗巢
        9600 哨卫巢
        """

        # 红方毁伤数值表
        damage_point_table = {
            21000:{
                9400:20,
                9600:20,
                9500:20
            },
            21001:{
                9400:5,
                9600:5,
                9500:5
            },
            21002:{
                9400:0,
                9600:0,
                9500:1
            }
        }

        # 红方命中率表
        hit_rate_table = {
            # 毁灭者
            21000: {
                9400: 0.8,
                9600: 0.6,
                9500: 0
            },
            # 清道夫
            21001: {
                9400: 0.8,
                9600: 0.6,
                9500: 0
            },
            # 清道夫
            21002: {
                9400: 0.05,
                9600: 0.05,
                9500: 0.8
            }
        }

        # 伤害来源
        prev_trigger_id = command.prevTriggerId
        prev_simulator = self.get_simulator_by_id(prev_trigger_id)
        if not prev_simulator:
            return command

        prev_simulator_type = prev_simulator.entity_ext.entity.entityType

        # 被击中的对象
        executor_id = command.executorId
        executor_simulator = self.get_simulator_by_id(executor_id)
        if not executor_simulator:
            return command
        executor_simulator_type = executor_simulator.entity_ext.entity.entityType

        # 获取命中率
        if prev_simulator_type not in hit_rate_table:
            return command
        if executor_simulator_type not in hit_rate_table[prev_simulator_type]:
            return command
        hit_rate = hit_rate_table[prev_simulator_type][executor_simulator_type]
        hit_rate = self.recalculate_rate(prev_simulator, executor_simulator, hit_rate)
        # damage_point = json.loads(command.commandAttributes)["cmd"]["damagePoint"]
        damage_point = damage_point_table[prev_simulator_type][executor_simulator_type]
        command.commandAttributes = json.dumps({"cmd": {"damagePoint": damage_point * hit_rate}})
        return command

    def recalculate_rate(self, prev_simulator: ISimulator | None, executor_simulator: ISimulator | None, hit_rate:float):
        """
        命中率调整
        :param prev_simulator:      进攻方
        :param executor_simulator:  被集中目标
        :param hit_rate:            原始命中率
        :return:
        """

        if not executor_simulator or not prev_simulator:
            return hit_rate

        # 使用天眼期间，毁灭者对母巢和阵地的命中率提升至 100%
        if (prev_simulator.entity_ext.entity.entityType == 21000
                and executor_simulator.entity_ext.entity.entityType in [9400, 9600]
                and self.is_using_satellite()):
            return 1

        # 命中率提升条件
        hit_rate = self.hit_increase(prev_simulator, executor_simulator, hit_rate)
        hit_rate = self.hit_decrease(prev_simulator, executor_simulator, hit_rate)

        return hit_rate

    def hit_increase(self, prev_simulator: ISimulator, executor_simulator: ISimulator, hit_rate:float)->float:
        """
        命中率提升

        当任意两枚红方单位在一定时间内命中同一目标，且入射夹角大于30°时，
        命中率最大提升20%，按照时间间隔线性递减，时间间隔0为20%

        :param prev_simulator: 进攻方
        :param executor_simulator: 被击中目标
        :param hit_rate: 原始命中率
        :return: 新的命中率
        """

        if hit_rate == 0:
            return hit_rate

        if prev_simulator.entity_ext.entity.entityType not in [21000, 21001]:
            return hit_rate

        if executor_simulator.entity_ext.entity.entityType != 9400:
            return hit_rate

        if executor_simulator.entity_ext.entity.id not in self.target_hit_relation:
            return hit_rate

        # 时间间隔
        time_interval = self.profile.imagineProfile.missileRateIncreaseTimeIntervalMinutes *60*1000
        min_angle = self.profile.imagineProfile.missileRateIncreaseMinAngle # 最小入射夹角（度）
        max_increase = self.profile.imagineProfile.missileRateIncreaseMaxValue # 最大提升比例

        last_hit_time = 0
        for item in self.target_hit_relation[executor_simulator.entity_ext.entity.id]:
            if (item["entity_type"] in [21000, 21001] and
                    prev_simulator.sim_time - item["sim_time"] <= time_interval and
                    self.get_vector_angle(prev_simulator.entity_ext.entity.velEcf, item["velEcf"]) > min_angle):
                # 满足提升条件，记录时间间隔最小的一次命中
                last_hit_time = max(last_hit_time, item["sim_time"])

        if last_hit_time == 0:
            return hit_rate

        # 计算命中率提升比例
        increase_rate = max_increase * (1-(prev_simulator.sim_time - last_hit_time) / time_interval)
        return hit_rate * (1+increase_rate)

    def hit_decrease(self, prev_simulator: ISimulator, executor_simulator: ISimulator, hit_rate:float)->float:
        """
        命中率降低

        当任意两枚毁灭者/清道夫在一定时间内夹击同一目标，且入射夹角小于10°时，
        命中率最大降低40%，按照时间间隔线性递增，时间间隔0为40%

        :param prev_simulator: 进攻方
        :param executor_simulator: 被击中目标
        :param hit_rate: 原始命中率
        :return: 新的命中率
        """

        if hit_rate == 0:
            return hit_rate

        if prev_simulator.entity_ext.entity.entityType not in [21000, 21001]:
            return hit_rate

        if executor_simulator.entity_ext.entity.entityType != 9400:
            return hit_rate

        if executor_simulator.entity_ext.entity.id not in self.target_hit_relation:
            return hit_rate

        # 时间间隔
        time_interval = self.profile.imagineProfile.missileRateDecreaseTimeIntervalMinutes *60*1000
        max_angle = self.profile.imagineProfile.missileRateDecreaseMaxAngle # 最大入射夹角（度）
        max_decrease = self.profile.imagineProfile.missileRateDecreaseMaxValue

        last_hit_time = 0
        for item in self.target_hit_relation[executor_simulator.entity_ext.entity.id]:
            if (item["entity_type"] in [21000, 21001] and
                    prev_simulator.sim_time - item["sim_time"] <= time_interval and
                    self.get_vector_angle(prev_simulator.entity_ext.entity.velEcf, executor_simulator.entity_ext.entity.velEcf) < max_angle):
                # 满足条件，记录时间间隔最小的一次命中
                last_hit_time = max(last_hit_time, item["sim_time"])

        if last_hit_time == 0:
            return hit_rate

        # 计算命中率提升比例
        decrease_rate = max_decrease * (1-(prev_simulator.sim_time - last_hit_time) / time_interval)
        return hit_rate * (1-decrease_rate)

    @staticmethod
    def get_vector_angle(vector1: Vector3d, vector2: Vector3d) -> float:
        """
        计算两个向量的夹角
        :param vector1: 向量1
        :param vector2: 向量2
        :return: 夹角（度）
        """
        # 点积
        dot = vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z
        # 模长
        norm1 = math.sqrt(vector1.x ** 2 + vector1.y ** 2 + vector1.z ** 2)
        norm2 = math.sqrt(vector2.x ** 2 + vector2.y ** 2 + vector2.z ** 2)
        # 零向量时夹角视为 0 度
        if norm1 == 0 or norm2 == 0:
            return 0.0
        # 余弦值（截断到 [-1, 1] 防止浮点误差）
        cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        return math.degrees(math.acos(cos_theta))

    def process_ai_commands(self, ai_commands: list[Command]):
        """
        引擎调用： 收集并分发执行AI的所有指令
        :return:
        """
        for ai_command in ai_commands:
            target_sim = self.get_simulator_by_id(ai_command.executorId)
            if not target_sim:
                continue

            if ai_command.commandTypeId == SimmerCommandType.EXECUTE_SATELLITE_DETECTION:
                # 使用天眼是一个全局计数、全局视角的事件
                self._process_missile_use_satellite(ai_command, target_sim)
            else:
                target_sim.command_received(ai_command)

    def _process_missile_use_satellite(self, command: Command, target_simulator: ISimulator):
        """
        处理使用天眼的指令
        :param command:
        :return:
        """

        if self.red_sat_use_count >= self.red_sat_max_use_count:
            logger.warning(f"使用天眼次数已经超出最大使用次数，来自【{target_simulator.entity_ext.entity.nameChn}】的天眼使用指令无效")
            return

        self.red_sat_use_count += 1
        logger.info(f"entity_id:{target_simulator.entity_ext.entity.id}, name:{target_simulator.entity_ext.entity.nameChn} 使用天眼，剩余次数：{self.red_sat_max_use_count - self.red_sat_use_count}")

        # 每次指令可以使用天眼 Y 分钟
        self._satellite_use_end_time = self.sim_time + self.sat_use_minutes * 60 * 1000  # Y分钟 * 60秒 * 1000毫秒

    def is_using_satellite(self) -> bool:
        """
        是否正在使用天眼
        :return:
        """
        return self.sim_time < self._satellite_use_end_time

    def _send_events_callback(self, event: dict):
        """
        即时发送
        :param event: 事件
        :return:
        """
        write_event(event, str(self.current_round) + ".json")
        # logging.info(f"[仿真器工厂] 事件显示: {event}")

    def get_simulator_by_id(self, id: int) -> Optional[ISimulator]:
        """
        按ID获取仿真器
        :param id: 实体ID
        :return: 仿真器实例，不存在返回None
        """
        return self._simulators.get(id)

    def get_simulators_by_type(self, entity_type: int) -> list[ISimulator]:
        """
        按类型获取仿真器列表
        :param entity_type: 实体类型
        :return: 仿真器列表
        """
        return self._simulators_by_type.get(entity_type, [])

    def get_lived_simulators_by_type(self, entity_type: int) -> list[ISimulator]:
        """
        按类型获取有生命的仿真器列表
        :param entity_type: 实体类型
        :return: 仿真器列表
        """
        simulators: list[ISimulator] = self._simulators_by_type.get(entity_type, [])

        return [simulator for simulator in simulators if simulator.entity_ext.entity.survivePoints > 0]

    def get_simulators_by_side(self, side_id: int) -> list[ISimulator]:
        """
        按作战方获取仿真器列表
        :param side_id: 作战方ID
        :return: 仿真器列表
        """
        return self._simulators_by_side.get(side_id, [])

    def get_all_simulators(self) -> list[ISimulator]:
        """获取所有仿真器"""
        return list(self._simulators.values())

    def get_all_lived_simulators(self) -> list[ISimulator]:
        """获取所有有生命的仿真器"""
        simulators: list[ISimulator] = list(self._simulators.values())
        return [simulator for simulator in simulators if simulator.entity_ext.entity.survivePoints > 0]

    def get_simulator_count(self) -> int:
        """获取仿真器总数"""
        return len(self._simulators)

    def get_profile_entity_count_by_type(self, entity_type: int):
        """获取想定中的某种类型实体个数"""
        return sum(1 for e in self.profile.imagineProfile.entityList if e.entity.entityType == entity_type)

    def update_sim_time(self, sim_time:float):
        self.sim_time = sim_time
        for simulator in self._simulators.values():
            simulator.sim_time = sim_time

    def reset_all(self):
        """重置所有仿真器"""

        # 必须先重置所有仿真器，再统一进行初始化
        for simulator in self._simulators.values():
            simulator.reset()

        for simulator in self._simulators.values():
            simulator.init_model()

        self._command_queue.clear()
        self.current_round += 1
        self.target_hit_relation.clear()
        self._satellite_use_end_time = 0
        self.red_sat_use_count = 0
        logging.info("[仿真器工厂] 所有仿真器已重置")

    def remove_simulator(self, id: int) -> bool:
        """
        移除仿真器
        :param id: 实体ID
        :return: 是否成功移除
        """
        if id not in self._simulators:
            return False

        simulator = self._simulators.pop(id)

        # 从类型索引中移除
        if simulator.entity_ext.entity.entityType in self._simulators_by_type:
            self._simulators_by_type[simulator.entity_ext.entity.entityType] = [
                s for s in self._simulators_by_type[simulator.entity_ext.entity.entityType]
                if s.entity_ext.entity.id != id
            ]

        # 从作战方索引中移除
        if simulator.entity_ext.entity.sideId in self._simulators_by_side:
            self._simulators_by_side[simulator.entity_ext.entity.sideId] = [
                s for s in self._simulators_by_side[simulator.entity_ext.entity.sideId]
                if s.entity_ext.entity.id != id
            ]

        return True

    def modify_simulator_position(self, id: int, lla: dict) -> bool:
        """
        修改仿真器位置
        :param id: 模型id
        :param lla: 新的模型位置
        :return:
        """
        if id not in self._simulators:
            return False
        self._simulators[id].set_lla(Vector3d(lla["x"], lla["y"], lla["z"]))

    def modify_simulator_speed(self, id: int, speed: float) -> bool:
        """
        修改仿真器速度
        :param id: 模型id
        :param speed: 新的速度
        :return:
        """
        if id not in self._simulators:
            return False
        self._simulators[id].set_speed(speed)
