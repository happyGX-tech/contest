# -*-coding:utf-8 -*-

import logging
from typing import Optional

import numpy as np

from envengine.agent_manager import AgentManager
from envengine.agent_manager.actions.deploy_action import SetLLA, DeployCompleted
from envengine.engine import Engine
from envengine.environment.command_adapter import CommandAdapter
from envengine.environment.command_converter import CommandConverter
from envengine.render import Renderer
from envengine.sdk.base_struct.profile.profile import Profile
from envengine.sdk.writer import write_state, write_ai_action, write_config, write_immediately
from envengine.simulator.util import SpeedDistributor
from user_agents.base_agent import AgentType
from envengine.common import SimmerCommandType, Vector3d

logger = logging.getLogger(__name__)


class TrainingEnv:
    """
    训练环境 - 封装Engine，提供标准RL接口
    支持多智能体隔离观测
    """

    def __init__(self, profile: Profile, render_mode: Optional[str] = None):
        self.engine = Engine(profile, 0)
        self.current_round = 0
        self.current_step = 0
        self.max_steps = self.engine.total_steps
        self.render_mode = render_mode
        self.renderer = None
        if render_mode == "human":
            map_area = profile.imagineProfile.mapArea
            self.renderer = Renderer(map_area.lonMin, map_area.lonMax, map_area.latMin, map_area.latMax)
            self.renderer.start()

        # 初始化智能体管理器
        self.agent_manager = AgentManager()
        # 初始化线性分配器
        polygon = profile.imagineProfile.redArea.coordinates[0]
        lons = [point[0] for point in polygon]
        lats = [point[1] for point in polygon]
        for p in profile.imagineProfile.redArea.coordinatesHM[0]:
            lons.append(p[0])
            lats.append(p[1])
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        self.speed_distributor = SpeedDistributor(min_lon=min_lon, max_lon=max_lon,
                                                  min_lat=min_lat, max_lat=max_lat)

        # 初始化上一帧状态缓存（用于奖励计算）
        self._last_observation = None
        self._last_actions = None
        self._shared_learning_policy = None

    def set_shared_learning_policy(self, policy) -> None:
        """Register a training-only centralized critic lifecycle hook.

        Platform agents still receive only isolated observations.  The shared
        policy is supplied the full state solely when explicitly training.
        """

        self._shared_learning_policy = policy

    def reset(self) -> dict:
        """
        重置环境
        :return: 初始观测
        """
        if self._shared_learning_policy is not None:
            self._shared_learning_policy.reset_episode()

        # 重置引擎
        self.engine.reset()
        self.current_step = 0
        self.current_round += 1

        # 重置所有智能体
        self.agent_manager.reset_all()

        # 获取初始态势作为观测
        observation = self._get_observation()

        # 将想定配置数据写入文件
        imagine_profile = self.engine.profile.imagineProfile
        red_area = imagine_profile.redArea
        map_area = imagine_profile.mapArea
        write_config({"mapArea": map_area.__dict__, "redArea": red_area.__dict__}, str(self.current_round) + ".json")

        # 将态势数据写入文件
        write_state(observation, str(self.current_round) + ".json")

        # 重置上一帧状态缓存
        self._last_observation = observation.copy()
        self._last_actions = None

        logger.info(f"[训练环境] 环境已重置")
        if self.render_mode == 'human' and self.renderer:
            self.renderer.update_data(observation["entities"])
            # 更新红蓝方区域
            self.renderer.update_area(self.engine.profile.imagineProfile.redArea.coordinates, self.engine.profile.imagineProfile.redArea.coordinatesHM)
        return observation

    def red_model_deploy(self):
        """
        红方模型部署
        :return:
        """

        while True:
            np_action: dict = self._generate_actions_from_deploy_agent()
            # 转换np.array动作到结构体动作
            action = []
            for row in np_action:
                action_type = int(row[0])

                if action_type == 0:  # 移动动作
                    action.append(
                        SetLLA(
                            executor_id=int(row[1]),
                            lla=Vector3d(
                                x=float(row[2]),
                                y=float(row[3]),
                                z=float(row[4])
                            )
                        ).to_dict()
                    )
                elif action_type == 1:  # 完成动作
                    action.append(DeployCompleted().to_dict())
            # 判断是否部署结束
            deploy_completed_flag = False
            for action_item in action:
                if action_item["commandType_id"] == SimmerCommandType.DEPLOY_COMPLETED:
                    deploy_completed_flag = True
                if not deploy_completed_flag:
                    # 将AI动作数据写入文件
                    write_ai_action(action_item, str(self.current_round) + ".json")

                    # 判断部署位置是否在区域内
                    # in_region = self.speed_distributor.is_in_region(action_item["lla"]["x"], action_item["lla"]["y"])
                    # if not in_region:
                    #     logger.error(f"[训练环境] 部署位置不在区域内")
                    #     continue

                    # 更改模型位置
                    self.engine.simulator_factory.modify_simulator_position(action_item["executor_id"],
                                                                            action_item["lla"])
                    # 更改模型初始速度
                    speed = self.speed_distributor.get_speed(action_item["lla"]["x"])
                    self.engine.simulator_factory.modify_simulator_speed(action_item["executor_id"], speed)
                    # 渲染
                    if self.render_mode == 'human' and self.renderer:
                        # 获取观测
                        observation = self._get_observation()
                        self.renderer.update_data(observation["entities"])
            if deploy_completed_flag:
                break

    def step(self) -> tuple[dict, dict, bool, dict]:
        """
        执行一步
        """

        action: list[dict] = self._generate_actions_from_agents()
        # 记录数据：将AI动作数据写入文件
        for i in action:
            write_ai_action(i, str(self.current_round) + ".json")

        # 对AI指令到引擎可接收指令进行适配
        action_adapter = CommandAdapter.common_adapter(action)
        # 执行仿真步进
        self.engine.step(action_adapter)
        self.current_step += 1

        # 获取观测
        observation = self._get_observation()

        # 记录数据：将态势数据写入文件
        write_state(observation, str(self.current_round) + ".json")

        # 计算奖励
        rewards = self._compute_reward()

        # 判断是否结束
        done = self.get_is_done()

        # 额外信息
        info = {
            "step": self.current_step,
            "max_steps": self.max_steps,
            "done": done,
        }

        if self._shared_learning_policy is not None:
            self._shared_learning_policy.end_environment_step(observation)

        # 通知每个智能体它们的奖励，并记录历史
        for agent in self.agent_manager.get_all_agents():
            agent_id = agent.agent_id
            reward = rewards.get(agent_id, 0.0)

            # 为该智能体提取隔离观测
            isolated_obs = self.agent_manager.extract_observation_for_agent(
                observation, agent_id
            )

            # 获取该智能体刚才执行的动作
            entity_id = agent.entity_id
            agent_action = None
            for cmd in action:
                if isinstance(cmd, dict) and "executor_id" in cmd and cmd["executor_id"] == entity_id:
                    agent_action = cmd
                    break
            # 记录这一步的经验
            agent.record_step(
                observation=isolated_obs,
                action=agent_action,
                reward=reward,
                info=info
            )

        if self._shared_learning_policy is not None:
            self._shared_learning_policy.finish_environment_step()

        if self.render_mode == 'human' and self.renderer:
            self.renderer.update_data(observation["entities"])

        # 更新上一帧状态
        self._last_observation = observation.copy()
        self._last_actions = action.copy() if action else None
        return observation, rewards, done, info

    def get_is_done(self) -> bool:
        if self.current_step >= self.max_steps:
            return True

        # 所有 红方弹、自爆虫群 结束之后，仿真结束
        for h in self.engine.simulator_factory.get_simulators_by_type(21000):
            if h.entity_ext.entity.isVisible:
                return False

        for l in self.engine.simulator_factory.get_simulators_by_type(21001):
            if l.entity_ext.entity.isVisible:
                return False

        for m in self.engine.simulator_factory.get_simulators_by_type(21002):
            if m.entity_ext.entity.isVisible:
                return False

        for la in self.engine.simulator_factory.get_simulators_by_type(24000):
            if la.entity_ext.entity.isVisible:
                return False

        return True

    def _generate_actions_from_agents(self) -> list[np.array]:
        """
        使用智能体工厂生成所有智能体的动作
        :return: actions
        """
        # 获取当前完整观测
        full_observation = self._get_observation()
        if self._shared_learning_policy is not None:
            self._shared_learning_policy.begin_environment_step(full_observation)

        # 为每个智能体(前提是模型存活)提取隔离观测并收集动作
        agent_observations = {}
        for agent in self.agent_manager.get_all_agents():
            isolated_obs = self.agent_manager.extract_observation_for_agent(
                full_observation,
                agent.agent_id
            )

            if isolated_obs:
                isolated_obs["is_using_satellite"] = self.engine.simulator_factory.is_using_satellite()
                agent_observations[agent.agent_id] = isolated_obs

        commanders = {
            agent.commander
            for agent in self.agent_manager.get_all_agents()
            if getattr(agent, "commander", None) is not None
        }
        for commander in commanders:
            begin_step = getattr(commander, "begin_step", None)
            if begin_step is not None:
                begin_step(tuple(agent_observations.values()))

        # 收集所有智能体的动作
        actions = self.agent_manager.collect_actions_from_agents(agent_observations)
        # 转换所有智能体动作到结构体类型
        return CommandConverter.common_converter(actions)

    def _generate_actions_from_deploy_agent(self) -> np.array:
        """
        使用智能体工厂生成部署智能体的动作
        :return: actions
        """
        # 为部署智能体提取隔离观测并收集动作
        agent_observations = {}

        for agent in self.agent_manager.get_agent_by_type(AgentType.DEPLOY):
            agent_observations[agent.agent_id] = self._get_red_observation()

        # 收集部署智能体的动作
        action = self.agent_manager.collect_actions_from_deploy_agent(agent_observations)

        return action

    def _get_init_ship_observation(self) -> dict:
        """获取初始化时，给红方 AI 的蓝方信息
        仅传递，蓝方高中价值单位（即：母巢、哨卫巢）

        9400：母巢
        9500：暗巢
        9600：哨卫巢
        """

        simulators = self.engine.simulator_factory.get_all_simulators()

        observation = {
            "step": self.current_step,
            "entities": {}
        }

        for simulator in simulators:
            entity_data = simulator.entity_ext.entity
            if entity_data.entityType not in (9400, 9600):
                continue

            observation["entities"][entity_data.id] = {
                "nameChn": entity_data.nameChn,
                "position": {
                    "lon": entity_data.lla.x,
                    "lat": entity_data.lla.y,
                    "alt": entity_data.lla.z
                },
                "health": entity_data.survivePoints,
                "isVisible": entity_data.isVisible,
                "type": entity_data.entityType,
                "side": entity_data.sideId,
                "detectInfo": entity_data.detectInfo,
                "commRangeInfo": entity_data.commRangeInfo
            }

        return observation

    def _get_observation(self) -> dict:
        """获取当前观测（态势）"""
        simulators = self.engine.simulator_factory.get_all_simulators()

        observation = {
            "step": self.current_step,
            "entities": {}
        }

        for simulator in simulators:
            entity_data = simulator.entity_ext.entity
            observation["entities"][entity_data.id] = {
                "nameChn": entity_data.nameChn,
                "position": {
                    "lon": entity_data.lla.x,
                    "lat": entity_data.lla.y,
                    "alt": entity_data.lla.z
                },
                "pos_ecf":{
                    "x":entity_data.posEcf.x,
                    "y":entity_data.posEcf.y,
                    "z":entity_data.posEcf.z,
                },
                "stage":entity_data.stage,
                "health": entity_data.survivePoints,
                "isVisible": entity_data.isVisible,
                "type": entity_data.entityType,
                "side": entity_data.sideId,
                "detectInfo": entity_data.detectInfo,
                "commRangeInfo": entity_data.commRangeInfo
            }

        return observation

    def _get_red_observation(self) -> dict:
        """获取红方观测（态势）"""
        simulators = self.engine.simulator_factory.get_simulators_by_side(0)

        observation = {
            "step": self.current_step,
            "entities": {}
        }

        for simulator in simulators:
            entity_data = simulator.entity_ext.entity
            observation["entities"][entity_data.id] = {
                "nameChn": entity_data.nameChn,
                "position": {
                    "lon": entity_data.lla.x,
                    "lat": entity_data.lla.y,
                    "alt": entity_data.lla.z
                },
                "health": entity_data.survivePoints,
                "type": entity_data.entityType,
                "side": entity_data.sideId,
                "detectInfo": entity_data.detectInfo
            }

        return observation

    def _compute_reward(self) -> dict[int, float]:
        """
        计算每个智能体的奖励
        :return: {agent_id: reward}
        """
        rewards = {}

        # 获取所有智能体
        for agent in self.agent_manager.get_all_agents():
            agent_id = agent.agent_id
            entity_id = agent.entity_id

            # 获取该实体的仿真器
            simulator = self.engine.get_simulator_by_id(entity_id)
            if not simulator:
                rewards[agent_id] = 0.0
                continue

            entity_data = simulator.entity_ext.entity

            # ========== 示例：个体奖励设计 ==========
            reward = 0.0

            # 1. 生存奖励（活着就有基础分）
            if entity_data.survivePoints > 0:
                reward += 1.0

            # 2. 健康值变化奖励（可选，需要记录上一帧）
            # if entity_data.survivePoints > last_health:
            #     reward += 0.5  # 健康值增加

            # 3. 位置奖励（例如：接近目标）
            # TODO: 根据任务目标设计

            # 4. 击杀奖励（如果有战斗）
            # TODO: 检测是否击毁敌方

            # 5. 团队协作奖励（全局奖励）
            # TODO: 团队完成任务时给所有人加分

            rewards[agent_id] = reward

        return rewards

    def close(self):
        """关闭环境"""
        if self.renderer:
            self.renderer.stop()
        print("[TrainingEnv] 环境已关闭")

    # 委托方法
    def get_simulator(self, entity_id: int):
        return self.engine.get_simulator(entity_id)

    def get_simulators_by_type(self, entity_type: int):
        return self.engine.get_simulators_by_type(entity_type)

    def get_simulators_by_side(self, side_id: int):
        return self.engine.get_simulators_by_side(side_id)
