# -*-coding:utf-8 -*-

import logging
from copy import deepcopy
from typing import Callable, Optional, Protocol

import numpy as np

from envengine.agent_manager import AgentManager
from envengine.agent_manager.actions.deploy_action import SetLLA
from envengine.engine import Engine
from envengine.environment.command_adapter import CommandAdapter
from envengine.environment.command_converter import CommandConverter
from envengine.render import Renderer
from envengine.sdk.base_struct.profile.profile import Profile
from envengine.sdk.writer import write_state, write_ai_action, write_config, write_immediately
from envengine.simulator.util import SpeedDistributor
from user_agents.base_agent import AgentType
from envengine.common import Vector3d

logger = logging.getLogger(__name__)


class TeamController(Protocol):
    """Planner for exactly one engine-authorized communication component."""

    def register_platform(self, entity_id: int) -> None:
        """Register a live member of this component."""

    def reset(self) -> None:
        """Discard this component's planning state at round end."""

    def begin_step(self, observations: tuple[dict, ...]) -> None:
        """Receive only observations from members of this component."""


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
        self._team_controller_factory: Callable[[], TeamController] | None = None
        self._team_controllers: dict[tuple[int, ...], TeamController] = {}
        self._deployment_coordinates = profile.imagineProfile.redArea.coordinates
        self._deployment_coordinates_hm = profile.imagineProfile.redArea.coordinatesHM
        self._deployment_launch_altitude: float | None = None

    def set_team_controller_factory(self, factory: Callable[[], TeamController]) -> None:
        """Create independent planners as communication components form.

        A factory may capture public mission configuration. It must return a
        fresh instance: disconnected components cannot share mutable planners.
        """
        if not callable(factory):
            raise TypeError("通信群指挥器工厂必须可调用")
        self._team_controller_factory = factory
        self._team_controllers.clear()

    def set_deployment_constraints(
        self,
        coordinates,
        coordinates_hm,
        launch_altitude: float | None = None,
    ) -> None:
        """Set environment-owned bounds used to validate red deployment."""

        self._deployment_coordinates = coordinates
        self._deployment_coordinates_hm = coordinates_hm
        self._deployment_launch_altitude = launch_altitude

    def set_shared_learning_policy(self, policy) -> None:
        """Register a training-only centralized critic lifecycle hook.

        Platform agents still receive only isolated observations.  The shared
        policy is supplied the full state solely when explicitly training.
        """

        if policy is not None and not bool(getattr(policy, "training", False)):
            raise ValueError("全局 Critic 状态仅允许提供给显式训练的学习策略")
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

        for controller in self._team_controllers.values():
            controller.reset()
        self._team_controllers.clear()

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

        expected_ids = {
            simulator.entity_ext.entity.id
            for entity_type in (21000, 21001, 21002)
            for simulator in self.engine.simulator_factory.get_simulators_by_type(entity_type)
        }
        deployed_ids: set[int] = set()

        while True:
            raw_actions = self._generate_actions_from_deploy_agent()
            try:
                rows = np.asarray(raw_actions, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("部署动作必须是有限数值组成的 N x 5 数组") from exc
            if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != 5:
                raise ValueError("部署动作必须是非空 N x 5 数组")
            if not np.isfinite(rows).all():
                raise ValueError("部署动作不能包含 NaN 或无穷值")

            actions = []
            batch_ids: set[int] = set()
            deploy_completed = False
            for row in rows:
                if not float(row[0]).is_integer():
                    raise ValueError(f"未知部署动作类型 {row[0]}")
                action_type = int(row[0])
                if deploy_completed:
                    raise ValueError("部署完成动作必须是本批次最后一项")

                if action_type == 0:
                    if not float(row[1]).is_integer():
                        raise ValueError(f"部署实体编号必须是整数: {row[1]}")
                    entity_id = int(row[1])
                    if entity_id not in expected_ids:
                        raise ValueError(f"实体 {entity_id} 不属于红方作战部署单位")
                    if entity_id in deployed_ids or entity_id in batch_ids:
                        raise ValueError(f"实体 {entity_id} 收到了重复部署动作")

                    lon, lat, altitude = map(float, row[2:5])
                    self._validate_deployment_position(
                        entity_id,
                        lon,
                        lat,
                        altitude,
                    )
                    actions.append(
                        SetLLA(
                            executor_id=entity_id,
                            lla=Vector3d(x=lon, y=lat, z=altitude),
                        ).to_dict()
                    )
                    batch_ids.add(entity_id)
                elif action_type == 1:
                    missing_ids = expected_ids - deployed_ids - batch_ids
                    if missing_ids:
                        preview = ", ".join(map(str, sorted(missing_ids)[:10]))
                        raise ValueError(
                            f"部署尚未覆盖全部红方作战单位，缺少 {len(missing_ids)} 个"
                            f"（{preview}）"
                        )
                    deploy_completed = True
                else:
                    raise ValueError(f"未知部署动作类型 {action_type}")

            # Validate the complete batch before mutating simulator state.
            for action_item in actions:
                write_ai_action(action_item, str(self.current_round) + ".json")
                self.engine.simulator_factory.modify_simulator_position(
                    action_item["executor_id"], action_item["lla"]
                )
                speed = self.speed_distributor.get_speed(action_item["lla"]["x"])
                self.engine.simulator_factory.modify_simulator_speed(
                    action_item["executor_id"], speed
                )
                if self.render_mode == 'human' and self.renderer:
                    observation = self._get_observation()
                    self.renderer.update_data(observation["entities"])
            deployed_ids.update(batch_ids)
            if deploy_completed:
                break

        # Red flight models are initialized during reset and repositioned during
        # deployment. Start them once, only after every final position is known.
        self.engine.simulator_factory.start_red_missions()

    def _validate_deployment_position(
        self,
        entity_id: int,
        lon: float,
        lat: float,
        altitude: float,
    ) -> None:
        simulator = self.engine.simulator_factory.get_simulator_by_id(entity_id)
        if simulator is None:
            raise ValueError(f"部署动作引用了不存在的实体 {entity_id}")

        entity_type = simulator.entity_ext.entity.entityType
        if entity_type in (21000, 21001):
            regions = getattr(
                self,
                "_deployment_coordinates_hm",
                self.engine.profile.imagineProfile.redArea.coordinatesHM,
            )
        elif entity_type == 21002:
            regions = getattr(
                self,
                "_deployment_coordinates",
                self.engine.profile.imagineProfile.redArea.coordinates,
            )
        else:
            raise ValueError(f"实体 {entity_id} 的类型 {entity_type} 不允许红方部署")

        if not any(self._point_in_polygon(lon, lat, polygon) for polygon in (regions or ())):
            raise ValueError(
                f"实体 {entity_id} 的部署位置 ({lon}, {lat}) 超出允许区域"
            )

        expected_altitude = getattr(self, "_deployment_launch_altitude", None)
        if expected_altitude is not None and abs(altitude - expected_altitude) > 1e-6:
            raise ValueError(
                f"实体 {entity_id} 的部署高度 {altitude} 不符合要求 {expected_altitude}"
            )

    @staticmethod
    def _point_in_polygon(lon: float, lat: float, polygon) -> bool:
        """Return whether a point lies inside or on a polygon boundary."""

        if len(polygon) < 3:
            return False
        inside = False
        previous = polygon[-1]
        for current in polygon:
            x1, y1 = float(previous[0]), float(previous[1])
            x2, y2 = float(current[0]), float(current[1])
            cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
            if (
                abs(cross) <= 1e-12
                and min(x1, x2) - 1e-12 <= lon <= max(x1, x2) + 1e-12
                and min(y1, y2) - 1e-12 <= lat <= max(y1, y2) + 1e-12
            ):
                return True
            if (y1 > lat) != (y2 > lat):
                intersection = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
                if lon < intersection:
                    inside = not inside
            previous = current
        return inside

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

    def _generate_actions_from_agents(self) -> list[dict]:
        """Plan from communication-scoped observations, then convert actions."""
        groups = self.engine.refresh_red_communications()
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

        self._prepare_team_controllers(groups, agent_observations)

        # 收集所有智能体的动作
        actions = self.agent_manager.collect_actions_from_agents(agent_observations)
        # 转换所有智能体动作到结构体类型
        return CommandConverter.common_converter(actions)

    def _prepare_team_controllers(
        self,
        groups: tuple[tuple[int, ...], ...],
        observations: dict[int, dict],
    ) -> None:
        """Bind each agent to its current component without cross-domain state.

        Stable components retain their planner. Merges and splits create fresh
        planners from public priors and the members' engine-owned track memory.
        A split therefore retains previously received intelligence but cannot
        receive any new information from the disconnected component.
        """
        factory = self._team_controller_factory
        if factory is None:
            return
        by_entity = {int(item["entity_id"]): item for item in observations.values()}
        next_controllers: dict[tuple[int, ...], TeamController] = {}
        for group in groups:
            key = tuple(sorted(group))
            members = tuple(by_entity[entity_id] for entity_id in key if entity_id in by_entity)
            if not members:
                continue
            controller = self._team_controllers.get(key)
            if controller is None:
                controller = factory()
                if any(controller is prior for prior in (
                    *self._team_controllers.values(), *next_controllers.values()
                )):
                    raise ValueError("不同通信群必须使用独立的指挥器实例")
                for observation in members:
                    controller.register_platform(int(observation["entity_id"]))
            next_controllers[key] = controller
            controller.begin_step(deepcopy(members))
            for observation in members:
                agent = self.agent_manager.get_agent(observation["agent_id"])
                agent.set_commander(controller)
        covered_ids = {entity_id for group in next_controllers for entity_id in group}
        if by_entity.keys() - covered_ids:
            raise ValueError("红方观测中存在未被引擎通信分群覆盖的作战单位")
        self._team_controllers = next_controllers

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

    def get_public_targets_observation(self) -> dict:
        """Return the public mother/sentry catalogue before combat starts."""
        if self.current_step != 0:
            raise RuntimeError("公开目标初始目录只能在战斗开始前读取")

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
                "detectInfo": deepcopy(entity_data.detectInfo),
                "commRangeInfo": deepcopy(entity_data.commRangeInfo)
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
                "detectInfo": deepcopy(entity_data.detectInfo)
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

            # Per-step learning reward; competition scoring lives in scenarios.
            rewards[agent_id] = float(entity_data.survivePoints > 0)

        return rewards

    def close(self):
        """关闭环境"""
        if self.renderer:
            self.renderer.stop()
        print("[TrainingEnv] 环境已关闭")

    # 委托方法
    def get_simulator(self, entity_id: int):
        return self.engine.get_simulator_by_id(entity_id)

    def get_simulators_by_type(self, entity_type: int):
        return self.engine.get_simulators_by_type(entity_type)

    def get_simulators_by_side(self, side_id: int):
        return self.engine.get_simulators_by_side(side_id)
