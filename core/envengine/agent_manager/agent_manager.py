# -*- coding: utf-8 -*-#
import logging
from copy import deepcopy
from typing import Dict, Optional, List
import numpy as np
from envengine.action_validation import IllegalActionError
from user_agents.base_agent import BaseAgent, AgentType

logger = logging.getLogger(__name__)


class AgentManager:
    """
    智能体工厂
    负责创建、管理和查询智能体实例
    为每个智能体分发对应的隔离观测
    """

    def __init__(self):
        self._agents: Dict[int, BaseAgent] = {}  # agent_id -> agent
        self._entity_to_agent: Dict[int, int] = {}  # entity_id -> agent_id
        self._agent_to_entity: Dict[int, int] = {}  # agent_id -> entity_id

    def register_agent(self, agent: BaseAgent):
        """
        注册智能体
        :param agent: 智能体实例
        """
        agent_id = agent.agent_id
        entity_id = agent.get_entity_id()

        self._agents[agent_id] = agent
        self._entity_to_agent[entity_id] = agent_id
        self._agent_to_entity[agent_id] = entity_id

        logger.info(f"[智能体工厂] 注册智能体 {agent_id} -> 实体 {entity_id}")

    def get_agent(self, agent_id: int) -> Optional[BaseAgent]:
        """获取指定智能体"""
        return self._agents.get(agent_id)

    def get_agent_by_entity(self, entity_id: int) -> Optional[BaseAgent]:
        """通过实体ID获取对应的智能体"""
        agent_id = self._entity_to_agent.get(entity_id)
        if agent_id is not None:
            return self._agents.get(agent_id)
        return None

    def get_agent_by_type(self, agent_type: AgentType):
        """
        通过智能体类型获取智能体
        :param agent_type:
        :return:
        """
        agents: List[BaseAgent] = self.get_all_agents()
        return [agent for agent in agents if agent.agent_type == agent_type]

    def get_all_agents(self) -> List[BaseAgent]:
        """获取所有智能体"""
        return list(self._agents.values())

    def get_agent_count(self) -> int:
        """获取智能体总数"""
        return len(self._agents)

    def extract_observation_for_agent(self, full_observation: dict, agent_id: int) -> dict:
        """
        从完整观测中提取指定智能体的隔离观测
        :param full_observation: 完整的观测信息
        :param agent_id: 智能体ID
        :return: 该智能体的隔离观测
        """
        entity_id = self._agent_to_entity.get(agent_id)
        if entity_id is None:
            logger.warning(f"[智能体工厂] 智能体 {agent_id} 未找到对应实体")
            return {}

        # 只返回该智能体对应实体的观测
        entities = full_observation.get("entities", {})
        if entity_id in entities:
            self_info = entities[entity_id]
            if self_info["health"] > 0 and self_info["isVisible"]:
                return {
                    "step": full_observation.get("step", 0),
                    "entity_id": entity_id,
                    "agent_id": agent_id,
                    "self": deepcopy(entities[entity_id])
                }
            return {}
        else:
            # logger.warning(f"[智能体工厂] 实体 {entity_id} 不在观测中")
            return {}

    def collect_actions_from_agents(self, agent_observations: Dict[int, dict]) -> list[np.array]:
        """
        向所有智能体传递观测信息，并收集智能体的动作
        :param agent_observations: {agent_id: observation}
        :return: {entity_id: actions} 格式的总动作字典
        """
        actions = []
        for agent_id, observation in agent_observations.items():
            agent = self._agents.get(agent_id)
            if agent is None:
                raise IllegalActionError(f"智能体 {agent_id} 未注册")
            agent.set_observation(deepcopy(observation))

        for agent_id in agent_observations:
            # Ownership comes from registration, never from the mutable agent.
            entity_id = self._agent_to_entity[agent_id]
            raw_action = self._agents[agent_id].get_action()
            try:
                action = np.asarray(raw_action, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise IllegalActionError(f"智能体 {agent_id} 的动作必须是数值数组") from exc
            if action.shape in ((0,), (0, 4)):
                actions.append(np.empty((0, 4), dtype=np.float64))
                continue
            if action.ndim != 2 or action.shape[1] != 4 or not np.isfinite(action).all():
                raise IllegalActionError(f"智能体 {agent_id} 的动作必须是有限数值 N x 4 数组")
            if np.any(action[:, 1] != entity_id):
                raise IllegalActionError(f"智能体 {agent_id} 只能控制自身实体 {entity_id}")
            actions.append(action.copy())

        return actions

    def collect_actions_from_deploy_agent(self, agent_observations: Dict[int, dict]) -> np.array:
        """
        收集部署智能体的动作
        :param agent_observations: {agent_id: observation}
        :return: {entity_id: actions} 格式的总动作字典
        """
        agent = self._agents.get(-1)
        if agent is None:
            raise RuntimeError("部署智能体不存在")
        if -1 not in agent_observations:
            raise RuntimeError("部署智能体未收到部署观测")

        # Deployment defines the initial state of the whole round. Propagate
        # failures instead of swallowing them and returning an empty object,
        # which would otherwise leave TrainingEnv in an infinite deploy loop.
        agent.set_observation(agent_observations[-1])
        return agent.get_action()

    def reset_all(self):
        """重置所有智能体"""
        for agent in self._agents.values():
            agent.reset()
        logger.info("[智能体工厂] 所有智能体已重置")

    def clear(self):
        """清空所有智能体"""
        self._agents.clear()
        self._entity_to_agent.clear()
        self._agent_to_entity.clear()
