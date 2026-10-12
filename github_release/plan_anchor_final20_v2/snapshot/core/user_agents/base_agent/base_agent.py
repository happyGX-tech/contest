# -*- coding: utf-8 -*-#
from abc import ABC, abstractmethod
from typing import Any, List
from collections import deque

import numpy as np

from user_agents.base_agent.agent_type import AgentType


class BaseAgent(ABC):
    """智能体基类"""

    def __init__(self, agent_id: int, entity_id: int, agent_type: AgentType, init_observation:dict, history_length: int = 10):
        """
        :param agent_id: 智能体ID
        :param entity_id: 对应的实体ID
        :param history_length: 历史记录长度（默认10步）
        """
        self.agent_id = agent_id
        self.entity_id = entity_id
        # 智能体类型
        self.agent_type: AgentType = agent_type

        self.init_observation = init_observation

        # 历史记录管理器
        self.history_length = history_length
        self.observation_history = deque(maxlen=history_length)
        self.action_history = deque(maxlen=history_length)
        self.reward_history = deque(maxlen=history_length)
        self.info_history = deque(maxlen=history_length)

    @abstractmethod
    def set_observation(self, observation: dict) -> None:
        """
        传入观测信息
        :param observation: 该智能体对应的观测信息
        """
        pass

    @abstractmethod
    def get_action(self) -> np.array:
        """
        获取动作
        :return: 动作
        """
        pass

    def record_step(self, observation: dict, action: Any, reward: float, info: dict = None):
        """
        记录一步的经验（由环境自动调用）
        :param observation: 当前观测
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param info: 额外信息
        """
        self.observation_history.append(observation)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.info_history.append(info or {})

    def get_recent_observations(self, steps: int = None) -> List[dict]:
        """
        获取最近的观测历史
        :param steps: 获取最近多少步，None表示全部
        :return: 观测历史列表
        """
        if steps is None:
            return list(self.observation_history)
        return list(self.observation_history)[-steps:]

    def get_recent_actions(self, steps: int = None) -> List[Any]:
        """获取最近的动作历史"""
        if steps is None:
            return list(self.action_history)
        return list(self.action_history)[-steps:]

    def get_recent_rewards(self, steps: int = None) -> List[float]:
        """获取最近的奖励历史"""
        if steps is None:
            return list(self.reward_history)
        return list(self.reward_history)[-steps:]

    def get_cumulative_reward(self, steps: int = None) -> float:
        """
        获取累积奖励
        :param steps: 最近多少步，None表示全部
        :return: 累积奖励值
        """
        rewards = self.get_recent_rewards(steps)
        return sum(rewards)

    @abstractmethod
    def reset(self):
        """重置智能体状态"""
        # 清空历史
        self.observation_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.info_history.clear()

    def get_entity_id(self) -> int:
        """获取该智能体控制的实体ID"""
        return self.entity_id
