# -*-coding:utf-8 -*-
import logging
import random
import time
import numpy as np

from user_agents.base_agent import BaseAgent, AgentType

ACTION_DEPLOY = 0
ACTION_COMPLETE = 1


class DeployAgent(BaseAgent):
    """部署智能体（示例）"""

    def __init__(self, agent_id: int, entity_id: int, init_observation: dict,
                 deploy_coordinates: list[list[list[float]]], deploy_coordinatesHM: list[list[list[float]]],
                 unified_policy=None):
        super().__init__(agent_id, entity_id, AgentType.DEPLOY, init_observation)
        # 红方可部署区域
        self.deploy_coordinates: list[list[list[float]]] = deploy_coordinates
        self.deploy_coordinatesHM: list[list[list[float]]] = deploy_coordinatesHM
        # 统一策略（学习式部署头）；None 时退回随机部署
        self.unified_policy = unified_policy
        self.index = 0
        self.latest_observation = None

    def set_observation(self, observation: dict) -> None:
        self.latest_observation = observation

    def get_action(self) -> np.ndarray:
        """
        生成动作

        Returns:
            np.ndarray: 二维数组，每行格式为 [action_type, entity_id, lon, lat, alt]
            行数表示同时执行的动作数量
            当 action_type=1 时，只有一行，且 entity_id=-1
        """
        # random.seed(42)

        observation = self.latest_observation["entities"]
        keys = list(observation.keys())

        # 可以配置每次部署的实体数量
        batch_size = 10

        actions = []
        for i in range(min(batch_size, len(keys) - self.index)):
            entity_id = keys[self.index + i]

            entity_type = observation[entity_id]["type"]
            if (entity_type == 21000 or entity_type == 21001) and len(self.deploy_coordinatesHM) > 0:
                zone = self.deploy_coordinatesHM
            else:
                zone = self.deploy_coordinates

            if self.unified_policy is not None:
                # 学习式部署头：tanh 高斯采样映射到部署区 bbox；
                # scripted 阶段返回 None → 回退规则部署（课程前期固定规则）
                deploy_choice = self.unified_policy.select_deploy(
                    entity_id=int(entity_id),
                    entity_index=self.index + i,
                    total_entities=len(keys),
                    missile_type=int(entity_type),
                    zone_bbox=self._zone_bbox(zone),
                )
                if deploy_choice is not None:
                    lon, lat = deploy_choice
                else:
                    lon, lat = self.random_point_in_rect(zone)
            else:
                lon, lat = self.random_point_in_rect(zone)

            if entity_type == 21002:
                actions.append([
                    ACTION_DEPLOY,
                    entity_id,
                    lon,
                    lat,
                    10000  # observation[entity_id]["position"]["alt"]
                ])
            else:
                actions.append([
                    ACTION_DEPLOY,
                    entity_id,
                    lon,
                    lat,
                    0
                ])

        self.index += len(actions)

        # 没有更多实体，返回完成标志
        if not actions:
            return np.array([[ACTION_COMPLETE, -1, 0, 0, 0]], dtype=np.float64)
        time.sleep(0.1)
        return np.array(actions, dtype=np.float64)

    @staticmethod
    def _zone_bbox(coordinates) -> tuple[float, float, float, float]:
        """部署区多边形 → (min_lon, max_lon, min_lat, max_lat)"""
        polygon = coordinates[0]
        lons = [point[0] for point in polygon]
        lats = [point[1] for point in polygon]
        return min(lons), max(lons), min(lats), max(lats)

    def random_point_in_rect(self, coordinates):
        """
        在矩形区域内生成随机点

        Args:
            coordinates: [[[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon4, lat4]]]

        Returns:
            (lon, lat): 随机经纬度
        """
        # 提取矩形的四个角
        polygon = coordinates[0]

        # 获取经纬度范围
        lons = [point[0] for point in polygon]
        lats = [point[1] for point in polygon]

        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # 在范围内随机生成
        lon = random.uniform(min_lon, max_lon)
        lat = random.uniform(min_lat, max_lat)

        return lon, lat

    def reset(self):
        """重置状态"""
        super().reset()
        self.index = 0
