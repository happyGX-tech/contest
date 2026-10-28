# -*-coding:utf-8 -*-
import random
import time

import numpy as np

from user_agents.base_agent import BaseAgent, AgentType

ACTION_DEPLOY = 0
ACTION_COMPLETE = 1
RED_PLATFORM_TYPES = (21000, 21001, 21002)


class DeployAgent(BaseAgent):
    """Deploy every red unit at a legal random point in its continuous area."""

    def __init__(self, agent_id: int, entity_id: int, init_observation: dict,
                 deploy_coordinates: list[list[list[float]]], deploy_coordinatesHM: list[list[list[float]]]):
        super().__init__(agent_id, entity_id, AgentType.DEPLOY, init_observation)
        self.deploy_coordinates = deploy_coordinates
        self.deploy_coordinatesHM = deploy_coordinatesHM
        self.index = 0
        self.latest_observation = None

    def set_observation(self, observation: dict) -> None:
        self.latest_observation = observation

    def get_action(self) -> np.ndarray:
        """Return deployment rows ``[0, entity_id, lon, lat, altitude]``."""

        observation = self.latest_observation["entities"]
        # The red-side observation also contains support assets (for example,
        # type 9202 satellites). Only the three contestant-controlled flight
        # platform types participate in deployment.
        keys = [
            entity_id
            for entity_id, entity in observation.items()
            if entity.get("type") in RED_PLATFORM_TYPES
        ]
        batch_size = 10
        actions = []
        for offset in range(min(batch_size, len(keys) - self.index)):
            entity_id = keys[self.index + offset]
            entity_type = observation[entity_id]["type"]
            regions = (
                self.deploy_coordinatesHM
                if entity_type in (21000, 21001) and self.deploy_coordinatesHM
                else self.deploy_coordinates
            )
            lon, lat = self.random_point_in_region(regions)
            actions.append([
                ACTION_DEPLOY,
                entity_id,
                lon,
                lat,
                observation[entity_id]["position"]["alt"],
            ])

        self.index += len(actions)
        if not actions:
            return np.array([[ACTION_COMPLETE, -1, 0, 0, 0]], dtype=np.float64)
        time.sleep(0.1)
        return np.array(actions, dtype=np.float64)

    @classmethod
    def random_point_in_region(cls, coordinates):
        if not coordinates:
            raise ValueError("红方部署区域不能为空")
        # Preserve the historic random stream for the common single-polygon
        # competition area; choose randomly only when multiple areas exist.
        polygon = coordinates[0] if len(coordinates) == 1 else random.choice(coordinates)
        if len(polygon) < 3:
            raise ValueError("红方部署区域至少需要三个顶点")
        lons = [float(point[0]) for point in polygon]
        lats = [float(point[1]) for point in polygon]
        for _ in range(10_000):
            lon = random.uniform(min(lons), max(lons))
            lat = random.uniform(min(lats), max(lats))
            if cls._point_in_polygon(lon, lat, polygon):
                return lon, lat
        raise ValueError("无法在红方部署区域内生成合法位置")

    @staticmethod
    def _point_in_polygon(lon: float, lat: float, polygon) -> bool:
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

    # Compatibility for callers that used the old rectangle-specific helper.
    random_point_in_rect = random_point_in_region

    def reset(self):
        super().reset()
        self.index = 0
