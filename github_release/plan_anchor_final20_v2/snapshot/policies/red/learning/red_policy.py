"""红方学习基线所需的观测、动作和共享策略抽象。

高层动作空间固定为：等待、对 5 个槽位发射、对 5 个槽位改目标、
左/右/停止机动、使用卫星。适配器负责将高层动作转换为仿真引擎已有的
四元动作，因此学习算法不需要直接回归经纬度。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_TARGET_SLOTS = 5


class HighLevelAction(IntEnum):
    MANEUVER_LEFT = 0
    STOP_MANEUVER = 1
    MANEUVER_RIGHT = 2


ACTION_DIM = len(HighLevelAction)


@dataclass(frozen=True)
class PolicyTransition:
    """一条单智能体经验；共享策略会收到所有红方导弹的经验。"""

    agent_id: int
    observation: np.ndarray
    action: int
    action_mask: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    global_state: np.ndarray | None = None
    next_global_state: np.ndarray | None = None


class SharedPolicy(ABC):
    """参数共享策略接口。

    PPO/MAPPO 实现只需实现 ``select_action``，并按需覆盖其余钩子。
    """

    @abstractmethod
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """在合法动作中选择一个高层动作。"""

    def observe(self, transition: PolicyTransition) -> None:
        """接收环境 transition；无在线学习需求的策略可以忽略。"""

    def reset_episode(self) -> None:
        """一轮开始时调用。共享策略应能安全地被多个 Agent 重复调用。"""

    def begin_environment_step(self, full_observation: Mapping[str, Any]) -> None:
        """联合动作生成前保存全局状态的可选钩子。"""

    def end_environment_step(self, full_observation: Mapping[str, Any]) -> None:
        """环境步进后保存下一全局状态的可选钩子。"""

    def finish_environment_step(self) -> None:
        """所有智能体 transition 均提交后的可选钩子。"""

    def get_global_state_context(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return None, None

    def save(self, path: str) -> None:
        raise NotImplementedError("该策略没有实现模型保存")

    def load(self, path: str) -> None:
        raise NotImplementedError("该策略没有实现模型加载")


class RandomMaskedPolicy(SharedPolicy):
    """无需深度学习依赖的烟雾测试策略。

    它不是性能基线，而是用来验证观测、掩码、动作转换和经验回传链路。
    """

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self.transition_count = 0

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        del observation
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            return int(HighLevelAction.STOP_MANEUVER)
        return int(self._rng.choice(valid_actions))

    def observe(self, transition: PolicyTransition) -> None:
        del transition
        self.transition_count += 1


class ObservationEncoder:
    """将可变结构的原始观测编码为固定长度 float32 向量。"""

    SELF_FEATURES = 21
    TARGET_FEATURES = 12
    DETECTION_FEATURES = 4
    TASK_FEATURES = 5

    def __init__(
        self,
        init_observation: Mapping[str, Any],
        target_slots: int = DEFAULT_TARGET_SLOTS,
        max_steps: int = 1000,
        coordinate_scale: float = 1.0,
        agent_id: int = 0,
        team_size: int = 58,
        hierarchical_task_context: bool = False,
    ):
        self.target_slots = target_slots
        self.max_steps = max(1, max_steps)
        self.coordinate_scale = max(float(coordinate_scale), 1e-6)
        self.agent_id = int(agent_id)
        self.team_size = max(1, int(team_size))
        self.hierarchical_task_context = bool(hierarchical_task_context)
        entities = init_observation.get("entities", {})
        self.targets = sorted(
            (dict(value, entity_id=int(key)) for key, value in entities.items()),
            key=lambda item: item["entity_id"],
        )[:target_slots]

    @property
    def observation_dim(self) -> int:
        base_dim = (
            self.SELF_FEATURES
            + self.target_slots * self.TARGET_FEATURES
            + self.DETECTION_FEATURES
        )
        return base_dim + (self.TASK_FEATURES if self.hierarchical_task_context else 0)

    def encode(
        self,
        observation: Mapping[str, Any],
        *,
        launched: bool,
        launch_step: int,
        satellite_used: bool,
        maneuver_state: int,
        current_target_index: int | None = None,
        target_switch_elapsed: int = 0,
        task_context: Sequence[float] | None = None,
    ) -> np.ndarray:
        self_info = observation["self"]
        position = self_info["position"]
        lon = float(position["lon"])
        lat = float(position["lat"])
        step = int(observation.get("step", 0))
        entity_type = int(self_info.get("type", 0))
        elapsed = 0 if not launched else max(0, step - launch_step)
        velocity = self_info.get("velocity") or {}
        speed = float(velocity.get("speed", 0.0))
        vertical_speed = float(velocity.get("up", 0.0))
        heading = float(velocity.get("heading", 0.0))
        agent_slot = np.clip(
            max(0, self.agent_id - 1) / max(self.team_size - 1, 1), 0.0, 1.0
        )
        role_index = max(0, self.agent_id - 1) % 3

        features = [
            np.clip(step / self.max_steps, 0.0, 1.0),
            lon / 180.0,
            lat / 90.0,
            np.clip(float(position.get("alt", 0.0)) / 20000.0, -1.0, 1.0),
            np.clip(float(self_info.get("health", 0.0)) / 100.0, 0.0, 1.0),
            float(entity_type == 21000),
            float(entity_type == 21001),
            float(entity_type == 21002),
            float(launched),
            np.clip(elapsed / self.max_steps, 0.0, 1.0),
            float(satellite_used),
            np.clip(float(maneuver_state) / 2.0, -1.0, 1.0),
            np.clip(float(target_switch_elapsed) / 50.0, 0.0, 1.0),
            np.clip(speed / 1500.0, 0.0, 2.0),
            np.clip(vertical_speed / 500.0, -1.0, 1.0),
            np.sin(heading),
            np.cos(heading),
            agent_slot,
            float(role_index == 0),
            float(role_index == 1),
            float(role_index == 2),
        ]

        detect_info = self_info.get("detectInfo") or {}
        detections = {
            int(getattr(info, "entity_id", key)): info
            for key, info in detect_info.items()
        }

        for index in range(self.target_slots):
            if index >= len(self.targets):
                features.extend([0.0] * self.TARGET_FEATURES)
                continue
            target = self.targets[index]
            target_id = int(target["entity_id"])
            detection = detections.get(target_id)
            if detection is not None:
                lla = getattr(detection, "lla", None)
                target_position = {
                    "lon": getattr(lla, "x", target["position"]["lon"]),
                    "lat": getattr(lla, "y", target["position"]["lat"]),
                }
                detection_age = max(0, step - int(getattr(detection, "time", step)))
            else:
                target_position = target["position"]
                detection_age = self.max_steps
            name = str(target.get("nameChn", ""))
            delta_lon = float(target_position["lon"]) - lon
            delta_lat = float(target_position["lat"]) - lat
            mean_latitude = np.radians((float(target_position["lat"]) + lat) / 2.0)
            east_km = delta_lon * 111.32 * np.cos(mean_latitude)
            north_km = delta_lat * 110.57
            distance_km = float(np.hypot(east_km, north_km))
            bearing = float(np.arctan2(east_km, north_km))
            relative_bearing = bearing - heading
            features.extend([
                np.clip(delta_lon / 10.0, -1.0, 1.0),
                np.clip(delta_lat / 10.0, -1.0, 1.0),
                np.clip(distance_km / 1000.0, 0.0, 1.0),
                np.sin(bearing),
                np.cos(bearing),
                float(detection is not None),
                np.clip(detection_age / self.max_steps, 0.0, 1.0),
                float(name.startswith("目标")),
                float(name.startswith("拦截阵地")),
                float(current_target_index == index),
                np.sin(relative_bearing),
                np.cos(relative_bearing),
            ])

        detected_names = [str(getattr(info, "nameChn", "")) for info in detect_info.values()]
        comm_count = len(self_info.get("commRangeInfo") or [])
        features.extend([
            np.clip(len(detect_info) / 200.0, 0.0, 1.0),
            np.clip(sum(name.startswith("标6") for name in detected_names) / 200.0, 0.0, 1.0),
            np.clip(sum(name.startswith("无人船") for name in detected_names) / 10.0, 0.0, 1.0),
            np.clip(comm_count / 64.0, 0.0, 1.0),
        ])
        if self.hierarchical_task_context:
            context = tuple(task_context or ())[:self.TASK_FEATURES]
            features.extend(context)
            features.extend([0.0] * (self.TASK_FEATURES - len(context)))
        return np.asarray(features, dtype=np.float32)


class GlobalStateEncoder:
    """将完整态势编码为集中式 Critic 使用的固定长度全局状态。"""

    RED_TYPES = (21000, 21001, 21002)
    BLUE_PREFIXES = ("目标", "拦截阵地", "普通雷达", "标6", "无人船")
    EXTRA_RED_FEATURES = 9

    def __init__(self, max_steps: int = 1000, target_slots: int = DEFAULT_TARGET_SLOTS):
        self.max_steps = max(1, int(max_steps))
        self.target_slots = int(target_slots)

    @property
    def state_dim(self) -> int:
        return (
            1
            + len(self.RED_TYPES) * 5
            + len(self.BLUE_PREFIXES) * 3
            + self.target_slots * 6
            + self.EXTRA_RED_FEATURES
        )

    @staticmethod
    def _mean(items, getter):
        values = [float(getter(item)) for item in items]
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _distance_km(first_position, second_position):
        lon1, lat1 = np.radians([
            float(first_position.get("lon", 0.0)),
            float(first_position.get("lat", 0.0)),
        ])
        lon2, lat2 = np.radians([
            float(second_position.get("lon", 0.0)),
            float(second_position.get("lat", 0.0)),
        ])
        delta_lon = lon2 - lon1
        delta_lat = lat2 - lat1
        value = (
            np.sin(delta_lat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
        )
        return float(6371.0 * 2.0 * np.arctan2(np.sqrt(value), np.sqrt(max(0.0, 1.0 - value))))

    def encode(self, observation: Mapping[str, Any]) -> np.ndarray:
        entities_map = observation.get("entities") or {}
        entities = [dict(item, entity_id=int(entity_id)) for entity_id, item in entities_map.items()]
        features = [np.clip(float(observation.get("step", 0)) / self.max_steps, 0.0, 1.0)]

        for entity_type in self.RED_TYPES:
            group = [item for item in entities if int(item.get("type", 0)) == entity_type]
            alive = [item for item in group if float(item.get("health", 0.0)) > 0]
            features.extend([
                len(alive) / max(len(group), 1),
                np.clip(self._mean(group, lambda x: x.get("health", 0.0)) / 100.0, 0.0, 1.0),
                self._mean(alive, lambda x: x.get("position", {}).get("lon", 0.0)) / 180.0,
                self._mean(alive, lambda x: x.get("position", {}).get("lat", 0.0)) / 90.0,
                np.clip(self._mean(alive, lambda x: x.get("position", {}).get("alt", 0.0)) / 20000.0, -1.0, 1.0),
            ])

        for prefix in self.BLUE_PREFIXES:
            group = [item for item in entities if str(item.get("nameChn", "")).startswith(prefix)]
            alive_count = sum(float(item.get("health", 0.0)) > 0 for item in group)
            features.extend([
                np.clip(len(group) / 200.0, 0.0, 1.0),
                alive_count / max(len(group), 1),
                np.clip(self._mean(group, lambda x: x.get("health", 0.0)) / 100.0, 0.0, 1.0),
            ])

        alive_red = [
            item for item in entities
            if int(item.get("type", 0)) in self.RED_TYPES and float(item.get("health", 0.0)) > 0
        ]
        key_targets = [
            item for item in entities
            if str(item.get("nameChn", "")).startswith("目标")
            and float(item.get("health", 0.0)) > 0
        ]
        nearest_distances = [
            min(
                self._distance_km(item.get("position", {}), target.get("position", {}))
                for target in key_targets
            )
            for item in alive_red
        ] if key_targets else []
        speeds = [float((item.get("velocity") or {}).get("speed", 0.0)) for item in alive_red]
        longitudes = [float(item.get("position", {}).get("lon", 0.0)) for item in alive_red]
        latitudes = [float(item.get("position", {}).get("lat", 0.0)) for item in alive_red]
        altitudes = [float(item.get("position", {}).get("alt", 0.0)) for item in alive_red]
        features.extend([
            np.clip(float(np.mean(speeds)) / 1500.0, 0.0, 2.0) if speeds else 0.0,
            np.clip(float(np.std(speeds)) / 1500.0, 0.0, 1.0) if speeds else 0.0,
            np.clip(float(np.std(longitudes)) / 10.0, 0.0, 1.0) if longitudes else 0.0,
            np.clip(float(np.std(latitudes)) / 10.0, 0.0, 1.0) if latitudes else 0.0,
            np.clip(float(np.std(altitudes)) / 20000.0, 0.0, 1.0) if altitudes else 0.0,
            np.clip(min(nearest_distances) / 1000.0, 0.0, 2.0) if nearest_distances else 0.0,
            np.clip(float(np.mean(nearest_distances)) / 1000.0, 0.0, 2.0) if nearest_distances else 0.0,
            np.clip(float(np.std(nearest_distances)) / 1000.0, 0.0, 1.0) if nearest_distances else 0.0,
            sum(speed > 1.0 for speed in speeds) / max(len(alive_red), 1),
        ])

        targets = sorted(
            [item for item in entities if str(item.get("nameChn", "")).startswith(("目标", "拦截阵地"))],
            key=lambda item: item["entity_id"],
        )[:self.target_slots]
        for index in range(self.target_slots):
            if index >= len(targets):
                features.extend([0.0] * 6)
                continue
            target = targets[index]
            position = target.get("position", {})
            name = str(target.get("nameChn", ""))
            features.extend([
                float(position.get("lon", 0.0)) / 180.0,
                float(position.get("lat", 0.0)) / 90.0,
                np.clip(float(position.get("alt", 0.0)) / 20000.0, -1.0, 1.0),
                np.clip(float(target.get("health", 0.0)) / 100.0, 0.0, 1.0),
                float(float(target.get("health", 0.0)) > 0),
                float(name.startswith("目标")),
            ])
        return np.asarray(features, dtype=np.float32)


class LearningActionAdapter:
    """构造动作掩码，并将高层动作翻译成引擎四元动作。"""

    ACTION_SET_ACC_Z = 0
    ACTION_LAUNCH = 1

    def __init__(self, targets: Sequence[Mapping[str, Any]], target_slots: int = DEFAULT_TARGET_SLOTS):
        self.targets = list(targets)[:target_slots]
        self.target_slots = target_slots

    def build_action_mask(
        self, *, launched: bool, satellite_used: bool, target_switch_allowed: bool = True
    ) -> np.ndarray:
        del launched, satellite_used, target_switch_allowed
        return np.ones(ACTION_DIM, dtype=np.bool_)

    def to_engine_action(self, action: int, entity_id: int) -> np.ndarray:
        action = int(action)
        if action == HighLevelAction.MANEUVER_LEFT:
            return np.asarray([[self.ACTION_SET_ACC_Z, entity_id, -1.0, 0.0]], dtype=np.float64)
        if action == HighLevelAction.MANEUVER_RIGHT:
            return np.asarray([[self.ACTION_SET_ACC_Z, entity_id, 1.0, 0.0]], dtype=np.float64)
        if action == HighLevelAction.STOP_MANEUVER:
            return np.asarray([[self.ACTION_SET_ACC_Z, entity_id, 0.0, 0.0]], dtype=np.float64)
        raise ValueError(f"未知的高层动作: {action}")

    def launch_target(self, target_index: int, entity_id: int) -> np.ndarray:
        """Build a launch command for the target assigned by the team rule."""
        return self._target_action(self.ACTION_LAUNCH, int(target_index), entity_id)

    def _target_action(self, action_type: int, target_index: int, entity_id: int) -> np.ndarray:
        if target_index >= len(self.targets):
            return np.empty((0, 4), dtype=np.float64)
        position = self.targets[target_index]["position"]
        return np.asarray([[
            action_type,
            entity_id,
            float(position["lon"]),
            float(position["lat"]),
        ]], dtype=np.float64)
