# -*-coding:utf-8 -*-
import math

import numpy as np

from envengine.sdk.base_struct.Basic import Vector3d
from user_agents.base_agent import BaseAgent, AgentType
from policies.red.learning import (
    LearningActionAdapter,
    ObservationEncoder,
    PolicyTransition,
    SharedPolicy,
)

ACTION_SET_ACC_Z = 0  # 设置Z轴加速度
ACTION_USE_SAT = 3  # 使用卫星


class AttackMissileAgent(BaseAgent):
    """
    动作数组格式: np.array([a, b, c, ...], dtype=np.float64)

    横向加速度指令：(加速度值的 1 表示 20G)
    [ACTION_SET_ACC_Z, 实体id, 加速度值, 0]

    使用卫星指令：
    [ACTION_USE_SAT, 实体id, 0, 0]

    Attributes:
        launch_step: 发射时的帧数，为发射为 -1
    """

    def __init__(self, agent_id: int, entity_id: int, init_observation:dict, commander,
                 motion_policy: str = "reactive_evasion", learning_policy: SharedPolicy | None = None,
                 learning_max_steps: int = 1000, hierarchical_learning: bool = False):
        super().__init__(agent_id, entity_id, AgentType.AIRCRAFT, init_observation)
        self.commander = commander
        if motion_policy not in {"straight", "reactive_evasion", "random_masked", "ppo", "mappo"}:
            raise ValueError(f"Unsupported red motion policy: {motion_policy}")
        if motion_policy in {"random_masked", "ppo", "mappo"} and learning_policy is None:
            raise ValueError(f"Motion policy '{motion_policy}' requires a shared learning policy")
        self.motion_policy = motion_policy
        self.learning_policy = learning_policy
        self.hierarchical_learning = bool(hierarchical_learning)
        self.learning_encoder = (
            ObservationEncoder(
                init_observation,
                max_steps=learning_max_steps,
                agent_id=agent_id,
                hierarchical_task_context=self.hierarchical_learning,
            )
            if learning_policy is not None
            else None
        )
        self.learning_adapter = (
            LearningActionAdapter(
                [
                    dict(target, entity_id=int(entity_id))
                    for entity_id, target in init_observation.get("entities", {}).items()
                ]
            )
            if learning_policy is not None
            else None
        )
        self.set_acc_z_z = 0
        self.acc_start_step = 0
        self.launch_step = -1
        self.sat_used = False
        self.latest_observation: dict | None = None
        self._learning_transition: tuple[np.ndarray, int, np.ndarray] | None = None

    def set_observation(self, observation: dict) -> None:
        self.latest_observation = observation

    def get_action(self) -> np.array:
        """生成动作"""
        observation = self.latest_observation
        actions = []  # 存储多个动作

        step = observation["step"]

        self.commander.report(observation)
        launch = self.commander.action_for(self.entity_id, step)
        if launch is not None:
            actions.append(launch)
            self.launch_step = step
            if self.commander.should_use_satellite(self.entity_id):
                actions.append([ACTION_USE_SAT, self.entity_id, 0, 0])
                self.sat_used = True

        if self.learning_policy is not None:
            self._set_acc_z_learning(actions, observation)
        elif self.motion_policy == "reactive_evasion":
            self._set_acc_z_avoid(actions, observation)

        return np.array(actions, dtype=np.float64)

    def _set_acc_z_learning(self, actions: list[list[float]], observation: dict) -> None:
        """Run a shared inference actor on this platform's isolated observation."""

        if self.launch_step < 0:
            return
        assert self.learning_encoder is not None
        assert self.learning_adapter is not None
        assert self.learning_policy is not None
        target_id = self.commander.target_id_for(self.entity_id)
        target_index = next(
            (
                index
                for index, target in enumerate(self.learning_adapter.targets)
                if int(target.get("entity_id", -1)) == target_id
            ),
            None,
        )
        encoded = self.learning_encoder.encode(
            observation,
            launched=True,
            launch_step=self.launch_step,
            satellite_used=self.sat_used,
            maneuver_state=self.set_acc_z_z,
            current_target_index=target_index,
            task_context=(
                self.commander.learning_task_context(self.entity_id)
                if self.hierarchical_learning else None
            ),
        )
        action = self.learning_policy.select_action(
            encoded,
            self.learning_adapter.build_action_mask(
                launched=True,
                satellite_used=self.sat_used,
            ),
        )
        if bool(getattr(self.learning_policy, "training", False)):
            self._learning_transition = (encoded, int(action), self.learning_adapter.build_action_mask(
                launched=True,
                satellite_used=self.sat_used,
            ))
        engine_actions = self.learning_adapter.to_engine_action(int(action), self.entity_id)
        actions.extend(engine_actions.tolist())

    def record_step(self, observation: dict, action, reward: float, info: dict = None):
        """Send an isolated-observation transition to an online learning policy."""

        super().record_step(observation, action, reward, info)
        if self._learning_transition is None or self.learning_policy is None:
            return
        encoded, selected_action, action_mask = self._learning_transition
        self._learning_transition = None
        if not bool(getattr(self.learning_policy, "training", False)):
            return
        assert self.learning_encoder is not None
        assert self.learning_adapter is not None
        target_id = self.commander.target_id_for(self.entity_id)
        target_index = next(
            (
                index
                for index, target in enumerate(self.learning_adapter.targets)
                if int(target.get("entity_id", -1)) == target_id
            ),
            None,
        )
        next_observation = (
            self.learning_encoder.encode(
                observation,
                launched=True,
                launch_step=self.launch_step,
                satellite_used=self.sat_used,
                maneuver_state=self.set_acc_z_z,
                current_target_index=target_index,
                task_context=(
                    self.commander.learning_task_context(self.entity_id)
                    if self.hierarchical_learning else None
                ),
            )
            if observation.get("self")
            else np.zeros_like(encoded)
        )
        global_state, next_global_state = self.learning_policy.get_global_state_context()
        self.learning_policy.observe(
            PolicyTransition(
                agent_id=self.agent_id,
                observation=encoded,
                action=selected_action,
                action_mask=action_mask,
                reward=float(reward),
                next_observation=next_observation,
                done=bool((info or {}).get("done", False) or not observation.get("self")),
                global_state=global_state,
                next_global_state=next_global_state,
            )
        )

    def _set_acc_z_avoid(self, actions, observation: dict):
        """
        躲避拦截弹机动
        :param actions:
        :param observation:
        :return:
        """

        # 当探测到拦截弹时，进行机动规避
        if self.launch_step < 0:
            return

        interceptors = [
            target
            for target in observation["self"].get("detectInfo", {}).values()
            if (target.get("entity_type") if isinstance(target, dict) else getattr(target, "entity_type", None)) == 24000 and
            self.is_self_interceptor(target, observation["self"])
        ]

        if not interceptors:
            return

        self.acc_start_step += 1

        old_status = self.set_acc_z_z
        if self.set_acc_z_z == 0:
            # 发现来袭拦截弹后立即横向机动。
            actions.append([ACTION_SET_ACC_Z, self.entity_id, 1, 0])
            self.set_acc_z_z = 1
        elif self.set_acc_z_z == 1 and self.acc_start_step >= 10:
            # 加速一定帧数后，停止加速
            actions.append([ACTION_SET_ACC_Z, self.entity_id, 0, 0])
            self.set_acc_z_z = 2

        if old_status != self.set_acc_z_z:
            self.acc_start_step = 0

        if self.sat_used:
            return
        actions.append([ACTION_USE_SAT, self.entity_id, 0, 0])
        self.sat_used = True

    def is_self_interceptor(self, target:dict, entity:dict)->bool:
        re_pos = Vector3d(entity["pos_ecf"]["x"] - target.pos_ecf.x, entity["pos_ecf"]["y"] - target.pos_ecf.y, entity["pos_ecf"]["z"] - target.pos_ecf.z)
        re_vel = target.vel_ecf
        return self.get_vector_angle(re_pos, re_vel) < 20

    def reset(self):
        """重置状态"""
        super().reset()
        self.set_acc_z_z = 0
        self.acc_start_step = 0
        self.launch_step = -1
        self.sat_used = False
        self._learning_transition = None

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
