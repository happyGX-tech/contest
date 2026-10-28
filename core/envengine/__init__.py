# envengine/__init__.py
"""
EnvEngine - Multi-Agent Reinforcement Learning Simulation Engine

Usage:
    from envengine import TrainingEnv, Profile
"""

# 核心类
from .environment.training_env import TrainingEnv
from .engine.engine import Engine
from .simulator.interfaces.isimulator import ISimulator

# 工具类
from .sdk.base_struct.profile.profile import Profile

# 版本信息
__version__ = "0.1.0"
__all__ = [
    "TrainingEnv",
    "Engine",
    "ISimulator",
    "Profile",
]
