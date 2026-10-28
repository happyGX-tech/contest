from dataclasses import dataclass
from datetime import datetime


@dataclass
class WriteConfig:
    """全局写入配置"""
    # 是否启用写入
    enable_config: bool = True  # 是否写配置
    enable_state: bool = True  # 是否写态势
    enable_event: bool = True  # 是否写事件
    enable_ai_action: bool = True  # 是否写AI动作

    # 性能配置
    batch_size: int = 100
    output_dir: str = "results"

    # 是否打印日志
    verbose: bool = True

    # 日志频率
    log_frequency:int = 1

    def is_enabled(self, data_type: str) -> bool:
        """检查某类型数据是否启用写入"""
        if data_type == "config":
            return self.enable_config
        elif data_type == "state":
            return self.enable_state
        elif data_type == "event":
            return self.enable_event
        elif data_type == "aiAction":
            return self.enable_ai_action
        return False


