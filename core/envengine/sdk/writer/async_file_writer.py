# -*-coding:utf-8 -*-
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import logging
from typing import List, Any, Optional
from threading import Lock

# import msgpack

from envengine.sdk.QZPython.customized_binary import AutoBinary

from envengine.sdk.writer.write_config import WriteConfig


class AsyncJsonFileWriter:
    """
    单文件异步写入器（单例模式）
    """

    _instance = None
    _lock = Lock()  # 线程安全锁

    def __new__(cls, config: Optional[WriteConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[WriteConfig] = None):
        """
        参数:
            output_dir: 输出目录
            batch_size: 每多少条数据写一次文件
        """
        # 防止重复初始化
        if self._initialized:
            return

        # 使用传入的配置或默认配置
        self.config = config or WriteConfig()

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.output_dir = Path(self.config.output_dir)
        self.batch_size = self.config.batch_size
        self.buffer = []
        self._current_filename = None  # 当前使用的文件名

        # 确保目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._initialized = True
        logging.info(f"初始化写入器成功，目录: {self.output_dir}")

    def _write_batch(self, batch_data: List[Any], filename: Path):
        """在子线程中执行的实际写入操作"""
        if not batch_data:
            return
        try:
            # 格式化数据（每行一条JSON）
            lines = [
                json.dumps(
                    item,
                    ensure_ascii=False,
                    default=str,  # 解决所有自定义类型
                    indent=None
                )
                for item in batch_data
            ]
            content = "\n".join(lines) + "\n"

            # 追加写入同一个文件
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(content)
            if self.config.verbose:
                logging.info(f"[写入完成] {filename} (+{len(batch_data)}条)")
        except Exception as e:
            logging.error(f"[写入失败] {filename}: {e}")

    def add(self, data: Any, data_type: str, filename: str) -> bool:
        """
        添加一条数据到写入队列
        参数:
            data: 要写入的数据
            filename: 文件名（相对于dir_name）
        返回: 是否触发了写入
        """
        # 检查是否启用
        if not self.config.is_enabled(data_type):
            return False
        if not filename:
            logging.error("[写入失败] 未指定文件名")
            return False

        # 构建完整文件路径
        full_path = Path(self.output_dir, filename)
        if filename:
            self._current_filename = full_path

        self.buffer.append(data)

        # 检查是否需要刷新
        if len(self.buffer) >= self.batch_size:
            self.flush()
            return True
        return False

    def flush(self):
        """立即提交缓冲区数据写入"""
        if not self.buffer or not self._current_filename:
            return

        # 注意：_write_batch 需要接收文件名参数
        self.executor.submit(self._write_batch, self.buffer.copy(), self._current_filename)
        self.buffer = []

    def close(self):
        """关闭写入器，等待所有数据写入完成"""
        self.flush()
        self.executor.shutdown(wait=True)
        if self.config.verbose:
            logging.info("写入器已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncBinaryFileWriter:
    """
    单文件异步写入器（单例模式）
    """

    _instance = None
    _lock = Lock()  # 线程安全锁

    def __new__(cls, config: Optional[WriteConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[WriteConfig] = None):
        """
        参数:
            output_dir: 输出目录
            batch_size: 每多少条数据写一次文件
        """
        # 防止重复初始化
        if self._initialized:
            return

        # 使用传入的配置或默认配置
        self.config = config or WriteConfig()

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.output_dir = Path(self.config.output_dir)
        self.batch_size = self.config.batch_size
        self.buffer = []
        self._current_filename = None  # 当前使用的文件名

        # 确保目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._initialized = True
        logging.info(f"初始化写入器成功，目录: {self.output_dir}")

    # def _write_batch(self, batch_data: List[Any], filename: Path):
    #     """在子线程中执行的实际写入操作"""
    #     if not batch_data:
    #         return
    #     try:
    #         # 格式化数据（每行一条JSON）
    #         lines = [
    #             json.dumps(
    #                 item,
    #                 ensure_ascii=False,
    #                 default=str,  # 解决所有自定义类型
    #                 indent=None
    #             )
    #             for item in batch_data
    #         ]
    #         content = "\n".join(lines) + "\n"
    #
    #         # 追加写入同一个文件
    #         with open(filename, 'a', encoding='utf-8') as f:
    #             f.write(content)
    #         if self.config.verbose:
    #             logging.info(f"[写入完成] {filename} (+{len(batch_data)}条)")
    #     except Exception as e:
    #         logging.error(f"[写入失败] {filename}: {e}")

    @staticmethod
    def msgpack_serializer(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return obj.__dict__
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"不支持的类型: {type(obj)}")

    def _write_batch(self, batch_data: List[Any], filename: Path):
        if not batch_data:
            return
        try:
            content = b''.join(batch_data)
            with open(filename, 'ab') as f:
                f.write(content)
        except Exception as e:
            logging.error(f"[写入失败] {filename}: {e}")

    def add(self, data: Any, data_type: str, filename: str) -> bool:
        """
        添加一条数据到写入队列
        参数:
            data: 要写入的数据
            filename: 文件名（相对于dir_name）
        返回: 是否触发了写入
        """
        # 检查是否启用
        if not self.config.is_enabled(data_type):
            return False
        if not filename:
            logging.error("[写入失败] 未指定文件名")
            return False

        # 构建完整文件路径
        full_path = Path(self.output_dir, filename)
        if filename:
            self._current_filename = full_path

        # self.buffer.append(msgpack.packb(data, default=self.msgpack_serializer))

        # 检查是否需要刷新
        if len(self.buffer) >= self.batch_size:
            self.flush()
            return True
        return False

    def flush(self):
        """立即提交缓冲区数据写入"""
        if not self.buffer or not self._current_filename:
            return

        # 注意：_write_batch 需要接收文件名参数
        self.executor.submit(self._write_batch, self.buffer.copy(), self._current_filename)
        self.buffer = []

    def close(self):
        """关闭写入器，等待所有数据写入完成"""
        self.flush()
        self.executor.shutdown(wait=True)
        if self.config.verbose:
            logging.info("写入器已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


_config: Optional[WriteConfig] = None


def init_writer(config: WriteConfig):
    """初始化写入器（在程序启动时调用一次）"""
    global _config
    _config = config
    return AsyncJsonFileWriter(config)


def get_writer() -> AsyncJsonFileWriter:
    """获取写入器单例"""
    return AsyncJsonFileWriter()


# ========== 便捷写入函数==========
def write_config(data: dict, filename: str):
    """写配置数据"""
    data["data_type"] = "config"
    writer = get_writer()
    writer.add(data, "config", filename)


def write_state(data: dict, filename: str):
    """写态势数据"""

    if _config and _config.log_frequency > 1:
        if data["step"] % _config.log_frequency != 0:
            return

    data["data_type"] = "state"
    writer = get_writer()
    writer.add(data, "state", filename)


def write_event(data: dict, filename: Optional[str] = None):
    """写事件数据"""
    data["data_type"] = "event"
    writer = get_writer()
    writer.add(data, "event", filename)


def write_ai_action(data: dict, filename: str):
    """写AI动作数据"""
    if data is None:
        return
    data["data_type"] = "aiAction"
    writer = get_writer()
    writer.add(data, "aiAction", filename)


def write_immediately():
    """立即提交所有缓冲区数据写入"""
    writer = get_writer()
    writer.flush()
