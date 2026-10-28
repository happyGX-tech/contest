# -*-coding:utf-8 -*-
import time
from json import JSONDecodeError
from typing import Callable, Final, Never
from envengine.sdk.Transport import MQTT
from envengine.sdk.Struct import Invoker, LogicInput, Negotiation


class TOPICS:
    class SUBSCRIBE:
        ONLINE: Final[str] = "JT_APP/online/e2a/"

        class STATE:
            SYNC: Final[str] = "JT_APP/state/sync/"
            SYNC_STREAM: Final[str] = "JT_APP/state/sync_stream/"
            ASYNC: Final[str] = "JT_APP/state/sync_stream/"

        ROUND: Final[str] = "JT_APP/round/"

    class PUBLISH:
        ONLINE: Final[str] = "JT_APP/online/a2e/"

        class COMMAND:
            SYNC: Final[str] = "JT_APP/command/sync/"
            SYNC_STREAM: Final[str] = "JT_APP/command/sync_stream/"
            ASYNC: Final[str] = "JT_APP/command/async/"


class Environment:
    def __init__(self, class_name: str, domain: str, sub_domain: str, mqtt_ip: str, mqtt_port: int):
        r"""
        连接环境
        :param class_name 类型名:
        :param domain: 域
        :param sub_domain: 子域
        :param mqtt_ip: MQTT host
        :param mqtt_port: MQTT port
        """
        """回调"""
        self.__round_count__: int = 0
        """轮次数"""
        self.__prefix__: Final[str] = class_name
        """TOPIC前缀"""
        self.__postfix__: Final[str] = domain + "_" + sub_domain
        """TOPIC后缀"""
        self.__online__: bool = False
        """上线状态"""
        name: str = "PYTHON_" + self.__prefix__ + "_" + self.__postfix__
        """名称"""
        self.__mqtt__ = MQTT(name, mqtt_ip, mqtt_port, self.__on_subscribe__)
        """mqtt"""
        self.__state__: dict[str, Invoker] = {
            "SYNC": Invoker(self.__mqtt__, False, False),
            "SYNC_STREAM": Invoker(self.__mqtt__, True, False),
            "ASYNC": Invoker(self.__mqtt__, False, True)
        }
        """接收内容"""
        self.__subscribe__()

    def run(self, reset: Callable[[int], None], logic: Callable[[LogicInput], str]) -> Never:
        """
        运行
        :param reset: 重置函数
        :param logic: 逻辑函数
        :return: Never
        """
        if not isinstance(reset, Callable):
            raise "Except in run: reset must be a Callable[[int], None]"
        if not isinstance(logic, Callable):
            raise "Except in run: logic must be a Callable[[LogicInput], str]"
        while True:
            self.__round_count__ += 1
            reset(self.__round_count__)
            self.__waiting_online__()
            while self.__online__:
                command_topic = TOPICS.PUBLISH.COMMAND
                self.__state__["SYNC"].invoke(self.__real_topic__(command_topic.__dict__["SYNC"]), logic)
                self.__state__["SYNC_STREAM"].invoke(self.__real_topic__(command_topic.__dict__["SYNC_STREAM"]), logic)
                self.__state__["ASYNC"].invoke(self.__real_topic__(command_topic.__dict__["ASYNC"]), logic)

    def __subscribe__(self):
        r"""
        启动订阅
        :return: NoReturn
        """
        self.__mqtt__.subscribe(self.__real_topic__(TOPICS.SUBSCRIBE.ONLINE), 0)
        self.__mqtt__.subscribe(self.__real_topic__(TOPICS.SUBSCRIBE.STATE.SYNC), 0)
        self.__mqtt__.subscribe(self.__real_topic__(TOPICS.SUBSCRIBE.STATE.SYNC_STREAM), 0)
        self.__mqtt__.subscribe(self.__real_topic__(TOPICS.SUBSCRIBE.STATE.ASYNC), 0)
        self.__mqtt__.subscribe(self.__real_topic__(TOPICS.SUBSCRIBE.ROUND), 0)

    def __waiting_online__(self) -> None:
        """
        等待上线
        :return: None
        """
        print("Waiting online...")
        if not self.__online__:
            while not self.__online__:
                self.__mqtt__.publish(self.__real_topic__(TOPICS.PUBLISH.ONLINE),
                                      Negotiation.current().to_json(), 0)
                time.sleep(1)
        print("Online OK!")

    def __on_subscribe__(self, topic: str, payload: str) -> None:
        """
        订阅到达事件
        :param topic: 主题
        :param payload: 负载
        :return: None
        """
        if topic == self.__real_topic__(TOPICS.SUBSCRIBE.ONLINE):
            self.__online_received__(payload)
            return
        if topic == self.__real_topic__(TOPICS.SUBSCRIBE.ROUND):
            self.__online__ = False
            return
        if topic == self.__real_topic__(TOPICS.SUBSCRIBE.STATE.SYNC):
            self.__state__["SYNC"].save_payload(payload)
            return
        if topic == self.__real_topic__(TOPICS.SUBSCRIBE.STATE.SYNC_STREAM):
            self.__state__["SYNC_STREAM"].save_payload(payload)
            return
        if topic == self.__real_topic__(TOPICS.SUBSCRIBE.STATE.ASYNC):
            self.__state__["ASYNC"].save_payload(payload)
            return

    def __real_topic__(self, topic: str):
        r"""
        真实TOPIC
        :param topic: 中间TOPIC
        :return: 完整TOPIC
        """
        return topic + self.__prefix__ + "/" + self.__postfix__

    def __online_received__(self, payload: str):
        """
        收到上线信息
        """
        if self.__online__:
            return
        self.__online__ = True
        try:
            negotiation = Negotiation.from_json(payload)
            if negotiation.version != Negotiation.current().version:
                print("[ERROR]\n[ONLINE]\n",
                      "Version is not matched! ",
                      "    Instance: ", self.__prefix__, "\n",
                      "    Local: ", Negotiation.current().version, "\n",
                      "    Remote: ", negotiation.version)
        except KeyError:
            print("[ERROR]\n[ONLINE]\n",
                  "Version is not matched! ",
                  "    Instance: ", self.__prefix__, "\n",
                  "    Local: ", Negotiation.current().version, "\n",
                  "    Remote: UNKNOWN")
        except JSONDecodeError:
            print("[ERROR]\n[ONLINE]\n",
                  "Version is not matched! ",
                  "    Instance: ", self.__prefix__, "\n",
                  "    Local: ", Negotiation.current().version, "\n",
                  "    Remote: UNKNOWN")

