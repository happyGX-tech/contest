# -*- coding: utf-8 -*-#
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from enum import IntEnum


class SimmerCommandType(IntEnum):
    """
    Attributes:
        ATTACK_COMMANDER_START: 进攻指控装订
    """
    MISSILE_LAUNCH = 200  # 进攻弹发射
    INTERCEPTOR_LAUNCH = 201  # 自爆虫群发射
    MISSILE_GUIDANCE = 202  # 进攻弹制导
    INTERCEPTOR_GUIDANCE = 203  # 自爆虫群制导
    MISSILE_CHANGE_ROUTE = 204  # 进攻弹修改航迹
    LASER_EMIT = 205  # 发射激光
    PAINT_EMIT = 206  # 发射喷涂
    NET_EMIT = 207  # 发射捕网
    MOVE = 300  # 开始移动
    PATROL_ALONG_ROUTE = 301  # 按路径点巡逻
    PATROL_CIRCLE = 302  # 大圆巡逻
    CHANGE_ORBIT = 303  # 天眼变轨
    GROUP_MOVE = 310  # 编组移动
    GROUP_PATROL_ROUTE = 311  # 编组按路径巡逻
    GROUP_PATROL_CIRCLE = 312  # 编组大圆巡逻
    GROUP = 400  # 编组
    CHANGE_GROUP = 401  # 修改编组
    DISSOLUTION = 402  # 解散编组
    EXEC_GROUP = 410  # 单机执行编组指令
    START_DETECT = 500  # 启动探测
    DAMAGE = 600  # 伤害
    DESTROY = 700  # 摧毁
    EASY_EVENT_SEND = 800  # 简易事件转发
    MESSAGE_CHAIN_CREATE = 802  # 通讯链创建
    MESSAGE_CHAIN_DESTROY = 803  # 删除通讯链路
    MESSAGE_CHAIN_ACTIVE = 804  # 通讯链路激活
    HEALTH_STATE_CHANGE = 900  # 健康状态切换
    POWER_STATE_CHANGE = 901  # 电源状态切换
    b = 999  # b
    INTERCEPTOR_INFO = 1000  # 拦截指控指令
    CE_SHI = 1005  # 测试指令勿删
    RELEASE_JAM = 1100  # 释放干扰
    GR_RELEASE = 1101  # GR释放指令
    CHANGE_JAMPOS = 1103  # 干扰位置改变
    RETRIEVE_CHILDREN = 1200  # 回收子物体
    RELEASE_CHILDREN = 1201  # 释放子物体
    SET_RELATED_ENTITY = 1202  # 设置相关实体
    DETECT_ENTITY = 1300  # 预警指控信息
    ENGINE_SPEED_SET = 1400  # 引擎倍速切换
    SCENE_VIEW_CHANGE = 1401  # 视景视角切换
    SCENE_VIEW_FOCUS = 1402  # 视景聚焦
    CHILD_SYNC = 1403  # 子物体位置速度同步
    JAMMMER_COMMANDER_START = 2100  # 干扰指控开始工作
    GR_COMMAND_START = 2101  # 干扰指控开始工作指令
    DEFEND_COMMANDER_START = 2200  # 防御指控装订
    TRANS_LAUNCH_WINDOW = 2202  # 传递拦截窗口
    ATTACK_COMMANDER_START = 2300  # 进攻指控装订
    EnvironmentInfoSet = 3000  # 环境信息装订
    EntityInfoSet = 3001  # 实体信息装订
    TransmissionInfoSet = 3002  # 传输信息装订
    InterfereInfoSet = 3003  # 干扰信息装订
    TargetCharacterInfo = 3004  # 目标特性指令装订
    ChangePose = 3005  # 改变实体位姿
    DETECT_STATUS_UPDATE = 3006  # 探测器探测状态更新
    SET_DESIRED_ACC_Z = 3007  # 设置飞行器侧向加速度
    DEPLOY = 3008  # 部署
    DEPLOY_COMPLETED = 3009  # 部署完成
    RADAR_DETECT_STATUS_UPDATE = 3010  # 视野塔探测状态更新
    EXECUTE_DETECTION = 3011  # 执行探测
    SET_DESIRED_VEL_X = 3012  # 设置飞行器轴向速度
    EXECUTE_SATELLITE_DETECTION = 3013  # 请求天眼探测
    CHANGE_MISSILE_TARGET = 3014  # 修改导弹目标点


