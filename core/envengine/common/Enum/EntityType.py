# -*- coding: utf-8 -*-#
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from enum import IntEnum


class EntityType(IntEnum):
	pppp = 855 # pppp
	MotionCarrier = 9000 # 运动载体
	FixedWingFlight = 9100 # 飞机
	Satellite = 9200 # 天眼
	FixedSatellite = 9201 # 地球同步天眼
	TleSatellite = 9202 # Tle天眼
	MotionPlatform = 9300 # 车
	GeneralShip = 9400 # 舰船
	oooo = 9888 # ooo
	Entity = 10000 # 实体
	Ceshi = 10005 # 测试专用
	Base = 11000 # 基地
	DefenceWorks = 11001 # 工事碉堡
	CommandCenterSkillModel = 11002 # 指控中心技能模型
	City = 11003 # 城市
	BH_Missile = 12345 # BH弹
	Weapon = 20000 # 武器
	CruiseMissile = 21000 # 毁灭者
	CruiseMissile2 = 21001 # 突击者
	CruiseMissileWRJ = 21002 # 清道夫
	Painter = 21003 # 喷枪
	Laser = 21004 # 激光器
	NetCather = 21005 # 网捕器
	RocketMissile = 21006 # 火箭弹
	BallisticMissile = 22000 # 弹道弹
	GlidingMissile = 23000 # 滑翔弹
	GlidingMissile2 = 23001 # 滑翔弹（复杂）
	BH_missile_1 = 23456 # BH导弹
	Interceptor = 24000 # 自爆虫群
	DataMissile = 25000 # 数据型测试弹
	Commander = 30000 # 指控
	GroupCommander = 31000 # 编组指控
	JammerCommander = 32000 # 干扰指控
	DetectCommander = 33000 # 预警指控
	MsgNet = 34000 # 通讯链指控
	Controller = 34567 # 控制
	DefendCommander = 35000 # 拦截指控
	AttackCommander = 36000 # 进攻指控
	JammerCenterCommander = 37000 # 新干扰指控
	Sensor = 40000 # 传感器
	MsgRepeater = 41000 # 通讯中继器
	InfaredSeeker = 42000 # 红外
	Seeker = 43000 # 导引头
	JTC_Sensor_Seeker = 43001 # JT导引头
	NormalRadar = 44000 # 视野塔
	SatelliteProb = 45000 # 天眼探测器
	Controller_1 = 45678 # 控制_1
	GeneralJammerS = 50000 # 干扰
	ChaffS = 51000 # 箔条
	AngularReversalS = 52000 # 充气式角反
	NarkaS = 53000 # 舷外有源
	ShipboardActive = 54000 # 舰载有源
	qqq = 55555 # qqq
	WeaponCarrier = 60000 # 武器挂架
	AttackWeaponCarrier = 61000 # 进攻武器挂架
	DefendWeaponCarrier = 62000 # 防御武器挂架
	JammerWeaponCarrier = 63000 # 干扰武器挂架
	EnvironmentModel = 70000 # 云雨雾雪霾环境模型
	AlgorithmAgent = 90000 # 算法代理
	AStartAlgorithmAgent = 90001 # A星算法
	SimulatorAgent = 91000 # 模型代理
	AgentTest = 91001 # 代理模型测试
	SeekerAgent_GS = 91002 # 导引头代理
	LXN_AgentTest = 91003 # LXN代理模型测试
	GSSeekerAgent_Radar = 91004 # GS导引头视野塔代理
	GSSeekerAgent_Optics = 91005 # GS导引头光学代理
	MikuSoftAgent = 91006 # MikuSoft代理测试-Class是Example
	LargeDataTransporter = 1000000 # 大数据传输测试模型
	TestData = 1000001 # 数据测试
	LinkTestSimulator = 19201080 # 挂接测试
	LL = 33333333 # 瞎几把剑
