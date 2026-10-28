# -*- coding: utf-8 -*-#
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from enum import IntEnum


class TypeInfo(IntEnum):
	#qqq
	eee1 = 55555 # eee1
	#基地
	base = 11000001 # 基地
	Anderson_Air_Force_Base = 11000002 # 安德森空军基地
	Apra_Naval_Base = 11000003 # 阿普拉海军基地
	Camp_Zama = 11000004 # 座间兵营
	Aomori_Prefecture = 11000005 # 青森县车力分屯基地
	JASDFK_yougamisaki_Sub_Base = 11000006 # 经岬分屯基地
	Yokosuka_Naval_Base = 11000007 # 横须贺海军基地
	Sasebo_Naval_Base = 11000008 # 佐世保海军基地
	Futenma_Marine_Corps_Air_Station = 11000009 # 普天间海军陆战队航空站
	Naval_Air_Station_Atsugi = 11000010 # 厚木海军航空站
	Iwakuni_Naval_Base = 11000011 # 岩国海军基地
	Yokota_Air_Base = 11000012 # 横田空军基地
	Misawa_Air_Base = 11000013 # 三泽空军基地
	Kadena_Air_Base = 11000014 # 嘉手纳空军基地
	#编组指控
	GroupCommander = 31000001 # 编组指控
	#天眼
	SatelliteA = 9200001 # 天眼1
	SatelliteB = 9200002 # 天眼2
	SatelliteC = 9200003 # 天眼3
	SatelliteD = 9200004 # 天眼4
	SatelliteE = 9200005 # 天眼5
	SatelliteF = 9200006 # 天眼6
	SatelliteG = 9200007 # 天眼7
	SatelliteH = 9200008 # 天眼8
	#车
	Vehicle = 9300001 # 普通车辆
	df_fsc_01 = 6 # 导弹发射车
	#通讯中继器
	CommonMesgRepeater = 41000001 # 通用通信中继器
	#通讯链指控
	CommonMsgNet = 34000001 # 通用通信链
	#舰船
	GeneralShip = 9400001 # 普通舰船
	#干扰指控
	JammerCenter = 32000001 # 干扰指控
	#红外
	CommonInfraredSeeker = 42000001 # 通用红外
	#预警指控
	DetectCommander = 33000001 # 预警指控
	#导引头
	CommonSeeker = 43000001 # 普通导引头
	#城市
	beijing = 11003001 # 北京
	taiwan = 11003002 # 台湾
	tokyo = 11003003 # 东京
	seoul = 11003004 # 首尔
	#进攻武器挂架
	AttackWeaponCarrier = 61000001 # 进攻武器挂架
	#防御武器挂架
	DefendWeaponCarrier = 62000001 # 防御武器挂架
	#飞机
	FWflight = 9100001 # 通用飞机
	#干扰武器挂架
	JammerWeaponCarrier = 63000001 # 干扰武器挂架
	#工事碉堡
	DefenceWorks = 11001001 # 工事碉堡
	#地球同步天眼
	fixorbitSat = 9201001 # 同步天眼
	orbitVariableSat = 9201002 # 通用变轨
	LasetSat = 9201003 # 激光变轨
	NetCatcherSat = 9201004 # 网捕变轨
	PaintSat = 9201005 # 喷涂变轨
	#大数据传输测试模型
	LargeDataTransporter = 0 # 大数据传输测试模型
	#喷枪
	defaultPaintGun = 21003001 # 默认喷枪
	#激光器
	defaultLaser = 21004001 # 默认激光器
	#网捕器
	defaultNetCatcher = 21005001 # 默认网捕器
	#天眼探测器
	defaultSatProber = 45000001 # 默认天眼探测器
	#自爆虫群
	SM6 = 24000001 # 自爆虫群
	#拦截指控
	DefendCommander = 35000001 # 拦截指控
	#进攻指控
	DefaultAttackCommander = 36000001 # 默认进攻指控
	#A星算法
	AStartAlgorithm = 90001001 # A星算法
	#JT导引头
	JTSeeker = 43001001 # JT导引头
	radar = 43001002 # 视野塔探测器
	sar = 43001003 # sar探测器
	infrared = 43001004 # 红外探测器
	#新干扰指控
	JamCenterNew = 37000001 # GR指控
	#舰载有源
	ANSQL = 54000001 # 通用舰载有源
	ShipActive_SuppressionJam = 54000002 # 舰载压制GR
	#舷外有源
	Nulka = 53000001 # 通用舷外有源
	Active_DeceptionJam = 53000003 # 欺骗GR
	#箔条
	MK36 = 51000001 # 通用箔条
	Chaff_Jam = 51000002 # 箔条GR
	Chaff_DiluteJam = 51000003 # 箔条冲淡GR
	Chaff_CentroidJam = 51000004 # 箔条质心GR
	#充气式角反
	MK59 = 52000001 # 通用充气式角反
	CornerReflector_Jam = 52000002 # 角反GR
	#Tle天眼
	G_TEL_1 = 9202001 # G_TEL_1
	G_TEL_2 = 9202002 # G_TEL_2
	G_TEL_3 = 9202003 # G_TEL_3
	G_TEL_4 = 9202004 # G_TEL_4
	G_TEL_5 = 9202005 # G_TEL_5
	G_TEL_6 = 9202006 # G_TEL_6
	G_TEL_7 = 9202007 # G_TEL_7
	G_TEL_8 = 9202008 # G_TEL_8
	G_TEL_9 = 9202009 # G_TEL_9
	G_TEL_10 = 9202010 # G_TEL_10
	G_TEL_11 = 9202011 # G_TEL_11
	G_TEL_12 = 9202012 # G_TEL_12
	G_TEL_13 = 9202013 # G_TEL_13
	G_TEL_14 = 9202014 # G_TEL_14
	G_TEL_15 = 9202015 # G_TEL_15
	G_TEL_16 = 9202016 # G_TEL_16
	G_TEL_17 = 9202017 # G_TEL_17
	G_TEL_18 = 9202018 # G_TEL_18
	G_TEL_19 = 9202019 # G_TEL_19
	G_TEL_20 = 9202020 # G_TEL_20
	G_TEL_21 = 9202021 # G_TEL_21
	G_TEL_22 = 9202022 # G_TEL_22
	G_TEL_23 = 9202023 # G_TEL_23
	G_TEL_24 = 9202024 # G_TEL_24
	G_TEL_25 = 9202025 # G_TEL_25
	G_TEL_26 = 9202026 # G_TEL_26
	G_TEL_27 = 9202027 # G_TEL_27
	G_TEL_28 = 9202028 # G_TEL_28
	G_TEL_29 = 9202029 # G_TEL_29
	G_TEL_30 = 9202030 # G_TEL_30
	G_TEL_31 = 9202031 # G_TEL_31
	G_TEL_32 = 9202032 # G_TEL_32
	G_TEL_33 = 9202033 # G_TEL_33
	G_TEL_34 = 9202034 # G_TEL_34
	G_TEL_35 = 9202035 # G_TEL_35
	G_TEL_36 = 9202036 # G_TEL_36
	G_TEL_37 = 9202037 # G_TEL_37
	G_TEL_38 = 9202038 # G_TEL_38
	G_TEL_39 = 9202039 # G_TEL_39
	G_TEL_40 = 9202040 # G_TEL_40
	G_TEL_41 = 9202041 # G_TEL_41
	G_TEL_42 = 9202042 # G_TEL_42
	G_TEL_43 = 9202043 # G_TEL_43
	G_TEL_44 = 9202044 # G_TEL_44
	G_TEL_45 = 9202045 # G_TEL_45
	G_TEL_46 = 9202046 # G_TEL_46
	G_TEL_47 = 9202047 # G_TEL_47
	G_TEL_48 = 9202048 # G_TEL_48
	G_TEL_49 = 9202049 # G_TEL_49
	#数据测试
	test1 = 10000001001 # 测试1
	#滑翔弹
	lrhw_ez = 23000001 # LRHW
	#滑翔弹（复杂）
	lrhw = 23001001 # LRHW
	#代理模型测试
	agentTest = 91001001 # 代理模型测试
	#导引头代理
	agentSeeker_GS = 91002001 # 导引头
	#LXN代理模型测试
	lxnAgentTest = 91003001 # lxn代理模型
	#挂接测试
	default = 1024768 # 默认
	#GS导引头视野塔代理
	gsSeekerAgent_Radar = 91004001 # GS导引头视野塔代理
	#GS导引头光学代理
	gsSeekerAgent_Optics = 91005001 # GS导引头光学代理
	#云雨雾雪霾环境模型
	environment = 70000001 # 环境信息模型
	#MikuSoft代理测试-Class是Example
	mikuSoftAgent = 91006001 # MikuSoftAgent
	#测试专用
	ceshi_unique = 23440 # 1015测试专用
	#数据型测试弹
	zk = 25000001 # 中科
