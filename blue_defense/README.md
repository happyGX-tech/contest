# 蓝方探测信息与策略接口

`BlueInformationStore` 由每个 `SimulatorFactory` 独立持有，在仿真重置时清空。十种内置策略只接收 `BlueObservation`，目标信息由蓝方雷达探测记录产生。

| 文件与接口 | 职责 |
|---|---|
| `information.py`：`receive_tracks()` | 检查记录身份、时间与有限坐标，合并全部雷达来源，保留各目标的最新记录 |
| `information.py`：`get_track()`、`snapshot()` | 根据记录位置和速度预测当前位置，清除超过30秒的记录，返回不可修改的数据 |
| `information.py`：`update_own_state()` | 接收己方虫巢、自爆虫状态和已经发射或预留的清单 |
| `config.py`：`BlueDefenseConfig` | 配置策略、随机种子、决策间隔及波次参数 |
| `controller.py`：`BlueDefenseController.decide()` | 控制决策间隔，将相同格式的观测交给配置的策略 |
| `base.py`：`ScheduledBluePolicy` | 推进B7—B9的延迟发射计划，释放时使用信息库提供的当前位置预测 |

## 运行过程

1. `RadarModelSimulator.execute_detection()` 按现有20公里范围及飞行阶段条件生成探测记录，坐标与速度保存为独立快照。
2. `DefendCommanderModelSimulator.handel_detect_info()` 校验探测来源属于己方雷达，将记录送入信息库。
3. 指控器每秒调用 `update()`，更新己方状态并读取信息库观测。B0—B9均使用这条输入路径，各自保留目标分配和发射计划算法。
4. 到期计划由 `_current_assignment()` 重新读取信息库中的目标预测。目标记录失效、己方自爆虫已经发射或被摧毁时，取消对应计划。发射仍需通过原有距离和角度条件。
5. `InterceptorSimulator.set_target_ecf()` 使用信息库更新制导；`GeneralShipSimulator.detect_attacker()` 和 `InterceptFieldSimulator.detect_attacker()` 使用信息库中距离自身20公里以内的最近目标进行避让。

探测信息可以在离开雷达范围后继续用于最多30秒的预测。新的位置和速度必须等待新的合法探测。信息库不通过目标真实存活状态提前删除记录。记录失效后，自爆虫原生模型继续使用最后一次设置的目标；母巢和哨卫巢在没有适用记录时返回初始位置。

信息库为进程内数据接口，当前平台没有隔离任意第三方Python代码的安全沙箱。

## 本地验证

原生模型使用CPython 3.11 x64。运行 `python -m unittest discover -s tests -p test_blue_information.py -v`，测试会加载真实E01场景、运行原生红方与自爆虫模型，并通过真实雷达产生记录。传感器测试中将真实红方模型放到雷达附近，该过程用于接口验证，不计入比赛成绩。测试摘要写入忽略提交的 `.blue_validation/`。
