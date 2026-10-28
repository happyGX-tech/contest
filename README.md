# Competition Platform All

本项目用于运行红蓝对抗仿真。入口是 `run.py`，Windows 下建议通过 `run_local.cmd` 调用。运行链路使用标准 CPython 3.11 x64。红方可选择 R0--R9 基线，蓝方可选择四种拦截策略，场景内置九张竞赛地图。

## 快速开始

首次使用时，在项目根目录创建标准 Python 虚拟环境并安装依赖（核心原生模块要求 CPython 3.11 x64）：

```powershell
# 将 py -3.11 替换为本机 CPython 3.11 的 python.exe 路径也可以。
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

随后运行一局无渲染仿真：

```powershell
.\run_local.cmd run --scenario easy/E01 --red-policy r0_random --blue-policy b0_fixed_ratio_random --red-motion-policy reactive_evasion --run-id demo -- --total-rounds 1 --render-mode none
```

运行器会将结果写入：

```text
results/runs/<run-id>/<时间戳>/summary.json
```

控制台同时输出一行 `FINAL_SUMMARY`，其中包含红方发射/存活数、蓝方剩余总血量、各计分目标血量与得分。

`run_local.cmd` 会依次选择 `COMPETITION_PYTHON`、已激活虚拟环境、项目 `.venv` 和系统 `python.exe`。如果需要指定其他 CPython 3.11 x64 运行时：

```powershell
$env:COMPETITION_PYTHON = "C:\path\to\python.exe"
.\run_local.cmd run --help
```

## 常用命令

快速检查策略是否能启动：

```powershell
.\run_local.cmd run --scenario easy/E01 --red-policy r1_priority --blue-policy b2_threat_priority --run-id smoke -- --total-rounds 1 --max-steps 2 --render-mode none
```

对同一场景重复多局，并固定随机性：

```powershell
.\run_local.cmd run --scenario medium/M01 --red-policy r2_static_assignment --blue-policy b3_min_cost_assignment --red-motion-policy straight --seed 20260826 --run-id m01_r2_vs_mincost -- --total-rounds 10 --render-mode none
```

启动本地说明页：

```powershell
.\run_local.cmd ui
```

`--` 后面的参数会原样传递给 core；它不能省略，否则 core 参数会被外层运行器当作未知参数。

## 外层运行参数

这些参数写在 `run` 后、`--` 前。

| 参数 | 可选值/示例 | 默认值 | 作用 |
|---|---|---|---|
| `--scenario` | `easy/E01`、`easy/E02`、`easy/E03`、`medium/M01`--`M03`、`hard/H01`--`H03`、`default`、或场景 JSON 路径 | `default` | 选择场景。竞赛场景若未显式设置 `--max-steps`，会自动使用该地图的正式时限。 |
| `--red-policy` | `r0_random`--`r8_satellite_packages` | `r0_random` | 选择红方目标分配/波次基线。 |
| `--blue-policy` | `b0_fixed_ratio_random`、`b1_nearest_interceptor`、`b2_threat_priority`、`b3_min_cost_assignment` | `b0_fixed_ratio_random` | 选择蓝方拦截分配策略。 |
| `--red-motion-policy` | `straight`、`reactive_evasion`、`random_masked`、`ppo`、`mappo` | `reactive_evasion` | 红方平台发射后的底层飞行方式。`ppo`/`mappo` 为共享学习机动层，需提供模型。 |
| `--red-learning-model` | checkpoint 路径 | 无 | `ppo`/`mappo` 推理时必填；训练时是读取旧权重或写出新权重的位置。 |
| `--red-learning-train` | 开关 | 关闭 | 启用 PPO/MAPPO 在线训练；训练期的全局状态只提供给集中式 Critic。 |
| `--seed` | 整数，例如 `20260826` | 未固定 | 同时设置红方、蓝方和仿真的随机种子，用于可复现实验。 |
| `--run-id` | 任意目录名，例如 `e01_r1` | `manual` | 本次结果在 `results/runs/` 下的归档名称。 |

## 红方 R0--R9

红方开局读取 core 的 `TrainingEnv._get_init_ship_observation()`。函数名含 `ship`，但实际提供完整的初始可用目标目录：`9400` 母巢、`9500` 暗巢、`9600` 哨卫巢，包含真实初始 ID、位置、血量和类型。

| 基线 | 行为 |
|---|---|
| `r0_random` | 在完整初始目录中可复现地随机分配，是随机对照。 |
| `r1_priority` | 基于目标价值、距离、已覆盖数量及平台—目标基础命中能力逐平台贪心分配。 |
| `r2_static_assignment` | 开局只求解一次分配；小规模使用精确分配，大规模使用确定性贪心。 |
| `r3_wave_schedule` | 开局一次性确定三波固定编成；每类平台均匀分入各波，波间隔 40 步，H/M/L 在每波内再延后 0/10/20 步。无后续重规划。 |
| `r4_rolling_rules` | 每 20 步对未发射平台重新分配，每次最多释放当时剩余平台的 25%。只使用静态初始目录，是时间驱动的滚动对照。 |
| `r5_event_rolling` | 仅在己方收到的目标航迹位置变化或平台失联时重分配未发射平台；事件在 5 个仿真步内合并，冷却期满后最多重规划一次。不使用全局态势，也不臆测目标血量。 |
| `r6_frontload_decoy` | 每 5 步根据已观测自爆虫群密度、己方损失和剩余兵力，滚动决定有限 L 前导批次、继续吸收压力或扩大主攻批次；不是独立侦察策略。 |
| `r7_strike_packages` | 继承 R6 的滚动指标和批次控制；主攻阶段将本轮 H/M 配对到同一 9400/9600 目标。编组时间不等于终端同时到达。 |
| `r8_satellite_packages` | 继承 R7；每一轮新生成的 9400/9600 打击包中选择一枚 H 于发射时请求天眼，core 会将该 H 对该类目标的命中率提升至 1。 |
| `r9_hierarchical_learning` | R8 作为合法上层任务/天眼调度器；共享 PPO/MAPPO Actor 按本机隔离观测和上层上下文控制横向机动。必须搭配 `random_masked`、`ppo` 或 `mappo`。 |

R0--R5 的共同发射延后为 H/M/L 的 0/10/20 步。R6--R9 以 5 步为重规划周期：初始探针批次为剩余兵力的 15%；低压力主攻批次为 30%，高压力时收缩到 12%。R7--R9 的 H/M 包内 M 先发、H 延后 3 步。R1--R9 使用 core 当前基础毁伤能力进行匹配：H/M 优先 `9400/9600`，L 优先 `9500`；R0 则不引入该偏好。

R3 与 R4 分别控制“固定多波次编成”和“固定周期滚动”两个维度；R5 再引入基于合法航迹与平台状态变化的事件滚动，详见 [policies/red/README.md](policies/red/README.md)。

## 红方学习机动层

R0--R8 可选配学习机动层；R9 则将该组合固化为正式分层策略。其 Actor 输入是本平台隔离观测，加上上层已分配目标类型、红方压力值和已发射比例，维度为 90；MAPPO 的全局状态仅在显式训练时传给 Critic，不能进入部署决策。R9 权重与此前 85 维 R0--R8 学习权重不兼容，应单独训练和保存。

```powershell
# 无 PyTorch 依赖的接线验证；不代表有效策略。
.\run_local.cmd run --scenario easy/E01 --red-policy r3_wave_schedule --red-motion-policy random_masked --blue-policy b2_threat_priority --run-id learning_smoke -- --total-rounds 1 --render-mode none

# 使用已训练模型推理；运行 Python 必须具备兼容的 PyTorch。
.\run_local.cmd run --scenario easy/E01 --red-policy r4_rolling_rules --red-motion-policy mappo --red-learning-model models\red_mappo.pt --blue-policy b2_threat_priority --run-id mappo_eval -- --total-rounds 10 --render-mode none

# 从头训练；模型写入指定路径。
.\run_local.cmd run --scenario easy/E01 --red-policy r3_wave_schedule --red-motion-policy ppo --red-learning-train --red-learning-model models\red_ppo.pt --run-id ppo_train -- --total-rounds 100 --render-mode none

# R9 分层 MAPPO；必须使用 R9 专用 90 维 checkpoint。
.\run_local.cmd run --scenario easy/E01 --red-policy r9_hierarchical_learning --red-motion-policy mappo --red-learning-train --red-learning-model models\r9_mappo.pt --run-id r9_train -- --total-rounds 100 --render-mode none
```

默认仿真运行时不含 PyTorch。需要训练或推理时，先在项目 `.venv` 中安装与 CPython 3.11 兼容的 PyTorch；模型权重不纳入 Git：

```powershell
.\.venv\Scripts\python.exe -m pip install torch
```

## 蓝方策略

| 策略 | 行为 |
|---|---|
| `b0_fixed_ratio_random` | 对每个已探测来袭平台发射固定数量的自爆虫群，自爆虫群随机选择。 |
| `b1_nearest_interceptor` | 对每个来袭平台优先分配距离最近的可用自爆虫群。 |
| `b2_threat_priority` | 依据来袭平台类型、距受保资产距离和预计到达时间排序。 |
| `b3_min_cost_assignment` | 综合估算拦截时间、目标紧迫度、资产距离及重复覆盖惩罚。 |

蓝方策略只接收视野塔已上报的来袭航迹。它目前没有显式的拦截成功概率模型；core 通过实际飞行轨迹和末端几何关系结算拦截。

## 传递给 core 的参数

这些参数必须写在 `--` 后。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--total-rounds N` | `100` | 连续运行 N 局；多局时分别产生 `summary_round_<N>.json`。 |
| `--max-steps N` | 场景正式时限或 `1000` | 每局最大步数。显式传入时覆盖竞赛场景的自动时限。 |
| `--render-mode none` | `human` | `none` 关闭渲染，适合批量试验；`human` 打开可视化。 |
| `--batch-size N` | `100` | 仿真写入器的批量写入大小。 |
| `--disable-log-color` | 关闭颜色 | 适合将日志重定向到文本文件。 |
| `--verbose` | 关闭 | 输出更详细的写入内容。 |
| `--enable-config` | 关闭 | 写出运行配置。 |
| `--enable-state` | 关闭 | 写出状态数据。 |
| `--enable-event` | 关闭 | 写出事件数据。 |
| `--enable-ai-action` | 关闭 | 写出 AI 动作数据。 |

例如，记录一局完整动作和事件：

```powershell
.\run_local.cmd run --scenario hard/H01 --red-policy r4_rolling_rules --blue-policy b2_threat_priority --seed 7 --run-id h01_trace -- --total-rounds 1 --render-mode none --enable-event --enable-ai-action
```

## 目录说明

```text
core/          仿真引擎、观测、通信、毁伤与拦截结算
policies/      红蓝运行时策略（含 policies/red/learning 学习机动层）
scenarios/     九张竞赛地图、任务信息与计分规则
tools/         离线辅助工具，例如 evaluate.py
experiments/   尚未接入运行链路的预测与离线实验代码
tests/         策略契约测试
results/       本地运行结果
```
