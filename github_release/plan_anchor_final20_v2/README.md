# PlanSpec-v2 final20 计划锚点实验复现包

这个目录对应报告 `plan_anchor_final20_all_12x5_v2`。它把报告使用的计划、关键源码快照、冻结底层模型、逐局分数、逐目标血量和来源哈希放在一起，适合作为 GitHub 发布目录。

## 先回答两个问题

### A00 是规则还是学习策略？

顶层 `A00_balanced` 是确定性规则计划，不是训练得到的策略：

- 部署：`role_layered`，中心锚点 `anchor_forward=0`、`anchor_lateral=0`，正面宽度 `0.67`、纵深 `0.43`；
- H/M 主攻：固定在 `0/120/240` 步释放三波；波内按 `H -> M -> L` 的角色次序和固定间隔编组；
- 公开目标分配：每个公开目标先至少分到 1 枚；H 对 9400、M 对 9600 使用 `3:1` 偏好权重，再按确定性规则补齐；
- L 侦察数：easy/medium/hard 固定为 `32/48/72`；执行 `legal_grid_search_v1`；
- 剩余 L：全部作为预备队。只有所有合法隐藏目标都被发现后，延迟 20 步按隐藏目标分组释放；局内不换靶；
- 天眼选择：固定规则 `fixed_lowest_live_unused_h`。

但整个系统并非“全规则”：高频运动/局部规避使用冻结的学习模型 `local_avoid_ppo_v13_gate_aligned_s3_best.pt`。因此最准确的表述是：**规则式顶层 A00 + 确定性 PlanSpec 执行器 + 冻结学习型底层运动策略**。蓝方是 `b0_fixed_ratio_random`，拦截比例为 2，随机性由公共 seed 控制。

### 报告分数怎么得到？

每局官方分数只看各计分目标的最终受损比例，不含时间奖励：

```text
damage_i = clip((initial_hp_i - final_hp_i) / initial_hp_i, 0, 1)
score = 100 * sum(weight_i * damage_i) / sum(weight_i)
```

目标权重为：9400 母巢/target = `5`，9600 哨卫巢/site = `2`，9500 暗巢/ship = `1`。例如 E01、seed=1 的 A00 得分是 `95.40229885057471`。A00 的报告均分 `92.18081954253798` 是 `20 张图 × 5 个 seed = 100 局`的算术平均；表中 SD 是样本标准差。其他锚点与同一地图、同一 seed 的 A00 做配对差，95% CI 使用固定 seed `20261010` 的 10,000 次 bootstrap。

`scripts/verify_bundle.py` 会从 `data/objective_health.csv` 重新计算全部 1,200 局分数；当前验证应为 0 个公式不一致。

## 目录

```text
data/
  scores.csv                 1,200 局紧凑统计输入
  objective_health.csv       每局每个计分目标的初始/最终血量与权重
plans/                       20 图 × 12 锚点的原始 PlanSpec-v2 token
results/
  report.md                  已核对的 5-seed 报告
  analysis.json              原始完整分析产物
provenance/
  protocol_identity.json     运行时关键文件和场景 SHA-256
  run_index.json             1,200 局原始索引（含原机器绝对路径，仅供审计）
snapshot/
  run.py                     原项目命令行包装器
  tools/                     当时的生成、运行、分析脚本
  policies/.../plan_skill/   与 run_index 哈希一致的 schema/执行器
  policies/.../unified/      编码器、网络、动作定义与共享信息中心
  policies/.../skills/       冻结 local_avoid_s3 的加载与网络代码
  policies/blue/             固定蓝方策略实现
  core/                      与 run_index 哈希一致的入口和 agent
  grid_maps/                 合法部署网格与动作编译
  models/                    与 run_index 哈希一致的冻结底层 checkpoint
  scenarios/                 final20 场景与计分代码
scripts/
  reproduce_analysis.py      不依赖模拟器，重建统计表和报告
  verify_bundle.py           哈希、PlanSpec、矩阵与计分公式检查
  verify_python_snapshot.py  补漏 Python 模块的导入烟测
```

## 2026-10-13 源码补漏

初版确实漏传了运行时模块。导师指出的以下四个文件现已补到原路径：

```text
snapshot/policies/red/learning/unified/info_center.py
snapshot/policies/red/learning/unified/unified_encoder.py
snapshot/policies/red/learning/unified/unified_networks.py
snapshot/policies/red/learning/unified/action_spec.py
```

再次从 `core/main.py`、`unified_missile_agent.py`、PlanSpec executor 和 `local_avoid_s3` 加载入口反向检查后，还补入了 `unified/` 其余源码、红/蓝公共策略、基础 Agent、`grid_maps/`、`RunSummary` 和底层模型加载代码。具体证据等级见 `provenance/recovered_dependencies.md`。

## 一键复核统计

统计复核只需要 Python 3 标准库；Python 快照导入烟测还需要 NumPy 和 PyTorch：

```bash
python scripts/verify_bundle.py
python scripts/verify_python_snapshot.py
python scripts/reproduce_analysis.py
diff -u results/report.md reproduced/report.md
```

预期关键输出：

```text
protocol_files_checked: 25
snapshot_python_files_checked: 71
plans_checked: 240
score_rows_checked: 1200
formula_runs_checked: 1200
A00_runs: 100
A00_mean: 92.18081954253798
```

## 重新运行模拟器

本目录现在包含计划生成/执行、编码器/网络、红蓝策略、部署和底层模型加载所需的 Python 侧快照，但仍不是完整模拟器发行版。完整重跑需要原项目的 `envengine` 及 Linux 原生动态库。实验环境的关键版本是 Python `3.11.16`、NumPy `2.4.6`、PyTorch `2.6.0+cu124`、SciPy `1.17.1`、Requests `2.34.2`。

建议在一个干净的原项目副本中：

1. 用 `snapshot/` 中同路径文件覆盖项目对应文件，例如 `rsync -a snapshot/ /path/to/competition-platform-env-train/`；
2. 将 `plans/` 放到 `experiments/plan_skill/anchors_final20_all_v2/`；
3. 将冻结模型放到 `models/local_avoid_ppo_v13_gate_aligned_s3_best.pt`；
4. 先运行两个 verify 脚本，再执行：

```bash
python tools/run_plan_anchor_all_maps_v2.py \
  --plans experiments/plan_skill/anchors_final20_all_v2 \
  --prefix plan_anchor_12x5_final20_representative_early_v2 \
  --max-jobs 40

python tools/analyze_plan_anchor_all_maps_v2.py \
  --index results/runs/plan_anchor_12x5_final20_representative_early_v2_driver_logs/run_index.json \
  --output results/analysis/plan_anchor_final20_all_12x5_v2
```

`protocol_identity.json` 只证明当时显式记录的关键源码、场景和 checkpoint；它没有给整个模拟器 checkout 做全仓库快照。补漏模块的来源和证据等级单独记录在 `provenance/recovered_dependencies.md`。因此，紧凑数据可以精确复现统计报告；在另一台机器上重跑得到逐局完全相同的轨迹，还要求相同原生库和运行环境。

## 发布注意

- `results/report.md` 已按真实矩阵校正为每图 5 个 seed；若其他副本写成 12，这是报告文字错误，因为数据实际是 20×12×5=1,200 局。
- 原始 1,200 份 summary 和日志约 1 GB，没有放入本目录；逐目标计分证据已压缩到 CSV。
- 公开上传前请确认原项目、场景数据和 checkpoint 的许可；本目录没有擅自添加新许可证。
