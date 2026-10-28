# PPO Evasion Skill

这是一个与上层目标分配解耦的单弹规避技能。R9 上层仅负责部署、目标分配和发射；技能只输出横向 `-20G / 0 / +20G` 三个动作。

## 约束

- Actor 只读取本弹状态和 `detect_from == 本弹 ID` 的拦截弹航迹。
- 来袭几何统一投影到本弹局部坐标，并显式编码预计最近交会距离（CPA）。
- V3 只把 `CPA <= 15 km`、`TGO <= 120 s` 且闭合速度不低于 `50 m/s` 的航迹视为入射威胁，避免被其他导弹附近的过航拦截弹污染奖励。
- V4 对每条本机经验生成左右镜像经验并交换动作标签，抑制策略退化成“始终向同一侧转”；镜像样本使用独立轨迹 ID，不会串接不同导弹的 GAE。
- 不读取其他红弹状态、通信群信息、团队得分或全局 Critic。
- 技能不会请求卫星，因此执行阶段不存在导弹间航迹共享。
- 所有导弹可以共享同一套网络参数，但每枚弹独立推理、独立轨迹、独立奖励。
- 当前平台只有高性能弹能自行探测拦截弹。因此训练基准默认移除中低性能弹，避免把“无传感器输入”误当作 PPO 能解决的问题。

## 指标

一次来袭事件从首次出现有效本机拦截弹航迹开始。连续 3 步脱离威胁判为成功；在拦截弹最近距离 2 km 内消失判为失败。最终门槛要求滚动 2000 次事件中：

1. 至少完成 1000 次事件；
2. 躲避率不低于 99%；
3. 95% Wilson 置信下界不低于 99%。

训练和评估会输出 `EVASION_METRICS` JSON。只有独立评估满足门槛，才能宣称达到 99%。

## 入口

```bash
python tools/train_evasion_ppo.py --rounds 100
python tools/prepare_evasion_v3_checkpoint.py \
  models/ppo_evasion_v2_final20_e01_best.pt \
  models/ppo_evasion_v3_final20_e01.pt
python tools/prepare_evasion_v4_checkpoint.py \
  models/ppo_evasion_v3_final20_e01.pt \
  models/ppo_evasion_v4_final20_e01.pt
python tools/evaluate_evasion_skill.py \
  --model models/ppo_evasion_final20_e01_best.pt \
  --seed-count 20 --rounds-per-seed 2
```

## 局部四阶段训练

`LocalAvoidEnv` 是独立于 R9 上层和完整仿真 SDK 的中层规避训练环境，对应《人族远征_分层决策与中层PPO规避训练方案》的第一阶段：

- Stage 0：无自爆虫，先学会稳定直飞；
- Stage 1：单个正面来袭自爆虫；
- Stage 2：单个任意方向来袭自爆虫；
- Stage 3：双弹同时来袭、左右夹击或同向延迟 `5--15` 步来袭；
- Actor 使用本机局部观测，Critic 额外使用训练期特权状态；
- 动作为 `LEFT / STRAIGHT / RIGHT`，只输出相对上层航向的残差机动；
- 速度、转弯率、碰撞半径、课程长度和全部奖励权重均可在配置中调整。

```bash
python tools/train_local_avoid_ppo.py --smoke --device cpu
python tools/train_local_avoid_ppo.py --device cuda
python tools/evaluate_local_avoid_ppo.py \
  --model models/local_avoid_ppo_final.pt --episodes 2000 --stages 2
```

V2 从 Stage 1 checkpoint 继续，把任意方向威胁拆成三个渐进阶段，并启用危险交会筛选、累积转向、CPA 风险势函数奖励和 Stage 0/1 混合回放：

```bash
python tools/train_local_avoid_ppo_v2.py --device cuda
```

Stage 3 从 Stage 2 最优点继续，默认混合回放 `30%` Stage 2 和 `70%`
Stage 3。训练入口支持只运行指定课程，并可选左右镜像一致性损失：

```bash
python tools/train_local_avoid_ppo_v4_s3.py \
  --resume models/local_avoid_ppo_v3_stage2_best.pt \
  --model models/local_avoid_ppo_v4_stage3.pt \
  --device cuda

python tools/evaluate_local_avoid_ppo.py \
  --model models/local_avoid_ppo_v4_stage3_best.pt \
  --episodes 2000 --stages 2,3 --seed 20269901 --device cuda
```

当前 Stage 3 选择模型是 `models/local_avoid_ppo_stage3_selected.pt`，内容与 V9
恢复性微调 best checkpoint 相同。训练先混合正常 S3 与同侧双弹 hard case，再冻结
V4 的观测归一化统计，并以 `90%` Stage 2 回放做单次恢复 update。固定 seed
`20269901` 的 2000 局结果如下：

| 模型 | Stage 2 躲避率 | Stage 3 躲避率 | Stage 3 命中率 |
|---|---:|---:|---:|
| V4 baseline | `98.68%` | `79.35%` | `19.35%` |
| V9 selected | `98.14%` | `83.30%` | `16.00%` |

V9 在维持 Stage 2 超过 `98%` 的同时，将 Stage 3 提升约 `3.95` 个百分点。结果见
`results/local_avoid_ppo/evaluation_v9_stage2_recovery_best_2000.json`。当前仍不能宣称
Stage 3 达到 `99%`。

固定 seed `20271001` 的 3000 局分模式诊断显示：同时来袭从 `74.88%` 提升到
`78.60%`，夹击从 `91.20%` 提升到 `93.65%`，延迟接续从 `76.01%` 提升到
`79.63%`，同侧双弹从 `74.21%` 提升到 `78.71%`。代价是 Stage 3 平均额外航程
从约 `14.1 km` 增至 `17.7 km`。诊断结果见
`results/local_avoid_ppo/stage3_diagnosis_v9_selected_3000.json`。

该环境用于先验证规避技能的可学习性，不替代完整仿真验收。达到局部环境指标后，还需接回真实观测、动力学和拦截弹模型做迁移训练与固定种子评估。

## 原生动力学微型遭遇环境

`NativeEncounterEnv` 是局部二维环境和完整对局之间的中间层。它绕过部署、上层规划、蓝方指挥、渲染和逐步日志，只保留一枚 H/M/L 红弹、零到两枚拦截弹以及平台原生导弹和 SM6 模型。第一版使用 oracle 拦截航迹，目标是先验证原生动力学最小闭环；探测链路和并行训练将在后续阶段接入。

宿主系统的 glibc 版本不足以加载最新版原生库，应在项目使用的 Python 3.11 容器中运行：

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -v /home/ubuntu/yunpan/miniconda3/envs/competition:/opt/competition-env:ro \
  -w /workspace \
  -e PYTHONPATH=/workspace/core:/workspace \
  -e LD_LIBRARY_PATH=/opt/competition-env/lib:/workspace/core:/workspace/core/envengine/simulator/models/CompCruiseMissileModel \
  docker.m.daocloud.io/library/python:3.11-slim-trixie \
  /opt/competition-env/bin/python tools/smoke_native_encounter.py --steps 20
```

## V15 真实失败轨迹微调

V15 从 M03/B0、M03/B7 的真实失败轨迹构造短时域专家标签，在 V14 GRU
策略上做低学习率行为克隆，并使用 KL 锚限制策略漂移。最终在 E01/M03、
B0/B3/B7/B9、固定种子 709--711 的 24 局真实仿真门控中，V15 谨慎版平均分
为 `74.973`，低于 V14 的 `76.285`；躲避率、到达率和每发有效命中率也均未
超过 V14。因此 V15 不覆盖任何现有 selected checkpoint。

这说明短时域几何避险标签不能直接代表整局收益。完整数据、训练口径和分场景结果见
`results/real_trajectories_v15/V15_REAL_EXPERT_REPORT.md`。

## 当前验收结果

当前选择模型为 `models/ppo_evasion_selected.pt`，内容与 V3 final checkpoint 相同。V4 的左右镜像增强在独立评估中没有超过 V3，因此未被选中。

V3 final 在 B0、全新固定 seed 401--418 上完成 1080 次入射事件：成功 1025 次、失败 55 次，躲避率 `94.91%`，95% Wilson 下界 `93.43%`。结果见 `results/evasion_ppo/eval_v3_final_heldout_1000.json`。

这还没有达到 99% 目标。训练中出现的单局 100% 不能替代大样本验收；在指标达到门槛前，模型只应称为当前候选，不应称为 99% 躲避模型。

## V10 完整仿真多地图训练

V10 不再使用微型仿真的终局代理。每局结束后，训练器以
`MissionOutcomeTracker` 的逐弹结果覆盖 `arrived_proxy/lost_proxy`：只有真实
`arrived/effective_hit` 轨迹保留归一化目标距离减少量，真实
`intercepted/lost` 轨迹的整段回报为零。正式训练拒绝任何未成功回写真实终局
的轨迹，并排除时间截断轨迹。

地图固定划分为：

- 训练：E01--E05、M01--M05、H01--H06，共 16 图；
- 严格 heldout：E06、M06、H07、H08，共 4 图；
- 每轮按 E/M/H 均衡轮转，H/M/L 导弹 transition 再做中位数重采样；
- 环境固定为 `static + B0 + gate=none`，PPO learner 使用 GPU；
- JSONL 和 TensorBoard 同时记录真实到达率、规避率、命中率和 PPO 指标。

正式训练完成后，必须在 4 张 heldout 地图和固定独立种子
`11901/11902/11903` 上，将 V10 与输入模型和纯直飞做同协议比较。训练 loss 或
训练图上的提升不能单独作为模型入选依据。
