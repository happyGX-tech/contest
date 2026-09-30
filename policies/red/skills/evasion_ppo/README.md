# S3 中层规避 PPO

S3 是与 R9 上层目标分配解耦的单弹规避技能。R9 继续负责部署、目标分配、发射和天眼；S3 只在导弹发射后输出 `LEFT / STRAIGHT / RIGHT` 三个横向残差动作，不修改目标和全局任务。

## 当前发布版本

- 模型：`models/local_avoid_ppo_v11_trusted_mission_s3_best.pt`
- 算法：独立 Actor、集中训练 Critic 的 PPO
- Actor 输入：33 维合法局部观测
- 动作：`-20G / 0 / +20G`
- 支持弹种：H、M、L
- 威胁输入：本机探测、通信簇共享航迹和天眼权威航迹
- 无危险威胁时：动作掩码强制直飞
- 有危险威胁时：根据距离、闭合速度、CPA 和目标方向执行短时规避

天眼信息由平台融合层判定为最高优先级。S3 保留航迹来源，但不会用本地观测覆盖天眼结果。

## 运行

```bash
python run.py run \
  --scenario final20/easy/E01 \
  --blue-policy b0_fixed_ratio_random \
  --red-policy r9_hierarchical_learning \
  --red-motion-policy local_avoid_s3 \
  --red-learning-model models/local_avoid_ppo_v11_trusted_mission_s3_best.pt \
  --seed 701 \
  --run-id s3_v11_e01_b0 \
  -- --total-rounds 1 --render-mode none
```

`local_avoid_s3` 是推理策略，不能通过完整仿真的在线 `--red-learning-train` 参数继续训练。继续训练应使用独立局部环境入口。

## 继续训练

仓库保留 V10 DAgger warm start 和 V11 任务型训练入口：

```bash
python tools/train_local_avoid_ppo_v11_trusted_mission.py \
  --resume models/local_avoid_ppo_v10_dagger_s3.pt \
  --device cuda
```

训练目标不是单纯生存，而是同时优化规避、持续向原目标推进、快速恢复航向、到达目标以及额外航程。

## 最新完整仿真验收

协议：Final20 全部 20 张地图、B0-B9 全部 10 种蓝方策略、固定种子 701、R9 上层不变；直飞和 S3 各 200 局，共 400 局，失败 0 局。

| 指标 | 直飞 | S3 V11 | 差值 |
|---|---:|---:|---:|
| 平均得分 | 42.74 | 62.97 | +20.24 |
| 任务有效躲避率 | 19.79% | 38.28% | +18.49pp |
| 到达率 | 63.78% | 68.00% | +4.22pp |
| 发射命中率 | 10.46% | 14.17% | +3.71pp |

200 个地图与防守策略组合中，S3 得分 176 胜、5 平、19 负。完整表格和逐局原始数据位于：

`results/evaluations_latest_s3/final20_all_b0_b9_v11_s3_vs_straight/`

这是单固定种子的全覆盖验收，不是多种子置信区间。S3 V11 是当前任务型中层 baseline，不应称为 99% 躲避模型。
