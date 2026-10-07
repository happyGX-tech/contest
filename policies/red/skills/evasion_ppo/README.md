# S3 中层规避 PPO

S3 是与 R9 上层目标分配解耦的单弹规避技能。R9 继续负责部署、目标分配、
发射和天眼；S3 只在导弹发射后输出 `LEFT / STRAIGHT / RIGHT` 三个横向残差
动作，不修改目标和全局任务。

## 当前发布版本

- 模型：`models/local_avoid_ppo_v13_gate_aligned_s3_best.pt`
- 算法：独立 Actor、集中训练 Critic 的前馈 PPO
- Actor 输入：33 维合法局部观测
- 动作：`-20G / 0 / +20G`
- 支持弹种：H、M、L
- 威胁输入：本机探测、通信簇共享航迹和天眼权威航迹
- 门控：`mission_v12`，只在紧急威胁窗口交给 PPO，规避后确定性回正
- SHA-256：`632662d47fb35edfdb10d680a359cced03fab75af09cff666631a97f0ccf87f3`

天眼信息由平台融合层判定为最高优先级。S3 保留航迹来源，但不会用本地观测
覆盖天眼结果。可通过 `LOCAL_AVOID_GATE_MODE=legacy` 回退到旧门控，默认使用
`mission_v12`。

## 模型选择

V13 与 V14 使用同一 R9 上层、同一 mission gate，在 Final20 的 E01/M03、
B0/B3/B7/B9、固定 seed 701 上直接配对评估：

| 指标 | V13 | V14 | V14 - V13 |
|---|---:|---:|---:|
| 最终得分 | 75.45 | 74.67 | -0.78 |
| 躲避率 | 33.71% | 32.35% | -1.36 pp |
| 到达率 | 76.42% | 76.42% | 0.00 pp |
| 发射命中率 | 14.15% | 13.68% | -0.47 pp |

V14 的 GRU 在独立局部环境中更强，但没有转化为整局得分，因此当前发布选择
V13。V15 真实失败轨迹微调也未通过多种子仿真门控。

详细报告：

- `results/evaluations_v13/V13_EVALUATION.md`
- `results/evaluations_v14/V14_EVALUATION.md`

## 运行

```bash
python run.py run \
  --scenario easy/E01 \
  --blue-policy b0_fixed_ratio_random \
  --red-policy r9_hierarchical_learning \
  --red-motion-policy local_avoid_s3 \
  --red-learning-model models/local_avoid_ppo_v13_gate_aligned_s3_best.pt \
  --seed 701 \
  --run-id s3_v13_e01_b0 \
  -- --total-rounds 1 --render-mode none
```

`local_avoid_s3` 是推理策略，不能通过完整仿真的在线
`--red-learning-train` 参数继续训练。

## 历史全覆盖基线

V11 曾在 Final20 全部 20 张地图、B0-B9、固定 seed 701 上完成 200 局 S3
评估，平均分 `62.97`，相对直飞增加 `20.24` 分。V13/V14 的直接选择实验只覆盖
E01/M03 四种防守，因此不要把两组绝对分数直接当作全地图同比结果。

旧 V11 权重保留为训练起点与全地图历史基线：
`models/local_avoid_ppo_v11_trusted_mission_s3_best.pt`。
