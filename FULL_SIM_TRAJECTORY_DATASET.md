# 完整仿真轨迹库

## 1. 目标

将完整仿真的 CPU 采样与 GPU 学习解耦。轨迹库按 SHA-256 保存原始
`jsonl.gz` 字节，同时记录策略、地图、种子、蓝方策略、目标运动方式及终局
标签来源。采集失败或只有代理终局标签的轨迹也会保留，但不会被标记为可训练。

默认位置：

```text
datasets/local_avoid_full_sim_v1/
├── dataset.json
├── manifest.json
├── policies/                 # 内容寻址的行为策略快照
├── objects/sha256/           # 内容寻址的不可变轨迹
├── records/                  # 每个轨迹对象的验证与来源元数据
└── runs/                     # 独立采集任务的协议、日志及原始轨迹
```

`manifest.json` 中只有 `usable_objects` 对应的记录可以直接进入真实终局训练。
`quarantined_objects` 包括超时、损坏或缺少 `MissionOutcomeTracker` 标签的数据，
必须补齐/审核标签后才能使用。

完整性检查：

```bash
python tools/collect_local_avoid_full_sim_dataset.py verify
```

## 2. 导入已有训练轨迹

导入不会移动或修改原文件，并会对重复内容去重：

```bash
conda activate competition
python tools/collect_local_avoid_full_sim_dataset.py import-run \
  results/local_avoid_set_attention_v10_fullsim_multimap_v2/\
local_avoid_set_attention_v10_realoutcome_multimap_w12_20261014113747
```

## 3. 独立 CPU 多地图采集

下面的采集器不执行 PPO 更新。每个 worker 一结束就立刻归档，因此其他地图
超时不会导致已完成数据丢失。行为策略会先复制到内容寻址的 `policies/`，采集
期间不会引用可能被训练进程覆盖的模型文件。

```bash
conda activate competition
python tools/collect_local_avoid_full_sim_dataset.py collect \
  --model models/local_avoid_set_attention_v9_fullsim_static_w16.pt \
  --rounds 1 \
  --games-per-round 16 \
  --workers 12 \
  --timeout-seconds 7200 \
  --objective-motion static \
  --gate-mode none
```

如果完整仿真需要 Docker 兼容层，采集器默认把当前执行脚本的 Python/Conda
环境只读挂载到容器。也可以显式指定，避免依赖某台服务器的绝对路径：

```bash
export COMPETITION_PYTHON_ENV="$CONDA_PREFIX"
python tools/collect_local_avoid_full_sim_dataset.py collect \
  --runtime docker \
  --python-env "$CONDA_PREFIX" \
  --model models/local_avoid_set_attention_v9_fullsim_static_w16.pt
```

另一台服务器至少需要 Linux x86-64、Python 3.11、Docker（宿主机 glibc
不足时）和包含 `numpy`、`torch` 及项目依赖的 `competition` 环境。轨迹数据目录
已被 Git 忽略；应使用对象存储、`rsync` 或独立数据盘同步，而不是提交到代码仓库。

默认地图池为训练集 E01-E05、M01-M05、H01-H06；E06、M06、H07、H08 不在
默认采集池中，继续作为严格 heldout。默认保留至少 10 GiB 磁盘空间，空间不足
时拒绝开始下一轮。

## 4. 训练边界

- 固定轨迹可反复用于成功轨迹加权行为克隆、价值函数预训练及奖励重算。
- 普通 PPO 更新只能使用生成它的行为策略对应的新鲜批次；不能把整个历史库
  当作当前策略的 on-policy 数据反复更新。
- 离线预训练后的模型仍需使用新鲜完整仿真轨迹做少量 PPO 校正和 heldout 评估。
