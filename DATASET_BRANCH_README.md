# `dataset` 分支：完整仿真轨迹采集

这个分支只负责在 CPU 上运行完整仿真、记录逐弹轨迹，并立即归档到内容寻址的
永久数据集。它不会执行 PPO 更新，也不会把轨迹提交到 Git。

## 1. 在采集服务器上准备

```bash
git clone --branch dataset --single-branch \
  git@github.com:jia065941-blip/contest.git competition-platform-env
cd competition-platform-env

conda activate competition
python --version                 # 必须是 Python 3.11.x
python -c 'import numpy, torch; print(numpy.__version__, torch.__version__)'
```

环境需要安装 `requirements-dataset.txt` 中的依赖。原生动力学库只支持 Linux
x86-64。若宿主机不能直接加载仓库内的 `.so`，
安装 Docker，并使用下面的 Docker 模式。

```bash
python -m pip install -r requirements-dataset.txt
```

分支内自带的 V9 采集策略校验值为：

```text
d23c1eac2bc18756f66ea808ec2b3c5167b151f075085a7d683df0d051a24a6d  models/local_avoid_set_attention_v9_fullsim_static_w16.pt
```

## 2. 先做检查

```bash
conda activate competition
python -m unittest \
  tests.test_full_sim_trajectory_dataset \
  tests.test_full_sim_local_avoid_training \
  tests.test_mission_outcomes -v

python tools/collect_local_avoid_full_sim_dataset.py verify
```

空数据集的 `verify` 也应返回 `valid: true`。

## 3. 启动独立采集

宿主机可以直接加载原生库时：

```bash
conda activate competition
python tools/collect_local_avoid_full_sim_dataset.py collect \
  --runtime host \
  --model models/local_avoid_set_attention_v9_fullsim_static_w16.pt \
  --rounds 10 \
  --games-per-round 16 \
  --workers 12 \
  --seed-start 200001 \
  --run-name server2_static \
  --objective-motion static \
  --gate-mode none
```

需要 Docker 兼容层时：

```bash
conda activate competition
export COMPETITION_PYTHON_ENV="$CONDA_PREFIX"
python tools/collect_local_avoid_full_sim_dataset.py collect \
  --runtime docker \
  --python-env "$CONDA_PREFIX" \
  --model models/local_avoid_set_attention_v9_fullsim_static_w16.pt \
  --rounds 10 \
  --games-per-round 16 \
  --workers 12 \
  --seed-start 200001 \
  --run-name server2_static \
  --objective-motion static \
  --gate-mode none
```

不同服务器必须使用不重叠的 `--seed-start` 区间和不同的 `--run-name`。默认地图池
为 E01-E05、M01-M05、H01-H06；E06、M06、H07、H08 留作 heldout。

## 4. 数据位置与同步

数据默认写入：

```text
datasets/local_avoid_full_sim_v1/
```

每个 worker 完成后会立即把原始 `jsonl.gz` 轨迹按 SHA-256 归档。超时、损坏和
缺少真实终局标签的轨迹仍会保存，但放入 quarantine，不会被误标为可训练数据。

采集结束或定期同步前先检查：

```bash
python tools/collect_local_avoid_full_sim_dataset.py verify
```

推荐把整个目录同步回主服务器。例如在采集服务器执行：

```bash
rsync -a --partial --info=progress2 \
  datasets/local_avoid_full_sim_v1/ \
  USER@MAIN_SERVER:/PATH/competition-platform-env/datasets/local_avoid_full_sim_v1/
```

对象文件按内容哈希命名，可以安全去重；但两台服务器不应同时直接写同一个网络
文件系统上的 `manifest.json`。应各自在本地采集，再同步对象，或在主服务器使用
采集器的 `import-run` 归档完整 run。

更多数据格式与训练边界见 `FULL_SIM_TRAJECTORY_DATASET.md`。
