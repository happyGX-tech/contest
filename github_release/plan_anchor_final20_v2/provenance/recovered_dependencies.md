# 补漏源码与证据等级

2026-10-13 复核发现，初版发布包只保存了实验直接改动的文件，遗漏了这些文件导入的 Python 运行依赖。现已按原项目路径补到 `snapshot/`，包括：

- 导师指出的 `info_center.py`、`unified_encoder.py`、`unified_networks.py`、`action_spec.py`；
- `unified/` 的其余源码、PlanSpec 所需红方公共模块和基础 Agent；
- 冻结 `local_avoid_s3` checkpoint 的加载/网络代码；
- 固定蓝方策略、网格部署、运行包装器和 `RunSummary`。

证据等级不能混写：

- `core/main.py`、`unified_missile_agent.py`、PlanSpec schema/executor、场景和 checkpoint 的 SHA-256 在原始 `run_index.json` 中有记录，发布快照逐项匹配。
- `action_spec.py`、`unified_encoder.py`、`unified_networks.py` 在当前工作树、2026-10-09 顶层训练归档和训练侧备份中的 SHA-256 一致。
- 原实验协议没有记录 `info_center.py` 的 SHA-256。当前工作树在实验后于 2026-10-12 又修改过该文件；发布版本由修改记录逆向恢复到这些后续修改之前，并与实验快照接口一致。因此它是“恢复版本”，不能宣称具有原始运行索引直接记录的逐字节证明。

`scripts/verify_bundle.py` 会检查补漏文件存在性、4 个关键恢复文件的固定哈希及全部 snapshot Python 文件的语法；`scripts/verify_python_snapshot.py` 会在安装 NumPy/PyTorch 的环境中执行关键导入烟测。

完整仿真重放仍需要原项目的 `envengine`、Linux 原生动态库和相同 Python/NumPy/PyTorch 环境。它们不是本 GitHub 目录的一部分；本补丁关闭的是 Python 策略/执行器源码缺失问题，不把原生模拟器依赖伪装成已解决。
