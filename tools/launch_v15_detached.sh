#!/usr/bin/env bash
# 在独立 systemd transient service 里启动 v15 训练（防 systemd-oomd 误杀 + 脱离终端）
#
# 背景：2026-10-19 17:21 的两次中断不是代码崩溃，而是 systemd-oomd 判定
#   /user.slice/user@1000.service 内存压力 81.52% > 50% 持续 >20s，做掉了
#   我们所在的 dsh-subprocess scope（峰值 13.3 GB）。本脚本用
#   ManagedOOMPreference=avoid 把训练进程标记为“最后才杀”，并放进独立 unit。
#
# 用法:
#   bash tools/launch_v15_detached.sh                 # 从头采集
#   INITIAL_TRACE_DIR=results/v15/<run>/traces/iteration_0001 \
#       bash tools/launch_v15_detached.sh             # 复用已采到的一轮 trace
#   UNIT=v15_try2 bash tools/launch_v15_detached.sh   # 指定 unit 名
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UNIT="${UNIT:-v15_train_$(date +%m%d_%H%M%S)}"
INITIAL_TRACE_DIR="${INITIAL_TRACE_DIR:-}"

if systemctl --user is-active --quiet "$UNIT" 2>/dev/null; then
  echo "FATAL: unit $UNIT 已在运行" >&2
  exit 1
fi
if pgrep -f "tools/train_local_avoid_full_sim.py" >/dev/null; then
  echo "FATAL: 已有训练进程在跑，先等它结束或手动停掉" >&2
  pgrep -af "tools/train_local_avoid_full_sim.py" >&2
  exit 1
fi

SETENV=(--setenv=DEVICE="${DEVICE:-cuda:3}")
[ -n "$INITIAL_TRACE_DIR" ] && SETENV+=(--setenv=INITIAL_TRACE_DIR="$INITIAL_TRACE_DIR")

systemd-run --user \
  --unit="$UNIT" --collect \
  -p ManagedOOMPreference=avoid \
  -p Nice=5 \
  "${SETENV[@]}" \
  -- bash -lc "cd '$ROOT' && exec bash tools/run_v15_full20.sh"

echo
echo "unit      : $UNIT"
echo "状态      : systemctl --user status $UNIT"
echo "日志      : journalctl --user -u $UNIT -f"
echo "           （脚本内部还会 tee 一份到 results/v15/logs/）"
echo "停止      : systemctl --user stop $UNIT"
echo "看板      : /home/amax/miniconda3/envs/macomp/bin/python tools/v15_progress.py"
