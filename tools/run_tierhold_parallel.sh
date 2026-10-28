#!/usr/bin/env bash
# 三路并行训练 tier-holdout 模型（每个 tier 一个 tmux 会话，默认 easy/medium/hard -> cuda:0/1/2）
#
# 用法（在任意目录）：
#   bash tools/run_tierhold_parallel.sh check    # 预检：python/torch/显卡/输出文件
#   bash tools/run_tierhold_parallel.sh dry      # 只打印三路命令，不启动
#   bash tools/run_tierhold_parallel.sh          # 启动三路（tmux 会话名 th_easy / th_medium / th_hard）
#   bash tools/run_tierhold_parallel.sh easy     # 只启动一个 tier
#   bash tools/run_tierhold_parallel.sh full20   # 全 20 图训练（无留出，cuda:7，提交候选）
#   bash tools/run_tierhold_parallel.sh status   # 看会话状态与进度
#   bash tools/run_tierhold_parallel.sh stop     # 停掉三路（只杀本脚本启动的会话）
#
# 环境变量：WORKERS=16 ITERATIONS=12 GAMES=10 PYTHON=/path/to/python USE_TMUX=0（无 tmux 时用 setsid nohup）
# 说明见 local_plan/69_全量训练计划.md §3/§7。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/amax/miniconda3/envs/macomp/bin/python}"
WORKERS="${WORKERS:-16}"
GAMES="${GAMES:-10}"
ITERATIONS="${ITERATIONS:-12}"
USE_TMUX="${USE_TMUX:-1}"
BASE_MODEL="models/local_avoid_set_attention_v12_r3_timely_arrival.pt"
LOGDIR="results/tierhold_logs"
TIERS_ALL=(easy medium hard)
TIERS_RUN=(easy medium hard full20)
# full20 = 全 20 图训练（无留出），作为真正提交候选；池子 6E+6M+8H
SCEN_ALL="final20/easy/E01,final20/easy/E02,final20/easy/E03,final20/easy/E04,final20/easy/E05,final20/easy/E06,final20/medium/M01,final20/medium/M02,final20/medium/M03,final20/medium/M04,final20/medium/M05,final20/medium/M06,final20/hard/H01,final20/hard/H02,final20/hard/H03,final20/hard/H04,final20/hard/H05,final20/hard/H06,final20/hard/H07,final20/hard/H08"

export PYTHONPATH="$ROOT:$ROOT/core${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/core:$ROOT/core/envengine/sdk/Util:/home/amax/miniconda3/envs/macomp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLUE_OBJECTIVE_MOTION=static
export LOCAL_AVOID_GATE_MODE=none

# 本机的 conda lib 会让 tmux 报 libtinfo/NCURSES 版本错，调 tmux 时剥掉这些变量
tmux_() { env -u LD_LIBRARY_PATH -u PYTHONPATH -u OMP_NUM_THREADS tmux "$@"; }

tier_gpu() { case "$1" in easy) echo 0 ;; medium) echo 1 ;; hard) echo 2 ;; full20) echo 7 ;; *) echo 0 ;; esac; }
out_model() {
  if [ "$1" = "full20" ]; then echo "models/local_avoid_set_attention_full20.pt"
  else echo "models/local_avoid_set_attention_tierhold_$1.pt"; fi
}
session_name() { echo "th_$1"; }

train_args() {
  local tier="$1" gpu="$2"
  if [ "$tier" = "full20" ]; then
    echo "--input-model $BASE_MODEL --output-model $(out_model full20) \
--tier-holdout none --scenarios $SCEN_ALL --workers 20 --games 20 --iterations ${FULL20_ITERATIONS:-6} \
--device cuda:$gpu --runtime host \
--gamma 0.99 --gae-lambda 0.95 \
--target-kl 0.02 --clip-range-value 0.2 --advantage-standardize --reward-clip 10 \
--anneal-lr --anneal-final-scale 0.1 --entropy-coef-final 0.002 \
--eval-every 3 --eval-cells E01/b0,M03/b0 --eval-final-cells H07/b0 --eval-jobs 3 \
--results-root results/full20 --run-name full20"
    return 0
  fi
  echo "--input-model $BASE_MODEL --output-model $(out_model "$tier") \
--tier-holdout $tier --workers $WORKERS --games $GAMES --iterations $ITERATIONS \
--device cuda:$gpu --runtime host \
--gamma 0.99 --gae-lambda 0.95 \
--target-kl 0.02 --clip-range-value 0.2 --advantage-standardize --reward-clip 10 \
--anneal-lr --anneal-final-scale 0.1 --entropy-coef-final 0.002 \
--eval-every 6 --eval-cells E01/b0,M03/b0 --eval-final-cells H07/b0 --eval-jobs 2 \
--results-root results/tierhold_$tier --run-name tierhold_$tier"
}

do_check() {
  echo "== 仓库: $ROOT"
  echo "== python: $PYTHON"
  "$PYTHON" - <<'PY'
import sys
import torch
print(f"   python {sys.version.split()[0]}, torch {torch.__version__}, cuda={torch.cuda.is_available()}, "
      f"devices={torch.cuda.device_count()}")
PY
  echo "== 起点权重: $BASE_MODEL"
  [ -f "$BASE_MODEL" ] || { echo "   缺失！"; return 1; }
  echo "== 显卡（MiB used）"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/   /'
  echo "== 输出文件（已存在会被跳过）"
  local t
  for t in "${TIERS_RUN[@]}"; do
    local g; g="$(tier_gpu "$t")"
    local used; used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')"
    printf '   %-7s cuda:%s  used=%s MiB  输出=%s %s\n' "$t" "$g" "$used" "$(out_model "$t")" \
      "$([ -e "$(out_model "$t")" ] && echo '[已存在，将跳过]' || echo '[待生成]')"
    if [ "${used:-0}" -gt 20000 ]; then
      echo "   !! cuda:$g 已被占用 ${used} MiB，换卡或先停别人"
    fi
  done
}

launch_one() {
  local tier="$1" gpu log wrapper
  gpu="$(tier_gpu "$tier")"
  log="$LOGDIR/$tier.log"
  wrapper="$LOGDIR/run_$tier.sh"
  mkdir -p "$LOGDIR"

  if [ -e "$(out_model "$tier")" ]; then
    echo "[skip] $(out_model "$tier") 已存在（训练器防覆盖），先挪走再跑"
    return 0
  fi

  cat >"$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLUE_OBJECTIVE_MOTION=static LOCAL_AVOID_GATE_MODE=none
echo "[\$(date '+%F %T')] start tier=$tier gpu=cuda:$gpu workers=$WORKERS iterations=$ITERATIONS" | tee -a "$log"
set +e
"$PYTHON" tools/train_local_avoid_full_sim.py $(train_args "$tier" "$gpu") 2>&1 | tee -a "$log"
rc=\${PIPESTATUS[0]}
set -e
echo "[\$(date '+%F %T')] exit=\$rc tier=$tier" | tee -a "$log"
exit "\$rc"
EOF
  chmod +x "$wrapper"

  if [ "$USE_TMUX" = "1" ] && command -v tmux >/dev/null 2>&1; then
    if tmux_ has-session -t "$(session_name "$tier")" 2>/dev/null; then
      echo "[skip] tmux 会话 $(session_name "$tier") 已存在，先 stop"
      return 0
    fi
    tmux_ new-session -d -s "$(session_name "$tier")" -c "$ROOT" "bash '$wrapper'"
    echo "[start] tmux 会话 $(session_name "$tier")  (attach: tmux attach -t $(session_name "$tier"))  日志: $log"
  else
    setsid nohup bash "$wrapper" >/dev/null 2>&1 &
    echo "[start] 后台 pid=$!  tier=$tier  日志: $log"
  fi
}

do_status() {
  echo "== tmux 会话"
  local t
  for t in "${TIERS_RUN[@]}"; do
    if tmux_ has-session -t "$(session_name "$t")" 2>/dev/null; then
      echo "   $(session_name "$t"): 运行中"
    else
      echo "   $(session_name "$t"): 无"
    fi
  done
  echo "== 进度（日志尾部）"
  for t in "${TIERS_RUN[@]}"; do
    local log="$LOGDIR/$t.log"
    if [ -f "$log" ]; then
      printf '   --- %s\n' "$log"
      grep -E "ITERATION_COMPLETE|EVAL |PIPELINE_COMPLETE|MODEL |Traceback|Error" "$log" 2>/dev/null | tail -n 4 | sed 's/^/      /'
      tail -n 1 "$log" | cut -c1-160 | sed 's/^/      … /'
    fi
  done
  echo "== 产物"
  ls -la models/*tierhold*.pt 2>/dev/null | sed 's/^/   /' || echo "   还没有"
}

do_stop() {
  local t
  for t in "${TIERS_RUN[@]}"; do
    if tmux_ has-session -t "$(session_name "$t")" 2>/dev/null; then
      tmux_ kill-session -t "$(session_name "$t")" && echo "[stop] tmux 会话 $(session_name "$t")"
    fi
  done
  pkill -f "train_local_avoid_full_sim.py --input-model $BASE_MODEL" 2>/dev/null && echo "[stop] 残留训练进程已清理" || true
}

target="${1:-all}"
case "$target" in
  check) do_check ;;
  dry)
    # 只生成 wrapper 并打印命令，不启动（核对用）
    do_check
    for t in "${TIERS_RUN[@]}"; do
      g="$(tier_gpu "$t")"
      mkdir -p "$LOGDIR"
      echo
      echo "== $t -> cuda:$g"
      echo "   $PYTHON tools/train_local_avoid_full_sim.py $(train_args "$t" "$g")"
    done
    ;;
  status) do_status ;;
  stop) do_stop ;;
  all)
    do_check
    for t in "${TIERS_ALL[@]}"; do launch_one "$t"; sleep 3; done
    echo
    echo "全部启动。看进度：bash tools/run_tierhold_parallel.sh status"
    ;;
  easy|medium|hard|full20) do_check && launch_one "$target" ;;
  full20) do_check && launch_one full20 ;;
  *) echo "用法: $0 [check|dry|status|stop|easy|medium|hard|full20|all]" >&2; exit 2 ;;
esac
