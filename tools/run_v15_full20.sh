#!/usr/bin/env bash
# v15_search_enhanced —— launch20 全 20 图训练启动器（Stage A，10-19 引擎）
#
# 用法:
#   bash tools/run_v15_full20.sh                 # 直接开跑
#   bash tools/run_v15_full20.sh --dry           # 只打印命令，不执行
#   ITERATIONS=12 PPO_EPOCHS=6 bash tools/run_v15_full20.sh
#
# 设计原则:
#   - 位置参数之前的一切都可用环境变量覆盖，不改代码就能扫参
#   - 默认值与 local_plan/72 的 Stage A 完全一致
#   - 只暴露“调了不会破坏可比性”的旋钮；--scenarios/--tier-holdout 等
#     影响协议口径的项不在默认覆盖范围内（SCENARIOS 留了逃生口）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export LD_LIBRARY_PATH="$ROOT/core:$ROOT/core/envengine/sdk/Util:/home/amax/miniconda3/envs/macomp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DRY=0
for arg in "$@"; do
  case "$arg" in
    --dry) DRY=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

PYTHON="${PYTHON:-/home/amax/miniconda3/envs/macomp/bin/python}"
INPUT_MODEL="${INPUT_MODEL:-models/local_avoid_set_attention_v12_r3_timely_arrival.pt}"
OUTPUT_MODEL="${OUTPUT_MODEL:-models/local_avoid_v15_search_enhanced.pt}"
DEVICE="${DEVICE:-cuda:3}"

# —— 数据规模（每轮 = WORKERS 局并发）——
WORKERS="${WORKERS:-20}"
GAMES="${GAMES:-20}"
ITERATIONS="${ITERATIONS:-6}"

# —— PPO 优化 ——
PPO_EPOCHS="${PPO_EPOCHS:-3}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-4096}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"            # 空 = 沿用 ckpt 的 1e-5
GAMMA="${GAMMA:-0.99}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
TARGET_KL="${TARGET_KL:-0.02}"                    # 空 = 关闭 KL 早停
ENTROPY_COEF="${ENTROPY_COEF:-}"              # 空 = 沿用 ckpt 的 0.02
ENTROPY_COEF_FINAL="${ENTROPY_COEF_FINAL:-0.002}"
ANNEAL_LR="${ANNEAL_LR:-1}"
ANNEAL_FINAL_SCALE="${ANNEAL_FINAL_SCALE:-0.3}"
CLIP_RANGE="${CLIP_RANGE:-}"                  # 空 = 沿用 ckpt 的 0.15

# —— 奖励 ——
REWARD_MODE="${REWARD_MODE:-progress_plus_score}"
SCORE_BONUS_COEF="${SCORE_BONUS_COEF:-1.0}"
REWARD_CLIP="${REWARD_CLIP:-}"

# —— 环境与协议（改这些会破坏与 R3 的可比性，谨慎）——
MAX_STEPS="${MAX_STEPS:-3000}"
BLUE_POLICY="${BLUE_POLICY:-b0_fixed_ratio_random}"
OBJECTIVE_MOTION="${OBJECTIVE_MOTION:-static}"
SEED_START="${SEED_START:-8101}"

# —— 评估 ——
EVAL_EVERY="${EVAL_EVERY:-3}"
EVAL_CELLS="${EVAL_CELLS:-E01/b0,M03/b0}"
EVAL_FINAL_CELLS="${EVAL_FINAL_CELLS:-H07/b0}"
EVAL_JOBS="${EVAL_JOBS:-2}"

# —— 断点续采（复用某轮已采集的 trace，仅第 1 轮生效）——
INITIAL_TRACE_DIR="${INITIAL_TRACE_DIR:-}"

# —— 产物 ——
RESULTS_ROOT="${RESULTS_ROOT:-results/v15}"
RUN_NAME="${RUN_NAME:-v15_full20}"

if [ -z "${SCENARIOS:-}" ]; then
  SCENARIOS="$(ls -d scenarios/cases/launch20/*/*/scenario.json 2>/dev/null \
    | sed 's#^scenarios/cases/##; s#/scenario.json$##' | sort | paste -sd, -)"
fi

if [ -z "$SCENARIOS" ]; then
  echo "FATAL: no launch20 scenarios found" >&2
  exit 1
fi
MAP_COUNT="$(printf '%s\n' "$SCENARIOS" | tr ',' '\n' | grep -c .)"
if [ "$MAP_COUNT" -ne 20 ]; then
  echo "WARN: expected 20 launch20 maps, got $MAP_COUNT" >&2
fi
if [ ! -f "$INPUT_MODEL" ]; then
  echo "FATAL: input checkpoint not found: $INPUT_MODEL" >&2
  exit 1
fi

ARGS=(
  --input-model "$INPUT_MODEL"
  --output-model "$OUTPUT_MODEL"
  --tier-holdout none
  --scenarios "$SCENARIOS"
  --heldout-scenarios ""
  --blue-policy "$BLUE_POLICY"
  --objective-motion "$OBJECTIVE_MOTION"
  --workers "$WORKERS" --games "$GAMES" --iterations "$ITERATIONS"
  --seed-start "$SEED_START"
  --device "$DEVICE" --runtime host
  --gamma "$GAMMA" --gae-lambda "$GAE_LAMBDA"
  --ppo-epochs "$PPO_EPOCHS" --minibatch-size "$MINIBATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --reward-mode "$REWARD_MODE" --score-bonus-coef "$SCORE_BONUS_COEF"
  --eval-every "$EVAL_EVERY" --eval-cells "$EVAL_CELLS"
  --eval-final-cells "$EVAL_FINAL_CELLS" --eval-jobs "$EVAL_JOBS"
  --results-root "$RESULTS_ROOT" --run-name "$RUN_NAME"
)
[ -n "$LEARNING_RATE" ] && ARGS+=(--learning-rate "$LEARNING_RATE")
[ -n "$TARGET_KL" ] && ARGS+=(--target-kl "$TARGET_KL")
[ -n "$ENTROPY_COEF" ] && ARGS+=(--entropy-coef "$ENTROPY_COEF")
[ -n "$CLIP_RANGE" ] && ARGS+=(--clip-range-value "$CLIP_RANGE")
[ -n "$REWARD_CLIP" ] && ARGS+=(--reward-clip "$REWARD_CLIP")
[ -n "$INITIAL_TRACE_DIR" ] && ARGS+=(--initial-trace-dir "$INITIAL_TRACE_DIR")
if [ "$ANNEAL_LR" = "1" ]; then
  ARGS+=(--anneal-lr --anneal-final-scale "$ANNEAL_FINAL_SCALE" --entropy-coef-final "$ENTROPY_COEF_FINAL")
fi

CMD=("$PYTHON" tools/train_local_avoid_full_sim.py "${ARGS[@]}")

echo "== v15_search_enhanced =="
echo "maps      : $MAP_COUNT"
echo "rounds    : $ITERATIONS x ($WORKERS workers / $GAMES games)"
echo "reward    : $REWARD_MODE (bonus coef $SCORE_BONUS_COEF)"
echo "device    : $DEVICE"
echo "output    : $OUTPUT_MODEL"
echo "log dir   : $RESULTS_ROOT/logs"
echo
printf '%q ' "${CMD[@]}"; echo

if [ "$DRY" = "1" ]; then
  exit 0
fi

mkdir -p "$RESULTS_ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RESULTS_ROOT/logs/${RUN_NAME}_${STAMP}.log"
echo "log       : $LOG"
"${CMD[@]}" 2>&1 | tee "$LOG"
