#!/usr/bin/env bash
# 批量评估：固定「r9_search_enhanced（扫描层）+ local_avoid_s3 + 指定网络」口径，
# 跑一组 (label=ckpt:scenario)，并行执行，结束后把 summary.json 的分数汇总成 TSV。
#
# 用法：
#   bash tools/eval_ckpt_batch.sh TAG SEED "label=ckpt:scenario" ["label=ckpt:scenario" ...]
# 示例：
#   bash tools/eval_ckpt_batch.sh v15screen 701 \
#     "r3=models/local_avoid_set_attention_v12_r3_timely_arrival.pt:medium/M03" \
#     "it1=results/v15/v15_full20_20261019174553/checkpoints/iteration_0001.pt:medium/M03"
# 场景写 `medium/M03` 会自动补上 SCENARIO_SET 前缀（默认 launch20）；10-19 引擎只认 launch20，
# 旧 `final20/...` 仅旧 core 可读，需要对照时显式写全名。
# 环境变量：
#   SCENARIO_SET=launch20  裸 tier 场景的前缀
#   JOBS=8            并行度
#   SEARCH=1          1=r9_search_enhanced，0=r9_hierarchical_learning（A/B 用）
#   MAX_STEPS=3000    步数上限（clean 口径）
#   DRY=1             只打印命令
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=/home/amax/miniconda3/envs/macomp/bin/python
export LD_LIBRARY_PATH=$ROOT/core:$ROOT/core/envengine/sdk/Util:/home/amax/miniconda3/envs/macomp/lib
export BLUE_OBJECTIVE_MOTION=static LOCAL_AVOID_GATE_MODE=none
# 固定 Python 哈希顺序：否则策略内部集合/字典遍历顺序随进程变化，同 seed 两次会跑出不同分数
# （探针：M03/701 固定时两次均 91.6667，随机时 86.11/91.67 波动）。0 可被外部覆盖。
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

TAG=${1:?用法: bash tools/eval_ckpt_batch.sh TAG SEED "label=ckpt:scenario" ...}
SEED=${2:?缺少 seed}
shift 2
JOBS=${JOBS:-8}
SEARCH=${SEARCH:-1}
MAX_STEPS=${MAX_STEPS:-3000}
SCENARIO_SET=${SCENARIO_SET:-launch20}
if [ "$SEARCH" = "1" ]; then POLICY=r9_search_enhanced; else POLICY=r9_hierarchical_learning; fi

# 裸 `easy/E01` → `launch20/easy/E01`；已带集合名或 JSON 路径则原样返回。
resolve_scenario() {
  case "$1" in
    easy/*|medium/*|hard/*) printf '%s/%s' "$SCENARIO_SET" "$1" ;;
    *) printf '%s' "$1" ;;
  esac
}

OUT=results/v15/ckpt_screen
mkdir -p "$OUT"
SUM=$OUT/${TAG}_seed${SEED}.tsv
[ -f "$SUM" ] || printf "label\tscenario\tseed\tscore\tdestroyed_weight\ttotal_weight\tsteps\ttermination\trun_id\n" > "$SUM"

run_one() {
  local label=$1 model=$2 scenario run_id
  scenario=$(resolve_scenario "$3")
  run_id="screen_${TAG}_${label}_$(basename "$scenario")_${SEED}"
  local t0; t0=$(date +%s)
  if [ ! -f "$model" ]; then echo "[MISS] $label $model"; return 1; fi
  OMP_NUM_THREADS=1 "$PY" run.py run --scenario "$scenario" \
    --blue-policy b0_fixed_ratio_random \
    --red-policy "$POLICY" --red-motion-policy local_avoid_s3 \
    --red-learning-model "$model" --seed "$SEED" --run-id "$run_id" \
    -- --total-rounds 1 --max-steps "$MAX_STEPS" --render-mode none --disable-log-color \
    > "$OUT/${run_id}.log" 2>&1
  local rc=$? dt=$(( $(date +%s) - t0 ))
  local js; js=$(ls -t results/runs/"$run_id"/*/summary.json 2>/dev/null | head -1)
  if [ -n "$js" ]; then
    "$PY" - "$js" "$label" "$scenario" "$SEED" "$run_id" "$dt" "$SUM" <<'PY'
import json, sys
js, label, scen, seed, run_id, dt, out = sys.argv[1:8]
d = json.load(open(js)); s = d.get("score") or {}
row = [label, scen.split("/")[-1], seed, f"{s.get('score', float('nan')):.4f}",
       s.get("destroyed_weight"), s.get("total_objective_weight"),
       d.get("steps_executed"), d.get("termination_reason"), run_id]
open(out, "a").write("\t".join(str(x) for x in row) + "\n")
print(f"[DONE] {label:>6} {row[1]:4} score={row[3]} ({dt}s)")
PY
  else
    echo "[FAIL] $label $scenario rc=$rc ${dt}s（见 $OUT/${run_id}.log）"
  fi
}

if [ "${DRY:-0}" = "1" ]; then
  for spec in "$@"; do
    label=${spec%%=*}; rest=${spec#*=}; model=${rest%%:*}; scenario=$(resolve_scenario "${rest#*:}")
    echo "OMP_NUM_THREADS=1 python run.py run --scenario $scenario --red-policy $POLICY --red-motion-policy local_avoid_s3 --red-learning-model $model --seed $SEED --run-id screen_${TAG}_${label}_$(basename "$scenario")_${SEED} -- --total-rounds 1 --max-steps $MAX_STEPS --render-mode none"
  done
  exit 0
fi

echo "== $TAG seed=$SEED policy=$POLICY jobs=$JOBS  汇总: $SUM"
running=0
for spec in "$@"; do
  label=${spec%%=*}; rest=${spec#*=}; model=${rest%%:*}; scenario=$(resolve_scenario "${rest#*:}")
  run_one "$label" "$model" "$scenario" &
  running=$((running+1))
  if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running-1)); fi
done
wait
echo "== 全部完成，汇总："
column -t -s $'\t' "$SUM"
