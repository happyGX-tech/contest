#!/usr/bin/env bash
# 全矩阵评估：it3 + r9_search_enhanced，final20 全部 20 图 × 3 seed，JOBS=8。
# 口径与 tools/eval_ckpt_batch.sh 一致（clean、3000 步、目标静止、gate off）。
# 用法：bash tools/run_v15_fullmatrix.sh            # 默认三 seed 701/8101/9101
#       DRY=1 bash tools/run_v15_fullmatrix.sh      # 只打印 60 条命令
#       SEEDS="701" JOBS=4 bash tools/run_v15_fullmatrix.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

JOBS=${JOBS:-8}
CKPT=${CKPT:-results/v15/v15_full20_20261019174553/checkpoints/iteration_0003.pt}
SEEDS=${SEEDS:-"701 8101 9101"}
TAG_PREFIX=${TAG_PREFIX:-v15mat}
LOG=${LOG:-results/v15/logs/matrix_$(date +%Y%m%d_%H%M%S).log}

[ -f "$CKPT" ] || { echo "缺少 ckpt: $CKPT" >&2; exit 2; }
mkdir -p results/v15/logs

# case_id 必须带 final20/ 前缀：scenarios/cases 下还有同名的历史 tier 目录
# （easy/hard/medium，其中 hard 只有 H01-H03），裸 easy/E01 会解析到旧图。
mapfile -t SCEN < <(for d in scenarios/cases/final20/*/*/; do
  printf "final20/%s/%s\n" "$(basename "$(dirname "$d")")" "$(basename "$d")"
done | sort)
[ "${#SCEN[@]}" -eq 20 ] || { echo "期望 20 张图，实际 ${#SCEN[@]}" >&2; exit 2; }

say() { echo "$*" | tee -a "$LOG"; }

say "== 全矩阵 it3+扫描 ${#SCEN[@]} 图 × seeds=[$SEEDS]，JOBS=$JOBS"
say "   ckpt=$CKPT"
say "   起跑 $(date '+%F %H:%M:%S')"

if [ "${DRY:-0}" = "1" ]; then
  for seed in $SEEDS; do
    args=()
    for s in "${SCEN[@]}"; do args+=("it3=$CKPT:$s"); done
    DRY=1 bash tools/eval_ckpt_batch.sh "${TAG_PREFIX}_${seed}" "$seed" "${args[@]}" | tee -a "$LOG"
  done
  say "== DRY 结束（未跑任何评估）"
  exit 0
fi

for seed in $SEEDS; do
  args=()
  for s in "${SCEN[@]}"; do args+=("it3=$CKPT:$s"); done
  say "---- seed $seed 起跑 $(date '+%H:%M:%S')（${#args[@]} 局）"
  JOBS=$JOBS bash tools/eval_ckpt_batch.sh "${TAG_PREFIX}_${seed}" "$seed" "${args[@]}" 2>&1 | tee -a "$LOG"
  say "---- seed $seed 完成 $(date '+%H:%M:%S')"
done

say "MATRIX_COMPLETE $(date '+%F %H:%M:%S')"
say "== 三份汇总：results/v15/ckpt_screen/${TAG_PREFIX}_<seed>_seed<seed>.tsv"
