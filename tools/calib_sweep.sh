#!/usr/bin/env bash
# Calibration sweep: score every intermediate phase-1 checkpoint in the whole game
# on E01 x b0 x seed 701, paired against `straight` on the same map.
#
# Reference point: results/evaluations_v13/v13_best_scoped_probe holds the same
# map/blue/seed cell for V13 (s3 74.47 / straight 43.07 in the original protocol).
#
# usage: tools/calib_sweep.sh <ckpt-dir> <run-prefix> [workers]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${1:?usage: calib_sweep.sh <ckpt-dir> <run-prefix> [workers]}"
PREFIX="${2:?usage: calib_sweep.sh <ckpt-dir> <run-prefix> [workers]}"
WORKERS="${3:-4}"

export PYTHONPATH="$ROOT:$ROOT/core"
export LD_LIBRARY_PATH="$ROOT/core:$ROOT/core/envengine/sdk/Util:/home/amax/miniconda3/envs/macomp/lib"
PY=/home/amax/miniconda3/envs/macomp/bin/python

run_one() {
  local motion="$1" model="$2" tag="$3"
  local extra=()
  if [[ "$motion" == "local_avoid_s3" ]]; then
    extra=(--red-learning-model "$model")
  fi
  "$PY" "$ROOT/run.py" run \
    --scenario easy/E01 \
    --blue-policy b0_fixed_ratio_random \
    --red-policy r9_hierarchical_learning \
    --red-motion-policy "$motion" \
    "${extra[@]}" \
    --seed 701 \
    --run-id "${PREFIX}_${tag}" \
    -- --total-rounds 1 >/dev/null 2>&1
  echo "done ${PREFIX}_${tag}"
}
export -f run_one

run_one straight "" straight &
pids=($!)
for model in "$CKPT_DIR"/ckpt_update*.pt; do
  base="$(basename "$model" .pt)"
  run_one local_avoid_s3 "$model" "$base" &
  pids+=($!)
  while [[ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]]; do wait -n; done
done
wait
echo "ALL_DONE"
