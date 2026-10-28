#!/usr/bin/env bash
# A/B: legacy PPO update (arm A) vs the eight implementation details (arm B).
# Same warm start, same seed, same 1,048,576-step three-phase budget.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/core"
export LD_LIBRARY_PATH="$ROOT/core:$ROOT/core/envengine/sdk/Util:/home/amax/miniconda3/envs/macomp/lib"
PY=/home/amax/miniconda3/envs/macomp/bin/python
OUT="$ROOT/results/ab_tricks"
COMMON=(--resume models/local_avoid_ppo_v10_dagger_s3.pt --device cuda:7
        --phase-steps 262144,262144,393216)

echo "=== arm A (legacy) start $(date +%H:%M:%S) ==="
"$PY" tools/train_local_avoid_ppo_ab.py --run-name legacy --out-dir "$OUT/legacy" "${COMMON[@]}"

echo "=== arm B (tricks) start $(date +%H:%M:%S) ==="
"$PY" tools/train_local_avoid_ppo_ab.py --run-name tricks --out-dir "$OUT/tricks" "${COMMON[@]}" \
  --target-kl 0.02 --clip-range-value 0.2 --orthogonal-init --dual-clip -0.5 \
  --anneal-lr --advantage-standardize --reward-clip 10 --entropy-coef-final 0.001

echo "=== both arms done $(date +%H:%M:%S) ==="
