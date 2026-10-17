# V12-R3 Set-Attention PPO release

This branch publishes the deployable V12-R3 low-level avoidance checkpoint,
compatible full-simulator training/evaluation code, compact training artifacts,
and aggregate E/M/H evaluation reports.

## Checkpoint contract

- Checkpoint: `models/local_avoid_set_attention_v12_r3_timely_arrival.pt`
- SHA-256: `2338b69fd5bb39c617d5a3fbee7c6c68e96cfe7c75faaf4dac70ade1f6358140`
- Algorithm: `local_avoid_set_attention_ppo_v1`
- Actor input: 85 dimensions, up to 8 masked threat tokens
- Critic input: 125 dimensions
- Action space: 3 discrete actions
- Architecture: 2-layer, 4-head Set-Attention policy
- Intended runtime: `r9_hierarchical_learning + local_avoid_s3`
- Gate: `none`
- Blue objective motion during training/evaluation: `static`

The checkpoint is standalone for inference. The source checkpoint used to begin
the R3 continuation is identified in the training protocol but is not needed to
deploy R3.

The original R3 training run did not archive a byte-exact source tree. The code
on this branch is the frozen source used for the completed R3 evaluation/resume
workflow and contains the R3 reward implementation; it must not be presented as
a byte-for-byte attestation of the historical training checkout.

## Training protocol

R3 was fine-tuned in the complete simulator on E01-E05 and M01-M05 only. It did
not train on H maps. The run used B0, 10 simulator workers, 10 games per PPO
iteration, 12 iterations, and a CUDA learner.

The `timely_arrival` objective keeps normalized target-distance reduction but
makes terminal behavior explicit:

- reaching the official 500 m arrival boundary and producing effective damage
  receive equivalent success treatment;
- entering 100 m is recorded as the damage-zone diagnostic;
- interception retains only 20% of distance progress and adds `-0.15`;
- lost/timeout trajectories receive no distance progress and add `-0.30`;
- elapsed time contributes up to `-0.05` at the horizon;
- timely arrival adds `+0.10`.

The exact run contract and all effective reward fields are in:

`results/local_avoid_set_attention_v12_r3_timely_arrival/local_avoid_set_attention_v12_r3_timely_arrival_20261015170601/protocol.json`

The same directory includes `training_history.jsonl`, `run_status.json`, and a
TensorBoard event file. Iteration checkpoints and raw simulator traces are not
included in this branch.

## Full-simulator evaluation

All rows below use R9 as the upper policy, `local_avoid_s3`, `gate=none`, static
objectives, and `max_steps=1200`. Rates come from MissionOutcomeTracker rather
than the PPO proxy reward.

| Evaluation | Valid games | Score | Evasion | Arrival <=500 m | Sampled <=100 m | Hits / launch |
|---|---:|---:|---:|---:|---:|---:|
| E01-E05/M01-M05, B0 | 30/30 | 75.31 | 30.86% | 64.40% | 63.69% | 18.20% |
| E01-E05/M01-M05, B3/B6/B9 | 90/90 | 84.17 | 42.53% | 79.13% | 78.33% | 18.25% |
| Held-out E06/M06, B0-B9 | 60/60 | 85.92 | 46.88% | 77.04% | 76.23% | 18.22% |
| Held-out H01-H08, B0-B9 | 233/240 | 80.71 | 49.48% | 73.40% | 72.80% | 22.58% |

The H matrix is deliberately marked `partial`: 240 jobs were scheduled, 233
were valid, and 7 timed out/failed. No missing game is silently counted as a
valid success. Detailed aggregate reports are under
`results/evaluations_set_attention_v12/`.

## Run one game

From the repository root, with the competition environment available:

```bash
LOCAL_AVOID_GATE_MODE=none BLUE_OBJECTIVE_MOTION=static \
conda run -n competition python run.py run \
  --scenario final20/easy/E01 \
  --blue-policy b0_fixed_ratio_random \
  --red-policy r9_hierarchical_learning \
  --red-motion-policy local_avoid_s3 \
  --red-learning-model models/local_avoid_set_attention_v12_r3_timely_arrival.pt \
  --seed 11901 --run-id v12_r3_smoke -- \
  --total-rounds 1 --max-steps 1200 --render-mode none --disable-log-color
```

## Verify the release

```bash
python releases/v12_r3/verify_release.py
```

The verifier checks every path and SHA-256 recorded in
`releases/v12_r3/MANIFEST.sha256`. Loading the checkpoint or running the native
simulator additionally requires the project Python/native runtime dependencies.
