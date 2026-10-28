#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""精读 tier-holdout / full20 训练曲线：114 个标签里只取决策相关的十几个。

用法（仓库根目录，用带 tensorboard 的 conda python）：
    /home/amax/miniconda3/envs/macomp/bin/python tools/tb_read.py
    ... tools/tb_read.py --dirs results/tierhold_easy results/full20 --all-keys

默认每个 run 打印一张小表：
    iter  score  arrive  evade  raw_ret  progress  kl     ent    ev     eval_E01 eval_M03
含义
    score    actual_outcomes/score_mean      训练局按判分公式算的真实均分（最重要）
    arrive   actual_outcomes/arrival_rate    真实到达率
    evade    actual_outcomes/evasion_rate    已结算弹的规避率
    raw_ret  full_sim_rollout/mean_raw_episode_return  每弹归一化回报（学没学会）
    progress full_sim_rollout/progress_positive_rate   朝目标推进的步占比
    kl       ppo/approx_kl                   策略步长，目标 ~target_kl(0.02)
    ent      ppo/entropy                     探索度，应缓降
    ev       ppo/explained_variance          critic 拟合度，应上升
    eval_*   eval/<cell>/score               留出格真实分
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEY_ROWS = [
    ("score", "actual_outcomes/score_mean", "{:.2f}", 1),
    ("arrive", "actual_outcomes/arrival_rate", "{:.3f}", 1),
    ("evade", "actual_outcomes/evasion_rate", "{:.3f}", 1),
    ("raw_ret", "full_sim_rollout/mean_raw_episode_return", "{:.3f}", 1),
    ("progress", "full_sim_rollout/progress_positive_rate", "{:.3f}", 1),
    ("kl", "ppo/approx_kl", "{:.4f}", 1),
    ("ent", "ppo/entropy", "{:.3f}", 1),
    ("ev", "ppo/explained_variance", "{:.3f}", 1),
]


def load(path: str) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    values: dict[str, list[float]] = {}
    steps: dict[str, list[int]] = {}
    for tag in ea.Tags().get("scalars", []):
        items = ea.Scalars(tag)
        values[tag] = [item.value for item in items]
        steps[tag] = [item.step for item in items]
    return values, steps


def find_runs(dirs: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in dirs:
        for hit in sorted(glob.glob(os.path.join(pattern, "**", "tensorboard"), recursive=True)):
            out.append(hit)
        if os.path.isdir(os.path.join(pattern, "tensorboard")):
            out.append(os.path.join(pattern, "tensorboard"))
    # 去重、保持稳定顺序
    seen: set[str] = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def print_run(path: str, all_keys: bool) -> None:
    name = os.path.basename(os.path.dirname(os.path.abspath(path)))
    try:
        series, steps_by_tag = load(path)
    except Exception as exc:  # noqa: BLE001
        print(f"== {name}: 读不了（{exc}）")
        return
    if not series:
        print(f"== {name}: 还没有标量（第 1 轮采集未结束）")
        return

    # 以 ppo/* 的点数作为轮次轴
    n_iter = len(steps_by_tag.get("ppo/entropy", [])) or max(
        (len(v) for v in series.values()), default=0
    )

    if all_keys:
        print(f"== {name}: {len(series)} 个标签")
        for tag in sorted(series):
            values = series[tag]
            print(f"   {tag:56s} n={len(values)} last={values[-1]:.4g}" if values else f"   {tag}")
        return

    cols = [c for c, tag, _, _ in KEY_ROWS if tag in series]
    eval_tags = sorted(t for t in series if t.startswith("eval/") and t.endswith("/score"))
    header = "iter " + " ".join(f"{c:>9s}" for c in cols) + "".join(
        f" {t.replace('eval/', 'ev_').replace('/score', ''):>9s}" for t in eval_tags
    )
    print(f"== {name}")
    print(header)
    for idx in range(n_iter):
        row = f"{idx + 1:>4d} "
        for col, tag, fmt, _ in KEY_ROWS:
            if tag not in series:
                continue
            values = series[tag]
            row += f" {(fmt.format(values[idx]) if idx < len(values) else '-'):>9s}"
        for tag in eval_tags:
            values = series[tag]
            row += f" {(f'{values[idx]:.2f}' if idx < len(values) else '-'):>9s}"
        print(row)
    missing = [c for c, tag, _, _ in KEY_ROWS if tag not in series]
    if missing:
        print(f"   (缺: {', '.join(missing)})")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["results/tierhold_easy", "results/tierhold_medium", "results/tierhold_hard", "results/full20"],
        help="训练结果根目录（会递归找 tensorboard/）",
    )
    parser.add_argument("--all-keys", action="store_true", help="列出全部标签而不是精读表")
    parser.add_argument("--list-tags", action="store_true", help="只打印某 run 的标签清单")
    args = parser.parse_args(argv)

    runs = find_runs(args.dirs)
    if not runs:
        print("没找到 tensorboard 目录，检查 --dirs", file=sys.stderr)
        return 1
    for path in runs:
        print_run(path, args.all_keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
