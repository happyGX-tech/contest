#!/usr/bin/env python3
"""把 v15 全矩阵评估（20 图 × 3 seed）汇总成 Markdown / CSV。

数据源：results/v15/ckpt_screen/v15mat_<seed>_seed<seed>.tsv
输出：  results/v15/FULL_MATRIX.md、results/v15/full_matrix.csv
"""
from __future__ import annotations
import collections, glob, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v15"
SEEDS = ["701", "8101", "9101"]
TIERS = {"E": "easy", "M": "medium", "H": "hard"}
# 既有复评（同配置、但跑在不同批次、PYTHONHASHSEED 随机）
OLD = {("M03", "701"): 91.6667, ("M03", "8101"): 86.8056, ("M03", "9101"): 88.1944,
       ("H07", "701"): 85.9438, ("H07", "8101"): 81.1245, ("H07", "9101"): 87.5502,
       ("H01", "701"): 86.3454, ("H01", "8101"): 85.1406, ("H01", "9101"): 77.1084,
       ("E01", "701"): 98.8506}


def load():
    rows = []
    for p in sorted(glob.glob(str(OUT / "ckpt_screen/v15mat_*_seed*.tsv"))):
        for line in Path(p).read_text().splitlines()[1:]:
            f = line.split("\t")
            if len(f) < 9:
                continue
            rows.append(dict(scen=f[1], seed=f[2], score=float(f[3]),
                             weight=float(f[5]), steps=int(f[6]), term=f[7]))
    return rows


def main():
    rows = load()
    by = collections.defaultdict(dict)
    weight = {}
    for r in rows:
        by[r["scen"]][r["seed"]] = r["score"]
        weight[r["scen"]] = r["weight"]
    if len(rows) != 60:
        raise SystemExit(f"期望 60 局，实际 {len(rows)}")

    md = ["# v15 全矩阵评估（it3 + r9_search_enhanced）", "",
          "- 组合：`results/v15/v15_full20_20261019174553/checkpoints/iteration_0003.pt` + `r9_search_enhanced`（85/240/3）",
          "- 口径：clean，`--max-steps 3000`，目标静止，`LOCAL_AVOID_GATE_MODE=none`，60 局全部 `environment_done`、1801 步",
          "- seed：701 / 8101 / 9101；每图 3 局", "",
          "## 逐图成绩", "", "| 图 | 权重 | 701 | 8101 | 9101 | 均值 | 极差 |", "|---|---|---|---|---|---|---|"]
    csv = ["scenario,weight,seed701,seed8101,seed9101,mean,range"]
    for scen in sorted(by, key=lambda s: (TIERS[s[0]], s)):
        v = [by[scen].get(s) for s in SEEDS]
        m = st.mean(v); rng = max(v) - min(v)
        md.append(f"| {scen} | {weight[scen]:.0f} | " + " | ".join(f"{x:.2f}" for x in v)
                  + f" | **{m:.2f}** | {rng:.2f} |")
        csv.append(f"{scen},{weight[scen]:.0f}," + ",".join(f"{x:.4f}" for x in v)
                   + f",{m:.4f},{rng:.4f}")

    md += ["", "## 分档", "", "| 档位 | 图数 | 局数 | 均值 | 最低 | 最高 |", "|---|---|---|---|---|---|"]
    for t, name in TIERS.items():
        vs = [r["score"] for r in rows if r["scen"][0] == t]
        md.append(f"| {name} | {len({r['scen'] for r in rows if r['scen'][0] == t})} | {len(vs)} "
                  f"| **{st.mean(vs):.2f}** | {min(vs):.2f} | {max(vs):.2f} |")
    allv = [r["score"] for r in rows]
    md.append(f"| 全体 | 20 | 60 | **{st.mean(allv):.2f}** | {min(allv):.2f} | {max(allv):.2f} |")

    diffs = [by[s][seed] - v for (s, seed), v in OLD.items()]
    md += ["", "## 可复现性警告", "",
           "- 同组合、同 seed 跑两次分数不同：新矩阵与旧复评的 10 个重合单元差 "
           f"{min(diffs):+.2f} ~ {max(diffs):+.2f}（均值 {st.mean(diffs):+.2f}，sd {st.stdev(diffs):.2f}）。",
           "- 探针：`M03/seed701` 固定 `PYTHONHASHSEED=0` 跑两次均 **91.6667**（destroyed_weight 39.0），"
           "而随机 hash seed 为 86.11 → 不确定性来自 Python 哈希顺序（策略内部集合/字典遍历与并列裁决）。",
           "- 结论：逐局噪声 σ≈3 分，**单 seed 差 <5 分不可作为改进证据**；跨批次对比必须先固定 `PYTHONHASHSEED`。", ""]
    (OUT / "FULL_MATRIX.md").write_text("\n".join(md))
    (OUT / "full_matrix.csv").write_text("\n".join(csv) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
