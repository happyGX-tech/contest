#!/usr/bin/env python3
"""汇总 v15 复评（2026-10-19）的全部评估数字，生成 Markdown + 自包含 HTML 报告。

数据源：
  results/v15/ckpt_screen/*.tsv        —— 带扫描层的批量评估（label/scenario/seed/score）
  训练期 eval（无扫描层）               —— 结果目录 eval_logs，下面表中硬编码并注明来源
输出：
  results/v15/EVAL_SUMMARY.md
  results/v15/eval_summary.html
"""

from __future__ import annotations

import glob
import html
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v15"
CELLS = ["E01", "M03", "H07", "H01"]
SEEDS = ["701", "8101", "9101"]

# 训练期评估（无扫描层，训练器自带 eval，seed 701，3000 步）
TRAIN_EVAL = {
    "R3 基线（无扫描层）": {"E01": 100.00, "M03": 67.19, "H07": 62.44, "H01": 59.35},
    "v15 it3（训练中，无扫描层）": {"E01": 86.21, "M03": 64.25},
    "v15 it6（最终，无扫描层）": {"E01": 73.56, "M03": 67.89, "H07": 61.68},
}

# 搜索层整体增益（R3 权重，干净口径，见 doc 69 §14.6 与本次复核）
SEARCH_GAIN = {
    # 格: (无扫描层, 加扫描层, 来源)
    "E01": (100.00, 100.00, "29 权重，双方均触顶"),
    "M03": (59.94, 87.73, "三 seed 均值（R3 67.19/62.15/50.49）"),
    "H07": (62.44, 80.72, "本次同日对照（旧记录 83.13）"),
    "H01": (59.35, 82.73, "单 seed 701"),
}


def load_screen() -> dict[tuple[str, str, str], float]:
    rows: dict[tuple[str, str, str], float] = {}
    for path in glob.glob(str(OUT / "ckpt_screen/*.tsv")):
        for line in Path(path).read_text().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            rows[(parts[0], parts[1], parts[2])] = float(parts[3])
    return rows


def paired(rows, cell, label, seeds=SEEDS):
    picks = []
    for seed in seeds:
        if cell == "M03" and seed == "701":
            a, b = rows.get(("r3", cell, seed)), rows.get(("it3", cell, seed))
        else:
            a, b = rows.get((f"r3_{cell.lower()}", cell, seed)), rows.get((f"{label}_{cell.lower()}", cell, seed))
        if a is not None and b is not None:
            picks.append((seed, a, b, b - a))
    if len(picks) < 2:
        return None
    deltas = [p[3] for p in picks]
    n = len(deltas)
    mean = st.mean(deltas)
    sd = st.stdev(deltas) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}[n]
    return {
        "picks": picks,
        "mean": mean,
        "sd": sd,
        "se": se,
        "lo": mean - t * se,
        "hi": mean + t * se,
        "win": sum(1 for d in deltas if d > 0),
        "lose": sum(1 for d in deltas if d < 0),
    }


def fmt(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def main() -> None:
    rows = load_screen()

    # 筛选：M03 / seed 701
    screen_labels = [("R3", "r3"), ("it1", "it1"), ("it2", "it2"), ("it3", "it3"),
                     ("it4", "it4"), ("it5", "it5"), ("it6", "it6")]
    screen = [(name, rows.get((label, "M03", "701"))) for name, label in screen_labels]

    # seed 701 四格
    four = {}
    for name, label in (("R3", "r3"), ("it2", "it2"), ("it3", "it3")):
        four[name] = {c: rows.get((f"{label}_{c.lower()}", c, "701")) or rows.get((label, c, "701")) for c in CELLS}
    four["it6"] = {"E01": rows.get(("it6_e01", "E01", "701")), "M03": rows.get(("it6", "M03", "701"))}
    # R3 的 E01 沿用 doc 69 §14.6 的 clean 记录（本次筛选批次未复跑该格，所有 seed 均 100.00）
    four["R3"]["E01"] = 100.00

    paired_stats = {c: paired(rows, c, "it3") for c in ("H07", "M03", "H01")}

    md = ["# v15 复评汇总（2026-10-19）", "",
          "口径：clean（`--max-steps 3000`、蓝方 `b0_fixed_ratio_random` + 目标静止）；"
          "「带扫描层」= `--red-policy r9_search_enhanced`（85 车道 / 240 lead / 3 认领）。", "",
          "## 1 训练期评估（无扫描层，seed 701）", "",
          "| 配置 | E01 | M03 | H07 | H01 |", "|---|---|---|---|---|"]
    for name, vals in TRAIN_EVAL.items():
        md.append(f"| {name} | " + " | ".join(fmt(vals.get(c)) for c in CELLS) + " |")

    md += ["", "## 2 ckpt 筛选（带扫描层，M03，seed 701）", "",
           "| ckpt | " + " | ".join(n for n, _ in screen_labels) + " |",
           "|---|" + "---|" * len(screen_labels)]
    md.append("| M03 | " + " | ".join(fmt(v) for _, v in screen) + " |")

    md += ["", "## 3 seed 701 四格（带扫描层）", "", "| 配置 | E01 | M03 | H07 | H01 |", "|---|---|---|---|---|"]
    for name in ("R3", "it2", "it3", "it6"):
        md.append(f"| {name} | " + " | ".join(fmt(four[name].get(c)) for c in CELLS) + " |")

    md += ["", "## 4 三 seed 配对（it3 − R3，带扫描层）", "",
           "| 格 | 701 | 8101 | 9101 | Δ 均值 | sd | 95% CI | 胜/负 |", "|---|---|---|---|---|---|---|---|"]
    for cell, s in paired_stats.items():
        if not s:
            continue
        by_seed = {p[0]: p[3] for p in s["picks"]}
        md.append(f"| {cell} | " + " | ".join(fmt(by_seed.get(seed)) for seed in SEEDS[:3])
                  + f" | **{s['mean']:+.2f}** | {s['sd']:.2f} | [{s['lo']:+.2f}, {s['hi']:+.2f}] | {s['win']}/{s['lose']} |")

    md += ["", "## 5 搜索层整体增益（R3 权重，无扫描层 → 加扫描层）", "",
           "| 格 | 无扫描层 | 加扫描层 | Δ | 说明 |", "|---|---|---|---|---|"]
    for cell, (a, b, note) in SEARCH_GAIN.items():
        md.append(f"| {cell} | {a:.2f} | {b:.2f} | {b - a:+.2f} | {note} |")
    md += ["", "> 噪声：M03 seed 摆动可达 ±17 分、H 档单次运行 ±2~3 分；"
               "单 seed 差 <3 分不下结论。", ""]
    (OUT / "EVAL_SUMMARY.md").write_text("\n".join(md))

    # ---------- HTML ----------
    def table(headers, body, highlight=None):
        head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        out = [f"<table><thead><tr>{head}</tr></thead><tbody>"]
        for row in body:
            cls = ' class="hl"' if highlight and highlight(row) else ""
            out.append("<tr" + cls + ">" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
        out.append("</tbody></table>")
        return "".join(out)

    def td(v):
        return fmt(v) if isinstance(v, (int, float)) or v is None else html.escape(str(v))

    t1 = table(["配置", "E01", "M03", "H07", "H01"],
               [[html.escape(n)] + [td(v.get(c)) for c in CELLS] for n, v in TRAIN_EVAL.items()])
    t2 = table(["ckpt"] + [n for n, _ in screen_labels],
               [["M03（seed 701）"] + [td(v) for _, v in screen]],
               highlight=lambda r: r[0] == "M03（seed 701）")
    t3 = table(["配置", "E01", "M03", "H07", "H01"],
               [[n] + [td(four[n].get(c)) for c in CELLS] for n in ("R3", "it2", "it3", "it6")],
               highlight=lambda r: r[0] == "it3")
    body4 = []
    for cell, s in paired_stats.items():
        if not s:
            continue
        by_seed = {p[0]: p[3] for p in s["picks"]}
        body4.append([cell] + [f"{by_seed[seed]:+.2f}" if seed in by_seed else "—" for seed in SEEDS]
                     + [f"<b>{s['mean']:+.2f}</b>", f"{s['sd']:.2f}",
                        f"[{s['lo']:+.2f}, {s['hi']:+.2f}]", f"{s['win']}/{s['lose']}"])
    t4 = table(["格", "701", "8101", "9101", "Δ 均值", "sd", "95% CI", "胜/负"], body4,
               highlight=lambda r: r[0] == "H07")
    t5 = table(["格", "无扫描层", "加扫描层", "Δ", "说明"],
               [[c, f"{a:.2f}", f"{b:.2f}", f"<b>{b - a:+.2f}</b>", html.escape(note)]
                for c, (a, b, note) in SEARCH_GAIN.items()])

    # Δ ± 95% CI 条形图（SVG）
    chart_rows = []
    for cell, s in paired_stats.items():
        if s:
            chart_rows.append((cell, s["mean"], s["lo"], s["hi"]))
    lo_all = min(r[2] for r in chart_rows) if chart_rows else -10
    hi_all = max(r[3] for r in chart_rows) if chart_rows else 10
    pad = max(2.0, (hi_all - lo_all) * 0.15)
    lo_all, hi_all = lo_all - pad, hi_all + pad
    width, left, right, row_h = 760, 70, 700, 54
    def sx(v):
        return left + (v - lo_all) / (hi_all - lo_all) * (right - left)
    svg = [f'<svg viewBox="0 0 {width} {row_h * len(chart_rows) + 46}" role="img" aria-label="配对差与 95% 置信区间">']
    zero = sx(0)
    svg.append(f'<line x1="{zero:.1f}" y1="10" x2="{zero:.1f}" y2="{row_h * len(chart_rows) + 20}" stroke="#94a3b8" stroke-dasharray="4 4"/>')
    for i, (cell, mean, lo, hi) in enumerate(chart_rows):
        y = 32 + i * row_h
        color = "#15803d" if lo > 0 else ("#b91c1c" if hi < 0 else "#64748b")
        svg.append(f'<text x="8" y="{y + 4}" font-size="14" fill="#0f172a">{cell}</text>')
        svg.append(f'<line x1="{sx(lo):.1f}" y1="{y}" x2="{sx(hi):.1f}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        for edge in (lo, hi):
            svg.append(f'<line x1="{sx(edge):.1f}" y1="{y - 8}" x2="{sx(edge):.1f}" y2="{y + 8}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<circle cx="{sx(mean):.1f}" cy="{y}" r="6" fill="{color}"/>')
        svg.append(f'<text x="{sx(mean):.1f}" y="{y - 14}" font-size="13" text-anchor="middle" fill="{color}">'
                   f'{mean:+.2f}</text>')
    for tick in (lo_all, 0, hi_all):
        svg.append(f'<text x="{sx(tick):.1f}" y="{row_h * len(chart_rows) + 38}" font-size="12" '
                   f'text-anchor="middle" fill="#64748b">{tick:+.1f}</text>')
    svg.append("</svg>")

    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>v15 复评汇总 · 2026-10-19</title>
<style>
 body{{font-family:-apple-system,"Segoe UI","Noto Sans SC",sans-serif;margin:0;background:#f8fafc;color:#0f172a}}
 main{{max-width:980px;margin:0 auto;padding:32px 20px 64px}}
 h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:18px;margin:34px 0 10px}}
 p.note{{color:#475569;font-size:13px;line-height:1.7;margin:6px 0 0}}
 table{{border-collapse:collapse;width:100%;font-size:14px;background:#fff;box-shadow:0 1px 2px rgba(15,23,42,.08);border-radius:8px;overflow:hidden}}
 th,td{{padding:9px 12px;text-align:right;border-bottom:1px solid #eef2f7}}
 th:first-child,td:first-child{{text-align:left}}
 thead th{{background:#f1f5f9;font-weight:600;color:#334155}}
 tr.hl td{{background:#ecfdf5;font-weight:600}}
 svg{{width:100%;height:auto;background:#fff;border-radius:8px;box-shadow:0 1px 2px rgba(15,23,42,.08);margin-top:8px}}
 .tag{{display:inline-block;font-size:12px;padding:2px 8px;border-radius:999px;background:#e2e8f0;color:#334155;margin-left:8px;vertical-align:middle}}
</style></head><body><main>
<h1>v15 复评汇总 <span class="tag">2026-10-19</span></h1>
<p class="note">口径：clean（<code>--max-steps 3000</code>、蓝方 <code>b0_fixed_ratio_random</code> + 目标静止）。
「带扫描层」= <code>--red-policy r9_search_enhanced</code>（85 车道 / 240 lead / 3 认领）。
噪声：M03 seed 摆动可达 ±17 分，H 档单次运行 ±2~3 分。</p>

<h2>1 训练期评估（无扫描层，seed 701）</h2>{t1}
<p class="note">训练器自带 eval，与交付组合（权重 + 扫描层）不是同一口径，不能直接当验收。</p>

<h2>2 ckpt 筛选（带扫描层，M03，seed 701）</h2>{t2}
<p class="note">it2 / it3 并列 91.67，比 R3 的 83.33 高 8.33。</p>

<h2>3 seed 701 四格（带扫描层）</h2>{t3}
<p class="note">it2 在 H 档被推翻（H07 78.31、H01 77.51）；it3 四格全不弱于 R3。</p>

<h2>4 三 seed 配对（it3 − R3，带扫描层）</h2>{t4}
<p class="note">H07 三 seed 全正、95% CI 不跨 0；M03/H01 的 CI 跨 0，只能算「不弱于 R3」。</p>
{''.join(svg)}

<h2>5 搜索层整体增益（R3 权重）</h2>{t5}
<p class="note">规则层是 +18~+27 分的主力；网络从 R3 换到 it3 只在 H07 多拿 5.22 分。</p>
</main></body></html>"""
    (OUT / "eval_summary.html").write_text(html_doc)
    print("写出:", OUT / "EVAL_SUMMARY.md", "和", OUT / "eval_summary.html")
    print(json.dumps({c: {k: round(v, 2) for k, v in s.items() if k in ("mean", "lo", "hi", "win", "lose")}
                      for c, s in paired_stats.items() if s}, ensure_ascii=False))


if __name__ == "__main__":
    main()
