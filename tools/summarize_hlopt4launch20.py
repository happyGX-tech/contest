#!/usr/bin/env python3
"""汇总 ``HLopt4launch20`` 全量矩阵，沿用 ``results/v15/FULL_MATRIX.md`` 的记录风格。

每图一行 + 分档（easy/medium/hard）均值 + 存活缺口；数据来自
``results/runs/HLopt4launch20/seed_<seed>/<MAP>/<timestamp>/``。

    python tools/summarize_hlopt4launch20.py --seed 701
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios/cases/launch20"
SUITE = "HLopt4launch20"
TYPE_NAME = {21000: "H", 21001: "M", 21002: "L"}
KIND_NAME = {"target": "母巢", "site": "阵地", "ship": "暗船"}
BAND = {"E": "easy", "M": "medium", "H": "hard"}


def latest_summary(map_id: str, seed: int) -> Path | None:
    matches = sorted((ROOT / "results/runs" / SUITE / f"seed_{seed}" / map_id).glob("*/summary.json"))
    return matches[-1] if matches else None


def load_case_info(map_id: str) -> dict:
    for path in SCENARIOS.glob(f"*/*{map_id}/case_info.json"):
        info = json.loads(path.read_text(encoding="utf-8"))
        info["_roots"] = {item["id"]: item for item in info["roots"] if item.get("objective")}
        info["_band"] = BAND.get(path.parent.name[0].upper(), path.parent.name)
        return info
    return {"_roots": {}, "_band": "?"}


def ledger_kills(summary_path: Path) -> dict[str, int]:
    kills = {"H": 0, "M": 0, "L": 0}
    ledger = summary_path.parent / "1.json"
    if not ledger.is_file():
        return kills
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)["content"]["event"]
        entity = event.get("entity") or {}
        source = event.get("destroySrc") or {}
        if entity.get("type") == source.get("type"):
            continue
        name = TYPE_NAME.get(source.get("type"))
        if name:
            kills[name] += 1
    return kills


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    maps = sorted({path.parent.name for path in SCENARIOS.glob("*/*/scenario.json")})
    rows = []
    for map_id in maps:
        summary_path = latest_summary(map_id, args.seed)
        info = load_case_info(map_id)
        if summary_path is None:
            rows.append({"map": map_id, "band": info["_band"], "score": None})
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        score = payload["score"]
        destroyed = set(score.get("destroyed_ids", []))
        survivors = [
            f"{ident}{KIND_NAME.get(item.get('kind'), item.get('kind', ''))}"
            for ident, item in info["_roots"].items()
            if ident not in destroyed
        ]
        rows.append({
            "map": map_id,
            "band": info["_band"],
            "score": float(score["score"]),
            "weight": f"{score['destroyed_weight']:g}/{score['total_objective_weight']:g}",
            "completed": bool(score.get("completed")),
            "kills": ledger_kills(summary_path),
            "survivors": survivors,
            "steps": payload.get("steps_executed"),
        })

    scored = [row for row in rows if row.get("score") is not None]
    lines = [
        f"# {SUITE} 全量矩阵（seed {args.seed}）",
        "",
        "- 口径：`--max-steps 979`，`r11_lethal_package` + `local_avoid_s3`"
        "（`local_avoid_set_attention_v12_r3_timely_arrival.pt`），蓝方 `b0_fixed_ratio_random`；",
        "  开关 `R11_MEDIUM_FOCUS=2 R11_HEAVY_ETA_GUARD=1 R11_SATELLITE_STEP=280`，"
        "`LOCAL_AVOID_GATE_MODE=terminal_lockout`。",
        f"- 生成于 {datetime.now().isoformat(timespec='seconds')}；完成 {len(scored)}/{len(rows)} 图；"
        f"运行目录 `results/runs/{SUITE}/seed_{args.seed}/<MAP>/`。",
        "",
        "## 逐图成绩",
        "",
        "| 图 | 档 | 分数 | 权重 | 全灭 | H 杀 | M 杀 | L 杀 | 步数 | 存活缺口 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row.get("score") is None:
            lines.append(f"| {row['map']} | {row['band']} | FAIL | - | - | - | - | - | - | - |")
            continue
        kills = row["kills"]
        gaps = "、".join(row["survivors"]) or "-"
        lines.append(
            f"| {row['map']} | {row['band']} | {row['score']:.2f} | {row['weight']} | "
            f"{'是' if row['completed'] else '否'} | {kills['H']} | {kills['M']} | {kills['L']} | "
            f"{row['steps']} | {gaps} |"
        )

    lines += ["", "## 分档", "", "| 档位 | 图数 | 完成 | 均值 | 最低 | 最高 |", "| --- | --- | --- | --- | --- | --- |"]
    for band in ("easy", "medium", "hard"):
        subset = [row for row in scored if row["band"] == band]
        if not subset:
            continue
        values = [row["score"] for row in subset]
        lines.append(
            f"| {band} | {len(subset)} | {sum(1 for row in subset if row['completed'])} | "
            f"**{sum(values) / len(values):.2f}** | {min(values):.2f} | {max(values):.2f} |"
        )
    if scored:
        values = [row["score"] for row in scored]
        lines.append(
            f"| 全体 | {len(scored)} | {sum(1 for row in scored if row['completed'])} | "
            f"**{sum(values) / len(values):.2f}** | {min(values):.2f} | {max(values):.2f} |"
        )
    if len(scored) < len(rows):
        lines += ["", f"> 仍有 {len(rows) - len(scored)} 图在跑或失败，FAIL 行待补。"]

    text = "\n".join(lines) + "\n"
    out = args.out or (ROOT / "results" / SUITE / f"FULL_MATRIX_seed{args.seed}_{SUITE}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
