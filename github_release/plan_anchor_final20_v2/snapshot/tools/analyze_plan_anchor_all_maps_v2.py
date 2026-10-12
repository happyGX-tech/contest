#!/usr/bin/env python3
"""Analyze paired all-map PlanSpec-v2 anchor sensitivity results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / (
    "results/runs/plan_anchor_12x5_final20_all_v2_driver_logs/run_index.json"
)
DEFAULT_OUTPUT = ROOT / "results/analysis/plan_anchor_12x5_final20_all_v2"
BASELINE = "A00_balanced"


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _bootstrap_mean_ci(
    values: list[float], *, seed: int = 20261010, samples: int = 10000
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def _read_record(index_record: dict) -> dict:
    path = Path(index_record["summary"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    score = payload.get("score") or {}
    execution = payload.get("plan_execution") or {}
    telemetry = payload.get("episode_telemetry") or {}
    lifecycle = telemetry.get("red_lifecycle") or {}
    by_role = lifecycle.get("by_role") or {}
    hidden_trigger = execution.get("hidden_trigger")
    return {
        "scenario": index_record["scenario"],
        "scenario_id": index_record["scenario_id"],
        "difficulty": index_record["difficulty"],
        "anchor_id": index_record["anchor_id"],
        "seed": int(index_record["seed"]),
        "summary": str(path),
        "score": float(score.get("score", 0.0)),
        "completed": bool(score.get("completed", False)),
        "destroyed_count": len(score.get("destroyed_ids") or ()),
        "hidden_triggered": hidden_trigger is not None,
        "hidden_discoveries": len(
            (hidden_trigger or {}).get("target_ids") or ()
        ),
        "search_completions": len(
            execution.get("search_completions") or {}
        ),
        "semantic_retarget_violations": len(
            execution.get("semantic_retarget_violations") or ()
        ),
        "deployment_all_match": bool(
            (execution.get("deployment") or {}).get("all_match")
        ),
        "l_arrival_rate": float(
            (by_role.get("L") or {}).get("arrival_rate", 0.0)
        ),
        "l_intercept_rate": float(
            (by_role.get("L") or {}).get("intercept_rate", 0.0)
        ),
    }


def _summary_for_rows(rows: list[dict], anchor_id: str) -> dict:
    selected = [row for row in rows if row["anchor_id"] == anchor_id]
    scores = [row["score"] for row in selected]
    return {
        "runs": len(selected),
        "mean_score": _mean(scores),
        "sd_score": _sd(scores),
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "completed_rate": (
            sum(row["completed"] for row in selected) / len(selected)
            if selected else 0.0
        ),
        "all_hidden_trigger_rate": (
            sum(row["hidden_triggered"] for row in selected) / len(selected)
            if selected else 0.0
        ),
        "mean_search_completions": _mean([
            float(row["search_completions"]) for row in selected
        ]),
        "mean_l_arrival_rate": _mean([
            row["l_arrival_rate"] for row in selected
        ]),
        "mean_l_intercept_rate": _mean([
            row["l_intercept_rate"] for row in selected
        ]),
    }


def _paired_stats(rows: list[dict], anchor_id: str) -> dict:
    lookup = {
        (row["scenario_id"], row["seed"], row["anchor_id"]): row["score"]
        for row in rows
    }
    pairs = []
    for (scenario_id, seed, current_anchor), value in lookup.items():
        if current_anchor != anchor_id:
            continue
        baseline = lookup.get((scenario_id, seed, BASELINE))
        if baseline is not None:
            pairs.append(value - baseline)
    low, high = _bootstrap_mean_ci(pairs)
    tolerance = 1e-9
    return {
        "pairs": len(pairs),
        "mean_delta": _mean(pairs),
        "sd_delta": _sd(pairs),
        "ci95": [low, high],
        "wins": sum(value > tolerance for value in pairs),
        "ties": sum(abs(value) <= tolerance for value in pairs),
        "losses": sum(value < -tolerance for value in pairs),
    }


def _map_stats(rows: list[dict], anchors: list[dict]) -> dict:
    result = {}
    by_map: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_map[row["scenario_id"]].append(row)
    for scenario_id, selected in sorted(by_map.items()):
        baseline = _summary_for_rows(selected, BASELINE)
        result[scenario_id] = {
            "difficulty": selected[0]["difficulty"],
            "baseline": baseline,
            "anchors": {
                anchor["anchor_id"]: {
                    **_summary_for_rows(selected, anchor["anchor_id"]),
                    "paired": _paired_stats(selected, anchor["anchor_id"]),
                }
                for anchor in anchors
            },
        }
    return result


def _difficulty_stats(rows: list[dict], anchors: list[dict]) -> dict:
    result = {}
    for difficulty in ("easy", "medium", "hard"):
        selected = [row for row in rows if row["difficulty"] == difficulty]
        result[difficulty] = {
            anchor["anchor_id"]: {
                **_summary_for_rows(selected, anchor["anchor_id"]),
                "paired": _paired_stats(selected, anchor["anchor_id"]),
            }
            for anchor in anchors
        }
    return result


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.3f}"


def _write_report(result: dict, path: Path) -> None:
    lines = [
        "# final20 全图顶层计划敏感性实验",
        "",
        "## 协议",
        "",
        f"- 地图：{result['map_count']} 张 final20 地图；每个锚点每图 "
        f"{len(result['seeds'])} 个公共种子。",
        "- 底层固定：`local_avoid_s3`；蓝方固定："
        "`b0_fixed_ratio_random`, ratio=2。",
        "- 同一地图内所有锚点共享 H/M 公开目标分配、L 侦察比例和"
        "全暗巢发现门控；只改变锚点声明的部署、波次或编队字段。",
        "- 隐藏目标不出现在战前 token 中；全部合法暗巢发现后，预备 L "
        "才按不同暗巢分组释放；局内不换靶。",
        "",
        "## 全图配对结果",
        "",
        "| 锚点 | 语义 | 因素 | 均分 | SD | 相对 A00 配对均差 | "
        "95% bootstrap CI | 胜/平/负 | 全目标完成率 | 全暗巢触发率 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for anchor in result["anchors"]:
        row = result["overall"][anchor["anchor_id"]]
        paired = row["paired"]
        lines.append(
            f"| `{anchor['anchor_id']}` | {anchor['label']} | "
            f"{anchor['factor']} | {_fmt(row['mean_score'])} | "
            f"{_fmt(row['sd_score'])} | {_fmt(paired['mean_delta'])} | "
            f"[{_fmt(paired['ci95'][0])}, {_fmt(paired['ci95'][1])}] | "
            f"{paired['wins']}/{paired['ties']}/{paired['losses']} | "
            f"{100.0 * row['completed_rate']:.1f}% | "
            f"{100.0 * row['all_hidden_trigger_rate']:.1f}% |"
        )

    lines.extend(["", "## 分难度配对均差", ""])
    headers = " | ".join(
        f"{difficulty} delta" for difficulty in ("easy", "medium", "hard")
    )
    lines.extend([
        f"| 锚点 | {headers} |",
        "|---|---:|---:|---:|",
    ])
    for anchor in result["anchors"]:
        anchor_id = anchor["anchor_id"]
        values = [
            _fmt(result["by_difficulty"][difficulty][anchor_id]["paired"]["mean_delta"])
            for difficulty in ("easy", "medium", "hard")
        ]
        lines.append(f"| `{anchor_id}` | " + " | ".join(values) + " |")

    lines.extend(["", "## 各地图最佳锚点", ""])
    lines.extend([
        "| 地图 | 难度 | A00 均分 | 最佳锚点 | 最佳均分 | 相对 A00 |",
        "|---|---|---:|---|---:|---:|",
    ])
    for scenario_id, map_row in result["by_map"].items():
        best_anchor, best = max(
            map_row["anchors"].items(),
            key=lambda item: (item[1]["mean_score"], item[0] == BASELINE),
        )
        baseline_score = map_row["baseline"]["mean_score"]
        lines.append(
            f"| `{scenario_id}` | {map_row['difficulty']} | "
            f"{_fmt(baseline_score)} | `{best_anchor}` | "
            f"{_fmt(best['mean_score'])} | "
            f"{_fmt(best['mean_score'] - baseline_score)} |"
        )

    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 配对均差和置信区间用于判断跨图稳定关系；单图最高分只用于定位"
        "后续细分区间，不直接作为通用 skill。",
        "- 如果全暗巢触发率低或搜索完成数接近零，部署/波次结论只能解释"
        "当前搜索执行器下的系统效果，不能声称完成了严格全域搜索。",
        "- 该实验是确定性计划敏感性实验，不是 MAT 在线学习结果。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(index_path: Path, output: Path) -> dict:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if len(index["records"]) != int(index["expected_runs"]):
        raise ValueError(
            f"incomplete index: {len(index['records'])}/"
            f"{index['expected_runs']} records"
        )
    failed = [row for row in index["records"] if row["status"] == "failed"]
    if failed:
        raise ValueError(f"cannot analyze {len(failed)} failed runs")
    rows = [_read_record(record) for record in index["records"]]
    if any(row["semantic_retarget_violations"] for row in rows):
        raise ValueError("semantic retarget violations found")
    if not all(row["deployment_all_match"] for row in rows):
        raise ValueError("deployment mismatches found")

    plans = Path(index["plans"])
    root_manifest = json.loads(
        (plans / "manifest.json").read_text(encoding="utf-8")
    )
    anchors = list(root_manifest["anchors"])
    overall = {
        anchor["anchor_id"]: {
            **_summary_for_rows(rows, anchor["anchor_id"]),
            "paired": _paired_stats(rows, anchor["anchor_id"]),
        }
        for anchor in anchors
    }
    result = {
        "experiment": root_manifest["experiment"],
        "run_index": str(index_path),
        "protocol_identity": index["protocol_identity"],
        "seeds": index["seeds"],
        "run_count": len(rows),
        "map_count": len({row["scenario_id"] for row in rows}),
        "anchors": anchors,
        "overall": overall,
        "by_difficulty": _difficulty_stats(rows, anchors),
        "by_map": _map_stats(rows, anchors),
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(result, output / "report.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.index.resolve(), args.output.resolve())
    print(json.dumps({
        "run_count": result["run_count"],
        "report": str((args.output.resolve() / "report.md")),
        "analysis": str((args.output.resolve() / "analysis.json")),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
