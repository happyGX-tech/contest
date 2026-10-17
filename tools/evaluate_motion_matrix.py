#!/usr/bin/env python3
"""Evaluate straight flight and S3 over every final20 map and blue policy."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scenarios" / "cases" / "final20" / "manifest.json"
BLUE_POLICIES = (
    "b0_fixed_ratio_random",
    "b1_nearest_interceptor",
    "b2_threat_priority",
    "b3_min_cost_assignment",
    "b4_joint_timing_assignment",
    "b5_successive_depth_coordination",
    "b6_mixed_fire_coordination",
    "b7_planned_nearest",
    "b8_threat_multi_wave",
    "b9_min_cost_planned_multi_wave",
)
MOTIONS = ("straight", "local_avoid_s3")
COUNT_KEYS = (
    "launched",
    "engaged",
    "engaged_resolved",
    "engaged_censored",
    "evaded",
    "intercepted",
    "arrived",
    "sampled_damage_zone_100",
    "effective_hits",
    "alive_at_end",
)


@dataclass(frozen=True)
class Task:
    map_id: str
    tier: str
    scenario: str
    blue_policy: str
    motion: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seeds", default="701")
    parser.add_argument("--maps", help="Comma-separated map IDs; default is all final20 maps")
    parser.add_argument("--blue-policies", help="Comma-separated policies; default is B0-B9")
    parser.add_argument(
        "--motions",
        help="Comma-separated motion policies; default is straight,local_avoid_s3",
    )
    parser.add_argument(
        "--s3-checkpoint",
        default="models/local_avoid_ppo_v11_trusted_mission_s3_best.pt",
    )
    parser.add_argument(
        "--s3-gate-mode",
        choices=("none", "legacy", "mission_v12"),
        default="mission_v12",
        help="Runtime mission gate wrapped around the frozen S3 checkpoint.",
    )
    parser.add_argument("--run-prefix", default="eval_final20_all_b0_b9_v11_s3_vs_straight")
    parser.add_argument(
        "--trace-dir",
        help="Optional directory for compressed per-missile S3 runtime trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/evaluations_latest_s3/final20_all_b0_b9_v11_s3_vs_straight",
    )
    return parser.parse_args()


def load_maps(selected: set[str] | None) -> list[tuple[str, str, str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    maps = []
    for item in manifest["maps"]:
        map_id = str(item["map"])
        if selected is not None and map_id not in selected:
            continue
        tier = str(item["tier"])
        maps.append((map_id, tier, f"final20/{tier}/{map_id}"))
    if selected is not None and {item[0] for item in maps} != selected:
        missing = sorted(selected - {item[0] for item in maps})
        raise ValueError(f"Unknown map IDs: {', '.join(missing)}")
    return maps


def latest_summary(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("*/summary.json"), key=lambda path: path.stat().st_mtime)
    for path in reversed(candidates):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(summary.get("outcomes"), dict):
            return path
    return None


def task_slug(task: Task) -> str:
    blue = task.blue_policy.split("_", 1)[0]
    motion = "s3" if task.motion == "local_avoid_s3" else "straight"
    return f"{task.map_id.lower()}_{blue}_{motion}_s{task.seed:04d}"


def run_task(args: argparse.Namespace, task: Task, output_dir: Path) -> dict[str, Any]:
    slug = task_slug(task)
    run_id = f"{args.run_prefix}_{slug}"
    run_dir = ROOT / "results" / "runs" / run_id
    summary_path = latest_summary(run_dir)
    cached = summary_path is not None
    if summary_path is None:
        command = [
            sys.executable,
            str(ROOT / "run.py"),
            "run",
            "--scenario",
            task.scenario,
            "--blue-policy",
            task.blue_policy,
            "--red-policy",
            "r9_hierarchical_learning",
            "--red-motion-policy",
            task.motion,
            "--seed",
            str(task.seed),
            "--run-id",
            run_id,
        ]
        if task.motion == "local_avoid_s3":
            command.extend(["--red-learning-model", str((ROOT / args.s3_checkpoint).resolve())])
        command.extend(
            [
                "--",
                "--total-rounds",
                "1",
                "--render-mode",
                "none",
                "--disable-log-color",
            ]
        )
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        # Native simulator extensions link against the active interpreter's
        # shared libpython.  Non-interactive subprocesses do not inherit the
        # conda activation hook, so make that library location explicit.
        python_lib = str(Path(sys.executable).resolve().parents[1] / "lib")
        existing_library_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{python_lib}:{existing_library_path}"
            if existing_library_path
            else python_lib
        )
        if task.motion == "local_avoid_s3":
            env["LOCAL_AVOID_GATE_MODE"] = args.s3_gate_mode
            if args.trace_dir:
                trace_dir = (ROOT / args.trace_dir).resolve()
                env["LOCAL_AVOID_TRACE_PATH"] = str(
                    trace_dir / f"{slug}.jsonl.gz"
                )
                env["LOCAL_AVOID_TRACE_OVERWRITE"] = "1"
        with (log_dir / f"{slug}.log").open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        summary_path = latest_summary(run_dir)
        if completed.returncode != 0 or summary_path is None:
            return {
                **asdict(task),
                "error": f"exit={completed.returncode}, summary={summary_path}",
            }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outcomes = summary["outcomes"]
    return {
        **asdict(task),
        "cached": cached,
        "score": float(summary["score"]["score"]),
        "completed": bool(summary["score"]["completed"]),
        "steps": int(summary["steps_executed"]),
        "summary_path": str(summary_path),
        "outcomes": {key: outcomes.get(key) for key in outcomes},
    }


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    scores = [float(row["score"]) for row in rows]
    counts = {
        key: sum(int(row["outcomes"].get(key, 0)) for row in rows)
        for key in COUNT_KEYS
    }
    return {
        "games": len(rows),
        "score_mean": statistics.mean(scores),
        "score_min": min(scores),
        "score_max": max(scores),
        **counts,
        "evasion_rate": counts["evaded"] / max(counts["engaged_resolved"], 1),
        "conservative_evasion_rate": counts["evaded"] / max(counts["engaged"], 1),
        "arrival_rate": counts["arrived"] / max(counts["launched"], 1),
        "sampled_damage_zone_100_rate": counts["sampled_damage_zone_100"]
        / max(counts["launched"], 1),
        "hit_rate": counts["effective_hits"] / max(counts["arrived"], 1),
        "hit_rate_per_launch": counts["effective_hits"] / max(counts["launched"], 1),
    }


def grouped(records: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        key = "|".join(str(row[item]) for item in keys)
        buckets.setdefault(key, []).append(row)
    return {key: summarize(value) for key, value in sorted(buckets.items())}


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def comparison_row(
    label: str,
    straight: dict[str, Any] | None,
    s3: dict[str, Any] | None,
) -> str:
    if straight is None or s3 is None:
        return f"| {label} | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |"
    return (
        f"| {label} | {straight['score_mean']:.2f} | {s3['score_mean']:.2f} | "
        f"{s3['score_mean'] - straight['score_mean']:+.2f} | "
        f"{pct(straight['evasion_rate'])} | {pct(s3['evasion_rate'])} | "
        f"{pct(straight['arrival_rate'])} | {pct(s3['arrival_rate'])} | "
        f"{pct(straight['hit_rate'])} | {pct(s3['hit_rate'])} | "
        f"{pct(straight['hit_rate_per_launch'])} | {pct(s3['hit_rate_per_launch'])} |"
    )


def comparison_table_header(first_column: str) -> list[str]:
    return [
        f"| {first_column} | 直飞得分 | S3得分 | 得分差 | 直飞躲避率 | S3躲避率 | 直飞到达率 | S3到达率 | 直飞命中率 | S3命中率 | 直飞发射命中率 | S3发射命中率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    records = report["records"]
    valid = [row for row in records if "error" not in row]
    overall = report["overall"]
    scope_label = (
        f"{report['protocol']['map_count']}图 × "
        f"{report['protocol']['blue_policy_count']}策略"
    )
    blue_labels = ", ".join(
        str(value).split("_", 1)[0].upper()
        for value in report["protocol"]["blue_policies"]
    )
    lines = [
        "# Final20 S3 与直飞数据对比",
        "",
        f"- 地图：Final20 选定 {report['protocol']['map_count']} 张",
        f"- 蓝方：{blue_labels}，共 {report['protocol']['blue_policy_count']} 种",
        f"- 固定种子：{', '.join(str(value) for value in report['protocol']['seeds'])}",
        f"- 有效局数：{len(valid)} / {report['protocol']['expected_games']}",
        (
            f"- 成对比较：{report['protocol']['paired_games']} 局 / "
            f"{report['protocol']['paired_cases']} 组（同地图、同蓝方、同 seed）"
        ),
        "- 命中率：到达后造成指定目标实际耐久下降的比例",
        "- 发射命中率：有效命中数 / 总发射数",
        "",
        "## 总体",
        "",
        *comparison_table_header("范围"),
        comparison_row(scope_label, overall["straight"], overall["local_avoid_s3"]),
        "",
        "## 按难度",
        "",
        *comparison_table_header("难度"),
    ]
    by_tier = report["by_tier"]
    for tier in ("easy", "medium", "hard"):
        lines.append(
            comparison_row(tier, by_tier.get(f"{tier}|straight"), by_tier.get(f"{tier}|local_avoid_s3"))
        )

    lines.extend(["", "## 按地图", "", *comparison_table_header("地图")])
    by_map = report["by_map"]
    for map_id in report["protocol"]["maps"]:
        lines.append(
            comparison_row(map_id, by_map.get(f"{map_id}|straight"), by_map.get(f"{map_id}|local_avoid_s3"))
        )

    lines.extend(["", "## 按蓝方策略", "", *comparison_table_header("蓝方")])
    by_blue = report["by_blue_policy"]
    for blue_policy in report["protocol"]["blue_policies"]:
        label = blue_policy.split("_", 1)[0].upper()
        lines.append(
            comparison_row(
                label,
                by_blue.get(f"{blue_policy}|straight"),
                by_blue.get(f"{blue_policy}|local_avoid_s3"),
            )
        )

    lines.extend(["", "## 全矩阵", "", *comparison_table_header("地图 / 蓝方")])
    cells = report["by_cell"]
    for map_id in report["protocol"]["maps"]:
        for blue_policy in report["protocol"]["blue_policies"]:
            blue = blue_policy.split("_", 1)[0].upper()
            lines.append(
                comparison_row(
                    f"{map_id} / {blue}",
                    cells.get(f"{map_id}|{blue_policy}|straight"),
                    cells.get(f"{map_id}|{blue_policy}|local_avoid_s3"),
                )
            )
    if report["failures"]:
        lines.extend(["", "## 失败任务", ""])
        for row in report["failures"]:
            task = Task(**{key: row[key] for key in Task.__dataclass_fields__})
            lines.append(f"- `{task_slug(task)}`: {row['error']}")
    return "\n".join(lines) + "\n"


def write_outputs(args: argparse.Namespace, tasks: list[Task], records: list[dict[str, Any]]) -> None:
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = [row for row in records if "error" not in row]
    failures = [row for row in records if "error" in row]
    motions_by_case: dict[tuple[str, str, int], set[str]] = {}
    for row in valid:
        case = (str(row["map_id"]), str(row["blue_policy"]), int(row["seed"]))
        motions_by_case.setdefault(case, set()).add(str(row["motion"]))
    paired_cases = {
        case for case, motions in motions_by_case.items() if set(MOTIONS).issubset(motions)
    }
    paired = [
        row
        for row in valid
        if (str(row["map_id"]), str(row["blue_policy"]), int(row["seed"]))
        in paired_cases
    ]
    report = {
        "protocol": {
            "maps": list(dict.fromkeys(task.map_id for task in tasks)),
            "map_count": len({task.map_id for task in tasks}),
            "blue_policies": list(dict.fromkeys(task.blue_policy for task in tasks)),
            "blue_policy_count": len({task.blue_policy for task in tasks}),
            "motions": list(dict.fromkeys(task.motion for task in tasks)),
            "seeds": sorted({task.seed for task in tasks}),
            "expected_games": len(tasks),
            "valid_games": len(valid),
            "paired_games": len(paired),
            "paired_cases": len(paired_cases),
            "red_upper_policy": "r9_hierarchical_learning",
            "s3_checkpoint": args.s3_checkpoint,
            "s3_gate_mode": args.s3_gate_mode,
            "isolated_process_per_game": True,
        },
        "overall": {
            motion: (
                summarize(rows)
                if (rows := [row for row in paired if row["motion"] == motion])
                else None
            )
            for motion in MOTIONS
        },
        "by_tier": grouped(paired, ("tier", "motion")),
        "by_map": grouped(paired, ("map_id", "motion")),
        "by_blue_policy": grouped(paired, ("blue_policy", "motion")),
        "by_cell": grouped(paired, ("map_id", "blue_policy", "motion")),
        "failures": failures,
        "records": sorted(
            valid,
            key=lambda row: (row["map_id"], row["blue_policy"], row["seed"], row["motion"]),
        ),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "per_run.json").write_text(
        json.dumps(report["records"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "COMPARISON.md").write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    args = parse_args()
    selected_maps = set(filter(None, (args.maps or "").split(","))) or None
    maps = load_maps(selected_maps)
    policies = tuple(filter(None, (args.blue_policies or "").split(","))) or BLUE_POLICIES
    unknown_policies = sorted(set(policies) - set(BLUE_POLICIES))
    if unknown_policies:
        raise ValueError(f"Unknown blue policies: {', '.join(unknown_policies)}")
    motions = tuple(filter(None, (args.motions or "").split(","))) or MOTIONS
    unknown_motions = sorted(set(motions) - set(MOTIONS))
    if unknown_motions:
        raise ValueError(f"Unknown motions: {', '.join(unknown_motions)}")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    tasks = [
        Task(map_id, tier, scenario, policy, motion, seed)
        for map_id, tier, scenario in maps
        for policy in policies
        for seed in seeds
        for motion in motions
    ]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, args, task, output_dir): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            status = "failed" if "error" in record else "cached" if record.get("cached") else "done"
            print(f"PROGRESS {index}/{len(tasks)} {task_slug(futures[future])} {status}", flush=True)
    write_outputs(args, tasks, records)
    print(f"REPORT {output_dir / 'COMPARISON.md'}", flush=True)
    return 1 if any("error" in row for row in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
