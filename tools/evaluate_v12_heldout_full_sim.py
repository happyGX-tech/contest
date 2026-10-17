#!/usr/bin/env python3
"""Paired complete-simulator evaluation for V12-family checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

from evaluate_motion_matrix import BLUE_POLICIES, summarize
from evaluate_v8_vs_v13_full_sim import (
    DEFAULT_GATE_MODES,
    Condition,
    EvaluationTask,
    assert_runtime_environment_unchanged,
    assert_task_files_unchanged,
    atomic_json,
    canonical_json,
    commit_completed_task_sidecars,
    current_source_identities,
    file_sha256,
    load_maps,
    resolve_runtime_environment_identity,
    resolve_scenario_protocol,
    resolve_under_root,
    run_task,
    select_runtime,
    summarize_by_type,
    task_slug,
    terminate_all_active,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONDITION_ORDER = ("v9", "v12")
DEFAULT_GATE_MODES.setdefault("v12", "none")
DEFAULT_GATE_MODES.setdefault("v12_base", "none")
DEFAULT_GATE_MODES.setdefault("v12_r2", "none")
DEFAULT_GATE_MODES.setdefault("v12_r3", "none")
DEFAULT_GATE_MODES.setdefault("mlp2", "none")
DEFAULT_V9 = ROOT / "models/local_avoid_set_attention_v9_fullsim_static_w16.pt"
DEFAULT_V12 = ROOT / "models/local_avoid_set_attention_v12_fullsim_em_distance_r2.pt"
DEFAULT_V12_R2 = (
    ROOT / "models/local_avoid_set_attention_v12_fullsim_r2_reward_finetune.pt"
)
DEFAULT_V12_R3 = (
    ROOT / "models/local_avoid_set_attention_v12_r3_timely_arrival.pt"
)
DEFAULT_V13 = ROOT / "models/local_avoid_ppo_v13_gate_aligned_s3_best.pt"
DEFAULT_MLP2 = ROOT / "models/local_avoid_mlp2_v13_fullsim_em_distance_r1.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--kill-grace-seconds", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--runtime", choices=("auto", "host", "docker"), default="auto")
    parser.add_argument(
        "--docker-image", default="docker.m.daocloud.io/library/python:3.11-slim-trixie"
    )
    parser.add_argument("--maps", default="E06,M06")
    parser.add_argument("--seeds", default="11901,11902,11903")
    parser.add_argument("--blue-policies", default="b0_fixed_ratio_random")
    parser.add_argument("--objective-motion", choices=("static",), default="static")
    parser.add_argument(
        "--evaluation-label",
        default="E06/M06 Held-out",
        help="Human-readable label used in the Markdown report title.",
    )
    parser.add_argument(
        "--map-role",
        choices=("heldout", "trained"),
        default="heldout",
        help="Whether the selected maps were excluded from or included in training.",
    )
    parser.add_argument("--v9-checkpoint", type=Path, default=DEFAULT_V9)
    parser.add_argument("--v12-checkpoint", type=Path, default=DEFAULT_V12)
    parser.add_argument("--v12-r2-checkpoint", type=Path, default=DEFAULT_V12_R2)
    parser.add_argument("--v12-r3-checkpoint", type=Path, default=DEFAULT_V12_R3)
    parser.add_argument("--v13-checkpoint", type=Path, default=DEFAULT_V13)
    parser.add_argument("--mlp2-checkpoint", type=Path, default=DEFAULT_MLP2)
    parser.add_argument(
        "--comparison-mode",
        choices=(
            "v9-v12",
            "v12-only",
            "v12-r2-only",
            "v12-r2",
            "v12-r3-only",
            "mlp2",
        ),
        default="v9-v12",
    )
    parser.add_argument("--v9-gate-mode", choices=("none",), default="none")
    parser.add_argument("--v12-gate-mode", choices=("none",), default="none")
    parser.add_argument("--run-prefix", default="heldout_em_static_b0_v9_v12")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/evaluations_set_attention_v12/heldout_e06_m06_static_b0_s11901_11903",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.max_steps < 2 or args.timeout_seconds <= 0:
        parser.error("workers must be positive, max-steps > 1 and timeout > 0")
    return args


def paired_records(
    records: list[dict[str, Any]], condition_order: tuple[str, ...]
) -> list[dict[str, Any]]:
    by_case: dict[tuple[str, str, int], set[str]] = {}
    for row in records:
        case = (str(row["map_id"]), str(row["blue_policy"]), int(row["seed"]))
        by_case.setdefault(case, set()).add(str(row["condition"]))
    complete_cases = {
        case for case, labels in by_case.items() if set(condition_order).issubset(labels)
    }
    return [
        row
        for row in records
        if (str(row["map_id"]), str(row["blue_policy"]), int(row["seed"]))
        in complete_cases
    ]


def grouped(records: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        key = "|".join(str(row[item]) for item in keys)
        buckets.setdefault(key, []).append(row)
    return {key: summarize(rows) for key, rows in sorted(buckets.items())}


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def metric_row(label: str, value: dict[str, Any]) -> str:
    return (
        f"| {label.upper()} | {value['games']} | {value['score_mean']:.2f} | "
        f"{pct(value['evasion_rate'])} | {pct(value['arrival_rate'])} | "
        f"{pct(value['sampled_damage_zone_100_rate'])} | "
        f"{pct(value['hit_rate'])} | {pct(value['hit_rate_per_launch'])} | "
        f"{value['intercepted']} |"
    )


def delta_row(
    left_label: str,
    left: dict[str, Any],
    right_label: str,
    right: dict[str, Any],
) -> str:
    return (
        f"| {left_label.upper()} - {right_label.upper()} | - | "
        f"{left['score_mean'] - right['score_mean']:+.2f} | "
        f"{pct(left['evasion_rate'] - right['evasion_rate'])} | "
        f"{pct(left['arrival_rate'] - right['arrival_rate'])} | "
        f"{pct(left['sampled_damage_zone_100_rate'] - right['sampled_damage_zone_100_rate'])} | "
        f"{pct(left['hit_rate'] - right['hit_rate'])} | "
        f"{pct(left['hit_rate_per_launch'] - right['hit_rate_per_launch'])} | "
        f"{left['intercepted'] - right['intercepted']:+d} |"
    )


def render_markdown(report: dict[str, Any]) -> str:
    protocol = report["protocol"]
    condition_order = tuple(protocol["condition_order"])
    delta_pairs = tuple(tuple(item) for item in protocol["delta_pairs"])
    blue_labels = [item.split("_", 1)[0].upper() for item in protocol["blue_policies"]]
    blue_summary = (
        f"{blue_labels[0]}-{blue_labels[-1]}"
        if len(blue_labels) > 1
        else blue_labels[0]
    )
    lines = [
        f"# {protocol['comparison_label']} {protocol['evaluation_label']} 完整仿真评估（{blue_summary}）",
        "",
        f"- 状态：`{report['status']}`",
        f"- 有效局数：{protocol['valid_games']} / {protocol['expected_games']}",
        f"- 完整配对 case：{protocol['paired_cases']} / {protocol['expected_cases']}",
        f"- 地图：{', '.join(protocol['maps'])}"
        + ("（均参与训练）" if protocol["map_role"] == "trained" else "（均未参与训练）"),
        f"- Seeds：{', '.join(str(seed) for seed in protocol['seeds'])}",
        f"- 环境：`static + {blue_summary} + gate=none + max_steps={protocol['max_steps']}`",
        "- 上层固定为 `r9_hierarchical_learning`。",
        "- 指标来自完整仿真 MissionOutcomeTracker，不是训练代理奖励。",
        "",
        "## 总体",
        "",
        "| 条件 | 局数 | 均分 | 规避率 | 到达率(500m) | 进入圈率(采样100m) | 到达后命中率 | 每发命中率 | 被拦截数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in condition_order:
        lines.append(metric_row(condition, report["overall"][condition]))
    for left_label, right_label in delta_pairs:
        lines.append(
            delta_row(
                left_label,
                report["overall"][left_label],
                right_label,
                report["overall"][right_label],
            )
        )
    lines.extend(["", "## 按地图", ""])
    for map_id in protocol["maps"]:
        lines.extend(
            [
                f"### {map_id}",
                "",
                "| 条件 | 局数 | 均分 | 规避率 | 到达率(500m) | 进入圈率(采样100m) | 到达后命中率 | 每发命中率 | 被拦截数 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        values = {
            condition: report["by_condition_map"][f"{condition}|{map_id}"]
            for condition in condition_order
        }
        for condition in condition_order:
            lines.append(metric_row(condition, values[condition]))
        for left_label, right_label in delta_pairs:
            lines.append(
                delta_row(left_label, values[left_label], right_label, values[right_label])
            )
        lines.append("")
    lines.extend(["## 按地图难度", ""])
    for tier in sorted({item.split("/", 1)[0] if "/" in item else item for item in report["tiers"]}):
        lines.extend(
            [
                f"### {tier.capitalize()}",
                "",
                "| 条件 | 局数 | 均分 | 规避率 | 到达率(500m) | 进入圈率(采样100m) | 到达后命中率 | 每发命中率 | 被拦截数 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        values = {
            condition: report["by_condition_tier"][f"{condition}|{tier}"]
            for condition in condition_order
        }
        for condition in condition_order:
            lines.append(metric_row(condition, values[condition]))
        for left_label, right_label in delta_pairs:
            lines.append(
                delta_row(left_label, values[left_label], right_label, values[right_label])
            )
        lines.append("")
    lines.extend(["## 按蓝方策略", ""])
    for blue_policy in protocol["blue_policies"]:
        blue_label = blue_policy.split("_", 1)[0].upper()
        lines.extend(
            [
                f"### {blue_label}: `{blue_policy}`",
                "",
                "| 条件 | 局数 | 均分 | 规避率 | 到达率(500m) | 进入圈率(采样100m) | 到达后命中率 | 每发命中率 | 被拦截数 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        values = {
            condition: report["by_condition_blue"][f"{condition}|{blue_policy}"]
            for condition in condition_order
        }
        for condition in condition_order:
            lines.append(metric_row(condition, values[condition]))
        for left_label, right_label in delta_pairs:
            lines.append(
                delta_row(left_label, values[left_label], right_label, values[right_label])
            )
        lines.append("")
    lines.extend(
        [
            "## 按 H/M/L 型号",
            "",
            "| 型号 | 条件 | 发射数 | 到达率 | 到达后命中率 | 每发命中率 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for type_label in ("H", "M", "L"):
        for condition in condition_order:
            value = report["by_condition_type"][f"{condition}|{type_label}"]
            lines.append(
                f"| {type_label} | {condition.upper()} | {value['launched']} | "
                f"{pct(value['arrival_rate'])} | {pct(value['hit_rate'])} | "
                f"{pct(value['hit_rate_per_launch'])} |"
            )
    if report["failures"]:
        lines.extend(["", "## 失败任务", ""])
        for row in report["failures"]:
            lines.append(
                f"- `{row['condition']} {row['map_id']} seed={row['seed']}`: "
                f"{row['error']}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.runtime = select_runtime(args.runtime)
    selected_maps = {item for item in args.maps.split(",") if item}
    maps = load_maps(selected_maps)
    blue_policies = tuple(item for item in args.blue_policies.split(",") if item)
    unknown_blue = sorted(set(blue_policies) - set(BLUE_POLICIES))
    if unknown_blue:
        raise ValueError(f"unknown blue policies: {', '.join(unknown_blue)}")
    seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    if args.comparison_mode == "mlp2":
        condition_order = ("v13", "mlp2", "v12")
        delta_pairs = (("mlp2", "v13"), ("v12", "v13"), ("mlp2", "v12"))
        comparison_label = "MLP2 / V13 / V12"
        conditions = (
            Condition("v13", resolve_under_root(args.v13_checkpoint), "none", ""),
            Condition("mlp2", resolve_under_root(args.mlp2_checkpoint), "none", ""),
            Condition("v12", resolve_under_root(args.v12_checkpoint), "none", ""),
        )
    elif args.comparison_mode == "v12-only":
        condition_order = ("v12",)
        delta_pairs = ()
        comparison_label = "V12"
        conditions = (
            Condition("v12", resolve_under_root(args.v12_checkpoint), "none", ""),
        )
    elif args.comparison_mode == "v12-r2-only":
        condition_order = ("v12_r2",)
        delta_pairs = ()
        comparison_label = "V12-R2 Reward"
        conditions = (
            Condition(
                "v12_r2", resolve_under_root(args.v12_r2_checkpoint), "none", ""
            ),
        )
    elif args.comparison_mode == "v12-r2":
        condition_order = ("v12_base", "v12_r2")
        delta_pairs = (("v12_r2", "v12_base"),)
        comparison_label = "V12-R2 Reward / V12 Base"
        conditions = (
            Condition(
                "v12_base", resolve_under_root(args.v12_checkpoint), "none", ""
            ),
            Condition(
                "v12_r2", resolve_under_root(args.v12_r2_checkpoint), "none", ""
            ),
        )
    elif args.comparison_mode == "v12-r3-only":
        condition_order = ("v12_r3",)
        delta_pairs = ()
        comparison_label = "V12-R3 Timely Arrival"
        conditions = (
            Condition(
                "v12_r3", resolve_under_root(args.v12_r3_checkpoint), "none", ""
            ),
        )
    else:
        condition_order = DEFAULT_CONDITION_ORDER
        delta_pairs = (("v12", "v9"),)
        comparison_label = "V12 / V9"
        conditions = (
            Condition("v9", resolve_under_root(args.v9_checkpoint), "none", ""),
            Condition("v12", resolve_under_root(args.v12_checkpoint), "none", ""),
        )
    conditions = tuple(
        Condition(item.label, item.checkpoint, item.gate_mode, file_sha256(item.checkpoint))
        for item in conditions
    )
    runtime_environment_identity = resolve_runtime_environment_identity(
        args.runtime,
        args.docker_image,
    )
    if args.runtime == "docker":
        args.docker_run_reference = runtime_environment_identity[
            "docker_run_reference"
        ]
    scenario_protocols = {
        scenario: resolve_scenario_protocol(scenario, args.max_steps)
        for _map_id, _tier, scenario in maps
    }
    code_identity, observation_encoding_identity = current_source_identities()
    tasks = [
        EvaluationTask(
            condition=condition.label,
            checkpoint=str(condition.checkpoint),
            checkpoint_sha256=condition.checkpoint_sha256,
            gate_mode=condition.gate_mode,
            map_id=map_id,
            tier=tier,
            scenario=scenario,
            blue_policy=blue_policy,
            seed=seed,
            scenario_sha256=scenario_protocols[scenario][0],
            scenario_case_info_sha256=scenario_protocols[scenario][1],
            max_steps=scenario_protocols[scenario][2],
        )
        for map_id, tier, scenario in maps
        for blue_policy in blue_policies
        for seed in seeds
        for condition in conditions
    ]
    preview = {
        "maps": [item[0] for item in maps],
        "seeds": list(seeds),
        "blue_policies": list(blue_policies),
        "objective_motion": args.objective_motion,
        "evaluation_label": args.evaluation_label,
        "map_role": args.map_role,
        "comparison_mode": args.comparison_mode,
        "comparison_label": comparison_label,
        "condition_order": list(condition_order),
        "delta_pairs": [list(item) for item in delta_pairs],
        "max_steps": args.max_steps,
        "conditions": {
            item.label: {
                "checkpoint": str(item.checkpoint),
                "checkpoint_sha256": item.checkpoint_sha256,
                "gate_mode": item.gate_mode,
            }
            for item in conditions
        },
        "expected_games": len(tasks),
        "workers": args.workers,
        "runtime": args.runtime,
        "runtime_environment_identity": runtime_environment_identity,
        "code_identity": code_identity,
        "observation_encoding_identity": observation_encoding_identity,
    }
    preview["matrix_protocol_fingerprint"] = hashlib.sha256(
        canonical_json(preview).encode("utf-8")
    ).hexdigest()
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    output_dir = resolve_under_root(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "protocol.json", preview)
    atomic_json(
        output_dir / "run_status.json",
        {"status": "running", "completed_games": 0, "expected_games": len(tasks)},
    )
    records: list[dict[str, Any]] = []
    interrupted = False
    cancel_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(
            run_task,
            args,
            task,
            output_dir,
            cancel_event,
            runtime_environment_identity,
            code_identity,
            observation_encoding_identity,
        ): task
        for task in tasks
    }
    try:
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                record = future.result()
            except Exception as error:
                record = {**asdict(task), "error": f"{type(error).__name__}: {error}"}
            records.append(record)
            atomic_json(output_dir / "per_run.partial.json", records)
            atomic_json(
                output_dir / "run_status.json",
                {
                    "status": "running",
                    "completed_games": index,
                    "expected_games": len(tasks),
                    "last_task": task_slug(task),
                    "last_result": "failed" if "error" in record else "done",
                },
            )
            print(
                f"PROGRESS {index}/{len(tasks)} {task_slug(task)} "
                f"{'failed' if 'error' in record else 'done'}",
                flush=True,
            )
    except KeyboardInterrupt:
        interrupted = True
        terminate_all_active(
            args.kill_grace_seconds,
            cancel_event=cancel_event,
        )
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)
    if interrupted:
        atomic_json(
            output_dir / "run_status.json",
            {
                "status": "interrupted",
                "completed_games": len(records),
                "expected_games": len(tasks),
            },
        )
        return 130

    try:
        assert_runtime_environment_unchanged(args, runtime_environment_identity)
        final_code_identity, final_observation_identity = current_source_identities()
        if canonical_json(final_code_identity) != canonical_json(code_identity):
            raise RuntimeError("Simulator source tree changed during evaluation")
        if canonical_json(final_observation_identity) != canonical_json(
            observation_encoding_identity
        ):
            raise RuntimeError("Observation encoding changed during evaluation")
        checked_task_files: set[tuple[str, str]] = set()
        for task in tasks:
            key = (task.checkpoint, task.scenario)
            if key not in checked_task_files:
                assert_task_files_unchanged(task)
                checked_task_files.add(key)
        commit_completed_task_sidecars(
            records,
            args,
            runtime_environment_identity,
            code_identity,
            observation_encoding_identity,
        )
    except Exception as error:
        atomic_json(
            output_dir / "run_status.json",
            {
                "status": "protocol_guard_failed",
                "completed_games": len(records),
                "expected_games": len(tasks),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        return 1

    valid = [row for row in records if "error" not in row]
    failures = [row for row in records if "error" in row]
    paired = paired_records(valid, condition_order)
    overall = {
        condition: summarize([row for row in paired if row["condition"] == condition])
        for condition in condition_order
    }
    report = {
        "status": "complete" if len(valid) == len(tasks) and not failures else "partial",
        "protocol": {
            **preview,
            "valid_games": len(valid),
            "expected_cases": len(tasks) // len(condition_order),
            "paired_cases": len(paired) // len(condition_order),
        },
        "overall": overall,
        "tiers": sorted({str(row["tier"]) for row in paired}),
        "by_condition_map": grouped(paired, ("condition", "map_id")),
        "by_condition_tier": grouped(paired, ("condition", "tier")),
        "by_condition_blue": grouped(paired, ("condition", "blue_policy")),
        "by_condition_type": summarize_by_type(paired),
        "failures": failures,
    }
    atomic_json(output_dir / "comparison.json", report)
    atomic_json(output_dir / "per_run.json", records)
    (output_dir / "COMPARISON.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    atomic_json(
        output_dir / "run_status.json",
        {
            "status": report["status"],
            "completed_games": len(records),
            "valid_games": len(valid),
            "expected_games": len(tasks),
            "paired_cases": report["protocol"]["paired_cases"],
        },
    )
    print(f"V12_HELDOUT_EVALUATION {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
