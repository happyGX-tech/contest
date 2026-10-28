"""Run the full map x blue matrix for one or more checkpoints and aggregate it.

The output layout and the aggregate schema deliberately mirror the historical
400-game bundle (`results/evaluations_latest_s3/final20_all_b0_b9_v11_s3_vs_straight`):
a matrix cell is (map, blue_policy, arm), every game runs in its own process, and
per-game summaries land under `results/runs/<run_name>/seed_<seed>/<timestamp>/`.

Usage
-----
    python tools/evaluate_matrix.py \
        --run-name v13_all24 --model v13=models/local_avoid_ppo_v13_gate_aligned_s3_best.pt \
        --run-name tricks_all24 --model tricks=results/ab_tricks/tricks/ckpt_phase3_*.pt \
        --workers 12 --seed 701 --output-dir results/evaluations_all24

Resumable: a cell whose summary.json already exists is reused and marked cached.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BLUES = (
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
TIERS = ("easy", "medium", "hard")


def discover_maps(suite: str, only: list[str] | None = None) -> list[tuple[str, str]]:
    """Return (tier, map_name) for every map directory in the suite.

    ``only`` filters by map id (E01, M03, ...) so a cheap subset such as
    ``--maps E01 M03 H07`` can be measured through the same instrument as the
    full suite.
    """
    base = ROOT / "scenarios" / "cases" / suite
    wanted = {m.strip().upper() for m in (only or [])}
    found: list[tuple[str, str]] = []
    for tier in TIERS:
        tier_dir = base / tier
        if not tier_dir.is_dir():
            continue
        for path in sorted(tier_dir.iterdir()):
            if path.is_dir() and (path / "scenario.json").is_file():
                if wanted and path.name.upper() not in wanted:
                    continue
                found.append((tier, path.name))
    if not found:
        raise SystemExit(f"no maps found under {base}" + (f" matching {sorted(wanted)}" if wanted else ""))
    missing = wanted - {name.upper() for _, name in found}
    if missing:
        raise SystemExit(f"maps not present in suite '{suite}': {sorted(missing)}")
    return found


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ARM=path")
    arm, raw = value.split("=", 1)
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return arm, path


def existing_summary(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("**/summary.json"))
    return candidates[-1] if candidates else None


def run_cell(job: dict[str, Any]) -> dict[str, Any]:
    stem = f"{job['arm']}_{job['map'].lower()}_{job['blue'].split('_')[0]}_s{job['seed']}"
    run_id = f"{job['run_name']}/seed_{job['seed']:04d}/{stem}"
    run_dir = ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = existing_summary(run_dir)
    record: dict[str, Any] = {
        "arm": job["arm"], "tier": job["tier"], "map_id": job["map"], "blue_policy": job["blue"],
        "seed": job["seed"], "run_dir": str(run_dir.relative_to(ROOT)),
    }
    if summary_path is not None and not job["force"]:
        record.update(cached=True, summary_path=str(summary_path))
        return record
    command = [
        sys.executable, str(ROOT / "run.py"), "run",
        "--scenario", f"{job['suite']}/{job['tier']}/{job['map']}",
        "--red-policy", job["red_policy"],
        "--blue-policy", job["blue"],
        "--red-motion-policy", job["red_motion"],
    ]
    if job.get("model"):
        command.extend(["--red-learning-model", str(job["model"])])
    command.extend([
        "--seed", str(job["seed"]),
        "--run-id", run_id,
        "--", "--total-rounds", "1", "--render-mode", "none", "--disable-log-color",
    ])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    if job.get("model"):
        env["RED_LEARNING_MODEL"] = str(job["model"])
        env["EVASION_MODEL"] = str(job["model"])
    log_path = run_dir / "run.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log_file,
                                   stderr=subprocess.STDOUT, check=False)
    record["seconds"] = round(time.time() - started, 2)
    record["returncode"] = completed.returncode
    summary_path = existing_summary(run_dir)
    if summary_path is None:
        record["error"] = f"no summary.json (returncode {completed.returncode}); see {log_path.name}"
        return record
    record.update(cached=False, summary_path=str(summary_path))
    return record


def collect(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten a finished cell into the row shape used by the aggregate."""
    summary = json.loads(Path(record["summary_path"]).read_text(encoding="utf-8"))
    score = summary.get("score") or {}
    red = summary.get("red") or {}
    return {
        "arm": record["arm"], "tier": record["tier"], "map_id": record["map_id"],
        "blue_policy": record["blue_policy"], "seed": record["seed"],
        "cached": bool(record.get("cached")), "seconds": record.get("seconds"),
        "run_dir": record["run_dir"], "summary_path": record["summary_path"],
        "score": float(score.get("score", 0.0)),
        "K": float(score.get("K", 0.0)),
        "steps_executed": int(summary.get("steps_executed", 0)),
        "termination_reason": summary.get("termination_reason"),
        "total_objective_weight": float(score.get("total_objective_weight", 0.0)),
        "destroyed_weight": float(score.get("destroyed_weight", 0.0)),
        "launched": int(red.get("launched", 0)),
        "alive": int(red.get("alive", 0)),
        "lost": int(red.get("lost", 0)),
        "completed": bool(score.get("completed", False)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"games": 0}
    scores = [row["score"] for row in rows]
    return {
        "games": len(rows),
        "score_mean": statistics.fmean(scores),
        "score_min": min(scores),
        "score_max": max(scores),
        "score_median": statistics.median(scores),
        "score_stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "launched": sum(row["launched"] for row in rows),
        "lost": sum(row["lost"] for row in rows),
        "alive": sum(row["alive"] for row in rows),
        "completed": sum(1 for row in rows if row["completed"]),
    }


def build_comparison(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    arms = sorted({row["arm"] for row in rows})
    comparison: dict[str, Any] = {
        "protocol": protocol,
        "overall": {arm: summarize([r for r in rows if r["arm"] == arm]) for arm in arms},
        "by_tier": {}, "by_map": {}, "by_blue_policy": {}, "by_cell": {},
    }
    for arm in arms:
        for tier in TIERS:
            subset = [r for r in rows if r["arm"] == arm and r["tier"] == tier]
            if subset:
                comparison["by_tier"][f"{tier}|{arm}"] = summarize(subset)
        for map_id in sorted({r["map_id"] for r in rows if r["arm"] == arm}):
            subset = [r for r in rows if r["arm"] == arm and r["map_id"] == map_id]
            comparison["by_map"][f"{map_id}|{arm}"] = summarize(subset)
        for blue in sorted({r["blue_policy"] for r in rows if r["arm"] == arm}):
            subset = [r for r in rows if r["arm"] == arm and r["blue_policy"] == blue]
            comparison["by_blue_policy"][f"{blue}|{arm}"] = summarize(subset)
    for row in rows:
        key = f"{row['map_id']}|{row['blue_policy']}|{row['arm']}"
        comparison["by_cell"][key] = {**row}
    # Paired deltas against the first arm (the reference).
    if len(arms) >= 2:
        base_arm = arms[0]
        paired = {}
        for arm in arms[1:]:
            cells = {}
            for row in rows:
                if row["arm"] not in (base_arm, arm):
                    continue
                cells.setdefault((row["map_id"], row["blue_policy"]), {})[row["arm"]] = row["score"]
            deltas = [v[arm] - v[base_arm] for v in cells.values() if len(v) == 2]
            if deltas:
                paired[arm] = {
                    "vs": base_arm, "pairs": len(deltas),
                    "delta_mean": statistics.fmean(deltas),
                    "delta_min": min(deltas), "delta_max": max(deltas),
                    "wins": sum(1 for d in deltas if d > 0.01),
                    "losses": sum(1 for d in deltas if d < -0.01),
                    "ties": sum(1 for d in deltas if abs(d) <= 0.01),
                    "per_map": {},
                }
                for map_id in sorted({m for m, _ in cells}):
                    map_deltas = [v[arm] - v[base_arm] for (m, _), v in cells.items()
                                  if m == map_id and len(v) == 2]
                    if map_deltas:
                        paired[arm]["per_map"][map_id] = {
                            "pairs": len(map_deltas),
                            "delta_mean": statistics.fmean(map_deltas),
                            "wins": sum(1 for d in map_deltas if d > 0.01),
                            "losses": sum(1 for d in map_deltas if d < -0.01),
                        }
        comparison["paired"] = paired
    return comparison


def write_markdown(comparison: dict[str, Any], path: Path) -> None:
    protocol = comparison["protocol"]
    arms = sorted(comparison["overall"])
    lines = [
        "# 全地图矩阵：S3 权重对比",
        "",
        f"- 套件：`{protocol['suite']}`，{protocol['maps']} 张图 "
        f"（easy {protocol['tier_counts']['easy']} / medium {protocol['tier_counts']['medium']} / "
        f"hard {protocol['tier_counts']['hard']}）",
        f"- 蓝方：{protocol['blues']} 种；seed：{protocol['seed']}；"
        f"每臂 {protocol['expected_games']} 局（每局独立进程）",
        f"- 上层策略：`{protocol['red_policy']}`；运动层：`{protocol['red_motion']}`",
    ]
    for arm in arms:
        lines.append(f"- `{arm}` = `{protocol['models'][arm]}`")
    lines += ["", "## 总体", "", "| 臂 | 局数 | 均分 | 中位数 | 最低 | 最高 | 标准差 | 发射 | 损失 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        s = comparison["overall"][arm]
        lines.append(f"| {arm} | {s['games']} | {s['score_mean']:.3f} | {s['score_median']:.3f} | "
                     f"{s['score_min']:.3f} | {s['score_max']:.3f} | {s['score_stdev']:.3f} | "
                     f"{s['launched']} | {s['lost']} |")
    if "paired" in comparison:
        lines += ["", "## 配对对比（同图同蓝方）", "",
                  "| 臂 | 对比基准 | 配对数 | 均差 | 胜 | 负 | 平 |", "|---|---|---|---|---|---|---|"]
        for arm, block in sorted(comparison["paired"].items()):
            lines.append(f"| {arm} | {block['vs']} | {block['pairs']} | {block['delta_mean']:+.3f} | "
                         f"{block['wins']} | {block['losses']} | {block['ties']} |")
    lines += ["", "## 按难度", "", "| 难度 | 臂 | 局数 | 均分 |", "|---|---|---|---|"]
    for tier in TIERS:
        for arm in arms:
            s = comparison["by_tier"].get(f"{tier}|{arm}")
            if s:
                lines.append(f"| {tier} | {arm} | {s['games']} | {s['score_mean']:.3f} |")
    lines += ["", "## 按地图", "", "| 图 | 难度 | " + " | ".join(arms) + " | 差值 |",
              "|---" * (2 + len(arms) + 1) + "|"]
    for map_id in sorted({key.split("|")[0] for key in comparison["by_map"]}):
        values = [comparison["by_map"].get(f"{map_id}|{arm}") for arm in arms]
        if not all(values):
            continue
        tier = next(row["tier"] for row in comparison["by_cell"].values() if row["map_id"] == map_id)
        cells = " | ".join(f"{v['score_mean']:.2f}" for v in values)
        delta = values[-1]["score_mean"] - values[0]["score_mean"] if len(values) > 1 else 0.0
        lines.append(f"| {map_id} | {tier} | {cells} | {delta:+.2f} |")
    lines += ["", "## 按蓝方", "", "| 蓝方 | " + " | ".join(arms) + " | 差值 |",
              "|---" * (1 + len(arms) + 1) + "|"]
    for blue in sorted({key.split("|")[0] for key in comparison["by_blue_policy"]}):
        values = [comparison["by_blue_policy"].get(f"{blue}|{arm}") for arm in arms]
        if not all(values):
            continue
        cells = " | ".join(f"{v['score_mean']:.2f}" for v in values)
        delta = values[-1]["score_mean"] - values[0]["score_mean"] if len(values) > 1 else 0.0
        lines.append(f"| {blue} | {cells} | {delta:+.2f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", default=[], type=parse_model,
                        metavar="ARM=PATH",
                        help="repeat once per arm; omit only for a checkpoint-free motion "
                             "layer such as --red-motion straight")
    parser.add_argument("--suite", default="final20")
    parser.add_argument("--maps", nargs="+", default=None, metavar="MAP",
                        help="restrict to these map ids (E01 M03 H07 ...); default is the whole suite")
    parser.add_argument("--blues", nargs="*", default=list(DEFAULT_BLUES))
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--red-policy", default="r9_hierarchical_learning")
    parser.add_argument("--red-motion", default="local_avoid_s3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="aggregate bundle root (default: results/evaluations_<suite>)",
    )
    parser.add_argument("--name", default=None, help="bundle name (default: timestamp)")
    parser.add_argument("--force", action="store_true", help="re-run cells that already have a summary")
    args = parser.parse_args(argv)

    models = dict(args.model)
    if len(models) != len(args.model):
        raise SystemExit("duplicate arm names")
    if not models:
        models = {args.red_motion: None}
    for arm, path in models.items():
        if path is None and args.red_motion == "local_avoid_s3":
            raise SystemExit(f"arm '{arm}' needs a checkpoint: --model {arm}=<path>")
    maps = discover_maps(args.suite, args.maps)
    tier_counts = {tier: sum(1 for t, _ in maps if t == tier) for tier in TIERS}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"{'_vs_'.join(sorted(models))}_{stamp}"
    output_dir = args.output_dir or Path("results") / f"evaluations_{args.suite}"
    bundle = ROOT / output_dir / name
    bundle.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for arm, model in models.items():
        run_name = f"matrix_{name}_{arm}"
        for tier, map_id in maps:
            for blue in args.blues:
                jobs.append({
                    "arm": arm, "model": str(model), "suite": args.suite, "tier": tier,
                    "map": map_id, "blue": blue, "seed": args.seed, "run_name": run_name,
                    "red_policy": args.red_policy, "red_motion": args.red_motion,
                    "force": args.force,
                })
    protocol = {
        "name": name, "suite": args.suite, "maps": len(maps), "tier_counts": tier_counts,
        "blues": len(args.blues), "seed": args.seed, "arms": sorted(models),
        "expected_games": len(jobs) // len(models),
        "red_policy": args.red_policy, "red_motion": args.red_motion,
        "models": {
            arm: ("(none)" if path is None else
                  str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path))
            for arm, path in models.items()
        },
        "isolated_process_per_game": True,
        "generated_at": stamp,
    }
    print(f"MATRIX_START {json.dumps({'name': name, 'jobs': len(jobs), 'workers': args.workers})}",
          flush=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            if "summary_path" not in record:
                failures.append(record)
                print(f"MATRIX_FAIL [{index}/{len(jobs)}] {record['arm']} {record['map_id']} "
                      f"{record['blue_policy']}: {record.get('error')}", flush=True)
                continue
            rows.append(collect(record))
            if index % 20 == 0 or index == len(jobs):
                print(f"MATRIX_PROGRESS {index}/{len(jobs)} elapsed={time.time()-started:.0f}s "
                      f"cached={sum(1 for r in rows if r['cached'])}", flush=True)
    comparison = build_comparison(rows, protocol)
    comparison["failures"] = failures
    (bundle / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    (bundle / "per_run.json").write_text(
        json.dumps({arm: [r for r in rows if r["arm"] == arm] for arm in sorted(models)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(comparison, bundle / "COMPARISON.md")
    elapsed = time.time() - started
    print(f"MATRIX_DONE {bundle} games={len(rows)} failures={len(failures)} "
          f"elapsed={elapsed:.0f}s", flush=True)
    for arm in sorted(models):
        s = comparison["overall"][arm]
        print(f"MATRIX_ARM {arm} games={s['games']} mean={s['score_mean']:.3f} "
              f"median={s['score_median']:.3f} min={s['score_min']:.3f}", flush=True)
    if "paired" in comparison:
        for arm, block in sorted(comparison["paired"].items()):
            print(f"MATRIX_PAIRED {arm} vs {block['vs']} delta={block['delta_mean']:+.3f} "
                  f"wins={block['wins']} losses={block['losses']} ties={block['ties']}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
