#!/usr/bin/env python3
"""HLopt4launch20 full-suite runner: launch20 全 20 图并行评测。

一局 = 一个独立的 ``run.py run`` 子进程（不改 ``core/``、不改 ``run.py``）。
每张图先用 ``tools/build_deployment_plan.py`` 生成 8x4 网格部署计划，然后带
``r11_lethal_package`` 策略与三件套开关跑 979 步。

并发模型（沿用 ``tools/parallel_eval.py``，见 local_plan/65）：
每局只占 1 核，``OMP_NUM_THREADS=1``，``CUDA_VISIBLE_DEVICES=""``（窗口模型在
CPU 上单次 act 约 1 ms，避免和其它 GPU 任务抢卡）。

用法::

    python tools/run_hlopt4launch20.py --jobs 20 --seed 701
    python tools/run_hlopt4launch20.py --map E01,E03 --jobs 2      # 子集重跑
    python tools/run_hlopt4launch20.py --dry                       # 只打印计划

结果：``results/runs/<suite>/seed_<seed>/<MAP>/<timestamp>/``，
汇总写到 ``results/<suite>/summary_seed<seed>_<stamp>.{json,md}``。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SCENARIO_ROOT = ROOT / "scenarios/cases/launch20"
PLAN_DIR = ROOT / "results/deployment_plans"
RUNS_ROOT = ROOT / "results/runs"
SUITE_NAME = os.environ.get("HLOPT_SUITE", "HLopt4launch20")
RED_POLICY = "r11_lethal_package"
MOTION_POLICY = "local_avoid_s3"
MODEL = "models/local_avoid_set_attention_v12_r3_timely_arrival.pt"
BLUE_POLICY = "b0_fixed_ratio_random"
MAX_STEPS = 979

SWITCHES = {
    "LOCAL_AVOID_GATE_MODE": "terminal_lockout",
    "R11_MEDIUM_FOCUS": "2",
    "R11_HEAVY_ETA_GUARD": "1",
    "R11_SATELLITE_STEP": "280",
}
# 实验变体只改环境变量：R11_* 与 LOCAL_AVOID_GATE_MODE 会被覆盖进子进程，
# 套件名用 HLOPT_SUITE 区分，保证旧基线目录不被覆盖。
SWITCHES.update({key: value for key, value in os.environ.items()
                 if key.startswith("R11_") or key == "LOCAL_AVOID_GATE_MODE"})


def all_maps() -> list[str]:
    return sorted({path.parent.name for path in SCENARIO_ROOT.glob("*/*/scenario.json")})


def scenario_arg(map_id: str) -> str:
    """run.py 的 ``--scenario`` 参数：相对 ``scenarios/cases`` 的目录，如 ``launch20/easy/E01``。"""
    for path in SCENARIO_ROOT.glob(f"*/*{map_id}/scenario.json"):
        return str(path.parent.relative_to(ROOT / "scenarios/cases"))
    raise SystemExit(f"scenario not found for {map_id}")


def plan_path(map_id: str) -> Path:
    return PLAN_DIR / f"{map_id}_{SUITE_NAME}.json"


def ensure_plan(map_id: str, dry: bool) -> Path:
    out = plan_path(map_id)
    if os.environ.get("HLOPT_PLAN_POLICY", "0").lower() not in {"0", "false", "off", "no"}:
        picked = subprocess.run(
            [PY, "tools/select_deployment_plan.py", "--map", map_id],
            cwd=ROOT, capture_output=True, text=True,
        )
        if picked.returncode != 0:
            raise SystemExit(f"plan policy failed for {map_id}: {picked.stderr.strip()}")
        return ROOT / picked.stdout.strip()
    if out.exists():
        return out
    scenario = SCENARIO_ROOT.glob(f"*/*{map_id}/scenario.json")
    scenario_path = next(iter(scenario))
    command = [PY, "tools/build_deployment_plan.py", "--scenario", str(scenario_path.relative_to(ROOT)),
               "--out", str(out.relative_to(ROOT))]
    if os.environ.get("HLOPT_MEDIUM_NEAR", "0").lower() not in {"0", "false", "off", "no"}:
        command.append("--medium-near")
    if dry:
        print("  build", " ".join(command))
        return out
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"plan build failed for {map_id}: {result.stderr[-800:]}")
    print(f"  plan {map_id} -> {out.relative_to(ROOT)}", flush=True)
    return out


def worker_env(switches: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    repo = str(ROOT)
    env["PYTHONPATH"] = f"{repo}:{repo}/core" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    env["LD_LIBRARY_PATH"] = (
        f"{repo}/core:{repo}/core/envengine/sdk/Util:/home/amax/miniconda3/envs/macomp/lib"
        + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else "")
    )
    env["OMP_NUM_THREADS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.update(switches)
    return env


def run_map(map_id: str, seed: int, switches: dict[str, str], plan: Path,
            timeout: int, dry: bool) -> dict:
    run_id = f"{SUITE_NAME}/seed_{seed}/{map_id}"
    log_path = Path(f"/tmp/{SUITE_NAME}_{map_id}_s{seed}.log")
    command = [
        PY, str(ROOT / "run.py"), "run",
        "--scenario", scenario_arg(map_id),
        "--red-policy", RED_POLICY,
        "--red-motion-policy", MOTION_POLICY,
        "--red-learning-model", MODEL,
        "--blue-policy", BLUE_POLICY,
        "--seed", str(seed),
        "--run-id", run_id,
        "--",
        "--total-rounds", "1",
        "--max-steps", str(MAX_STEPS),
        "--render-mode", "none",
        "--disable-log-color",
        "--enable-event",
        "--deployment-plan", str(plan.relative_to(ROOT)),
    ]
    if dry:
        print(" ", " ".join(command))
        return {"map": map_id, "dry": True}
    started = time.monotonic()
    with log_path.open("wb") as log:
        try:
            result = subprocess.run(command, cwd=ROOT, env=worker_env(switches),
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = -9
    elapsed = round(time.monotonic() - started, 1)
    found = sorted(RUNS_ROOT.glob(f"{run_id}/*/summary.json"))
    if not found:
        return {"map": map_id, "exit": exit_code, "seconds": elapsed,
                "summary": None, "log": str(log_path)}
    summary = json.loads(found[-1].read_text(encoding="utf-8"))["score"]
    return {
        "map": map_id, "exit": exit_code, "seconds": elapsed,
        "summary": str(found[-1].relative_to(ROOT)),
        "score": float(summary["score"]),
        "weight": f"{summary['destroyed_weight']}/{summary['total_objective_weight']}",
        "completed": bool(summary.get("completed")),
        "destroyed": summary.get("destroyed_ids", []),
        "log": str(log_path),
    }


def write_report(results: list[dict], seed: int, wall: float, stamp: str) -> tuple[Path, Path]:
    out_dir = ROOT / "results" / SUITE_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda item: item["map"])
    scores = [item["score"] for item in ordered if item.get("score") is not None]
    payload = {
        "suite": SUITE_NAME, "seed": seed, "max_steps": MAX_STEPS,
        "switches": SWITCHES, "wall_seconds": wall,
        "maps": len(ordered), "mean": round(sum(scores) / len(scores), 4) if scores else None,
        "results": ordered,
    }
    json_path = out_dir / f"summary_seed{seed}_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# {SUITE_NAME} 全量结果（seed {seed}）",
        "",
        f"> 自动生成于 {datetime.now().isoformat(timespec='seconds')}；wall {wall:.0f}s；"
        f"{MAX_STEPS} 步/局；策略 `{RED_POLICY}` + `{MOTION_POLICY}`；蓝方 `{BLUE_POLICY}`。",
        "",
        "| 地图 | 分数 | 权重 | 全灭 | 用时(s) |",
        "|---|---|---|---|---|",
    ]
    for item in ordered:
        score = f"{item['score']:.2f}" if item.get("score") is not None else "FAIL"
        lines.append(
            f"| {item['map']} | {score} | {item.get('weight', '-')} | "
            f"{'是' if item.get('completed') else '否'} | {item.get('seconds', '-')} |"
        )
    if scores:
        full = sum(1 for item in ordered if item.get("completed"))
        lines += ["", f"均值 **{sum(scores) / len(scores):.2f}**；"
                      f"满分 {full}/{len(ordered)}；失败 {len(ordered) - len(scores)}。"]
    md_path = out_dir / f"summary_seed{seed}_{stamp}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="", help="逗号分隔的地图 id，默认全部 20 张")
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=5400)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--switches", default="", help="覆盖开关，形如 K=V,K=V")
    args = parser.parse_args(argv)

    switches = dict(SWITCHES)
    for pair in filter(None, (item.strip() for item in args.switches.split(","))):
        key, _, value = pair.partition("=")
        switches[key.strip()] = value.strip()

    maps = [item.strip() for item in args.map.split(",") if item.strip()] or all_maps()
    print(f"{SUITE_NAME}: {len(maps)} maps, jobs={args.jobs}, seed={args.seed}, dry={args.dry}")
    plans = {map_id: ensure_plan(map_id, args.dry) for map_id in maps}
    if args.dry:
        for map_id in maps:
            run_map(map_id, args.seed, switches, plans[map_id], args.timeout, dry=True)
        return 0

    started = time.monotonic()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(run_map, map_id, args.seed, switches, plans[map_id], args.timeout, args.dry): map_id
            for map_id in maps
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result.get("score") is None:
                print(f"  FAIL  {result['map']} exit={result.get('exit')} {result.get('seconds')}s "
                      f"log={result.get('log')}", flush=True)
            else:
                print(f"  OK    {result['map']} score={result['score']:.2f} "
                      f"weight={result['weight']} steps={result['seconds']}s", flush=True)
    wall = round(time.monotonic() - started, 1)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path, md_path = write_report(results, args.seed, wall, stamp)
    print(f"\nwall {wall:.0f}s -> {json_path.relative_to(ROOT)} | {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
