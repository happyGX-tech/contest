#!/usr/bin/env python3
"""v15 训练实时进度看板（只读，不干扰训练进程）。

用法:
    bash tools/v15_progress.sh            # 或 python tools/v15_progress.py
    ... --interval 5                      # 刷新间隔
    ... --once                            # 只打一次
    ... --log results/v15/logs/xxx.log    # 指定日志（默认自动找最新）
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINER = "tools/train_local_avoid_full_sim.py"


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def latest_log() -> Path | None:
    logs = sorted(Path(ROOT / "results/v15/logs").glob("*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def read_text(path: Path) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []


def parse_log(lines: list[str]) -> dict:
    info: dict = {"iter": None, "total": None, "workers": 0, "complete": None,
                  "evals": [], "final": [], "done": False, "error": None}
    for line in lines:
        m = re.match(r"ITERATION (\d+)/(\d+) COLLECT", line)
        if m:
            info.update(iter=int(m.group(1)), total=int(m.group(2)),
                        workers=0, complete=None, evals=[], final=[])
            continue
        if line.startswith("WORKER "):
            info["workers"] += 1
            continue
        m = re.match(r"ITERATION_COMPLETE (\d+)/(\d+) samples=(\d+) step=(\d+)", line)
        if m:
            info["complete"] = (int(m.group(1)), int(m.group(3)), int(m.group(4)))
            continue
        m = re.match(r"EVAL (\S+) score=([-\d.]+)", line)
        if m:
            info["evals"].append((m.group(1), float(m.group(2))))
            continue
        m = re.match(r"EVAL_FINAL mean_score=([-\d.]+)", line)
        if m:
            info["final"].append(float(m.group(1)))
            continue
        if line.startswith("PIPELINE_COMPLETE"):
            info["done"] = True
        if re.search(r"Traceback|Segmentation fault|Error:", line):
            info["error"] = line[:160]
    return info


def run_dir(log_path: Path) -> Path | None:
    """按日志与结果目录的 mtime 就近匹配（脚本启动到训练器建目录会差几秒）。"""
    cands = list(Path(ROOT / "results/v15").glob("v15_full20_*"))
    if not cands:
        return None
    ref = log_path.stat().st_mtime
    best = min(cands, key=lambda d: abs(d.stat().st_mtime - ref))
    return best if abs(best.stat().st_mtime - ref) < 600 else None


def tier_of(name: str) -> str:
    return {"E": "easy", "M": "medium", "H": "hard"}.get(name[:1], "?")


def fmt_dur(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def collect_status(trainer_pid: str, pgrep_workers: int, info: dict,
                   run: Path | None, elapsed: float, rss_mb: float,
                   free_gb: float, gpu: str) -> dict:
    stat = {"alive": bool(trainer_pid), "workers": pgrep_workers, "rss_mb": rss_mb,
            "free_gb": free_gb, "gpu": gpu, "elapsed": elapsed}
    if run is None:
        return stat
    it = info.get("iter") or 1
    tdir = run / "traces" / f"iteration_{it:04d}"
    files = sorted(tdir.glob("*.jsonl.gz")) if tdir.exists() else []
    stat["trace_files"] = len(files)
    stat["trace_mb"] = sum(f.stat().st_size for f in files) / 1e6
    stat["tiers"] = {t: {"done": 0, "total": 0, "mb": 0.0} for t in ("easy", "medium", "hard")}
    for f in files:
        t = tier_of(f.name)
        if t in stat["tiers"]:
            stat["tiers"][t]["done"] += 1
            stat["tiers"][t]["mb"] += f.stat().st_size / 1e6
    counts = run / ("workers_iteration_%04d.json" % it)
    if counts.exists():
        stat["expected_workers"] = counts.read_text().count("status")
    ckpts = list((run / "checkpoints").glob("*.pt")) if (run / "checkpoints").exists() else []
    stat["checkpoints"] = len(ckpts)
    # 轮次起点：第 1 轮用目录名里的时间戳；之后用上一轮 checkpoint 的 mtime
    round_start = None
    if it == 1:
        m = re.search(r"_(\d{8})_(\d{6})$", run.name)
        if m:
            round_start = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    else:
        prev = run / "checkpoints" / f"iteration_{it - 1:04d}.pt"
        if prev.exists():
            round_start = datetime.fromtimestamp(prev.stat().st_mtime)
    if round_start is None and files:
        round_start = datetime.fromtimestamp(min(f.stat().st_mtime for f in files))
    stat["round_start"] = round_start
    return stat


def render(info: dict, stat: dict, log_path: Path, run: Path | None, interval: int) -> str:
    out = []
    bar = "=" * 72
    out.append(bar)
    out.append(f"v15_search_enhanced   {datetime.now():%H:%M:%S}   刷新间隔 {interval}s")
    out.append(bar)
    alive = "● 运行中" if stat["alive"] else "○ 已退出"
    out.append(f"训练进程 : {alive}"
               + (f"  已跑 {fmt_dur(stat['elapsed'])} · RSS {stat['rss_mb'] / 1024:.1f} GB" if stat["alive"] else ""))
    out.append(f"worker   : {stat['workers']} 个仿真子进程 · 可用内存 {stat['free_gb']:.0f} GB")
    out.append(f"GPU      : {stat['gpu']}")
    out.append(f"日志     : {log_path.name}")
    if run:
        out.append(f"结果目录 : {run.relative_to(ROOT)}")
    out.append("")

    it, total = info.get("iter"), info.get("total")
    if it:
        done_workers = info["workers"]
        expected = stat.get("expected_workers") or 20
        stage = "采集中"
        if info["complete"]:
            stage = "轮次完成"
        elif info["evals"]:
            stage = "评估中"
        out.append(f"进度     : 第 {it}/{total} 轮 · {stage} · worker 完成 {done_workers}/{expected}")
        if stat.get("round_start"):
            out.append(f"本轮耗时 : {fmt_dur((datetime.now() - stat['round_start']).total_seconds())}"
                       f" · trace {stat.get('trace_files', 0)} 个 / {stat.get('trace_mb', 0):.0f} MB")
        for tier, s in (stat.get("tiers") or {}).items():
            bar_len = 12
            out.append(f"  {tier:<7}: {'█' * min(bar_len, s['done']) + '·' * max(0, bar_len - s['done'])} "
                       f"{s['done']} 份 · {s['mb']:.0f} MB")
        if info["complete"]:
            i, samples, step = info["complete"]
            out.append(f"最近完成 : 第 {i} 轮 · samples={samples:,} · step={step:,}")
        for name, score in info["evals"][-4:]:
            out.append(f"评估     : {name} = {score:.2f}")
        for score in info["final"][-1:]:
            out.append(f"末轮评估 : mean_score = {score:.2f}")
        if stat.get("checkpoints"):
            out.append(f"checkpoint: {stat['checkpoints']} 个（每轮 1 个）")
        if info["done"]:
            out.append("状态     : ✅ PIPELINE_COMPLETE，训练已全部完成")
        if info["error"]:
            out.append(f"⚠️ 异常  : {info['error']}")
        elif not stat["alive"] and not info["done"]:
            out.append("⚠️ 状态  : 这份日志对应的训练进程已不在运行（日志停在上面最后一行）")
            out.append("           若训练已换到新 unit，看板会自动切到最新日志")
    else:
        out.append("进度     : 日志里还没有 ITERATION 行（刚启动）")

    out.append("")
    out.append("提示: 这个脚本只读日志/目录，Ctrl-C 退出不影响训练；")
    out.append("      训练用 setsid+nohup 脱离终端，关掉当前终端也不会被 kill。")
    out.append(bar)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else latest_log()
    if log_path is None:
        print("找不到日志：results/v15/logs/*.log", file=sys.stderr)
        return 1
    last_log: Path | None = None

    while True:
        if args.log is None:
            fresh = latest_log()
            if fresh is not None:
                log_path = fresh
        if log_path != last_log and last_log is not None:
            print(f"\n[看板] 已切换到更新的日志: {log_path.name}\n")
        last_log = log_path
        run = run_dir(log_path)
        lines = read_text(log_path)
        info = parse_log(lines)
        pid = sh(f'pgrep -f "{TRAINER}" | head -1')
        elapsed = 0.0
        rss_mb = 0.0
        if pid:
            stats = sh(f"ps -o etimes=,rss= -p {pid}").split()
            if len(stats) == 2:
                elapsed, rss_mb = float(stats[0]), float(stats[1]) / 1024
        nworkers = int(sh('pgrep -fc "run.py run" || echo 0') or 0)
        free_gb = 0.0
        for line in sh("free -g | awk 'NR==2{print $7}'").split():
            free_gb = float(line)
        gpu = sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\\n' ' '")
        stat = collect_status(pid, nworkers, info, run, elapsed, rss_mb, free_gb, gpu)
        stat["alive"] = bool(pid)
        # 若进程已退出但日志显示完成，交给渲染层判断
        if args.once:
            print(render(info, stat, log_path, run, args.interval))
            return 0
        print("\033[2J\033[H" + render(info, stat, log_path, run, args.interval), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
