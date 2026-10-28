"""把编译产物补齐成一个可直接 import 的完整包目录。

.pyd 只能替换 .py，包内的二进制依赖（.so / .pyd / .dll）、类型存根（.pyi）、
资源文件（.jpg / .png / .js）以及被排除编译的目录（SWIG 生成壳）都必须原样带上。

所有配置写死在下面的常量区，运行时不接受自定义参数。

用法:
    python build_tools/finalize_dist.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 参与的顶层包（与 cython_setup.py 的 PACKAGES 保持一致）
PACKAGES = ["envengine"]
# 产物目录（与 cython_setup.py 的 DIST_DIR 保持一致）
DIST_ROOT = ROOT / "build" / "envengine_dist"
# 这些子树下的 .py 原样保留（SWIG 生成壳，与 cython_setup.py 的 SKIP_DIRS 保持一致）
KEEP_PY_DIRS = ["envengine/simulator/models", "envengine/sdk/Util"]
# 运行期缓存
TRASH_DIRS = {"__pycache__"}
TRASH_SUFFIXES = {".pyc", ".pyo", ".c", ".cpp"}


def should_keep(path: Path, rel: str) -> bool:
    if any(part in TRASH_DIRS for part in path.parts):
        return False
    if path.suffix in TRASH_SUFFIXES:
        return False
    if path.suffix == ".py":
        return any(rel.startswith(directory + "/") for directory in KEEP_PY_DIRS)
    return True


def main() -> None:
    total = 0
    for package in PACKAGES:
        base = ROOT / package
        if not base.is_dir():
            raise SystemExit(f"[error] package not found: {base}")
        for src in sorted(base.rglob("*")):
            if src.is_dir():
                continue
            rel = src.relative_to(ROOT).as_posix()
            if not should_keep(src, rel):
                continue
            target = DIST_ROOT / rel
            if target.exists() and target.stat().st_mtime >= src.stat().st_mtime:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            total += 1
    print(f"[info] copied {total} non-source file(s) into {DIST_ROOT}")

    missing_init = [package for package in PACKAGES
                    if not any((DIST_ROOT / package).rglob("__init__.*"))]
    if missing_init:
        print(f"[warn] packages missing __init__ in dist: {missing_init}")
        sys.exit(1)


if __name__ == "__main__":
    main()
