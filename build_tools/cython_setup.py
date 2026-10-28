"""把 envengine 编译为 CPython 原生扩展模块（.pyd / .so）。

所有配置写死在下面的常量区，运行时不接受自定义参数。
必须在目标平台的 CPython 3.11 x64 下执行；Windows 侧先进入
"x64 Native Tools Command Prompt for VS 2019"（等价于调用 vcvarsall.bat x64）。

用法:
    python build_tools/cython_setup.py build_ext
"""

from __future__ import annotations

import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent.parent

# 参与编译的顶层包
PACKAGES = ["envengine"]
# 跳过 SWIG 生成壳与第三方原生库所在子树，这些源码必须保持原样
SKIP_DIRS = ["envengine/simulator/models", "envengine/sdk/Util"]
# Cython 生成的 C 文件目录
GEN_DIR = ROOT / "build" / "cython_gen"
# 编译产物目录（.pyd / .so 输出到这里）
DIST_DIR = ROOT / "build" / "envengine_dist"
# 编译中间产物目录（.obj / .lib）
BUILD_TEMP = ROOT / "build" / "cython_tmp"
# 编译并行度
THREADS = 8


def collect_sources() -> list[tuple[str, str]]:
    """返回 [(模块点分名, 源码相对路径)]。"""
    sources = []
    for package in PACKAGES:
        base = ROOT / package
        if not base.is_dir():
            raise SystemExit(f"[error] package not found: {base}")
        for py in sorted(base.rglob("*.py")):
            rel = py.relative_to(ROOT).as_posix()
            if "__pycache__" in rel:
                continue
            if any(rel.startswith(directory + "/") for directory in SKIP_DIRS):
                continue
            sources.append((rel[:-3].replace("/", "."), rel))
    return sources


def main() -> None:
    sources = collect_sources()
    if not sources:
        raise SystemExit("[error] no source file matched")

    os.chdir(ROOT)
    print(f"[info] modules to compile: {len(sources)}")
    setup(
        name="competition-platform-native",
        ext_modules=cythonize(
            [Extension(name, [source]) for name, source in sources],
            compiler_directives={"language_level": "3"},
            build_dir=str(GEN_DIR),
            nthreads=THREADS,
        ),
        options={"build_ext": {
            "build_lib": str(DIST_DIR),
            "build_temp": str(BUILD_TEMP),
        }},
    )


if __name__ == "__main__":
    main()
