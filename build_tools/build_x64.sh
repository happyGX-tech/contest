#!/usr/bin/env bash
# ===========================================================================
#  envengine 原生编译（Linux x86_64）—— 只做两步: 编译 -> 补齐保留文件
#
#  用法:
#      bash build_tools/build_x64.sh
# ===========================================================================

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

echo "[1/2] 开始编译: .py -> .so"
python build_tools/cython_setup.py build_ext || {
    echo "[ERROR] 编译失败，请查看上方日志"
    exit 1
}

echo "[2/2] 编译完成，开始复制保留文件"
python build_tools/finalize_dist.py || {
    echo "[ERROR] 保留文件复制失败，请查看上方日志"
    exit 1
}

echo "全部完成，产物位于 build/envengine_dist"
