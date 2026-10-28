"""Run main.py against the compiled envengine build.

The dist tree is put first on sys.path so `envengine` resolves to the compiled
.pyd modules, while the remaining packages stay on their source form.

Usage:
    python build_tools/run_compiled.py --scenario launch20/easy/E01 --max-steps 100 ...
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "build" / "envengine_dist"


def main() -> None:
    if not DIST.is_dir():
        raise SystemExit(f"[error] dist tree not found: {DIST}\n        run build_tools\build_x64.cmd first")
    if not any(DIST.rglob("*.pyd")) and not any(DIST.rglob("*.so")):
        raise SystemExit(f"[error] no compiled module in {DIST}")

    # dist 优先，root 兜底；其余顶层包（blue_defense / scenarios / user_agents）走源码
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(DIST))
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
