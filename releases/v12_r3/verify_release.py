#!/usr/bin/env python3
"""Verify the pinned files in the V12-R3 release manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("MANIFEST.sha256")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    checked = 0
    for line_number, raw_line in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            failures.append(f"line {line_number}: malformed manifest entry")
            continue
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        checked += 1
        if actual != expected:
            failures.append(
                f"sha256 mismatch: {relative}: expected {expected}, got {actual}"
            )
    if failures:
        print("V12-R3 release verification FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"V12-R3 release verification passed: {checked} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
