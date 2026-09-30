#!/usr/bin/env python3
"""Train mission-first S3 with authoritative satellite track provenance."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_local_avoid_ppo_v10_mission_s3 import main  # noqa: E402


if __name__ == "__main__":
    defaults = [
        "--resume",
        str(ROOT / "models/local_avoid_ppo_v10_dagger_s3.pt"),
        "--model",
        str(ROOT / "models/local_avoid_ppo_v11_trusted_mission_s3.pt"),
        "--log-dir",
        str(ROOT / "results/local_avoid_ppo/tensorboard_v11_trusted_mission"),
        # Preserve the V10 input ABI during warm start. Track provenance is
        # retained by the encoder, but explicit source features need a fresh
        # policy or a staged adapter instead of an abrupt distribution shift.
        "--preserve-normalizers",
        "--peer-track-probability",
        "0.25",
        "--satellite-track-probability",
        "0.50",
    ]
    raise SystemExit(main([*defaults, *sys.argv[1:]]))
