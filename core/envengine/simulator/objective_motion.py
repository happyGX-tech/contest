"""Runtime switch for blue objective movement."""

from __future__ import annotations

import os


ENV_NAME = "BLUE_OBJECTIVE_MOTION"
VALID_MODES = frozenset({"static", "rule"})


def objective_motion_mode() -> str:
    """Return the configured objective-motion mode, failing on typos."""

    mode = os.getenv(ENV_NAME, "rule").strip().lower()
    if mode not in VALID_MODES:
        choices = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"{ENV_NAME} must be one of: {choices}; got {mode!r}")
    return mode
