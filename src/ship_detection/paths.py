"""Resolve runtime files without embedding a machine-specific workspace path."""

import os
from pathlib import Path

_checkout = Path(__file__).resolve().parents[2]
ROOT = (
    Path(
        os.environ.get(
            "SHIP_DETECTION_HOME",
            str(_checkout if (_checkout / "pyproject.toml").is_file() else Path.cwd()),
        )
    )
    .expanduser()
    .resolve()
)
