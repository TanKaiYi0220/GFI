from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist and return the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
