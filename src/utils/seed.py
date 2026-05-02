from __future__ import annotations

import random


def set_python_seed(seed: int) -> None:
    """Seed the Python standard library RNG."""
    random.seed(seed)
