from __future__ import annotations

import random


def set_python_seed(seed: int) -> None:
    """Seed the Python standard library RNG."""
    random.seed(seed)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    set_python_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
