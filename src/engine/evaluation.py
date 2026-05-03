from __future__ import annotations

from dataclasses import dataclass
import math


def calculate_psnr(img_gt: object, img_pred: object) -> float:
    import numpy as np

    gt_float = img_gt.astype(np.float32, copy=False)
    pred_float = img_pred.astype(np.float32, copy=False)
    mse = float(np.mean((gt_float - pred_float) ** 2))
    if mse == 0.0:
        return float("inf")

    return float(-10.0 * math.log10(mse))


@dataclass
class AverageMeter:
    sum: float = 0.0
    count: int = 0

    def update(self, value: float, n: int) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0

        return self.sum / self.count
