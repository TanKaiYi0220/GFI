from __future__ import annotations

from dataclasses import dataclass

def calculate_psnr(img1, img2):
    import torch

    psnr = -10 * torch.log10(((img1 - img2) * (img1 - img2)).mean())
    return psnr

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
