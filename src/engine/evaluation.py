from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
import pandas as pd
import torch

from src.models.external.IFRNet.metric import calculate_psnr as ifrnet_calculate_psnr

VFI_METRICS: tuple[str, ...] = ("psnr",)


def compute_batch_psnr(img_gt: torch.Tensor, img_pred: torch.Tensor) -> torch.Tensor:
    mse = ((img_gt - img_pred) * (img_gt - img_pred)).mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse)


@dataclass
class TaskEvaluator:
    """Small evaluator that collects per-sample VFI metrics."""

    task_name: str
    metrics: tuple[str, ...]
    records: list[dict[str, float]] = field(default_factory=list)

    def evaluate_batch(self, img_gt: torch.Tensor, img_pred: torch.Tensor) -> None:
        psnr_batch = compute_batch_psnr(img_gt, img_pred)
        self.records.extend({"psnr": float(psnr)} for psnr in psnr_batch.detach().cpu().tolist())

    def evaluate(
        self,
        meta: dict[str, Any],
        img_gt: Any,
        img_pred: Any,
        flow_1_to_0: Any,
        flow_1_to_2: Any,
        bmv: Any,
        fmv: Any,
    ) -> None:
        gt_tensor = torch.from_numpy(img_gt).to(dtype=torch.float32).permute(2, 0, 1)
        pred_tensor = torch.from_numpy(img_pred).to(dtype=torch.float32).permute(2, 0, 1)
        psnr = float(ifrnet_calculate_psnr(gt_tensor, pred_tensor))
        self.records.append({"psnr": psnr})

    def to_dataframe(self) -> Any:
        return pd.DataFrame(self.records)
