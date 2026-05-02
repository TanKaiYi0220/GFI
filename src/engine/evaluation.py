from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from typing import Any

VFI_METRICS: tuple[str, ...] = ("psnr",)


@dataclass
class TaskEvaluator:
    """Small evaluator that collects per-sample VFI metrics."""

    task_name: str
    metrics: tuple[str, ...]
    records: list[dict[str, float]] = field(default_factory=list)

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
        del meta
        del flow_1_to_0
        del flow_1_to_2
        del bmv
        del fmv

        diff = img_gt.astype("float32") - img_pred.astype("float32")
        mse = float((diff * diff).mean())
        psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)
        self.records.append({"psnr": psnr})

    def to_dataframe(self) -> Any:
        import pandas as pd

        return pd.DataFrame(self.records)

