from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .external.IFRNet.models.IFRNet_Residual import Model as ExternalIFRNetResidualModel
from .external.IFRNet.models.IFRNet_Residual import pad_to_multiple

InitFlowPyramid = dict[int, tuple[Tensor, Tensor]]


def masked_area_flow_resize(
    flow: Tensor,
    mask: Tensor,
    scale_factor: float,
    epsilon: float,
) -> Tensor:
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape [B, 1, H, W], got {tuple(mask.shape)}")
    if flow.shape[0] != mask.shape[0] or flow.shape[-2:] != mask.shape[-2:]:
        raise ValueError(
            f"flow and mask dimensions do not match: flow={tuple(flow.shape)} mask={tuple(mask.shape)}"
        )
    if scale_factor == 1.0:
        return flow

    normalized_mask = mask.to(device=flow.device, dtype=flow.dtype)
    weighted_flow = F.interpolate(flow * normalized_mask, scale_factor=scale_factor, mode="area")
    valid_area = F.interpolate(normalized_mask, scale_factor=scale_factor, mode="area")
    return weighted_flow / (valid_area + epsilon) * scale_factor


def build_masked_area_pyramid_init_flows(
    init_flow0_full: Tensor,
    init_flow1_full: Tensor,
    init_flow0_mask: Tensor,
    init_flow1_mask: Tensor,
    scale_factor: float,
    epsilon: float,
) -> InitFlowPyramid:
    level_scales: dict[int, float] = {
        1: scale_factor,
        2: scale_factor * 0.5,
        3: scale_factor * 0.25,
        4: scale_factor * 0.125,
    }
    return {
        level: (
            masked_area_flow_resize(init_flow0_full, init_flow0_mask, level_scale, epsilon),
            masked_area_flow_resize(init_flow1_full, init_flow1_mask, level_scale, epsilon),
        )
        for level, level_scale in level_scales.items()
    }


class Model(ExternalIFRNetResidualModel):
    def __init__(self, local_rank: int = -1, lr: float = 1e-4, init_flow_layer: int = 4) -> None:
        super().__init__(local_rank=local_rank, lr=lr, init_flow_layer=init_flow_layer)
        self._inference_init_flow0_mask: Tensor | None = None
        self._inference_init_flow1_mask: Tensor | None = None
        self._inference_init_flow_mask_epsilon: float = 1e-6

    def _get_init_flows(
        self,
        init_flow0: Tensor | None = None,
        init_flow1: Tensor | None = None,
        scale_factor: float = 1.0,
    ) -> InitFlowPyramid:
        if self._inference_init_flow0_mask is None and self._inference_init_flow1_mask is None:
            return super()._get_init_flows(
                init_flow0=init_flow0,
                init_flow1=init_flow1,
                scale_factor=scale_factor,
            )
        if init_flow0 is None or init_flow1 is None:
            raise ValueError("init_flow0 and init_flow1 must both be provided for masked-area downscaling.")
        if self._inference_init_flow0_mask is None or self._inference_init_flow1_mask is None:
            raise ValueError("init_flow0_mask and init_flow1_mask must either both be provided or both be omitted.")
        return build_masked_area_pyramid_init_flows(
            init_flow0_full=init_flow0,
            init_flow1_full=init_flow1,
            init_flow0_mask=self._inference_init_flow0_mask,
            init_flow1_mask=self._inference_init_flow1_mask,
            scale_factor=scale_factor,
            epsilon=self._inference_init_flow_mask_epsilon,
        )

    def inference(
        self,
        img0: Tensor,
        img1: Tensor,
        embt: Tensor,
        scale_factor: float = 1.0,
        init_flow0: Tensor | None = None,
        init_flow1: Tensor | None = None,
        init_flow0_mask: Tensor | None = None,
        init_flow1_mask: Tensor | None = None,
        init_flow_mask_epsilon: float = 1e-6,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if (init_flow0_mask is None) != (init_flow1_mask is None):
            raise ValueError("init_flow0_mask and init_flow1_mask must either both be provided or both be omitted.")

        padded_init_flow0_mask = None
        padded_init_flow1_mask = None
        if init_flow0_mask is not None and init_flow1_mask is not None:
            padded_init_flow0_mask, _ = pad_to_multiple(init_flow0_mask, multiple=8, mode="replicate")
            padded_init_flow1_mask, _ = pad_to_multiple(init_flow1_mask, multiple=8, mode="replicate")

        self._inference_init_flow0_mask = padded_init_flow0_mask
        self._inference_init_flow1_mask = padded_init_flow1_mask
        self._inference_init_flow_mask_epsilon = init_flow_mask_epsilon
        try:
            result: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] = super().inference(
                img0=img0,
                img1=img1,
                embt=embt,
                scale_factor=scale_factor,
                init_flow0=init_flow0,
                init_flow1=init_flow1,
            )
        finally:
            self._inference_init_flow0_mask = None
            self._inference_init_flow1_mask = None
            self._inference_init_flow_mask_epsilon = 1e-6
        return result

__all__ = ["Model"]
