from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .external.IFRNet.models.IFRNet_Residual import Model as ExternalIFRNetResidualModel
from .external.IFRNet.models.IFRNet_Residual import pad_to_multiple
from .external.IFRNet.utils import warp
from .convex_upsampling import crop_like
from .convex_upsampling import resize_bilinear
from .convex_upsampling import upsample_decoder_flow

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
    def __init__(
        self,
        local_rank: int = -1,
        lr: float = 1e-4,
        init_flow_layer: int = 4,
        convex_upsampling: bool = False,
    ) -> None:
        super().__init__(local_rank=local_rank, lr=lr, init_flow_layer=init_flow_layer)
        self.convex_upsampling = bool(convex_upsampling)
        self._inference_init_flow0_mask: Tensor | None = None
        self._inference_init_flow1_mask: Tensor | None = None
        self._inference_init_flow_mask_epsilon: float = 1e-6

    def set_convex_upsampling(self, enabled: bool) -> None:
        self.convex_upsampling = bool(enabled)

    def _prepare_runtime_init_flow_masks(
        self,
        init_flow0_mask: Tensor | None,
        init_flow1_mask: Tensor | None,
        init_flow_mask_epsilon: float,
    ) -> tuple[Tensor | None, Tensor | None]:
        if (init_flow0_mask is None) != (init_flow1_mask is None):
            raise ValueError("init_flow0_mask and init_flow1_mask must either both be provided or both be omitted.")
        if init_flow_mask_epsilon <= 0:
            raise ValueError(f"init_flow_mask_epsilon must be positive, got {init_flow_mask_epsilon}")
        if init_flow0_mask is None or init_flow1_mask is None:
            return None, None

        padded_init_flow0_mask, _ = pad_to_multiple(init_flow0_mask, multiple=8, mode="replicate")
        padded_init_flow1_mask, _ = pad_to_multiple(init_flow1_mask, multiple=8, mode="replicate")
        return padded_init_flow0_mask, padded_init_flow1_mask

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
        padded_init_flow0_mask, padded_init_flow1_mask = self._prepare_runtime_init_flow_masks(
            init_flow0_mask=init_flow0_mask,
            init_flow1_mask=init_flow1_mask,
            init_flow_mask_epsilon=init_flow_mask_epsilon,
        )

        self._inference_init_flow0_mask = padded_init_flow0_mask
        self._inference_init_flow1_mask = padded_init_flow1_mask
        self._inference_init_flow_mask_epsilon = init_flow_mask_epsilon
        try:
            height, width = img0.shape[-2:]

            mean = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
            img0 = img0 - mean
            img1 = img1 - mean

            img0, _ = pad_to_multiple(img0, multiple=8, mode="replicate")
            img1, _ = pad_to_multiple(img1, multiple=8, mode="replicate")
            if init_flow0 is None or init_flow1 is None:
                raise ValueError("init_flow0 and init_flow1 must both be provided.")
            init_flow0, _ = pad_to_multiple(init_flow0, multiple=8, mode="replicate")
            init_flow1, _ = pad_to_multiple(init_flow1, multiple=8, mode="replicate")

            img0_scaled = resize_bilinear(tensor=img0, scale_factor=scale_factor)
            img1_scaled = resize_bilinear(tensor=img1, scale_factor=scale_factor)

            f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_scaled)
            f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_scaled)

            init_flows = self._get_init_flows(
                init_flow0=init_flow0,
                init_flow1=init_flow1,
                scale_factor=scale_factor,
            )
            init0_1, init1_1 = init_flows[1]
            init0_2, init1_2 = init_flows[2]
            init0_3, init1_3 = init_flows[3]
            init0_4, init1_4 = init_flows[4]

            out4 = self.decoder4(f0_4, f1_4, embt)
            ft_3 = out4[:, :]
            flow0_4 = init0_4
            flow1_4 = init1_4

            out3 = self.decoder3(ft_3, f0_3, f1_3, flow0_4, flow1_4)
            dflow0_3 = out3[:, 0:2]
            dflow1_3 = out3[:, 2:4]
            ft_2 = out3[:, 4:]

            if self.init_flow_layer >= 2:
                flow0_3 = init0_3 + dflow0_3
                flow1_3 = init1_3 + dflow1_3
            else:
                flow0_3 = dflow0_3 + upsample_decoder_flow(flow0_4, ft_2, self.convex_upsampling)
                flow1_3 = dflow1_3 + upsample_decoder_flow(flow1_4, ft_2, self.convex_upsampling)

            out2 = self.decoder2(ft_2, f0_2, f1_2, flow0_3, flow1_3)
            dflow0_2 = out2[:, 0:2]
            dflow1_2 = out2[:, 2:4]
            ft_1 = out2[:, 4:]

            if self.init_flow_layer >= 3:
                flow0_2 = init0_2 + dflow0_2
                flow1_2 = init1_2 + dflow1_2
            else:
                flow0_2 = dflow0_2 + upsample_decoder_flow(flow0_3, ft_1, self.convex_upsampling)
                flow1_2 = dflow1_2 + upsample_decoder_flow(flow1_3, ft_1, self.convex_upsampling)

            out1 = self.decoder1(ft_1, f0_1, f1_1, flow0_2, flow1_2)
            dflow0_1 = out1[:, 0:2]
            dflow1_1 = out1[:, 2:4]
            up_mask_1 = torch.sigmoid(out1[:, 4:5])
            up_res_1 = out1[:, 5:8]

            if self.init_flow_layer >= 4:
                flow0_1 = init0_1 + dflow0_1
                flow1_1 = init1_1 + dflow1_1
            else:
                flow0_1 = dflow0_1 + upsample_decoder_flow(flow0_2, out1, self.convex_upsampling)
                flow1_1 = dflow1_1 + upsample_decoder_flow(flow1_2, out1, self.convex_upsampling)

            img0_warp = warp(img0, flow0_1)
            img1_warp = warp(img1, flow1_1)
            imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean
            imgt_pred = torch.clamp(imgt_merge + up_res_1, 0, 1)

            result: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] = (
                crop_like(imgt_pred, height, width),
                crop_like(flow0_1, height, width),
                crop_like(flow1_1, height, width),
                crop_like(up_mask_1, height, width),
                crop_like(up_res_1, height, width),
                crop_like(imgt_merge, height, width),
            )
        finally:
            self._inference_init_flow0_mask = None
            self._inference_init_flow1_mask = None
            self._inference_init_flow_mask_epsilon = 1e-6
        return result

    def forward(
        self,
        img0: Tensor,
        img1: Tensor,
        embt: Tensor,
        imgt: Tensor,
        init_flow0: Tensor | None = None,
        init_flow1: Tensor | None = None,
        init_flow0_mask: Tensor | None = None,
        init_flow1_mask: Tensor | None = None,
        init_flow_mask_epsilon: float = 1e-6,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        padded_init_flow0_mask, padded_init_flow1_mask = self._prepare_runtime_init_flow_masks(
            init_flow0_mask=init_flow0_mask,
            init_flow1_mask=init_flow1_mask,
            init_flow_mask_epsilon=init_flow_mask_epsilon,
        )

        self._inference_init_flow0_mask = padded_init_flow0_mask
        self._inference_init_flow1_mask = padded_init_flow1_mask
        self._inference_init_flow_mask_epsilon = init_flow_mask_epsilon
        try:
            height, width = img0.shape[-2:]
            mean = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
            img0 = img0 - mean
            img1 = img1 - mean
            imgt_centered = imgt - mean

            img0, _ = pad_to_multiple(img0, multiple=8, mode="replicate")
            img1, _ = pad_to_multiple(img1, multiple=8, mode="replicate")
            imgt_centered, _ = pad_to_multiple(imgt_centered, multiple=8, mode="replicate")
            if init_flow0 is None or init_flow1 is None:
                raise ValueError("init_flow0 and init_flow1 must both be provided.")
            init_flow0, _ = pad_to_multiple(init_flow0, multiple=8, mode="replicate")
            init_flow1, _ = pad_to_multiple(init_flow1, multiple=8, mode="replicate")

            f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
            f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)
            ft_1, _ft_2, _ft_3, _ft_4 = self.encoder(imgt_centered)

            init_flows = self._get_init_flows(
                init_flow0=init_flow0,
                init_flow1=init_flow1,
                scale_factor=1.0,
            )
            init0_1, init1_1 = init_flows[1]
            init0_2, init1_2 = init_flows[2]
            init0_3, init1_3 = init_flows[3]
            init0_4, init1_4 = init_flows[4]

            out4 = self.decoder4(f0_4, f1_4, embt)
            ft_3_pred = out4[:, :]
            flow0_4 = init0_4
            flow1_4 = init1_4

            out3 = self.decoder3(ft_3_pred, f0_3, f1_3, flow0_4, flow1_4)
            dflow0_3 = out3[:, 0:2]
            dflow1_3 = out3[:, 2:4]
            ft_2_pred = out3[:, 4:]

            if self.init_flow_layer >= 2:
                flow0_3 = init0_3 + dflow0_3
                flow1_3 = init1_3 + dflow1_3
            else:
                flow0_3 = dflow0_3 + upsample_decoder_flow(flow0_4, ft_2_pred, self.convex_upsampling)
                flow1_3 = dflow1_3 + upsample_decoder_flow(flow1_4, ft_2_pred, self.convex_upsampling)

            out2 = self.decoder2(ft_2_pred, f0_2, f1_2, flow0_3, flow1_3)
            dflow0_2 = out2[:, 0:2]
            dflow1_2 = out2[:, 2:4]
            ft_1_pred = out2[:, 4:]

            if self.init_flow_layer >= 3:
                flow0_2 = init0_2 + dflow0_2
                flow1_2 = init1_2 + dflow1_2
            else:
                flow0_2 = dflow0_2 + upsample_decoder_flow(flow0_3, ft_1_pred, self.convex_upsampling)
                flow1_2 = dflow1_2 + upsample_decoder_flow(flow1_3, ft_1_pred, self.convex_upsampling)

            out1 = self.decoder1(ft_1_pred, f0_1, f1_1, flow0_2, flow1_2)
            dflow0_1 = out1[:, 0:2]
            dflow1_1 = out1[:, 2:4]
            up_mask_1 = torch.sigmoid(out1[:, 4:5])
            up_res_1 = out1[:, 5:8]

            if self.init_flow_layer >= 4:
                flow0_1 = init0_1 + dflow0_1
                flow1_1 = init1_1 + dflow1_1
            else:
                flow0_1 = dflow0_1 + upsample_decoder_flow(flow0_2, out1, self.convex_upsampling)
                flow1_1 = dflow1_1 + upsample_decoder_flow(flow1_2, out1, self.convex_upsampling)

            img0_warp = warp(img0, flow0_1)
            img1_warp = warp(img1, flow1_1)
            imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean
            imgt_pred = torch.clamp(imgt_merge + up_res_1, 0, 1)

            imgt_pred = crop_like(imgt_pred, height, width)
            img0_warp = crop_like(img0_warp, height, width)
            img1_warp = crop_like(img1_warp, height, width)
            up_mask_1 = crop_like(up_mask_1, height, width)
            flow0_1 = crop_like(flow0_1, height, width)
            flow1_1 = crop_like(flow1_1, height, width)

            loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)
            loss_geo = 0.01 * self.gc_loss(ft_1_pred, ft_1)
            loss_dis = self.l1_loss(img0_warp + mean - imgt)
            loss_dis = loss_dis + self.l1_loss(img1_warp + mean - imgt)

            result: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] = (
                imgt_pred,
                loss_rec,
                loss_geo,
                loss_dis,
                flow0_1,
                flow1_1,
                up_mask_1,
            )
        finally:
            self._inference_init_flow0_mask = None
            self._inference_init_flow1_mask = None
            self._inference_init_flow_mask_epsilon = 1e-6
        return result


__all__ = ["Model"]
