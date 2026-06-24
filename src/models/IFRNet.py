from __future__ import annotations

import torch
from torch import Tensor

from .convex_upsampling import crop_like
from .convex_upsampling import resize_bilinear
from .convex_upsampling import upsample_decoder_flow
from .external.IFRNet.models.IFRNet import Model as ExternalIFRNetModel
from .external.IFRNet.models.IFRNet import pad_to_multiple
from .external.IFRNet.utils import get_robust_weight
from .external.IFRNet.utils import warp


class Model(ExternalIFRNetModel):
    def __init__(
        self,
        local_rank: int = -1,
        lr: float = 1e-4,
        convex_upsampling: bool = False,
    ) -> None:
        super().__init__(local_rank=local_rank, lr=lr)
        self.convex_upsampling = bool(convex_upsampling)

    def set_convex_upsampling(self, enabled: bool) -> None:
        self.convex_upsampling = bool(enabled)

    def inference(
        self,
        img0: Tensor,
        img1: Tensor,
        embt: Tensor,
        scale_factor: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        height, width = img0.shape[-2:]
        mean = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean
        img1 = img1 - mean

        img0, _ = pad_to_multiple(img0, 8, mode="replicate")
        img1, _ = pad_to_multiple(img1, 8, mode="replicate")

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3 = out4[:, 4:]

        out3 = self.decoder3(ft_3, f0_3, f1_3, up_flow0_4, up_flow1_4)
        ft_2 = out3[:, 4:]
        up_flow0_3 = out3[:, 0:2] + upsample_decoder_flow(up_flow0_4, ft_2, self.convex_upsampling)
        up_flow1_3 = out3[:, 2:4] + upsample_decoder_flow(up_flow1_4, ft_2, self.convex_upsampling)

        out2 = self.decoder2(ft_2, f0_2, f1_2, up_flow0_3, up_flow1_3)
        ft_1 = out2[:, 4:]
        up_flow0_2 = out2[:, 0:2] + upsample_decoder_flow(up_flow0_3, ft_1, self.convex_upsampling)
        up_flow1_2 = out2[:, 2:4] + upsample_decoder_flow(up_flow1_3, ft_1, self.convex_upsampling)

        out1 = self.decoder1(ft_1, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + upsample_decoder_flow(up_flow0_2, out1, self.convex_upsampling)
        up_flow1_1 = out1[:, 2:4] + upsample_decoder_flow(up_flow1_2, out1, self.convex_upsampling)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]

        up_flow0_1 = resize_bilinear(up_flow0_1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
        up_flow1_1 = resize_bilinear(up_flow1_1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
        up_mask_1 = resize_bilinear(up_mask_1, scale_factor=(1.0 / scale_factor))
        up_res_1 = resize_bilinear(up_res_1, scale_factor=(1.0 / scale_factor))

        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean
        imgt_pred = torch.clamp(imgt_merge + up_res_1, 0, 1)

        return (
            crop_like(imgt_pred, height, width),
            crop_like(up_flow0_1, height, width),
            crop_like(up_flow1_1, height, width),
            crop_like(up_mask_1, height, width),
        )

    def forward(
        self,
        img0: Tensor,
        img1: Tensor,
        embt: Tensor,
        imgt: Tensor,
        flow: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        height, width = img0.shape[-2:]
        mean = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean
        img1 = img1 - mean
        imgt_centered = imgt - mean

        img0, _ = pad_to_multiple(img0, 8, mode="replicate")
        img1, _ = pad_to_multiple(img1, 8, mode="replicate")
        imgt_centered, _ = pad_to_multiple(imgt_centered, 8, mode="replicate")
        if flow is not None:
            flow, _ = pad_to_multiple(flow, 8, mode="replicate")

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)
        ft_1, ft_2, ft_3, _ft_4 = self.encoder(imgt_centered)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_pred = out4[:, 4:]

        out3 = self.decoder3(ft_3_pred, f0_3, f1_3, up_flow0_4, up_flow1_4)
        ft_2_pred = out3[:, 4:]
        up_flow0_3 = out3[:, 0:2] + upsample_decoder_flow(up_flow0_4, ft_2_pred, self.convex_upsampling)
        up_flow1_3 = out3[:, 2:4] + upsample_decoder_flow(up_flow1_4, ft_2_pred, self.convex_upsampling)

        out2 = self.decoder2(ft_2_pred, f0_2, f1_2, up_flow0_3, up_flow1_3)
        ft_1_pred = out2[:, 4:]
        up_flow0_2 = out2[:, 0:2] + upsample_decoder_flow(up_flow0_3, ft_1_pred, self.convex_upsampling)
        up_flow1_2 = out2[:, 2:4] + upsample_decoder_flow(up_flow1_3, ft_1_pred, self.convex_upsampling)

        out1 = self.decoder1(ft_1_pred, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + upsample_decoder_flow(up_flow0_2, out1, self.convex_upsampling)
        up_flow1_1 = out1[:, 2:4] + upsample_decoder_flow(up_flow1_2, out1, self.convex_upsampling)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]

        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean
        imgt_pred = torch.clamp(imgt_merge + up_res_1, 0, 1)

        imgt_pred = crop_like(imgt_pred, height, width)
        up_flow0_1 = crop_like(up_flow0_1, height, width)
        up_flow1_1 = crop_like(up_flow1_1, height, width)
        up_mask_1 = crop_like(up_mask_1, height, width)

        loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)
        loss_geo = 0.01 * (self.gc_loss(ft_1_pred, ft_1) + self.gc_loss(ft_2_pred, ft_2) + self.gc_loss(ft_3_pred, ft_3))

        if flow is not None:
            flow = crop_like(flow, height, width)
            robust_weight0 = get_robust_weight(up_flow0_1, flow[:, 0:2], beta=0.3)
            robust_weight1 = get_robust_weight(up_flow1_1, flow[:, 2:4], beta=0.3)
            up_flow0_2 = crop_like(2.0 * resize_bilinear(up_flow0_2, 2.0), height, width)
            up_flow1_2 = crop_like(2.0 * resize_bilinear(up_flow1_2, 2.0), height, width)
            up_flow0_3 = crop_like(4.0 * resize_bilinear(up_flow0_3, 4.0), height, width)
            up_flow1_3 = crop_like(4.0 * resize_bilinear(up_flow1_3, 4.0), height, width)
            up_flow0_4 = crop_like(8.0 * resize_bilinear(up_flow0_4, 8.0), height, width)
            up_flow1_4 = crop_like(8.0 * resize_bilinear(up_flow1_4, 8.0), height, width)
            loss_dis = 0.01 * (
                self.rb_loss(up_flow0_2 - flow[:, 0:2], weight=robust_weight0)
                + self.rb_loss(up_flow1_2 - flow[:, 2:4], weight=robust_weight1)
            )
            loss_dis = loss_dis + 0.01 * (
                self.rb_loss(up_flow0_3 - flow[:, 0:2], weight=robust_weight0)
                + self.rb_loss(up_flow1_3 - flow[:, 2:4], weight=robust_weight1)
            )
            loss_dis = loss_dis + 0.01 * (
                self.rb_loss(up_flow0_4 - flow[:, 0:2], weight=robust_weight0)
                + self.rb_loss(up_flow1_4 - flow[:, 2:4], weight=robust_weight1)
            )
        else:
            loss_dis = 0.00 * loss_geo

        return imgt_pred, loss_rec, loss_geo, loss_dis, up_flow0_1, up_flow1_1, up_mask_1


__all__ = ["Model"]
