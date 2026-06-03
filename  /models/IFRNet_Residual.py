import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import warp, get_robust_weight
from ..loss import *


# -------------------------
# helpers
# -------------------------
def pad_to_multiple(x, multiple=16, mode="replicate"):
    # x: [B,C,H,W]
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)  # (pad_h, pad_w)
    # pad format: (left, right, top, bottom)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (pad_h, pad_w)

def crop_like(x, h, w):
    return x[..., :h, :w]

def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.PReLU(out_channels)
    )

def flow_resize(flow, scale_factor):
    """
    Resize flow and scale displacement accordingly.
    If spatial size scales by s, displacement (pixel units) scales by s.
    """
    return resize(flow, scale_factor=scale_factor) * scale_factor

def build_pyramid_init_flows(init_flow0_full, init_flow1_full, scale_factor=1.0):
    """
    Build init flow pyramid aligned to DECODER scales w.r.t scaled img0_:
      lv1: full
      lv2: 1/2
      lv3: 1/4
      lv4: 1/8
    If scale_factor != 1, it means img0_ is scaled from original img0.
    The init flow should be scaled consistently too.
    """
    level_scales = {
        1: scale_factor * 1.0,
        2: scale_factor * 0.5,
        3: scale_factor * 0.25,
        4: scale_factor * 0.125,
    }
    init = {}
    for lv, s in level_scales.items():
        init0 = flow_resize(init_flow0_full, s)
        init1 = flow_resize(init_flow1_full, s)
        init[lv] = (init0, init1)
    return init


# -------------------------
# regularization on dflow
# -------------------------
def sobel_grad(x: torch.Tensor):
    """
    x: [B,C,H,W]
    return gx, gy: [B,C,H,W]
    """
    device, dtype = x.device, x.dtype
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=device, dtype=dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],
                       [ 0, 0, 0],
                       [ 1, 2, 1]], device=device, dtype=dtype).view(1,1,3,3)

    C = x.shape[1]
    kx = kx.repeat(C, 1, 1, 1)
    ky = ky.repeat(C, 1, 1, 1)

    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return gx, gy

def dflow_reg_loss(dflow: torch.Tensor, lam_mag=1.0, lam_grad=0.0, eps=1e-3):
    """
    dflow: [B,2,H,W]
    Charbonnier-like magnitude + sobel spatial grad
    """
    mag = torch.sqrt(dflow * dflow + eps * eps).mean()
    gx, gy = sobel_grad(dflow)
    grad = torch.sqrt(gx * gx + gy * gy + eps * eps).mean()
    return lam_mag * mag + lam_grad * grad


# -------------------------
# network blocks
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super().__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, 3, 1, 1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, 3, 1, 1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :].clone())
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :].clone())
        out = self.prelu(x + self.conv5(out))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, 32, 3, 2, 1),
            convrelu(32, 32, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(32, 48, 3, 2, 1),
            convrelu(48, 48, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(48, 72, 3, 2, 1),
            convrelu(72, 72, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(72, 96, 3, 2, 1),
            convrelu(96, 96, 3, 1, 1)
        )

    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


# -------------------------
# decoders (bias-only residual)
# out channels must match slicing:
#   level4: ft_3_(72) => 72
#   level3: dflow0(2) + dflow1(2) + ft_2_(48) => 52
#   level2: dflow0(2) + dflow1(2) + ft_1_(32) => 36
#   level1: dflow0(2) + dflow1(2) + mask(1) + res(3) => 8
# -------------------------
class Decoder4(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock = nn.Sequential(
            convrelu(192 + 1, 192),
            ResBlock(192, 32),
            nn.ConvTranspose2d(192, 72, 4, 2, 1, bias=True)
        )

    def forward(self, f0, f1, embt):
        b, c, h, w = f0.shape
        embt = embt.repeat(1, 1, h, w)
        f_in = torch.cat([f0, f1, embt], 1)
        return self.convblock(f_in)


class Decoder3(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock = nn.Sequential(
            convrelu(220, 216),
            ResBlock(216, 32),
            nn.ConvTranspose2d(216, 52, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        return self.convblock(f_in)


class Decoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock = nn.Sequential(
            convrelu(148, 144),
            ResBlock(144, 32),
            nn.ConvTranspose2d(144, 36, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        return self.convblock(f_in)


class Decoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock = nn.Sequential(
            convrelu(100, 96),
            ResBlock(96, 32),
            nn.ConvTranspose2d(96, 8, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        return self.convblock(f_in)


# -------------------------
# Model
# -------------------------
class Model(nn.Module):
    def __init__(self, local_rank=-1, lr=1e-4, init_flow_layer=4):
        super().__init__()
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()

        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)

        self.init_flow_layer = init_flow_layer

    @torch.no_grad()
    def _make_zero_flow(self, B, H, W, device, dtype):
        return torch.zeros(B, 2, H, W, device=device, dtype=dtype)

    def _get_init_flows(self, init_flow0=None, init_flow1=None, scale_factor=1.0):
        """
        img0_: scaled image tensor (the same spatial size used for encoder/decoder at lv1)
        Priority:
        1) init_flows dict (lv1..lv4)
        2) full-res init_flow*_full -> build pyramid (with scale_factor)
        3) zeros
        """
        if (init_flow0 is not None) and (init_flow1 is not None):
            return build_pyramid_init_flows(init_flow0, init_flow1, scale_factor=scale_factor)
        
        raise ValueError("Either init_flows or init_flow*_full must be provided.")


    # -------------------------
    # inference
    # -------------------------
    def inference(self, img0, img1, embt, scale_factor=1.0,
                  init_flow0=None, init_flow1=None):
        H, W = img0.shape[-2:]  # original

        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_

        # img0_ = resize(img0, scale_factor=scale_factor)
        # img1_ = resize(img1, scale_factor=scale_factor)

        img0, _ = pad_to_multiple(img0, multiple=8, mode="replicate")
        img1, _ = pad_to_multiple(img1, multiple=8, mode="replicate")
        init_flow0, _ = pad_to_multiple(init_flow0, multiple=8, mode="replicate")
        init_flow1, _ = pad_to_multiple(init_flow1, multiple=8, mode="replicate")

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)

        init = self._get_init_flows(
            init_flow0=init_flow0,
            init_flow1=init_flow1,
            scale_factor=scale_factor
        )
        init0_1, init1_1 = init[1]
        init0_2, init1_2 = init[2]
        init0_3, init1_3 = init[3]
        init0_4, init1_4 = init[4]

        # level4
        out4     = self.decoder4(f0_4, f1_4, embt)
        ft_3_    = out4[:, :]

        flow0_4 = init0_4
        flow1_4 = init1_4

        # level3
        out3     = self.decoder3(ft_3_, f0_3, f1_3, flow0_4, flow1_4)
        dflow0_3 = out3[:, 0:2]
        dflow1_3 = out3[:, 2:4]
        ft_2_    = out3[:, 4:]

        if self.init_flow_layer >= 2:
            flow0_3 = init0_3 + dflow0_3
            flow1_3 = init1_3 + dflow1_3
        else:
            flow0_3 = dflow0_3 + 2.0 * resize(flow0_4, scale_factor=2.0)
            flow1_3 = dflow1_3 + 2.0 * resize(flow1_4, scale_factor=2.0)

        # level2
        out2     = self.decoder2(ft_2_, f0_2, f1_2, flow0_3, flow1_3)
        dflow0_2 = out2[:, 0:2]
        dflow1_2 = out2[:, 2:4]
        ft_1_    = out2[:, 4:]

        if self.init_flow_layer >= 3:
            flow0_2 = init0_2 + dflow0_2
            flow1_2 = init1_2 + dflow1_2
        else:
            flow0_2 = dflow0_2 + 2.0 * resize(flow0_3, scale_factor=2.0)
            flow1_2 = dflow1_2 + 2.0 * resize(flow1_3, scale_factor=2.0)

        # level1
        out1      = self.decoder1(ft_1_, f0_1, f1_1, flow0_2, flow1_2)
        dflow0_1  = out1[:, 0:2]
        dflow1_1  = out1[:, 2:4]
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1  = out1[:, 5:8]

        if self.init_flow_layer >= 4:
            flow0_1 = init0_1 + dflow0_1
            flow1_1 = init1_1 + dflow1_1
        else:
            flow0_1 = dflow0_1 + 2.0 * resize(flow0_2, scale_factor=2.0)
            flow1_1 = dflow1_1 + 2.0 * resize(flow1_2, scale_factor=2.0)

        img0_warp = warp(img0, flow0_1)
        img1_warp = warp(img1, flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = torch.clamp(imgt_merge + up_res_1, 0, 1)

        # crop
        imgt_pred = crop_like(imgt_pred, H, W)
        img0_warp = crop_like(img0_warp, H, W)
        img1_warp = crop_like(img1_warp, H, W)
        imgt_merge = crop_like(imgt_merge, H, W)
        up_mask_1 = crop_like(up_mask_1, H, W)
        up_res_1  = crop_like(up_res_1, H, W)
        flow0_1 = crop_like(flow0_1, H, W)
        flow1_1 = crop_like(flow1_1, H, W)

        return imgt_pred, flow0_1, flow1_1, up_mask_1, up_res_1, imgt_merge

    # -------------------------
    # training forward
    # -------------------------
    def forward(self, img0, img1, embt, imgt, init_flow0=None, init_flow1=None):
        """
        Training forward.
        Optional supervised GT flow: `flow` shape [B,4,H,W] (flow0 GT + flow1 GT).
        """
        H, W = img0.shape[-2:]  # original
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0  = img0 - mean_
        img1  = img1 - mean_
        imgt_ = imgt - mean_

        img0, _ = pad_to_multiple(img0, multiple=8, mode="replicate")
        img1, _ = pad_to_multiple(img1, multiple=8, mode="replicate")
        imgt_, _ = pad_to_multiple(imgt_, multiple=8, mode="replicate")
        init_flow0, _ = pad_to_multiple(init_flow0, multiple=8, mode="replicate")
        init_flow1, _ = pad_to_multiple(init_flow1, multiple=8, mode="replicate")

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)
        ft_1, ft_2, ft_3, ft_4 = self.encoder(imgt_)

        init = self._get_init_flows(
            init_flow0=init_flow0,
            init_flow1=init_flow1,
            scale_factor=1.0
        )
        init0_1, init1_1 = init[1]
        init0_2, init1_2 = init[2]
        init0_3, init1_3 = init[3]
        init0_4, init1_4 = init[4]

        # level4
        out4     = self.decoder4(f0_4, f1_4, embt)
        ft_3_    = out4[:, :]

        flow0_4 = init0_4
        flow1_4 = init1_4

        # level3
        out3     = self.decoder3(ft_3_, f0_3, f1_3, flow0_4, flow1_4)
        dflow0_3 = out3[:, 0:2]
        dflow1_3 = out3[:, 2:4]
        ft_2_    = out3[:, 4:]

        if self.init_flow_layer >= 2:
            flow0_3 = init0_3 + dflow0_3
            flow1_3 = init1_3 + dflow1_3
        else:
            flow0_3 = dflow0_3 + 2.0 * resize(flow0_4, scale_factor=2.0)
            flow1_3 = dflow1_3 + 2.0 * resize(flow1_4, scale_factor=2.0)

        # level2
        out2     = self.decoder2(ft_2_, f0_2, f1_2, flow0_3, flow1_3)
        dflow0_2 = out2[:, 0:2]
        dflow1_2 = out2[:, 2:4]
        ft_1_    = out2[:, 4:]

        if self.init_flow_layer >= 3:
            flow0_2 = init0_2 + dflow0_2
            flow1_2 = init1_2 + dflow1_2
        else:
            flow0_2 = dflow0_2 + 2.0 * resize(flow0_3, scale_factor=2.0)
            flow1_2 = dflow1_2 + 2.0 * resize(flow1_3, scale_factor=2.0)

        # level1
        out1      = self.decoder1(ft_1_, f0_1, f1_1, flow0_2, flow1_2)
        dflow0_1  = out1[:, 0:2]
        dflow1_1  = out1[:, 2:4]
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1  = out1[:, 5:8]

        if self.init_flow_layer >= 4:
            flow0_1 = init0_1 + dflow0_1
            flow1_1 = init1_1 + dflow1_1
        else:
            flow0_1 = dflow0_1 + 2.0 * resize(flow0_2, scale_factor=2.0)
            flow1_1 = dflow1_1 + 2.0 * resize(flow1_2, scale_factor=2.0)

        img0_warp = warp(img0, flow0_1)
        img1_warp = warp(img1, flow1_1)

        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        # imgt_merge = img0_warp + mean_
        imgt_pred = torch.clamp(imgt_merge + up_res_1, 0, 1)

        # crop
        imgt_pred = crop_like(imgt_pred, H, W)
        img0_warp = crop_like(img0_warp, H, W)
        img1_warp = crop_like(img1_warp, H, W)
        up_mask_1 = crop_like(up_mask_1, H, W)
        up_res_1  = crop_like(up_res_1, H, W)
        flow0_1 = crop_like(flow0_1, H, W)
        flow1_1 = crop_like(flow1_1, H, W)

        # losses
        loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)
        # loss_geo = 0.01 * (self.gc_loss(ft_1_, ft_1) + self.gc_loss(ft_2_, ft_2) + self.gc_loss(ft_3_, ft_3))
        loss_geo = 0.01 * (self.gc_loss(ft_1_, ft_1))

        # dflow regularization (optional but recommended for stability)
        loss_dflow = 0.0
        # loss_dflow += dflow_reg_loss(dflow0_3) + dflow_reg_loss(dflow1_3)
        # loss_dflow += dflow_reg_loss(dflow0_2) + dflow_reg_loss(dflow1_2)
        # loss_dflow += dflow_reg_loss(dflow0_1) + dflow_reg_loss(dflow1_1)
        # loss_dflow = 0.0 * loss_dflow  # set to zero by default, tune weight in dflow_reg_loss()
        loss_dflow = self.l1_loss(img0_warp + mean_ - imgt)  # alternative dflow reg
        loss_dflow += self.l1_loss(img1_warp + mean_ - imgt)  # alternative dflow reg

        if init_flow0 is not None and init_flow1 is not None:
            # robust weights use finest-level total flow
            # robust_weight0 = get_robust_weight(flow0_1, flow[:, 0:2], beta=0.3)
            # robust_weight1 = get_robust_weight(flow1_1, flow[:, 2:4], beta=0.3)

            # multi-scale distillation comparing total flow against GT
            loss_dis = 0.0
            # loss_dis  += 0.01 * (
            #     self.rb_loss(2.0 * resize(flow0_2, 2.0) - flow[:, 0:2], weight=robust_weight0) +
            #     self.rb_loss(2.0 * resize(flow1_2, 2.0) - flow[:, 2:4], weight=robust_weight1)
            # )
            # loss_dis += 0.01 * (
            #     self.rb_loss(4.0 * resize(flow0_3, 4.0) - flow[:, 0:2], weight=robust_weight0) +
            #     self.rb_loss(4.0 * resize(flow1_3, 4.0) - flow[:, 2:4], weight=robust_weight1)
            # )
            # loss_dis += 0.01 * (
            #     self.rb_loss(8.0 * resize(flow0_4, 8.0) - flow[:, 0:2], weight=robust_weight0) +
            #     self.rb_loss(8.0 * resize(flow1_4, 8.0) - flow[:, 2:4], weight=robust_weight1)
            # )

            # add residual regularization with a small weight (tunable)
            loss_dis += loss_dflow
        else:
            # no GT flow supervision, keep dis as 0 (or you can still apply loss_dflow here)
            loss_dis = 0.00 * loss_geo

        return imgt_pred, loss_rec, loss_geo, loss_dis, flow0_1, flow1_1, up_mask_1
