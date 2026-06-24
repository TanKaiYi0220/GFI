from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def crop_like(tensor: Tensor, height: int, width: int) -> Tensor:
    return tensor[..., :height, :width]


def resize_bilinear(tensor: Tensor, scale_factor: float) -> Tensor:
    return F.interpolate(tensor, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def validate_convex_upsampling_mask(flow: Tensor, mask: Tensor, scale_factor: int) -> None:
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must have shape [B, 2, H, W], got {tuple(flow.shape)}")
    if mask.ndim != 4 or mask.shape[1] != 9:
        raise ValueError(f"mask must have shape [B, 9, H*scale, W*scale], got {tuple(mask.shape)}")

    expected_shape = (
        int(flow.shape[0]),
        9,
        int(flow.shape[2] * scale_factor),
        int(flow.shape[3] * scale_factor),
    )
    if tuple(mask.shape) != expected_shape:
        raise ValueError(f"convex upsampling mask must have shape {expected_shape}, got {tuple(mask.shape)}")


def build_convex_upsampling_mask_logits(features: Tensor) -> Tensor:
    if features.ndim != 4:
        raise ValueError(f"features must have shape [B, C, H, W], got {tuple(features.shape)}")
    if features.shape[1] >= 9:
        return features[:, :9]

    missing_channels = 9 - int(features.shape[1])
    padding = torch.zeros(
        features.shape[0],
        missing_channels,
        features.shape[2],
        features.shape[3],
        device=features.device,
        dtype=features.dtype,
    )
    return torch.cat([features, padding], dim=1)


def convex_upsample_flow(flow: Tensor, mask: Tensor, scale_factor: int) -> Tensor:
    validate_convex_upsampling_mask(flow=flow, mask=mask, scale_factor=scale_factor)

    batch_size = int(flow.shape[0])
    channel_count = int(flow.shape[1])
    height = int(flow.shape[2])
    width = int(flow.shape[3])
    target_height = int(height * scale_factor)
    target_width = int(width * scale_factor)
    weights = torch.softmax(mask, dim=1)

    flow_neighbors = F.unfold(flow, kernel_size=3, padding=1)
    flow_neighbors = flow_neighbors.view(batch_size, channel_count * 9, height, width)
    flow_neighbors = F.interpolate(flow_neighbors, size=(target_height, target_width), mode="nearest")
    flow_neighbors = flow_neighbors.view(batch_size, channel_count, 9, target_height, target_width)
    return torch.sum(flow_neighbors * weights.unsqueeze(1), dim=2) * float(scale_factor)


def upsample_decoder_flow(flow: Tensor, mask_features: Tensor, convex_upsampling: bool) -> Tensor:
    if convex_upsampling:
        mask_logits = build_convex_upsampling_mask_logits(features=mask_features)
        return convex_upsample_flow(flow=flow, mask=mask_logits, scale_factor=2)

    return resize_bilinear(tensor=flow, scale_factor=2.0) * 2.0
