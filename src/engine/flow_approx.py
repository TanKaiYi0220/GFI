from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor

FLOW_APPROX_METHODS: tuple[str, ...] = ("single", "combination", "splatting")
SPLATTING_FLOW_APPROX_METHODS: tuple[str, ...] = ("splatting", "linear_splatting")
SPLATTING_FILL_STRATEGIES: tuple[str, ...] = (
    "zero",
    "outside_in_4direction",
    "outside_in_4neighbor",
    "outside_in_8neighbor",
)
SPLATTING_FILL_METHOD_ALIASES: dict[str, str] = {
    "splatting_zero_fill": "zero",
    "splatting_outside_in_4direction": "outside_in_4direction",
    "splatting_outside_in_4neighbor": "outside_in_4neighbor",
    "splatting_outside_in_8neighbor": "outside_in_8neighbor",
}
FLOW_APPROX_METHOD_CHOICES: tuple[str, ...] = FLOW_APPROX_METHODS + ("linear_splatting",) + tuple(SPLATTING_FILL_METHOD_ALIASES.keys())
DEPTH_REDUCE_MODE: str = "amax"
DEFAULT_SPLATTING_FILL_STRATEGY: str = "outside_in_4neighbor"


@dataclass(frozen=True)
class FlowInitResult:
    bmv: Tensor
    fmv: Tensor
    masks: Tensor | None


def flow_approx(flow: Tensor, time: Tensor, forward: bool) -> Tensor:
    return time * flow if forward else (1 - time) * flow


def flow_approx_combination(fmv: Tensor, bmv: Tensor, time: Tensor, forward: bool) -> Tensor:
    if forward:
        return (1 - time) * (1 - time) * fmv - time * (1 - time) * bmv

    return -(1 - time) * time * fmv + time * time * bmv


def is_splatting_flow_approx_method(flow_approx_method: str) -> bool:
    return flow_approx_method in SPLATTING_FLOW_APPROX_METHODS or flow_approx_method in SPLATTING_FILL_METHOD_ALIASES


def resolve_splatting_fill_strategy(flow_approx_method: str, splatting_fill_strategy: str) -> str:
    if flow_approx_method in SPLATTING_FILL_METHOD_ALIASES:
        return SPLATTING_FILL_METHOD_ALIASES[flow_approx_method]

    if splatting_fill_strategy not in SPLATTING_FILL_STRATEGIES:
        available_strategies = ", ".join(SPLATTING_FILL_STRATEGIES)
        raise ValueError(f"Unsupported splatting_fill_strategy '{splatting_fill_strategy}'. Available strategies: {available_strategies}")

    return splatting_fill_strategy


def validate_flow_tensor(name: str, tensor: Tensor) -> None:
    if tensor.ndim != 4 or tensor.shape[1] != 2:
        raise ValueError(f"{name} must have shape [B, 2, H, W], got {tuple(tensor.shape)}.")


def normalize_depth_tensor(name: str, depth: Tensor, flow: Tensor) -> Tensor:
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)

    expected_shape = (flow.shape[0], 1, flow.shape[2], flow.shape[3])
    if tuple(depth.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(depth.shape)}.")

    return depth.to(device=flow.device, dtype=flow.dtype)


def make_source_grid(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    import torch

    y_coords = torch.arange(height, device=device, dtype=dtype)
    x_coords = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    source_grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)
    return source_grid.expand(batch_size, 2, height, width)


def flatten_target_index(target_x: Tensor, target_y: Tensor, width: int) -> Tensor:
    return target_y * width + target_x


def validate_coverage_mask(name: str, mask: Tensor, flow: Tensor) -> None:
    expected_shape = (flow.shape[0], 1, flow.shape[2], flow.shape[3])
    if tuple(mask.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(mask.shape)}.")


def build_neighbor_kernel(neighbor_mode: str, device: torch.device, dtype: torch.dtype) -> Tensor:
    import torch

    if neighbor_mode == "4neighbor":
        return torch.tensor(
            [[[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]],
            device=device,
            dtype=dtype,
        )

    if neighbor_mode == "8neighbor":
        return torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]],
            device=device,
            dtype=dtype,
        )

    raise ValueError(f"Unsupported neighbor_mode: {neighbor_mode}.")


def outside_in_fill_flow_holes_by_neighbors(flow: Tensor, mask: Tensor, neighbor_mode: str) -> Tensor:
    import torch
    import torch.nn.functional as functional

    validate_flow_tensor("flow", flow)
    validate_coverage_mask("mask", mask, flow)

    valid = mask.bool()
    filled = torch.where(valid, flow, torch.zeros_like(flow))
    max_iterations = int(flow.shape[2] + flow.shape[3])
    neighbor_kernel = build_neighbor_kernel(neighbor_mode=neighbor_mode, device=flow.device, dtype=flow.dtype)
    flow_kernel = neighbor_kernel.expand(flow.shape[1], 1, 3, 3)

    for _iteration in range(max_iterations):
        if bool(valid.all().detach().cpu().item()):
            break

        valid_float = valid.to(dtype=flow.dtype)
        neighbor_count = functional.conv2d(valid_float, neighbor_kernel, padding=1)
        fillable = (~valid) & (neighbor_count > 0)
        if not bool(fillable.any().detach().cpu().item()):
            break

        neighbor_sum = functional.conv2d(filled * valid_float, flow_kernel, padding=1, groups=flow.shape[1])
        neighbor_average = neighbor_sum / neighbor_count.clamp_min(1.0)
        filled = torch.where(fillable.expand_as(flow), neighbor_average, filled)
        valid = valid | fillable

    return filled


def outside_in_fill_flow_holes(flow: Tensor, mask: Tensor) -> Tensor:
    return outside_in_fill_flow_holes_by_neighbors(flow=flow, mask=mask, neighbor_mode="4neighbor")


def gather_nearest_valid_along_width(flow: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    import torch

    batch_size = int(flow.shape[0])
    height = int(flow.shape[2])
    width = int(flow.shape[3])
    x_coords = torch.arange(width, device=flow.device, dtype=torch.long).reshape(1, 1, 1, width)
    x_coords = x_coords.expand(batch_size, 1, height, width)
    valid_indices = torch.where(valid, x_coords, torch.full_like(x_coords, -1))
    nearest_indices = torch.cummax(valid_indices, dim=3).values
    has_value = nearest_indices >= 0
    gathered = flow.gather(dim=3, index=nearest_indices.clamp_min(0).expand(batch_size, flow.shape[1], height, width))
    return torch.where(has_value.expand_as(flow), gathered, torch.zeros_like(flow)), has_value


def gather_nearest_valid_along_height(flow: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    import torch

    batch_size = int(flow.shape[0])
    height = int(flow.shape[2])
    width = int(flow.shape[3])
    y_coords = torch.arange(height, device=flow.device, dtype=torch.long).reshape(1, 1, height, 1)
    y_coords = y_coords.expand(batch_size, 1, height, width)
    valid_indices = torch.where(valid, y_coords, torch.full_like(y_coords, -1))
    nearest_indices = torch.cummax(valid_indices, dim=2).values
    has_value = nearest_indices >= 0
    gathered = flow.gather(dim=2, index=nearest_indices.clamp_min(0).expand(batch_size, flow.shape[1], height, width))
    return torch.where(has_value.expand_as(flow), gathered, torch.zeros_like(flow)), has_value


def outside_in_fill_flow_holes_by_4direction(flow: Tensor, mask: Tensor) -> Tensor:
    import torch

    validate_flow_tensor("flow", flow)
    validate_coverage_mask("mask", mask, flow)

    valid = mask.bool()
    filled = torch.where(valid, flow, torch.zeros_like(flow))
    left_flow, has_left = gather_nearest_valid_along_width(flow=filled, valid=valid)
    right_flow_reversed, has_right_reversed = gather_nearest_valid_along_width(
        flow=torch.flip(filled, dims=(3,)),
        valid=torch.flip(valid, dims=(3,)),
    )
    up_flow, has_up = gather_nearest_valid_along_height(flow=filled, valid=valid)
    down_flow_reversed, has_down_reversed = gather_nearest_valid_along_height(
        flow=torch.flip(filled, dims=(2,)),
        valid=torch.flip(valid, dims=(2,)),
    )

    right_flow = torch.flip(right_flow_reversed, dims=(3,))
    down_flow = torch.flip(down_flow_reversed, dims=(2,))
    has_right = torch.flip(has_right_reversed, dims=(3,))
    has_down = torch.flip(has_down_reversed, dims=(2,))
    direction_sum = left_flow + right_flow + up_flow + down_flow
    direction_count = (
        has_left.to(dtype=flow.dtype)
        + has_right.to(dtype=flow.dtype)
        + has_up.to(dtype=flow.dtype)
        + has_down.to(dtype=flow.dtype)
    )
    direction_average = direction_sum / direction_count.clamp_min(1.0)
    fillable = (~valid) & (direction_count > 0)
    return torch.where(fillable.expand_as(flow), direction_average, filled)


def fill_splatting_flow_holes(flow: Tensor, mask: Tensor, fill_strategy: str) -> Tensor:
    if fill_strategy == "zero":
        return flow

    if fill_strategy == "outside_in_4direction":
        return outside_in_fill_flow_holes_by_4direction(flow=flow, mask=mask)

    if fill_strategy == "outside_in_4neighbor":
        return outside_in_fill_flow_holes_by_neighbors(flow=flow, mask=mask, neighbor_mode="4neighbor")

    if fill_strategy == "outside_in_8neighbor":
        return outside_in_fill_flow_holes_by_neighbors(flow=flow, mask=mask, neighbor_mode="8neighbor")

    available_strategies = ", ".join(SPLATTING_FILL_STRATEGIES)
    raise ValueError(f"Unsupported splatting fill_strategy '{fill_strategy}'. Available strategies: {available_strategies}")


def nearest_depth_splat_flow(source_motion: Tensor, source_depth: Tensor) -> tuple[Tensor, Tensor]:
    import torch

    validate_flow_tensor("source_motion", source_motion)
    source_depth = normalize_depth_tensor("source_depth", source_depth, source_motion)

    batch_size = int(source_motion.shape[0])
    height = int(source_motion.shape[2])
    width = int(source_motion.shape[3])
    pixel_count = height * width
    source_grid = make_source_grid(batch_size, height, width, source_motion.device, source_motion.dtype)
    target_position = source_grid + source_motion
    target_x = target_position[:, 0].round().long()
    target_y = target_position[:, 1].round().long()
    valid = (target_x >= 0) & (target_x < width) & (target_y >= 0) & (target_y < height)

    flat_target_index = flatten_target_index(
        target_x=target_x.clamp(0, width - 1),
        target_y=target_y.clamp(0, height - 1),
        width=width,
    ).reshape(batch_size, -1)
    flat_depth = source_depth.reshape(batch_size, -1)
    flat_valid = valid.reshape(batch_size, -1)
    invalid_depth = torch.full_like(flat_depth, -torch.inf)
    flat_depth = torch.where(flat_valid, flat_depth, invalid_depth)

    depth_buffer = torch.full(
        (batch_size, pixel_count),
        -torch.inf,
        device=source_motion.device,
        dtype=source_motion.dtype,
    )
    if not hasattr(depth_buffer, "scatter_reduce_"):
        raise RuntimeError("linear_splatting requires torch.Tensor.scatter_reduce_; use a PyTorch build that provides it.")

    depth_buffer.scatter_reduce_(
        dim=1,
        index=flat_target_index,
        src=flat_depth,
        reduce=DEPTH_REDUCE_MODE,
        include_self=True,
    )

    selected_depth = depth_buffer.gather(dim=1, index=flat_target_index)
    selected = flat_valid & torch.isfinite(flat_depth) & (flat_depth == selected_depth)
    flat_selected = selected.to(dtype=source_motion.dtype)
    flat_value = (-source_motion).reshape(batch_size, 2, -1) * flat_selected.unsqueeze(1)

    splatted_flow_sum = torch.zeros(
        (batch_size, 2, pixel_count),
        device=source_motion.device,
        dtype=source_motion.dtype,
    )
    splatted_flow_sum.scatter_add_(
        dim=2,
        index=flat_target_index.unsqueeze(1).expand(batch_size, 2, pixel_count),
        src=flat_value,
    )

    hit_count = torch.zeros(
        (batch_size, pixel_count),
        device=source_motion.device,
        dtype=source_motion.dtype,
    )
    hit_count.scatter_add_(dim=1, index=flat_target_index, src=flat_selected)
    coverage_flat = hit_count > 0
    safe_hit_count = hit_count.clamp_min(1.0)
    splatted_flow = splatted_flow_sum / safe_hit_count.unsqueeze(1)
    coverage = coverage_flat.reshape(batch_size, 1, height, width).to(dtype=source_motion.dtype)
    return splatted_flow.reshape(batch_size, 2, height, width), coverage


def build_linear_splatting_flow_init(
    fmv_30: Tensor,
    bmv_30: Tensor,
    embt: Tensor,
    source_depth0: Tensor,
    source_depth1: Tensor,
) -> FlowInitResult:
    return build_linear_splatting_flow_init_with_fill_strategy(
        fmv_30=fmv_30,
        bmv_30=bmv_30,
        embt=embt,
        source_depth0=source_depth0,
        source_depth1=source_depth1,
        fill_strategy=DEFAULT_SPLATTING_FILL_STRATEGY,
    )


def build_linear_splatting_flow_init_with_fill_strategy(
    fmv_30: Tensor,
    bmv_30: Tensor,
    embt: Tensor,
    source_depth0: Tensor,
    source_depth1: Tensor,
    fill_strategy: str,
) -> FlowInitResult:
    import torch

    time = embt.reshape(embt.shape[0], 1, 1, 1)
    source_depth0 = normalize_depth_tensor("source_depth0", source_depth0, fmv_30)
    source_depth1 = normalize_depth_tensor("source_depth1", source_depth1, bmv_30)
    partial_fmv = time * fmv_30
    partial_bmv = (1 - time) * bmv_30

    approx_bmv, bmv_mask = nearest_depth_splat_flow(partial_fmv, source_depth0)
    approx_fmv, fmv_mask = nearest_depth_splat_flow(partial_bmv, source_depth1)

    approx_fmv = fill_splatting_flow_holes(flow=approx_fmv, mask=fmv_mask, fill_strategy=fill_strategy)
    approx_bmv = fill_splatting_flow_holes(flow=approx_bmv, mask=bmv_mask, fill_strategy=fill_strategy)
    return FlowInitResult(bmv=approx_bmv, fmv=approx_fmv, masks=torch.cat((bmv_mask, fmv_mask), dim=1))


def build_flow_init_result(
    fmv_30: Tensor,
    bmv_30: Tensor,
    embt: Tensor,
    flow_approx_method: str,
    source_depth0: Tensor | None,
    source_depth1: Tensor | None,
) -> FlowInitResult:
    return build_flow_init_result_with_fill_strategy(
        fmv_30=fmv_30,
        bmv_30=bmv_30,
        embt=embt,
        flow_approx_method=flow_approx_method,
        source_depth0=source_depth0,
        source_depth1=source_depth1,
        splatting_fill_strategy=DEFAULT_SPLATTING_FILL_STRATEGY,
    )


def build_flow_init_result_with_fill_strategy(
    fmv_30: Tensor,
    bmv_30: Tensor,
    embt: Tensor,
    flow_approx_method: str,
    source_depth0: Tensor | None,
    source_depth1: Tensor | None,
    splatting_fill_strategy: str,
) -> FlowInitResult:
    time = embt.reshape(embt.shape[0], 1, 1, 1)

    if flow_approx_method == "single":
        approx_fmv = flow_approx(fmv_30, time, True)
        approx_bmv = flow_approx(bmv_30, time, False)
        return FlowInitResult(bmv=approx_bmv, fmv=approx_fmv, masks=None)

    if flow_approx_method == "combination":
        approx_fmv = flow_approx_combination(fmv_30, bmv_30, time, True)
        approx_bmv = flow_approx_combination(fmv_30, bmv_30, time, False)
        return FlowInitResult(bmv=approx_bmv, fmv=approx_fmv, masks=None)

    if is_splatting_flow_approx_method(flow_approx_method=flow_approx_method):
        if source_depth0 is None or source_depth1 is None:
            raise ValueError("linear_splatting flow approximation requires source_depth0 and source_depth1 tensors.")

        return build_linear_splatting_flow_init_with_fill_strategy(
            fmv_30=fmv_30,
            bmv_30=bmv_30,
            embt=embt,
            source_depth0=source_depth0,
            source_depth1=source_depth1,
            fill_strategy=resolve_splatting_fill_strategy(
                flow_approx_method=flow_approx_method,
                splatting_fill_strategy=splatting_fill_strategy,
            ),
        )

    available_methods = ", ".join(FLOW_APPROX_METHOD_CHOICES)
    raise ValueError(f"Unsupported flow_approx_method '{flow_approx_method}'. Available methods: {available_methods}")


def build_flow_init(
    fmv_30: Tensor,
    bmv_30: Tensor,
    embt: Tensor,
    flow_approx_method: str,
) -> tuple[Tensor, Tensor]:
    result = build_flow_init_result(
        fmv_30=fmv_30,
        bmv_30=bmv_30,
        embt=embt,
        flow_approx_method=flow_approx_method,
        source_depth0=None,
        source_depth1=None,
    )
    return result.bmv, result.fmv
