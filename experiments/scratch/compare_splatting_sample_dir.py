from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch

from src.data.dataset_loader import depth_to_tensor
from src.data.dataset_loader import flow_to_tensor
from src.data.dataset_loader import image_to_tensor
from src.data.image_ops import flow_to_image
from src.data.image_ops import load_backward_velocity
from src.data.image_ops import load_png
from src.data.image_ops import save_image
from src.engine.evaluation import calculate_psnr
from src.engine.flow_approx import FLOW_APPROX_METHODS
from src.engine.flow_approx import build_flow_init_result
from src.engine.flow_approx import flatten_target_index
from src.engine.flow_approx import make_source_grid
from src.models.external.IFRNet.utils import warp


RegionMaps = dict[str, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare flow approximation modes on one flat sample directory.")
    parser.add_argument("--sample-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--frame-0", required=True, type=int)
    parser.add_argument("--frame-t", required=True, type=int)
    parser.add_argument("--frame-1", required=True, type=int)
    parser.add_argument("--warmup-iters", required=True, type=int)
    parser.add_argument("--timing-iters", required=True, type=int)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: path={path}")

    return path


def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = load_png(require_file(path))[:, :, :3]
    return image_to_tensor(image).unsqueeze(0).to(device)


def load_flow_and_depth_tensors(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    flow, depth = load_backward_velocity(require_file(path))
    flow_tensor = flow_to_tensor(flow).unsqueeze(0).to(device)
    depth_tensor = depth_to_tensor(depth).unsqueeze(0).to(device)
    return flow_tensor, depth_tensor


def blend_warps(img0_warped: torch.Tensor, img1_warped: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
    if masks is None:
        return 0.5 * img0_warped + 0.5 * img1_warped

    bmv_mask = masks[:, 0:1]
    fmv_mask = masks[:, 1:2]
    weight_sum = bmv_mask + fmv_mask
    average_blend = 0.5 * img0_warped + 0.5 * img1_warped
    coverage_blend = (img0_warped * bmv_mask + img1_warped * fmv_mask) / weight_sum.clamp_min(1.0)
    return torch.where(weight_sum > 0, coverage_blend, average_blend)


def calculate_mae(target: torch.Tensor, prediction: torch.Tensor) -> float:
    return float((target - prediction).abs().mean().detach().cpu().item())


def calculate_epe(flow0: torch.Tensor, flow1: torch.Tensor, target_flow0: torch.Tensor, target_flow1: torch.Tensor) -> float:
    epe0 = (flow0 - target_flow0).norm(dim=1)
    epe1 = (flow1 - target_flow1).norm(dim=1)
    return float(torch.cat((epe0.reshape(1, -1), epe1.reshape(1, -1)), dim=1).mean().detach().cpu().item())


def calculate_masked_psnr(target: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor) -> float:
    mask_float = mask.to(device=target.device, dtype=target.dtype)
    if mask_float.ndim == 2:
        mask_float = mask_float.unsqueeze(0)

    selected_pixels = mask_float.sum()
    if float(selected_pixels.detach().cpu().item()) <= 0.0:
        return -1.0

    squared_error = (target - prediction) * (target - prediction)
    channel_count = int(target.shape[0])
    mse = (squared_error * mask_float).sum() / (selected_pixels * channel_count)
    return float((-10 * torch.log10(mse)).detach().cpu().item())


def build_nearest_splat_hit_count(source_motion: torch.Tensor) -> torch.Tensor:
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
    flat_valid = valid.reshape(batch_size, -1).to(dtype=source_motion.dtype)
    hit_count = torch.zeros((batch_size, pixel_count), device=source_motion.device, dtype=source_motion.dtype)
    hit_count.scatter_add_(dim=1, index=flat_target_index, src=flat_valid)
    return hit_count.reshape(batch_size, 1, height, width)


def build_region_maps(fmv_30: torch.Tensor, bmv_30: torch.Tensor, embt: torch.Tensor, masks: torch.Tensor) -> RegionMaps:
    time_tensor = embt.reshape(embt.shape[0], 1, 1, 1)
    bmv_hit_count = build_nearest_splat_hit_count(time_tensor * fmv_30)
    fmv_hit_count = build_nearest_splat_hit_count((1 - time_tensor) * bmv_30)
    bmv_hit = masks[:, 0:1] > 0
    fmv_hit = masks[:, 1:2] > 0
    hit_both = bmv_hit & fmv_hit
    return {
        "all": torch.ones_like(hit_both, dtype=torch.bool),
        "hit_both": hit_both,
        "hole_any": ~hit_both,
        "bmv_hit": bmv_hit,
        "fmv_hit": fmv_hit,
        "bmv_hole": ~bmv_hit,
        "fmv_hole": ~fmv_hit,
        "bmv_many_to_one": bmv_hit_count > 1,
        "fmv_many_to_one": fmv_hit_count > 1,
        "many_to_one_any": (bmv_hit_count > 1) | (fmv_hit_count > 1),
        "bmv_hit_count": bmv_hit_count,
        "fmv_hit_count": fmv_hit_count,
    }


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_method(
    method_name: str,
    img0: torch.Tensor,
    img1: torch.Tensor,
    fmv_30: torch.Tensor,
    bmv_30: torch.Tensor,
    source_depth0: torch.Tensor,
    source_depth1: torch.Tensor,
    embt: torch.Tensor,
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float, float]:
    flow_init = build_flow_init_result(fmv_30, bmv_30, embt, method_name, source_depth0, source_depth1)
    for _index in range(warmup_iters):
        flow_init = build_flow_init_result(fmv_30, bmv_30, embt, method_name, source_depth0, source_depth1)
        _blend = blend_warps(warp(img0, flow_init.bmv), warp(img1, flow_init.fmv), flow_init.masks)

    init_elapsed_ms: list[float] = []
    total_elapsed_ms: list[float] = []
    for _index in range(timing_iters):
        synchronize_if_needed(device)
        init_start = time.perf_counter()
        flow_init = build_flow_init_result(fmv_30, bmv_30, embt, method_name, source_depth0, source_depth1)
        synchronize_if_needed(device)
        init_elapsed_ms.append((time.perf_counter() - init_start) * 1000.0)

        synchronize_if_needed(device)
        total_start = time.perf_counter()
        flow_init = build_flow_init_result(fmv_30, bmv_30, embt, method_name, source_depth0, source_depth1)
        _blend = blend_warps(warp(img0, flow_init.bmv), warp(img1, flow_init.fmv), flow_init.masks)
        synchronize_if_needed(device)
        total_elapsed_ms.append((time.perf_counter() - total_start) * 1000.0)

    init_elapsed_ms.sort()
    total_elapsed_ms.sort()
    return (
        flow_init.bmv,
        flow_init.fmv,
        flow_init.masks,
        init_elapsed_ms[len(init_elapsed_ms) // 2],
        total_elapsed_ms[len(total_elapsed_ms) // 2],
    )


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    return np.round(tensor[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def mask_to_uint8(mask: torch.Tensor) -> np.ndarray:
    return np.round(mask[0, 0].detach().cpu().numpy().astype(np.float32) * 255.0).astype(np.uint8)


def hit_count_to_color(hit_count: torch.Tensor) -> np.ndarray:
    count_np = hit_count[0, 0].detach().cpu().numpy().astype(np.float32)
    scale = max(float(np.percentile(count_np, 99.0)), 1.0)
    count_u8 = np.round(np.clip(count_np / scale, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(count_u8, cv2.COLORMAP_TURBO)


def build_direction_region_label_map(hit: torch.Tensor, many_to_one: torch.Tensor) -> np.ndarray:
    hit_np = hit[0, 0].detach().cpu().numpy().astype(bool)
    many_to_one_np = many_to_one[0, 0].detach().cpu().numpy().astype(bool)
    hole = ~hit_np
    label = np.zeros(hole.shape, dtype=np.uint8)
    label[hole] = 128
    label[many_to_one_np] = 255
    return label


def colorize_region_label(label: np.ndarray, hit_count: torch.Tensor) -> np.ndarray:
    hit_count_np = hit_count[0, 0].detach().cpu().numpy().astype(np.float32)
    color = np.zeros((*label.shape, 3), dtype=np.uint8)
    color[label == 0] = np.array([24, 24, 24], dtype=np.uint8)
    color[label == 128] = np.array([220, 220, 220], dtype=np.uint8)
    many_to_one = label == 255
    color[many_to_one & (hit_count_np <= 2)] = np.array([80, 210, 255], dtype=np.uint8)
    color[many_to_one & (hit_count_np == 3)] = np.array([40, 200, 80], dtype=np.uint8)
    color[many_to_one & (hit_count_np == 4)] = np.array([0, 170, 255], dtype=np.uint8)
    color[many_to_one & (hit_count_np >= 5)] = np.array([40, 40, 220], dtype=np.uint8)

    height, width = label.shape
    total_pixels = float(label.size)
    normal_ratio = float(np.count_nonzero(label == 0) / total_pixels * 100.0)
    hole_ratio = float(np.count_nonzero(label == 128) / total_pixels * 100.0)
    multi2_ratio = float(np.count_nonzero(many_to_one & (hit_count_np <= 2)) / total_pixels * 100.0)
    multi3_ratio = float(np.count_nonzero(many_to_one & (hit_count_np == 3)) / total_pixels * 100.0)
    multi4_ratio = float(np.count_nonzero(many_to_one & (hit_count_np == 4)) / total_pixels * 100.0)
    multi5_ratio = float(np.count_nonzero(many_to_one & (hit_count_np >= 5)) / total_pixels * 100.0)

    legend_width = 170
    canvas = np.full((height, width + legend_width, 3), 255, dtype=np.uint8)
    canvas[:, :width] = color
    legend_items = (
        (f"normal {normal_ratio:.1f}%", (24, 24, 24)),
        (f"hole {hole_ratio:.1f}%", (220, 220, 220)),
        (f"multi=2 {multi2_ratio:.1f}%", (80, 210, 255)),
        (f"multi=3 {multi3_ratio:.1f}%", (40, 200, 80)),
        (f"multi=4 {multi4_ratio:.1f}%", (0, 170, 255)),
        (f"multi>=5 {multi5_ratio:.1f}%", (40, 40, 220)),
    )
    cv2.putText(canvas, "legend", (width + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for index, (label_text, bgr_color) in enumerate(legend_items):
        y = 54 + index * 40
        cv2.rectangle(canvas, (width + 12, y - 18), (width + 34, y + 4), bgr_color, -1)
        cv2.rectangle(canvas, (width + 12, y - 18), (width + 34, y + 4), (0, 0, 0), 1)
        cv2.putText(
            canvas,
            label_text,
            (width + 42, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def build_record(
    method_name: str,
    img0: torch.Tensor,
    img1: torch.Tensor,
    imgt: torch.Tensor,
    bmv: torch.Tensor,
    fmv: torch.Tensor,
    masks: torch.Tensor | None,
    bmv_60: torch.Tensor,
    fmv_60: torch.Tensor,
    region_maps: RegionMaps,
    init_runtime_ms: float,
    total_runtime_ms: float,
) -> tuple[dict[str, object], torch.Tensor]:
    img0_warped = warp(img0, bmv)
    img1_warped = warp(img1, fmv)
    blend = blend_warps(img0_warped, img1_warped, masks)
    record: dict[str, object] = {
        "method": method_name,
        "psnr_all": float(calculate_psnr(imgt[0], blend[0]).detach().cpu().item()),
        "mae_all": calculate_mae(imgt, blend),
        "flow_epe_all": calculate_epe(bmv, fmv, bmv_60, fmv_60),
        "init_runtime_ms": init_runtime_ms,
        "total_runtime_ms": total_runtime_ms,
    }
    for region_name in ("hit_both", "hole_any", "bmv_hole", "fmv_hole", "many_to_one_any", "bmv_many_to_one", "fmv_many_to_one"):
        region_mask = region_maps[region_name][0, 0]
        record[f"{region_name}_ratio"] = float(region_mask.detach().cpu().float().mean().item())
        record[f"psnr_{region_name}"] = calculate_masked_psnr(imgt[0], blend[0], region_mask)

    if masks is None:
        record["coverage_bmv"] = 1.0
        record["coverage_fmv"] = 1.0
    else:
        record["coverage_bmv"] = float(masks[:, 0:1].mean().detach().cpu().item())
        record["coverage_fmv"] = float(masks[:, 1:2].mean().detach().cpu().item())

    record["coverage_mean"] = (float(record["coverage_bmv"]) + float(record["coverage_fmv"])) / 2.0
    return record, blend


def save_region_maps(output_dir: Path, region_maps: RegionMaps) -> None:
    bmv_label = build_direction_region_label_map(region_maps["bmv_hit"], region_maps["bmv_many_to_one"])
    fmv_label = build_direction_region_label_map(region_maps["fmv_hit"], region_maps["fmv_many_to_one"])
    save_image(output_dir / "region_splatting_label_bmv.png", bmv_label)
    save_image(
        output_dir / "region_splatting_label_bmv_color.png",
        colorize_region_label(bmv_label, region_maps["bmv_hit_count"]),
    )
    save_image(output_dir / "region_splatting_label_fmv.png", fmv_label)
    save_image(
        output_dir / "region_splatting_label_fmv_color.png",
        colorize_region_label(fmv_label, region_maps["fmv_hit_count"]),
    )
    save_image(output_dir / "region_hit_both.png", mask_to_uint8(region_maps["hit_both"]))
    save_image(output_dir / "region_hole_any.png", mask_to_uint8(region_maps["hole_any"]))
    save_image(output_dir / "region_bmv_hit.png", mask_to_uint8(region_maps["bmv_hit"]))
    save_image(output_dir / "region_fmv_hit.png", mask_to_uint8(region_maps["fmv_hit"]))
    save_image(output_dir / "region_bmv_hole.png", mask_to_uint8(region_maps["bmv_hole"]))
    save_image(output_dir / "region_fmv_hole.png", mask_to_uint8(region_maps["fmv_hole"]))
    save_image(output_dir / "region_many_to_one_any.png", mask_to_uint8(region_maps["many_to_one_any"]))
    save_image(output_dir / "region_bmv_many_to_one.png", mask_to_uint8(region_maps["bmv_many_to_one"]))
    save_image(output_dir / "region_fmv_many_to_one.png", mask_to_uint8(region_maps["fmv_many_to_one"]))
    save_image(output_dir / "region_bmv_hit_count.png", hit_count_to_color(region_maps["bmv_hit_count"]))
    save_image(output_dir / "region_fmv_hit_count.png", hit_count_to_color(region_maps["fmv_hit_count"]))


def run_experiment(args: argparse.Namespace) -> list[dict[str, object]]:
    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img0 = load_image_tensor(sample_dir / f"colorNoScreenUI_{args.frame_0}.png", device)
    imgt = load_image_tensor(sample_dir / f"colorNoScreenUI_{args.frame_t}.png", device)
    img1 = load_image_tensor(sample_dir / f"colorNoScreenUI_{args.frame_1}.png", device)
    bmv_60, _depth_t_bmv = load_flow_and_depth_tensors(sample_dir / f"backwardVel_Depth_{args.frame_t}.exr", device)
    fmv_60, _depth_t_fmv = load_flow_and_depth_tensors(sample_dir / f"forwardVel_Depth_{args.frame_t}.exr", device)
    bmv_30, source_depth1 = load_flow_and_depth_tensors(sample_dir / f"backwardVel_Depth_{args.frame_1 // 2}_fps30.exr", device)
    fmv_30, source_depth0 = load_flow_and_depth_tensors(sample_dir / f"forwardVel_Depth_{args.frame_0 // 2}_fps30.exr", device)
    embt = torch.tensor([[[0.5]]], dtype=torch.float32, device=device)

    splatting_init = build_flow_init_result(fmv_30, bmv_30, embt, "splatting", source_depth0, source_depth1)
    if splatting_init.masks is None:
        raise ValueError("Splatting method did not return masks.")

    region_maps = build_region_maps(fmv_30, bmv_30, embt, splatting_init.masks)
    save_region_maps(output_dir, region_maps)
    save_image(output_dir / "image_0.png", tensor_to_uint8_image(img0))
    save_image(output_dir / "image_t.png", tensor_to_uint8_image(imgt))
    save_image(output_dir / "image_1.png", tensor_to_uint8_image(img1))

    records: list[dict[str, object]] = []
    for method_name in FLOW_APPROX_METHODS:
        bmv, fmv, masks, init_runtime_ms, total_runtime_ms = measure_method(
            method_name,
            img0,
            img1,
            fmv_30,
            bmv_30,
            source_depth0,
            source_depth1,
            embt,
            device,
            int(args.warmup_iters),
            int(args.timing_iters),
        )
        record, blend = build_record(
            method_name,
            img0,
            img1,
            imgt,
            bmv,
            fmv,
            masks,
            bmv_60,
            fmv_60,
            region_maps,
            init_runtime_ms,
            total_runtime_ms,
        )
        records.append(record)
        save_image(output_dir / f"blend_{method_name}.png", tensor_to_uint8_image(blend))
        save_image(output_dir / f"flow_t_to_0_{method_name}.png", flow_to_image(bmv[0].detach().cpu().permute(1, 2, 0).numpy()))
        save_image(output_dir / f"flow_t_to_1_{method_name}.png", flow_to_image(fmv[0].detach().cpu().permute(1, 2, 0).numpy()))

    direct_total_ms: list[float] = []
    for _index in range(int(args.warmup_iters)):
        _blend = blend_warps(warp(img0, bmv_60), warp(img1, fmv_60), None)

    for _index in range(int(args.timing_iters)):
        synchronize_if_needed(device)
        start_time = time.perf_counter()
        _blend = blend_warps(warp(img0, bmv_60), warp(img1, fmv_60), None)
        synchronize_if_needed(device)
        direct_total_ms.append((time.perf_counter() - start_time) * 1000.0)

    direct_total_ms.sort()
    direct_record, direct_blend = build_record(
        "direct_fps60_target_flow",
        img0,
        img1,
        imgt,
        bmv_60,
        fmv_60,
        None,
        bmv_60,
        fmv_60,
        region_maps,
        0.0,
        direct_total_ms[len(direct_total_ms) // 2],
    )
    records.append(direct_record)
    save_image(output_dir / "blend_direct_fps60_target_flow.png", tensor_to_uint8_image(direct_blend))

    payload = {
        "sample_dir": str(sample_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "frames": {"frame_0": int(args.frame_0), "frame_t": int(args.frame_t), "frame_1": int(args.frame_1)},
        "records": records,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(output_dir / "metrics.csv", index=False)
    return records


def main() -> None:
    args = parse_args()
    records = run_experiment(args)
    print(json.dumps({"records": records}, indent=2))


if __name__ == "__main__":
    main()
