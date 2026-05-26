from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset_loader import depth_to_tensor
from src.data.dataset_loader import flow_to_tensor
from src.data.dataset_loader import image_to_tensor
from src.data.image_ops import load_backward_velocity
from src.data.image_ops import load_png
from src.engine.evaluation import calculate_psnr
from src.engine.flow_approx import FLOW_APPROX_METHODS
from src.engine.flow_approx import build_flow_init_result
from src.models.external.IFRNet.utils import warp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare linear flow approximation modes on one fps60 triplet.")
    parser.add_argument("--dataset-root-dir", required=True, type=str)
    parser.add_argument("--record", default="ARPG_3", type=str)
    parser.add_argument("--mode", default="1_Medium/1_Medium_5", type=str)
    parser.add_argument("--frame-0", default=452, type=int)
    parser.add_argument("--frame-t", default=453, type=int)
    parser.add_argument("--frame-1", default=454, type=int)
    parser.add_argument("--warmup-iters", default=5, type=int)
    parser.add_argument("--timing-iters", default=30, type=int)
    parser.add_argument("--output-json", default="", type=str)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: path={path}")

    return path


def build_frame_path(dataset_root_dir: Path, record: str, mode: str, fps: int, prefix: str, frame_index: int, ext: str) -> Path:
    return dataset_root_dir / record / mode / f"fps_{fps}" / f"{prefix}_{frame_index}{ext}"


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


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_method(
    method_name: str,
    img0: torch.Tensor,
    img1: torch.Tensor,
    bmv_30: torch.Tensor,
    fmv_30: torch.Tensor,
    source_depth0: torch.Tensor,
    source_depth1: torch.Tensor,
    embt: torch.Tensor,
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    flow_init = build_flow_init_result(
        fmv_30=fmv_30,
        bmv_30=bmv_30,
        embt=embt,
        flow_approx_method=method_name,
        source_depth0=source_depth0,
        source_depth1=source_depth1,
    )

    for _warmup_index in range(warmup_iters):
        flow_init = build_flow_init_result(
            fmv_30=fmv_30,
            bmv_30=bmv_30,
            embt=embt,
            flow_approx_method=method_name,
            source_depth0=source_depth0,
            source_depth1=source_depth1,
        )
        _img0_warped = warp(img0, flow_init.bmv)
        _img1_warped = warp(img1, flow_init.fmv)
        _blend = blend_warps(_img0_warped, _img1_warped, flow_init.masks)

    elapsed_values_ms: list[float] = []
    for _timing_index in range(timing_iters):
        synchronize_if_needed(device)
        start_time = time.perf_counter()
        flow_init = build_flow_init_result(
            fmv_30=fmv_30,
            bmv_30=bmv_30,
            embt=embt,
            flow_approx_method=method_name,
            source_depth0=source_depth0,
            source_depth1=source_depth1,
        )
        _img0_warped = warp(img0, flow_init.bmv)
        _img1_warped = warp(img1, flow_init.fmv)
        _blend = blend_warps(_img0_warped, _img1_warped, flow_init.masks)
        synchronize_if_needed(device)
        elapsed_values_ms.append((time.perf_counter() - start_time) * 1000.0)

    elapsed_values_ms.sort()
    elapsed_ms = elapsed_values_ms[len(elapsed_values_ms) // 2]
    return flow_init.bmv, flow_init.fmv, flow_init.masks, elapsed_ms


def build_method_record(
    method_name: str,
    img0: torch.Tensor,
    img1: torch.Tensor,
    imgt: torch.Tensor,
    bmv: torch.Tensor,
    fmv: torch.Tensor,
    masks: torch.Tensor | None,
    bmv_60: torch.Tensor,
    fmv_60: torch.Tensor,
    runtime_ms: float,
) -> dict[str, object]:
    img0_warped = warp(img0, bmv)
    img1_warped = warp(img1, fmv)
    blend = blend_warps(img0_warped, img1_warped, masks)
    coverage_bmv = 1.0
    coverage_fmv = 1.0
    if masks is not None:
        coverage_bmv = float(masks[:, 0].mean().detach().cpu().item())
        coverage_fmv = float(masks[:, 1].mean().detach().cpu().item())

    return {
        "method": method_name,
        "psnr": float(calculate_psnr(imgt[0], blend[0]).detach().cpu().item()),
        "mae": calculate_mae(imgt, blend),
        "coverage_bmv": coverage_bmv,
        "coverage_fmv": coverage_fmv,
        "coverage_mean": (coverage_bmv + coverage_fmv) / 2.0,
        "flow_epe": calculate_epe(bmv, fmv, bmv_60, fmv_60),
        "runtime_ms": runtime_ms,
    }


def run_comparison(args: argparse.Namespace) -> list[dict[str, object]]:
    dataset_root_dir = Path(args.dataset_root_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps60_mode = str(args.mode)
    fps30_mode = fps60_mode

    img0 = load_image_tensor(
        build_frame_path(dataset_root_dir, args.record, fps60_mode, 60, "colorNoScreenUI", args.frame_0, ".png"),
        device,
    )
    imgt = load_image_tensor(
        build_frame_path(dataset_root_dir, args.record, fps60_mode, 60, "colorNoScreenUI", args.frame_t, ".png"),
        device,
    )
    img1 = load_image_tensor(
        build_frame_path(dataset_root_dir, args.record, fps60_mode, 60, "colorNoScreenUI", args.frame_1, ".png"),
        device,
    )
    bmv_60, _depth_t_bmv = load_flow_and_depth_tensors(
        build_frame_path(dataset_root_dir, args.record, fps60_mode, 60, "backwardVel_Depth", args.frame_t, ".exr"),
        device,
    )
    fmv_60, _depth_t_fmv = load_flow_and_depth_tensors(
        build_frame_path(dataset_root_dir, args.record, fps60_mode, 60, "forwardVel_Depth", args.frame_t, ".exr"),
        device,
    )
    bmv_30, source_depth1 = load_flow_and_depth_tensors(
        build_frame_path(dataset_root_dir, args.record, fps30_mode, 30, "backwardVel_Depth", args.frame_1 // 2, ".exr"),
        device,
    )
    fmv_30, source_depth0 = load_flow_and_depth_tensors(
        build_frame_path(dataset_root_dir, args.record, fps30_mode, 30, "forwardVel_Depth", args.frame_0 // 2, ".exr"),
        device,
    )
    embt = torch.tensor([[[0.5]]], dtype=torch.float32, device=device)

    records: list[dict[str, object]] = []
    for method_name in FLOW_APPROX_METHODS:
        bmv, fmv, masks, runtime_ms = time_method(
            method_name=method_name,
            img0=img0,
            img1=img1,
            bmv_30=bmv_30,
            fmv_30=fmv_30,
            source_depth0=source_depth0,
            source_depth1=source_depth1,
            embt=embt,
            device=device,
            warmup_iters=int(args.warmup_iters),
            timing_iters=int(args.timing_iters),
        )
        records.append(
            build_method_record(
                method_name=method_name,
                img0=img0,
                img1=img1,
                imgt=imgt,
                bmv=bmv,
                fmv=fmv,
                masks=masks,
                bmv_60=bmv_60,
                fmv_60=fmv_60,
                runtime_ms=runtime_ms,
            )
        )

    for _warmup_index in range(int(args.warmup_iters)):
        _img0_warped = warp(img0, bmv_60)
        _img1_warped = warp(img1, fmv_60)
        _blend = blend_warps(_img0_warped, _img1_warped, None)

    direct_elapsed_values_ms: list[float] = []
    for _timing_index in range(int(args.timing_iters)):
        synchronize_if_needed(device)
        start_time = time.perf_counter()
        _img0_warped = warp(img0, bmv_60)
        _img1_warped = warp(img1, fmv_60)
        _blend = blend_warps(_img0_warped, _img1_warped, None)
        synchronize_if_needed(device)
        direct_elapsed_values_ms.append((time.perf_counter() - start_time) * 1000.0)

    direct_elapsed_values_ms.sort()
    direct_runtime_ms = direct_elapsed_values_ms[len(direct_elapsed_values_ms) // 2]
    records.append(
        build_method_record(
            method_name="direct_fps60_target_flow",
            img0=img0,
            img1=img1,
            imgt=imgt,
            bmv=bmv_60,
            fmv=fmv_60,
            masks=None,
            bmv_60=bmv_60,
            fmv_60=fmv_60,
            runtime_ms=direct_runtime_ms,
        )
    )
    return records


def main() -> None:
    args = parse_args()
    records = run_comparison(args)
    payload: dict[str, Any] = {"records": records}
    print(json.dumps(payload, indent=2))
    if args.output_json != "":
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
