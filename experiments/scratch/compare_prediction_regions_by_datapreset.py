from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

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
from src.data.image_ops import load_backward_velocity
from src.engine.flow_approx import build_flow_init_result
from src.engine.flow_approx import flatten_target_index
from src.engine.flow_approx import make_source_grid


class SamplePreset(TypedDict):
    sample_id: str
    sample_dir: str
    candidate_result_dir: str
    baseline_result_dir: str
    frame_0: int
    frame_t: int
    frame_1: int


class DataPreset(TypedDict):
    candidate_name: str
    baseline_name: str
    samples: list[SamplePreset]


RegionMaps = dict[str, np.ndarray]

KERNEL_SIZE: int = 9
CONFIDENCE_THRESHOLD: float = 0.08
SAVE_OVERLAY_IMAGES: bool = True
OUTPUT_ROOT: Path = Path(tempfile.gettempdir()) / "GFI_prediction_region_compare"

DATA_PRESETS: dict[str, DataPreset] = {
    "arpg_3_1_difficult_5_0452_0454": {
        "candidate_name": "Splat Training",
        "baseline_name": "FineTuning",
        "samples": [
            {
                "sample_id": "ARPG_3_1_Difficult_5_frame_0452_0454",
                "sample_dir": (
                    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260527 - Lab Meeting"
                    r"\warping_testing_case\ARPG_3_1_Difficult_5\Medium_frame_0452_0454"
                ),
                "candidate_result_dir": (
                    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260527 - Lab Meeting"
                    r"\warping_testing_case\ARPG_3_1_Difficult_5_Splat"
                ),
                "baseline_result_dir": (
                    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260527 - Lab Meeting"
                    r"\warping_testing_case\ARPG_3_1_Difficult_5_FineTuning"
                ),
                "frame_0": 452,
                "frame_t": 453,
                "frame_1": 454,
            },
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare prediction quality by splatting regions for one data preset.")
    parser.add_argument("--data-preset", required=True, choices=tuple(sorted(DATA_PRESETS.keys())))
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: path={path}")

    return path


def read_rgb_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(require_file(path)), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: path={path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    image_u8 = np.clip(np.round(image_rgb * 255.0), 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def load_flow_and_depth_tensors(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    flow, depth = load_backward_velocity(require_file(path))
    flow_tensor = flow_to_tensor(flow).unsqueeze(0).to(device)
    depth_tensor = depth_to_tensor(depth).unsqueeze(0).to(device)
    return flow_tensor, depth_tensor


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


def tensor_mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    return mask[0, 0].detach().cpu().numpy().astype(bool)


def build_region_maps(sample: SamplePreset, device: torch.device) -> RegionMaps:
    sample_dir = Path(sample["sample_dir"])
    frame_0 = int(sample["frame_0"])
    frame_1 = int(sample["frame_1"])
    bmv_30, source_depth1 = load_flow_and_depth_tensors(sample_dir / f"backwardVel_Depth_{frame_1 // 2}_fps30.exr", device)
    fmv_30, source_depth0 = load_flow_and_depth_tensors(sample_dir / f"forwardVel_Depth_{frame_0 // 2}_fps30.exr", device)
    embt = torch.tensor([[[0.5]]], dtype=torch.float32, device=device)
    flow_init = build_flow_init_result(fmv_30, bmv_30, embt, "splatting", source_depth0, source_depth1)
    if flow_init.masks is None:
        raise ValueError("Splatting flow init did not return coverage masks.")

    time_tensor = embt.reshape(embt.shape[0], 1, 1, 1)
    bmv_hit_count = build_nearest_splat_hit_count(time_tensor * fmv_30)
    fmv_hit_count = build_nearest_splat_hit_count((1 - time_tensor) * bmv_30)
    bmv_hit = flow_init.masks[:, 0:1] > 0
    fmv_hit = flow_init.masks[:, 1:2] > 0
    bmv_many_to_one = bmv_hit_count > 1
    fmv_many_to_one = fmv_hit_count > 1
    bmv_hole = ~bmv_hit
    fmv_hole = ~fmv_hit
    return {
        "all": np.ones_like(tensor_mask_to_numpy(bmv_hit), dtype=bool),
        "hit_both": tensor_mask_to_numpy(bmv_hit & fmv_hit),
        "clean_both_no_hole_no_multi": tensor_mask_to_numpy(~(bmv_hole | fmv_hole | bmv_many_to_one | fmv_many_to_one)),
        "hole_any": tensor_mask_to_numpy(bmv_hole | fmv_hole),
        "many_to_one_any": tensor_mask_to_numpy(bmv_many_to_one | fmv_many_to_one),
        "bmv_hole": tensor_mask_to_numpy(bmv_hole),
        "fmv_hole": tensor_mask_to_numpy(fmv_hole),
        "bmv_many_to_one": tensor_mask_to_numpy(bmv_many_to_one),
        "fmv_many_to_one": tensor_mask_to_numpy(fmv_many_to_one),
    }


def validate_same_shape(name: str, expected: np.ndarray, current: np.ndarray) -> None:
    if expected.shape != current.shape:
        raise ValueError(f"Shape mismatch for {name}: expected={expected.shape} actual={current.shape}")


def masked_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if mask.dtype != bool:
        raise TypeError(f"Mask must be bool: dtype={mask.dtype}")

    if int(mask.sum()) == 0:
        return float("nan")

    diff = prediction[mask] - target[mask]
    return float(np.mean(diff * diff))


def masked_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if mask.dtype != bool:
        raise TypeError(f"Mask must be bool: dtype={mask.dtype}")

    if int(mask.sum()) == 0:
        return float("nan")

    return float(np.mean(np.abs(prediction[mask] - target[mask])))


def mse_to_psnr(mse: float) -> float:
    if np.isnan(mse):
        return float("nan")

    if mse <= 0.0:
        return float("inf")

    return float(-10.0 * np.log10(mse))


def build_local_winner_masks(
    target: np.ndarray,
    candidate_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_error = np.mean((candidate_prediction - target) ** 2, axis=2)
    baseline_error = np.mean((baseline_prediction - target) ** 2, axis=2)
    candidate_local_error = cv2.blur(candidate_error, (KERNEL_SIZE, KERNEL_SIZE))
    baseline_local_error = cv2.blur(baseline_error, (KERNEL_SIZE, KERNEL_SIZE))
    confidence = np.abs(baseline_local_error - candidate_local_error) / (baseline_local_error + candidate_local_error + 1e-12)
    candidate_better = (candidate_local_error < baseline_local_error) & (confidence >= CONFIDENCE_THRESHOLD)
    baseline_better = (baseline_local_error < candidate_local_error) & (confidence >= CONFIDENCE_THRESHOLD)
    neutral = ~(candidate_better | baseline_better)
    return candidate_better, baseline_better, neutral


def build_region_row(
    data_preset_name: str,
    sample: SamplePreset,
    candidate_name: str,
    baseline_name: str,
    region_name: str,
    region_mask: np.ndarray,
    target: np.ndarray,
    candidate_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_better: np.ndarray,
    baseline_better: np.ndarray,
    neutral: np.ndarray,
) -> dict[str, object]:
    total_pixels = int(target.shape[0] * target.shape[1])
    region_pixels = int(region_mask.sum())
    candidate_mse = masked_mse(candidate_prediction, target, region_mask)
    baseline_mse = masked_mse(baseline_prediction, target, region_mask)
    if region_pixels == 0:
        candidate_better_ratio = float("nan")
        baseline_better_ratio = float("nan")
        neutral_ratio = float("nan")
    else:
        candidate_better_ratio = float(np.count_nonzero(candidate_better & region_mask) / region_pixels)
        baseline_better_ratio = float(np.count_nonzero(baseline_better & region_mask) / region_pixels)
        neutral_ratio = float(np.count_nonzero(neutral & region_mask) / region_pixels)

    return {
        "data_preset": data_preset_name,
        "sample_id": sample["sample_id"],
        "region": region_name,
        "kernel_size": KERNEL_SIZE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "candidate_name": candidate_name,
        "baseline_name": baseline_name,
        "pixels": region_pixels,
        "region_ratio": float(region_pixels / total_pixels),
        "candidate_better_ratio": candidate_better_ratio,
        "baseline_better_ratio": baseline_better_ratio,
        "neutral_ratio": neutral_ratio,
        "candidate_psnr": mse_to_psnr(candidate_mse),
        "baseline_psnr": mse_to_psnr(baseline_mse),
        "psnr_delta_candidate_minus_baseline": mse_to_psnr(candidate_mse) - mse_to_psnr(baseline_mse),
        "candidate_mae": masked_mae(candidate_prediction, target, region_mask),
        "baseline_mae": masked_mae(baseline_prediction, target, region_mask),
    }


def draw_overlay_legend(
    overlay: np.ndarray,
    candidate_name: str,
    baseline_name: str,
    candidate_ratio: float,
    baseline_ratio: float,
    neutral_ratio: float,
) -> np.ndarray:
    height, width, _channels = overlay.shape
    panel_width = 330
    canvas = np.ones((height, width + panel_width, 3), dtype=np.float32)
    canvas[:, :width] = overlay
    x0 = width + 28
    cv2.putText(canvas, "Local winner", (x0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0.05, 0.05, 0.05), 2, cv2.LINE_AA)
    legend_items = (
        ((0.02, 0.38, 0.95), f"blue: {candidate_name} better"),
        ((0.95, 0.08, 0.04), f"red: {baseline_name} better"),
        ((0.55, 0.55, 0.55), "gray: small difference"),
    )
    for index, (color, text) in enumerate(legend_items):
        y = 92 + index * 45
        cv2.rectangle(canvas, (x0, y - 18), (x0 + 30, y + 12), color, -1)
        cv2.rectangle(canvas, (x0, y - 18), (x0 + 30, y + 12), (0.0, 0.0, 0.0), 1)
        cv2.putText(canvas, text, (x0 + 42, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0.05, 0.05, 0.05), 1, cv2.LINE_AA)

    text_lines = (
        f"kernel {KERNEL_SIZE}x{KERNEL_SIZE}",
        f"threshold {CONFIDENCE_THRESHOLD:.2f}",
        f"candidate {candidate_ratio * 100.0:.1f}%",
        f"baseline {baseline_ratio * 100.0:.1f}%",
        f"neutral {neutral_ratio * 100.0:.1f}%",
    )
    for index, text in enumerate(text_lines):
        cv2.putText(canvas, text, (x0, 260 + index * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0.05, 0.05, 0.05), 1, cv2.LINE_AA)

    return canvas


def build_local_winner_overlay(
    target: np.ndarray,
    candidate_better: np.ndarray,
    baseline_better: np.ndarray,
    neutral: np.ndarray,
    candidate_name: str,
    baseline_name: str,
) -> np.ndarray:
    overlay = target.copy()
    blue = np.array([0.02, 0.38, 0.95], dtype=np.float32)
    red = np.array([0.95, 0.08, 0.04], dtype=np.float32)
    gray = np.array([0.55, 0.55, 0.55], dtype=np.float32)
    overlay[candidate_better] = 0.38 * overlay[candidate_better] + 0.62 * blue
    overlay[baseline_better] = 0.38 * overlay[baseline_better] + 0.62 * red
    overlay[neutral] = 0.65 * overlay[neutral] + 0.35 * gray
    return draw_overlay_legend(
        overlay=overlay,
        candidate_name=candidate_name,
        baseline_name=baseline_name,
        candidate_ratio=float(candidate_better.mean()),
        baseline_ratio=float(baseline_better.mean()),
        neutral_ratio=float(neutral.mean()),
    )


def analyze_sample(
    data_preset_name: str,
    sample: SamplePreset,
    candidate_name: str,
    baseline_name: str,
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, object]]:
    candidate_dir = Path(sample["candidate_result_dir"])
    baseline_dir = Path(sample["baseline_result_dir"])
    target = read_rgb_image(baseline_dir / "image_gt.png")
    candidate_target = read_rgb_image(candidate_dir / "image_gt.png")
    candidate_prediction = read_rgb_image(candidate_dir / "image_pred.png")
    baseline_prediction = read_rgb_image(baseline_dir / "image_pred.png")
    validate_same_shape("candidate_target", target, candidate_target)
    validate_same_shape("candidate_prediction", target, candidate_prediction)
    validate_same_shape("baseline_prediction", target, baseline_prediction)
    if not np.allclose(target, candidate_target, atol=1.0 / 255.0):
        raise ValueError(f"Ground-truth mismatch: sample_id={sample['sample_id']}")

    region_maps = build_region_maps(sample=sample, device=device)
    candidate_better, baseline_better, neutral = build_local_winner_masks(
        target=target,
        candidate_prediction=candidate_prediction,
        baseline_prediction=baseline_prediction,
    )

    if SAVE_OVERLAY_IMAGES:
        overlay = build_local_winner_overlay(
            target=target,
            candidate_better=candidate_better,
            baseline_better=baseline_better,
            neutral=neutral,
            candidate_name=candidate_name,
            baseline_name=baseline_name,
        )
        save_rgb_image(output_dir / f"{sample['sample_id']}_winner_overlay.png", overlay)

    rows: list[dict[str, object]] = []
    for region_name, region_mask in region_maps.items():
        rows.append(
            build_region_row(
                data_preset_name=data_preset_name,
                sample=sample,
                candidate_name=candidate_name,
                baseline_name=baseline_name,
                region_name=region_name,
                region_mask=region_mask,
                target=target,
                candidate_prediction=candidate_prediction,
                baseline_prediction=baseline_prediction,
                candidate_better=candidate_better,
                baseline_better=baseline_better,
                neutral=neutral,
            )
        )

    return rows


def write_run_config(data_preset_name: str, data_preset: DataPreset, output_dir: Path) -> None:
    run_config = {
        "data_preset": data_preset_name,
        "candidate_name": data_preset["candidate_name"],
        "baseline_name": data_preset["baseline_name"],
        "kernel_size": KERNEL_SIZE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "save_overlay_images": SAVE_OVERLAY_IMAGES,
        "output_dir": str(output_dir),
        "samples": data_preset["samples"],
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


def run_analysis(data_preset_name: str) -> Path:
    data_preset = DATA_PRESETS[data_preset_name]
    output_dir = OUTPUT_ROOT / data_preset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(data_preset_name=data_preset_name, data_preset=data_preset, output_dir=output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for sample in data_preset["samples"]:
        rows.extend(
            analyze_sample(
                data_preset_name=data_preset_name,
                sample=sample,
                candidate_name=data_preset["candidate_name"],
                baseline_name=data_preset["baseline_name"],
                output_dir=output_dir,
                device=device,
            )
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "region_winner_ratios.csv", index=False)
    summary = metrics[metrics["region"] == "all"].copy()
    summary.to_csv(output_dir / "summary.csv", index=False)
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = run_analysis(data_preset_name=str(args.data_preset))
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
