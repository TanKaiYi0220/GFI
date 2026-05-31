from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from typing import TypedDict

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch

from src.data.dataset_config import get_dataset_preset
from src.data.dataset_config import iter_dataset_configs
from src.data.dataset_config import list_dataset_presets
from src.data.dataset_loader import depth_to_tensor
from src.data.dataset_loader import flow_to_tensor
from src.data.image_ops import load_backward_velocity
from src.engine.flow_approx import build_flow_init_result
from src.engine.flow_approx import flatten_target_index
from src.engine.flow_approx import make_source_grid


class SamplePreset(TypedDict):
    sample_id: str
    sample_dir: str
    dataset_root_dir: str
    record: str
    mode: str
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
                "dataset_root_dir": "",
                "record": "",
                "mode": "",
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


def load_config_defaults(argv: list[str] | None) -> dict[str, Any]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str)
    config_args, _remaining = config_parser.parse_known_args(argv)
    if config_args.config is None:
        return {}

    config_path = resolve_input_path(str(config_args.config))
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a JSON object: path={config_path}")

    config["config_path"] = str(config_path)
    return config


def config_default(config: dict[str, Any], key: str, fallback: Any) -> Any:
    return config[key] if key in config else fallback


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config = load_config_defaults(argv)
    parser = argparse.ArgumentParser(description="Compare prediction quality by splatting regions.")
    parser.add_argument("--config", default=config.get("config_path"), type=str, help="JSON config path. CLI arguments override matching config values.")
    preset_group = parser.add_mutually_exclusive_group(required=False)
    preset_group.add_argument("--data-preset", default=config.get("data_preset"), choices=tuple(sorted(DATA_PRESETS.keys())), help="Scratch-local comparison preset.")
    preset_group.add_argument("--dataset-preset", default=config.get("dataset_preset"), choices=list_dataset_presets(), help="Project dataset preset, e.g. train_minor_0507.")
    parser.add_argument("--root-dir", default=config.get("root_dir"), type=str, help="Directory containing preprocessed CSV indexes.")
    parser.add_argument("--dataset-root-dir", default=config.get("dataset_root_dir"), type=str, help="Root directory containing raw frame and velocity assets.")
    parser.add_argument("--candidate-result-root", default=config.get("candidate_result_root"), type=str, help="Inference/sample result root for the candidate model.")
    parser.add_argument("--baseline-result-root", default=config.get("baseline_result_root"), type=str, help="Inference/sample result root for the baseline model.")
    parser.add_argument("--candidate-name", default=config_default(config, "candidate_name", "candidate"), type=str)
    parser.add_argument("--baseline-name", default=config_default(config, "baseline_name", "baseline"), type=str)
    parser.add_argument("--only-fps", default=int(config_default(config, "only_fps", 60)), type=int)
    parser.add_argument("--limit", default=int(config_default(config, "limit", 0)), type=int, help="Optional max sample count after filtering. 0 means no limit.")
    parser.add_argument("--mode-filter", default=config_default(config, "mode_filter", ""), type=str, help="Optional exact mode path filter.")
    parser.add_argument(
        "--skip-missing-results",
        default=bool(config_default(config, "skip_missing_results", False)),
        action="store_true",
        help="Skip rows whose prediction artifacts are not saved.",
    )
    parser.add_argument("--result-layout", default=config_default(config, "result_layout", "auto"), choices=("auto", "inference", "training-samples"))
    parser.add_argument("--sample-split", default=config_default(config, "sample_split", "train"), choices=("train", "test"))
    parser.add_argument("--candidate-epoch", default=config_default(config, "candidate_epoch", "latest"), type=str, help="Training-samples epoch dir, e.g. latest or epoch_0061.")
    parser.add_argument("--baseline-epoch", default=config_default(config, "baseline_epoch", "latest"), type=str, help="Training-samples epoch dir, e.g. latest or epoch_0061.")
    parser.add_argument("--output-path", default=config.get("output_path"), type=str, help="Final output directory. Defaults to a temp directory grouped by preset name.")
    parser.add_argument("--kernel-size", default=int(config_default(config, "kernel_size", KERNEL_SIZE)), type=int)
    parser.add_argument("--confidence-threshold", default=float(config_default(config, "confidence_threshold", CONFIDENCE_THRESHOLD)), type=float)
    parser.add_argument("--save-overlay-images", default=bool(config_default(config, "save_overlay_images", SAVE_OVERLAY_IMAGES)), action="store_true")
    parser.add_argument("--no-save-overlay-images", dest="save_overlay_images", action="store_false")
    args = parser.parse_args(argv)
    if args.data_preset is None and args.dataset_preset is None:
        parser.error("one of --data-preset or --dataset-preset is required, either in CLI or config")
    if args.data_preset is not None and args.dataset_preset is not None:
        parser.error("only one of --data-preset or --dataset-preset can be set")
    if int(args.kernel_size) <= 0:
        parser.error("--kernel-size must be positive")
    if int(args.kernel_size) % 2 == 0:
        parser.error("--kernel-size must be odd")
    if float(args.confidence_threshold) < 0.0:
        parser.error("--confidence-threshold must be non-negative")
    return args


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


def resolve_input_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_dataset_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    missing_names = [
        name
        for name in ("root_dir", "dataset_root_dir", "candidate_result_root", "baseline_result_root")
        if getattr(args, name) is None
    ]
    if len(missing_names) > 0:
        raise ValueError(f"--dataset-preset requires these arguments: {', '.join('--' + name.replace('_', '-') for name in missing_names)}")

    return (
        resolve_input_path(str(args.root_dir)),
        resolve_input_path(str(args.dataset_root_dir)),
        resolve_input_path(str(args.candidate_result_root)),
        resolve_input_path(str(args.baseline_result_root)),
    )


def load_dataset_preset_dataframe(root_dir: Path, dataset_preset_name: str, only_fps: int) -> pd.DataFrame:
    dataset_preset = get_dataset_preset(dataset_preset_name)
    dataframe_list: list[pd.DataFrame] = []
    for dataset_config in iter_dataset_configs(dataset_preset):
        if dataset_config.fps != only_fps:
            continue

        csv_path = root_dir / f"{dataset_config.record_name}_preprocessed" / f"{dataset_config.mode_index}_raw_sequence_frame_index.csv"
        if not csv_path.is_file():
            continue

        dataframe = pd.read_csv(csv_path)
        dataframe["record"] = dataset_config.record
        dataframe["mode"] = dataset_config.mode_path
        dataframe_list.append(dataframe)

    if len(dataframe_list) == 0:
        raise RuntimeError(f"No dataset CSV found under root_dir={root_dir} for preset={dataset_preset_name} only_fps={only_fps}")

    return pd.concat(dataframe_list, ignore_index=True)


def build_frame_range(frame_0: int, frame_1: int) -> str:
    return f"frame_{frame_0:04d}_{frame_1:04d}"


def build_training_sample_key(record: str, mode: str, frame_0: int) -> str:
    mode_parts = mode.split("/")
    if len(mode_parts) < 2:
        raise ValueError(f"Mode path must contain difficulty and sequence parts: mode={mode}")

    sequence_parts = mode_parts[1].split("_")
    if len(sequence_parts) < 3:
        raise ValueError(f"Mode sequence part must look like 1_Medium_5: mode={mode}")

    record_suffix = record.replace("AnimeFantasyRPG_", "ARPG_")
    return f"{record_suffix}_{sequence_parts[0]}_{sequence_parts[1]}_{sequence_parts[2]}_{frame_0:04d}"


def resolve_epoch_dir(frame_dir: Path, epoch_name: str) -> Path:
    if epoch_name != "latest":
        return frame_dir / epoch_name

    if not frame_dir.is_dir():
        return frame_dir / "latest"

    epoch_dirs = sorted(
        (path for path in frame_dir.iterdir() if path.is_dir() and path.name.startswith("epoch_")),
        key=lambda path: path.name,
    )
    if len(epoch_dirs) == 0:
        return frame_dir / "latest"

    return epoch_dirs[-1]


def build_inference_result_dir(result_root: Path, record: str, mode: str, frame_range: str) -> Path:
    return result_root / record / mode / frame_range


def build_training_sample_result_dir(result_root: Path, split_name: str, frame_key: str, epoch_name: str) -> Path:
    return resolve_epoch_dir(result_root / "samples" / split_name / frame_key, epoch_name)


def choose_result_dir(
    result_root: Path,
    record: str,
    mode: str,
    frame_range: str,
    frame_key: str,
    split_name: str,
    epoch_name: str,
    result_layout: str,
) -> Path:
    inference_dir = build_inference_result_dir(result_root, record, mode, frame_range)
    training_dir = build_training_sample_result_dir(result_root, split_name, frame_key, epoch_name)
    if result_layout == "inference":
        return inference_dir
    if result_layout == "training-samples":
        return training_dir
    if result_files_exist_in_dir(training_dir):
        return training_dir
    return inference_dir


def build_dataset_sample(
    row: pd.Series,
    dataset_root_dir: Path,
    candidate_result_root: Path,
    baseline_result_root: Path,
    args: argparse.Namespace,
) -> SamplePreset:
    record = str(row["record"])
    mode = str(row["mode"])
    frame_0 = int(row["img0"])
    frame_t = int(row["img1"])
    frame_1 = int(row["img2"])
    frame_range = build_frame_range(frame_0, frame_1)
    frame_key = build_training_sample_key(record, mode, frame_0)
    sample_id = f"{record}_{mode.replace('/', '_')}_{frame_range}"
    candidate_result_dir = choose_result_dir(
        result_root=candidate_result_root,
        record=record,
        mode=mode,
        frame_range=frame_range,
        frame_key=frame_key,
        split_name=str(args.sample_split),
        epoch_name=str(args.candidate_epoch),
        result_layout=str(args.result_layout),
    )
    baseline_result_dir = choose_result_dir(
        result_root=baseline_result_root,
        record=record,
        mode=mode,
        frame_range=frame_range,
        frame_key=frame_key,
        split_name=str(args.sample_split),
        epoch_name=str(args.baseline_epoch),
        result_layout=str(args.result_layout),
    )
    return {
        "sample_id": sample_id,
        "sample_dir": "",
        "dataset_root_dir": str(dataset_root_dir),
        "record": record,
        "mode": mode,
        "candidate_result_dir": str(candidate_result_dir),
        "baseline_result_dir": str(baseline_result_dir),
        "frame_0": frame_0,
        "frame_t": frame_t,
        "frame_1": frame_1,
    }


def result_files_exist_in_dir(result_dir: Path) -> bool:
    return (result_dir / "image_gt.png").is_file() and (result_dir / "image_pred.png").is_file()


def result_files_exist(sample: SamplePreset) -> bool:
    candidate_dir = Path(sample["candidate_result_dir"])
    baseline_dir = Path(sample["baseline_result_dir"])
    return result_files_exist_in_dir(candidate_dir) and result_files_exist_in_dir(baseline_dir)


def build_missing_result_preview(sample: SamplePreset) -> dict[str, object]:
    candidate_dir = Path(sample["candidate_result_dir"])
    baseline_dir = Path(sample["baseline_result_dir"])
    return {
        "sample_id": sample["sample_id"],
        "candidate_result_dir": str(candidate_dir),
        "candidate_files_exist": result_files_exist_in_dir(candidate_dir),
        "baseline_result_dir": str(baseline_dir),
        "baseline_files_exist": result_files_exist_in_dir(baseline_dir),
    }


def build_samples_from_dataset_preset(args: argparse.Namespace) -> DataPreset:
    root_dir, dataset_root_dir, candidate_result_root, baseline_result_root = require_dataset_args(args)
    dataframe = load_dataset_preset_dataframe(root_dir, str(args.dataset_preset), int(args.only_fps))
    if "valid" in dataframe.columns:
        dataframe = dataframe[dataframe["valid"] == True].reset_index(drop=True)
    if str(args.mode_filter) != "":
        dataframe = dataframe[dataframe["mode"] == str(args.mode_filter)].reset_index(drop=True)

    samples: list[SamplePreset] = []
    missing_previews: list[dict[str, object]] = []
    for _index, row in dataframe.iterrows():
        sample = build_dataset_sample(row, dataset_root_dir, candidate_result_root, baseline_result_root, args)
        if bool(args.skip_missing_results) and not result_files_exist(sample):
            if len(missing_previews) < 5:
                missing_previews.append(build_missing_result_preview(sample))
            continue
        samples.append(sample)
        if int(args.limit) > 0 and len(samples) >= int(args.limit):
            break

    if len(samples) == 0:
        missing_preview_text = json.dumps(missing_previews, indent=2)
        raise RuntimeError(
            "No comparable samples found. Check result roots, mode_filter, and whether inference artifacts were saved. "
            f"candidate_result_root={candidate_result_root} baseline_result_root={baseline_result_root} "
            f"missing_preview={missing_preview_text}"
        )

    return {
        "candidate_name": str(args.candidate_name),
        "baseline_name": str(args.baseline_name),
        "samples": samples,
    }


def build_flow_path(sample: SamplePreset, mode: str, prefix: str, frame_index: int) -> Path:
    if sample["dataset_root_dir"] != "":
        return Path(sample["dataset_root_dir"]) / sample["record"] / mode / f"{prefix}_{frame_index}.exr"

    return Path(sample["sample_dir"]) / f"{prefix}_{frame_index}_fps30.exr"


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
    frame_0 = int(sample["frame_0"])
    frame_1 = int(sample["frame_1"])
    mode_30 = sample["mode"].replace("fps_60", "fps_30")
    bmv_30, source_depth1 = load_flow_and_depth_tensors(build_flow_path(sample, mode_30, "backwardVel_Depth", frame_1 // 2), device)
    fmv_30, source_depth0 = load_flow_and_depth_tensors(build_flow_path(sample, mode_30, "forwardVel_Depth", frame_0 // 2), device)
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
    kernel_size: int,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_error = np.mean((candidate_prediction - target) ** 2, axis=2)
    baseline_error = np.mean((baseline_prediction - target) ** 2, axis=2)
    candidate_local_error = cv2.blur(candidate_error, (kernel_size, kernel_size))
    baseline_local_error = cv2.blur(baseline_error, (kernel_size, kernel_size))
    confidence = np.abs(baseline_local_error - candidate_local_error) / (baseline_local_error + candidate_local_error + 1e-12)
    candidate_better = (candidate_local_error < baseline_local_error) & (confidence >= confidence_threshold)
    baseline_better = (baseline_local_error < candidate_local_error) & (confidence >= confidence_threshold)
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
    kernel_size: int,
    confidence_threshold: float,
    sample_winner: str,
    overlay_path: str,
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
        "record": sample["record"] if sample["record"] != "" else "scratch",
        "mode": sample["mode"],
        "sample_winner": sample_winner,
        "overlay_path": overlay_path,
        "region": region_name,
        "kernel_size": kernel_size,
        "confidence_threshold": confidence_threshold,
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
    kernel_size: int,
    confidence_threshold: float,
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
        f"kernel {kernel_size}x{kernel_size}",
        f"threshold {confidence_threshold:.2f}",
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
    kernel_size: int,
    confidence_threshold: float,
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
        kernel_size=kernel_size,
        confidence_threshold=confidence_threshold,
    )


def choose_sample_winner(target: np.ndarray, candidate_prediction: np.ndarray, baseline_prediction: np.ndarray) -> str:
    full_mask = np.ones(target.shape[:2], dtype=bool)
    candidate_psnr = mse_to_psnr(masked_mse(candidate_prediction, target, full_mask))
    baseline_psnr = mse_to_psnr(masked_mse(baseline_prediction, target, full_mask))
    if np.isclose(candidate_psnr, baseline_psnr, atol=1e-6):
        return "neutral"
    if candidate_psnr > baseline_psnr:
        return "candidate_win"
    return "baseline_win"


def build_sample_output_dir(output_dir: Path, sample: SamplePreset, sample_winner: str) -> Path:
    record_name = sample["record"] if sample["record"] != "" else "scratch"
    return output_dir / record_name / sample_winner


def analyze_sample(
    data_preset_name: str,
    sample: SamplePreset,
    candidate_name: str,
    baseline_name: str,
    output_dir: Path,
    device: torch.device,
    kernel_size: int,
    confidence_threshold: float,
    save_overlay_images: bool,
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
        kernel_size=kernel_size,
        confidence_threshold=confidence_threshold,
    )

    sample_winner = choose_sample_winner(target=target, candidate_prediction=candidate_prediction, baseline_prediction=baseline_prediction)
    sample_output_dir = build_sample_output_dir(output_dir=output_dir, sample=sample, sample_winner=sample_winner)
    overlay_path = ""
    if save_overlay_images:
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = str(sample_output_dir / f"{sample['sample_id']}_winner_overlay.png")
        overlay = build_local_winner_overlay(
            target=target,
            candidate_better=candidate_better,
            baseline_better=baseline_better,
            neutral=neutral,
            candidate_name=candidate_name,
            baseline_name=baseline_name,
            kernel_size=kernel_size,
            confidence_threshold=confidence_threshold,
        )
        save_rgb_image(Path(overlay_path), overlay)

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
                kernel_size=kernel_size,
                confidence_threshold=confidence_threshold,
                sample_winner=sample_winner,
                overlay_path=overlay_path,
            )
        )

    return rows


def build_effective_config(args: argparse.Namespace, data_preset_name: str, output_dir: Path) -> dict[str, object]:
    return {
        "config": str(args.config) if args.config is not None else "",
        "data_preset": str(args.data_preset) if args.data_preset is not None else "",
        "dataset_preset": str(args.dataset_preset) if args.dataset_preset is not None else "",
        "root_dir": str(args.root_dir) if args.root_dir is not None else "",
        "dataset_root_dir": str(args.dataset_root_dir) if args.dataset_root_dir is not None else "",
        "candidate_result_root": str(args.candidate_result_root) if args.candidate_result_root is not None else "",
        "baseline_result_root": str(args.baseline_result_root) if args.baseline_result_root is not None else "",
        "candidate_name": str(args.candidate_name),
        "baseline_name": str(args.baseline_name),
        "only_fps": int(args.only_fps),
        "limit": int(args.limit),
        "mode_filter": str(args.mode_filter),
        "skip_missing_results": bool(args.skip_missing_results),
        "result_layout": str(args.result_layout),
        "sample_split": str(args.sample_split),
        "candidate_epoch": str(args.candidate_epoch),
        "baseline_epoch": str(args.baseline_epoch),
        "output_path": str(output_dir),
        "kernel_size": int(args.kernel_size),
        "confidence_threshold": float(args.confidence_threshold),
        "save_overlay_images": bool(args.save_overlay_images),
        "resolved_data_preset_name": data_preset_name,
    }


def write_run_config(data_preset_name: str, data_preset: DataPreset, output_dir: Path, args: argparse.Namespace) -> None:
    effective_config = build_effective_config(args=args, data_preset_name=data_preset_name, output_dir=output_dir)
    run_config = {
        "config": effective_config,
        "data_preset": data_preset_name,
        "candidate_name": data_preset["candidate_name"],
        "baseline_name": data_preset["baseline_name"],
        "kernel_size": int(args.kernel_size),
        "confidence_threshold": float(args.confidence_threshold),
        "save_overlay_images": bool(args.save_overlay_images),
        "output_dir": str(output_dir),
        "samples": data_preset["samples"],
    }
    (output_dir / "analysis_config.json").write_text(json.dumps(effective_config, indent=2), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


def resolve_output_dir(args: argparse.Namespace, data_preset_name: str) -> Path:
    if args.output_path is None or str(args.output_path) == "":
        return OUTPUT_ROOT / data_preset_name

    return resolve_input_path(str(args.output_path))


def write_record_metrics(output_dir: Path, metrics: pd.DataFrame) -> None:
    for record_name, record_metrics in metrics.groupby("record", sort=False):
        record_dir = output_dir / str(record_name)
        record_dir.mkdir(parents=True, exist_ok=True)
        record_metrics.to_csv(record_dir / "region_winner_ratios.csv", index=False)
        record_summary = record_metrics[record_metrics["region"] == "all"].copy()
        record_summary.to_csv(record_dir / "summary.csv", index=False)


def run_analysis(data_preset_name: str, data_preset: DataPreset, args: argparse.Namespace) -> Path:
    output_dir = resolve_output_dir(args=args, data_preset_name=data_preset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(data_preset_name=data_preset_name, data_preset=data_preset, output_dir=output_dir, args=args)
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
                kernel_size=int(args.kernel_size),
                confidence_threshold=float(args.confidence_threshold),
                save_overlay_images=bool(args.save_overlay_images),
            )
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "region_winner_ratios.csv", index=False)
    summary = metrics[metrics["region"] == "all"].copy()
    summary.to_csv(output_dir / "summary.csv", index=False)
    write_record_metrics(output_dir=output_dir, metrics=metrics)
    return output_dir


def main() -> None:
    args = parse_args()
    if args.data_preset is not None:
        data_preset_name = str(args.data_preset)
        data_preset = DATA_PRESETS[data_preset_name]
    else:
        data_preset_name = str(args.dataset_preset)
        data_preset = build_samples_from_dataset_preset(args)

    output_dir = run_analysis(data_preset_name=data_preset_name, data_preset=data_preset, args=args)
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
