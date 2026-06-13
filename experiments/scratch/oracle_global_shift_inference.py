from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inference import BASELINE_MODEL_NAME
from scripts.inference import RESIDUAL_MODEL_NAME
from scripts.inference import run_inference_batch
from scripts.train import build_merged_dataframe
from scripts.train import read_model_init_args
from scripts.train import resolve_model_class
from scripts.train import set_seed
from src.data.dataset_loader import FlowEstimationTrainDataset
from src.data.dataset_loader import VFITrainDataset
from src.engine.flow_approx import SPLATTING_FLOW_APPROX_METHODS
from src.utils.config import load_yaml_file
from src.utils.logger import build_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a GT-oracle global-shift diagnostic.",
    )
    parser.add_argument("--config", required=True, type=str, help="Oracle-shift experiment config path.")
    parser.add_argument("--project-root", type=str, help="Optional project root override.")
    parser.add_argument("--inference-config", type=str, help="Optional referenced inference config override.")
    parser.add_argument("--output-dir", type=str, help="Optional output directory override.")
    parser.add_argument("--inference-preset", type=str, help="Optional dataset preset override.")
    parser.add_argument("--merged-csv", type=str, help="Optional merged preset CSV override.")
    parser.add_argument("--dataset-root-dir", type=str, help="Optional dataset asset root override.")
    parser.add_argument("--checkpoint-path", type=str, help="Optional checkpoint path override.")
    parser.add_argument("--flow-approx-method", type=str, help="Optional flow approximation method override.")
    parser.add_argument("--max-shift", type=int, help="Optional shift search radius override.")
    parser.add_argument("--improvement-threshold-db", type=float, help="Optional selected-case threshold override.")
    parser.add_argument("--save-top-k-improved", type=int, help="Optional top-k selected improved case count override.")
    parser.add_argument("--limit", type=int, help="Optional sample limit override. 0 means no limit.")
    parser.add_argument("--batch-size", type=int, help="Optional batch size override.")
    parser.add_argument("--record-filter", type=str, help="Optional exact record filter override.")
    parser.add_argument("--mode-filter", type=str, help="Optional exact mode filter override.")
    parser.add_argument("--save-heatmap-all-selected", action="store_true", help="Save shift heatmaps for selected cases.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit before loading the model.")
    return parser.parse_args()


def override_config(config: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    if value is not None:
        updated = dict(config)
        updated[key] = value
        return updated
    return config


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing oracle-shift config: {config_path}")
    config = load_yaml_file(config_path)
    if not isinstance(config, dict):
        raise TypeError(f"Oracle-shift config must be a JSON object: path={config_path}")
    return config


def require_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"Config key must be a non-empty string: key={key}, value={value}")
    return value


def optional_string(config: dict[str, Any], key: str, fallback: str) -> str:
    value = config.get(key, fallback)
    if not isinstance(value, str):
        raise TypeError(f"Config key must be a string: key={key}, value={value}")
    return value


def optional_int(config: dict[str, Any], key: str, fallback: int) -> int:
    value = config.get(key, fallback)
    if not isinstance(value, int):
        raise TypeError(f"Config key must be an int: key={key}, value={value}")
    return value


def optional_float(config: dict[str, Any], key: str, fallback: float) -> float:
    value = config.get(key, fallback)
    if not isinstance(value, (float, int)):
        raise TypeError(f"Config key must be numeric: key={key}, value={value}")
    return float(value)


def optional_bool(config: dict[str, Any], key: str, fallback: bool) -> bool:
    value = config.get(key, fallback)
    if not isinstance(value, bool):
        raise TypeError(f"Config key must be bool: key={key}, value={value}")
    return value


def expand_path_text(path_text: str) -> str:
    return os.path.expandvars(path_text).replace("\\", "/")


def resolve_path(path_text: str, project_root: Path) -> Path:
    expanded = expand_path_text(path_text)
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def resolve_optional_path(path_text: str, project_root: Path) -> Path | None:
    if path_text == "":
        return None
    return resolve_path(path_text, project_root)


def read_inference_presets(config: dict[str, Any]) -> list[str]:
    if "inference_presets" in config:
        value = config["inference_presets"]
    elif "inference_preset" in config:
        value = config["inference_preset"]
    else:
        raise KeyError("Inference config must contain inference_preset or inference_presets.")

    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if len(value) == 0:
            raise ValueError("inference_presets must contain at least one preset name.")
        return value
    raise TypeError("inference_preset must be a string or inference_presets must be a list of strings.")


def load_referenced_inference_config(experiment_config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    inference_config_path = resolve_path(require_string(experiment_config, "inference_config"), project_root)
    inference_config = load_yaml_file(inference_config_path)
    if not isinstance(inference_config, dict):
        raise TypeError(f"Inference config must be a JSON object: path={inference_config_path}")
    inference_config["config_path"] = str(inference_config_path)
    inference_config["inference_preset"] = [require_string(experiment_config, "inference_preset")]

    dataset_root_dir = optional_string(experiment_config, "dataset_root_dir", "")
    checkpoint_path = optional_string(experiment_config, "checkpoint_path", "")
    flow_approx_method = optional_string(experiment_config, "flow_approx_method", "")
    batch_size = optional_int(experiment_config, "batch_size", int(inference_config.get("batch_size", 1)))

    if dataset_root_dir != "":
        inference_config["dataset_root_dir"] = dataset_root_dir
    if checkpoint_path != "":
        inference_config["checkpoint_path"] = checkpoint_path
    if flow_approx_method != "":
        inference_config["flow_approx_method"] = flow_approx_method
    inference_config["batch_size"] = batch_size
    return inference_config


def tensor_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((prediction - target) ** 2).clamp_min(1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).detach().cpu().item())


def centered_crop(tensor: torch.Tensor, margin: int) -> torch.Tensor:
    if margin == 0:
        return tensor
    return tensor[:, margin:-margin, margin:-margin]


def shifted_crop(tensor: torch.Tensor, dx: int, dy: int, margin: int) -> torch.Tensor:
    height = int(tensor.shape[1])
    width = int(tensor.shape[2])
    return tensor[:, margin + dy : height - margin + dy, margin + dx : width - margin + dx]


def compute_oracle_shift_grid(prediction: torch.Tensor, target: torch.Tensor, max_shift: int) -> dict[str, Any]:
    target_crop = centered_crop(target, max_shift)
    original_crop = centered_crop(prediction, max_shift)
    original_psnr = tensor_psnr(original_crop, target_crop)
    mse_grid = torch.empty((2 * max_shift + 1, 2 * max_shift + 1), dtype=torch.float32, device=prediction.device)

    for y_index, dy in enumerate(range(-max_shift, max_shift + 1)):
        for x_index, dx in enumerate(range(-max_shift, max_shift + 1)):
            candidate_crop = shifted_crop(prediction, dx, dy, max_shift)
            mse_grid[y_index, x_index] = torch.mean((candidate_crop - target_crop) ** 2)

    psnr_grid = 10.0 * torch.log10(1.0 / mse_grid.clamp_min(1e-12))
    flat_best_index = int(torch.argmax(psnr_grid).detach().cpu().item())
    best_y, best_x = np.unravel_index(flat_best_index, tuple(psnr_grid.shape))
    best_dx = int(best_x - max_shift)
    best_dy = int(best_y - max_shift)
    best_psnr = float(psnr_grid[best_y, best_x].detach().cpu().item())

    return {
        "grid": psnr_grid.detach().cpu().numpy(),
        "original_crop_psnr": original_psnr,
        "best_shift_psnr": best_psnr,
        "best_shift_gain": best_psnr - original_psnr,
        "best_shift_dx": best_dx,
        "best_shift_dy": best_dy,
        "best_shift_magnitude": float((best_dx * best_dx + best_dy * best_dy) ** 0.5),
    }


def apply_shift_for_display(prediction: torch.Tensor, dx: int, dy: int, margin: int) -> torch.Tensor:
    shifted = torch.full_like(prediction, 0.5)
    height = int(prediction.shape[1])
    width = int(prediction.shape[2])
    shifted[:, margin : height - margin, margin : width - margin] = shifted_crop(prediction, dx, dy, margin)
    return shifted


def tensor_to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.round(array * 255.0).astype(np.uint8)


def save_rgb(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_rgb(tensor), mode="RGB").save(path)


def save_error_map(path: Path, prediction: torch.Tensor, target: torch.Tensor) -> None:
    error = torch.mean(torch.abs(prediction.detach().cpu() - target.detach().cpu()), dim=0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    image = ax.imshow(error, cmap="magma", vmin=0.0, vmax=0.3)
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_shifted_error_map(path: Path, prediction: torch.Tensor, target: torch.Tensor, dx: int, dy: int, margin: int) -> None:
    target_crop = centered_crop(target, margin)
    prediction_crop = shifted_crop(prediction, dx, dy, margin)
    error_crop = torch.mean(torch.abs(prediction_crop.detach().cpu() - target_crop.detach().cpu()), dim=0).numpy()
    error = np.zeros(tuple(target.shape[1:]), dtype=np.float32)
    if margin == 0:
        error[:, :] = error_crop
    else:
        error[margin:-margin, margin:-margin] = error_crop
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    image = ax.imshow(error, cmap="magma", vmin=0.0, vmax=0.3)
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_shift_heatmap(path: Path, grid: np.ndarray, original_crop_psnr: float, max_shift: int) -> None:
    gain_grid = grid - original_crop_psnr
    max_abs_gain = max(float(np.max(np.abs(gain_grid))), 0.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(gain_grid, cmap="RdYlGn", origin="lower", vmin=-max_abs_gain, vmax=max_abs_gain)
    tick_step = max(1, max_shift // 2)
    ticks = list(range(0, 2 * max_shift + 1, tick_step))
    tick_labels = [str(value - max_shift) for value in ticks]
    best_y, best_x = np.unravel_index(int(np.argmax(grid)), grid.shape)
    ax.scatter([best_x], [best_y], c="black", marker="x", s=80)
    ax.set_xticks(ticks, tick_labels)
    ax.set_yticks(ticks, tick_labels)
    ax.set_xlabel("dx")
    ax.set_ylabel("dy")
    ax.set_title("Oracle shift PSNR gain vs original")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="gain (dB)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_sample_key(inference_preset: str, record: str, mode: str, frame_range: str) -> str:
    return f"{inference_preset}|{record}|{mode}|{frame_range}"


def build_base_row(row: pd.Series, inference_preset: str, sample_index: int) -> dict[str, object]:
    record = str(row["record"])
    mode = str(row["mode"])
    frame_range = f"frame_{int(row['img0']):04d}_{int(row['img2']):04d}"
    return {
        "sample_index": sample_index,
        "sample_key": build_sample_key(inference_preset, record, mode, frame_range),
        "inference_preset": inference_preset,
        "record": record,
        "mode": mode,
        "record_name": f"{record}_{mode}",
        "frame_range": frame_range,
        "valid": bool(row["valid"]) if "valid" in row.index else True,
        "distance_index_mean": float(row["D_index Mean"]) if "D_index Mean" in row.index else -1.0,
        "distance_index_median": float(row["D_index Median"]) if "D_index Median" in row.index else -1.0,
    }


def build_dataframe(
    inference_config: dict[str, Any],
    experiment_config: dict[str, Any],
    project_root: Path,
    output_dir: Path,
) -> pd.DataFrame:
    logger = build_logger("oracle_global_shift_inference")
    inference_preset = require_string(experiment_config, "inference_preset")
    available_presets = read_inference_presets(inference_config)
    if inference_preset not in available_presets:
        raise ValueError(f"Requested inference_preset={inference_preset} is not listed in inference config presets={available_presets}")

    merged_csv = optional_string(experiment_config, "merged_csv", "")
    if merged_csv != "":
        merged_csv_path = resolve_path(merged_csv, project_root)
        if not merged_csv_path.exists():
            raise FileNotFoundError(f"Missing merged CSV: {merged_csv_path}")
        dataframe = pd.read_csv(merged_csv_path)
    else:
        root_dir = resolve_path(str(inference_config["root_dir"]), project_root)
        only_fps = int(inference_config.get("only_fps", 60))
        dataframe = build_merged_dataframe(root_dir, output_dir, inference_preset, only_fps, logger)

    dataframe = dataframe.copy()
    dataframe["inference_preset"] = inference_preset
    if "valid" in dataframe.columns:
        dataframe = dataframe[dataframe["valid"] == True].reset_index(drop=True)

    record_filter = optional_string(experiment_config, "record_filter", "")
    mode_filter = optional_string(experiment_config, "mode_filter", "")
    limit = optional_int(experiment_config, "limit", 0)
    if record_filter != "":
        dataframe = dataframe[dataframe["record"].astype(str) == record_filter].reset_index(drop=True)
    if mode_filter != "":
        dataframe = dataframe[dataframe["mode"].astype(str) == mode_filter].reset_index(drop=True)
    if limit > 0:
        dataframe = dataframe.head(limit).reset_index(drop=True)
    if len(dataframe) == 0:
        raise ValueError("No samples left after filtering.")
    return dataframe


def load_model(inference_config: dict[str, Any], project_root: Path, device: torch.device) -> Any:
    model_name = str(inference_config["model_name"])
    model_init_args = read_model_init_args(inference_config)
    model_class = resolve_model_class(model_name)
    model = model_class(**model_init_args).to(device)
    checkpoint_path = resolve_path(str(inference_config["checkpoint_path"]), project_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def make_dataset(group_dataframe: pd.DataFrame, inference_config: dict[str, Any], project_root: Path) -> Any:
    model_name = str(inference_config["model_name"])
    flow_approx_method = str(inference_config.get("flow_approx_method", "none"))
    dataset_root_dir = resolve_path(str(inference_config["dataset_root_dir"]), project_root)
    if not dataset_root_dir.exists():
        raise FileNotFoundError(f"Missing dataset_root_dir: {dataset_root_dir}")
    input_fps = int(inference_config.get("input_fps", 30))
    if model_name in (BASELINE_MODEL_NAME, RESIDUAL_MODEL_NAME):
        return VFITrainDataset(group_dataframe, str(dataset_root_dir), False, input_fps)
    include_source_depths = flow_approx_method in SPLATTING_FLOW_APPROX_METHODS
    return FlowEstimationTrainDataset(group_dataframe, str(dataset_root_dir), input_fps, False, include_source_depths)


def run_metrics_pass(
    experiment_config: dict[str, Any],
    inference_config: dict[str, Any],
    dataframe: pd.DataFrame,
    model: Any,
    project_root: Path,
    device: torch.device,
) -> pd.DataFrame:
    model_name = str(inference_config["model_name"])
    flow_approx_method = str(inference_config.get("flow_approx_method", "none"))
    scale_factor = float(inference_config.get("scale_factor", 1.0))
    batch_size = optional_int(experiment_config, "batch_size", int(inference_config.get("batch_size", 1)))
    max_shift = optional_int(experiment_config, "max_shift", 6)

    rows: list[dict[str, object]] = []
    sample_index = 0
    with torch.no_grad():
        for (inference_preset, record, mode), group_dataframe in dataframe.groupby(["inference_preset", "record", "mode"], sort=False):
            group_dataframe = group_dataframe.reset_index(drop=True)
            dataset = make_dataset(group_dataframe, inference_config, project_root)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            sample_offset = 0
            progress = tqdm(loader, desc=f"oracle_metrics_{inference_preset}_{record}_{mode}", leave=True)
            for batch in progress:
                inference_result = run_inference_batch(batch, device, flow_approx_method, model, model_name, scale_factor)
                target_batch = inference_result["imgt"]
                prediction_batch = inference_result["imgt_pred"]
                for batch_index in range(int(prediction_batch.shape[0])):
                    row = group_dataframe.iloc[sample_offset + batch_index]
                    target = target_batch[batch_index]
                    prediction = prediction_batch[batch_index].clamp(0.0, 1.0)
                    shift_result = compute_oracle_shift_grid(prediction, target, max_shift)
                    base_row = build_base_row(row, str(inference_preset), sample_index)
                    base_row.update(
                        {
                            "full_psnr_before_db": tensor_psnr(prediction, target),
                            "crop_psnr_before_db": float(shift_result["original_crop_psnr"]),
                            "crop_psnr_after_oracle_shift_db": float(shift_result["best_shift_psnr"]),
                            "oracle_shift_gain_db": float(shift_result["best_shift_gain"]),
                            "best_shift_dx": int(shift_result["best_shift_dx"]),
                            "best_shift_dy": int(shift_result["best_shift_dy"]),
                            "best_shift_magnitude_px": float(shift_result["best_shift_magnitude"]),
                            "selected_for_save": False,
                            "save_reason": "",
                        }
                    )
                    rows.append(base_row)
                    sample_index += 1
                sample_offset += int(prediction_batch.shape[0])
    return pd.DataFrame(rows)


def select_cases_for_saving(metrics: pd.DataFrame, experiment_config: dict[str, Any]) -> set[str]:
    save_top_k_improved = optional_int(experiment_config, "save_top_k_improved", 0)
    improvement_threshold_db = optional_float(experiment_config, "improvement_threshold_db", 0.5)
    if save_top_k_improved <= 0:
        return set()
    candidates = metrics[metrics["oracle_shift_gain_db"] >= improvement_threshold_db].copy()
    candidates = candidates.sort_values("oracle_shift_gain_db", ascending=False).head(save_top_k_improved)
    return set(candidates["sample_key"].astype(str).tolist())


def save_case_artifacts(
    case_dir: Path,
    inference_result: dict[str, Any],
    shift_result: dict[str, Any],
    row_metrics: dict[str, object],
    max_shift: int,
    save_heatmap: bool,
) -> dict[str, str]:
    img0 = inference_result["img0"][0]
    img1 = inference_result["img1"][0]
    target = inference_result["imgt"][0]
    prediction = inference_result["imgt_pred"][0].clamp(0.0, 1.0)
    shifted_prediction = apply_shift_for_display(
        prediction,
        int(shift_result["best_shift_dx"]),
        int(shift_result["best_shift_dy"]),
        max_shift,
    )
    paths: dict[str, Path] = {
        "image_0_path": case_dir / "image_0.png",
        "image_1_path": case_dir / "image_1.png",
        "image_gt_path": case_dir / "image_gt.png",
        "image_pred_before_path": case_dir / "image_pred_before.png",
        "image_pred_after_oracle_shift_path": case_dir / "image_pred_after_oracle_shift.png",
        "error_before_path": case_dir / "error_before.png",
        "error_after_oracle_shift_path": case_dir / "error_after_oracle_shift.png",
        "oracle_shift_heatmap_path": case_dir / "oracle_shift_gain_heatmap.png",
        "oracle_shift_grid_csv_path": case_dir / "oracle_shift_psnr_grid.csv",
        "case_metrics_path": case_dir / "case_metrics.json",
    }
    save_rgb(paths["image_0_path"], img0)
    save_rgb(paths["image_1_path"], img1)
    save_rgb(paths["image_gt_path"], target)
    save_rgb(paths["image_pred_before_path"], prediction)
    save_rgb(paths["image_pred_after_oracle_shift_path"], shifted_prediction)
    save_error_map(paths["error_before_path"], prediction, target)
    save_shifted_error_map(
        paths["error_after_oracle_shift_path"],
        prediction,
        target,
        int(shift_result["best_shift_dx"]),
        int(shift_result["best_shift_dy"]),
        max_shift,
    )
    np.savetxt(paths["oracle_shift_grid_csv_path"], shift_result["grid"], delimiter=",", fmt="%.8f")
    if save_heatmap:
        save_shift_heatmap(paths["oracle_shift_heatmap_path"], shift_result["grid"], float(shift_result["original_crop_psnr"]), max_shift)
    case_dir.mkdir(parents=True, exist_ok=True)
    with paths["case_metrics_path"].open("w", encoding="utf-8") as handle:
        json.dump(row_metrics, handle, indent=2)
    return {name: str(path) for name, path in paths.items()}


def save_selected_cases(
    experiment_config: dict[str, Any],
    inference_config: dict[str, Any],
    dataframe: pd.DataFrame,
    model: Any,
    metrics: pd.DataFrame,
    selected_keys: set[str],
    project_root: Path,
    output_dir: Path,
    device: torch.device,
) -> pd.DataFrame:
    if len(selected_keys) == 0:
        return metrics

    model_name = str(inference_config["model_name"])
    flow_approx_method = str(inference_config.get("flow_approx_method", "none"))
    scale_factor = float(inference_config.get("scale_factor", 1.0))
    max_shift = optional_int(experiment_config, "max_shift", 6)
    save_heatmap = optional_bool(experiment_config, "save_heatmap_all_selected", True)
    metrics = metrics.copy()
    metrics_index_by_key = {str(row.sample_key): index for index, row in metrics.iterrows()}

    with torch.no_grad():
        for (inference_preset, record, mode), group_dataframe in dataframe.groupby(["inference_preset", "record", "mode"], sort=False):
            group_dataframe = group_dataframe.reset_index(drop=True)
            row_keys = [
                build_sample_key(str(inference_preset), str(row["record"]), str(row["mode"]), f"frame_{int(row['img0']):04d}_{int(row['img2']):04d}")
                for _, row in group_dataframe.iterrows()
            ]
            selected_indices = [index for index, sample_key in enumerate(row_keys) if sample_key in selected_keys]
            if len(selected_indices) == 0:
                continue

            dataset = make_dataset(group_dataframe, inference_config, project_root)
            selected_dataset = Subset(dataset, selected_indices)
            loader = DataLoader(selected_dataset, batch_size=1, shuffle=False)
            progress = tqdm(loader, desc=f"oracle_save_{inference_preset}_{record}_{mode}", leave=True)
            for selected_loader_index, batch in enumerate(progress):
                original_index = selected_indices[selected_loader_index]
                row = group_dataframe.iloc[original_index]
                frame_range = f"frame_{int(row['img0']):04d}_{int(row['img2']):04d}"
                sample_key = build_sample_key(str(inference_preset), str(row["record"]), str(row["mode"]), frame_range)
                inference_result = run_inference_batch(batch, device, flow_approx_method, model, model_name, scale_factor)
                prediction = inference_result["imgt_pred"][0].clamp(0.0, 1.0)
                target = inference_result["imgt"][0]
                shift_result = compute_oracle_shift_grid(prediction, target, max_shift)
                metrics_row_index = metrics_index_by_key[sample_key]
                row_metrics = metrics.loc[metrics_row_index].to_dict()
                case_dir = output_dir / "selected_cases" / str(inference_preset) / str(row["record"]) / str(row["mode"]) / frame_range
                saved_paths = save_case_artifacts(case_dir, inference_result, shift_result, row_metrics, max_shift, save_heatmap)
                for path_name, path_value in saved_paths.items():
                    metrics.loc[metrics_row_index, path_name] = path_value
                metrics.loc[metrics_row_index, "selected_for_save"] = True
                metrics.loc[metrics_row_index, "save_reason"] = "top_oracle_shift_gain"
    return metrics


def summarize_metrics(metrics: pd.DataFrame, experiment_config: dict[str, Any]) -> pd.DataFrame:
    improvement_threshold_db = optional_float(experiment_config, "improvement_threshold_db", 0.5)
    improved = metrics[metrics["oracle_shift_gain_db"] > 0.0]
    threshold_improved = metrics[metrics["oracle_shift_gain_db"] >= improvement_threshold_db]
    return pd.DataFrame(
        [
            {
                "sample_count": len(metrics),
                "mean_full_psnr_before_db": float(metrics["full_psnr_before_db"].mean()),
                "mean_crop_psnr_before_db": float(metrics["crop_psnr_before_db"].mean()),
                "mean_crop_psnr_after_oracle_shift_db": float(metrics["crop_psnr_after_oracle_shift_db"].mean()),
                "mean_oracle_shift_gain_db": float(metrics["oracle_shift_gain_db"].mean()),
                "median_oracle_shift_gain_db": float(metrics["oracle_shift_gain_db"].median()),
                "improved_case_count": int(len(improved)),
                "improved_case_rate": float(len(improved) / len(metrics)),
                "threshold_improved_case_count": int(len(threshold_improved)),
                "threshold_improved_case_rate": float(len(threshold_improved) / len(metrics)),
                "mean_best_shift_magnitude_px": float(metrics["best_shift_magnitude_px"].mean()),
                "nonzero_shift_case_rate": float((metrics["best_shift_magnitude_px"] > 0.0).mean()),
            }
        ]
    )


def summarize_records(metrics: pd.DataFrame, experiment_config: dict[str, Any]) -> pd.DataFrame:
    improvement_threshold_db = optional_float(experiment_config, "improvement_threshold_db", 0.5)
    rows: list[dict[str, object]] = []
    for (inference_preset, record, mode), group in metrics.groupby(["inference_preset", "record", "mode"], sort=False):
        rows.append(
            {
                "inference_preset": inference_preset,
                "record": record,
                "mode": mode,
                "sample_count": len(group),
                "mean_full_psnr_before_db": float(group["full_psnr_before_db"].mean()),
                "mean_crop_psnr_before_db": float(group["crop_psnr_before_db"].mean()),
                "mean_crop_psnr_after_oracle_shift_db": float(group["crop_psnr_after_oracle_shift_db"].mean()),
                "mean_oracle_shift_gain_db": float(group["oracle_shift_gain_db"].mean()),
                "median_oracle_shift_gain_db": float(group["oracle_shift_gain_db"].median()),
                "improved_case_rate": float((group["oracle_shift_gain_db"] > 0.0).mean()),
                "threshold_improved_case_rate": float((group["oracle_shift_gain_db"] >= improvement_threshold_db).mean()),
                "mean_best_shift_magnitude_px": float(group["best_shift_magnitude_px"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_output_dir(experiment_config: dict[str, Any], project_root: Path) -> Path:
    output_dir_text = optional_string(experiment_config, "output_dir", "")
    if output_dir_text != "":
        return resolve_path(output_dir_text, project_root)
    return Path(tempfile.gettempdir()) / "GFI_oracle_shift_inference" / require_string(experiment_config, "inference_preset")


def build_project_root(experiment_config: dict[str, Any], cli_project_root: str | None) -> Path:
    if cli_project_root is not None:
        return Path(expand_path_text(cli_project_root)).expanduser().resolve()
    project_root_text = optional_string(experiment_config, "project_root", "")
    if project_root_text != "":
        return Path(expand_path_text(project_root_text)).expanduser().resolve()
    return PROJECT_ROOT.resolve()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    for key in (
        "inference_config",
        "output_dir",
        "inference_preset",
        "merged_csv",
        "dataset_root_dir",
        "checkpoint_path",
        "flow_approx_method",
        "max_shift",
        "improvement_threshold_db",
        "save_top_k_improved",
        "limit",
        "batch_size",
        "record_filter",
        "mode_filter",
    ):
        updated = override_config(updated, key, getattr(args, key))
    if args.save_heatmap_all_selected:
        updated["save_heatmap_all_selected"] = True
    if args.dry_run:
        updated["dry_run"] = True
    return updated


def main() -> None:
    args = parse_args()
    raw_config = load_experiment_config(Path(args.config))
    experiment_config = apply_cli_overrides(raw_config, args)
    project_root = build_project_root(experiment_config, args.project_root)
    inference_config = load_referenced_inference_config(experiment_config, project_root)
    output_dir = build_output_dir(experiment_config, project_root)
    dry_run = optional_bool(experiment_config, "dry_run", False)

    run_summary = {
        "script_kind": "GT-oracle global shift diagnostic. Not deployable inference.",
        "config_path": str(Path(args.config).resolve()),
        "project_root": str(project_root),
        "inference_config": inference_config["config_path"],
        "model_name": inference_config["model_name"],
        "checkpoint_path": str(resolve_path(str(inference_config["checkpoint_path"]), project_root)),
        "dataset_root_dir": str(resolve_path(str(inference_config["dataset_root_dir"]), project_root)),
        "flow_approx_method": str(inference_config.get("flow_approx_method", "")),
        "inference_preset": require_string(experiment_config, "inference_preset"),
        "merged_csv": optional_string(experiment_config, "merged_csv", ""),
        "max_shift": optional_int(experiment_config, "max_shift", 6),
        "improvement_threshold_db": optional_float(experiment_config, "improvement_threshold_db", 0.5),
        "save_top_k_improved": optional_int(experiment_config, "save_top_k_improved", 0),
        "limit": optional_int(experiment_config, "limit", 0),
        "output_dir": str(output_dir),
    }
    if dry_run:
        print(json.dumps(run_summary, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    set_seed(int(inference_config.get("seed", 1234)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataframe = build_dataframe(inference_config, experiment_config, project_root, output_dir)
    model = load_model(inference_config, project_root, device)
    metrics = run_metrics_pass(experiment_config, inference_config, dataframe, model, project_root, device)
    selected_keys = select_cases_for_saving(metrics, experiment_config)
    metrics = save_selected_cases(experiment_config, inference_config, dataframe, model, metrics, selected_keys, project_root, output_dir, device)

    summary = summarize_metrics(metrics, experiment_config)
    record_summary = summarize_records(metrics, experiment_config)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    record_summary.to_csv(output_dir / "record_metrics.csv", index=False)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"metrics={output_dir / 'metrics.csv'}")
    print(f"summary={output_dir / 'summary.csv'}")
    print(f"record_metrics={output_dir / 'record_metrics.csv'}")
    print(f"selected_cases={output_dir / 'selected_cases'}")


if __name__ == "__main__":
    main()
