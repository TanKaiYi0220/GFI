from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any
from typing import TypedDict

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ProfileRow(TypedDict):
    name: str
    y: int
    x_start: int
    x_end: int
    color_bgr: tuple[int, int, int]


class ProfileRenderConfig(TypedDict):
    rows: list[ProfileRow]
    line_thickness: int
    time_scale: int
    profile_width: int
    source_scale: float
    include_invalid: bool


class FrameSequence(TypedDict):
    gt_frames: dict[int, Any]
    pred_frames: dict[int, Any]
    interpolated_frames: list[int]


DEFAULT_LINE_THICKNESS: int = 5
DEFAULT_TIME_SCALE: int = 14
DEFAULT_SOURCE_SCALE: float = 0.28
DEFAULT_PROFILE_WIDTH: int = 980
BACKGROUND_BGR: tuple[int, int, int] = (255, 255, 255)
TEXT_BGR: tuple[int, int, int] = (32, 32, 32)
MUTED_TEXT_BGR: tuple[int, int, int] = (96, 96, 96)
BLUE_TEXT_BGR: tuple[int, int, int] = (190, 88, 20)
GRID_BGR: tuple[int, int, int] = (230, 230, 230)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for one frame range and export row-wise temporal profile visualizations.",
    )
    parser.add_argument("--config", required=True, type=str, help="Path to one inference config file.")
    parser.add_argument("--frame-start", required=False, type=int, help="First input frame index. Overrides temporal_profile.frame_start.")
    parser.add_argument("--frame-end", required=False, type=int, help="Last input frame index. Overrides temporal_profile.frame_end.")
    parser.add_argument("--inference-preset", required=False, type=str, help="Preset override. Falls back to temporal_profile.inference_preset.")
    parser.add_argument("--record", required=False, type=str, help="Optional record override. Falls back to temporal_profile.record.")
    parser.add_argument("--mode", required=False, type=str, help="Optional exact mode override. Falls back to temporal_profile.mode.")
    parser.add_argument("--output-dir", required=False, type=str, help="Optional output directory override.")
    parser.add_argument("--include-invalid", action="store_true", help="Keep samples whose CSV valid column is false.")
    return parser.parse_args(argv)


def resolve_project_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def require_mapping(value: object, section_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{section_name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def require_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer, got {type(value).__name__}")
    return int(value)


def read_optional_int(value: object, key: str, fallback: int) -> int:
    if value is None:
        return fallback
    return require_int(value, key)


def read_optional_float(value: object, key: str, fallback: float) -> float:
    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number, got {type(value).__name__}")
    return float(value)


def read_optional_bool(value: object, key: str, fallback: bool) -> bool:
    if value is None:
        return fallback
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean, got {type(value).__name__}")
    return bool(value)


def read_x_range(value: object, key: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{key} must be a two-item list: [x_start, x_end]")
    x_start = require_int(value[0], f"{key}[0]")
    x_end = require_int(value[1], f"{key}[1]")
    if x_end <= x_start:
        raise ValueError(f"{key} must satisfy x_end > x_start, got {value}")
    return x_start, x_end


def read_rgb_color(value: object, key: str, row_index: int) -> tuple[int, int, int]:
    if value is None:
        palette = (
            (230, 80, 40),
            (40, 150, 230),
            (90, 180, 90),
            (200, 110, 210),
            (40, 190, 190),
            (210, 160, 60),
        )
        color_rgb = palette[row_index % len(palette)]
        return color_rgb[2], color_rgb[1], color_rgb[0]

    if not isinstance(value, list) or len(value) != 3:
        raise TypeError(f"{key} must be an RGB list with three integers.")

    rgb_values = [require_int(channel, f"{key}[{index}]") for index, channel in enumerate(value)]
    if any(channel < 0 or channel > 255 for channel in rgb_values):
        raise ValueError(f"{key} values must be in [0, 255], got {value}")
    return rgb_values[2], rgb_values[1], rgb_values[0]


def read_profile_rows(profile_config: dict[str, Any]) -> list[ProfileRow]:
    raw_rows = profile_config.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) == 0:
        raise KeyError("temporal_profile.rows must contain at least one row config.")

    global_x_range = profile_config.get("x_range")
    parsed_rows: list[ProfileRow] = []
    for row_index, raw_row in enumerate(raw_rows):
        row = require_mapping(raw_row, f"temporal_profile.rows[{row_index}]")
        row_name = row.get("name")
        if not isinstance(row_name, str) or row_name == "":
            raise TypeError(f"temporal_profile.rows[{row_index}].name must be a non-empty string.")
        y = require_int(row.get("y"), f"temporal_profile.rows[{row_index}].y")
        x_range_value = row.get("x_range", global_x_range)
        if x_range_value is None:
            raise KeyError(
                f"temporal_profile.rows[{row_index}] needs x_range because temporal_profile.x_range is missing.",
            )
        x_start, x_end = read_x_range(x_range_value, f"temporal_profile.rows[{row_index}].x_range")
        color_bgr = read_rgb_color(row.get("color"), f"temporal_profile.rows[{row_index}].color", row_index)
        parsed_rows.append(
            {
                "name": row_name,
                "y": y,
                "x_start": x_start,
                "x_end": x_end,
                "color_bgr": color_bgr,
            },
        )

    return parsed_rows


def read_profile_render_config(config: dict[str, Any], include_invalid_arg: bool) -> ProfileRenderConfig:
    raw_profile_config = config.get("temporal_profile")
    if raw_profile_config is None:
        raise KeyError("Missing required config section: temporal_profile")
    profile_config = require_mapping(raw_profile_config, "temporal_profile")
    include_invalid_config = read_optional_bool(profile_config.get("include_invalid"), "temporal_profile.include_invalid", False)
    include_invalid = include_invalid_arg or include_invalid_config
    line_thickness = read_optional_int(
        profile_config.get("line_thickness"),
        "temporal_profile.line_thickness",
        DEFAULT_LINE_THICKNESS,
    )
    time_scale = read_optional_int(profile_config.get("time_scale"), "temporal_profile.time_scale", DEFAULT_TIME_SCALE)
    profile_width = read_optional_int(
        profile_config.get("profile_width"),
        "temporal_profile.profile_width",
        DEFAULT_PROFILE_WIDTH,
    )
    source_scale = read_optional_float(
        profile_config.get("source_scale"),
        "temporal_profile.source_scale",
        DEFAULT_SOURCE_SCALE,
    )
    if line_thickness <= 0:
        raise ValueError(f"temporal_profile.line_thickness must be positive, got {line_thickness}")
    if time_scale <= 0:
        raise ValueError(f"temporal_profile.time_scale must be positive, got {time_scale}")
    if profile_width <= 0:
        raise ValueError(f"temporal_profile.profile_width must be positive, got {profile_width}")
    if source_scale <= 0.0:
        raise ValueError(f"temporal_profile.source_scale must be positive, got {source_scale}")
    return {
        "rows": read_profile_rows(profile_config),
        "line_thickness": line_thickness,
        "time_scale": time_scale,
        "profile_width": profile_width,
        "source_scale": source_scale,
        "include_invalid": include_invalid,
    }


def read_temporal_profile_config(config: dict[str, Any]) -> dict[str, Any]:
    raw_profile_config = config.get("temporal_profile")
    if raw_profile_config is None:
        raise KeyError("Missing required config section: temporal_profile")
    return require_mapping(raw_profile_config, "temporal_profile")


def read_profile_output_dir(config: dict[str, Any], output_dir_arg: object, project_root: Path) -> Path:
    if isinstance(output_dir_arg, str):
        return resolve_project_path(output_dir_arg, project_root)
    profile_config = read_temporal_profile_config(config)
    output_dir_value = profile_config.get("output_dir")
    if isinstance(output_dir_value, str):
        return resolve_project_path(output_dir_value, project_root)
    return resolve_project_path(str(config["output_dir"]), project_root) / "temporal_profiles"


def select_inference_preset(config: dict[str, Any], requested_preset: object) -> str:
    from scripts.inference import read_inference_presets

    presets = read_inference_presets(config)
    profile_config = read_temporal_profile_config(config)
    preset = requested_preset
    if preset is None:
        preset = profile_config.get("inference_preset")
    if isinstance(preset, str):
        if preset not in presets:
            raise ValueError(f"Requested inference preset {preset!r} is not in config presets: {presets}")
        return preset
    if len(presets) != 1:
        raise ValueError(
            "Config has multiple inference presets and temporal_profile.inference_preset is missing. "
            f"Available presets: {presets}",
        )
    return presets[0]


def read_filter_value(cli_value: object, config: dict[str, Any], key: str) -> str | None:
    if isinstance(cli_value, str):
        return cli_value
    profile_config = read_temporal_profile_config(config)
    value = profile_config.get(key)
    if value is None and key == "record":
        value = profile_config.get("record_filter")
    if value is None and key == "mode":
        value = profile_config.get("mode_filter")
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"temporal_profile.{key} must be a string or null, got {type(value).__name__}")
    return value


def read_frame_bound(cli_value: object, config: dict[str, Any], key: str) -> int:
    if isinstance(cli_value, int):
        return cli_value
    profile_config = read_temporal_profile_config(config)
    value = profile_config.get(key)
    if value is None:
        raise KeyError(f"Missing required value: --{key.replace('_', '-')} or temporal_profile.{key}")
    return require_int(value, f"temporal_profile.{key}")


def validate_frame_range(frame_start: int, frame_end: int) -> None:
    if frame_end <= frame_start:
        raise ValueError(f"frame_end must be larger than frame_start, got start={frame_start} end={frame_end}")
    if (frame_end - frame_start) % 2 != 0:
        raise ValueError(
            f"frame range must cover complete 30fps-to-60fps pairs; got start={frame_start} end={frame_end}",
        )


def filter_dataframe_for_range(
    dataframe: Any,
    frame_start: int,
    frame_end: int,
    record_filter: str | None,
    mode_filter: str | None,
    include_invalid: bool,
) -> Any:
    filtered = dataframe.copy()
    if not include_invalid and "valid" in filtered.columns:
        filtered = filtered[filtered["valid"] == True].reset_index(drop=True)
    if record_filter is not None:
        filtered = filtered[filtered["record"] == record_filter].reset_index(drop=True)
    if mode_filter is not None:
        filtered = filtered[filtered["mode"] == mode_filter].reset_index(drop=True)
    filtered = filtered[
        (filtered["img0"].astype(int) >= frame_start)
        & (filtered["img2"].astype(int) <= frame_end)
    ].reset_index(drop=True)

    if len(filtered) == 0:
        raise RuntimeError(
            "No samples matched "
            f"frame_start={frame_start} frame_end={frame_end} record={record_filter} mode={mode_filter}.",
        )

    group_columns = ["record", "mode"]
    group_count = int(filtered[group_columns].drop_duplicates().shape[0])
    if group_count != 1:
        preview = filtered[group_columns].drop_duplicates().head(12).to_dict("records")
        raise RuntimeError(
            "Frame range matches multiple record/mode groups; pass --record and --mode. "
            f"groups={preview}",
        )

    sorted_dataframe = filtered.sort_values(["img0", "img1", "img2"]).reset_index(drop=True)
    expected_pairs = [(frame_index, frame_index + 1, frame_index + 2) for frame_index in range(frame_start, frame_end, 2)]
    actual_pairs = [
        (int(row["img0"]), int(row["img1"]), int(row["img2"]))
        for _, row in sorted_dataframe.iterrows()
    ]
    if actual_pairs != expected_pairs:
        raise RuntimeError(f"Frame range is not contiguous. expected={expected_pairs} actual={actual_pairs}")

    return sorted_dataframe


def tensor_to_bgr_uint8(tensor: Any) -> Any:
    import numpy as np

    return np.round(tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def calculate_psnr(gt_image: Any, pred_image: Any) -> float:
    import numpy as np

    diff = gt_image.astype(np.float32) - pred_image.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def build_dataset(dataframe: Any, dataset_root_dir: Path, input_fps: int, model_name: str, flow_approx_method: str) -> Any:
    from scripts.inference import BASELINE_MODEL_NAME
    from src.data.dataset_loader import FlowEstimationTrainDataset
    from src.data.dataset_loader import VFITrainDataset
    from src.engine.flow_approx import is_splatting_flow_approx_method

    if model_name == BASELINE_MODEL_NAME:
        return VFITrainDataset(dataframe, str(dataset_root_dir), False, input_fps)

    include_source_depths = is_splatting_flow_approx_method(flow_approx_method=flow_approx_method)
    return FlowEstimationTrainDataset(dataframe, str(dataset_root_dir), input_fps, False, include_source_depths)


def run_sequence_inference(
    selected_dataframe: Any,
    config: dict[str, Any],
    dataset_root_dir: Path,
    checkpoint_path: Path,
    logger: Any,
) -> FrameSequence:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from scripts.inference import DEFAULT_INIT_FLOW_DOWNSCALE_STRATEGY
    from scripts.inference import DEFAULT_INIT_FLOW_MASK_EPSILON
    from scripts.inference import DEFAULT_SPLATTING_FILL_STRATEGY
    from scripts.inference import run_inference_batch_with_fill_strategy
    from scripts.train import read_model_init_args
    from scripts.train import resolve_model_class
    from scripts.train import set_seed

    seed = int(config.get("seed", 1234))
    model_name = str(config["model_name"])
    model_init_args = read_model_init_args(config)
    input_fps = int(config["input_fps"])
    batch_size = int(config.get("batch_size", 1))
    scale_factor = float(config.get("scale_factor", 1.0))
    flow_approx_method = str(config.get("flow_approx_method", "combination"))
    splatting_fill_strategy = str(config.get("splatting_fill_strategy", DEFAULT_SPLATTING_FILL_STRATEGY))
    init_flow_downscale_strategy = str(config.get("init_flow_downscale_strategy", DEFAULT_INIT_FLOW_DOWNSCALE_STRATEGY))
    init_flow_mask_epsilon = float(config.get("init_flow_mask_epsilon", DEFAULT_INIT_FLOW_MASK_EPSILON))

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s model=%s checkpoint=%s", device, model_name, checkpoint_path)

    model_class = resolve_model_class(model_name)
    model = model_class(**model_init_args).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    dataset = build_dataset(selected_dataframe, dataset_root_dir, input_fps, model_name, flow_approx_method)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    gt_frames: dict[int, Any] = {}
    pred_frames: dict[int, Any] = {}
    interpolated_frames: list[int] = []
    row_offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="temporal_profile", leave=True):
            inference_result = run_inference_batch_with_fill_strategy(
                batch,
                device,
                flow_approx_method,
                splatting_fill_strategy,
                init_flow_downscale_strategy,
                init_flow_mask_epsilon,
                model,
                model_name,
                scale_factor,
            )
            img0 = inference_result["img0"]
            imgt = inference_result["imgt"]
            img1 = inference_result["img1"]
            imgt_pred = inference_result["imgt_pred"]
            batch_count = int(imgt_pred.shape[0])
            for batch_index in range(batch_count):
                row = selected_dataframe.iloc[row_offset + batch_index]
                frame0 = int(row["img0"])
                framet = int(row["img1"])
                frame1 = int(row["img2"])
                img0_np = tensor_to_bgr_uint8(img0[batch_index])
                imgt_np = tensor_to_bgr_uint8(imgt[batch_index])
                img1_np = tensor_to_bgr_uint8(img1[batch_index])
                pred_np = tensor_to_bgr_uint8(imgt_pred[batch_index])
                gt_frames[frame0] = img0_np
                gt_frames[framet] = imgt_np
                gt_frames[frame1] = img1_np
                pred_frames[frame0] = img0_np
                pred_frames[framet] = pred_np
                pred_frames[frame1] = img1_np
                interpolated_frames.append(framet)
            row_offset += batch_count

    return {
        "gt_frames": gt_frames,
        "pred_frames": pred_frames,
        "interpolated_frames": interpolated_frames,
    }


def validate_profile_rows(rows: list[ProfileRow], image_shape: tuple[int, int, int]) -> None:
    height = int(image_shape[0])
    width = int(image_shape[1])
    for row in rows:
        y = row["y"]
        x_start = row["x_start"]
        x_end = row["x_end"]
        if y < 0 or y >= height:
            raise ValueError(f"Profile row {row['name']!r} has y={y}, outside image height={height}")
        if x_start < 0 or x_end > width:
            raise ValueError(
                f"Profile row {row['name']!r} has x_range=[{x_start}, {x_end}], outside image width={width}",
            )


def extract_temporal_profile(frames: dict[int, Any], frame_indices: list[int], row: ProfileRow, line_thickness: int) -> Any:
    import numpy as np

    first_frame = frames[frame_indices[0]]
    height = int(first_frame.shape[0])
    half_thickness = max(0, line_thickness // 2)
    y_start = max(0, row["y"] - half_thickness)
    y_end = min(height, row["y"] + half_thickness + 1)
    profiles = []
    for frame_index in frame_indices:
        frame = frames[frame_index]
        band = frame[y_start:y_end, row["x_start"] : row["x_end"], :].astype(np.float32)
        profiles.append(np.round(band.mean(axis=0)).astype(np.uint8))
    return np.stack(profiles, axis=0)


def scale_profile(profile: Any, profile_width: int, time_scale: int, cv2: Any) -> Any:
    target_height = int(profile.shape[0]) * time_scale
    return cv2.resize(profile, (profile_width, target_height), interpolation=cv2.INTER_NEAREST)


def draw_text(canvas: Any, text: str, x: int, y: int, scale: float, color_bgr: tuple[int, int, int], thickness: int, cv2: Any) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, thickness, cv2.LINE_AA)


def paste_image(canvas: Any, image: Any, x: int, y: int) -> None:
    canvas[y : y + image.shape[0], x : x + image.shape[1]] = image


def fit_text(text: str, max_width: int, scale: float, thickness: int, cv2: Any) -> str:
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    candidate = text
    while len(candidate) > 1:
        candidate = candidate[:-1]
        trial = f"{candidate}{suffix}"
        if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= max_width:
            return trial
    return suffix


def build_source_preview(first_frame: Any, rows: list[ProfileRow], source_scale: float, line_thickness: int, cv2: Any) -> Any:
    source_width = max(1, int(round(first_frame.shape[1] * source_scale)))
    source_height = max(1, int(round(first_frame.shape[0] * source_scale)))
    preview = cv2.resize(first_frame, (source_width, source_height), interpolation=cv2.INTER_AREA)
    scaled_thickness = max(1, int(round(line_thickness * source_scale)))
    for row in rows:
        y = int(round(row["y"] * source_scale))
        x_start = int(round(row["x_start"] * source_scale))
        x_end = int(round(row["x_end"] * source_scale))
        color = row["color_bgr"]
        cv2.line(preview, (x_start, y), (x_end, y), color, scaled_thickness, cv2.LINE_AA)
        label = fit_text(row["name"], max(30, source_width - x_start - 6), 0.45, 1, cv2)
        draw_text(preview, label, min(x_start + 5, source_width - 30), max(14, y - 5), 0.45, color, 1, cv2)
    return preview


def draw_frame_labels(
    canvas: Any,
    frame_indices: list[int],
    x: int,
    y: int,
    time_scale: int,
    stream_label: str,
    interpolated_frames: list[int],
    cv2: Any,
) -> None:
    interpolated_set = set(interpolated_frames)
    for frame_offset, frame_index in enumerate(frame_indices):
        y_mid = y + frame_offset * time_scale + int(time_scale * 0.68)
        if frame_index in interpolated_set:
            suffix = stream_label
        else:
            suffix = "input"
        label = f"{frame_index} {suffix}"
        draw_text(canvas, label, x, y_mid, 0.43, BLUE_TEXT_BGR, 1, cv2)


def draw_profile_grid(canvas: Any, x: int, y: int, width: int, frame_count: int, time_scale: int, cv2: Any) -> None:
    height = frame_count * time_scale
    cv2.rectangle(canvas, (x, y), (x + width, y + height), GRID_BGR, 1)
    for index in range(1, frame_count):
        line_y = y + index * time_scale
        cv2.line(canvas, (x, line_y), (x + width, line_y), (255, 255, 255), 1, cv2.LINE_AA)


def render_temporal_profile_figure(
    sequence: FrameSequence,
    rows: list[ProfileRow],
    frame_indices: list[int],
    render_config: ProfileRenderConfig,
    title: str,
    output_path: Path,
    cv2: Any,
    np: Any,
) -> None:
    if len(frame_indices) == 0:
        raise ValueError("frame_indices must contain at least one frame.")

    first_frame = sequence["gt_frames"][frame_indices[0]]
    validate_profile_rows(rows, tuple(first_frame.shape))
    source_preview = build_source_preview(first_frame, rows, render_config["source_scale"], render_config["line_thickness"], cv2)

    margin = 18
    title_height = 62
    source_title_height = 26
    source_width = int(source_preview.shape[1])
    source_height = int(source_preview.shape[0])
    profile_width = render_config["profile_width"]
    time_scale = render_config["time_scale"]
    label_width = 94
    panel_gap = 22
    row_header_height = 28
    panel_title_height = 22
    panel_height = len(frame_indices) * time_scale
    row_block_height = row_header_height + (panel_title_height + panel_height) * 2 + panel_gap + 18
    content_x = margin * 2 + source_width + 18
    profile_x = content_x + label_width
    canvas_width = profile_x + profile_width + margin
    canvas_height = max(
        title_height + source_title_height + source_height + margin,
        title_height + len(rows) * row_block_height + margin,
    )
    canvas = np.full((canvas_height, canvas_width, 3), BACKGROUND_BGR, dtype=np.uint8)

    fitted_title = fit_text(title, canvas_width - 2 * margin, 0.72, 2, cv2)
    draw_text(canvas, fitted_title, margin, 36, 0.72, TEXT_BGR, 2, cv2)
    draw_text(canvas, "Sample rows", margin, title_height + 4, 0.52, TEXT_BGR, 1, cv2)
    paste_image(canvas, source_preview, margin, title_height + source_title_height)

    current_y = title_height
    for row in rows:
        draw_text(canvas, row["name"], content_x, current_y + 18, 0.56, TEXT_BGR, 1, cv2)
        current_y += row_header_height
        stream_specs = (
            ("Ground truth", "GT", sequence["gt_frames"]),
            ("Prediction", "pred", sequence["pred_frames"]),
        )
        for stream_index, (panel_label, frame_suffix, frames) in enumerate(stream_specs):
            panel_y = current_y + stream_index * (panel_title_height + panel_height + panel_gap)
            draw_text(canvas, panel_label, profile_x, panel_y + 15, 0.5, TEXT_BGR, 1, cv2)
            profile = extract_temporal_profile(frames, frame_indices, row, render_config["line_thickness"])
            scaled_profile = scale_profile(profile, profile_width, time_scale, cv2)
            image_y = panel_y + panel_title_height
            paste_image(canvas, scaled_profile, profile_x, image_y)
            draw_profile_grid(canvas, profile_x, image_y, profile_width, len(frame_indices), time_scale, cv2)
            draw_frame_labels(canvas, frame_indices, content_x, image_y, time_scale, frame_suffix, sequence["interpolated_frames"], cv2)
        current_y += row_block_height - row_header_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    is_written = cv2.imwrite(str(output_path), canvas)
    if not is_written:
        raise ValueError(f"Failed to write temporal profile figure: {output_path}")


def build_metrics_records(sequence: FrameSequence, rows: list[ProfileRow], line_thickness: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for frame_index in sequence["interpolated_frames"]:
        gt_image = sequence["gt_frames"][frame_index]
        pred_image = sequence["pred_frames"][frame_index]
        records.append(
            {
                "scope": "full_frame",
                "row_name": "",
                "frame": frame_index,
                "psnr": calculate_psnr(gt_image, pred_image),
            },
        )
        for row in rows:
            gt_profile = extract_temporal_profile(sequence["gt_frames"], [frame_index], row, line_thickness)[0]
            pred_profile = extract_temporal_profile(sequence["pred_frames"], [frame_index], row, line_thickness)[0]
            records.append(
                {
                    "scope": "profile_row",
                    "row_name": row["name"],
                    "frame": frame_index,
                    "psnr": calculate_psnr(gt_profile, pred_profile),
                },
            )
    return records


def write_metrics_csv(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scope", "row_name", "frame", "psnr"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def sanitize_path_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if sanitized == "":
        raise ValueError(f"Cannot build a safe path part from value: {value!r}")
    return sanitized[:120]


def format_checkpoint_title(checkpoint_path: Path, project_root: Path) -> str:
    try:
        return checkpoint_path.relative_to(project_root).as_posix()
    except ValueError:
        return str(checkpoint_path)


def write_summary(
    output_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    inference_preset: str,
    selected_dataframe: Any,
    frame_start: int,
    frame_end: int,
    render_config: ProfileRenderConfig,
) -> None:
    first_row = selected_dataframe.iloc[0]
    summary = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "inference_preset": inference_preset,
        "record": str(first_row["record"]),
        "mode": str(first_row["mode"]),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "rows": [
            {
                "name": row["name"],
                "y": row["y"],
                "x_range": [row["x_start"], row["x_end"]],
            }
            for row in render_config["rows"]
        ],
        "line_thickness": render_config["line_thickness"],
        "time_scale": render_config["time_scale"],
        "profile_width": render_config["profile_width"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    import cv2
    import numpy as np

    from scripts.train import build_merged_dataframe
    from src.utils.config import load_yaml_file
    from src.utils.logger import build_logger

    config_path = resolve_project_path(str(args.config), PROJECT_ROOT)
    config = load_yaml_file(config_path)
    frame_start = read_frame_bound(args.frame_start, config, "frame_start")
    frame_end = read_frame_bound(args.frame_end, config, "frame_end")
    validate_frame_range(frame_start, frame_end)
    render_config = read_profile_render_config(config, bool(args.include_invalid))
    inference_preset = select_inference_preset(config, args.inference_preset)
    record_filter = read_filter_value(args.record, config, "record")
    mode_filter = read_filter_value(args.mode, config, "mode")

    root_dir = resolve_project_path(str(config["root_dir"]), PROJECT_ROOT)
    dataset_root_dir = resolve_project_path(str(config["dataset_root_dir"]), PROJECT_ROOT)
    checkpoint_path = resolve_project_path(str(config["checkpoint_path"]), PROJECT_ROOT)
    output_root = read_profile_output_dir(config, args.output_dir, PROJECT_ROOT)
    only_fps = int(config["only_fps"])

    logger = build_logger("scripts.inference_temporal_profile")
    staging_dir = output_root / "_merged_csv"
    staging_dir.mkdir(parents=True, exist_ok=True)
    dataframe = build_merged_dataframe(root_dir, staging_dir, inference_preset, only_fps, logger)
    selected_dataframe = filter_dataframe_for_range(
        dataframe,
        frame_start,
        frame_end,
        record_filter,
        mode_filter,
        render_config["include_invalid"],
    )

    sequence = run_sequence_inference(selected_dataframe, config, dataset_root_dir, checkpoint_path, logger)
    first_row = selected_dataframe.iloc[0]
    weights_title = format_checkpoint_title(checkpoint_path, PROJECT_ROOT)
    record_tag = sanitize_path_part(str(first_row["record"]))
    mode_tag = sanitize_path_part(str(first_row["mode"]))
    weights_tag = sanitize_path_part(weights_title)
    range_tag = f"frames_{frame_start:04d}_{frame_end:04d}"
    output_dir = output_root / weights_tag / record_tag / mode_tag / range_tag

    all_frame_indices = list(range(frame_start, frame_end + 1))
    predicted_frame_indices = sequence["interpolated_frames"]
    title_prefix = f"Weights: {weights_title}"
    render_temporal_profile_figure(
        sequence,
        render_config["rows"],
        all_frame_indices,
        render_config,
        f"{title_prefix} | GT / prediction with input frames",
        output_dir / "temporal_profile_with_inputs.png",
        cv2,
        np,
    )
    render_temporal_profile_figure(
        sequence,
        render_config["rows"],
        predicted_frame_indices,
        render_config,
        f"{title_prefix} | GT / prediction only",
        output_dir / "temporal_profile_predictions_only.png",
        cv2,
        np,
    )
    metric_records = build_metrics_records(sequence, render_config["rows"], render_config["line_thickness"])
    write_metrics_csv(metric_records, output_dir / "temporal_profile_metrics.csv")
    write_summary(
        output_dir / "temporal_profile_summary.json",
        config_path,
        checkpoint_path,
        inference_preset,
        selected_dataframe,
        frame_start,
        frame_end,
        render_config,
    )

    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "with_inputs": str(output_dir / "temporal_profile_with_inputs.png"),
            "predictions_only": str(output_dir / "temporal_profile_predictions_only.png"),
            "metrics": str(output_dir / "temporal_profile_metrics.csv"),
            "summary": str(output_dir / "temporal_profile_summary.json"),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main(sys.argv[1:])
