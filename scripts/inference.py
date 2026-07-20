from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.checkpoints import load_inference_checkpoint
from src.engine.dataset_runs import build_inference_dataset
from src.engine.dataset_runs import build_merged_dataframe
from src.engine.dataset_runs import filter_valid_dataframe
from src.engine.evaluation import average_metric_values
from src.engine.evaluation import build_flip_evaluator
from src.engine.evaluation import build_flolpips_model
from src.engine.evaluation import build_lpips_model
from src.engine.evaluation import build_metric_meters
from src.engine.evaluation import build_vfips_model
from src.engine.evaluation import calculate_batch_metrics
from src.engine.evaluation import calculate_psnr_div_sample
from src.engine.evaluation import calculate_vfips_sequence_sample_values
from src.engine.evaluation import format_metric_averages
from src.engine.evaluation import prepare_psnr_div_sample
from src.engine.evaluation import PSNR_DIV_METRIC_NAME
from src.engine.evaluation import VFIPS_METRIC_NAME
from src.engine.interpolation_batch import InterpolationBatchResult
from src.engine.interpolation_batch import run_inference_batch
from src.engine.model_registry import resolve_model_class
from src.engine.model_registry import RESIDUAL_FLOW_APPROX_MODEL_NAME
from src.engine.model_registry import RESIDUAL_MODEL_NAME
from src.engine.model_registry import set_model_convex_upsampling
from src.engine.model_registry import uses_image_only_vfi_model
from src.engine.run_config import build_inference_dry_run_summary
from src.engine.run_config import build_inference_run_config
from src.engine.run_config import RgbSequenceConfig
from src.utils.logger import build_logger
from src.utils.model_stats import build_model_parameter_summary
from src.utils.seed import set_seed
# Model variants:
# - IFRNet: baseline
# - IFRNet_Residual: residual model initialized by bmv/fmv from the 60fps motion labels
# - IFRNet_Residual_FlowApprox: residual model initialized by approximated 60fps motion from bmv_30/fmv_30

SplattingRegionMaps = dict[str, Any]
VfipsSegment = dict[str, Any]

SELECTED_ARTIFACT_COLUMNS: tuple[str, ...] = (
    "image_0_path",
    "image_1_path",
    "image_gt_path",
    "image_pred_path",
    "image_merge_path",
    "bmv_path",
    "fmv_path",
    "flow_1_to_0_path",
    "flow_1_to_2_path",
    "flow_mask_path",
    "image_0_warped_path",
    "image_1_warped_path",
    "image_0_bmv_warped_path",
    "image_1_fmv_warped_path",
    "image_0_init_warped_path",
    "image_1_init_warped_path",
    "image_init_warped_merge_path",
    "splatting_region_label_bmv_color_path",
    "splatting_region_label_fmv_color_path",
    "splatting_hit_count_bmv_path",
    "splatting_hit_count_fmv_path",
)


def update_metric_meter(metric_meters: dict[str, Any], metric_name: str, metric_value: float) -> None:
    if metric_name in (PSNR_DIV_METRIC_NAME, VFIPS_METRIC_NAME) and not math.isfinite(metric_value):
        return

    metric_meters[metric_name].update(metric_value, 1)


def start_vfips_segment(frame_index: int, frame: Any) -> VfipsSegment:
    source_frame = frame.detach().cpu()
    return {
        "last_frame_index": frame_index,
        "reference_frames": [source_frame],
        "prediction_frames": [source_frame],
        "sample_indices": [],
        "sample_frame_positions": [],
    }


def append_vfips_sample(
    vfips_segments: list[VfipsSegment],
    sample_index: int,
    frame_0_index: int,
    frame_2_index: int,
    img0: Any,
    imgt: Any,
    img1: Any,
    imgt_pred: Any,
) -> None:
    if len(vfips_segments) == 0 or int(vfips_segments[-1]["last_frame_index"]) != frame_0_index:
        vfips_segments.append(start_vfips_segment(frame_index=frame_0_index, frame=img0))

    current_segment = vfips_segments[-1]
    reference_frames = current_segment["reference_frames"]
    prediction_frames = current_segment["prediction_frames"]
    sample_frame_positions = current_segment["sample_frame_positions"]
    sample_indices = current_segment["sample_indices"]

    sample_frame_positions.append(len(reference_frames))
    sample_indices.append(sample_index)
    reference_frames.append(imgt.detach().cpu())
    prediction_frames.append(imgt_pred.detach().cpu())
    endpoint_frame = img1.detach().cpu()
    reference_frames.append(endpoint_frame)
    prediction_frames.append(endpoint_frame)
    current_segment["last_frame_index"] = frame_2_index


def calculate_group_vfips_values(
    vfips_segments: list[VfipsSegment],
    metric_config: dict[str, object],
    vfips_model: Any | None,
) -> dict[int, float]:
    if not bool(metric_config["enable_vfips"]):
        return {}
    if vfips_model is None:
        raise RuntimeError("metrics.enable_vfips=true requires a loaded VFIPS model.")

    values_by_sample: dict[int, float] = {}
    for vfips_segment in vfips_segments:
        segment_values = calculate_vfips_sequence_sample_values(
            reference_frames=vfips_segment["reference_frames"],
            prediction_frames=vfips_segment["prediction_frames"],
            sample_indices=vfips_segment["sample_indices"],
            sample_frame_positions=vfips_segment["sample_frame_positions"],
            vfips_model=vfips_model,
            clip_length=int(metric_config["vfips_clip_length"]),
            stride=int(metric_config["vfips_stride"]),
        )
        values_by_sample.update(segment_values)

    return values_by_sample


def apply_vfips_values_to_group_rows(
    group_rows: list[dict[str, object]],
    values_by_sample: dict[int, float],
    metric_meters: dict[str, Any],
    record_metric_meters: dict[str, Any],
) -> None:
    for sample_row in group_rows:
        sample_index = int(sample_row["sample_index"])
        vfips_value = float(values_by_sample.get(sample_index, float("nan")))
        sample_row[VFIPS_METRIC_NAME] = vfips_value
        update_metric_meter(metric_meters, VFIPS_METRIC_NAME, vfips_value)
        update_metric_meter(record_metric_meters, VFIPS_METRIC_NAME, vfips_value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference from one config file.")
    parser.add_argument("--config", required=True, type=str, help="Path to one inference config file.")
    return parser.parse_args(argv)


def save_flow_diff_visuals(
    save_dir: Path,
    name: str,
    bg_img_np: Any,
    init_flow_np: Any,
    final_flow_np: Any,
    flow_to_image: Any,
    save_image: Any,
    cv2: Any,
    np: Any,
    threshold: float,
    percentile: float,
) -> dict[str, float]:
    diff_flow_np = final_flow_np - init_flow_np
    diff_mag_np = np.linalg.norm(diff_flow_np, axis=2)
    scale = max(float(np.percentile(diff_mag_np, percentile)), 1e-6)
    diff_mag_u8 = np.round(np.clip(diff_mag_np / scale, 0.0, 1.0) * 255.0).astype(np.uint8)
    diff_mag_color = cv2.applyColorMap(diff_mag_u8, cv2.COLORMAP_TURBO)

    height, width = diff_mag_color.shape[:2]
    colorbar_values = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    colorbar_u8 = np.round(colorbar_values * 255.0).astype(np.uint8)
    colorbar = cv2.applyColorMap(colorbar_u8, cv2.COLORMAP_TURBO)
    colorbar = cv2.resize(colorbar, (28, height), interpolation=cv2.INTER_NEAREST)

    overlay = cv2.addWeighted(bg_img_np.astype(np.uint8), 0.2, diff_mag_color, 0.8, 0.0)
    overlay_with_colorbar = np.full((height, width + 126, 3), 255, dtype=np.uint8)
    overlay_with_colorbar[:, :width] = overlay
    overlay_with_colorbar[:, width + 8 : width + 36] = colorbar

    for canvas in (overlay_with_colorbar,):
        cv2.putText(canvas, "|dflow|", (width + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        for tick_index in range(5):
            tick_value = tick_index / 4.0
            tick_y = int((1.0 - tick_value) * (height - 1))
            actual_value = tick_value * scale
            cv2.line(canvas, (width + 8, tick_y), (width + 35, tick_y), (0, 0, 0), 1)
            cv2.putText(
                canvas,
                f"{actual_value:.2f}",
                (width + 42, min(max(tick_y + 4, 12), height - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    changed_mask = np.where(diff_mag_np > threshold, 255, 0).astype(np.uint8)
    save_image(save_dir / f"init_flow_{name}.png", flow_to_image(init_flow_np))
    save_image(save_dir / f"diff_flow_{name}.png", flow_to_image(diff_flow_np))
    save_image(save_dir / f"diff_mag_cb_overlay_{name}.png", overlay_with_colorbar)
    save_image(save_dir / f"diff_changed_thr_{threshold:.2f}_{name}.png", changed_mask)
    return {
        "diff_mag_mean": float(diff_mag_np.mean()),
        "diff_mag_max": float(diff_mag_np.max()),
        "diff_changed_ratio": float((changed_mask > 0).mean()),
        "diff_percentile_value": float(scale),
    }


def build_direction_splatting_region_label_image(hit: Any, many_to_one: Any, np: Any) -> Any:
    hit_np = hit[0, 0].detach().cpu().numpy().astype(bool)
    many_to_one_np = many_to_one[0, 0].detach().cpu().numpy().astype(bool)
    hole = ~hit_np
    label = np.zeros(hole.shape, dtype=np.uint8)
    label[hole] = 128
    label[many_to_one_np] = 255
    return label


def build_splatting_legend_canvas(category_map: Any, cv2: Any, np: Any) -> Any:
    height, width = category_map.shape
    total_pixels = float(category_map.size)
    normal_ratio = float(np.count_nonzero(category_map == 1) / total_pixels * 100.0)
    hole_ratio = float(np.count_nonzero(category_map == 0) / total_pixels * 100.0)
    multi2_ratio = float(np.count_nonzero(category_map == 2) / total_pixels * 100.0)
    multi3_ratio = float(np.count_nonzero(category_map == 3) / total_pixels * 100.0)
    multi4_ratio = float(np.count_nonzero(category_map == 4) / total_pixels * 100.0)
    multi5_ratio = float(np.count_nonzero(category_map == 5) / total_pixels * 100.0)
    legend_width = 190
    canvas = np.full((height, width + legend_width, 3), 255, dtype=np.uint8)
    legend_items = (
        ("normal", normal_ratio, (24, 24, 24)),
        ("hole", hole_ratio, (220, 220, 220)),
        ("multi=2", multi2_ratio, (255, 80, 40)),
        ("multi=3", multi3_ratio, (70, 190, 70)),
        ("multi=4", multi4_ratio, (0, 190, 255)),
        ("multi>=5", multi5_ratio, (40, 40, 220)),
    )
    cv2.putText(canvas, "legend", (width + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for index, (label_text, ratio, bgr_color) in enumerate(legend_items):
        y = 54 + index * 40
        cv2.rectangle(canvas, (width + 12, y - 18), (width + 34, y + 4), bgr_color, -1)
        cv2.rectangle(canvas, (width + 12, y - 18), (width + 34, y + 4), (0, 0, 0), 1)
        cv2.putText(
            canvas,
            f"{label_text} {ratio:.1f}%",
            (width + 42, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def colorize_splatting_category_map(category_map: Any, cv2: Any, np: Any) -> Any:
    _height, width = category_map.shape
    color = np.zeros((*category_map.shape, 3), dtype=np.uint8)
    color[category_map == 1] = np.array([24, 24, 24], dtype=np.uint8)
    color[category_map == 0] = np.array([220, 220, 220], dtype=np.uint8)
    color[category_map == 2] = np.array([255, 80, 40], dtype=np.uint8)
    color[category_map == 3] = np.array([70, 190, 70], dtype=np.uint8)
    color[category_map == 4] = np.array([0, 190, 255], dtype=np.uint8)
    color[category_map == 5] = np.array([40, 40, 220], dtype=np.uint8)
    canvas = build_splatting_legend_canvas(category_map, cv2, np)
    canvas[:, :width] = color
    return canvas


def build_splatting_category_map_from_hit_count(hit_count: Any, np: Any) -> Any:
    hit_count_np = hit_count.detach().cpu().numpy().astype(np.float32)
    category_map = np.ones(hit_count_np.shape, dtype=np.uint8)
    category_map[hit_count_np <= 0] = 0
    category_map[hit_count_np == 2] = 2
    category_map[hit_count_np == 3] = 3
    category_map[hit_count_np == 4] = 4
    category_map[hit_count_np >= 5] = 5
    return category_map


def colorize_splatting_hit_count(hit_count: Any, cv2: Any, np: Any) -> Any:
    return colorize_splatting_category_map(build_splatting_category_map_from_hit_count(hit_count, np), cv2, np)


def colorize_splatting_region_label(label: Any, hit_count: Any, cv2: Any, np: Any) -> Any:
    category_map = build_splatting_category_map_from_hit_count(hit_count, np)
    category_map[label == 128] = 0
    category_map[label == 0] = 1
    return colorize_splatting_category_map(category_map, cv2, np)


def save_splatting_region_visuals(
    cv2: Any,
    np: Any,
    region_maps: SplattingRegionMaps | None,
    save_dir: Path,
    save_image: Any,
) -> dict[str, str]:
    image_paths = {
        "splatting_region_label_bmv_color_path": "",
        "splatting_region_label_fmv_color_path": "",
        "splatting_hit_count_bmv_path": "",
        "splatting_hit_count_fmv_path": "",
    }
    if region_maps is None:
        return image_paths

    bmv_region_label = build_direction_splatting_region_label_image(
        region_maps["bmv_hit"],
        region_maps["bmv_many_to_one"],
        np,
    )
    fmv_region_label = build_direction_splatting_region_label_image(
        region_maps["fmv_hit"],
        region_maps["fmv_many_to_one"],
        np,
    )
    outputs = {
        "splatting_region_label_bmv_color_path": colorize_splatting_region_label(
            bmv_region_label,
            region_maps["bmv_hit_count"][0, 0],
            cv2,
            np,
        ),
        "splatting_region_label_fmv_color_path": colorize_splatting_region_label(
            fmv_region_label,
            region_maps["fmv_hit_count"][0, 0],
            cv2,
            np,
        ),
        "splatting_hit_count_bmv_path": colorize_splatting_hit_count(region_maps["bmv_hit_count"][0, 0], cv2, np),
        "splatting_hit_count_fmv_path": colorize_splatting_hit_count(region_maps["fmv_hit_count"][0, 0], cv2, np),
    }
    filenames = {
        "splatting_region_label_bmv_color_path": "splatting_region_label_bmv_color.png",
        "splatting_region_label_fmv_color_path": "splatting_region_label_fmv_color.png",
        "splatting_hit_count_bmv_path": "splatting_hit_count_bmv.png",
        "splatting_hit_count_fmv_path": "splatting_hit_count_fmv.png",
    }
    for key, image in outputs.items():
        path = save_dir / filenames[key]
        save_image(path, image)
        image_paths[key] = str(path)

    return image_paths


def build_blank_selected_artifact_paths() -> dict[str, str]:
    return {column_name: "" for column_name in SELECTED_ARTIFACT_COLUMNS}


def has_flow_artifacts(inference_result: InterpolationBatchResult) -> bool:
    return (
        inference_result.bmv is not None
        and inference_result.fmv is not None
        and inference_result.up_flow0_1 is not None
        and inference_result.up_flow1_1 is not None
        and inference_result.up_mask_1 is not None
    )


def validate_flow_specific_requests(model_name: str, save_topk_largest_flow_diff: int) -> None:
    if uses_image_only_vfi_model(model_name=model_name) and save_topk_largest_flow_diff > 0:
        raise ValueError(
            "save_topk_largest_flow_diff requires flow artifacts, "
            f"but model_name={model_name} returns image-only predictions. Set save_topk_largest_flow_diff=0."
        )


def build_path_safe_tag(value: Any) -> str:
    return str(value).replace("/", "__").replace("\\", "__")


def should_export_rgb_sequence(config: RgbSequenceConfig, record: Any, mode_name: Any) -> bool:
    if not config.enabled:
        return False
    if config.record_filter is not None and str(record) != config.record_filter:
        return False
    if config.mode_filter is not None and str(mode_name) != config.mode_filter:
        return False
    return True


def tensor_image_to_uint8(image: Any, np: Any) -> Any:
    return np.round(image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def save_rgb_endpoint_frame(
    output_dir: Path,
    frame_index: int,
    image: Any,
    written_endpoint_frames: set[int],
    dedupe_endpoints: bool,
    save_image: Any,
    np: Any,
) -> None:
    if dedupe_endpoints and frame_index in written_endpoint_frames:
        return

    save_image(output_dir / f"frame_{frame_index:04d}.png", tensor_image_to_uint8(image=image, np=np))
    written_endpoint_frames.add(frame_index)


def save_rgb_sequence_sample(
    config: RgbSequenceConfig,
    inference_preset: Any,
    record: Any,
    mode_name: Any,
    frame_0_index: int,
    frame_1_index: int,
    frame_2_index: int,
    img0: Any,
    imgt: Any,
    img1: Any,
    imgt_pred: Any,
    written_endpoint_frames: set[int],
    save_image: Any,
    np: Any,
) -> None:
    output_dir = config.output_dir / str(inference_preset) / str(record) / build_path_safe_tag(value=mode_name)
    save_rgb_endpoint_frame(
        output_dir=output_dir,
        frame_index=frame_0_index,
        image=img0,
        written_endpoint_frames=written_endpoint_frames,
        dedupe_endpoints=config.dedupe_endpoints,
        save_image=save_image,
        np=np,
    )
    if config.export_target:
        save_image(output_dir / f"frame_{frame_1_index:04d}_t.png", tensor_image_to_uint8(image=imgt, np=np))
    if config.export_prediction:
        save_image(output_dir / f"frame_{frame_1_index:04d}_p.png", tensor_image_to_uint8(image=imgt_pred, np=np))
    save_rgb_endpoint_frame(
        output_dir=output_dir,
        frame_index=frame_2_index,
        image=img1,
        written_endpoint_frames=written_endpoint_frames,
        dedupe_endpoints=config.dedupe_endpoints,
        save_image=save_image,
        np=np,
    )


def save_selected_sample_artifacts(
    cv2: Any,
    flow_diff_percentile: float,
    flow_diff_threshold: float,
    flow_to_image: Any,
    inference_result: InterpolationBatchResult,
    np: Any,
    save_dir: Path,
    save_image: Any,
) -> dict[str, str]:
    img0 = inference_result.img0
    img1 = inference_result.img1
    imgt = inference_result.imgt
    bmv = inference_result.bmv
    fmv = inference_result.fmv
    imgt_pred = inference_result.imgt_pred
    imgt_merge = inference_result.imgt_merge
    init_bmv = inference_result.init_bmv
    init_fmv = inference_result.init_fmv
    init_masks = inference_result.init_masks
    up_flow0_1 = inference_result.up_flow0_1
    up_flow1_1 = inference_result.up_flow1_1
    up_mask_1 = inference_result.up_mask_1
    splatting_region_maps = inference_result.splatting_region_maps

    img0_np = np.round(img0[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img1_np = np.round(img1[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    imgt_np = np.round(imgt[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img_pred_np = np.round(imgt_pred[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    imgt_merge_np = None if imgt_merge is None else np.round(imgt_merge[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    image_paths = build_blank_selected_artifact_paths()
    image_paths.update(
        {
            "image_0_path": str(save_dir / "image_0.png"),
            "image_1_path": str(save_dir / "image_1.png"),
            "image_gt_path": str(save_dir / "image_gt.png"),
            "image_pred_path": str(save_dir / "image_pred.png"),
            "image_merge_path": str(save_dir / "image_merge.png") if imgt_merge_np is not None else "",
        }
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    save_image(Path(image_paths["image_0_path"]), img0_np)
    save_image(Path(image_paths["image_1_path"]), img1_np)
    save_image(Path(image_paths["image_gt_path"]), imgt_np)
    save_image(Path(image_paths["image_pred_path"]), img_pred_np)
    if imgt_merge_np is not None:
        save_image(Path(image_paths["image_merge_path"]), imgt_merge_np)

    if not has_flow_artifacts(inference_result=inference_result):
        return image_paths

    from src.models.external.IFRNet.utils import warp

    img0_warped = warp(img0, up_flow0_1)
    img1_warped = warp(img1, up_flow1_1)
    img0_bmv_warped = warp(img0, bmv)
    img1_fmv_warped = warp(img1, fmv)
    init_img0_warped = None
    init_img1_warped = None
    init_merge = None
    if init_bmv is not None and init_fmv is not None:
        init_img0_warped = warp(img0, init_bmv)
        init_img1_warped = warp(img1, init_fmv)
        if init_masks is None:
            init_merge = 0.5 * init_img0_warped + 0.5 * init_img1_warped
        else:
            init_bmv_mask = init_masks[:, 0:1]
            init_fmv_mask = init_masks[:, 1:2]
            init_weight_sum = init_bmv_mask + init_fmv_mask
            init_average = 0.5 * init_img0_warped + 0.5 * init_img1_warped
            init_weighted = (
                init_img0_warped * init_bmv_mask + init_img1_warped * init_fmv_mask
            ) / init_weight_sum.clamp_min(1.0)
            init_merge = init_average.where(init_weight_sum <= 0, init_weighted)

    bmv_np = flow_to_image(bmv[0].detach().cpu().permute(1, 2, 0).numpy())
    fmv_np = flow_to_image(fmv[0].detach().cpu().permute(1, 2, 0).numpy())
    flow_1_to_0_np = flow_to_image(up_flow0_1[0].detach().cpu().permute(1, 2, 0).numpy())
    flow_1_to_2_np = flow_to_image(up_flow1_1[0].detach().cpu().permute(1, 2, 0).numpy())
    flow_mask_np = np.round(up_mask_1[0, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
    img0_warped_np = np.round(img0_warped[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img1_warped_np = np.round(img1_warped[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img0_bmv_warped_np = np.round(img0_bmv_warped[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img1_fmv_warped_np = np.round(img1_fmv_warped[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    init_img0_warped_np = None if init_img0_warped is None else np.round(init_img0_warped[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    init_img1_warped_np = None if init_img1_warped is None else np.round(init_img1_warped[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    init_merge_np = None if init_merge is None else np.round(init_merge[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    image_paths.update({
        "bmv_path": str(save_dir / "bmv.png"),
        "fmv_path": str(save_dir / "fmv.png"),
        "flow_1_to_0_path": str(save_dir / "flow_1_to_0.png"),
        "flow_1_to_2_path": str(save_dir / "flow_1_to_2.png"),
        "flow_mask_path": str(save_dir / "flow_mask.png"),
        "image_0_warped_path": str(save_dir / "image_0_warped.png"),
        "image_1_warped_path": str(save_dir / "image_1_warped.png"),
        "image_0_bmv_warped_path": str(save_dir / "image_0_bmv_warped.png"),
        "image_1_fmv_warped_path": str(save_dir / "image_1_fmv_warped.png"),
        "image_0_init_warped_path": str(save_dir / "image_0_init_warped.png") if init_img0_warped_np is not None else "",
        "image_1_init_warped_path": str(save_dir / "image_1_init_warped.png") if init_img1_warped_np is not None else "",
        "image_init_warped_merge_path": str(save_dir / "image_init_warped_merge.png") if init_merge_np is not None else "",
    })
    image_paths.update(save_splatting_region_visuals(cv2, np, splatting_region_maps, save_dir, save_image))

    save_image(Path(image_paths["bmv_path"]), bmv_np)
    save_image(Path(image_paths["fmv_path"]), fmv_np)
    save_image(Path(image_paths["flow_1_to_0_path"]), flow_1_to_0_np)
    save_image(Path(image_paths["flow_1_to_2_path"]), flow_1_to_2_np)
    save_image(Path(image_paths["flow_mask_path"]), flow_mask_np)
    save_image(Path(image_paths["image_0_warped_path"]), img0_warped_np)
    save_image(Path(image_paths["image_1_warped_path"]), img1_warped_np)
    save_image(Path(image_paths["image_0_bmv_warped_path"]), img0_bmv_warped_np)
    save_image(Path(image_paths["image_1_fmv_warped_path"]), img1_fmv_warped_np)
    if init_img0_warped_np is not None and init_img1_warped_np is not None and init_merge_np is not None:
        save_image(Path(image_paths["image_0_init_warped_path"]), init_img0_warped_np)
        save_image(Path(image_paths["image_1_init_warped_path"]), init_img1_warped_np)
        save_image(Path(image_paths["image_init_warped_merge_path"]), init_merge_np)

    if init_bmv is not None and init_fmv is not None:
        init_flow_1_to_0_np = init_bmv[0].detach().cpu().permute(1, 2, 0).numpy()
        init_flow_1_to_2_np = init_fmv[0].detach().cpu().permute(1, 2, 0).numpy()
        final_flow_1_to_0_np = up_flow0_1[0].detach().cpu().permute(1, 2, 0).numpy()
        final_flow_1_to_2_np = up_flow1_1[0].detach().cpu().permute(1, 2, 0).numpy()
        save_flow_diff_visuals(
            save_dir,
            "1_to_0",
            img0_np,
            init_flow_1_to_0_np,
            final_flow_1_to_0_np,
            flow_to_image,
            save_image,
            cv2,
            np,
            flow_diff_threshold,
            flow_diff_percentile,
        )
        save_flow_diff_visuals(
            save_dir,
            "1_to_2",
            img1_np,
            init_flow_1_to_2_np,
            final_flow_1_to_2_np,
            flow_to_image,
            save_image,
            cv2,
            np,
            flow_diff_threshold,
            flow_diff_percentile,
        )

    return image_paths


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    run_config = build_inference_run_config(config_path=config_path, project_root=PROJECT_ROOT)
    validate_flow_specific_requests(
        model_name=run_config.model.model_name,
        save_topk_largest_flow_diff=run_config.save_topk_largest_flow_diff,
    )
    if run_config.mode == "dry-run":
        print(json.dumps(build_inference_dry_run_summary(config=run_config), indent=2))
        return

    metric_config = dict(run_config.metrics.values)

    run_config.output_dir.mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import Subset
    from tqdm import tqdm

    from src.data.image_ops import flow_to_image
    from src.data.image_ops import save_image
    logger = build_logger("scripts.inference")
    set_seed(run_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = build_lpips_model(metric_config, device)
    flolpips_model = build_flolpips_model(metric_config, device)
    vfips_model = build_vfips_model(metric_config, device)
    build_flip_evaluator(metric_config)
    logger.info("device=%s model=%s", device, run_config.model.model_name)
    logger.info("metrics=%s", metric_config)
    if run_config.model.eval_convex_upsampling is not None:
        logger.info("eval_convex_upsampling=%s", run_config.model.eval_convex_upsampling)
    logger.info(
        "flow_approx_method=%s splatting_fill_strategy=%s init_flow_downscale_strategy=%s init_flow_mask_epsilon=%s",
        run_config.flow_approx.method,
        run_config.flow_approx.splatting_fill_strategy,
        run_config.flow_approx.init_flow_downscale_strategy,
        run_config.flow_approx.init_flow_mask_epsilon,
    )

    dataframe_list: list[Any] = []
    for inference_preset in run_config.inference_presets:
        preset_dataframe = build_merged_dataframe(
            run_config.root_dir,
            run_config.output_dir,
            inference_preset,
            run_config.only_fps,
            logger,
        )
        preset_dataframe["inference_preset"] = inference_preset
        dataframe_list.append(preset_dataframe)

    dataframe = pd.concat(dataframe_list, ignore_index=True)
    dataframe = filter_valid_dataframe(dataframe)
    model_class = resolve_model_class(run_config.model.model_name)
    model = model_class(**run_config.model.model_init_args).to(device)
    model_parameter_summary = build_model_parameter_summary(model)
    logger.info(
        "model_param_count=%s model_param_million=%.6f",
        model_parameter_summary["model_param_count"],
        model_parameter_summary["model_param_million"],
    )
    if hasattr(model, "init_flow_layer"):
        logger.info("model_init_flow_layer=%s", model.init_flow_layer)
    # print("Load Pretrained Weights from IFRNet_Vimeo90K.pth as Baseline")
    # state_dict = torch.load("src/models/external/IFRNet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth", map_location=device)
    load_inference_checkpoint(model=model, checkpoint_path=run_config.checkpoint_path, device=device)
    if run_config.model.eval_convex_upsampling is not None:
        set_model_convex_upsampling(
            model=model,
            enabled=run_config.model.eval_convex_upsampling,
            context="inference",
        )
    model.eval()

    metric_meters = build_metric_meters(metric_config)
    rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for (inference_preset, record, mode_name), group_dataframe in dataframe.groupby(["inference_preset", "record", "mode"], sort=False):
            group_dataframe = group_dataframe.reset_index(drop=True)
            dataset = build_inference_dataset(
                dataframe=group_dataframe,
                dataset_root_dir=run_config.dataset_root_dir,
                input_fps=run_config.input_fps,
                model_name=run_config.model.model_name,
                flow_approx_method=run_config.flow_approx.method,
            )

            loader = DataLoader(dataset, batch_size=run_config.batch_size, shuffle=False)
            record_metric_meters = build_metric_meters(metric_config)
            progress = tqdm(loader, desc=f"{inference_preset}_{record}_{mode_name}", leave=True)
            sample_offset = 0
            group_rows: list[dict[str, object]] = []
            psnr_div_enabled = bool(metric_config["enable_psnr_div"])
            psnr_div_threshold = float(metric_config["psnr_div_divergence_threshold"])
            vfips_enabled = bool(metric_config["enable_vfips"])
            vfips_segments: list[VfipsSegment] = []
            previous_psnr_div_prediction_rgb = None
            pending_psnr_div_row = None
            pending_psnr_div_target_y = None
            pending_psnr_div_prediction_y = None
            pending_psnr_div_prediction_rgb = None
            export_rgb_sequence = should_export_rgb_sequence(
                config=run_config.rgb_sequence,
                record=record,
                mode_name=mode_name,
            )
            written_rgb_endpoint_frames: set[int] = set()

            for batch in progress:
                inference_result = run_inference_batch(
                    config=run_config,
                    model=model,
                    batch=batch,
                    device=device,
                )
                imgt = inference_result.imgt
                imgt_pred = inference_result.imgt_pred
                init_bmv = inference_result.init_bmv
                init_fmv = inference_result.init_fmv
                up_flow0_1 = inference_result.up_flow0_1
                up_flow1_1 = inference_result.up_flow1_1
                batch_metric_values = calculate_batch_metrics(
                    target=imgt.detach(),
                    prediction=imgt_pred.detach(),
                    metric_config=metric_config,
                    lpips_model=lpips_model,
                    flolpips_model=flolpips_model,
                    img0=inference_result.img0.detach(),
                    img1=inference_result.img1.detach(),
                )

                for batch_index in range(int(imgt_pred.shape[0])):
                    row = group_dataframe.iloc[sample_offset + batch_index]
                    frame_0_index = int(row["img0"])
                    frame_1_index = int(row["img1"])
                    frame_2_index = int(row["img2"])
                    frame_range = f"frame_{int(row['img0']):04d}_{int(row['img2']):04d}"
                    sample_metric_values = {metric_name: float(metric_values[batch_index]) for metric_name, metric_values in batch_metric_values.items()}
                    for metric_name, metric_value in sample_metric_values.items():
                        update_metric_meter(metric_meters, metric_name, metric_value)
                        update_metric_meter(record_metric_meters, metric_name, metric_value)

                    diff_1_to_0 = {"diff_mag_mean": -1.0, "diff_mag_max": -1.0, "diff_changed_ratio": -1.0, "diff_percentile_value": -1.0}
                    diff_1_to_2 = {"diff_mag_mean": -1.0, "diff_mag_max": -1.0, "diff_changed_ratio": -1.0, "diff_percentile_value": -1.0}
                    if init_bmv is not None and init_fmv is not None and up_flow0_1 is not None and up_flow1_1 is not None:
                        init_flow_1_to_0_np = init_bmv[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                        init_flow_1_to_2_np = init_fmv[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                        final_flow_1_to_0_np = up_flow0_1[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                        final_flow_1_to_2_np = up_flow1_1[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                        diff_mag_1_to_0_np = np.linalg.norm(final_flow_1_to_0_np - init_flow_1_to_0_np, axis=2)
                        diff_mag_1_to_2_np = np.linalg.norm(final_flow_1_to_2_np - init_flow_1_to_2_np, axis=2)
                        diff_1_to_0 = {
                            "diff_mag_mean": float(diff_mag_1_to_0_np.mean()),
                            "diff_mag_max": float(diff_mag_1_to_0_np.max()),
                            "diff_changed_ratio": float((diff_mag_1_to_0_np > run_config.flow_diff_threshold).mean()),
                            "diff_percentile_value": float(
                                max(np.percentile(diff_mag_1_to_0_np, run_config.flow_diff_percentile), 1e-6)
                            ),
                        }
                        diff_1_to_2 = {
                            "diff_mag_mean": float(diff_mag_1_to_2_np.mean()),
                            "diff_mag_max": float(diff_mag_1_to_2_np.max()),
                            "diff_changed_ratio": float((diff_mag_1_to_2_np > run_config.flow_diff_threshold).mean()),
                            "diff_percentile_value": float(
                                max(np.percentile(diff_mag_1_to_2_np, run_config.flow_diff_percentile), 1e-6)
                            ),
                        }

                    sample_row = {
                        "sample_index": int(sample_offset + batch_index),
                        "inference_preset": str(inference_preset),
                        "record": str(record),
                        "mode": str(mode_name),
                        "record_name": f"{record}_{mode_name}",
                        **model_parameter_summary,
                        "frame_range": frame_range,
                        "valid": bool(row["valid"]) if "valid" in row.index else True,
                        "distance_index_mean": float(row["D_index Mean"]) if "D_index Mean" in row.index else -1.0,
                        "distance_index_median": float(row["D_index Median"]) if "D_index Median" in row.index else -1.0,
                        **sample_metric_values,
                        "flow_diff_1_to_0_mean": diff_1_to_0["diff_mag_mean"],
                        "flow_diff_1_to_0_max": diff_1_to_0["diff_mag_max"],
                        "flow_diff_1_to_0_changed_ratio": diff_1_to_0["diff_changed_ratio"],
                        "flow_diff_1_to_0_percentile_value": diff_1_to_0["diff_percentile_value"],
                        "flow_diff_1_to_2_mean": diff_1_to_2["diff_mag_mean"],
                        "flow_diff_1_to_2_max": diff_1_to_2["diff_mag_max"],
                        "flow_diff_1_to_2_changed_ratio": diff_1_to_2["diff_changed_ratio"],
                        "flow_diff_1_to_2_percentile_value": diff_1_to_2["diff_percentile_value"],
                    }
                    if export_rgb_sequence:
                        save_rgb_sequence_sample(
                            config=run_config.rgb_sequence,
                            inference_preset=inference_preset,
                            record=record,
                            mode_name=mode_name,
                            frame_0_index=frame_0_index,
                            frame_1_index=frame_1_index,
                            frame_2_index=frame_2_index,
                            img0=inference_result.img0[batch_index],
                            imgt=imgt[batch_index],
                            img1=inference_result.img1[batch_index],
                            imgt_pred=imgt_pred[batch_index],
                            written_endpoint_frames=written_rgb_endpoint_frames,
                            save_image=save_image,
                            np=np,
                        )
                    if vfips_enabled:
                        append_vfips_sample(
                            vfips_segments=vfips_segments,
                            sample_index=int(sample_row["sample_index"]),
                            frame_0_index=frame_0_index,
                            frame_2_index=frame_2_index,
                            img0=inference_result.img0[batch_index],
                            imgt=imgt[batch_index],
                            img1=inference_result.img1[batch_index],
                            imgt_pred=imgt_pred[batch_index],
                        )
                    if psnr_div_enabled:
                        target_y, prediction_y, prediction_rgb = prepare_psnr_div_sample(
                            target=imgt[batch_index],
                            prediction=imgt_pred[batch_index],
                        )
                        if pending_psnr_div_row is not None:
                            psnr_div_value = calculate_psnr_div_sample(
                                target_y=pending_psnr_div_target_y,
                                prediction_y=pending_psnr_div_prediction_y,
                                flow_start_prediction_rgb=pending_psnr_div_prediction_rgb,
                                flow_end_prediction_rgb=prediction_rgb,
                                divergence_threshold=psnr_div_threshold,
                            )
                            pending_psnr_div_row[PSNR_DIV_METRIC_NAME] = psnr_div_value
                            update_metric_meter(metric_meters, PSNR_DIV_METRIC_NAME, psnr_div_value)
                            update_metric_meter(record_metric_meters, PSNR_DIV_METRIC_NAME, psnr_div_value)
                            group_rows.append(pending_psnr_div_row)
                            previous_psnr_div_prediction_rgb = pending_psnr_div_prediction_rgb

                        pending_psnr_div_row = sample_row
                        pending_psnr_div_target_y = target_y
                        pending_psnr_div_prediction_y = prediction_y
                        pending_psnr_div_prediction_rgb = prediction_rgb
                    else:
                        group_rows.append(sample_row)

                sample_offset += int(imgt_pred.shape[0])
                progress.set_postfix({"mean_psnr": f"{record_metric_meters['psnr'].avg:.6f}"})

            if psnr_div_enabled and pending_psnr_div_row is not None:
                if previous_psnr_div_prediction_rgb is None:
                    psnr_div_value = float("nan")
                else:
                    psnr_div_value = calculate_psnr_div_sample(
                        target_y=pending_psnr_div_target_y,
                        prediction_y=pending_psnr_div_prediction_y,
                        flow_start_prediction_rgb=previous_psnr_div_prediction_rgb,
                        flow_end_prediction_rgb=pending_psnr_div_prediction_rgb,
                        divergence_threshold=psnr_div_threshold,
                    )
                pending_psnr_div_row[PSNR_DIV_METRIC_NAME] = psnr_div_value
                update_metric_meter(metric_meters, PSNR_DIV_METRIC_NAME, psnr_div_value)
                update_metric_meter(record_metric_meters, PSNR_DIV_METRIC_NAME, psnr_div_value)
                group_rows.append(pending_psnr_div_row)

            if vfips_enabled:
                vfips_values_by_sample = calculate_group_vfips_values(
                    vfips_segments=vfips_segments,
                    metric_config=metric_config,
                    vfips_model=vfips_model,
                )
                apply_vfips_values_to_group_rows(
                    group_rows=group_rows,
                    values_by_sample=vfips_values_by_sample,
                    metric_meters=metric_meters,
                    record_metric_meters=record_metric_meters,
                )

            group_metrics_df = pd.DataFrame(group_rows)
            record_metric_values = average_metric_values(record_metric_meters)
            record_rows.append(
                {
                    "record": str(record),
                    "inference_preset": str(inference_preset),
                    "mode": str(mode_name),
                    "record_name": f"{record}_{mode_name}",
                    "samples": int(len(group_dataframe)),
                    **model_parameter_summary,
                    **{f"mean_{metric_name}": metric_value for metric_name, metric_value in record_metric_values.items()},
                }
            )
            selected_sample_reasons: dict[int, list[str]] = {}

            if run_config.save_topk_worst_psnr > 0:
                for sample_index in group_metrics_df.nsmallest(run_config.save_topk_worst_psnr, "psnr")["sample_index"].tolist():
                    selected_sample_reasons.setdefault(int(sample_index), []).append("worst_psnr")
            if run_config.save_topk_best_psnr > 0:
                for sample_index in group_metrics_df.nlargest(run_config.save_topk_best_psnr, "psnr")["sample_index"].tolist():
                    selected_sample_reasons.setdefault(int(sample_index), []).append("best_psnr")
            if run_config.model.model_name in (RESIDUAL_MODEL_NAME, RESIDUAL_FLOW_APPROX_MODEL_NAME) and run_config.save_topk_largest_flow_diff > 0:
                for sample_index in group_metrics_df.nlargest(run_config.save_topk_largest_flow_diff, "flow_diff_1_to_0_changed_ratio")["sample_index"].tolist():
                    selected_sample_reasons.setdefault(int(sample_index), []).append("largest_flow_diff_1_to_0")
                for sample_index in group_metrics_df.nlargest(run_config.save_topk_largest_flow_diff, "flow_diff_1_to_2_changed_ratio")["sample_index"].tolist():
                    selected_sample_reasons.setdefault(int(sample_index), []).append("largest_flow_diff_1_to_2")

            group_metrics_df["selected_for_save"] = group_metrics_df["sample_index"].map(lambda sample_index: int(sample_index) in selected_sample_reasons)
            group_metrics_df["save_reason"] = group_metrics_df["sample_index"].map(
                lambda sample_index: ";".join(selected_sample_reasons.get(int(sample_index), [])),
            )
            for column_name in SELECTED_ARTIFACT_COLUMNS:
                group_metrics_df[column_name] = ""

            selected_indices = sorted(selected_sample_reasons.keys())
            if len(selected_indices) > 0:
                selected_dataset = Subset(dataset, selected_indices)
                selected_loader = DataLoader(selected_dataset, batch_size=1, shuffle=False)
                selected_progress = tqdm(selected_loader, desc=f"save_{inference_preset}_{record}_{mode_name}", leave=True)

                for selected_batch_index, batch in enumerate(selected_progress):
                    selected_row = group_metrics_df[group_metrics_df["sample_index"] == selected_indices[selected_batch_index]].iloc[0]
                    frame_range = str(selected_row["frame_range"])
                    save_dir = run_config.output_dir / str(record) / str(mode_name) / frame_range

                    inference_result = run_inference_batch(
                        config=run_config,
                        model=model,
                        batch=batch,
                        device=device,
                    )
                    image_paths = save_selected_sample_artifacts(
                        cv2,
                        run_config.flow_diff_percentile,
                        run_config.flow_diff_threshold,
                        flow_to_image,
                        inference_result,
                        np,
                        save_dir,
                        save_image,
                    )

                    for column_name, path_value in image_paths.items():
                        group_metrics_df.loc[group_metrics_df["sample_index"] == selected_indices[selected_batch_index], column_name] = path_value

            rows.extend(group_metrics_df.to_dict("records"))
            logger.info("record=%s mode=%s samples=%s metrics=%s", record, mode_name, len(group_dataframe), format_metric_averages(record_metric_meters))

    pd.DataFrame(rows).to_csv(run_config.output_dir / "metrics.csv", index=False)
    pd.DataFrame(record_rows).to_csv(run_config.output_dir / "record_metrics.csv", index=False)
    logger.info("samples=%s metrics=%s output_dir=%s", len(rows), format_metric_averages(metric_meters), run_config.output_dir)


if __name__ == "__main__":
    main()
