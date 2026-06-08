from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_merged_dataframe
from scripts.train import set_seed
from src.engine.evaluation import build_lpips_model
from src.engine.evaluation import calculate_batch_metrics
from src.engine.evaluation import read_metric_config
from src.engine.evaluation import require_psnr_enabled
from src.engine.flow_approx import build_linear_splatting_flow_init_with_fill_strategy
from src.engine.flow_approx import build_flow_init_result
from src.engine.flow_approx import FLOW_APPROX_METHODS
from src.utils.config import load_yaml_file
from src.utils.logger import build_logger

ANALYSIS_SPLATTING_FILL_METHODS: dict[str, str] = {
    "splatting_zero_fill": "zero",
    "splatting_outside_in_4direction": "outside_in_4direction",
    "splatting_outside_in_4neighbor": "outside_in_4neighbor",
    "splatting_outside_in_8neighbor": "outside_in_8neighbor",
    "splatting_ground_truth_fill": "ground_truth",
}
ANALYSIS_FLOW_APPROX_METHODS: tuple[str, ...] = FLOW_APPROX_METHODS + tuple(ANALYSIS_SPLATTING_FILL_METHODS.keys())
GROUND_TRUTH_METHOD: str = "ground_truth"
LAYER_ANALYSIS_METHODS: tuple[str, ...] = (GROUND_TRUTH_METHOD,) + ANALYSIS_FLOW_APPROX_METHODS
METHOD_DISPLAY_NAMES: dict[str, str] = {
    GROUND_TRUTH_METHOD: "GT target flow",
    "single": "single",
    "combination": "combination",
    "splatting": "splatting (4-neighbor)",
    "splatting_zero_fill": "zero fill",
    "splatting_outside_in_4direction": "outside-in 4-direction",
    "splatting_outside_in_4neighbor": "outside-in 4-neighbor",
    "splatting_outside_in_8neighbor": "outside-in 8-neighbor",
    "splatting_ground_truth_fill": "GT hole fill",
}
METHOD_COLORS: dict[str, str] = {
    GROUND_TRUTH_METHOD: "#222222",
    "single": "#4C78A8",
    "combination": "#E45756",
    "splatting": "#59A14F",
    "splatting_zero_fill": "#9D9D9D",
    "splatting_outside_in_4direction": "#F2CF5B",
    "splatting_outside_in_4neighbor": "#54A24B",
    "splatting_outside_in_8neighbor": "#B279A2",
    "splatting_ground_truth_fill": "#D62728",
}


@dataclass(frozen=True)
class AnalysisConfig:
    mode: str
    root_dir: Path
    dataset_root_dir: Path
    analysis_presets: tuple[str, ...]
    output_dir: Path
    batch_size: int
    only_fps: int
    input_fps: int
    seed: int
    filter_valid_only: bool
    layer_scales: tuple[tuple[int, float], ...]
    layer_scatter_methods: tuple[str, ...]
    layer_bar_methods: tuple[str, ...]
    metric_config: dict[str, object]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dataset motion and flow approximation metrics.")
    parser.add_argument("--config", required=True, type=str, help="Path to one dataset analysis config file.")
    return parser.parse_args(argv)


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def parse_analysis_presets(config_payload: dict[str, Any]) -> tuple[str, ...]:
    if "analysis_presets" in config_payload:
        raw_presets = config_payload["analysis_presets"]
        if not isinstance(raw_presets, list):
            raise TypeError("analysis_presets must be a list of dataset preset names.")

        presets = tuple(str(preset) for preset in raw_presets)
        if len(presets) == 0:
            raise ValueError("analysis_presets must contain at least one dataset preset name.")

        return presets

    return (str(config_payload["analysis_preset"]),)


def parse_layer_scales(config_payload: dict[str, Any]) -> tuple[tuple[int, float], ...]:
    raw_layer_scales = config_payload["layer_scales"]
    if not isinstance(raw_layer_scales, list):
        raise TypeError("layer_scales must be a list of objects containing layer and scale.")

    layer_scales: list[tuple[int, float]] = []
    seen_layers: set[int] = set()
    for raw_layer_scale in raw_layer_scales:
        if not isinstance(raw_layer_scale, dict):
            raise TypeError(f"Each layer_scales entry must be an object: value={raw_layer_scale!r}")

        layer = int(raw_layer_scale["layer"])
        scale = float(raw_layer_scale["scale"])
        if layer in seen_layers:
            raise ValueError(f"layer_scales contains duplicate layer: layer={layer}")
        if scale <= 0.0 or scale > 1.0:
            raise ValueError(f"Layer scale must be in (0, 1]: layer={layer} scale={scale}")

        layer_scales.append((layer, scale))
        seen_layers.add(layer)

    if len(layer_scales) == 0:
        raise ValueError("layer_scales must contain at least one layer.")

    return tuple(sorted(layer_scales))


def parse_layer_methods(config_payload: dict[str, Any], config_key: str) -> tuple[str, ...]:
    raw_methods = config_payload[config_key]
    if not isinstance(raw_methods, list):
        raise TypeError(f"{config_key} must be a list of method names.")

    methods = tuple(str(method) for method in raw_methods)
    if len(methods) == 0:
        raise ValueError(f"{config_key} must contain at least one method.")
    if len(set(methods)) != len(methods):
        raise ValueError(f"{config_key} contains duplicate methods: methods={methods}")

    unknown_methods = set(methods) - set(LAYER_ANALYSIS_METHODS)
    if unknown_methods:
        raise ValueError(
            f"{config_key} contains unsupported methods: methods={sorted(unknown_methods)} "
            f"available_methods={LAYER_ANALYSIS_METHODS}",
        )

    return methods


def build_analysis_config(config_payload: dict[str, Any]) -> AnalysisConfig:
    metric_config = read_metric_config(config_values=config_payload)
    require_psnr_enabled(metric_config=metric_config, pipeline_name="dataset analysis")
    return AnalysisConfig(
        mode=str(config_payload["mode"]),
        root_dir=resolve_project_path(str(config_payload["root_dir"])),
        dataset_root_dir=resolve_project_path(str(config_payload["dataset_root_dir"])),
        analysis_presets=parse_analysis_presets(config_payload=config_payload),
        output_dir=resolve_project_path(str(config_payload["output_dir"])),
        batch_size=int(config_payload["batch_size"]),
        only_fps=int(config_payload["only_fps"]),
        input_fps=int(config_payload["input_fps"]),
        seed=int(config_payload.get("seed", 1234)),
        filter_valid_only=bool(config_payload.get("filter_valid_only", True)),
        layer_scales=parse_layer_scales(config_payload=config_payload),
        layer_scatter_methods=parse_layer_methods(
            config_payload=config_payload,
            config_key="layer_scatter_methods",
        ),
        layer_bar_methods=parse_layer_methods(
            config_payload=config_payload,
            config_key="layer_bar_methods",
        ),
        metric_config=metric_config,
    )


def build_dry_run_summary(config: AnalysisConfig) -> dict[str, object]:
    return {
        "mode": config.mode,
        "root_dir": str(config.root_dir),
        "dataset_root_dir": str(config.dataset_root_dir),
        "analysis_presets": list(config.analysis_presets),
        "output_dir": str(config.output_dir),
        "batch_size": config.batch_size,
        "only_fps": config.only_fps,
        "input_fps": config.input_fps,
        "seed": config.seed,
        "filter_valid_only": config.filter_valid_only,
        "flow_approx_methods": list(ANALYSIS_FLOW_APPROX_METHODS),
        "layer_scales": [
            {"layer": layer, "scale": scale}
            for layer, scale in config.layer_scales
        ],
        "layer_scatter_methods": list(config.layer_scatter_methods),
        "layer_bar_methods": list(config.layer_bar_methods),
        "metrics": dict(config.metric_config),
    }


def summarize_flat_values(flat_values: Any) -> dict[str, Any]:
    import torch

    return {
        "mean": flat_values.mean(dim=1),
        "max": flat_values.max(dim=1).values,
        "p95": torch.quantile(flat_values, 0.95, dim=1),
    }


def build_motion_stats(bmv_60: Any, fmv_60: Any) -> dict[str, dict[str, Any]]:
    import torch

    bmv_magnitude = bmv_60.norm(dim=1)
    fmv_magnitude = fmv_60.norm(dim=1)

    flat_bmv = bmv_magnitude.reshape(int(bmv_magnitude.shape[0]), -1)
    flat_fmv = fmv_magnitude.reshape(int(fmv_magnitude.shape[0]), -1)
    pooled_flat = torch.cat([flat_bmv, flat_fmv], dim=1)

    return {
        "bmv": summarize_flat_values(flat_values=flat_bmv),
        "fmv": summarize_flat_values(flat_values=flat_fmv),
        "pooled": summarize_flat_values(flat_values=pooled_flat),
    }


def build_bidirectional_error_stats(
    approx_bmv: Any,
    approx_fmv: Any,
    bmv_60: Any,
    fmv_60: Any,
) -> dict[str, dict[str, Any]]:
    import torch

    bmv_error_map = (approx_bmv - bmv_60).norm(dim=1)
    fmv_error_map = (approx_fmv - fmv_60).norm(dim=1)
    flat_bmv_error = bmv_error_map.reshape(int(bmv_error_map.shape[0]), -1)
    flat_fmv_error = fmv_error_map.reshape(int(fmv_error_map.shape[0]), -1)
    pooled_flat = torch.cat([flat_bmv_error, flat_fmv_error], dim=1)

    return {
        "bmv": summarize_flat_values(flat_values=flat_bmv_error),
        "fmv": summarize_flat_values(flat_values=flat_fmv_error),
        "pooled": summarize_flat_values(flat_values=pooled_flat),
    }


def build_flow_init_result_with_runtime(
    fmv_30: Any,
    bmv_30: Any,
    embt: Any,
    flow_approx_method: str,
    source_depth0: Any | None,
    source_depth1: Any | None,
    ground_truth_bmv: Any,
    ground_truth_fmv: Any,
    device: Any,
) -> tuple[Any, float]:
    import time
    import torch

    def build_result() -> Any:
        if flow_approx_method in ANALYSIS_SPLATTING_FILL_METHODS:
            if source_depth0 is None or source_depth1 is None:
                raise ValueError(f"{flow_approx_method} requires source_depth0 and source_depth1 tensors.")

            return build_linear_splatting_flow_init_with_fill_strategy(
                fmv_30=fmv_30,
                bmv_30=bmv_30,
                embt=embt,
                source_depth0=source_depth0,
                source_depth1=source_depth1,
                fill_strategy=ANALYSIS_SPLATTING_FILL_METHODS[flow_approx_method],
                ground_truth_bmv=ground_truth_bmv,
                ground_truth_fmv=ground_truth_fmv,
            )

        return build_flow_init_result(
            fmv_30=fmv_30,
            bmv_30=bmv_30,
            embt=embt,
            flow_approx_method=flow_approx_method,
            source_depth0=source_depth0,
            source_depth1=source_depth1,
        )

    if device.type != "cuda":
        start_time = time.perf_counter()
        flow_init = build_result()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return flow_init, elapsed_ms

    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    flow_init = build_result()
    end_event.record()
    torch.cuda.synchronize(device)
    return flow_init, float(start_event.elapsed_time(end_event))


def resolve_flow_fill_strategy(flow_approx_method: str) -> str:
    if flow_approx_method in ANALYSIS_SPLATTING_FILL_METHODS:
        return ANALYSIS_SPLATTING_FILL_METHODS[flow_approx_method]

    if flow_approx_method == "splatting":
        return "outside_in_4neighbor"

    return ""


def calculate_warp_metrics(
    target: Any,
    prediction: Any,
    metric_config: dict[str, object],
    lpips_model: Any | None,
) -> dict[str, list[float]]:
    return calculate_batch_metrics(
        target=target.detach(),
        prediction=prediction.detach(),
        metric_config=metric_config,
        lpips_model=lpips_model,
    )


def build_bidirectional_metric_columns(
    metric_values_img0: dict[str, list[float]],
    metric_values_img1: dict[str, list[float]],
    batch_index: int,
    suffix: str,
) -> dict[str, float]:
    if set(metric_values_img0) != set(metric_values_img1):
        raise ValueError(
            f"Bidirectional metric names must match: "
            f"img0={sorted(metric_values_img0)} img1={sorted(metric_values_img1)}",
        )

    columns: dict[str, float] = {}
    for metric_name in metric_values_img0:
        img0_value = float(metric_values_img0[metric_name][batch_index])
        img1_value = float(metric_values_img1[metric_name][batch_index])
        columns[f"warp_{metric_name}_img0{suffix}"] = img0_value
        columns[f"warp_{metric_name}_img1{suffix}"] = img1_value
        columns[f"warp_{metric_name}_mean{suffix}"] = (img0_value + img1_value) / 2.0

    return columns


def build_metric_delta_columns(
    approx_columns: dict[str, float],
    gt_columns: dict[str, float],
    metric_names: tuple[str, ...],
) -> dict[str, float]:
    columns: dict[str, float] = {}
    for metric_name in metric_names:
        for direction in ("img0", "img1", "mean"):
            approx_key = f"warp_{metric_name}_{direction}"
            gt_key = f"{approx_key}_gt"
            columns[f"warp_{metric_name}_delta_{direction}_vs_gt"] = approx_columns[approx_key] - gt_columns[gt_key]

    return columns


def resize_image_for_layer(image: Any, scale: float) -> Any:
    import torch.nn.functional as functional

    return functional.interpolate(
        image,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
    )


def resize_flow_for_layer(flow: Any, scale: float) -> Any:
    return resize_image_for_layer(image=flow, scale=scale) * scale


def build_layer_contexts(
    img0: Any,
    imgt: Any,
    img1: Any,
    bmv_60: Any,
    fmv_60: Any,
    layer_scales: tuple[tuple[int, float], ...],
    metric_config: dict[str, object],
    lpips_model: Any | None,
    warp_fn: Any,
) -> dict[int, dict[str, Any]]:
    contexts: dict[int, dict[str, Any]] = {}
    for layer, scale in layer_scales:
        layer_img0 = resize_image_for_layer(image=img0, scale=scale)
        layer_imgt = resize_image_for_layer(image=imgt, scale=scale)
        layer_img1 = resize_image_for_layer(image=img1, scale=scale)
        layer_bmv_60 = resize_flow_for_layer(flow=bmv_60, scale=scale)
        layer_fmv_60 = resize_flow_for_layer(flow=fmv_60, scale=scale)
        gt_img0_warped = warp_fn(layer_img0, layer_bmv_60)
        gt_img1_warped = warp_fn(layer_img1, layer_fmv_60)
        contexts[layer] = {
            "scale": scale,
            "img0": layer_img0,
            "imgt": layer_imgt,
            "img1": layer_img1,
            "bmv_60": layer_bmv_60,
            "fmv_60": layer_fmv_60,
            "gt_img0_warp_metrics": calculate_warp_metrics(
                target=layer_imgt,
                prediction=gt_img0_warped,
                metric_config=metric_config,
                lpips_model=lpips_model,
            ),
            "gt_img1_warp_metrics": calculate_warp_metrics(
                target=layer_imgt,
                prediction=gt_img1_warped,
                metric_config=metric_config,
                lpips_model=lpips_model,
            ),
        }

    return contexts


def build_method_layer_rows(
    base_records: list[dict[str, object]],
    method: str,
    approx_bmv_full: Any,
    approx_fmv_full: Any,
    layer_contexts: dict[int, dict[str, Any]],
    metric_config: dict[str, object],
    lpips_model: Any | None,
    warp_fn: Any,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer, context in layer_contexts.items():
        scale = float(context["scale"])
        approx_bmv = resize_flow_for_layer(flow=approx_bmv_full, scale=scale)
        approx_fmv = resize_flow_for_layer(flow=approx_fmv_full, scale=scale)
        error_stats = build_bidirectional_error_stats(
            approx_bmv=approx_bmv,
            approx_fmv=approx_fmv,
            bmv_60=context["bmv_60"],
            fmv_60=context["fmv_60"],
        )
        approx_img0_warped = warp_fn(context["img0"], approx_bmv)
        approx_img1_warped = warp_fn(context["img1"], approx_fmv)
        approx_img0_warp_metrics = calculate_warp_metrics(
            target=context["imgt"],
            prediction=approx_img0_warped,
            metric_config=metric_config,
            lpips_model=lpips_model,
        )
        approx_img1_warp_metrics = calculate_warp_metrics(
            target=context["imgt"],
            prediction=approx_img1_warped,
            metric_config=metric_config,
            lpips_model=lpips_model,
        )

        for batch_index, base_record in enumerate(base_records):
            epe_layer_pixels = float(error_stats["pooled"]["mean"][batch_index].detach().cpu().item())
            gt_metric_columns = build_bidirectional_metric_columns(
                metric_values_img0=context["gt_img0_warp_metrics"],
                metric_values_img1=context["gt_img1_warp_metrics"],
                batch_index=batch_index,
                suffix="_gt",
            )
            approx_metric_columns = build_bidirectional_metric_columns(
                metric_values_img0=approx_img0_warp_metrics,
                metric_values_img1=approx_img1_warp_metrics,
                batch_index=batch_index,
                suffix="",
            )
            metric_delta_columns = build_metric_delta_columns(
                approx_columns=approx_metric_columns,
                gt_columns=gt_metric_columns,
                metric_names=tuple(approx_img0_warp_metrics),
            )
            rows.append(
                {
                    **base_record,
                    "layer": layer,
                    "scale": scale,
                    "method": method,
                    "epe_layer_pixels": epe_layer_pixels,
                    "epe_fullres_equivalent": epe_layer_pixels / scale,
                    **gt_metric_columns,
                    **approx_metric_columns,
                    **metric_delta_columns,
                }
            )

    return rows


def build_sample_base_record(row: Any, sample_index: int, analysis_preset: str) -> dict[str, object]:
    record = str(row["record"])
    mode = str(row["mode"])
    return {
        "sample_index": sample_index,
        "analysis_preset": analysis_preset,
        "record": record,
        "mode": mode,
        "record_name": f"{record}_{mode}",
        "frame_0": int(row["img0"]),
        "frame_1": int(row["img1"]),
        "frame_2": int(row["img2"]),
        "frame_range": f"frame_{int(row['img0']):04d}_{int(row['img2']):04d}",
        "valid": bool(row["valid"]) if "valid" in row.index else True,
        "distance_index_mean": float(row["D_index Mean"]) if "D_index Mean" in row.index else -1.0,
        "distance_index_median": float(row["D_index Median"]) if "D_index Median" in row.index else -1.0,
    }


def build_summary_dataframe(dataframe: Any, group_columns: list[str]) -> Any:
    import pandas as pd

    excluded_columns = set(group_columns)
    excluded_columns.update({"sample_index", "frame_0", "frame_1", "frame_2", "frame_range"})
    numeric_columns = [
        column_name
        for column_name in dataframe.columns
        if column_name not in excluded_columns and pd.api.types.is_numeric_dtype(dataframe[column_name])
    ]

    aggregated = (
        dataframe.groupby(group_columns, dropna=False)[numeric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    flattened_columns: list[str] = []
    for column_name in aggregated.columns:
        if not isinstance(column_name, tuple):
            flattened_columns.append(str(column_name))
            continue

        base_name, stat_name = column_name
        if stat_name == "":
            flattened_columns.append(str(base_name))
            continue

        flattened_columns.append(f"{base_name}_{stat_name}")

    aggregated.columns = flattened_columns
    sample_counts = dataframe.groupby(group_columns, dropna=False).size().reset_index(name="sample_count")
    summary = sample_counts.merge(aggregated, on=group_columns, how="left")
    return summary.sort_values(group_columns).reset_index(drop=True)


def build_record_plot_label(record: str, mode: str) -> str:
    mode_parts = mode.replace("\\", "/").split("/")
    scene_name = mode_parts[1] if len(mode_parts) > 1 else mode_parts[0]
    return f"{record}/{scene_name}"


def plot_layer_psnr_epe_scatter(
    layer_dataframe: Any,
    analysis_preset: str,
    layer_scales: tuple[tuple[int, float], ...],
    methods: tuple[str, ...],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    preset_dataframe = layer_dataframe[
        (layer_dataframe["analysis_preset"] == analysis_preset)
        & (layer_dataframe["method"].isin(methods))
    ]
    summary = (
        preset_dataframe.groupby(["layer", "scale", "method"], as_index=False)
        .agg(
            warp_psnr_mean=("warp_psnr_mean", "mean"),
            epe_fullres_equivalent=("epe_fullres_equivalent", "mean"),
        )
    )
    column_count = 2
    row_count = int(np.ceil(len(layer_scales) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(14, 5.5 * row_count),
        squeeze=False,
    )

    for axis_index, (layer, scale) in enumerate(layer_scales):
        axis = axes.flat[axis_index]
        layer_summary = summary[summary["layer"] == layer]
        for method in methods:
            method_row = layer_summary[layer_summary["method"] == method]
            if len(method_row) != 1:
                raise ValueError(
                    f"Expected one scatter summary row: preset={analysis_preset} "
                    f"layer={layer} method={method} rows={len(method_row)}",
                )

            psnr = float(method_row.iloc[0]["warp_psnr_mean"])
            epe = float(method_row.iloc[0]["epe_fullres_equivalent"])
            axis.scatter(
                psnr,
                epe,
                s=75,
                color=METHOD_COLORS[method],
                edgecolors="white",
                linewidths=0.7,
                label=METHOD_DISPLAY_NAMES[method],
                zorder=3,
            )
            axis.annotate(
                METHOD_DISPLAY_NAMES[method],
                (psnr, epe),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        axis.set_title(f"Layer {layer} (scale={scale:g})")
        axis.set_xlabel("Mean warped RGB PSNR (dB)")
        axis.set_ylabel("Mean EPE (full-resolution-equivalent pixels)")
        axis.grid(alpha=0.25)

    for unused_axis_index in range(len(layer_scales), row_count * column_count):
        axes.flat[unused_axis_index].set_visible(False)

    figure.suptitle(f"{analysis_preset}: Flow Approximation by IFRNet Layer", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_record_metric_by_layer(
    layer_dataframe: Any,
    analysis_preset: str,
    layer_scales: tuple[tuple[int, float], ...],
    methods: tuple[str, ...],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    preset_dataframe = layer_dataframe[
        (layer_dataframe["analysis_preset"] == analysis_preset)
        & (layer_dataframe["method"].isin(methods))
    ].copy()
    preset_dataframe["record_label"] = [
        build_record_plot_label(record=str(record), mode=str(mode))
        for record, mode in zip(
            preset_dataframe["record"],
            preset_dataframe["mode"],
            strict=True,
        )
    ]
    summary = (
        preset_dataframe.groupby(
            ["layer", "scale", "record_label", "method"],
            as_index=False,
        )
        .agg(metric_mean=(metric, "mean"))
    )
    record_labels = list(dict.fromkeys(summary["record_label"].tolist()))
    x_positions = np.arange(len(record_labels), dtype=np.float32)
    bar_width = min(0.8 / float(len(methods)), 0.18)
    figure, axes = plt.subplots(
        len(layer_scales),
        1,
        figsize=(max(13.0, len(record_labels) * 1.55), 4.2 * len(layer_scales)),
        squeeze=False,
    )

    for axis_index, (layer, scale) in enumerate(layer_scales):
        axis = axes[axis_index, 0]
        layer_summary = summary[summary["layer"] == layer]
        for method_index, method in enumerate(methods):
            method_summary = (
                layer_summary[layer_summary["method"] == method]
                .set_index("record_label")
                .reindex(record_labels)
            )
            if method_summary["metric_mean"].isna().any():
                missing_records = method_summary[method_summary["metric_mean"].isna()].index.tolist()
                raise ValueError(
                    f"Missing record metrics: preset={analysis_preset} layer={layer} "
                    f"method={method} records={missing_records}",
                )

            offsets = (
                x_positions
                + (method_index - (len(methods) - 1) / 2.0) * bar_width
            )
            axis.bar(
                offsets,
                method_summary["metric_mean"].to_numpy(),
                width=bar_width,
                color=METHOD_COLORS[method],
                label=METHOD_DISPLAY_NAMES[method],
            )

        axis.set_title(f"Layer {layer} (scale={scale:g})")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(record_labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if axis_index == 0:
            axis.legend(ncol=min(3, len(methods)), fontsize=9)

    figure.suptitle(f"{analysis_preset}: {title}", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_layer_analysis_outputs(
    layer_dataframe: Any,
    config: AnalysisConfig,
) -> None:
    layer_record_dataframe = build_summary_dataframe(
        dataframe=layer_dataframe,
        group_columns=[
            "analysis_preset",
            "record",
            "mode",
            "record_name",
            "layer",
            "scale",
            "method",
        ],
    )
    layer_global_dataframe = build_summary_dataframe(
        dataframe=layer_dataframe,
        group_columns=["analysis_preset", "layer", "scale", "method"],
    )
    layer_dataframe.to_csv(config.output_dir / "flow_approx_by_layer_sample.csv", index=False)
    layer_record_dataframe.to_csv(config.output_dir / "flow_approx_by_layer_record.csv", index=False)
    layer_global_dataframe.to_csv(config.output_dir / "flow_approx_by_layer_summary.csv", index=False)

    for analysis_preset in config.analysis_presets:
        plot_layer_psnr_epe_scatter(
            layer_dataframe=layer_dataframe,
            analysis_preset=analysis_preset,
            layer_scales=config.layer_scales,
            methods=config.layer_scatter_methods,
            output_path=config.output_dir / f"{analysis_preset}_layer_psnr_epe_scatter.png",
        )
        plot_record_metric_by_layer(
            layer_dataframe=layer_dataframe,
            analysis_preset=analysis_preset,
            layer_scales=config.layer_scales,
            methods=config.layer_bar_methods,
            metric="warp_psnr_mean",
            ylabel="Mean warped RGB PSNR (dB)",
            title="Warped RGB PSNR by Record and Layer",
            output_path=config.output_dir / f"{analysis_preset}_record_layer_psnr.png",
        )
        plot_record_metric_by_layer(
            layer_dataframe=layer_dataframe,
            analysis_preset=analysis_preset,
            layer_scales=config.layer_scales,
            methods=config.layer_bar_methods,
            metric="epe_fullres_equivalent",
            ylabel="Mean EPE (full-resolution-equivalent pixels)",
            title="Flow EPE by Record and Layer",
            output_path=config.output_dir / f"{analysis_preset}_record_layer_epe.png",
        )
        if bool(config.metric_config["enable_ssim"]):
            plot_record_metric_by_layer(
                layer_dataframe=layer_dataframe,
                analysis_preset=analysis_preset,
                layer_scales=config.layer_scales,
                methods=config.layer_bar_methods,
                metric="warp_ssim_mean",
                ylabel="Mean warped RGB SSIM (higher is better)",
                title="Warped RGB SSIM by Record and Layer",
                output_path=config.output_dir / f"{analysis_preset}_record_layer_ssim.png",
            )
        if bool(config.metric_config["enable_lpips"]):
            plot_record_metric_by_layer(
                layer_dataframe=layer_dataframe,
                analysis_preset=analysis_preset,
                layer_scales=config.layer_scales,
                methods=config.layer_bar_methods,
                metric="warp_lpips_mean",
                ylabel="Mean warped RGB LPIPS (lower is better)",
                title="Warped RGB LPIPS by Record and Layer",
                output_path=config.output_dir / f"{analysis_preset}_record_layer_lpips.png",
            )


def analyze_dataset(config: AnalysisConfig) -> None:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.data.dataset_loader import FlowEstimationTrainDataset
    from src.models.external.IFRNet.utils import warp

    logger = build_logger("scripts.analyze_dataset")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = build_lpips_model(metric_config=config.metric_config, device=device)
    logger.info("device=%s analysis_presets=%s", device, config.analysis_presets)
    logger.info("metrics=%s", config.metric_config)

    motion_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []

    for analysis_preset in config.analysis_presets:
        merged_dataframe = build_merged_dataframe(
            root_dir=config.root_dir,
            checkpoints_dir=config.output_dir,
            dataset_preset_name=analysis_preset,
            only_fps=config.only_fps,
            logger=logger,
        )

        if config.filter_valid_only and "valid" in merged_dataframe.columns:
            logger.info(
                "analysis_preset=%s valid_count=%s",
                analysis_preset,
                merged_dataframe["valid"].value_counts().to_dict(),
            )
            merged_dataframe = merged_dataframe[merged_dataframe["valid"] == True].reset_index(drop=True)
        else:
            merged_dataframe = merged_dataframe.reset_index(drop=True)

        dataset = FlowEstimationTrainDataset(
            dataframe=merged_dataframe,
            dataset_root_dir=str(config.dataset_root_dir),
            input_fps=config.input_fps,
            augment=False,
            include_source_depths=True,
        )
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        sample_offset = 0

        with torch.no_grad():
            progress = tqdm(loader, desc=f"Analyzing {analysis_preset}", leave=True)
            for batch in progress:
                img0, imgt, img1, bmv_60, fmv_60, bmv_30, fmv_30, embt, info = batch
                img0 = img0.to(device)
                imgt = imgt.to(device)
                img1 = img1.to(device)
                bmv_60 = bmv_60.to(device)
                fmv_60 = fmv_60.to(device)
                bmv_30 = bmv_30.to(device)
                fmv_30 = fmv_30.to(device)
                embt = embt.to(device)
                source_depth0 = info["source_depth0"].to(device)
                source_depth1 = info["source_depth1"].to(device)

                batch_size = int(img0.shape[0])
                batch_dataframe = merged_dataframe.iloc[sample_offset : sample_offset + batch_size].reset_index(drop=True)
                base_records = [
                    build_sample_base_record(
                        row=batch_dataframe.iloc[batch_index],
                        sample_index=sample_offset + batch_index,
                        analysis_preset=analysis_preset,
                    )
                    for batch_index in range(batch_size)
                ]

                motion_stats = build_motion_stats(bmv_60=bmv_60, fmv_60=fmv_60)
                gt_img0_warped = warp(img0, bmv_60)
                gt_img1_warped = warp(img1, fmv_60)
                gt_img0_warp_metrics = calculate_warp_metrics(
                    target=imgt,
                    prediction=gt_img0_warped,
                    metric_config=config.metric_config,
                    lpips_model=lpips_model,
                )
                gt_img1_warp_metrics = calculate_warp_metrics(
                    target=imgt,
                    prediction=gt_img1_warped,
                    metric_config=config.metric_config,
                    lpips_model=lpips_model,
                )
                layer_contexts = build_layer_contexts(
                    img0=img0,
                    imgt=imgt,
                    img1=img1,
                    bmv_60=bmv_60,
                    fmv_60=fmv_60,
                    layer_scales=config.layer_scales,
                    metric_config=config.metric_config,
                    lpips_model=lpips_model,
                    warp_fn=warp,
                )
                layer_rows.extend(
                    build_method_layer_rows(
                        base_records=base_records,
                        method=GROUND_TRUTH_METHOD,
                        approx_bmv_full=bmv_60,
                        approx_fmv_full=fmv_60,
                        layer_contexts=layer_contexts,
                        metric_config=config.metric_config,
                        lpips_model=lpips_model,
                        warp_fn=warp,
                    )
                )

                method_metrics: dict[str, dict[str, Any]] = {}
                for flow_approx_method in ANALYSIS_FLOW_APPROX_METHODS:
                    flow_init, flow_init_runtime_ms = build_flow_init_result_with_runtime(
                        fmv_30=fmv_30,
                        bmv_30=bmv_30,
                        embt=embt,
                        flow_approx_method=flow_approx_method,
                        source_depth0=source_depth0,
                        source_depth1=source_depth1,
                        ground_truth_bmv=bmv_60,
                        ground_truth_fmv=fmv_60,
                        device=device,
                    )
                    approx_bmv = flow_init.bmv
                    approx_fmv = flow_init.fmv
                    error_stats = build_bidirectional_error_stats(
                        approx_bmv=approx_bmv,
                        approx_fmv=approx_fmv,
                        bmv_60=bmv_60,
                        fmv_60=fmv_60,
                    )
                    approx_img0_warped = warp(img0, approx_bmv)
                    approx_img1_warped = warp(img1, approx_fmv)
                    approx_img0_warp_metrics = calculate_warp_metrics(
                        target=imgt,
                        prediction=approx_img0_warped,
                        metric_config=config.metric_config,
                        lpips_model=lpips_model,
                    )
                    approx_img1_warp_metrics = calculate_warp_metrics(
                        target=imgt,
                        prediction=approx_img1_warped,
                        metric_config=config.metric_config,
                        lpips_model=lpips_model,
                    )
                    layer_rows.extend(
                        build_method_layer_rows(
                            base_records=base_records,
                            method=flow_approx_method,
                            approx_bmv_full=approx_bmv,
                            approx_fmv_full=approx_fmv,
                            layer_contexts=layer_contexts,
                            metric_config=config.metric_config,
                            lpips_model=lpips_model,
                            warp_fn=warp,
                        )
                    )
                    method_metrics[flow_approx_method] = {
                        "error_stats": error_stats,
                        "img0_warp_metrics": approx_img0_warp_metrics,
                        "img1_warp_metrics": approx_img1_warp_metrics,
                        "masks": flow_init.masks,
                        "flow_init_runtime_ms": flow_init_runtime_ms,
                    }

                for batch_index in range(batch_size):
                    base_record = base_records[batch_index]
                    gt_metric_columns = build_bidirectional_metric_columns(
                        metric_values_img0=gt_img0_warp_metrics,
                        metric_values_img1=gt_img1_warp_metrics,
                        batch_index=batch_index,
                        suffix="_gt60",
                    )
                    motion_rows.append(
                        {
                            **base_record,
                            "motion_bmv_mean": float(motion_stats["bmv"]["mean"][batch_index].detach().cpu().item()),
                            "motion_bmv_max": float(motion_stats["bmv"]["max"][batch_index].detach().cpu().item()),
                            "motion_bmv_p95": float(motion_stats["bmv"]["p95"][batch_index].detach().cpu().item()),
                            "motion_fmv_mean": float(motion_stats["fmv"]["mean"][batch_index].detach().cpu().item()),
                            "motion_fmv_max": float(motion_stats["fmv"]["max"][batch_index].detach().cpu().item()),
                            "motion_fmv_p95": float(motion_stats["fmv"]["p95"][batch_index].detach().cpu().item()),
                            "motion_pooled_mean": float(motion_stats["pooled"]["mean"][batch_index].detach().cpu().item()),
                            "motion_pooled_max": float(motion_stats["pooled"]["max"][batch_index].detach().cpu().item()),
                            "motion_pooled_p95": float(motion_stats["pooled"]["p95"][batch_index].detach().cpu().item()),
                            **gt_metric_columns,
                        }
                    )

                    for flow_approx_method in ANALYSIS_FLOW_APPROX_METHODS:
                        flow_method_metric = method_metrics[flow_approx_method]
                        error_stats = flow_method_metric["error_stats"]
                        approx_metric_columns = build_bidirectional_metric_columns(
                            metric_values_img0=flow_method_metric["img0_warp_metrics"],
                            metric_values_img1=flow_method_metric["img1_warp_metrics"],
                            batch_index=batch_index,
                            suffix="_approx",
                        )
                        metric_delta_columns: dict[str, float] = {}
                        for metric_name in flow_method_metric["img0_warp_metrics"]:
                            for direction in ("img0", "img1", "mean"):
                                approx_key = f"warp_{metric_name}_{direction}_approx"
                                gt_key = f"warp_{metric_name}_{direction}_gt60"
                                metric_delta_columns[f"warp_{metric_name}_delta_{direction}_vs_gt60"] = (
                                    approx_metric_columns[approx_key] - gt_metric_columns[gt_key]
                                )
                        masks = flow_method_metric["masks"]
                        flow_init_runtime_ms = float(flow_method_metric["flow_init_runtime_ms"])
                        flow_init_runtime_ms_per_sample = flow_init_runtime_ms / float(batch_size)
                        coverage_bmv = 1.0
                        coverage_fmv = 1.0
                        if masks is not None:
                            coverage_bmv = float(masks[batch_index, 0].detach().cpu().mean().item())
                            coverage_fmv = float(masks[batch_index, 1].detach().cpu().mean().item())

                        flow_rows.append(
                            {
                                **base_record,
                                "method": flow_approx_method,
                                "splatting_fill_strategy": resolve_flow_fill_strategy(flow_approx_method=flow_approx_method),
                                "flow_init_runtime_ms": flow_init_runtime_ms_per_sample,
                                "flow_init_batch_runtime_ms": flow_init_runtime_ms,
                                "motion_pooled_mean": float(motion_stats["pooled"]["mean"][batch_index].detach().cpu().item()),
                                "motion_pooled_p95": float(motion_stats["pooled"]["p95"][batch_index].detach().cpu().item()),
                                "approx_error_bmv_mean": float(error_stats["bmv"]["mean"][batch_index].detach().cpu().item()),
                                "approx_error_bmv_max": float(error_stats["bmv"]["max"][batch_index].detach().cpu().item()),
                                "approx_error_bmv_p95": float(error_stats["bmv"]["p95"][batch_index].detach().cpu().item()),
                                "approx_error_fmv_mean": float(error_stats["fmv"]["mean"][batch_index].detach().cpu().item()),
                                "approx_error_fmv_max": float(error_stats["fmv"]["max"][batch_index].detach().cpu().item()),
                                "approx_error_fmv_p95": float(error_stats["fmv"]["p95"][batch_index].detach().cpu().item()),
                                "approx_error_pooled_mean": float(error_stats["pooled"]["mean"][batch_index].detach().cpu().item()),
                                "approx_error_pooled_max": float(error_stats["pooled"]["max"][batch_index].detach().cpu().item()),
                                "approx_error_pooled_p95": float(error_stats["pooled"]["p95"][batch_index].detach().cpu().item()),
                                **gt_metric_columns,
                                **approx_metric_columns,
                                "coverage_bmv": coverage_bmv,
                                "coverage_fmv": coverage_fmv,
                                "coverage_mean": (coverage_bmv + coverage_fmv) / 2.0,
                                **metric_delta_columns,
                            }
                        )

                sample_offset += batch_size

        logger.info("analysis_preset=%s samples=%s", analysis_preset, len(merged_dataframe))

    motion_dataframe = pd.DataFrame(motion_rows)
    flow_dataframe = pd.DataFrame(flow_rows)
    layer_dataframe = pd.DataFrame(layer_rows)
    motion_summary_dataframe = build_summary_dataframe(
        dataframe=motion_dataframe,
        group_columns=["analysis_preset", "record", "mode", "record_name"],
    )
    flow_summary_dataframe = build_summary_dataframe(
        dataframe=flow_dataframe,
        group_columns=["analysis_preset", "record", "mode", "record_name", "method"],
    )

    motion_dataframe.to_csv(config.output_dir / "motion_by_sample.csv", index=False)
    flow_dataframe.to_csv(config.output_dir / "flow_approx_by_sample.csv", index=False)
    motion_summary_dataframe.to_csv(config.output_dir / "motion_by_record.csv", index=False)
    flow_summary_dataframe.to_csv(config.output_dir / "flow_approx_by_record.csv", index=False)
    save_layer_analysis_outputs(layer_dataframe=layer_dataframe, config=config)

    logger.info(
        "motion_rows=%s flow_rows=%s layer_rows=%s output_dir=%s",
        len(motion_dataframe),
        len(flow_dataframe),
        len(layer_dataframe),
        config.output_dir,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = resolve_project_path(str(args.config))
    config_payload = load_yaml_file(config_path=config_path)
    config = build_analysis_config(config_payload=config_payload)

    if config.mode == "dry-run":
        print(json.dumps(build_dry_run_summary(config=config), indent=2))
        return

    if config.mode != "analyze":
        raise ValueError(f"Unsupported mode: {config.mode}")

    analyze_dataset(config=config)


if __name__ == "__main__":
    main()
