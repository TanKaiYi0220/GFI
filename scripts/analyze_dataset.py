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
from scripts.train_flow_approx import FLOW_APPROX_METHODS
from scripts.train_flow_approx import build_flow_init
from src.utils.config import load_yaml_file
from src.utils.logger import build_logger


@dataclass(frozen=True)
class AnalysisConfig:
    mode: str
    root_dir: Path
    dataset_root_dir: Path
    analysis_preset: str
    output_dir: Path
    batch_size: int
    only_fps: int
    input_fps: int
    seed: int
    filter_valid_only: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dataset motion and flow approximation metrics.")
    parser.add_argument("--config", required=True, type=str, help="Path to one dataset analysis config file.")
    return parser.parse_args(argv)


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def build_analysis_config(config_payload: dict[str, Any]) -> AnalysisConfig:
    return AnalysisConfig(
        mode=str(config_payload["mode"]),
        root_dir=resolve_project_path(str(config_payload["root_dir"])),
        dataset_root_dir=resolve_project_path(str(config_payload["dataset_root_dir"])),
        analysis_preset=str(config_payload["analysis_preset"]),
        output_dir=resolve_project_path(str(config_payload["output_dir"])),
        batch_size=int(config_payload["batch_size"]),
        only_fps=int(config_payload["only_fps"]),
        input_fps=int(config_payload["input_fps"]),
        seed=int(config_payload.get("seed", 1234)),
        filter_valid_only=bool(config_payload.get("filter_valid_only", True)),
    )


def build_dry_run_summary(config: AnalysisConfig) -> dict[str, object]:
    return {
        "mode": config.mode,
        "root_dir": str(config.root_dir),
        "dataset_root_dir": str(config.dataset_root_dir),
        "analysis_preset": config.analysis_preset,
        "output_dir": str(config.output_dir),
        "batch_size": config.batch_size,
        "only_fps": config.only_fps,
        "input_fps": config.input_fps,
        "seed": config.seed,
        "filter_valid_only": config.filter_valid_only,
        "flow_approx_methods": list(FLOW_APPROX_METHODS),
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


def calculate_batch_psnr(target: Any, prediction: Any, calculate_psnr_fn: Any) -> list[float]:
    batch_size = int(target.shape[0])
    psnr_values: list[float] = []

    for batch_index in range(batch_size):
        psnr_value = float(calculate_psnr_fn(target[batch_index], prediction[batch_index]).detach().cpu().item())
        psnr_values.append(psnr_value)

    return psnr_values


def build_sample_base_record(row: Any, sample_index: int) -> dict[str, object]:
    record = str(row["record"])
    mode = str(row["mode"])
    return {
        "sample_index": sample_index,
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


def analyze_dataset(config: AnalysisConfig) -> None:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.data.dataset_loader import FlowEstimationTrainDataset
    from src.engine.evaluation import calculate_psnr
    from src.models.external.IFRNet.utils import warp

    logger = build_logger("scripts.analyze_dataset")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s analysis_preset=%s", device, config.analysis_preset)

    merged_dataframe = build_merged_dataframe(
        root_dir=config.root_dir,
        checkpoints_dir=config.output_dir,
        dataset_preset_name=config.analysis_preset,
        only_fps=config.only_fps,
        logger=logger,
    )

    if config.filter_valid_only and "valid" in merged_dataframe.columns:
        logger.info("valid_count=%s", merged_dataframe["valid"].value_counts().to_dict())
        merged_dataframe = merged_dataframe[merged_dataframe["valid"] == True].reset_index(drop=True)
    else:
        merged_dataframe = merged_dataframe.reset_index(drop=True)

    dataset = FlowEstimationTrainDataset(
        dataframe=merged_dataframe,
        dataset_root_dir=str(config.dataset_root_dir),
        input_fps=config.input_fps,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    motion_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    sample_offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Analyzing dataset", leave=True):
            img0, imgt, img1, bmv_60, fmv_60, bmv_30, fmv_30, embt, _info = batch
            img0 = img0.to(device)
            imgt = imgt.to(device)
            img1 = img1.to(device)
            bmv_60 = bmv_60.to(device)
            fmv_60 = fmv_60.to(device)
            bmv_30 = bmv_30.to(device)
            fmv_30 = fmv_30.to(device)
            embt = embt.to(device)

            batch_size = int(img0.shape[0])
            batch_dataframe = merged_dataframe.iloc[sample_offset : sample_offset + batch_size].reset_index(drop=True)

            motion_stats = build_motion_stats(bmv_60=bmv_60, fmv_60=fmv_60)
            gt_img0_warped = warp(img0, bmv_60)
            gt_img1_warped = warp(img1, fmv_60)
            gt_img0_warp_psnr = calculate_batch_psnr(target=imgt, prediction=gt_img0_warped, calculate_psnr_fn=calculate_psnr)
            gt_img1_warp_psnr = calculate_batch_psnr(target=imgt, prediction=gt_img1_warped, calculate_psnr_fn=calculate_psnr)

            method_metrics: dict[str, dict[str, Any]] = {}
            for flow_approx_method in FLOW_APPROX_METHODS:
                approx_bmv, approx_fmv = build_flow_init(
                    fmv_30=fmv_30,
                    bmv_30=bmv_30,
                    embt=embt,
                    flow_approx_method=flow_approx_method,
                )
                error_stats = build_bidirectional_error_stats(
                    approx_bmv=approx_bmv,
                    approx_fmv=approx_fmv,
                    bmv_60=bmv_60,
                    fmv_60=fmv_60,
                )
                approx_img0_warped = warp(img0, approx_bmv)
                approx_img1_warped = warp(img1, approx_fmv)
                approx_img0_warp_psnr = calculate_batch_psnr(
                    target=imgt,
                    prediction=approx_img0_warped,
                    calculate_psnr_fn=calculate_psnr,
                )
                approx_img1_warp_psnr = calculate_batch_psnr(
                    target=imgt,
                    prediction=approx_img1_warped,
                    calculate_psnr_fn=calculate_psnr,
                )
                method_metrics[flow_approx_method] = {
                    "error_stats": error_stats,
                    "img0_warp_psnr": approx_img0_warp_psnr,
                    "img1_warp_psnr": approx_img1_warp_psnr,
                }

            for batch_index in range(batch_size):
                row = batch_dataframe.iloc[batch_index]
                base_record = build_sample_base_record(row=row, sample_index=sample_offset + batch_index)

                gt_warp_psnr_mean = (gt_img0_warp_psnr[batch_index] + gt_img1_warp_psnr[batch_index]) / 2.0
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
                        "warp_psnr_img0_gt60": gt_img0_warp_psnr[batch_index],
                        "warp_psnr_img1_gt60": gt_img1_warp_psnr[batch_index],
                        "warp_psnr_mean_gt60": gt_warp_psnr_mean,
                    }
                )

                for flow_approx_method in FLOW_APPROX_METHODS:
                    flow_method_metric = method_metrics[flow_approx_method]
                    error_stats = flow_method_metric["error_stats"]
                    approx_img0_psnr = float(flow_method_metric["img0_warp_psnr"][batch_index])
                    approx_img1_psnr = float(flow_method_metric["img1_warp_psnr"][batch_index])
                    approx_mean_psnr = (approx_img0_psnr + approx_img1_psnr) / 2.0

                    flow_rows.append(
                        {
                            **base_record,
                            "method": flow_approx_method,
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
                            "warp_psnr_img0_gt60": gt_img0_warp_psnr[batch_index],
                            "warp_psnr_img1_gt60": gt_img1_warp_psnr[batch_index],
                            "warp_psnr_mean_gt60": gt_warp_psnr_mean,
                            "warp_psnr_img0_approx": approx_img0_psnr,
                            "warp_psnr_img1_approx": approx_img1_psnr,
                            "warp_psnr_mean_approx": approx_mean_psnr,
                            "warp_psnr_delta_img0_vs_gt60": approx_img0_psnr - gt_img0_warp_psnr[batch_index],
                            "warp_psnr_delta_img1_vs_gt60": approx_img1_psnr - gt_img1_warp_psnr[batch_index],
                            "warp_psnr_delta_mean_vs_gt60": approx_mean_psnr - gt_warp_psnr_mean,
                        }
                    )

            sample_offset += batch_size

    motion_dataframe = pd.DataFrame(motion_rows)
    flow_dataframe = pd.DataFrame(flow_rows)
    motion_summary_dataframe = build_summary_dataframe(
        dataframe=motion_dataframe,
        group_columns=["record", "mode", "record_name"],
    )
    flow_summary_dataframe = build_summary_dataframe(
        dataframe=flow_dataframe,
        group_columns=["record", "mode", "record_name", "method"],
    )

    motion_dataframe.to_csv(config.output_dir / "motion_by_sample.csv", index=False)
    flow_dataframe.to_csv(config.output_dir / "flow_approx_by_sample.csv", index=False)
    motion_summary_dataframe.to_csv(config.output_dir / "motion_by_record.csv", index=False)
    flow_summary_dataframe.to_csv(config.output_dir / "flow_approx_by_record.csv", index=False)

    logger.info("motion_rows=%s flow_rows=%s output_dir=%s", len(motion_dataframe), len(flow_dataframe), config.output_dir)


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
