from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_merged_dataframe
from src.data.dataset_config import list_dataset_presets
from src.data.dataset_loader import FlowEstimationTrainDataset
from src.engine.evaluation import calculate_psnr_batch
from src.models.external.IFRNet.utils import warp


@dataclass(frozen=True)
class ExperimentConfig:
    root_dir: Path
    dataset_root_dir: Path
    dataset_preset: str
    output_csv: Path
    only_fps: int
    input_fps: int
    batch_size: int
    limit: int
    num_workers: int
    filter_valid_only: bool
    epsilon: float
    min_motion: float
    invalid_flow_scale: float
    device_name: str


@dataclass(frozen=True)
class FlowBatch:
    img0: torch.Tensor
    imgt: torch.Tensor
    img1: torch.Tensor
    bmv_60: torch.Tensor
    fmv_60: torch.Tensor
    bmv_30: torch.Tensor
    fmv_30: torch.Tensor
    embt: torch.Tensor
    info: dict[str, object]


@dataclass(frozen=True)
class ScaleResult:
    scale: torch.Tensor
    valid: torch.Tensor
    raw_scale: torch.Tensor


def parse_args(argv: list[str]) -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate oracle per-pixel effective time from 30fps motion vectors to rendered 60fps motion vectors, "
            "then compare fixed t=0.5 pseudo-flow warping against oracle t_eff pseudo-flow warping."
        )
    )
    parser.add_argument("--root-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset-root-dir", type=Path, default=Path("/workspace/datasets/Minor_0507"))
    parser.add_argument("--dataset-preset", choices=list_dataset_presets(), default="train_minor_0507")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("analysis_outputs/oracle_effective_time/train_minor_0507.csv"),
    )
    parser.add_argument("--only-fps", type=int, default=60)
    parser.add_argument("--input-fps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.set_defaults(filter_valid_only=True)
    parser.add_argument("--filter-valid-only", dest="filter_valid_only", action="store_true")
    parser.add_argument("--no-filter-valid-only", dest="filter_valid_only", action="store_false")
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--min-motion", type=float, default=1.0e-3)
    parser.add_argument("--invalid-flow-scale", type=float, default=0.5)
    parser.add_argument("--device", dest="device_name", default="auto")
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.limit < 0:
        raise ValueError(f"--limit must be non-negative, got {args.limit}")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be non-negative, got {args.num_workers}")
    if args.epsilon <= 0.0:
        raise ValueError(f"--epsilon must be positive, got {args.epsilon}")
    if args.min_motion < 0.0:
        raise ValueError(f"--min-motion must be non-negative, got {args.min_motion}")
    if args.invalid_flow_scale < 0.0 or args.invalid_flow_scale > 1.0:
        raise ValueError(f"--invalid-flow-scale must be in [0, 1], got {args.invalid_flow_scale}")

    return ExperimentConfig(
        root_dir=args.root_dir,
        dataset_root_dir=args.dataset_root_dir,
        dataset_preset=str(args.dataset_preset),
        output_csv=args.output_csv,
        only_fps=int(args.only_fps),
        input_fps=int(args.input_fps),
        batch_size=int(args.batch_size),
        limit=int(args.limit),
        num_workers=int(args.num_workers),
        filter_valid_only=bool(args.filter_valid_only),
        epsilon=float(args.epsilon),
        min_motion=float(args.min_motion),
        invalid_flow_scale=float(args.invalid_flow_scale),
        device_name=str(args.device_name),
    )


def build_script_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")
    return device


def build_dataframe(config: ExperimentConfig, logger: logging.Logger) -> pd.DataFrame:
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe = build_merged_dataframe(
        config.root_dir,
        config.output_csv.parent,
        config.dataset_preset,
        config.only_fps,
        logger,
    )

    if config.filter_valid_only:
        if "valid" not in dataframe.columns:
            raise KeyError("Cannot apply --filter-valid-only because the merged dataframe has no 'valid' column.")
        dataframe = dataframe[dataframe["valid"].astype(bool)].copy()

    if config.limit > 0:
        dataframe = dataframe.head(config.limit).copy()

    if len(dataframe) == 0:
        raise RuntimeError(
            f"No samples selected for preset={config.dataset_preset} filter_valid_only={config.filter_valid_only} "
            f"limit={config.limit}"
        )

    return dataframe


def expect_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected {name} to be torch.Tensor, got {type(value).__name__}")
    return value


def expect_info(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected batch info to be dict, got {type(value).__name__}")
    return dict(value)


def unpack_batch(batch: object, device: torch.device) -> FlowBatch:
    if not isinstance(batch, list) and not isinstance(batch, tuple):
        raise TypeError(f"Expected DataLoader batch tuple, got {type(batch).__name__}")
    if len(batch) != 9:
        raise ValueError(f"Expected FlowEstimationTrainDataset batch length 9, got {len(batch)}")

    return FlowBatch(
        img0=expect_tensor(batch[0], "img0").to(device, non_blocking=True),
        imgt=expect_tensor(batch[1], "imgt").to(device, non_blocking=True),
        img1=expect_tensor(batch[2], "img1").to(device, non_blocking=True),
        bmv_60=expect_tensor(batch[3], "bmv_60").to(device, non_blocking=True),
        fmv_60=expect_tensor(batch[4], "fmv_60").to(device, non_blocking=True),
        bmv_30=expect_tensor(batch[5], "bmv_30").to(device, non_blocking=True),
        fmv_30=expect_tensor(batch[6], "fmv_30").to(device, non_blocking=True),
        embt=expect_tensor(batch[7], "embt").to(device, non_blocking=True),
        info=expect_info(batch[8]),
    )


def projection_scale(
    endpoint_flow: torch.Tensor,
    target_flow: torch.Tensor,
    epsilon: float,
    min_motion: float,
    invalid_flow_scale: float,
) -> ScaleResult:
    if tuple(endpoint_flow.shape) != tuple(target_flow.shape):
        raise ValueError(
            f"Flow shape mismatch: endpoint_flow={tuple(endpoint_flow.shape)} target_flow={tuple(target_flow.shape)}"
        )

    denominator = (endpoint_flow * endpoint_flow).sum(dim=1, keepdim=True)
    raw_scale = (endpoint_flow * target_flow).sum(dim=1, keepdim=True) / (denominator + epsilon)
    valid = denominator.sqrt() > min_motion
    clamped_scale = raw_scale.clamp(0.0, 1.0)
    fallback_scale = torch.full_like(clamped_scale, invalid_flow_scale)
    scale = torch.where(valid, clamped_scale, fallback_scale)
    return ScaleResult(scale=scale, valid=valid, raw_scale=raw_scale)


def fixed_flow_scales(embt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    time = embt.reshape(int(embt.shape[0]), 1, 1, 1)
    fmv_scale = time
    bmv_scale = 1.0 - time
    return bmv_scale, fmv_scale


def mean_per_sample(value: torch.Tensor) -> list[float]:
    values = value.flatten(start_dim=1).mean(dim=1)
    return [float(item) for item in values.detach().cpu().tolist()]


def masked_mean_per_sample(value: torch.Tensor, mask: torch.Tensor, fallback_value: float) -> list[float]:
    if tuple(value.shape) != tuple(mask.shape):
        raise ValueError(f"Masked mean shape mismatch: value={tuple(value.shape)} mask={tuple(mask.shape)}")

    float_mask = mask.to(dtype=value.dtype)
    numerator = (value * float_mask).flatten(start_dim=1).sum(dim=1)
    denominator = float_mask.flatten(start_dim=1).sum(dim=1)
    fallback = torch.full_like(numerator, fallback_value)
    values = torch.where(denominator > 0.0, numerator / denominator.clamp_min(1.0), fallback)
    return [float(item) for item in values.detach().cpu().tolist()]


def ratio_per_sample(mask: torch.Tensor) -> list[float]:
    values = mask.to(dtype=torch.float32).flatten(start_dim=1).mean(dim=1)
    return [float(item) for item in values.detach().cpu().tolist()]


def clamped_ratio_per_sample(scale_result: ScaleResult) -> list[float]:
    clamped = scale_result.valid & ((scale_result.raw_scale < 0.0) | (scale_result.raw_scale > 1.0))
    valid_count = scale_result.valid.to(dtype=torch.float32).flatten(start_dim=1).sum(dim=1)
    clamped_count = clamped.to(dtype=torch.float32).flatten(start_dim=1).sum(dim=1)
    fallback = torch.zeros_like(valid_count)
    values = torch.where(valid_count > 0.0, clamped_count / valid_count.clamp_min(1.0), fallback)
    return [float(item) for item in values.detach().cpu().tolist()]


def flow_epe_batch(predicted_flow: torch.Tensor, target_flow: torch.Tensor) -> list[float]:
    if tuple(predicted_flow.shape) != tuple(target_flow.shape):
        raise ValueError(
            f"EPE shape mismatch: predicted_flow={tuple(predicted_flow.shape)} target_flow={tuple(target_flow.shape)}"
        )

    epe = ((predicted_flow - target_flow) ** 2).sum(dim=1, keepdim=True).sqrt()
    return mean_per_sample(epe)


def batched_text(info: dict[str, object], key: str, batch_size: int) -> list[str]:
    value = info[key]
    if isinstance(value, str):
        if batch_size != 1:
            raise ValueError(f"Scalar string info[{key}] is only valid for batch_size=1, got {batch_size}")
        return [value]
    if isinstance(value, list) or isinstance(value, tuple):
        if len(value) != batch_size:
            raise ValueError(f"Expected info[{key}] length {batch_size}, got {len(value)}")
        return [str(item) for item in value]
    raise TypeError(f"Expected info[{key}] to be str/list/tuple, got {type(value).__name__}")


def batched_bool(info: dict[str, object], key: str, batch_size: int) -> list[bool]:
    value = info[key]
    if isinstance(value, bool):
        if batch_size != 1:
            raise ValueError(f"Scalar bool info[{key}] is only valid for batch_size=1, got {batch_size}")
        return [value]
    if isinstance(value, torch.Tensor):
        if int(value.numel()) != batch_size:
            raise ValueError(f"Expected info[{key}] tensor numel {batch_size}, got {int(value.numel())}")
        return [bool(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, list) or isinstance(value, tuple):
        if len(value) != batch_size:
            raise ValueError(f"Expected info[{key}] length {batch_size}, got {len(value)}")
        return [bool(item) for item in value]
    raise TypeError(f"Expected info[{key}] to be bool/tensor/list/tuple, got {type(value).__name__}")


def append_batch_rows(rows: list[dict[str, object]], batch: FlowBatch, config: ExperimentConfig) -> None:
    batch_size = int(batch.img0.shape[0])
    fixed_bmv_scale, fixed_fmv_scale = fixed_flow_scales(batch.embt)
    fixed_bmv = fixed_bmv_scale * batch.bmv_30
    fixed_fmv = fixed_fmv_scale * batch.fmv_30

    oracle_bmv_scale = projection_scale(
        batch.bmv_30,
        batch.bmv_60,
        config.epsilon,
        config.min_motion,
        config.invalid_flow_scale,
    )
    oracle_fmv_scale = projection_scale(
        batch.fmv_30,
        batch.fmv_60,
        config.epsilon,
        config.min_motion,
        config.invalid_flow_scale,
    )
    oracle_bmv = oracle_bmv_scale.scale * batch.bmv_30
    oracle_fmv = oracle_fmv_scale.scale * batch.fmv_30

    fixed_bmv_warp = warp(batch.img0, fixed_bmv)
    fixed_fmv_warp = warp(batch.img1, fixed_fmv)
    fixed_merged = 0.5 * (fixed_bmv_warp + fixed_fmv_warp)

    oracle_bmv_warp = warp(batch.img0, oracle_bmv)
    oracle_fmv_warp = warp(batch.img1, oracle_fmv)
    oracle_merged = 0.5 * (oracle_bmv_warp + oracle_fmv_warp)

    gt_bmv_warp = warp(batch.img0, batch.bmv_60)
    gt_fmv_warp = warp(batch.img1, batch.fmv_60)
    gt_merged = 0.5 * (gt_bmv_warp + gt_fmv_warp)

    fixed_bmv_psnr = calculate_psnr_batch(batch.imgt, fixed_bmv_warp)
    fixed_fmv_psnr = calculate_psnr_batch(batch.imgt, fixed_fmv_warp)
    fixed_merged_psnr = calculate_psnr_batch(batch.imgt, fixed_merged)
    oracle_bmv_psnr = calculate_psnr_batch(batch.imgt, oracle_bmv_warp)
    oracle_fmv_psnr = calculate_psnr_batch(batch.imgt, oracle_fmv_warp)
    oracle_merged_psnr = calculate_psnr_batch(batch.imgt, oracle_merged)
    gt_bmv_psnr = calculate_psnr_batch(batch.imgt, gt_bmv_warp)
    gt_fmv_psnr = calculate_psnr_batch(batch.imgt, gt_fmv_warp)
    gt_merged_psnr = calculate_psnr_batch(batch.imgt, gt_merged)

    fixed_bmv_epe = flow_epe_batch(fixed_bmv, batch.bmv_60)
    fixed_fmv_epe = flow_epe_batch(fixed_fmv, batch.fmv_60)
    oracle_bmv_epe = flow_epe_batch(oracle_bmv, batch.bmv_60)
    oracle_fmv_epe = flow_epe_batch(oracle_fmv, batch.fmv_60)

    oracle_fmv_t_eff = oracle_fmv_scale.scale
    oracle_bmv_t_eff = 1.0 - oracle_bmv_scale.scale
    both_valid = oracle_bmv_scale.valid & oracle_fmv_scale.valid
    oracle_t_eff_gap = (oracle_fmv_t_eff - oracle_bmv_t_eff).abs()

    record_names = batched_text(batch.info, "record_name", batch_size)
    frame_ranges = batched_text(batch.info, "frame_range", batch_size)
    valid_flags = batched_bool(batch.info, "valid", batch_size)

    fixed_t_mean = mean_per_sample(fixed_fmv_scale)
    oracle_fmv_t_mean = masked_mean_per_sample(oracle_fmv_t_eff, oracle_fmv_scale.valid, config.invalid_flow_scale)
    oracle_bmv_t_mean = masked_mean_per_sample(oracle_bmv_t_eff, oracle_bmv_scale.valid, 1.0 - config.invalid_flow_scale)
    oracle_fmv_scale_mean = masked_mean_per_sample(oracle_fmv_scale.scale, oracle_fmv_scale.valid, config.invalid_flow_scale)
    oracle_bmv_scale_mean = masked_mean_per_sample(oracle_bmv_scale.scale, oracle_bmv_scale.valid, config.invalid_flow_scale)
    oracle_t_gap_mean = masked_mean_per_sample(oracle_t_eff_gap, both_valid, 0.0)
    oracle_fmv_valid_ratio = ratio_per_sample(oracle_fmv_scale.valid)
    oracle_bmv_valid_ratio = ratio_per_sample(oracle_bmv_scale.valid)
    oracle_fmv_clamped_ratio = clamped_ratio_per_sample(oracle_fmv_scale)
    oracle_bmv_clamped_ratio = clamped_ratio_per_sample(oracle_bmv_scale)

    for batch_index in range(batch_size):
        row: dict[str, object] = {
            "record_name": record_names[batch_index],
            "frame_range": frame_ranges[batch_index],
            "valid": valid_flags[batch_index],
            "fixed_t": fixed_t_mean[batch_index],
            "fixed_bmv_psnr": fixed_bmv_psnr[batch_index],
            "fixed_fmv_psnr": fixed_fmv_psnr[batch_index],
            "fixed_merged_psnr": fixed_merged_psnr[batch_index],
            "oracle_bmv_psnr": oracle_bmv_psnr[batch_index],
            "oracle_fmv_psnr": oracle_fmv_psnr[batch_index],
            "oracle_merged_psnr": oracle_merged_psnr[batch_index],
            "gt_bmv_psnr": gt_bmv_psnr[batch_index],
            "gt_fmv_psnr": gt_fmv_psnr[batch_index],
            "gt_merged_psnr": gt_merged_psnr[batch_index],
            "oracle_bmv_delta_psnr": oracle_bmv_psnr[batch_index] - fixed_bmv_psnr[batch_index],
            "oracle_fmv_delta_psnr": oracle_fmv_psnr[batch_index] - fixed_fmv_psnr[batch_index],
            "oracle_merged_delta_psnr": oracle_merged_psnr[batch_index] - fixed_merged_psnr[batch_index],
            "gt_merged_delta_psnr": gt_merged_psnr[batch_index] - fixed_merged_psnr[batch_index],
            "fixed_bmv_epe": fixed_bmv_epe[batch_index],
            "fixed_fmv_epe": fixed_fmv_epe[batch_index],
            "oracle_bmv_epe": oracle_bmv_epe[batch_index],
            "oracle_fmv_epe": oracle_fmv_epe[batch_index],
            "oracle_bmv_delta_epe": oracle_bmv_epe[batch_index] - fixed_bmv_epe[batch_index],
            "oracle_fmv_delta_epe": oracle_fmv_epe[batch_index] - fixed_fmv_epe[batch_index],
            "oracle_fmv_t_eff_mean": oracle_fmv_t_mean[batch_index],
            "oracle_bmv_t_eff_mean": oracle_bmv_t_mean[batch_index],
            "oracle_fmv_scale_mean": oracle_fmv_scale_mean[batch_index],
            "oracle_bmv_scale_mean": oracle_bmv_scale_mean[batch_index],
            "oracle_t_eff_gap_mean": oracle_t_gap_mean[batch_index],
            "oracle_fmv_valid_ratio": oracle_fmv_valid_ratio[batch_index],
            "oracle_bmv_valid_ratio": oracle_bmv_valid_ratio[batch_index],
            "oracle_fmv_clamped_ratio": oracle_fmv_clamped_ratio[batch_index],
            "oracle_bmv_clamped_ratio": oracle_bmv_clamped_ratio[batch_index],
        }
        rows.append(row)


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    metric_columns = [
        "fixed_merged_psnr",
        "oracle_merged_psnr",
        "gt_merged_psnr",
        "oracle_merged_delta_psnr",
        "gt_merged_delta_psnr",
        "fixed_bmv_epe",
        "fixed_fmv_epe",
        "oracle_bmv_epe",
        "oracle_fmv_epe",
        "oracle_bmv_delta_epe",
        "oracle_fmv_delta_epe",
        "oracle_fmv_t_eff_mean",
        "oracle_bmv_t_eff_mean",
        "oracle_t_eff_gap_mean",
        "oracle_fmv_valid_ratio",
        "oracle_bmv_valid_ratio",
        "oracle_fmv_clamped_ratio",
        "oracle_bmv_clamped_ratio",
    ]
    for column in metric_columns:
        summary_rows.append({"metric": column, "mean": float(results[column].mean())})

    oracle_win_ratio = float((results["oracle_merged_delta_psnr"] > 0.0).mean())
    summary_rows.append({"metric": "oracle_merged_win_ratio", "mean": oracle_win_ratio})
    summary_rows.append({"metric": "sample_count", "mean": float(len(results))})
    return pd.DataFrame(summary_rows)


def log_summary(summary: pd.DataFrame, logger: logging.Logger) -> None:
    lookup = {str(row["metric"]): float(row["mean"]) for row in summary.to_dict("records")}
    logger.info("Samples: %s", int(lookup["sample_count"]))
    logger.info("Fixed merged PSNR: %.4f", lookup["fixed_merged_psnr"])
    logger.info("Oracle merged PSNR: %.4f", lookup["oracle_merged_psnr"])
    logger.info("Oracle merged delta PSNR: %.4f", lookup["oracle_merged_delta_psnr"])
    logger.info("Oracle merged win ratio: %.4f", lookup["oracle_merged_win_ratio"])
    logger.info("GT-flow merged PSNR upper bound: %.4f", lookup["gt_merged_psnr"])
    logger.info("Mean oracle fmv t_eff: %.4f", lookup["oracle_fmv_t_eff_mean"])
    logger.info("Mean oracle bmv t_eff: %.4f", lookup["oracle_bmv_t_eff_mean"])
    logger.info("Mean forward/backward t_eff gap: %.4f", lookup["oracle_t_eff_gap_mean"])


def run_experiment(config: ExperimentConfig) -> None:
    logger = build_script_logger("OracleEffectiveTime")
    device = resolve_device(config.device_name)
    logger.info("Using device=%s", device)

    dataframe = build_dataframe(config, logger)
    dataset = FlowEstimationTrainDataset(dataframe, str(config.dataset_root_dir), config.input_fps, False, False)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            batch = unpack_batch(raw_batch, device)
            append_batch_rows(rows, batch, config)
            if (batch_index + 1) % 50 == 0:
                logger.info("Processed batches=%s samples=%s", batch_index + 1, len(rows))

    results = pd.DataFrame(rows)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output_csv, index=False)

    summary = build_summary(results)
    summary_csv = config.output_csv.with_name(f"{config.output_csv.stem}_summary.csv")
    summary.to_csv(summary_csv, index=False)

    logger.info("Wrote results_csv=%s", config.output_csv)
    logger.info("Wrote summary_csv=%s", summary_csv)
    log_summary(summary, logger)


def main(argv: list[str]) -> None:
    config = parse_args(argv)
    run_experiment(config)


if __name__ == "__main__":
    main(sys.argv[1:])
