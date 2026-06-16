from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_logger
from scripts.train import build_merged_dataframe
from scripts.train import build_record_name_summary
from scripts.train import build_training_dataset
from scripts.train import build_train_arg_parser
from scripts.train import get_lr
from scripts.train import load_train_run_config
from scripts.train import load_training_state
from scripts.train import read_model_init_args
from scripts.train import resolve_effective_init_flow_downscale_strategy
from scripts.train import resolve_effective_splatting_fill_strategy
from scripts.train import resolve_model_class
from scripts.train import save_checkpoint
from scripts.train import save_input_config
from scripts.train import set_lr
from scripts.train import set_seed
from scripts.train import uses_flow_approx_model
from src.engine.evaluation import AverageMeter
from src.engine.evaluation import average_metric_values
from src.engine.evaluation import build_lpips_model
from src.engine.evaluation import build_metric_meters
from src.engine.evaluation import calculate_batch_metrics
from src.engine.evaluation import format_metric_averages
from src.engine.evaluation import format_metric_values
from src.engine.evaluation import read_metric_config
from src.engine.evaluation import require_psnr_enabled
from src.engine.flow_approx import attach_effective_time_estimator
from src.engine.flow_approx import build_flow_init_result_with_fill_strategy
from src.engine.flow_approx import is_splatting_flow_approx_method

ORACLE_EFFECTIVE_TIME_SOURCES: tuple[str, ...] = ("average", "fmv", "bmv")
DEFAULT_ORACLE_EFFECTIVE_TIME_SOURCE: str = "average"
DEFAULT_ORACLE_EFFECTIVE_TIME_EPSILON: float = 1.0e-6
DEFAULT_ORACLE_EFFECTIVE_TIME_MIN_MOTION: float = 1.0e-3
DEFAULT_ORACLE_EFFECTIVE_TIME_INVALID_FLOW_SCALE: float = 0.5


@dataclass(frozen=True)
class ProjectionScaleResult:
    scale: Any
    valid: Any
    raw_scale: Any


@dataclass(frozen=True)
class OracleBatchStepOutput:
    imgt: Any
    imgt_pred: Any
    info: dict[str, Any]
    loss_rec: Any
    loss_geo: Any
    loss_dis: Any
    oracle_t_eff: Any


def validate_oracle_args(args: argparse.Namespace) -> None:
    if not uses_flow_approx_model(args.model_name):
        raise ValueError("Oracle effective-time training requires model_name=IFRNet_Residual_FlowApprox.")
    if args.oracle_effective_time_source not in ORACLE_EFFECTIVE_TIME_SOURCES:
        available_sources = ", ".join(ORACLE_EFFECTIVE_TIME_SOURCES)
        raise ValueError(
            f"Unsupported oracle_effective_time_source={args.oracle_effective_time_source}. "
            f"Available sources: {available_sources}"
        )
    if args.oracle_effective_time_epsilon <= 0.0:
        raise ValueError(f"oracle_effective_time_epsilon must be positive, got {args.oracle_effective_time_epsilon}")
    if args.oracle_effective_time_min_motion < 0.0:
        raise ValueError(
            f"oracle_effective_time_min_motion must be non-negative, got {args.oracle_effective_time_min_motion}"
        )
    if args.oracle_effective_time_invalid_flow_scale < 0.0 or args.oracle_effective_time_invalid_flow_scale > 1.0:
        raise ValueError(
            "oracle_effective_time_invalid_flow_scale must be in [0, 1], "
            f"got {args.oracle_effective_time_invalid_flow_scale}"
        )
    if args.init_flow_downscale_strategy == "masked_area" and not is_splatting_flow_approx_method(args.flow_approx_method):
        raise ValueError("init_flow_downscale_strategy=masked_area requires a splatting flow approximation method.")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default=None, type=str)
    bootstrap_args, _remaining_argv = bootstrap_parser.parse_known_args(argv)

    config_path = None if bootstrap_args.config is None else Path(bootstrap_args.config)
    config_defaults = load_train_run_config(config_path)
    parser = build_train_arg_parser(config_defaults)
    parser.description = "Train IFRNet_Residual_FlowApprox with oracle t_eff injected into flow initialization."
    parser.add_argument(
        "--oracle-effective-time-source",
        default=config_defaults.get("oracle_effective_time_source", DEFAULT_ORACLE_EFFECTIVE_TIME_SOURCE),
        choices=ORACLE_EFFECTIVE_TIME_SOURCES,
        help="Which oracle t_eff direction to inject into the shared effective-time map.",
    )
    parser.add_argument(
        "--oracle-effective-time-epsilon",
        default=config_defaults.get("oracle_effective_time_epsilon", DEFAULT_ORACLE_EFFECTIVE_TIME_EPSILON),
        type=float,
        help="Denominator epsilon for projecting rendered 60fps flow onto 30fps endpoint flow.",
    )
    parser.add_argument(
        "--oracle-effective-time-min-motion",
        default=config_defaults.get("oracle_effective_time_min_motion", DEFAULT_ORACLE_EFFECTIVE_TIME_MIN_MOTION),
        type=float,
        help="Minimum 30fps endpoint-flow magnitude required before oracle projection is trusted.",
    )
    parser.add_argument(
        "--oracle-effective-time-invalid-flow-scale",
        default=config_defaults.get(
            "oracle_effective_time_invalid_flow_scale",
            DEFAULT_ORACLE_EFFECTIVE_TIME_INVALID_FLOW_SCALE,
        ),
        type=float,
        help="Fallback scale for invalid oracle projection. 0.5 means fixed midpoint fallback.",
    )
    args = parser.parse_args(argv)
    args.model_init_args = read_model_init_args(config_defaults)
    args.metric_config = read_metric_config(config_defaults)
    args.input_config = dict(config_defaults)
    args.input_config["oracle_effective_time_source"] = args.oracle_effective_time_source
    args.input_config["oracle_effective_time_epsilon"] = args.oracle_effective_time_epsilon
    args.input_config["oracle_effective_time_min_motion"] = args.oracle_effective_time_min_motion
    args.input_config["oracle_effective_time_invalid_flow_scale"] = args.oracle_effective_time_invalid_flow_scale
    require_psnr_enabled(args.metric_config, "oracle effective-time training")
    validate_oracle_args(args)
    return args


def validate_flow_pair(endpoint_flow: Any, target_flow: Any) -> None:
    if endpoint_flow.ndim != 4 or endpoint_flow.shape[1] != 2:
        raise ValueError(f"endpoint_flow must have shape [B, 2, H, W], got {tuple(endpoint_flow.shape)}")
    if tuple(endpoint_flow.shape) != tuple(target_flow.shape):
        raise ValueError(
            f"Flow shape mismatch: endpoint_flow={tuple(endpoint_flow.shape)} target_flow={tuple(target_flow.shape)}"
        )


def projection_scale(
    endpoint_flow: Any,
    target_flow: Any,
    epsilon: float,
    min_motion: float,
    invalid_flow_scale: float,
) -> ProjectionScaleResult:
    import torch

    validate_flow_pair(endpoint_flow, target_flow)
    target_flow = target_flow.to(device=endpoint_flow.device, dtype=endpoint_flow.dtype)
    denominator = (endpoint_flow * endpoint_flow).sum(dim=1, keepdim=True)
    raw_scale = (endpoint_flow * target_flow).sum(dim=1, keepdim=True) / (denominator + epsilon)
    valid = denominator.sqrt() > min_motion
    clamped_scale = raw_scale.clamp(0.0, 1.0)
    fallback_scale = torch.full_like(clamped_scale, invalid_flow_scale)
    scale = torch.where(valid, clamped_scale, fallback_scale)
    return ProjectionScaleResult(scale=scale, valid=valid, raw_scale=raw_scale)


def build_fixed_time(embt: Any, reference_flow: Any) -> Any:
    return embt.reshape(int(embt.shape[0]), 1, 1, 1).to(
        device=reference_flow.device,
        dtype=reference_flow.dtype,
    ).expand(-1, 1, int(reference_flow.shape[2]), int(reference_flow.shape[3]))


def combine_oracle_t_eff(
    embt: Any,
    reference_flow: Any,
    fmv_t_eff: Any,
    fmv_valid: Any,
    bmv_t_eff: Any,
    bmv_valid: Any,
    oracle_effective_time_source: str,
) -> Any:
    import torch

    fixed_time = build_fixed_time(embt, reference_flow)
    if oracle_effective_time_source == "fmv":
        return torch.where(fmv_valid, fmv_t_eff, fixed_time)
    if oracle_effective_time_source == "bmv":
        return torch.where(bmv_valid, bmv_t_eff, fixed_time)
    if oracle_effective_time_source == "average":
        both_valid = fmv_valid & bmv_valid
        fmv_only = fmv_valid & ~bmv_valid
        bmv_only = bmv_valid & ~fmv_valid
        average_time = 0.5 * (fmv_t_eff + bmv_t_eff)
        oracle_time = torch.where(both_valid, average_time, fixed_time)
        oracle_time = torch.where(fmv_only, fmv_t_eff, oracle_time)
        return torch.where(bmv_only, bmv_t_eff, oracle_time)

    raise ValueError(f"Unsupported oracle_effective_time_source={oracle_effective_time_source}")


def build_oracle_t_eff(
    args: argparse.Namespace,
    embt: Any,
    bmv_30: Any,
    fmv_30: Any,
    bmv_60: Any,
    fmv_60: Any,
) -> Any:
    bmv_scale = projection_scale(
        endpoint_flow=bmv_30,
        target_flow=bmv_60,
        epsilon=args.oracle_effective_time_epsilon,
        min_motion=args.oracle_effective_time_min_motion,
        invalid_flow_scale=args.oracle_effective_time_invalid_flow_scale,
    )
    fmv_scale = projection_scale(
        endpoint_flow=fmv_30,
        target_flow=fmv_60,
        epsilon=args.oracle_effective_time_epsilon,
        min_motion=args.oracle_effective_time_min_motion,
        invalid_flow_scale=args.oracle_effective_time_invalid_flow_scale,
    )
    fmv_t_eff = fmv_scale.scale
    bmv_t_eff = 1.0 - bmv_scale.scale
    return combine_oracle_t_eff(
        embt=embt,
        reference_flow=fmv_30,
        fmv_t_eff=fmv_t_eff,
        fmv_valid=fmv_scale.valid,
        bmv_t_eff=bmv_t_eff,
        bmv_valid=bmv_scale.valid,
        oracle_effective_time_source=args.oracle_effective_time_source,
    ).clamp(0.0, 1.0)


def build_oracle_t_eff_stats(oracle_t_eff: Any) -> dict[str, list[float]]:
    flat_t_eff = oracle_t_eff.detach().float().flatten(start_dim=1)
    return {
        "oracle_t_eff_mean": [float(value) for value in flat_t_eff.mean(dim=1).cpu().tolist()],
        "oracle_t_eff_std": [float(value) for value in flat_t_eff.std(dim=1, unbiased=False).cpu().tolist()],
        "oracle_t_eff_min": [float(value) for value in flat_t_eff.min(dim=1).values.cpu().tolist()],
        "oracle_t_eff_max": [float(value) for value in flat_t_eff.max(dim=1).values.cpu().tolist()],
    }


def run_oracle_training_batch(
    args: argparse.Namespace,
    model: Any,
    batch: Any,
    device: Any,
) -> OracleBatchStepOutput:
    img0, imgt, img1, bmv_60, fmv_60, bmv_30, fmv_30, embt, info = batch
    img0 = img0.to(device)
    imgt = imgt.to(device)
    img1 = img1.to(device)
    bmv_60 = bmv_60.to(device)
    fmv_60 = fmv_60.to(device)
    bmv_30 = bmv_30.to(device)
    fmv_30 = fmv_30.to(device)
    embt = embt.to(device)

    source_depth0 = None
    source_depth1 = None
    if is_splatting_flow_approx_method(flow_approx_method=args.flow_approx_method):
        source_depth0 = info["source_depth0"].to(device)
        source_depth1 = info["source_depth1"].to(device)

    oracle_t_eff = build_oracle_t_eff(
        args=args,
        embt=embt,
        bmv_30=bmv_30,
        fmv_30=fmv_30,
        bmv_60=bmv_60,
        fmv_60=fmv_60,
    )
    flow_init = build_flow_init_result_with_fill_strategy(
        fmv_30=fmv_30,
        bmv_30=bmv_30,
        embt=embt,
        flow_approx_method=args.flow_approx_method,
        source_depth0=source_depth0,
        source_depth1=source_depth1,
        splatting_fill_strategy=args.splatting_fill_strategy,
        ground_truth_bmv=bmv_60,
        ground_truth_fmv=fmv_60,
        effective_time=oracle_t_eff,
    )

    init_bmv_mask = None
    init_fmv_mask = None
    if args.init_flow_downscale_strategy == "masked_area":
        if flow_init.masks is None:
            raise RuntimeError("init_flow_downscale_strategy=masked_area requires splatting coverage masks.")
        init_bmv_mask = flow_init.masks[:, 0:1]
        init_fmv_mask = flow_init.masks[:, 1:2]

    model_output = model(
        img0,
        img1,
        embt,
        imgt,
        init_flow0=flow_init.bmv,
        init_flow1=flow_init.fmv,
        init_flow0_mask=init_bmv_mask,
        init_flow1_mask=init_fmv_mask,
        init_flow_mask_epsilon=args.init_flow_mask_epsilon,
    )
    imgt_pred, loss_rec, loss_geo, loss_dis, _up_flow0_1, _up_flow1_1, _up_mask_1 = model_output
    return OracleBatchStepOutput(
        imgt=imgt,
        imgt_pred=imgt_pred,
        info=info,
        loss_rec=loss_rec,
        loss_geo=loss_geo,
        loss_dis=loss_dis,
        oracle_t_eff=oracle_t_eff,
    )


def build_loss_record(loss_rec: Any, loss_geo: Any, loss_dis: Any, total_loss: Any) -> dict[str, float]:
    return {
        "loss_rec": float(loss_rec.detach().cpu()),
        "loss_geo": float(loss_geo.detach().cpu()),
        "loss_dis": float(loss_dis.detach().cpu()),
        "loss_total": float(total_loss.detach().cpu()),
    }


def append_oracle_batch_metric_records(
    target_records: list[dict[str, object]],
    metric_meters: dict[str, AverageMeter],
    batch_output: OracleBatchStepOutput,
    loss_record: dict[str, float],
    metric_config: dict[str, object],
    lpips_model: Any | None,
) -> None:
    batch_size = int(batch_output.imgt_pred.shape[0])
    batch_metric_values = calculate_batch_metrics(
        batch_output.imgt.detach(),
        batch_output.imgt_pred.detach(),
        metric_config,
        lpips_model,
    )
    oracle_t_eff_stats = build_oracle_t_eff_stats(batch_output.oracle_t_eff)
    normalized_loss_record = {metric_name: float(metric_value) for metric_name, metric_value in loss_record.items()}

    for batch_index in range(batch_size):
        sample_metric_values = {
            metric_name: float(metric_values[batch_index])
            for metric_name, metric_values in batch_metric_values.items()
        }
        for metric_name, metric_value in sample_metric_values.items():
            metric_meters[metric_name].update(metric_value, 1)
        target_records.append(
            {
                "record_name": batch_output.info["record_name"][batch_index],
                "frame_range": batch_output.info["frame_range"][batch_index],
                **sample_metric_values,
                "oracle_t_eff_mean": oracle_t_eff_stats["oracle_t_eff_mean"][batch_index],
                "oracle_t_eff_std": oracle_t_eff_stats["oracle_t_eff_std"][batch_index],
                "oracle_t_eff_min": oracle_t_eff_stats["oracle_t_eff_min"][batch_index],
                "oracle_t_eff_max": oracle_t_eff_stats["oracle_t_eff_max"][batch_index],
                **normalized_loss_record,
            }
        )


def evaluate_oracle(args: argparse.Namespace, model: Any, loader: Any, device: Any, lpips_model: Any | None) -> tuple[float, Any, Any, dict[str, float]]:
    import pandas as pd
    import torch
    from tqdm import tqdm

    model.eval()
    metric_meters = build_metric_meters(args.metric_config)
    eval_records: list[dict[str, object]] = []

    with torch.no_grad():
        progress = tqdm(loader, desc="Evaluating oracle t_eff")
        for batch in progress:
            batch_output = run_oracle_training_batch(args, model, batch, device)
            total_loss = batch_output.loss_rec + batch_output.loss_geo + batch_output.loss_dis
            loss_record = build_loss_record(batch_output.loss_rec, batch_output.loss_geo, batch_output.loss_dis, total_loss)
            append_oracle_batch_metric_records(
                eval_records,
                metric_meters,
                batch_output,
                loss_record,
                args.metric_config,
                lpips_model,
            )
            progress.set_postfix({"eval_psnr": f"{metric_meters['psnr'].avg:.6f}"})

    eval_df = pd.DataFrame(eval_records)
    record_name_df = build_record_name_summary(eval_df, args.metric_config)
    return metric_meters["psnr"].avg, eval_df, record_name_df, average_metric_values(metric_meters)


def train_oracle(
    args: argparse.Namespace,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    logger: Any,
    training_state: Any,
    lpips_model: Any | None,
) -> None:
    import pandas as pd
    from tqdm import tqdm

    best_psnr = training_state.best_psnr
    global_step = training_state.global_step
    checkpoints_dir = Path(args.output_dir) / "checkpoints"

    for epoch in range(training_state.start_epoch, args.epochs):
        model.train()
        train_metric_meters = build_metric_meters(args.metric_config)
        train_loss_total_meter = AverageMeter()
        train_loss_rec_meter = AverageMeter()
        train_loss_geo_meter = AverageMeter()
        train_loss_dis_meter = AverageMeter()
        train_records: list[dict[str, object]] = []
        progress = tqdm(train_loader, desc=f"Oracle Epoch {epoch + 1}/{args.epochs}")

        for batch in progress:
            lr = get_lr(args, global_step)
            set_lr(optimizer, lr)
            optimizer.zero_grad()

            batch_output = run_oracle_training_batch(args, model, batch, device)
            total_loss = batch_output.loss_rec + batch_output.loss_geo + batch_output.loss_dis
            total_loss.backward()
            optimizer.step()

            batch_size = int(batch_output.imgt.shape[0])
            train_loss_total_meter.update(float(total_loss.detach().cpu()), batch_size)
            train_loss_rec_meter.update(float(batch_output.loss_rec.detach().cpu()), batch_size)
            train_loss_geo_meter.update(float(batch_output.loss_geo.detach().cpu()), batch_size)
            train_loss_dis_meter.update(float(batch_output.loss_dis.detach().cpu()), batch_size)

            global_step += 1
            progress.set_postfix(
                {
                    "train_loss": f"{float(total_loss.detach().cpu()):.6f}",
                    "avg_train_loss": f"{train_loss_total_meter.avg:.6f}",
                    "lr": f"{lr:.6f}",
                }
            )
            if (epoch + 1) % args.eval_interval != 0:
                continue

            loss_record = build_loss_record(batch_output.loss_rec, batch_output.loss_geo, batch_output.loss_dis, total_loss)
            append_oracle_batch_metric_records(
                train_records,
                train_metric_meters,
                batch_output,
                loss_record,
                args.metric_config,
                lpips_model,
            )

        if (epoch + 1) % args.eval_interval == 0:
            train_df = pd.DataFrame(train_records)
            train_record_name_df = build_record_name_summary(train_df, args.metric_config)
            train_df.to_csv(checkpoints_dir / f"train_epoch_{epoch + 1}.csv", index=False)
            train_record_name_df.to_csv(checkpoints_dir / f"train_epoch_{epoch + 1}_record_name.csv", index=False)
            logger.info(
                "Epoch %s train_loss_total=%.6f train_loss_rec=%.6f train_loss_geo=%.6f train_loss_dis=%.6f train_metrics=%s",
                epoch + 1,
                train_loss_total_meter.avg,
                train_loss_rec_meter.avg,
                train_loss_geo_meter.avg,
                train_loss_dis_meter.avg,
                format_metric_averages(train_metric_meters),
            )
        else:
            logger.info(
                "Epoch %s train_loss_total=%.6f train_loss_rec=%.6f train_loss_geo=%.6f train_loss_dis=%.6f",
                epoch + 1,
                train_loss_total_meter.avg,
                train_loss_rec_meter.avg,
                train_loss_geo_meter.avg,
                train_loss_dis_meter.avg,
            )

        if (epoch + 1) % args.eval_interval == 0:
            test_psnr, test_df, test_record_name_df, test_metric_values = evaluate_oracle(
                args,
                model,
                test_loader,
                device,
                lpips_model,
            )
            test_df.to_csv(checkpoints_dir / f"test_epoch_{epoch + 1}.csv", index=False)
            test_record_name_df.to_csv(checkpoints_dir / f"test_epoch_{epoch + 1}_record_name.csv", index=False)
            logger.info("Epoch %s test_metrics=%s", epoch + 1, format_metric_values(test_metric_values))

            if test_psnr > best_psnr:
                best_psnr = test_psnr
                save_checkpoint(checkpoints_dir / "best.pth", model, optimizer, epoch, best_psnr)
                logger.info("New Best PSNR - Epoch %s test_psnr=%.6f", epoch + 1, test_psnr)

        save_checkpoint(checkpoints_dir / f"epoch_{epoch + 1}.pth", model, optimizer, epoch, best_psnr)
        save_checkpoint(checkpoints_dir / "latest.pth", model, optimizer, epoch, best_psnr)


def build_dry_run_summary(args: argparse.Namespace) -> dict[str, object]:
    return {
        "mode": args.mode,
        "script": "experiments/scratch/train_oracle_effective_time.py",
        "model_name": args.model_name,
        "model_init_args": dict(args.model_init_args),
        "train_preset": args.train_preset,
        "test_preset": args.test_preset,
        "root_dir": args.root_dir,
        "dataset_root_dir": args.dataset_root_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_interval": args.eval_interval,
        "flow_approx_method": args.flow_approx_method,
        "splatting_fill_strategy": args.splatting_fill_strategy,
        "effective_splatting_fill_strategy": resolve_effective_splatting_fill_strategy(args),
        "init_flow_downscale_strategy": args.init_flow_downscale_strategy,
        "effective_init_flow_downscale_strategy": resolve_effective_init_flow_downscale_strategy(args),
        "oracle_effective_time_source": args.oracle_effective_time_source,
        "oracle_effective_time_epsilon": args.oracle_effective_time_epsilon,
        "oracle_effective_time_min_motion": args.oracle_effective_time_min_motion,
        "oracle_effective_time_invalid_flow_scale": args.oracle_effective_time_invalid_flow_scale,
        "pretrained_checkpoint_path": args.pretrained_checkpoint_path,
        "resume_path": args.resume_path,
        "metrics": dict(args.metric_config),
    }


def run_training(args: argparse.Namespace) -> None:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger, run_dir = build_logger(output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = build_lpips_model(args.metric_config, device)
    logger.info("device=%s", device)
    logger.info("oracle_effective_time_source=%s", args.oracle_effective_time_source)
    logger.info(
        "oracle_effective_time_epsilon=%s oracle_effective_time_min_motion=%s oracle_effective_time_invalid_flow_scale=%s",
        args.oracle_effective_time_epsilon,
        args.oracle_effective_time_min_motion,
        args.oracle_effective_time_invalid_flow_scale,
    )

    root_dir = Path(args.root_dir)
    train_df = build_merged_dataframe(root_dir, checkpoints_dir, args.train_preset, args.only_fps, logger)
    test_df = build_merged_dataframe(root_dir, checkpoints_dir, args.test_preset, args.only_fps, logger)

    if "valid" in train_df.columns:
        logger.info("Valid Count %s in %s", train_df["valid"].value_counts().to_dict(), args.train_preset)
        train_df = train_df[train_df["valid"] == True]
    if "valid" in test_df.columns:
        logger.info("Valid Count %s in %s", test_df["valid"].value_counts().to_dict(), args.test_preset)
        test_df = test_df[test_df["valid"] == True]

    train_dataset = build_training_dataset(
        train_df,
        args.dataset_root_dir,
        True,
        args.input_fps,
        args.model_name,
        args.flow_approx_method,
    )
    test_dataset = build_training_dataset(
        test_df,
        args.dataset_root_dir,
        False,
        args.input_fps,
        args.model_name,
        args.flow_approx_method,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    args.iters_per_epoch = len(train_loader)
    model_class = resolve_model_class(args.model_name)
    model = model_class(**dict(args.model_init_args))
    attach_effective_time_estimator(model=model, effective_time_mode="disabled", effective_time_hidden_channels=32)
    model = model.to(device)
    if hasattr(model, "init_flow_layer"):
        logger.info("model_init_flow_layer=%s", model.init_flow_layer)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0)
    training_state = load_training_state(args, model, optimizer, device, logger)
    logger.info(
        "model=%s mode=%s train_samples=%s test_samples=%s start_epoch=%s epochs=%s batch_size=%s",
        args.model_name,
        training_state.mode,
        len(train_dataset),
        len(test_dataset),
        training_state.start_epoch,
        args.epochs,
        args.batch_size,
    )
    logger.info("run_log_dir=%s", run_dir)
    input_config_path = save_input_config(target_dir=run_dir, input_config=args.input_config)
    logger.info("input_config_path=%s", input_config_path)

    train_oracle(args, model, optimizer, train_loader, test_loader, device, logger, training_state, lpips_model)


def main(argv: list[str] | None) -> None:
    args = parse_args(argv)
    if args.mode == "dry-run":
        print(json.dumps(build_dry_run_summary(args), indent=2))
        return

    run_training(args)


if __name__ == "__main__":
    main(sys.argv[1:])
