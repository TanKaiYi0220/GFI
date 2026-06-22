from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import ACTIVE_DATASET_ROOT_KEY
from src.data.dataset_config import get_dataset_preset
from src.data.dataset_config import iter_dataset_configs
from src.data.dataset_config import list_dataset_presets
from src.engine.evaluation import AverageMeter
from src.engine.evaluation import average_metric_values
from src.engine.evaluation import build_lpips_model
from src.engine.evaluation import build_metric_meters
from src.engine.evaluation import calculate_batch_metrics
from src.engine.evaluation import format_metric_averages
from src.engine.evaluation import format_metric_values
from src.engine.evaluation import get_enabled_metric_names
from src.engine.evaluation import read_metric_config
from src.engine.evaluation import require_psnr_enabled
from src.engine.flow_approx import build_flow_init_result_with_fill_strategy
from src.engine.flow_approx import DEFAULT_SPLATTING_FILL_STRATEGY
from src.engine.flow_approx import FLOW_APPROX_METHOD_CHOICES
from src.engine.flow_approx import FLOW_APPROX_METHODS
from src.engine.flow_approx import is_splatting_flow_approx_method
from src.engine.flow_approx import resolve_splatting_fill_strategy
from src.engine.flow_approx import SPLATTING_FILL_STRATEGIES
from src.utils.config import load_yaml_file

MODEL_NAMES: tuple[str, ...] = ("IFRNet", "IFRNet_Residual", "IFRNet_Residual_FlowApprox")
FLOW_APPROX_MODEL_NAMES: tuple[str, ...] = ("IFRNet_Residual_FlowApprox",)
DEFAULT_INIT_FLOW_DOWNSCALE_STRATEGY: str = "bilinear"
DEFAULT_INIT_FLOW_MASK_EPSILON: float = 1e-6
INIT_FLOW_DOWNSCALE_STRATEGIES: tuple[str, ...] = ("bilinear", "masked_area")


@dataclass(frozen=True)
class TrainingState:
    start_epoch: int
    global_step: int
    best_psnr: float
    mode: str


@dataclass(frozen=True)
class BatchStepOutput:
    imgt: Any
    imgt_pred: Any
    info: dict[str, Any]
    loss_rec: Any
    loss_geo: Any
    loss_dis: Any


def uses_flow_approx_model(model_name: str) -> bool:
    return model_name in FLOW_APPROX_MODEL_NAMES


def read_model_init_args(config_values: dict[str, Any]) -> dict[str, Any]:
    raw_model_init_args = config_values.get("model_init_args", {})
    if raw_model_init_args is None:
        return {}
    if not isinstance(raw_model_init_args, dict):
        raise TypeError(f"model_init_args must be a mapping, got {type(raw_model_init_args).__name__}")

    return dict(raw_model_init_args)


def parse_bool_value(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized_value = value.strip().lower()
        if normalized_value in ("true", "1", "yes", "on"):
            return True
        if normalized_value in ("false", "0", "no", "off"):
            return False
    raise TypeError(f"{key} must be a boolean, got {value!r}")


def parse_eval_convex_upsampling_arg(value: str) -> bool:
    return parse_bool_value(value, "eval_convex_upsampling")


def read_optional_bool(config_values: dict[str, Any], key: str) -> bool | None:
    if key not in config_values or config_values[key] is None:
        return None
    return parse_bool_value(config_values[key], key)


def set_model_convex_upsampling(model: Any, enabled: bool, context: str) -> bool:
    setter = getattr(model, "set_convex_upsampling", None)
    if not callable(setter):
        raise TypeError(
            f"{context} requested eval_convex_upsampling={enabled}, "
            f"but model type {type(model).__name__} does not support it."
        )
    previous_value = bool(getattr(model, "convex_upsampling"))
    setter(enabled)
    return previous_value


def resolve_model_class(model_name: str) -> type[Any]:
    if model_name == "IFRNet":
        from src.models.IFRNet import Model as IFRNetModel

        return IFRNetModel
    if model_name == "IFRNet_Residual":
        from src.models.IFRNet_Residual import Model as IFRNetResidualModel

        return IFRNetResidualModel
    if model_name == "IFRNet_Residual_FlowApprox":
        from src.models.IFRNet_Residual import Model as IFRNetResidualModel

        return IFRNetResidualModel

    available_models = ", ".join(MODEL_NAMES)
    raise KeyError(f"Unknown model '{model_name}'. Available models: {available_models}")


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(args: argparse.Namespace, step: int) -> float:
    total_steps = max(args.epochs * args.iters_per_epoch, 1)
    ratio = 0.5 * (1.0 + math.cos(step / total_steps * math.pi))
    return (args.lr_start - args.lr_end) * ratio + args.lr_end


def set_lr(optimizer: Any, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def build_logger(output_dir: Path) -> tuple[logging.Logger, Path]:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = logs_dir / time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("GFITrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, run_dir


def build_merged_dataframe(
    root_dir: Path,
    checkpoints_dir: Path,
    dataset_preset_name: str,
    only_fps: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    import pandas as pd

    dataset_preset = get_dataset_preset(dataset_preset_name)
    dataframe_list: list[pd.DataFrame] = []

    for dataset_config in iter_dataset_configs(dataset_preset):
        if dataset_config.fps != only_fps:
            continue

        csv_path = root_dir / f"{dataset_config.record_name}_preprocessed" / f"{dataset_config.mode_index}_raw_sequence_frame_index.csv"
        if not csv_path.is_file():
            logger.warning("Dataset CSV missing: %s", csv_path)
            continue

        dataframe = pd.read_csv(csv_path)
        dataframe["record"] = dataset_config.record
        dataframe["mode"] = dataset_config.mode_path
        dataframe_list.append(dataframe)
        logger.info("Loaded dataset CSV %s rows=%s", csv_path, len(dataframe))

    if len(dataframe_list) == 0:
        raise RuntimeError(f"No dataset CSV found under root_dir={root_dir} for preset={dataset_preset_name}")

    merged = pd.concat(dataframe_list, ignore_index=True)
    merged.to_csv(checkpoints_dir / f"{dataset_preset_name}_merged.csv", index=False)
    logger.info("Merged dataset size=%s preset=%s", len(merged), dataset_preset_name)
    return merged


def forward_model(
    model_name: str,
    model: Any,
    img0: Any,
    img1: Any,
    embt: Any,
    imgt: Any,
    source_bmv: Any,
    source_fmv: Any,
    flow_approx_method: str,
    splatting_fill_strategy: str,
    init_flow_downscale_strategy: str,
    init_flow_mask_epsilon: float,
    source_depth0: Any | None,
    source_depth1: Any | None,
    ground_truth_bmv: Any | None,
    ground_truth_fmv: Any | None,
) -> Any:
    import torch

    init_bmv = source_bmv
    init_fmv = source_fmv
    init_bmv_mask = None
    init_fmv_mask = None

    if uses_flow_approx_model(model_name):
        flow_init = build_flow_init_result_with_fill_strategy(
            fmv_30=source_fmv,
            bmv_30=source_bmv,
            embt=embt,
            flow_approx_method=flow_approx_method,
            source_depth0=source_depth0,
            source_depth1=source_depth1,
            splatting_fill_strategy=splatting_fill_strategy,
            ground_truth_bmv=ground_truth_bmv,
            ground_truth_fmv=ground_truth_fmv,
        )
        init_bmv = flow_init.bmv
        init_fmv = flow_init.fmv
        if init_flow_downscale_strategy == "masked_area":
            if flow_init.masks is None:
                raise RuntimeError(
                    "init_flow_downscale_strategy=masked_area requires splatting coverage masks, but none were produced."
                )
            init_bmv_mask = flow_init.masks[:, 0:1]
            init_fmv_mask = flow_init.masks[:, 1:2]

    if model_name == "IFRNet":
        flow = torch.cat([init_bmv, init_fmv], dim=1).float()
        return model(img0, img1, embt, imgt, flow)

    return model(
        img0,
        img1,
        embt,
        imgt,
        init_flow0=init_bmv,
        init_flow1=init_fmv,
        init_flow0_mask=init_bmv_mask,
        init_flow1_mask=init_fmv_mask,
        init_flow_mask_epsilon=init_flow_mask_epsilon,
    )


def build_loss_record(
    loss_rec: Any,
    loss_geo: Any,
    loss_dis: Any,
    total_loss: Any,
) -> dict[str, float]:
    return {
        "loss_rec": float(loss_rec.detach().cpu()),
        "loss_geo": float(loss_geo.detach().cpu()),
        "loss_dis": float(loss_dis.detach().cpu()),
        "loss_total": float(total_loss.detach().cpu()),
    }


def append_batch_metric_records(
    target_records: list[dict[str, object]],
    metric_meters: dict[str, AverageMeter],
    info: dict[str, Any],
    imgt_pred: Any,
    imgt: Any,
    loss_record: dict[str, float],
    metric_config: dict[str, object],
    lpips_model: Any | None,
) -> None:
    batch_size = int(imgt_pred.shape[0])
    normalized_loss_record = {metric_name: float(metric_value) for metric_name, metric_value in loss_record.items()}
    batch_metric_values = calculate_batch_metrics(imgt.detach(), imgt_pred.detach(), metric_config, lpips_model)

    for batch_index in range(batch_size):
        sample_metric_values = {metric_name: float(metric_values[batch_index]) for metric_name, metric_values in batch_metric_values.items()}
        for metric_name, metric_value in sample_metric_values.items():
            metric_meters[metric_name].update(metric_value, 1)
        target_records.append(
            {
                "record_name": info["record_name"][batch_index],
                "frame_range": info["frame_range"][batch_index],
                **sample_metric_values,
                **normalized_loss_record,
            }
        )


def build_record_name_summary(dataframe: Any, metric_config: dict[str, object]) -> Any:
    summary_columns = [*get_enabled_metric_names(metric_config), "loss_rec", "loss_geo", "loss_dis", "loss_total"]
    return (
        dataframe.groupby(["record_name"], as_index=False)[summary_columns]
        .mean()
        .sort_values(["record_name"])
        .reset_index(drop=True)
    )


def save_checkpoint(
    checkpoint_path: Path,
    model: Any,
    optimizer: Any,
    epoch: int,
    best_psnr: float,
) -> None:
    import torch

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
        },
        str(checkpoint_path),
    )


def build_training_dataset(
    dataframe: Any,
    dataset_root_dir: str,
    augment: bool,
    input_fps: int,
    model_name: str,
    flow_approx_method: str,
) -> Any:
    normalized_dataframe = dataframe.reset_index(drop=True)

    if uses_flow_approx_model(model_name):
        from src.data.dataset_loader import FlowEstimationTrainDataset

        include_source_depths = is_splatting_flow_approx_method(flow_approx_method=flow_approx_method)
        return FlowEstimationTrainDataset(normalized_dataframe, dataset_root_dir, input_fps, augment, include_source_depths)

    from src.data.dataset_loader import VFITrainDataset

    return VFITrainDataset(normalized_dataframe, dataset_root_dir, augment, input_fps)


def select_sample_rows(dataframe: Any, frame_keys: list[str]) -> Any:
    indices: list[int] = []
    for frame_key in frame_keys:
        parts = frame_key.split("_")
        if len(parts) != 6 or parts[0] != "ARPG":
            raise ValueError(f"sample frame key must look like ARPG_3_0_Medium_5_0404, got {frame_key}")
        record, main_index, difficulty, sub_index, frame_text = f"{parts[0]}_{parts[1]}", parts[2], parts[3], parts[4], parts[5]
        if difficulty not in ("Easy", "Medium", "Difficult"):
            raise ValueError(f"sample frame key difficulty must be Easy, Medium, or Difficult, got {frame_key}")
        base_matches = dataframe[(dataframe["record"] == record) & dataframe["mode"].str.startswith(f"{main_index}_") & dataframe["mode"].str.contains(f"_{sub_index}/fps_", regex=False)]
        base_matches = base_matches[base_matches["mode"].str.startswith(f"{main_index}_{difficulty}/")]
        matches = base_matches[base_matches["img0"].astype(int) == int(frame_text)]
        matches = matches.drop_duplicates(subset=["record", "mode", "img0", "img1", "img2"])
        if len(matches) != 1:
            mode_preview = dataframe[
                (dataframe["record"] == record)
                & dataframe["mode"].str.startswith(f"{main_index}_")
                & dataframe["mode"].str.contains(f"_{sub_index}/fps_", regex=False)
                & (dataframe["img0"].astype(int) == int(frame_text))
            ][["record", "mode", "img0", "img1", "img2"]].drop_duplicates().head(10).to_dict("records")
            preview = matches[["record", "mode", "img0", "img1", "img2"]].head(5).to_dict("records")
            raise RuntimeError(f"Expected exactly one row for sample frame {frame_key}, got {len(matches)} after matching img0. candidates={preview}, available_without_difficulty={mode_preview}")
        indices.append(int(matches.index[0]))
    return dataframe.loc[indices].reset_index(drop=True)


def save_epoch_samples(args: argparse.Namespace, model: Any, sample_dataframes: dict[str, Any], epoch: int, device: Any, logger: logging.Logger) -> None:
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from scripts.inference import run_inference_batch_with_fill_strategy
    from scripts.inference import save_selected_sample_artifacts
    from src.data.image_ops import flow_to_image, save_image

    frame_groups = {"train": args.sample_train_frames, "test": args.sample_test_frames}
    if sum(len(frame_keys) for frame_keys in frame_groups.values()) == 0:
        return
    if args.sample_interval_epoch <= 0:
        raise ValueError(f"sample_interval_epoch must be positive, got {args.sample_interval_epoch}")
    if (epoch + 1) % args.sample_interval_epoch != 0:
        return

    previous_convex_upsampling = None
    if args.eval_convex_upsampling is not None:
        previous_convex_upsampling = set_model_convex_upsampling(
            model=model,
            enabled=args.eval_convex_upsampling,
            context="sample evaluation",
        )
    try:
        model.eval()
        with torch.no_grad():
            for split_name, frame_keys in frame_groups.items():
                if len(frame_keys) == 0:
                    continue
                sample_dataframe = select_sample_rows(sample_dataframes[split_name], list(frame_keys))
                sample_dataset = build_training_dataset(
                    sample_dataframe,
                    args.dataset_root_dir,
                    False,
                    args.input_fps,
                    args.model_name,
                    args.flow_approx_method,
                )
                for frame_key, batch in zip(frame_keys, DataLoader(sample_dataset, batch_size=1, shuffle=False)):
                    save_dir = Path(args.output_dir) / "samples" / split_name / frame_key / f"epoch_{epoch + 1:04d}"
                    inference_result = run_inference_batch_with_fill_strategy(
                        batch,
                        device,
                        args.flow_approx_method,
                        args.splatting_fill_strategy,
                        args.init_flow_downscale_strategy,
                        args.init_flow_mask_epsilon,
                        model,
                        args.model_name,
                        1.0,
                    )
                    save_selected_sample_artifacts(cv2, 99.0, 1.0, flow_to_image, inference_result, np, save_dir, save_image)
                    logger.info("Saved sample frame split=%s frame=%s epoch=%s dir=%s", split_name, frame_key, epoch + 1, save_dir)
    finally:
        if previous_convex_upsampling is not None:
            set_model_convex_upsampling(
                model=model,
                enabled=previous_convex_upsampling,
                context="sample evaluation restore",
            )


def resolve_dataset_class_name(model_name: str) -> str:
    if uses_flow_approx_model(model_name):
        return "FlowEstimationTrainDataset"

    return "VFITrainDataset"


def run_training_batch(
    args: argparse.Namespace,
    model: Any,
    batch: Any,
    device: Any,
) -> BatchStepOutput:
    if uses_flow_approx_model(args.model_name):
        img0, imgt, img1, bmv_60, fmv_60, bmv_30, fmv_30, embt, info = batch
        source_bmv = bmv_30.to(device)
        source_fmv = fmv_30.to(device)
        ground_truth_bmv = bmv_60.to(device)
        ground_truth_fmv = fmv_60.to(device)
        if is_splatting_flow_approx_method(flow_approx_method=args.flow_approx_method):
            source_depth0 = info["source_depth0"].to(device)
            source_depth1 = info["source_depth1"].to(device)
        else:
            source_depth0 = None
            source_depth1 = None
    else:
        img0, imgt, img1, bmv, fmv, embt, info = batch
        source_bmv = bmv.to(device)
        source_fmv = fmv.to(device)
        source_depth0 = None
        source_depth1 = None
        ground_truth_bmv = None
        ground_truth_fmv = None

    img0 = img0.to(device)
    img1 = img1.to(device)
    imgt = imgt.to(device)
    embt = embt.to(device)

    model_output = forward_model(
        args.model_name,
        model,
        img0,
        img1,
        embt,
        imgt,
        source_bmv,
        source_fmv,
        args.flow_approx_method,
        args.splatting_fill_strategy,
        args.init_flow_downscale_strategy,
        args.init_flow_mask_epsilon,
        source_depth0,
        source_depth1,
        ground_truth_bmv,
        ground_truth_fmv,
    )
    imgt_pred, loss_rec, loss_geo, loss_dis, _up_flow0_1, _up_flow1_1, _up_mask_1 = model_output
    return BatchStepOutput(
        imgt=imgt,
        imgt_pred=imgt_pred,
        info=info,
        loss_rec=loss_rec,
        loss_geo=loss_geo,
        loss_dis=loss_dis,
    )


def evaluate(
    args: argparse.Namespace,
    model: Any,
    loader: Any,
    device: Any,
    lpips_model: Any | None,
) -> tuple[float, Any, Any, dict[str, float]]:
    import pandas as pd
    import torch
    from tqdm import tqdm

    previous_convex_upsampling = None
    if args.eval_convex_upsampling is not None:
        previous_convex_upsampling = set_model_convex_upsampling(
            model=model,
            enabled=args.eval_convex_upsampling,
            context="training evaluation",
        )

    metric_meters = build_metric_meters(args.metric_config)
    eval_records: list[dict[str, object]] = []

    try:
        model.eval()
        with torch.no_grad():
            pbar = tqdm(loader, desc="Evaluating")
            for batch in pbar:
                batch_output = run_training_batch(args, model, batch, device)
                total_loss = batch_output.loss_rec + batch_output.loss_geo + batch_output.loss_dis
                loss_record = build_loss_record(
                    batch_output.loss_rec,
                    batch_output.loss_geo,
                    batch_output.loss_dis,
                    total_loss,
                )
                append_batch_metric_records(
                    eval_records,
                    metric_meters,
                    batch_output.info,
                    batch_output.imgt_pred,
                    batch_output.imgt,
                    loss_record,
                    args.metric_config,
                    lpips_model,
                )
                pbar.set_postfix({"eval_psnr": f"{metric_meters['psnr'].avg:.6f}"})
    finally:
        if previous_convex_upsampling is not None:
            set_model_convex_upsampling(
                model=model,
                enabled=previous_convex_upsampling,
                context="training evaluation restore",
            )

    eval_df = pd.DataFrame(eval_records)
    record_name_df = build_record_name_summary(eval_df, args.metric_config)
    return metric_meters["psnr"].avg, eval_df, record_name_df, average_metric_values(metric_meters)


def train(
    args: argparse.Namespace,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    logger: logging.Logger,
    training_state: TrainingState,
    sample_dataframes: dict[str, Any],
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
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            lr = get_lr(args, global_step)
            set_lr(optimizer, lr)
            optimizer.zero_grad()

            batch_output = run_training_batch(args, model, batch, device)
            total_loss = batch_output.loss_rec + batch_output.loss_geo + batch_output.loss_dis
            total_loss.backward()
            optimizer.step()

            batch_size = int(batch_output.imgt.shape[0])
            train_loss_total_meter.update(float(total_loss.detach().cpu()), batch_size)
            train_loss_rec_meter.update(float(batch_output.loss_rec.detach().cpu()), batch_size)
            train_loss_geo_meter.update(float(batch_output.loss_geo.detach().cpu()), batch_size)
            train_loss_dis_meter.update(float(batch_output.loss_dis.detach().cpu()), batch_size)

            global_step += 1
            pbar.set_postfix(
                {
                    "train_loss": f"{float(total_loss.detach().cpu()):.6f}",
                    "avg_train_loss": f"{train_loss_total_meter.avg:.6f}",
                    "lr": f"{lr:.6f}",
                }
            )
            if (epoch + 1) % args.eval_interval != 0:
                continue

            loss_record = build_loss_record(
                batch_output.loss_rec,
                batch_output.loss_geo,
                batch_output.loss_dis,
                total_loss,
            )
            append_batch_metric_records(
                train_records,
                train_metric_meters,
                batch_output.info,
                batch_output.imgt_pred,
                batch_output.imgt,
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
            test_psnr, test_df, test_record_name_df, test_metric_values = evaluate(args, model, test_loader, device, lpips_model)
            test_df.to_csv(checkpoints_dir / f"test_epoch_{epoch + 1}.csv", index=False)
            test_record_name_df.to_csv(checkpoints_dir / f"test_epoch_{epoch + 1}_record_name.csv", index=False)
            logger.info("Epoch %s test_metrics=%s", epoch + 1, format_metric_values(test_metric_values))

            if test_psnr > best_psnr:
                best_psnr = test_psnr
                save_checkpoint(checkpoints_dir / "best.pth", model, optimizer, epoch, best_psnr)
                logger.info("New Best PSNR - Epoch %s test_psnr=%.6f", epoch + 1, test_psnr)

        save_checkpoint(checkpoints_dir / f"epoch_{epoch + 1}.pth", model, optimizer, epoch, best_psnr)
        save_checkpoint(checkpoints_dir / "latest.pth", model, optimizer, epoch, best_psnr)
        save_epoch_samples(args, model, sample_dataframes, epoch, device, logger)


def load_train_run_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    return load_yaml_file(config_path=config_path)


def build_train_arg_parser(config_defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train IFRNet variants on the VFI dataset.")
    parser.add_argument("--config", default=config_defaults.get("config"), type=str, help="Optional JSON-formatted YAML-compatible run config.")
    parser.add_argument("--mode", default=config_defaults.get("mode", "dry-run"), choices=["dry-run", "train"])
    parser.add_argument("--model-name", default=config_defaults.get("model_name", "IFRNet"), choices=MODEL_NAMES)
    parser.add_argument("--root-dir", default=config_defaults.get("root_dir", "./datasets/data"), help="Directory containing preprocessed CSV indexes.")
    parser.add_argument("--dataset-root-dir", default=config_defaults.get("dataset_root_dir"), type=str, help="Root directory containing frame and velocity assets.")
    parser.add_argument("--paths-config", default=config_defaults.get("paths_config"), type=str, help="Optional path to configs/paths/default.yaml.")
    parser.add_argument("--train-preset", default=config_defaults.get("train_preset", "train_vfx_0416"), choices=list_dataset_presets())
    parser.add_argument("--test-preset", default=config_defaults.get("test_preset", "test_vfx_0416"), choices=list_dataset_presets())
    parser.add_argument("--epochs", default=config_defaults.get("epochs", 60), type=int, help="Total number of epochs to run.")
    parser.add_argument("--resume-path", default=config_defaults.get("resume_path"), type=str, help="Checkpoint to resume from.")
    parser.add_argument(
        "--pretrained-checkpoint-path",
        default=config_defaults.get("pretrained_checkpoint_path", config_defaults.get("pretrained_checkpoints_path")),
        type=str,
        help="Checkpoint to load when resume_path is None.",
    )
    parser.add_argument("--eval-interval", default=config_defaults.get("eval_interval", 1), type=int, help="Run validation every N epochs.")
    parser.add_argument("--lr-start", default=config_defaults.get("lr_start", 1e-4), type=float, help="Initial learning rate.")
    parser.add_argument("--lr-end", default=config_defaults.get("lr_end", 1e-5), type=float, help="Final learning rate after cosine decay.")
    parser.add_argument("--seed", default=config_defaults.get("seed", 1234), type=int, help="Random seed.")
    parser.add_argument("--batch-size", default=config_defaults.get("batch_size", 8), type=int, help="Training batch size.")
    parser.add_argument("--output-dir", default=config_defaults.get("output_dir"), type=str, help="Output directory for checkpoints and logs.")
    parser.add_argument("--only-fps", default=config_defaults.get("only_fps", 60), type=int, help="Use CSV entries for this FPS only.")
    parser.add_argument("--input-fps", default=config_defaults.get("input_fps", 30), type=int, help="Input frame rate for the dataset loader.")
    parser.add_argument("--sample-train-frames", default=config_defaults.get("sample_train_frames", []), nargs="*", help="Training sample frame keys such as ARPG_2_4_2_052.")
    parser.add_argument("--sample-test-frames", default=config_defaults.get("sample_test_frames", []), nargs="*", help="Testing sample frame keys such as ARPG_2_4_2_052.")
    parser.add_argument("--sample-interval-epoch", default=config_defaults.get("sample_interval_epoch", config_defaults.get("eval_interval", 1)), type=int, help="Save configured sample frames every N epochs.")
    parser.add_argument(
        "--eval-convex-upsampling",
        default=config_defaults.get("eval_convex_upsampling"),
        type=parse_eval_convex_upsampling_arg,
        help="Optional evaluation-only override for residual-model convex flow upsampling.",
    )
    parser.add_argument(
        "--flow-approx-method",
        default=config_defaults.get("flow_approx_method", "combination"),
        choices=FLOW_APPROX_METHOD_CHOICES,
        help="How to approximate middle-frame flows from 30fps motion vectors.",
    )
    parser.add_argument(
        "--splatting-fill-strategy",
        default=config_defaults.get("splatting_fill_strategy", DEFAULT_SPLATTING_FILL_STRATEGY),
        choices=SPLATTING_FILL_STRATEGIES,
        help="How splatting fills no-hit target pixels when flow_approx_method is splatting or linear_splatting.",
    )
    parser.add_argument(
        "--init-flow-downscale-strategy",
        default=config_defaults.get("init_flow_downscale_strategy", DEFAULT_INIT_FLOW_DOWNSCALE_STRATEGY),
        choices=INIT_FLOW_DOWNSCALE_STRATEGIES,
        help="How init flows are downscaled inside IFRNet residual pyramids.",
    )
    parser.add_argument(
        "--init-flow-mask-epsilon",
        default=config_defaults.get("init_flow_mask_epsilon", DEFAULT_INIT_FLOW_MASK_EPSILON),
        type=float,
        help="Numerical epsilon for mask-normalized init-flow downscaling.",
    )
    return parser


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default=None, type=str, help="Optional JSON-formatted YAML-compatible run config.")
    bootstrap_args, _remaining_argv = bootstrap_parser.parse_known_args(argv)

    config_path = None if bootstrap_args.config is None else Path(bootstrap_args.config)
    config_defaults = load_train_run_config(config_path)
    parser = build_train_arg_parser(config_defaults)
    args = parser.parse_args(argv)
    args.model_init_args = read_model_init_args(config_defaults)
    args.metric_config = read_metric_config(config_defaults)
    args.eval_convex_upsampling = (
        None
        if args.eval_convex_upsampling is None
        else parse_bool_value(args.eval_convex_upsampling, "eval_convex_upsampling")
    )
    require_psnr_enabled(args.metric_config, "training")
    args.input_config = config_defaults
    validate_flow_approx_runtime_args(args)
    return args


def resolve_effective_splatting_fill_strategy(args: argparse.Namespace) -> str:
    if not uses_flow_approx_model(args.model_name):
        return ""

    if not is_splatting_flow_approx_method(flow_approx_method=args.flow_approx_method):
        return ""

    return resolve_splatting_fill_strategy(
        flow_approx_method=args.flow_approx_method,
        splatting_fill_strategy=args.splatting_fill_strategy,
    )


def resolve_effective_init_flow_downscale_strategy(args: argparse.Namespace) -> str:
    if not uses_flow_approx_model(args.model_name):
        return ""
    if not is_splatting_flow_approx_method(flow_approx_method=args.flow_approx_method):
        return ""
    return args.init_flow_downscale_strategy


def validate_flow_approx_runtime_args(args: argparse.Namespace) -> None:
    if args.init_flow_mask_epsilon <= 0:
        raise ValueError(f"init_flow_mask_epsilon must be positive, got {args.init_flow_mask_epsilon}")
    if args.init_flow_downscale_strategy == "masked_area":
        if not uses_flow_approx_model(args.model_name):
            raise ValueError(
                "init_flow_downscale_strategy=masked_area requires model_name=IFRNet_Residual_FlowApprox."
            )
        if not is_splatting_flow_approx_method(flow_approx_method=args.flow_approx_method):
            raise ValueError(
                "init_flow_downscale_strategy=masked_area requires a splatting flow approximation method."
            )


def log_run_summary(
    args: argparse.Namespace,
    train_dataset: Any,
    test_dataset: Any,
    training_state: TrainingState,
    device: Any,
    logger: logging.Logger,
) -> None:
    logger.info(
        "model=%s mode=%s dataset_class=%s device=%s output_dir=%s dataset_root_dir=%s train_samples=%s test_samples=%s start_epoch=%s epochs=%s batch_size=%s",
        args.model_name,
        training_state.mode,
        resolve_dataset_class_name(args.model_name),
        device,
        args.output_dir,
        args.dataset_root_dir,
        len(train_dataset),
        len(test_dataset),
        training_state.start_epoch,
        args.epochs,
        args.batch_size,
    )


def is_raw_model_state_dict(checkpoint: Any, torch_module: Any) -> bool:
    if not isinstance(checkpoint, dict) or len(checkpoint) == 0:
        return False

    return all(isinstance(key, str) and torch_module.is_tensor(value) for key, value in checkpoint.items())


def extract_pretrained_state_dict(checkpoint: Any, checkpoint_path: Path, torch_module: Any) -> Any:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]

    if is_raw_model_state_dict(checkpoint, torch_module):
        return checkpoint

    available_keys = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    raise KeyError(
        "pretrained_checkpoint_path must point to either a raw model state_dict or a full training checkpoint "
        f"containing a 'model' key: path={checkpoint_path}, keys={available_keys}"
    )


def load_training_state(
    args: argparse.Namespace,
    model: Any,
    optimizer: Any,
    device: Any,
    logger: logging.Logger,
) -> TrainingState:
    import torch

    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location=device)
        if "model" not in checkpoint or "optimizer" not in checkpoint or "epoch" not in checkpoint:
            raise KeyError(f"resume_path must point to a full training checkpoint with model, optimizer, and epoch: {args.resume_path}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = int(checkpoint["epoch"]) + 1
        logger.info("Resumed from %s at epoch %s", args.resume_path, start_epoch)
        return TrainingState(
            start_epoch=start_epoch,
            global_step=start_epoch * args.iters_per_epoch,
            best_psnr=float(checkpoint.get("best_psnr", 0.0)),
            mode="resume",
        )

    pretrained_path = args.pretrained_checkpoint_path
    if pretrained_path is not None:
        pretrained_path = Path(pretrained_path)
        logger.info("Loading pretrained checkpoint from %s", pretrained_path)
        checkpoint = torch.load(str(pretrained_path), map_location=device)
        model.load_state_dict(extract_pretrained_state_dict(checkpoint, pretrained_path, torch))
        return TrainingState(
            start_epoch=0,
            global_step=0,
            best_psnr=0.0,
            mode="pretrained",
        )

    logger.info("Training %s from scratch", args.model_name)
    return TrainingState(
        start_epoch=0,
        global_step=0,
        best_psnr=0.0,
        mode="scratch",
    )


def build_dry_run_summary(args: argparse.Namespace) -> dict[str, object]:
    summary: dict[str, object] = {
        "mode": args.mode,
        "model_name": args.model_name,
        "train_preset": args.train_preset,
        "test_preset": args.test_preset,
        "active_root_key": ACTIVE_DATASET_ROOT_KEY,
        "dataset_root_dir": args.dataset_root_dir,
        "csv_root_dir": str(Path(args.root_dir)),
        "output_dir": args.output_dir,
        "resume_path": args.resume_path,
        "pretrained_checkpoint_path": args.pretrained_checkpoint_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_interval": args.eval_interval,
        "input_fps": args.input_fps,
        "only_fps": args.only_fps,
        "dataset_class": resolve_dataset_class_name(args.model_name),
        "metrics": dict(args.metric_config),
    }
    if args.eval_convex_upsampling is not None:
        summary["eval_convex_upsampling"] = args.eval_convex_upsampling

    if len(args.model_init_args) > 0:
        summary["model_init_args"] = dict(args.model_init_args)

    if uses_flow_approx_model(args.model_name):
        summary["flow_approx_method"] = args.flow_approx_method
        summary["splatting_fill_strategy"] = args.splatting_fill_strategy
        summary["effective_splatting_fill_strategy"] = resolve_effective_splatting_fill_strategy(args=args)
        summary["init_flow_downscale_strategy"] = args.init_flow_downscale_strategy
        summary["effective_init_flow_downscale_strategy"] = resolve_effective_init_flow_downscale_strategy(args=args)
        summary["init_flow_mask_epsilon"] = args.init_flow_mask_epsilon

    return summary


def save_input_config(target_dir: Path, input_config: dict[str, Any]) -> Path:
    config_path = target_dir / "input_config.json"
    config_path.write_text(json.dumps(input_config, indent=2), encoding="utf-8")
    return config_path


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
    logger.info("model_name=%s model_init_args=%s", args.model_name, args.model_init_args)
    logger.info(
        "train_preset=%s test_preset=%s input_fps=%s only_fps=%s",
        args.train_preset,
        args.test_preset,
        args.input_fps,
        args.only_fps,
    )
    logger.info(
        "epochs=%s batch_size=%s eval_interval=%s lr_start=%s lr_end=%s seed=%s",
        args.epochs,
        args.batch_size,
        args.eval_interval,
        args.lr_start,
        args.lr_end,
        args.seed,
    )
    logger.info(
        "resume_path=%s pretrained_checkpoint_path=%s output_dir=%s",
        args.resume_path,
        args.pretrained_checkpoint_path,
        args.output_dir,
    )
    logger.info("metrics=%s", args.metric_config)
    if args.eval_convex_upsampling is not None:
        logger.info("eval_convex_upsampling=%s", args.eval_convex_upsampling)
    if uses_flow_approx_model(args.model_name):
        logger.info("flow_approx_method=%s", args.flow_approx_method)
        logger.info(
            "splatting_fill_strategy=%s effective_splatting_fill_strategy=%s",
            args.splatting_fill_strategy,
            resolve_effective_splatting_fill_strategy(args=args),
        )
        logger.info(
            "init_flow_downscale_strategy=%s effective_init_flow_downscale_strategy=%s init_flow_mask_epsilon=%s",
            args.init_flow_downscale_strategy,
            resolve_effective_init_flow_downscale_strategy(args=args),
            args.init_flow_mask_epsilon,
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
    model_init_args = dict(getattr(args, "model_init_args", {}))
    model = model_class(**model_init_args).to(device)
    if hasattr(model, "init_flow_layer"):
        logger.info("model_init_flow_layer=%s", model.init_flow_layer)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0)
    training_state = load_training_state(args, model, optimizer, device, logger)

    log_run_summary(args, train_dataset, test_dataset, training_state, device, logger)
    logger.info("run_log_dir=%s", run_dir)
    input_config_path = save_input_config(target_dir=run_dir, input_config=args.input_config)
    logger.info("input_config_path=%s", input_config_path)

    train(
        args,
        model,
        optimizer,
        train_loader,
        test_loader,
        device,
        logger,
        training_state,
        {"train": train_df, "test": test_df},
        lpips_model,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_train_args(argv)

    if args.mode == "dry-run":
        print(json.dumps(build_dry_run_summary(args), indent=2))
        return

    run_training(args)


if __name__ == "__main__":
    main()
