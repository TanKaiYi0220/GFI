from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import list_dataset_presets
from src.engine.dataset_runs import build_merged_dataframe
from src.engine.dataset_runs import build_training_dataset
from src.engine.dataset_runs import filter_valid_dataframe
from src.engine.dataset_runs import resolve_dataset_class_name
from src.engine.checkpoints import load_training_state
from src.engine.checkpoints import save_checkpoint
from src.engine.checkpoints import TrainingState
from src.engine.evaluation import AverageMeter
from src.engine.evaluation import average_metric_values
from src.engine.evaluation import build_flip_evaluator
from src.engine.evaluation import build_flolpips_model
from src.engine.evaluation import build_lpips_model
from src.engine.evaluation import build_metric_meters
from src.engine.evaluation import calculate_batch_metrics
from src.engine.evaluation import format_metric_averages
from src.engine.evaluation import format_metric_values
from src.engine.evaluation import get_enabled_metric_names
from src.engine.flow_approx import DEFAULT_SPLATTING_FILL_STRATEGY
from src.engine.flow_approx import FLOW_APPROX_METHOD_CHOICES
from src.engine.flow_approx import SPLATTING_FILL_STRATEGIES
from src.engine.interpolation_batch import run_training_batch
from src.engine.model_registry import MODEL_NAMES
from src.engine.model_registry import resolve_model_class
from src.engine.model_registry import set_model_convex_upsampling
from src.engine.model_registry import uses_flow_approx_model
from src.engine.run_config import build_train_dry_run_summary
from src.engine.run_config import build_train_run_config
from src.engine.run_config import DEFAULT_INIT_FLOW_DOWNSCALE_STRATEGY
from src.engine.run_config import DEFAULT_INIT_FLOW_MASK_EPSILON
from src.engine.run_config import INIT_FLOW_DOWNSCALE_STRATEGIES
from src.engine.run_config import parse_eval_convex_upsampling_arg
from src.engine.run_config import TrainRunConfig
from src.utils.config import load_yaml_file
from src.utils.seed import set_seed


def get_lr(config: TrainRunConfig, step: int, iters_per_epoch: int) -> float:
    total_steps = max(config.epochs * iters_per_epoch, 1)
    ratio = 0.5 * (1.0 + math.cos(step / total_steps * math.pi))
    return (config.lr_start - config.lr_end) * ratio + config.lr_end


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
    img0: Any,
    img1: Any,
    imgt: Any,
    loss_record: dict[str, float],
    metric_config: dict[str, object],
    lpips_model: Any | None,
    flolpips_model: Any | None,
) -> None:
    batch_size = int(imgt_pred.shape[0])
    normalized_loss_record = {metric_name: float(metric_value) for metric_name, metric_value in loss_record.items()}
    batch_metric_values = calculate_batch_metrics(
        target=imgt.detach(),
        prediction=imgt_pred.detach(),
        metric_config=metric_config,
        lpips_model=lpips_model,
        flolpips_model=flolpips_model,
        img0=img0.detach(),
        img1=img1.detach(),
    )

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


def build_sample_artifact_inputs(batch_output: Any) -> dict[str, Any]:
    return {
        "img0": batch_output.img0,
        "img1": batch_output.img1,
        "imgt": batch_output.imgt,
        "bmv": batch_output.bmv,
        "fmv": batch_output.fmv,
        "imgt_pred": batch_output.imgt_pred,
        "imgt_merge": batch_output.imgt_merge,
        "init_bmv": batch_output.init_bmv,
        "init_fmv": batch_output.init_fmv,
        "init_masks": batch_output.init_masks,
        "splatting_region_maps": batch_output.splatting_region_maps,
        "up_flow0_1": batch_output.up_flow0_1,
        "up_flow1_1": batch_output.up_flow1_1,
        "up_mask_1": batch_output.up_mask_1,
    }


def save_epoch_samples(
    config: TrainRunConfig,
    model: Any,
    sample_dataframes: dict[str, Any],
    epoch: int,
    device: Any,
    logger: logging.Logger,
) -> None:
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from scripts.inference import save_selected_sample_artifacts
    from src.data.image_ops import flow_to_image, save_image

    frame_groups = {"train": config.sample_train_frames, "test": config.sample_test_frames}
    if sum(len(frame_keys) for frame_keys in frame_groups.values()) == 0:
        return
    if config.sample_interval_epoch <= 0:
        raise ValueError(f"sample_interval_epoch must be positive, got {config.sample_interval_epoch}")
    if (epoch + 1) % config.sample_interval_epoch != 0:
        return

    previous_convex_upsampling = None
    if config.model.eval_convex_upsampling is not None:
        previous_convex_upsampling = set_model_convex_upsampling(
            model=model,
            enabled=config.model.eval_convex_upsampling,
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
                    config.dataset_root_dir,
                    False,
                    config.input_fps,
                    config.model.model_name,
                    config.flow_approx.method,
                )
                for frame_key, batch in zip(frame_keys, DataLoader(sample_dataset, batch_size=1, shuffle=False)):
                    save_dir = Path(config.output_dir) / "samples" / split_name / frame_key / f"epoch_{epoch + 1:04d}"
                    batch_output = run_training_batch(
                        config=config,
                        model=model,
                        batch=batch,
                        device=device,
                        collect_visual_artifacts=True,
                    )
                    save_selected_sample_artifacts(
                        cv2,
                        99.0,
                        1.0,
                        flow_to_image,
                        build_sample_artifact_inputs(batch_output),
                        np,
                        save_dir,
                        save_image,
                    )
                    logger.info("Saved sample frame split=%s frame=%s epoch=%s dir=%s", split_name, frame_key, epoch + 1, save_dir)
    finally:
        if previous_convex_upsampling is not None:
            set_model_convex_upsampling(
                model=model,
                enabled=previous_convex_upsampling,
                context="sample evaluation restore",
            )


def evaluate(
    config: TrainRunConfig,
    model: Any,
    loader: Any,
    device: Any,
    lpips_model: Any | None,
    flolpips_model: Any | None,
) -> tuple[float, Any, Any, dict[str, float]]:
    import pandas as pd
    import torch
    from tqdm import tqdm

    previous_convex_upsampling = None
    if config.model.eval_convex_upsampling is not None:
        previous_convex_upsampling = set_model_convex_upsampling(
            model=model,
            enabled=config.model.eval_convex_upsampling,
            context="training evaluation",
        )

    metric_meters = build_metric_meters(config.metrics.values)
    eval_records: list[dict[str, object]] = []

    try:
        model.eval()
        with torch.no_grad():
            pbar = tqdm(loader, desc="Evaluating")
            for batch in pbar:
                batch_output = run_training_batch(
                    config=config,
                    model=model,
                    batch=batch,
                    device=device,
                    collect_visual_artifacts=False,
                )
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
                    batch_output.img0,
                    batch_output.img1,
                    batch_output.imgt,
                    loss_record,
                    config.metrics.values,
                    lpips_model,
                    flolpips_model,
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
    record_name_df = build_record_name_summary(eval_df, config.metrics.values)
    return metric_meters["psnr"].avg, eval_df, record_name_df, average_metric_values(metric_meters)


def train(
    config: TrainRunConfig,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    logger: logging.Logger,
    training_state: TrainingState,
    sample_dataframes: dict[str, Any],
    lpips_model: Any | None,
    flolpips_model: Any | None,
) -> None:
    import pandas as pd
    from tqdm import tqdm

    best_psnr = training_state.best_psnr
    global_step = training_state.global_step
    checkpoints_dir = Path(config.output_dir) / "checkpoints"
    train_steps_per_epoch = len(train_loader)

    for epoch in range(training_state.start_epoch, config.epochs):
        model.train()
        train_metric_meters = build_metric_meters(config.metrics.values)
        train_loss_total_meter = AverageMeter()
        train_loss_rec_meter = AverageMeter()
        train_loss_geo_meter = AverageMeter()
        train_loss_dis_meter = AverageMeter()
        train_records: list[dict[str, object]] = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for batch in pbar:
            lr = get_lr(config, global_step, train_steps_per_epoch)
            set_lr(optimizer, lr)
            optimizer.zero_grad()

            batch_output = run_training_batch(
                config=config,
                model=model,
                batch=batch,
                device=device,
                collect_visual_artifacts=False,
            )
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
            if (epoch + 1) % config.eval_interval != 0:
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
                batch_output.img0,
                batch_output.img1,
                batch_output.imgt,
                loss_record,
                config.metrics.values,
                lpips_model,
                flolpips_model,
            )

        if (epoch + 1) % config.eval_interval == 0:
            train_df = pd.DataFrame(train_records)
            train_record_name_df = build_record_name_summary(train_df, config.metrics.values)
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

        if (epoch + 1) % config.eval_interval == 0:
            test_psnr, test_df, test_record_name_df, test_metric_values = evaluate(
                config=config,
                model=model,
                loader=test_loader,
                device=device,
                lpips_model=lpips_model,
                flolpips_model=flolpips_model,
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
        save_epoch_samples(config, model, sample_dataframes, epoch, device, logger)


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
    args.input_config = config_defaults
    return args


def parse_train_config(argv: list[str] | None) -> TrainRunConfig:
    args = parse_train_args(argv)
    return build_train_run_config(args=args, config_defaults=args.input_config)


def log_run_summary(
    config: TrainRunConfig,
    train_dataset: Any,
    test_dataset: Any,
    training_state: TrainingState,
    device: Any,
    logger: logging.Logger,
) -> None:
    logger.info(
        "model=%s mode=%s dataset_class=%s device=%s output_dir=%s dataset_root_dir=%s train_samples=%s test_samples=%s start_epoch=%s epochs=%s batch_size=%s",
        config.model.model_name,
        training_state.mode,
        resolve_dataset_class_name(config.model.model_name),
        device,
        config.output_dir,
        config.dataset_root_dir,
        len(train_dataset),
        len(test_dataset),
        training_state.start_epoch,
        config.epochs,
        config.batch_size,
    )


def save_input_config(target_dir: Path, input_config: dict[str, Any]) -> Path:
    config_path = target_dir / "input_config.json"
    config_path.write_text(json.dumps(input_config, indent=2), encoding="utf-8")
    return config_path


def run_training(config: TrainRunConfig) -> None:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    output_dir = Path(config.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger, run_dir = build_logger(output_dir)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_config = dict(config.metrics.values)
    lpips_model = build_lpips_model(metric_config, device)
    flolpips_model = build_flolpips_model(metric_config, device)
    build_flip_evaluator(metric_config)
    logger.info("device=%s", device)
    logger.info("model_name=%s model_init_args=%s", config.model.model_name, config.model.model_init_args)
    logger.info(
        "train_preset=%s test_preset=%s input_fps=%s only_fps=%s",
        config.train_preset,
        config.test_preset,
        config.input_fps,
        config.only_fps,
    )
    logger.info(
        "epochs=%s batch_size=%s eval_interval=%s lr_start=%s lr_end=%s seed=%s",
        config.epochs,
        config.batch_size,
        config.eval_interval,
        config.lr_start,
        config.lr_end,
        config.seed,
    )
    logger.info(
        "resume_path=%s pretrained_checkpoint_path=%s output_dir=%s",
        config.resume_path,
        config.pretrained_checkpoint_path,
        config.output_dir,
    )
    logger.info("metrics=%s", metric_config)
    if config.model.eval_convex_upsampling is not None:
        logger.info("eval_convex_upsampling=%s", config.model.eval_convex_upsampling)
    if uses_flow_approx_model(config.model.model_name):
        logger.info("flow_approx_method=%s", config.flow_approx.method)
        logger.info(
            "splatting_fill_strategy=%s effective_splatting_fill_strategy=%s",
            config.flow_approx.splatting_fill_strategy,
            config.flow_approx.effective_splatting_fill_strategy,
        )
        logger.info(
            "init_flow_downscale_strategy=%s effective_init_flow_downscale_strategy=%s init_flow_mask_epsilon=%s",
            config.flow_approx.init_flow_downscale_strategy,
            config.flow_approx.effective_init_flow_downscale_strategy,
            config.flow_approx.init_flow_mask_epsilon,
        )

    train_df = build_merged_dataframe(config.root_dir, checkpoints_dir, config.train_preset, config.only_fps, logger)
    test_df = build_merged_dataframe(config.root_dir, checkpoints_dir, config.test_preset, config.only_fps, logger)

    if "valid" in train_df.columns:
        logger.info("Valid Count %s in %s", train_df["valid"].value_counts().to_dict(), config.train_preset)
    if "valid" in test_df.columns:
        logger.info("Valid Count %s in %s", test_df["valid"].value_counts().to_dict(), config.test_preset)
    train_df = filter_valid_dataframe(train_df)
    test_df = filter_valid_dataframe(test_df)

    train_dataset = build_training_dataset(
        train_df,
        config.dataset_root_dir,
        True,
        config.input_fps,
        config.model.model_name,
        config.flow_approx.method,
    )
    test_dataset = build_training_dataset(
        test_df,
        config.dataset_root_dir,
        False,
        config.input_fps,
        config.model.model_name,
        config.flow_approx.method,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    iters_per_epoch = len(train_loader)
    model_class = resolve_model_class(config.model.model_name)
    model_init_args = dict(config.model.model_init_args)
    model = model_class(**model_init_args).to(device)
    if hasattr(model, "init_flow_layer"):
        logger.info("model_init_flow_layer=%s", model.init_flow_layer)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_start, weight_decay=0)
    training_state = load_training_state(
        resume_path=config.resume_path,
        pretrained_checkpoint_path=config.pretrained_checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=device,
        logger=logger,
        iters_per_epoch=iters_per_epoch,
        model_name=config.model.model_name,
    )

    log_run_summary(config, train_dataset, test_dataset, training_state, device, logger)
    logger.info("run_log_dir=%s", run_dir)
    input_config_path = save_input_config(target_dir=run_dir, input_config=config.input_config)
    logger.info("input_config_path=%s", input_config_path)

    train(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        logger=logger,
        training_state=training_state,
        sample_dataframes={"train": train_df, "test": test_df},
        lpips_model=lpips_model,
        flolpips_model=flolpips_model,
    )


def main(argv: list[str] | None = None) -> None:
    config = parse_train_config(argv)

    if config.mode == "dry-run":
        print(json.dumps(build_train_dry_run_summary(config=config), indent=2))
        return

    run_training(config=config)


if __name__ == "__main__":
    main()
