from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import TrainingState
from scripts.train import build_logger
from scripts.train import build_loss_record
from scripts.train import build_merged_dataframe
from scripts.train import build_train_arg_parser
from scripts.train import build_dry_run_summary
from scripts.train import forward_model
from scripts.train import get_lr
from scripts.train import load_train_run_config
from scripts.train import load_training_state
from scripts.train import log_run_summary
from scripts.train import prepare_args
from scripts.train import save_checkpoint
from scripts.train import set_lr
from scripts.train import set_seed
from src.data.dataset_loader import CachedVFITrainDataset
from src.data.dataset_loader import VFITrainDataset
from src.engine.evaluation import TaskEvaluator
from src.engine.evaluation import VFI_METRICS
from src.models.registry import get_model_class


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def move_tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    use_non_blocking = device.type == "cuda"
    return tensor.to(device, non_blocking=use_non_blocking)


def append_batch_loss_records(
    target_records: list[dict[str, float]],
    loss_record: dict[str, float],
    batch_size: int,
) -> None:
    target_records.extend(loss_record.copy() for _ in range(batch_size))


def parse_bool_argument(value: str) -> bool:
    normalized_value = value.strip().lower()
    if normalized_value in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"Expected a boolean-like value, got: {value}")


def build_timing_arg_parser(config_defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = build_train_arg_parser(config_defaults)
    parser.description = "Train IFRNet variants with detailed timing logs."
    parser.add_argument(
        "--num-workers",
        default=config_defaults.get("num_workers", 0),
        type=int,
        help="Training dataloader worker count.",
    )
    parser.add_argument(
        "--pin-memory",
        default=config_defaults.get("pin_memory", True),
        type=parse_bool_argument,
        help="Whether dataloaders should pin host memory.",
    )
    parser.add_argument(
        "--persistent-workers",
        default=config_defaults.get("persistent_workers", False),
        type=parse_bool_argument,
        help="Whether dataloader workers should stay alive between epochs.",
    )
    parser.add_argument(
        "--prefetch-factor",
        default=config_defaults.get("prefetch_factor", 2),
        type=int,
        help="Number of prefetched batches per worker when num_workers > 0.",
    )
    parser.add_argument(
        "--timing-log-interval",
        default=20,
        type=int,
        help="Log one timing summary every N training steps.",
    )
    parser.add_argument(
        "--timing-warmup-steps",
        default=5,
        type=int,
        help="Ignore the first N training steps in aggregated timing summaries.",
    )
    return parser


def parse_train_timing_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default=None, type=str, help="Optional JSON-formatted YAML-compatible run config.")
    bootstrap_args, _remaining_argv = bootstrap_parser.parse_known_args(argv)

    config_path = None if bootstrap_args.config is None else Path(bootstrap_args.config)
    config_defaults = load_train_run_config(config_path)
    parser = build_timing_arg_parser(config_defaults)
    return parser.parse_args(argv)


def build_step_timing_record(
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    batch_size: int,
    data_wait_seconds: float,
    host_to_device_seconds: float,
    forward_seconds: float,
    backward_seconds: float,
    metric_seconds: float,
    step_total_seconds: float,
) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "batch_size": batch_size,
        "data_wait_seconds": data_wait_seconds,
        "host_to_device_seconds": host_to_device_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "metric_seconds": metric_seconds,
        "step_total_seconds": step_total_seconds,
    }


def build_dataloader_kwargs(
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> dict[str, Any]:
    dataloader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return dataloader_kwargs


def build_timing_summary(records: list[dict[str, float | int]], timing_warmup_steps: int) -> dict[str, float]:
    if len(records) == 0:
        return {}

    timing_df = pd.DataFrame(records)
    filtered_df = timing_df[timing_df["global_step"] > timing_warmup_steps]
    summary_df = filtered_df if len(filtered_df) > 0 else timing_df

    total_batch = float(summary_df["batch_size"].sum())
    total_step_time = float(summary_df["step_total_seconds"].sum())
    samples_per_second = total_batch / total_step_time if total_step_time > 0.0 else 0.0

    return {
        "data_wait_seconds": float(summary_df["data_wait_seconds"].mean()),
        "host_to_device_seconds": float(summary_df["host_to_device_seconds"].mean()),
        "forward_seconds": float(summary_df["forward_seconds"].mean()),
        "backward_seconds": float(summary_df["backward_seconds"].mean()),
        "metric_seconds": float(summary_df["metric_seconds"].mean()),
        "step_total_seconds": float(summary_df["step_total_seconds"].mean()),
        "samples_per_second": samples_per_second,
    }


def log_timing_summary(
    logger: Any,
    label: str,
    summary: dict[str, float],
) -> None:
    if len(summary) == 0:
        return

    logger.info(
        "%s timing avg data_wait=%.4fs h2d=%.4fs forward=%.4fs backward=%.4fs metric=%.4fs step_total=%.4fs samples_per_second=%.2f",
        label,
        summary["data_wait_seconds"],
        summary["host_to_device_seconds"],
        summary["forward_seconds"],
        summary["backward_seconds"],
        summary["metric_seconds"],
        summary["step_total_seconds"],
        summary["samples_per_second"],
    )


def evaluate(
    model_name: str,
    model: Any,
    loader: Any,
    device: torch.device,
) -> tuple[float, Any]:
    model.eval()
    evaluator = TaskEvaluator("VFI", VFI_METRICS)
    loss_records: list[dict[str, float]] = []

    with torch.no_grad():
        for batch in tqdm(loader):
            img0, imgt, img1, bmv, fmv, embt, _info = batch
            img0 = move_tensor_to_device(img0, device)
            img1 = move_tensor_to_device(img1, device)
            imgt = move_tensor_to_device(imgt, device)
            bmv = move_tensor_to_device(bmv, device)
            fmv = move_tensor_to_device(fmv, device)
            embt = move_tensor_to_device(embt, device)

            imgt_pred, loss_rec, loss_geo, loss_dis, _up_flow0_1, _up_flow1_1, _up_mask_1 = forward_model(
                model_name,
                model,
                img0,
                img1,
                embt,
                imgt,
                bmv,
                fmv,
            )

            total_loss = loss_rec + loss_geo + loss_dis
            loss_record = build_loss_record(loss_rec, loss_geo, loss_dis, total_loss)
            batch_size = int(imgt_pred.shape[0])
            evaluator.evaluate_batch(img_gt=imgt, img_pred=imgt_pred)
            append_batch_loss_records(loss_records, loss_record, batch_size)

    metric_df = evaluator.to_dataframe()
    loss_df = pd.DataFrame(loss_records)
    eval_df = pd.concat([metric_df.reset_index(drop=True), loss_df.reset_index(drop=True)], axis=1)
    return float(eval_df["psnr"].mean()), eval_df


def train_with_timing(
    args: argparse.Namespace,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    test_loader: Any,
    device: torch.device,
    logger: Any,
    run_dir: Path,
    training_state: TrainingState,
) -> None:
    best_psnr = training_state.best_psnr
    global_step = training_state.global_step
    checkpoints_dir = Path(args.output_dir) / "checkpoints"
    step_timing_records: list[dict[str, float | int]] = []
    epoch_timing_records: list[dict[str, float | int]] = []

    for epoch in range(training_state.start_epoch, args.epochs):
        model.train()
        evaluator = TaskEvaluator("VFI", VFI_METRICS)
        train_loss_records: list[dict[str, float]] = []
        epoch_step_timing_records: list[dict[str, float | int]] = []
        previous_step_end = time.perf_counter()

        for step_in_epoch, batch in enumerate(tqdm(train_loader), start=1):
            batch_ready_time = time.perf_counter()
            data_wait_seconds = batch_ready_time - previous_step_end

            img0, imgt, img1, bmv, fmv, embt, _info = batch

            synchronize_device(device)
            host_to_device_start = time.perf_counter()
            img0 = move_tensor_to_device(img0, device)
            img1 = move_tensor_to_device(img1, device)
            imgt = move_tensor_to_device(imgt, device)
            bmv = move_tensor_to_device(bmv, device)
            fmv = move_tensor_to_device(fmv, device)
            embt = move_tensor_to_device(embt, device)
            synchronize_device(device)
            host_to_device_end = time.perf_counter()

            lr = get_lr(args, global_step)
            set_lr(optimizer, lr)
            optimizer.zero_grad()

            synchronize_device(device)
            forward_start = time.perf_counter()
            imgt_pred, loss_rec, loss_geo, loss_dis, _up_flow0_1, _up_flow1_1, _up_mask_1 = forward_model(
                args.model_name,
                model,
                img0,
                img1,
                embt,
                imgt,
                bmv,
                fmv,
            )
            synchronize_device(device)
            forward_end = time.perf_counter()

            total_loss = loss_rec + loss_geo + loss_dis

            synchronize_device(device)
            backward_start = time.perf_counter()
            total_loss.backward()
            optimizer.step()
            synchronize_device(device)
            backward_end = time.perf_counter()

            metric_start = time.perf_counter()
            loss_record = build_loss_record(loss_rec, loss_geo, loss_dis, total_loss)
            batch_size = int(imgt_pred.shape[0])
            evaluator.evaluate_batch(img_gt=imgt.detach(), img_pred=imgt_pred.detach())
            append_batch_loss_records(train_loss_records, loss_record, batch_size)
            metric_end = time.perf_counter()

            step_end = time.perf_counter()
            previous_step_end = step_end

            global_step += 1
            step_timing_record = build_step_timing_record(
                epoch=epoch + 1,
                step_in_epoch=step_in_epoch,
                global_step=global_step,
                batch_size=batch_size,
                data_wait_seconds=data_wait_seconds,
                host_to_device_seconds=host_to_device_end - host_to_device_start,
                forward_seconds=forward_end - forward_start,
                backward_seconds=backward_end - backward_start,
                metric_seconds=metric_end - metric_start,
                step_total_seconds=step_end - batch_ready_time,
            )
            step_timing_records.append(step_timing_record)
            epoch_step_timing_records.append(step_timing_record)

            if args.timing_log_interval > 0 and step_in_epoch % args.timing_log_interval == 0:
                timing_window = epoch_step_timing_records[-args.timing_log_interval :]
                timing_summary = build_timing_summary(timing_window, timing_warmup_steps=0)
                log_timing_summary(
                    logger=logger,
                    label=f"Epoch {epoch + 1} step {step_in_epoch}",
                    summary=timing_summary,
                )

        epoch_timing_summary = build_timing_summary(epoch_step_timing_records, args.timing_warmup_steps)
        log_timing_summary(logger=logger, label=f"Epoch {epoch + 1}", summary=epoch_timing_summary)

        if len(epoch_timing_summary) > 0:
            epoch_timing_records.append(
                {
                    "epoch": epoch + 1,
                    **epoch_timing_summary,
                }
            )

        if (epoch + 1) % args.eval_interval == 0:
            train_metric_df = evaluator.to_dataframe()
            train_loss_df = pd.DataFrame(train_loss_records)
            train_df = pd.concat([train_metric_df.reset_index(drop=True), train_loss_df.reset_index(drop=True)], axis=1)
            train_psnr = float(train_df["psnr"].mean())
            train_df.to_csv(checkpoints_dir / f"train_epoch_{epoch + 1}.csv", index=False)
            logger.info("Epoch %s train_psnr=%.6f", epoch + 1, train_psnr)

            test_psnr, test_df = evaluate(args.model_name, model, test_loader, device)
            test_df.to_csv(checkpoints_dir / f"test_epoch_{epoch + 1}.csv", index=False)
            logger.info("Epoch %s test_psnr=%.6f", epoch + 1, test_psnr)

            if test_psnr > best_psnr:
                best_psnr = test_psnr
                save_checkpoint(checkpoints_dir / "best.pth", model, optimizer, epoch, best_psnr)

        save_checkpoint(checkpoints_dir / "latest.pth", model, optimizer, epoch, best_psnr)
        pd.DataFrame(step_timing_records).to_csv(run_dir / "train_step_timing.csv", index=False)
        pd.DataFrame(epoch_timing_records).to_csv(run_dir / "train_epoch_timing.csv", index=False)

    overall_timing_summary = build_timing_summary(step_timing_records, args.timing_warmup_steps)
    log_timing_summary(logger=logger, label="Overall", summary=overall_timing_summary)


def run_training(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger, run_dir = build_logger(output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    root_dir = Path(args.root_dir)
    if args.train_cache_manifest is None:
        train_df = build_merged_dataframe(root_dir, checkpoints_dir, args.train_preset, args.only_fps, logger)
        if "valid" in train_df.columns:
            logger.info("Valid Count %s in %s", train_df["valid"].value_counts().to_dict(), args.train_preset)
            train_df = train_df[train_df["valid"] == True]
        train_dataset = VFITrainDataset(train_df, args.dataset_root_dir, True, args.input_fps)
    else:
        logger.info("Using cached training manifest %s", args.train_cache_manifest)
        train_dataset = CachedVFITrainDataset(args.train_cache_manifest, True)

    if args.test_cache_manifest is None:
        test_df = build_merged_dataframe(root_dir, checkpoints_dir, args.test_preset, args.only_fps, logger)
        if "valid" in test_df.columns:
            logger.info("Valid Count %s in %s", test_df["valid"].value_counts().to_dict(), args.test_preset)
            test_df = test_df[test_df["valid"] == True]
        test_dataset = VFITrainDataset(test_df, args.dataset_root_dir, False, args.input_fps)
    else:
        logger.info("Using cached validation manifest %s", args.test_cache_manifest)
        test_dataset = CachedVFITrainDataset(args.test_cache_manifest, False)
    train_loader_kwargs = build_dataloader_kwargs(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader_kwargs = build_dataloader_kwargs(
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        **test_loader_kwargs,
    )

    args.iters_per_epoch = len(train_loader)
    model_class = get_model_class(args.model_name)
    model = model_class().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0)
    training_state = load_training_state(args, model, optimizer, device, logger)

    log_run_summary(args, train_dataset, test_dataset, training_state, device, logger)
    logger.info("run_log_dir=%s", run_dir)
    logger.info(
        "dataloader_config num_workers=%s pin_memory=%s persistent_workers=%s prefetch_factor=%s",
        args.num_workers,
        args.pin_memory,
        args.persistent_workers,
        args.prefetch_factor,
    )

    train_with_timing(
        args=args,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        logger=logger,
        run_dir=run_dir,
        training_state=training_state,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_train_timing_args(argv)
    args = prepare_args(args)

    if args.mode == "dry-run":
        summary = build_dry_run_summary(args)
        summary["num_workers"] = args.num_workers
        summary["pin_memory"] = args.pin_memory
        summary["persistent_workers"] = args.persistent_workers
        summary["prefetch_factor"] = args.prefetch_factor
        summary["timing_log_interval"] = args.timing_log_interval
        summary["timing_warmup_steps"] = args.timing_warmup_steps
        print(json.dumps(summary, indent=2))
        return

    run_training(args)


if __name__ == "__main__":
    main()
