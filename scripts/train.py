from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import ACTIVE_DATASET_ROOT_KEY
from src.data.dataset_config import get_dataset_preset
from src.data.dataset_config import list_dataset_presets
from src.data.dataset_config import resolve_active_dataset_root
from src.data.dataset_config import iter_dataset_configs
from src.data.dataset_loader import VFITrainDataset
from src.engine.evaluation import TaskEvaluator
from src.engine.evaluation import VFI_METRICS
from src.engine.evaluation import calculate_psnr_skimage
from src.engine.evaluation import calculate_psnr_torch
from src.models.registry import get_model_class
from src.models.registry import get_model_config
from src.utils.config import load_yaml_file


@dataclass(frozen=True)
class TrainingState:
    start_epoch: int
    global_step: int
    best_psnr: float
    mode: str


def set_seed(seed: int) -> None:
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


def read_resume_start_epoch(checkpoint_path: Path) -> int | None:
    if not checkpoint_path.is_file():
        return None

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    epoch = checkpoint.get("epoch")
    if epoch is None:
        return None

    return int(epoch) + 1


def build_unique_output_dir(base_dir: Path) -> str:
    if not base_dir.exists():
        return str(base_dir)

    suffix_index = 1
    while True:
        candidate = base_dir.parent / f"{base_dir.name}_{suffix_index:02d}"
        if not candidate.exists():
            return str(candidate)
        suffix_index += 1


def build_resume_output_dir(resume_path: str, start_epoch: int | None) -> str:
    checkpoint_path = Path(resume_path)
    source_output_dir = checkpoint_path.parent.parent
    checkpoint_name = checkpoint_path.stem
    resume_epoch_label = f"e{start_epoch}" if start_epoch is not None else "resume"
    base_dir = source_output_dir.parent / f"{source_output_dir.name}_{checkpoint_name}_{resume_epoch_label}"
    return build_unique_output_dir(base_dir)


def resolve_resume_path(user_resume_path: str | None, default_resume_path: str | None) -> str | None:
    if user_resume_path is not None:
        resume_path = Path(user_resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        return str(resume_path)

    if default_resume_path is None:
        return None

    resume_path = Path(default_resume_path)
    if resume_path.is_file():
        return str(resume_path)

    return None


def resolve_output_dir(
    output_dir: str | None,
    resume_path: str | None,
    default_output_dir: str,
) -> tuple[str, str]:
    if output_dir is not None:
        return str(Path(output_dir)), "user"

    if resume_path is not None:
        start_epoch = read_resume_start_epoch(Path(resume_path))
        return build_resume_output_dir(resume_path, start_epoch), "auto_resume"

    return build_unique_output_dir(Path(default_output_dir)), "auto_fresh"


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
) -> Any:
    dataset_preset = get_dataset_preset(dataset_preset_name)
    dataframe_list: list[Any] = []

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
    bmv: Any,
    fmv: Any,
) -> Any:
    if model_name == "IFRNet":
        flow = torch.cat([bmv, fmv], dim=1).float()
        return model(img0, img1, embt, imgt, flow)

    return model(img0, img1, embt, imgt, init_flow0=bmv, init_flow1=fmv)


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


def append_batch_loss_records(
    target_records: list[dict[str, float]],
    loss_record: dict[str, float],
    batch_size: int,
) -> None:
    target_records.extend(loss_record.copy() for _ in range(batch_size))


def save_checkpoint(
    checkpoint_path: Path,
    model: Any,
    optimizer: Any,
    epoch: int,
    best_psnr: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
        },
        str(checkpoint_path),
    )


def evaluate(
    model_name: str,
    model: Any,
    loader: Any,
    device: Any,
) -> tuple[float, Any]:
    model.eval()
    evaluator = TaskEvaluator("VFI", VFI_METRICS)
    loss_records: list[dict[str, float]] = []

    with torch.no_grad():
        for batch in tqdm(loader):
            img0, imgt, img1, bmv, fmv, embt, _info = batch
            img0 = img0.to(device)
            img1 = img1.to(device)
            imgt = imgt.to(device)
            bmv = bmv.to(device)
            fmv = fmv.to(device)
            embt = embt.to(device)

            imgt_pred, loss_rec, loss_geo, loss_dis, up_flow0_1, up_flow1_1, up_mask_1 = forward_model(
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

            for b in range(batch_size):
                psnr = calculate_psnr_torch(img_gt=imgt[b], img_pred=imgt_pred[b])
                # sklearn_psnr = calculate_psnr_skimage(img_gt=imgt[b], img_pred=imgt_pred[b])
                evaluator.records.append({"psnr": float(psnr)})

            append_batch_loss_records(loss_records, loss_record, batch_size)

    metric_df = evaluator.to_dataframe()
    loss_df = pd.DataFrame(loss_records)
    eval_df = pd.concat([metric_df.reset_index(drop=True), loss_df.reset_index(drop=True)], axis=1)
    return float(eval_df["psnr"].mean()), eval_df


def train(
    args: argparse.Namespace,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    logger: logging.Logger,
    training_state: TrainingState,
) -> None:
    best_psnr = training_state.best_psnr
    global_step = training_state.global_step
    checkpoints_dir = Path(args.output_dir) / "checkpoints"

    for epoch in range(training_state.start_epoch, args.epochs):
        model.train()
        evaluator = TaskEvaluator("VFI", VFI_METRICS)
        train_loss_records: list[dict[str, float]] = []

        for batch in tqdm(train_loader):
            img0, imgt, img1, bmv, fmv, embt, _info = batch
            img0 = img0.to(device)
            img1 = img1.to(device)
            imgt = imgt.to(device)
            bmv = bmv.to(device)
            fmv = fmv.to(device)
            embt = embt.to(device)

            lr = get_lr(args, global_step)
            set_lr(optimizer, lr)
            optimizer.zero_grad()

            imgt_pred, loss_rec, loss_geo, loss_dis, up_flow0_1, up_flow1_1, up_mask_1 = forward_model(
                args.model_name,
                model,
                img0,
                img1,
                embt,
                imgt,
                bmv,
                fmv,
            )

            total_loss = loss_rec + loss_geo + loss_dis
            total_loss.backward()
            optimizer.step()

            loss_record = build_loss_record(loss_rec, loss_geo, loss_dis, total_loss)
            batch_size = int(imgt_pred.shape[0])

            for b in range(batch_size):
                psnr = calculate_psnr_torch(img_gt=imgt[b], img_pred=imgt_pred[b])
                evaluator.records.append({"psnr": float(psnr)})

            append_batch_loss_records(train_loss_records, loss_record, batch_size)

            global_step += 1

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


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default=None, type=str, help="Optional JSON-formatted YAML-compatible run config.")
    bootstrap_args, _remaining_argv = bootstrap_parser.parse_known_args(argv)

    config_path = None if bootstrap_args.config is None else Path(bootstrap_args.config)
    config_defaults = load_train_run_config(config_path)
    parser = build_train_arg_parser(config_defaults)
    return parser.parse_args(argv)


def build_train_arg_parser(config_defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train IFRNet variants on the VFI dataset.")
    parser.add_argument("--config", default=config_defaults.get("config"), type=str, help="Optional JSON-formatted YAML-compatible run config.")
    parser.add_argument("--mode", default=config_defaults.get("mode", "dry-run"), choices=["dry-run", "train"])
    parser.add_argument("--model-name", default=config_defaults.get("model_name", "IFRNet"), choices=["IFRNet", "IFRNet_Residual"])
    parser.add_argument(
        "--root-dir",
        default=config_defaults.get("root_dir", "./datasets/data"),
        help="Directory containing preprocessed CSV indexes.",
    )
    parser.add_argument(
        "--dataset-root-dir",
        default=config_defaults.get("dataset_root_dir"),
        type=str,
        help="Root directory containing frame and velocity assets.",
    )
    parser.add_argument(
        "--paths-config",
        default=config_defaults.get("paths_config"),
        type=str,
        help="Optional path to configs/paths/default.yaml.",
    )
    parser.add_argument(
        "--train-preset",
        default=config_defaults.get("train_preset", "train_vfx_0416"),
        choices=list_dataset_presets(),
    )
    parser.add_argument(
        "--test-preset",
        default=config_defaults.get("test_preset", "test_vfx_0416"),
        choices=list_dataset_presets(),
    )
    parser.add_argument("--epochs", default=config_defaults.get("epochs", 60), type=int, help="Total number of epochs to run.")
    parser.add_argument("--resume-path", default=config_defaults.get("resume_path"), type=str, help="Checkpoint to resume from.")
    parser.add_argument("--pretrained-checkpoint-path", default=config_defaults.get("pretrained_checkpoint_path"), type=str, help="Path to pretrained checkpoint.")
    parser.add_argument("--eval-interval", default=config_defaults.get("eval_interval", 1), type=int, help="Run validation every N epochs.")
    parser.add_argument("--lr-start", default=config_defaults.get("lr_start", 1e-4), type=float, help="Initial learning rate.")
    parser.add_argument("--lr-end", default=config_defaults.get("lr_end", 1e-5), type=float, help="Final learning rate after cosine decay.")
    parser.add_argument("--seed", default=config_defaults.get("seed", 1234), type=int, help="Random seed.")
    parser.add_argument("--batch-size", default=config_defaults.get("batch_size", 8), type=int, help="Training batch size.")
    parser.add_argument("--output-dir", default=config_defaults.get("output_dir"), type=str, help="Output directory for checkpoints and logs.")
    parser.add_argument("--only-fps", default=config_defaults.get("only_fps", 60), type=int, help="Use CSV entries for this FPS only.")
    parser.add_argument("--input-fps", default=config_defaults.get("input_fps", 30), type=int, help="Input frame rate for the dataset loader.")
    return parser


def load_train_run_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    return load_yaml_file(config_path=config_path)


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    model_config = get_model_config(args.model_name)
    args.resume_path = resolve_resume_path(args.resume_path, model_config["default_resume_path"])
    args.output_dir, args.output_dir_reason = resolve_output_dir(
        args.output_dir,
        args.resume_path,
        model_config["default_output_dir"],
    )

    if args.dataset_root_dir is None:
        paths_config_path = None if args.paths_config is None else Path(args.paths_config)
        args.dataset_root_dir = str(resolve_active_dataset_root(paths_config_path))

    return args


def load_training_state(
    args: argparse.Namespace,
    model: Any,
    optimizer: Any,
    device: Any,
    logger: logging.Logger,
) -> TrainingState:
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location=device)
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

    if args.pretrained_checkpoint_path is not None:
        pretrained_checkpoint_path = Path(args.pretrained_checkpoint_path)
        if not pretrained_checkpoint_path.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_checkpoint_path}")

        logger.info("Loading pretrained checkpoint from %s", pretrained_checkpoint_path)
        model.load_state_dict(torch.load(str(pretrained_checkpoint_path), map_location=device))
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


def log_run_summary(
    args: argparse.Namespace,
    train_dataset: Any,
    test_dataset: Any,
    training_state: TrainingState,
    device: Any,
    logger: logging.Logger,
) -> None:
    logger.info(
        "model=%s mode=%s device=%s output_dir=%s output_dir_reason=%s dataset_root_dir=%s train_samples=%s test_samples=%s start_epoch=%s epochs=%s batch_size=%s",
        args.model_name,
        training_state.mode,
        device,
        args.output_dir,
        args.output_dir_reason,
        args.dataset_root_dir,
        len(train_dataset),
        len(test_dataset),
        training_state.start_epoch,
        args.epochs,
        args.batch_size,
    )


def build_dry_run_summary(args: argparse.Namespace) -> dict[str, object]:
    return {
        "mode": args.mode,
        "model_name": args.model_name,
        "train_preset": args.train_preset,
        "test_preset": args.test_preset,
        "active_root_key": ACTIVE_DATASET_ROOT_KEY,
        "dataset_root_dir": args.dataset_root_dir,
        "csv_root_dir": str(Path(args.root_dir)),
        "output_dir": args.output_dir,
        "output_dir_reason": args.output_dir_reason,
        "resume_path": args.resume_path,
        "pretrained_checkpoint_path": args.pretrained_checkpoint_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_interval": args.eval_interval,
        "input_fps": args.input_fps,
        "only_fps": args.only_fps,
    }


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
    train_df = build_merged_dataframe(root_dir, checkpoints_dir, args.train_preset, args.only_fps, logger)
    test_df = build_merged_dataframe(root_dir, checkpoints_dir, args.test_preset, args.only_fps, logger)

    if "valid" in train_df.columns:
        logger.info("Valid Count %s in %s", train_df["valid"].value_counts().to_dict(), args.train_preset)
        train_df = train_df[train_df["valid"] == True]
    if "valid" in test_df.columns:
        logger.info("Valid Count %s in %s", test_df["valid"].value_counts().to_dict(), args.test_preset)
        test_df = test_df[test_df["valid"] == True]

    train_dataset = VFITrainDataset(train_df, args.dataset_root_dir, True, args.input_fps)
    test_dataset = VFITrainDataset(test_df, args.dataset_root_dir, False, args.input_fps)
    args.num_workers = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    args.iters_per_epoch = len(train_loader)
    model_class = get_model_class(args.model_name)
    model = model_class().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0)
    training_state = load_training_state(args, model, optimizer, device, logger)

    log_run_summary(args, train_dataset, test_dataset, training_state, device, logger)
    logger.info("run_log_dir=%s", run_dir)

    train(
        args,
        model,
        optimizer,
        train_loader,
        test_loader,
        device,
        logger,
        training_state,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_train_args(argv)
    args = prepare_args(args)

    if args.mode == "dry-run":
        print(json.dumps(build_dry_run_summary(args), indent=2))
        return

    run_training(args)


if __name__ == "__main__":
    main()
