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
from src.engine.evaluation import calculate_psnr
from src.utils.config import load_yaml_file

MODEL_NAMES: tuple[str, ...] = ("IFRNet", "IFRNet_Residual", "IFRNet_Residual_FlowApprox")


@dataclass(frozen=True)
class TrainingState:
    start_epoch: int
    global_step: int
    best_psnr: float
    mode: str


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
    img0: torch.Tensor,
    img1: torch.Tensor,
    embt: torch.Tensor,
    imgt: torch.Tensor,
    bmv: torch.Tensor,
    fmv: torch.Tensor,
) -> Any:
    import torch

    if model_name == "IFRNet":
        flow = torch.cat([bmv, fmv], dim=1).float()
        return model(img0, img1, embt, imgt, flow)

    return model(img0, img1, embt, imgt, init_flow0=bmv, init_flow1=fmv)


def build_loss_record(
    loss_rec: torch.Tensor,
    loss_geo: torch.Tensor,
    loss_dis: torch.Tensor,
    total_loss: torch.Tensor,
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


def append_batch_psnr_records(
    target_records: list[dict[str, float]],
    psnr_meter: AverageMeter,
    imgt_pred: torch.Tensor,
    imgt: torch.Tensor,
) -> None:
    batch_size = int(imgt_pred.shape[0])
    for batch_index in range(batch_size):
        # pred_np = imgt_pred[batch_index].detach().permute(1, 2, 0).cpu().numpy()
        # gt_np = imgt[batch_index].detach().permute(1, 2, 0).cpu().numpy()
        psnr_value = calculate_psnr(imgt[batch_index], imgt_pred[batch_index]).detach().cpu().item()
        psnr_meter.update(psnr_value, 1)
        target_records.append({"psnr": psnr_value})


def append_batch_metric_records(
    target_records: list[dict[str, object]],
    psnr_meter: AverageMeter,
    info: dict[str, Any],
    imgt_pred: Any,
    imgt: Any,
    loss_record: dict[str, float],
) -> None:
    batch_size = int(imgt_pred.shape[0])
    normalized_loss_record = {metric_name: float(metric_value) for metric_name, metric_value in loss_record.items()}

    for batch_index in range(batch_size):
        psnr_value = float(calculate_psnr(imgt[batch_index], imgt_pred[batch_index]).detach().cpu().item())
        psnr_meter.update(psnr_value, 1)
        target_records.append(
            {
                "record_name": info["record_name"][batch_index],
                "frame_range": info["frame_range"][batch_index],
                "psnr": psnr_value,
                **normalized_loss_record,
            }
        )


def build_record_name_summary(dataframe: Any) -> Any:
    summary_columns = ["psnr", "loss_rec", "loss_geo", "loss_dis", "loss_total"]
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


def evaluate(
    model_name: str,
    model: Any,
    loader: Any,
    device: Any,
) -> tuple[float, Any, Any]:
    import pandas as pd
    import torch
    from tqdm import tqdm

    model.eval()
    psnr_meter = AverageMeter()
    eval_records: list[dict[str, object]] = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for batch in pbar:
            img0, imgt, img1, bmv, fmv, embt, info = batch
            img0 = img0.to(device)
            img1 = img1.to(device)
            imgt = imgt.to(device)
            bmv = bmv.to(device)
            fmv = fmv.to(device)
            embt = embt.to(device)

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
            append_batch_metric_records(eval_records, psnr_meter, info, imgt_pred, imgt, loss_record)
            pbar.set_postfix({"eval_psnr": f"{psnr_meter.avg:.6f}"})

    eval_df = pd.DataFrame(eval_records)
    record_name_df = build_record_name_summary(eval_df)
    return psnr_meter.avg, eval_df, record_name_df


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
    import pandas as pd
    from tqdm import tqdm

    best_psnr = training_state.best_psnr
    global_step = training_state.global_step
    checkpoints_dir = Path(args.output_dir) / "checkpoints"

    for epoch in range(training_state.start_epoch, args.epochs):
        model.train()
        train_psnr_meter = AverageMeter()
        train_records: list[dict[str, object]] = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            img0, imgt, img1, bmv, fmv, embt, info = batch
            img0 = img0.to(device)
            img1 = img1.to(device)
            imgt = imgt.to(device)
            bmv = bmv.to(device)
            fmv = fmv.to(device)
            embt = embt.to(device)

            lr = get_lr(args, global_step)
            set_lr(optimizer, lr)
            optimizer.zero_grad()

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

            total_loss = loss_rec + loss_geo + loss_dis
            total_loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix({"train_loss": f"{float(total_loss.detach().cpu()):.6f}", "lr": f"{lr:.6f}"})
            if (epoch + 1) % args.eval_interval != 0:
                continue
            
            # During evaluation epochs, we also record training metrics and losses for analysis.
            loss_record = build_loss_record(loss_rec, loss_geo, loss_dis, total_loss)
            append_batch_metric_records(train_records, train_psnr_meter, info, imgt_pred, imgt, loss_record)

        if (epoch + 1) % args.eval_interval == 0:
            train_df = pd.DataFrame(train_records)
            train_record_name_df = build_record_name_summary(train_df)
            train_df.to_csv(checkpoints_dir / f"train_epoch_{epoch + 1}.csv", index=False)
            train_record_name_df.to_csv(checkpoints_dir / f"train_epoch_{epoch + 1}_record_name.csv", index=False)
            logger.info("Epoch %s train_psnr=%.6f", epoch + 1, train_psnr_meter.avg)

            test_psnr, test_df, test_record_name_df = evaluate(args.model_name, model, test_loader, device)
            test_df.to_csv(checkpoints_dir / f"test_epoch_{epoch + 1}.csv", index=False)
            test_record_name_df.to_csv(checkpoints_dir / f"test_epoch_{epoch + 1}_record_name.csv", index=False)
            logger.info("Epoch %s test_psnr=%.6f", epoch + 1, test_psnr)

            if test_psnr > best_psnr:
                best_psnr = test_psnr
                save_checkpoint(checkpoints_dir / "best.pth", model, optimizer, epoch, best_psnr)
                logger.info("New Best PSNR - Epoch %s test_psnr=%.6f", epoch + 1, test_psnr)
                            
        save_checkpoint(checkpoints_dir / f"epoch_{epoch + 1}.pth", model, optimizer, epoch, best_psnr)
        save_checkpoint(checkpoints_dir / "latest.pth", model, optimizer, epoch, best_psnr)


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
    return parser


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default=None, type=str, help="Optional JSON-formatted YAML-compatible run config.")
    bootstrap_args, _remaining_argv = bootstrap_parser.parse_known_args(argv)

    config_path = None if bootstrap_args.config is None else Path(bootstrap_args.config)
    config_defaults = load_train_run_config(config_path)
    parser = build_train_arg_parser(config_defaults)
    return parser.parse_args(argv)


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    return args


def load_training_state(
    args: argparse.Namespace,
    model: Any,
    optimizer: Any,
    device: torch.device,
    logger: logging.Logger,
) -> TrainingState:
    import torch

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

    pretrained_path = args.pretrained_checkpoint_path
    if pretrained_path is not None:
        pretrained_path = Path(pretrained_path)
        logger.info("Loading pretrained checkpoint from %s", pretrained_path)
        model.load_state_dict(torch.load(str(pretrained_path), map_location=device))
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
        "model=%s mode=%s device=%s output_dir=%s dataset_root_dir=%s train_samples=%s test_samples=%s start_epoch=%s epochs=%s batch_size=%s",
        args.model_name,
        training_state.mode,
        device,
        args.output_dir,
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
        "resume_path": args.resume_path,
        "pretrained_checkpoint_path": args.pretrained_checkpoint_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_interval": args.eval_interval,
        "input_fps": args.input_fps,
        "only_fps": args.only_fps,
    }


def run_training(args: argparse.Namespace) -> None:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from src.data.dataset_loader import VFITrainDataset

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

    # temporary code for quick testing
    # train_df = train_df[:100]
    # test_df = test_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)  # shuffle test_df for more representative quick tests
    # test_df = test_df[:10]

    train_dataset = VFITrainDataset(train_df, args.dataset_root_dir, True, args.input_fps)
    test_dataset = VFITrainDataset(test_df, args.dataset_root_dir, False, args.input_fps)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    args.iters_per_epoch = len(train_loader)
    model_class = resolve_model_class(args.model_name)
    model_init_args = dict(getattr(args, "model_init_args", {}))
    model = model_class(**model_init_args).to(device)
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
