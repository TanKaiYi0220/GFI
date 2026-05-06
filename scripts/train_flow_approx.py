from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import TrainingState
from scripts.train import build_logger
from scripts.train import build_merged_dataframe
from scripts.train import build_train_arg_parser
from scripts.train import load_train_run_config
from scripts.train import load_training_state
from scripts.train import log_run_summary
from scripts.train import resolve_model_class
from scripts.train import save_checkpoint
from scripts.train import set_lr
from scripts.train import set_seed
from scripts.train import get_lr
from src.data.dataset_config import ACTIVE_DATASET_ROOT_KEY
from src.engine.evaluation import AverageMeter
from src.engine.evaluation import calculate_psnr

FLOW_APPROX_METHODS: tuple[str, ...] = ("single", "combination")


def flow_approx(flow: Any, time: Any, forward: bool) -> Any:
    return time * flow if forward else (1 - time) * flow


def flow_approx_combination(fmv: Any, bmv: Any, time: Any, forward: bool) -> Any:
    if forward:
        return (1 - time) * (1 - time) * fmv - time * (1 - time) * bmv

    return -(1 - time) * time * fmv + time * time * bmv


def build_flow_init(
    fmv_30: Any,
    bmv_30: Any,
    embt: Any,
    flow_approx_method: str,
) -> tuple[Any, Any]:
    time = embt.reshape(embt.shape[0], 1, 1, 1)

    if flow_approx_method == "single":
        approx_fmv = flow_approx(fmv_30, time, True)
        approx_bmv = flow_approx(bmv_30, time, False)
        return approx_bmv, approx_fmv

    approx_fmv = flow_approx_combination(fmv_30, bmv_30, time, True)
    approx_bmv = flow_approx_combination(fmv_30, bmv_30, time, False)
    return approx_bmv, approx_fmv


def forward_model_with_approx(
    model_name: str,
    model: Any,
    img0: Any,
    img1: Any,
    embt: Any,
    imgt: Any,
    bmv_30: Any,
    fmv_30: Any,
    flow_approx_method: str,
) -> tuple[Any, Any, Any]:
    import torch

    approx_bmv, approx_fmv = build_flow_init(fmv_30, bmv_30, embt, flow_approx_method)

    if model_name == "IFRNet":
        flow = torch.cat([approx_bmv, approx_fmv], dim=1).float()
        model_output = model(img0, img1, embt, imgt, flow)
        return model_output, approx_bmv, approx_fmv

    model_output = model(
        img0,
        img1,
        embt,
        imgt,
        init_flow0=approx_bmv,
        init_flow1=approx_fmv,
    )
    return model_output, approx_bmv, approx_fmv


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


def evaluate(
    args: argparse.Namespace,
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
            img0, imgt, img1, _bmv_60, _fmv_60, bmv_30, fmv_30, embt, info = batch
            img0 = img0.to(device)
            img1 = img1.to(device)
            imgt = imgt.to(device)
            bmv_30 = bmv_30.to(device)
            fmv_30 = fmv_30.to(device)
            embt = embt.to(device)

            model_output, _approx_bmv, _approx_fmv = forward_model_with_approx(
                args.model_name,
                model,
                img0,
                img1,
                embt,
                imgt,
                bmv_30,
                fmv_30,
                args.flow_approx_method,
            )
            imgt_pred, loss_rec, loss_geo, loss_dis, _up_flow0_1, _up_flow1_1, _up_mask_1 = model_output

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
    logger: Any,
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
            img0, imgt, img1, _bmv_60, _fmv_60, bmv_30, fmv_30, embt, info = batch
            img0 = img0.to(device)
            img1 = img1.to(device)
            imgt = imgt.to(device)
            bmv_30 = bmv_30.to(device)
            fmv_30 = fmv_30.to(device)
            embt = embt.to(device)

            lr = get_lr(args, global_step)
            set_lr(optimizer, lr)
            optimizer.zero_grad()

            model_output, _approx_bmv, _approx_fmv = forward_model_with_approx(
                args.model_name,
                model,
                img0,
                img1,
                embt,
                imgt,
                bmv_30,
                fmv_30,
                args.flow_approx_method,
            )
            imgt_pred, loss_rec, loss_geo, loss_dis, _up_flow0_1, _up_flow1_1, _up_mask_1 = model_output

            total_loss = loss_rec + loss_geo + loss_dis
            total_loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix({"train_loss": f"{float(total_loss.detach().cpu()):.6f}", "lr": f"{lr:.6f}"})

            if (epoch + 1) % args.eval_interval != 0:
                continue

            loss_record = build_loss_record(loss_rec, loss_geo, loss_dis, total_loss)
            append_batch_metric_records(train_records, train_psnr_meter, info, imgt_pred, imgt, loss_record)

        if (epoch + 1) % args.eval_interval == 0:
            train_df = pd.DataFrame(train_records)
            train_record_name_df = build_record_name_summary(train_df)
            train_df.to_csv(checkpoints_dir / f"train_epoch_{epoch + 1}.csv", index=False)
            train_record_name_df.to_csv(checkpoints_dir / f"train_epoch_{epoch + 1}_record_name.csv", index=False)
            logger.info("Epoch %s train_psnr=%.6f", epoch + 1, train_psnr_meter.avg)

            test_psnr, test_df, test_record_name_df = evaluate(args, model, test_loader, device)
            test_df.to_csv(checkpoints_dir / f"test_epoch_{epoch + 1}.csv", index=False)
            test_record_name_df.to_csv(checkpoints_dir / f"test_epoch_{epoch + 1}_record_name.csv", index=False)
            logger.info("Epoch %s test_psnr=%.6f", epoch + 1, test_psnr)

            if test_psnr > best_psnr:
                best_psnr = test_psnr
                save_checkpoint(checkpoints_dir / "best.pth", model, optimizer, epoch, best_psnr)
                logger.info("New Best PSNR - Epoch %s test_psnr=%.6f", epoch + 1, test_psnr)

        save_checkpoint(checkpoints_dir / f"epoch_{epoch + 1}.pth", model, optimizer, epoch, best_psnr)
        save_checkpoint(checkpoints_dir / "latest.pth", model, optimizer, epoch, best_psnr)


def build_flow_approx_arg_parser(config_defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = build_train_arg_parser(config_defaults)
    parser.description = "Train IFRNet variants with 30fps flow approximation initializations."
    parser.add_argument(
        "--flow-approx-method",
        default=config_defaults.get("flow_approx_method", "combination"),
        choices=FLOW_APPROX_METHODS,
        help="How to approximate middle-frame flows from 30fps motion vectors.",
    )
    return parser


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default=None, type=str, help="Optional JSON-formatted YAML-compatible run config.")
    bootstrap_args, _remaining_argv = bootstrap_parser.parse_known_args(argv)

    config_path = None if bootstrap_args.config is None else Path(bootstrap_args.config)
    config_defaults = load_train_run_config(config_path)
    parser = build_flow_approx_arg_parser(config_defaults)
    return parser.parse_args(argv)


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
        "flow_approx_method": args.flow_approx_method,
        "dataset_class": "FlowEstimationTrainDataset",
    }


def run_training(args: argparse.Namespace) -> None:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from src.data.dataset_loader import FlowEstimationTrainDataset

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger, run_dir = build_logger(output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)
    logger.info("flow_approx_method=%s", args.flow_approx_method)

    root_dir = Path(args.root_dir)
    train_df = build_merged_dataframe(root_dir, checkpoints_dir, args.train_preset, args.only_fps, logger)
    test_df = build_merged_dataframe(root_dir, checkpoints_dir, args.test_preset, args.only_fps, logger)

    if "valid" in train_df.columns:
        logger.info("Valid Count %s in %s", train_df["valid"].value_counts().to_dict(), args.train_preset)
        train_df = train_df[train_df["valid"] == True]
    if "valid" in test_df.columns:
        logger.info("Valid Count %s in %s", test_df["valid"].value_counts().to_dict(), args.test_preset)
        test_df = test_df[test_df["valid"] == True]

    train_dataset = FlowEstimationTrainDataset(train_df.reset_index(drop=True), args.dataset_root_dir, args.input_fps, True)
    test_dataset = FlowEstimationTrainDataset(test_df.reset_index(drop=True), args.dataset_root_dir, args.input_fps, False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    args.iters_per_epoch = len(train_loader)
    model_class = resolve_model_class(args.model_name)
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

    if args.mode == "dry-run":
        print(json.dumps(build_dry_run_summary(args), indent=2))
        return

    run_training(args)


if __name__ == "__main__":
    main()
