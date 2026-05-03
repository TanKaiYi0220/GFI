from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_merged_dataframe
from scripts.train import build_train_arg_parser
from scripts.train import load_train_run_config
from scripts.train import prepare_args
from src.data.augment import random_crop
from src.data.augment import random_horizontal_flip
from src.data.augment import random_reverse_channel
from src.data.augment import random_rotate
from src.data.augment import random_vertical_flip
from src.data.dataset_loader import VFITrainDataset
from src.data.dataset_loader import build_distance_indexing
from src.data.dataset_loader import build_embedding_tensor
from src.data.dataset_loader import flow_to_tensor
from src.data.dataset_loader import image_to_tensor


def parse_bool_argument(value: str) -> bool:
    normalized_value = value.strip().lower()
    if normalized_value in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean-like value, got: {value}")


def build_benchmark_arg_parser(config_defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = build_train_arg_parser(config_defaults)
    parser.description = "Benchmark dataloader-only throughput and per-sample timing breakdown."
    parser.add_argument("--num-workers", default=config_defaults.get("num_workers", 0), type=int)
    parser.add_argument("--pin-memory", default=config_defaults.get("pin_memory", True), type=parse_bool_argument)
    parser.add_argument("--persistent-workers", default=config_defaults.get("persistent_workers", False), type=parse_bool_argument)
    parser.add_argument("--prefetch-factor", default=config_defaults.get("prefetch_factor", 2), type=int)
    parser.add_argument("--num-batches", default=50, type=int, help="Number of dataloader batches to benchmark.")
    parser.add_argument("--warmup-batches", default=5, type=int, help="Ignore the first N batches in summary statistics.")
    return parser


def parse_benchmark_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default=None, type=str)
    bootstrap_args, _remaining_argv = bootstrap_parser.parse_known_args(argv)

    config_path = None if bootstrap_args.config is None else Path(bootstrap_args.config)
    config_defaults = load_train_run_config(config_path)
    parser = build_benchmark_arg_parser(config_defaults)
    return parser.parse_args(argv)


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


class TimedVFITrainDataset(VFITrainDataset):
    def __getitem__(self, index: int) -> tuple[Any, ...]:
        sample_start = time.perf_counter()

        row_lookup_start = time.perf_counter()
        row = self.dataframe.iloc[index]
        frame_0_idx = int(row["img0"])
        frame_1_idx = int(row["img1"])
        frame_2_idx = int(row["img2"])
        record = str(row["record"])
        mode = str(row["mode"])

        info = {
            "frame_range": f"frame_{frame_0_idx:04d}_{frame_2_idx:04d}",
            "valid": bool(row["valid"]) if "valid" in row.index else True,
            "distance_indexing": build_distance_indexing(row),
        }

        img_0_path = self._build_modality_path(record, mode, frame_0_idx, "colorNoScreenUI")
        img_1_path = self._build_modality_path(record, mode, frame_1_idx, "colorNoScreenUI")
        img_2_path = self._build_modality_path(record, mode, frame_2_idx, "colorNoScreenUI")
        backward_path = self._build_modality_path(record, mode, frame_1_idx, "backwardVel_Depth")
        forward_path = self._build_modality_path(record, mode, frame_1_idx, "forwardVel_Depth")
        row_lookup_end = time.perf_counter()

        image_load_start = time.perf_counter()
        img0 = self._load_image(img_0_path)
        imgt = self._load_image(img_1_path)
        img1 = self._load_image(img_2_path)
        image_load_end = time.perf_counter()

        flow_load_start = time.perf_counter()
        bmv = self._load_game_motion(backward_path)
        fmv = self._load_game_motion(forward_path)
        flow_load_end = time.perf_counter()

        augment_start = time.perf_counter()
        if self.augment:
            img0, imgt, img1, bmv, fmv = random_crop(img0, imgt, img1, bmv, fmv, (224, 224))
            img0, imgt, img1, bmv, fmv = random_reverse_channel(img0, imgt, img1, bmv, fmv, 0.5)
            img0, imgt, img1, bmv, fmv = random_vertical_flip(img0, imgt, img1, bmv, fmv, 0.3)
            img0, imgt, img1, bmv, fmv = random_horizontal_flip(img0, imgt, img1, bmv, fmv, 0.5)
            img0, imgt, img1, bmv, fmv = random_rotate(img0, imgt, img1, bmv, fmv, 0.05)
        augment_end = time.perf_counter()

        tensor_start = time.perf_counter()
        img0_tensor = image_to_tensor(img0)
        imgt_tensor = image_to_tensor(imgt)
        img1_tensor = image_to_tensor(img1)
        bmv_tensor = flow_to_tensor(bmv)
        fmv_tensor = flow_to_tensor(fmv)
        embt_tensor = build_embedding_tensor()
        tensor_end = time.perf_counter()

        sample_end = time.perf_counter()
        info["timing"] = {
            "row_lookup_seconds": row_lookup_end - row_lookup_start,
            "image_load_seconds": image_load_end - image_load_start,
            "flow_load_seconds": flow_load_end - flow_load_start,
            "augment_seconds": augment_end - augment_start,
            "tensor_seconds": tensor_end - tensor_start,
            "sample_total_seconds": sample_end - sample_start,
        }

        return img0_tensor, imgt_tensor, img1_tensor, bmv_tensor, fmv_tensor, embt_tensor, info


def build_batch_record(
    batch_index: int,
    batch_size: int,
    batch_wait_seconds: float,
    timing_info: dict[str, Any],
) -> dict[str, float | int]:
    return {
        "batch_index": batch_index,
        "batch_size": batch_size,
        "batch_wait_seconds": batch_wait_seconds,
        "row_lookup_seconds_mean": float(timing_info["row_lookup_seconds"].float().mean().item()),
        "image_load_seconds_mean": float(timing_info["image_load_seconds"].float().mean().item()),
        "flow_load_seconds_mean": float(timing_info["flow_load_seconds"].float().mean().item()),
        "augment_seconds_mean": float(timing_info["augment_seconds"].float().mean().item()),
        "tensor_seconds_mean": float(timing_info["tensor_seconds"].float().mean().item()),
        "sample_total_seconds_mean": float(timing_info["sample_total_seconds"].float().mean().item()),
    }


def build_summary(records: list[dict[str, float | int]], warmup_batches: int) -> dict[str, float]:
    if len(records) == 0:
        raise RuntimeError("No dataloader benchmark records were collected.")

    summary_df = pd.DataFrame(records)
    filtered_df = summary_df[summary_df["batch_index"] > warmup_batches]
    benchmark_df = filtered_df if len(filtered_df) > 0 else summary_df

    average_batch_size = float(benchmark_df["batch_size"].mean())
    average_batch_wait = float(benchmark_df["batch_wait_seconds"].mean())
    end_to_end_samples_per_second = average_batch_size / average_batch_wait if average_batch_wait > 0.0 else 0.0

    return {
        "batch_wait_seconds": average_batch_wait,
        "row_lookup_seconds_mean": float(benchmark_df["row_lookup_seconds_mean"].mean()),
        "image_load_seconds_mean": float(benchmark_df["image_load_seconds_mean"].mean()),
        "flow_load_seconds_mean": float(benchmark_df["flow_load_seconds_mean"].mean()),
        "augment_seconds_mean": float(benchmark_df["augment_seconds_mean"].mean()),
        "tensor_seconds_mean": float(benchmark_df["tensor_seconds_mean"].mean()),
        "sample_total_seconds_mean": float(benchmark_df["sample_total_seconds_mean"].mean()),
        "end_to_end_samples_per_second": end_to_end_samples_per_second,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_benchmark_args(argv)
    args = prepare_args(args)

    root_dir = Path(args.root_dir)
    checkpoints_dir = PROJECT_ROOT / "outputs" / "dataloader_benchmark"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_df = build_merged_dataframe(root_dir, checkpoints_dir, args.train_preset, args.only_fps, logger=_SilentLogger())
    if "valid" in train_df.columns:
        train_df = train_df[train_df["valid"] == True]

    dataset = TimedVFITrainDataset(train_df, args.dataset_root_dir, True, args.input_fps)
    loader_kwargs = build_dataloader_kwargs(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    loader = DataLoader(dataset, **loader_kwargs)

    print(json.dumps(
        {
            "dataset_len": len(dataset),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "persistent_workers": args.persistent_workers,
            "prefetch_factor": args.prefetch_factor,
            "num_batches": args.num_batches,
            "warmup_batches": args.warmup_batches,
        },
        indent=2,
    ))

    batch_records: list[dict[str, float | int]] = []
    iterator = iter(loader)

    for batch_index in range(1, args.num_batches + 1):
        batch_start = time.perf_counter()
        try:
            img0, imgt, img1, bmv, fmv, embt, info = next(iterator)
        except StopIteration:
            break
        batch_end = time.perf_counter()

        batch_record = build_batch_record(
            batch_index=batch_index,
            batch_size=int(img0.shape[0]),
            batch_wait_seconds=batch_end - batch_start,
            timing_info=info["timing"],
        )
        batch_records.append(batch_record)

        if batch_index == 1 or batch_index == args.warmup_batches or batch_index == args.num_batches:
            print(json.dumps(batch_record, indent=2))

    summary = build_summary(batch_records, args.warmup_batches)
    print(json.dumps(summary, indent=2))


class _SilentLogger:
    def info(self, _message: str, *_args: Any) -> None:
        return

    def warning(self, _message: str, *_args: Any) -> None:
        return


if __name__ == "__main__":
    main()
