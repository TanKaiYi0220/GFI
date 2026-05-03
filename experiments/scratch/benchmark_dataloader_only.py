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
from src.data.dataset_loader import NPZVFITrainDataset
from src.data.dataset_loader import VFITrainDataset
from src.data.dataset_loader import apply_vfi_train_augment
from src.data.dataset_loader import build_distance_indexing
from src.data.dataset_loader import build_vfi_training_tensors
from src.data.dataset_loader import load_npz_arrays


RAW_TIMING_FIELDS: tuple[str, ...] = (
    "row_lookup_seconds",
    "image_load_seconds",
    "flow_load_seconds",
    "augment_seconds",
    "tensor_seconds",
    "sample_total_seconds",
)

NPZ_TIMING_FIELDS: tuple[str, ...] = (
    "row_lookup_seconds",
    "npz_load_seconds",
    "augment_seconds",
    "tensor_seconds",
    "sample_total_seconds",
)


def parse_bool_argument(value: str) -> bool:
    normalized_value = value.strip().lower()
    if normalized_value in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean-like value, got: {value}")


def build_benchmark_arg_parser(config_defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = build_train_arg_parser(config_defaults)
    parser.description = "Benchmark dataloader-only throughput and compare raw files against .npz caches."
    parser.add_argument("--data-source", default="raw", choices=["raw", "npz", "both"])
    parser.add_argument("--npz-manifest", default=None, type=str, help="Manifest CSV written by cache_vfi_samples_npz.py.")
    parser.add_argument("--npz-cache-dir", default=str(PROJECT_ROOT / "outputs" / "npz_cache"), type=str)
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


def build_npz_manifest_path(args: argparse.Namespace) -> Path:
    if args.npz_manifest is not None:
        return Path(args.npz_manifest)

    manifest_path = Path(args.npz_cache_dir) / args.train_preset / f"{args.train_preset}_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"NPZ manifest not found: {manifest_path}")

    return manifest_path


class TimedRawVFITrainDataset(VFITrainDataset):
    timing_fields: tuple[str, ...] = RAW_TIMING_FIELDS

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
            img0, imgt, img1, bmv, fmv = apply_vfi_train_augment(img0, imgt, img1, bmv, fmv)
        augment_end = time.perf_counter()

        tensor_start = time.perf_counter()
        img0_tensor, imgt_tensor, img1_tensor, bmv_tensor, fmv_tensor, embt_tensor = build_vfi_training_tensors(
            img0,
            imgt,
            img1,
            bmv,
            fmv,
        )
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


class TimedNPZVFITrainDataset(NPZVFITrainDataset):
    timing_fields: tuple[str, ...] = NPZ_TIMING_FIELDS

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        sample_start = time.perf_counter()

        row_lookup_start = time.perf_counter()
        row = self.manifest.iloc[index]
        cache_path = Path(str(row["cache_path"]))
        info = {
            "frame_range": str(row["frame_range"]),
            "valid": bool(row["valid"]),
            "distance_indexing": [
                float(row["distance_index_mean"]),
                float(row["distance_index_median"]),
            ],
        }
        row_lookup_end = time.perf_counter()

        npz_load_start = time.perf_counter()
        arrays = load_npz_arrays(cache_path)
        img0 = arrays["img0"]
        imgt = arrays["imgt"]
        img1 = arrays["img1"]
        bmv = arrays["bmv"]
        fmv = arrays["fmv"]
        npz_load_end = time.perf_counter()

        augment_start = time.perf_counter()
        if self.augment:
            img0, imgt, img1, bmv, fmv = apply_vfi_train_augment(img0, imgt, img1, bmv, fmv)
        augment_end = time.perf_counter()

        tensor_start = time.perf_counter()
        img0_tensor, imgt_tensor, img1_tensor, bmv_tensor, fmv_tensor, embt_tensor = build_vfi_training_tensors(
            img0,
            imgt,
            img1,
            bmv,
            fmv,
        )
        tensor_end = time.perf_counter()

        sample_end = time.perf_counter()
        info["timing"] = {
            "row_lookup_seconds": row_lookup_end - row_lookup_start,
            "npz_load_seconds": npz_load_end - npz_load_start,
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
    timing_fields: tuple[str, ...],
) -> dict[str, float | int]:
    batch_record: dict[str, float | int] = {
        "batch_index": batch_index,
        "batch_size": batch_size,
        "batch_wait_seconds": batch_wait_seconds,
    }
    for timing_field in timing_fields:
        batch_record[f"{timing_field}_mean"] = float(timing_info[timing_field].float().mean().item())

    return batch_record


def build_summary(
    records: list[dict[str, float | int]],
    warmup_batches: int,
    timing_fields: tuple[str, ...],
    data_source: str,
) -> dict[str, float | int | str]:
    if len(records) == 0:
        raise RuntimeError("No dataloader benchmark records were collected.")

    summary_df = pd.DataFrame(records)
    filtered_df = summary_df[summary_df["batch_index"] > warmup_batches]
    benchmark_df = filtered_df if len(filtered_df) > 0 else summary_df

    average_batch_size = float(benchmark_df["batch_size"].mean())
    average_batch_wait = float(benchmark_df["batch_wait_seconds"].mean())
    end_to_end_samples_per_second = average_batch_size / average_batch_wait if average_batch_wait > 0.0 else 0.0

    summary: dict[str, float | int | str] = {
        "data_source": data_source,
        "batch_wait_seconds": average_batch_wait,
        "end_to_end_samples_per_second": end_to_end_samples_per_second,
        "average_batch_size": average_batch_size,
        "measured_batch_count": int(len(benchmark_df)),
    }
    for timing_field in timing_fields:
        summary[f"{timing_field}_mean"] = float(benchmark_df[f"{timing_field}_mean"].mean())

    return summary


def build_comparison_summary(raw_summary: dict[str, Any], npz_summary: dict[str, Any]) -> dict[str, Any]:
    raw_wait = float(raw_summary["batch_wait_seconds"])
    npz_wait = float(npz_summary["batch_wait_seconds"])
    raw_throughput = float(raw_summary["end_to_end_samples_per_second"])
    npz_throughput = float(npz_summary["end_to_end_samples_per_second"])

    return {
        "raw_data_source": raw_summary["data_source"],
        "npz_data_source": npz_summary["data_source"],
        "batch_wait_speedup": (raw_wait / npz_wait) if npz_wait > 0.0 else float("inf"),
        "throughput_speedup": (npz_throughput / raw_throughput) if raw_throughput > 0.0 else float("inf"),
        "raw_batch_wait_seconds": raw_wait,
        "npz_batch_wait_seconds": npz_wait,
        "raw_end_to_end_samples_per_second": raw_throughput,
        "npz_end_to_end_samples_per_second": npz_throughput,
    }


def print_benchmark_header(args: argparse.Namespace, data_source: str, dataset_len: int) -> None:
    header = {
        "data_source": data_source,
        "dataset_len": dataset_len,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "num_batches": args.num_batches,
        "warmup_batches": args.warmup_batches,
    }
    if data_source == "npz":
        header["npz_manifest"] = str(build_npz_manifest_path(args))

    print(json.dumps(header, indent=2))


def run_one_benchmark(
    args: argparse.Namespace,
    data_source: str,
) -> dict[str, Any]:
    loader_kwargs = build_dataloader_kwargs(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    if data_source == "raw":
        root_dir = Path(args.root_dir)
        checkpoints_dir = PROJECT_ROOT / "outputs" / "dataloader_benchmark"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        train_df = build_merged_dataframe(root_dir, checkpoints_dir, args.train_preset, args.only_fps, logger=_SilentLogger())
        if "valid" in train_df.columns:
            train_df = train_df[train_df["valid"] == True]
        dataset = TimedRawVFITrainDataset(train_df, args.dataset_root_dir, True, args.input_fps)
    else:
        manifest_path = build_npz_manifest_path(args)
        dataset = TimedNPZVFITrainDataset(str(manifest_path), True)

    print_benchmark_header(args, data_source, len(dataset))

    loader = DataLoader(dataset, **loader_kwargs)
    iterator = iter(loader)
    batch_records: list[dict[str, float | int]] = []
    timing_fields = dataset.timing_fields

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
            timing_fields=timing_fields,
        )
        batch_records.append(batch_record)

        if batch_index == 1 or batch_index == args.warmup_batches or batch_index == args.num_batches:
            print(json.dumps(batch_record, indent=2))

    summary = build_summary(batch_records, args.warmup_batches, timing_fields, data_source)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_benchmark_args(argv)
    args = prepare_args(args)

    if args.data_source == "raw":
        run_one_benchmark(args, "raw")
        return

    if args.data_source == "npz":
        run_one_benchmark(args, "npz")
        return

    raw_summary = run_one_benchmark(args, "raw")
    npz_summary = run_one_benchmark(args, "npz")
    comparison_summary = build_comparison_summary(raw_summary, npz_summary)
    print(json.dumps(comparison_summary, indent=2))


class _SilentLogger:
    def info(self, _message: str, *_args: Any) -> None:
        return

    def warning(self, _message: str, *_args: Any) -> None:
        return


if __name__ == "__main__":
    main()
