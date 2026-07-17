from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from src.engine.checkpoints import load_inference_checkpoint
from src.engine.model_registry import resolve_model_class
from src.engine.model_registry import set_model_convex_upsampling
from src.engine.run_config import InferenceRunConfig
from src.engine.run_config import build_inference_run_config


@dataclass(frozen=True)
class BenchmarkSample:
    name: str
    img0_path: Path
    img2_path: Path
    timestep: float


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    median_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    fps: float


@dataclass(frozen=True)
class BenchmarkRunArgs:
    config_path: Path
    input_dir: Path
    output_dir: Path
    warmup: int
    repeat: int
    batch_size: int
    device: str


@dataclass(frozen=True)
class BenchmarkBatch:
    sample_names: list[str]
    img0: torch.Tensor
    img1: torch.Tensor
    embt: torch.Tensor
    image_shape: tuple[int, int]


def parse_benchmark_args(default_config: str, model_label: str, argv: list[str] | None) -> BenchmarkRunArgs:
    parser = argparse.ArgumentParser(description=f"Benchmark {model_label} inference time.")
    parser.add_argument("--config", default=default_config, type=str, help="Path to an existing inference YAML.")
    parser.add_argument("--input-dir", default="benchmarks/inputs", type=str, help="Folder containing benchmark samples.")
    parser.add_argument("--output-dir", default="benchmarks/outputs", type=str, help="Folder for CSV and JSON reports.")
    parser.add_argument("--warmup", default=5, type=int, help="Untimed warmup iterations.")
    parser.add_argument("--repeat", default=30, type=int, help="Timed iterations per batch.")
    parser.add_argument("--batch-size", default=1, type=int, help="Samples per timed forward pass.")
    parser.add_argument("--device", default="cuda", type=str, help="Torch device, usually cuda.")
    args = parser.parse_args(argv)
    if int(args.warmup) < 0:
        raise ValueError(f"Benchmark warmup must be non-negative, got {args.warmup}")
    if int(args.repeat) <= 0:
        raise ValueError(f"Benchmark repeat must be positive, got {args.repeat}")
    if int(args.batch_size) <= 0:
        raise ValueError(f"Benchmark batch_size must be positive, got {args.batch_size}")
    return BenchmarkRunArgs(
        config_path=Path(args.config),
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        warmup=int(args.warmup),
        repeat=int(args.repeat),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )


def load_rgb_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    image_tensor = image_tensor.reshape(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0
    return image_tensor


def read_sample_timestep(sample_dir: Path) -> float:
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        return 0.5
    raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(raw_meta, dict):
        raise TypeError(f"Benchmark sample meta.json must contain a mapping: path={meta_path}")
    timestep = float(raw_meta.get("timestep", 0.5))
    if not math.isfinite(timestep):
        raise ValueError(f"Benchmark timestep must be finite, got {timestep}: path={meta_path}")
    if timestep < 0.0 or timestep > 1.0:
        raise ValueError(f"Benchmark timestep must be in [0, 1], got {timestep}: path={meta_path}")
    return timestep


def discover_benchmark_samples(input_dir: Path) -> list[BenchmarkSample]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Benchmark input directory is missing: path={input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Benchmark input path must be a directory: path={input_dir}")

    sample_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if len(sample_dirs) == 0:
        raise ValueError(f"Benchmark input directory has no sample folders: path={input_dir}")

    samples: list[BenchmarkSample] = []
    for sample_dir in sample_dirs:
        img0_path = sample_dir / "img0.png"
        img2_path = sample_dir / "img2.png"
        if not img0_path.is_file():
            raise FileNotFoundError(f"Benchmark sample is missing img0.png: path={img0_path}")
        if not img2_path.is_file():
            raise FileNotFoundError(f"Benchmark sample is missing img2.png: path={img2_path}")
        samples.append(
            BenchmarkSample(
                name=sample_dir.name,
                img0_path=img0_path,
                img2_path=img2_path,
                timestep=read_sample_timestep(sample_dir=sample_dir),
            )
        )
    return samples


def build_benchmark_batches(samples: list[BenchmarkSample], batch_size: int) -> list[BenchmarkBatch]:
    if batch_size <= 0:
        raise ValueError(f"Benchmark batch_size must be positive, got {batch_size}")

    batches: list[BenchmarkBatch] = []
    expected_image_shape: tuple[int, int] | None = None
    for start_index in range(0, len(samples), batch_size):
        batch_samples = samples[start_index : start_index + batch_size]
        img0_tensors = [load_rgb_tensor(image_path=sample.img0_path) for sample in batch_samples]
        img1_tensors = [load_rgb_tensor(image_path=sample.img2_path) for sample in batch_samples]
        shapes = {(int(tensor.shape[-2]), int(tensor.shape[-1])) for tensor in img0_tensors + img1_tensors}
        if len(shapes) != 1:
            raise ValueError(
                "All benchmark images inside one batch must share shape, "
                f"sample_names={[sample.name for sample in batch_samples]}, shapes={sorted(shapes)}"
            )
        image_shape = next(iter(shapes))
        if expected_image_shape is None:
            expected_image_shape = image_shape
        elif image_shape != expected_image_shape:
            raise ValueError(
                "All benchmark images in one entire run must share shape, "
                f"expected_shape={expected_image_shape}, sample_names={[sample.name for sample in batch_samples]}, "
                f"actual_shape={image_shape}"
            )
        embt = torch.tensor([sample.timestep for sample in batch_samples], dtype=torch.float32).view(-1, 1, 1, 1)
        batches.append(
            BenchmarkBatch(
                sample_names=[sample.name for sample in batch_samples],
                img0=torch.stack(img0_tensors, dim=0),
                img1=torch.stack(img1_tensors, dim=0),
                embt=embt,
                image_shape=image_shape,
            )
        )
    return batches


def require_available_device(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Benchmark requested device=cuda, but CUDA is not available.")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def inference_context() -> Any:
    inference_mode = getattr(torch, "inference_mode", None)
    if callable(inference_mode):
        return inference_mode()
    no_grad = getattr(torch, "no_grad", None)
    if callable(no_grad):
        return no_grad()
    return nullcontext()


def load_benchmark_model(config: InferenceRunConfig, device: torch.device) -> torch.nn.Module:
    model_class = resolve_model_class(config.model.model_name)
    model = model_class(**config.model.model_init_args).to(device)
    load_inference_checkpoint(model=model, checkpoint_path=config.checkpoint_path, device=device)
    if config.model.eval_convex_upsampling is not None:
        set_model_convex_upsampling(
            model=model,
            enabled=config.model.eval_convex_upsampling,
            context="benchmark",
        )
    model.eval()
    return model


def build_inference_batch(batch: BenchmarkBatch) -> tuple[torch.Tensor, ...]:
    batch_size = int(batch.img0.shape[0])
    height = int(batch.img0.shape[-2])
    width = int(batch.img0.shape[-1])
    imgt = torch.zeros_like(batch.img0)
    flow = torch.zeros((batch_size, 2, height, width), dtype=batch.img0.dtype)
    return batch.img0, imgt, batch.img1, flow, flow.clone(), batch.embt, {}


def run_inference_batch_wrapper(
    config: InferenceRunConfig,
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> Any:
    from src.engine.interpolation_batch import run_inference_batch

    return run_inference_batch(config=config, model=model, batch=batch, device=device)


def run_single_inference(
    model: torch.nn.Module,
    config: InferenceRunConfig,
    batch: BenchmarkBatch,
    device: torch.device,
) -> Any:
    inference_batch = build_inference_batch(batch=batch)
    return run_inference_batch_wrapper(config=config, model=model, batch=inference_batch, device=device)


def measure_batch_ms(
    model: torch.nn.Module,
    config: InferenceRunConfig,
    batch: BenchmarkBatch,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> list[float]:
    with inference_context():
        for _index in range(warmup):
            run_single_inference(model=model, config=config, batch=batch, device=device)
        synchronize_device(device=device)

        durations_ms: list[float] = []
        for _index in range(repeat):
            synchronize_device(device=device)
            start_time = time.perf_counter()
            run_single_inference(model=model, config=config, batch=batch, device=device)
            synchronize_device(device=device)
            durations_ms.append((time.perf_counter() - start_time) * 1000.0)
    return durations_ms


def percentile_nearest(values: list[float], percentile: float) -> float:
    if len(values) == 0:
        raise ValueError("Cannot calculate percentile for an empty value list.")
    sorted_values = sorted(values)
    index = round((percentile / 100.0) * (len(sorted_values) - 1))
    return float(sorted_values[int(index)])


def summarize_durations_ms(durations_ms: list[float], batch_size: int) -> TimingStats:
    if len(durations_ms) == 0:
        raise ValueError("Cannot summarize an empty benchmark duration list.")
    if batch_size <= 0:
        raise ValueError(f"Benchmark batch_size must be positive, got {batch_size}")
    for duration_ms in durations_ms:
        if not math.isfinite(duration_ms) or duration_ms < 0.0:
            raise ValueError(
                f"Benchmark durations must be finite and non-negative, got {duration_ms}: "
                f"durations_ms={durations_ms}, batch_size={batch_size}"
            )

    mean_ms = float(statistics.mean(durations_ms))
    return TimingStats(
        mean_ms=mean_ms,
        median_ms=float(statistics.median(durations_ms)),
        p90_ms=percentile_nearest(values=durations_ms, percentile=90.0),
        min_ms=float(min(durations_ms)),
        max_ms=float(max(durations_ms)),
        fps=float(batch_size * 1000.0 / mean_ms),
    )


def summarize_benchmark_calls(durations_ms: list[float], sample_counts: list[int]) -> TimingStats:
    if len(durations_ms) == 0:
        raise ValueError("Cannot summarize an empty benchmark duration list.")
    if len(durations_ms) != len(sample_counts):
        raise ValueError(
            "Benchmark durations and sample counts must have the same length: "
            f"durations={len(durations_ms)}, sample_counts={len(sample_counts)}"
        )
    total_duration_ms = float(sum(durations_ms))
    if total_duration_ms <= 0.0:
        raise ValueError(f"Benchmark total duration must be positive, got {total_duration_ms}")
    total_samples = 0
    for sample_count in sample_counts:
        if sample_count <= 0:
            raise ValueError(f"Benchmark sample counts must be positive, got {sample_count}")
        total_samples += sample_count

    stats = summarize_durations_ms(durations_ms=durations_ms, batch_size=1)
    return TimingStats(
        mean_ms=stats.mean_ms,
        median_ms=stats.median_ms,
        p90_ms=stats.p90_ms,
        min_ms=stats.min_ms,
        max_ms=stats.max_ms,
        fps=float(total_samples * 1000.0 / total_duration_ms),
    )


def build_report_paths(output_dir: Path, model_label: str, timestamp: str) -> tuple[Path, Path]:
    safe_label = model_label.lower().replace("-", "").replace("_", "")
    return (
        output_dir / f"benchmark_{safe_label}_{timestamp}.csv",
        output_dir / f"benchmark_{safe_label}_{timestamp}.json",
    )


def format_timing_summary(
    model_label: str,
    config_path: Path,
    checkpoint_path: Path,
    device_name: str,
    gpu_name: str,
    input_shape: tuple[int, int],
    warmup: int,
    repeat: int,
    batch_size: int,
    stats: TimingStats,
) -> dict[str, object]:
    return {
        "model_label": model_label,
        "config_path": config_path.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
        "device": device_name,
        "gpu_name": gpu_name,
        "height": int(input_shape[0]),
        "width": int(input_shape[1]),
        "warmup": int(warmup),
        "repeat": int(repeat),
        "batch_size": int(batch_size),
        "mean_ms": stats.mean_ms,
        "median_ms": stats.median_ms,
        "p90_ms": stats.p90_ms,
        "min_ms": stats.min_ms,
        "max_ms": stats.max_ms,
        "fps": stats.fps,
    }


def write_benchmark_reports(
    output_dir: Path,
    model_label: str,
    summary: dict[str, object],
    rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path, json_path = build_report_paths(output_dir=output_dir, model_label=model_label, timestamp=timestamp)
    fieldnames = list(rows[0].keys()) if len(rows) > 0 else list(summary.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return csv_path, json_path


def get_gpu_name(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    return torch.cuda.get_device_name(device)


def run_benchmark(default_config: str, model_label: str, argv: list[str] | None) -> None:
    args = parse_benchmark_args(default_config=default_config, model_label=model_label, argv=argv)
    device = torch.device(args.device)
    require_available_device(device=device)
    config = build_inference_run_config(config_path=args.config_path, project_root=PROJECT_ROOT)
    samples = discover_benchmark_samples(input_dir=args.input_dir)
    batches = build_benchmark_batches(samples=samples, batch_size=args.batch_size)
    model = load_benchmark_model(config=config, device=device)

    rows: list[dict[str, object]] = []
    all_durations_ms: list[float] = []
    all_sample_counts: list[int] = []
    for batch_index, batch in enumerate(batches):
        durations_ms = measure_batch_ms(
            model=model,
            config=config,
            batch=batch,
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
        )
        stats = summarize_durations_ms(durations_ms=durations_ms, batch_size=len(batch.sample_names))
        all_durations_ms.extend(durations_ms)
        all_sample_counts.extend([len(batch.sample_names)] * len(durations_ms))
        rows.append(
            {
                "batch_index": batch_index,
                "sample_names": "|".join(batch.sample_names),
                "height": batch.image_shape[0],
                "width": batch.image_shape[1],
                "batch_size": len(batch.sample_names),
                "mean_ms": stats.mean_ms,
                "median_ms": stats.median_ms,
                "p90_ms": stats.p90_ms,
                "min_ms": stats.min_ms,
                "max_ms": stats.max_ms,
                "fps": stats.fps,
            }
        )

    summary_stats = summarize_benchmark_calls(durations_ms=all_durations_ms, sample_counts=all_sample_counts)
    summary = format_timing_summary(
        model_label=model_label,
        config_path=args.config_path,
        checkpoint_path=config.checkpoint_path,
        device_name=str(device),
        gpu_name=get_gpu_name(device=device),
        input_shape=batches[0].image_shape,
        warmup=args.warmup,
        repeat=args.repeat,
        batch_size=args.batch_size,
        stats=summary_stats,
    )
    csv_path, json_path = write_benchmark_reports(
        output_dir=args.output_dir,
        model_label=model_label,
        summary=summary,
        rows=rows,
    )
    print(f"Wrote benchmark CSV: {csv_path}")
    print(f"Wrote benchmark JSON: {json_path}")
