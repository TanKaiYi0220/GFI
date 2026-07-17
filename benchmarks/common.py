from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path


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


def read_sample_timestep(sample_dir: Path) -> float:
    meta_path = sample_dir / 'meta.json'
    if not meta_path.exists():
        return 0.5
    raw_meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if not isinstance(raw_meta, dict):
        raise TypeError(f'Benchmark sample meta.json must contain a mapping: path={meta_path}')
    timestep = float(raw_meta.get('timestep', 0.5))
    if timestep < 0.0 or timestep > 1.0:
        raise ValueError(f'Benchmark timestep must be in [0, 1], got {timestep}: path={meta_path}')
    return timestep


def discover_benchmark_samples(input_dir: Path) -> list[BenchmarkSample]:
    if not input_dir.exists():
        raise FileNotFoundError(f'Benchmark input directory is missing: path={input_dir}')
    if not input_dir.is_dir():
        raise NotADirectoryError(f'Benchmark input path must be a directory: path={input_dir}')

    sample_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if len(sample_dirs) == 0:
        raise ValueError(f'Benchmark input directory has no sample folders: path={input_dir}')

    samples: list[BenchmarkSample] = []
    for sample_dir in sample_dirs:
        img0_path = sample_dir / 'img0.png'
        img2_path = sample_dir / 'img2.png'
        if not img0_path.is_file():
            raise FileNotFoundError(f'Benchmark sample is missing img0.png: path={img0_path}')
        if not img2_path.is_file():
            raise FileNotFoundError(f'Benchmark sample is missing img2.png: path={img2_path}')
        samples.append(
            BenchmarkSample(
                name=sample_dir.name,
                img0_path=img0_path,
                img2_path=img2_path,
                timestep=read_sample_timestep(sample_dir=sample_dir),
            )
        )
    return samples


def percentile_nearest(values: list[float], percentile: float) -> float:
    if len(values) == 0:
        raise ValueError('Cannot calculate percentile for an empty value list.')
    sorted_values = sorted(values)
    index = round((percentile / 100.0) * (len(sorted_values) - 1))
    return float(sorted_values[int(index)])


def summarize_durations_ms(durations_ms: list[float], batch_size: int) -> TimingStats:
    if len(durations_ms) == 0:
        raise ValueError('Cannot summarize an empty benchmark duration list.')
    if batch_size <= 0:
        raise ValueError(f'Benchmark batch_size must be positive, got {batch_size}')

    mean_ms = float(statistics.mean(durations_ms))
    return TimingStats(
        mean_ms=mean_ms,
        median_ms=float(statistics.median(durations_ms)),
        p90_ms=percentile_nearest(values=durations_ms, percentile=90.0),
        min_ms=float(min(durations_ms)),
        max_ms=float(max(durations_ms)),
        fps=float(batch_size * 1000.0 / mean_ms),
    )
