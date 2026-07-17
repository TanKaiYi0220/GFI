# Inference Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `benchmarks/` folder with `benchmark_xxx.py` scripts that measure model inference time from existing inference YAML files and pre-stored real input samples.

**Architecture:** Put shared benchmark logic in `benchmarks/common.py` and keep each `benchmark_xxx.py` file as a thin entrypoint with a default config path and model label. Benchmark inputs are already-materialized sample folders, so no dataset DataFrame construction, metrics, image export, or video export runs during timing.

**Tech Stack:** Python, PyTorch, PIL, existing `src.engine.run_config`, existing `src.engine.model_registry`, existing `src.engine.checkpoints`, pytest smoke tests.

## Global Constraints

- Execute in a full branch checkout that contains the baseline model wrappers and `src/engine/checkpoints.py`; the temp architecture-review folder may not be a Git repository or full source checkout.
- Timing includes tensor transfer to the selected device and the model inference call, including wrapper padding, TTA, and timestep handling.
- Timing excludes dataset DataFrame creation, metric calculation, image saving, video export, RGB sequence export, and top-k artifact export.
- Inputs come from `benchmarks/inputs/<sample_name>/img0.png` and `benchmarks/inputs/<sample_name>/img2.png`.
- Optional `benchmarks/inputs/<sample_name>/meta.json` may define `{"timestep": 0.5}`.
- Fail fast for missing config, missing input directory, empty input directory, missing sample images, incompatible image shapes in a batch, unavailable CUDA when requested, and checkpoint/model loading errors.
- Use CUDA synchronization before and after measured calls when the selected device is CUDA.

---

## File Structure

- Create `benchmarks/common.py`: sample discovery, image loading, model loading, timed inference, statistics, and report writing.
- Create `benchmarks/benchmark_rife.py`: RIFE entrypoint.
- Create `benchmarks/benchmark_uprnet.py`: UPR-Net entrypoint.
- Create `benchmarks/benchmark_emavfi.py`: EMA-VFI entrypoint.
- Create `benchmarks/benchmark_sgmvfi.py`: SGM-VFI entrypoint.
- Create `tests/test_benchmark_common.py`: smoke tests for pure helper behavior that does not require CUDA or checkpoints.

---

### Task 1: Sample Discovery And Timing Statistics

**Files:**
- Create: `benchmarks/common.py`
- Create: `tests/test_benchmark_common.py`

**Interfaces:**
- Produces: `BenchmarkSample(name: str, img0_path: Path, img2_path: Path, timestep: float)`
- Produces: `TimingStats(mean_ms: float, median_ms: float, p90_ms: float, min_ms: float, max_ms: float, fps: float)`
- Produces: `discover_benchmark_samples(input_dir: Path) -> list[BenchmarkSample]`
- Produces: `summarize_durations_ms(durations_ms: list[float], batch_size: int) -> TimingStats`

- [ ] **Step 1: Write the failing helper tests**

Create `tests/test_benchmark_common.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.common import discover_benchmark_samples
from benchmarks.common import summarize_durations_ms


def test_discover_benchmark_samples_reads_sorted_samples_and_default_timestep(tmp_path: Path) -> None:
    later_sample = tmp_path / "sample_0002"
    first_sample = tmp_path / "sample_0001"
    later_sample.mkdir()
    first_sample.mkdir()
    for sample_dir in (later_sample, first_sample):
        (sample_dir / "img0.png").write_bytes(b"not-used-by-discovery")
        (sample_dir / "img2.png").write_bytes(b"not-used-by-discovery")
    (later_sample / "meta.json").write_text(json.dumps({"timestep": 0.25}), encoding="utf-8")

    samples = discover_benchmark_samples(input_dir=tmp_path)

    assert [sample.name for sample in samples] == ["sample_0001", "sample_0002"]
    assert samples[0].timestep == 0.5
    assert samples[1].timestep == 0.25


def test_discover_benchmark_samples_fails_when_required_image_is_missing(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_0001"
    sample_dir.mkdir()
    (sample_dir / "img0.png").write_bytes(b"not-used-by-discovery")

    with pytest.raises(FileNotFoundError, match="img2.png"):
        discover_benchmark_samples(input_dir=tmp_path)


def test_summarize_durations_ms_reports_batch_fps() -> None:
    stats = summarize_durations_ms(durations_ms=[10.0, 20.0, 30.0], batch_size=2)

    assert stats.mean_ms == pytest.approx(20.0)
    assert stats.median_ms == pytest.approx(20.0)
    assert stats.p90_ms == pytest.approx(30.0)
    assert stats.min_ms == pytest.approx(10.0)
    assert stats.max_ms == pytest.approx(30.0)
    assert stats.fps == pytest.approx(100.0)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
pytest tests/test_benchmark_common.py -q
```

Expected: FAIL because `benchmarks.common` does not exist.

- [ ] **Step 3: Add the minimal helper implementation**

Create `benchmarks/common.py`:

```python
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
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        return 0.5
    raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(raw_meta, dict):
        raise TypeError(f"Benchmark sample meta.json must contain a mapping: path={meta_path}")
    timestep = float(raw_meta.get("timestep", 0.5))
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

    mean_ms = float(statistics.mean(durations_ms))
    return TimingStats(
        mean_ms=mean_ms,
        median_ms=float(statistics.median(durations_ms)),
        p90_ms=percentile_nearest(values=durations_ms, percentile=90.0),
        min_ms=float(min(durations_ms)),
        max_ms=float(max(durations_ms)),
        fps=float(batch_size * 1000.0 / mean_ms),
    )
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```powershell
pytest tests/test_benchmark_common.py -q
```

Expected: PASS for all three tests.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add benchmarks/common.py tests/test_benchmark_common.py
git commit -m "Add benchmark sample helpers"
```

---

### Task 2: Model Loading, Input Batching, Timing, And Reports

**Files:**
- Modify: `benchmarks/common.py`
- Modify: `tests/test_benchmark_common.py`

**Interfaces:**
- Consumes: `BenchmarkSample`, `TimingStats`, `discover_benchmark_samples`, `summarize_durations_ms`
- Produces: `BenchmarkRunArgs`
- Produces: `parse_benchmark_args(default_config: str, model_label: str, argv: list[str] | None) -> BenchmarkRunArgs`
- Produces: `run_benchmark(default_config: str, model_label: str, argv: list[str] | None) -> None`
- Produces: `write_benchmark_reports(output_dir: Path, model_label: str, summary: dict[str, object], rows: list[dict[str, object]]) -> tuple[Path, Path]`

- [ ] **Step 1: Add report formatting tests**

Append to `tests/test_benchmark_common.py`:

```python
from benchmarks.common import build_report_paths
from benchmarks.common import format_timing_summary


def test_format_timing_summary_includes_config_and_checkpoint() -> None:
    stats = summarize_durations_ms(durations_ms=[20.0, 20.0], batch_size=1)

    summary = format_timing_summary(
        model_label="UPRNet",
        config_path=Path("configs/run/inference_uprnet_official.yaml"),
        checkpoint_path=Path("src/models/external/UPR-Net/checkpoints/upr-base.pkl"),
        device_name="cuda",
        gpu_name="NVIDIA RTX",
        input_shape=(720, 1280),
        warmup=5,
        repeat=30,
        batch_size=1,
        stats=stats,
    )

    assert summary["model_label"] == "UPRNet"
    assert summary["config_path"] == "configs/run/inference_uprnet_official.yaml"
    assert summary["checkpoint_path"] == "src/models/external/UPR-Net/checkpoints/upr-base.pkl"
    assert summary["mean_ms"] == 20.0
    assert summary["fps"] == 50.0


def test_build_report_paths_uses_model_label_and_timestamp(tmp_path: Path) -> None:
    csv_path, json_path = build_report_paths(
        output_dir=tmp_path,
        model_label="UPRNet",
        timestamp="20260717_010203",
    )

    assert csv_path == tmp_path / "benchmark_uprnet_20260717_010203.csv"
    assert json_path == tmp_path / "benchmark_uprnet_20260717_010203.json"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
pytest tests/test_benchmark_common.py -q
```

Expected: FAIL because `format_timing_summary` and `build_report_paths` do not exist.

- [ ] **Step 3: Extend `benchmarks/common.py` with runner helpers**

Add these imports near the top of `benchmarks/common.py`:

```python
import argparse
import csv
import sys
import time
from datetime import datetime
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

Add these imports after `PROJECT_ROOT` is defined:

```python
import torch
from PIL import Image

from src.engine.checkpoints import load_inference_checkpoint
from src.engine.model_registry import resolve_model_class
from src.engine.model_registry import set_model_convex_upsampling
from src.engine.run_config import InferenceRunConfig
from src.engine.run_config import build_inference_run_config
```

Add these dataclasses and functions:

```python
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


def build_benchmark_batches(samples: list[BenchmarkSample], batch_size: int, device: torch.device) -> list[BenchmarkBatch]:
    if batch_size <= 0:
        raise ValueError(f"Benchmark batch_size must be positive, got {batch_size}")

    batches: list[BenchmarkBatch] = []
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
        embt = torch.tensor([sample.timestep for sample in batch_samples], dtype=torch.float32).view(-1, 1, 1, 1)
        batches.append(
            BenchmarkBatch(
                sample_names=[sample.name for sample in batch_samples],
                img0=torch.stack(img0_tensors, dim=0).to(device),
                img1=torch.stack(img1_tensors, dim=0).to(device),
                embt=embt.to(device),
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


def run_single_inference(model: torch.nn.Module, config: InferenceRunConfig, batch: BenchmarkBatch) -> torch.Tensor:
    inference = getattr(model, "inference", None)
    if not callable(inference):
        raise TypeError(f"Benchmark model type {type(model).__name__} does not provide inference().")
    prediction = inference(batch.img0, batch.img1, batch.embt, config.scale_factor)
    if not isinstance(prediction, torch.Tensor):
        raise TypeError(f"Benchmark model inference must return a tensor, got {type(prediction).__name__}")
    return prediction


def measure_batch_ms(
    model: torch.nn.Module,
    config: InferenceRunConfig,
    batch: BenchmarkBatch,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> list[float]:
    with torch.inference_mode():
        for _index in range(warmup):
            run_single_inference(model=model, config=config, batch=batch)
        synchronize_device(device=device)

        durations_ms: list[float] = []
        for _index in range(repeat):
            synchronize_device(device=device)
            start_time = time.perf_counter()
            run_single_inference(model=model, config=config, batch=batch)
            synchronize_device(device=device)
            durations_ms.append((time.perf_counter() - start_time) * 1000.0)
    return durations_ms


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
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
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
    batches = build_benchmark_batches(samples=samples, batch_size=args.batch_size, device=device)
    model = load_benchmark_model(config=config, device=device)

    rows: list[dict[str, object]] = []
    all_durations_ms: list[float] = []
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

    summary_stats = summarize_durations_ms(durations_ms=all_durations_ms, batch_size=args.batch_size)
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
```

- [ ] **Step 4: Run the helper tests and verify they pass**

Run:

```powershell
pytest tests/test_benchmark_common.py -q
```

Expected: PASS.

- [ ] **Step 5: Run compile verification**

Run:

```powershell
python -m compileall benchmarks tests
```

Expected: exit code 0.

- [ ] **Step 6: Commit Task 2**

Run:

```powershell
git add benchmarks/common.py tests/test_benchmark_common.py
git commit -m "Add benchmark inference runner"
```

---

### Task 3: Model-Specific Benchmark Entrypoints

**Files:**
- Create: `benchmarks/benchmark_rife.py`
- Create: `benchmarks/benchmark_uprnet.py`
- Create: `benchmarks/benchmark_emavfi.py`
- Create: `benchmarks/benchmark_sgmvfi.py`

**Interfaces:**
- Consumes: `run_benchmark(default_config: str, model_label: str, argv: list[str] | None) -> None`
- Produces: CLI entrypoints for RIFE, UPR-Net, EMA-VFI, and SGM-VFI.

- [ ] **Step 1: Write a smoke test for default entrypoint imports**

Append to `tests/test_benchmark_common.py`:

```python
import importlib


def test_model_benchmark_entrypoints_import() -> None:
    for module_name in (
        "benchmarks.benchmark_rife",
        "benchmarks.benchmark_uprnet",
        "benchmarks.benchmark_emavfi",
        "benchmarks.benchmark_sgmvfi",
    ):
        module = importlib.import_module(module_name)
        assert callable(module.main)
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
pytest tests/test_benchmark_common.py::test_model_benchmark_entrypoints_import -q
```

Expected: FAIL because the entrypoint modules do not exist.

- [ ] **Step 3: Add the entrypoint scripts**

Create `benchmarks/benchmark_rife.py`:

```python
from __future__ import annotations

from benchmarks.common import run_benchmark


def main(argv: list[str] | None = None) -> None:
    run_benchmark(
        default_config="configs/run/inference_rife_official.yaml",
        model_label="RIFE",
        argv=argv,
    )


if __name__ == "__main__":
    main()
```

Create `benchmarks/benchmark_uprnet.py`:

```python
from __future__ import annotations

from benchmarks.common import run_benchmark


def main(argv: list[str] | None = None) -> None:
    run_benchmark(
        default_config="configs/run/inference_uprnet_official.yaml",
        model_label="UPRNet",
        argv=argv,
    )


if __name__ == "__main__":
    main()
```

Create `benchmarks/benchmark_emavfi.py`:

```python
from __future__ import annotations

from benchmarks.common import run_benchmark


def main(argv: list[str] | None = None) -> None:
    run_benchmark(
        default_config="configs/run/inference_emavfi_official.yaml",
        model_label="EMAVFI",
        argv=argv,
    )


if __name__ == "__main__":
    main()
```

Create `benchmarks/benchmark_sgmvfi.py`:

```python
from __future__ import annotations

from benchmarks.common import run_benchmark


def main(argv: list[str] | None = None) -> None:
    run_benchmark(
        default_config="configs/run/inference_sgmvfi_official.yaml",
        model_label="SGMVFI",
        argv=argv,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the entrypoint test and verify it passes**

Run:

```powershell
pytest tests/test_benchmark_common.py::test_model_benchmark_entrypoints_import -q
```

Expected: PASS.

- [ ] **Step 5: Run all benchmark helper tests**

Run:

```powershell
pytest tests/test_benchmark_common.py -q
```

Expected: PASS.

- [ ] **Step 6: Run compile verification**

Run:

```powershell
python -m compileall benchmarks tests
```

Expected: exit code 0.

- [ ] **Step 7: Optionally run one real benchmark on the remote CUDA machine**

After placing real samples under `benchmarks/inputs`, run:

```powershell
python benchmarks/benchmark_uprnet.py --config configs/run/inference_uprnet_official.yaml --input-dir benchmarks/inputs --warmup 2 --repeat 5 --batch-size 1
```

Expected: CSV and JSON report paths are printed. If CUDA/checkpoint files are unavailable on the current machine, skip this command and record that only smoke tests and compile checks were run locally.

- [ ] **Step 8: Commit Task 3**

Run:

```powershell
git add benchmarks tests/test_benchmark_common.py
git commit -m "Add model benchmark entrypoints"
```

---

## Self-Review Notes

- Spec coverage: sample folder contract, YAML reuse, CUDA timing, CSV/JSON reports, model-specific scripts, and fail-fast validation are covered.
- Placeholder scan: no open-ended implementation markers remain.
- Type consistency: `BenchmarkSample`, `TimingStats`, `BenchmarkRunArgs`, `BenchmarkBatch`, and `run_benchmark` are defined before entrypoint tasks consume them.
