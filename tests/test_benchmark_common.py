from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from benchmarks.common import BenchmarkBatch
from benchmarks.common import BenchmarkSample
from benchmarks.common import build_report_paths
from benchmarks.common import build_benchmark_batches
from benchmarks.common import discover_benchmark_samples
from benchmarks.common import format_timing_summary
from benchmarks.common import measure_batch_ms
from benchmarks.common import summarize_benchmark_calls
from benchmarks.common import summarize_durations_ms
from src.engine.run_config import FlowApproxConfig
from src.engine.run_config import InferenceRunConfig
from src.engine.run_config import MetricRunConfig
from src.engine.run_config import ModelRunConfig
from src.engine.run_config import RgbSequenceConfig


def _build_inference_config(model_name: str) -> InferenceRunConfig:
    return InferenceRunConfig(
        mode="inference",
        model=ModelRunConfig(
            model_name=model_name,
            model_init_args={},
            eval_convex_upsampling=None,
        ),
        flow_approx=FlowApproxConfig(
            method="combination",
            splatting_fill_strategy="none",
            effective_splatting_fill_strategy="",
            init_flow_downscale_strategy="bilinear",
            effective_init_flow_downscale_strategy="",
            init_flow_mask_epsilon=1e-6,
        ),
        metrics=MetricRunConfig(values={"psnr": True}),
        inference_presets=["benchmark"],
        root_dir=Path("."),
        dataset_root_dir=Path("."),
        checkpoint_path=Path("checkpoint.pt"),
        output_dir=Path("outputs"),
        seed=0,
        batch_size=1,
        only_fps=30,
        input_fps=15,
        scale_factor=1.0,
        flow_diff_threshold=1.0,
        flow_diff_percentile=99.0,
        save_topk_worst_psnr=0,
        save_topk_best_psnr=0,
        save_topk_largest_flow_diff=0,
        rgb_sequence=RgbSequenceConfig(
            enabled=False,
            output_dir=Path("outputs/rgb"),
            record_filter=None,
            mode_filter=None,
            dedupe_endpoints=True,
            export_target=False,
            export_prediction=False,
        ),
        input_config={},
    )


def _write_rgb_image(image_path: Path, width: int, height: int, value: int) -> None:
    image = Image.new("RGB", (width, height), color=(value, value, value))
    image.save(image_path)


def test_discover_benchmark_samples_reads_sorted_samples_and_default_timestep(tmp_path: Path) -> None:
    later_sample = tmp_path / 'sample_0002'
    first_sample = tmp_path / 'sample_0001'
    later_sample.mkdir()
    first_sample.mkdir()
    for sample_dir in (later_sample, first_sample):
        (sample_dir / 'img0.png').write_bytes(b'not-used-by-discovery')
        (sample_dir / 'img2.png').write_bytes(b'not-used-by-discovery')
    (later_sample / 'meta.json').write_text(json.dumps({'timestep': 0.25}), encoding='utf-8')

    samples = discover_benchmark_samples(input_dir=tmp_path)

    assert [sample.name for sample in samples] == ['sample_0001', 'sample_0002']
    assert samples[0].timestep == 0.5
    assert samples[1].timestep == 0.25


def test_discover_benchmark_samples_fails_when_required_image_is_missing(tmp_path: Path) -> None:
    sample_dir = tmp_path / 'sample_0001'
    sample_dir.mkdir()
    (sample_dir / 'img0.png').write_bytes(b'not-used-by-discovery')

    with pytest.raises(FileNotFoundError, match='img2.png'):
        discover_benchmark_samples(input_dir=tmp_path)


@pytest.mark.parametrize('timestep', [math.nan])
def test_read_sample_timestep_rejects_non_finite_values(tmp_path: Path, timestep: float) -> None:
    sample_dir = tmp_path / 'sample_0001'
    sample_dir.mkdir()
    (sample_dir / 'img0.png').write_bytes(b'not-used-by-discovery')
    (sample_dir / 'img2.png').write_bytes(b'not-used-by-discovery')
    (sample_dir / 'meta.json').write_text(json.dumps({'timestep': timestep}), encoding='utf-8')

    with pytest.raises(ValueError, match='finite'):
        discover_benchmark_samples(input_dir=tmp_path)


def test_summarize_durations_ms_reports_batch_fps() -> None:
    stats = summarize_durations_ms(durations_ms=[10.0, 20.0, 30.0], batch_size=2)

    assert stats.mean_ms == pytest.approx(20.0)
    assert stats.median_ms == pytest.approx(20.0)
    assert stats.p90_ms == pytest.approx(30.0)
    assert stats.min_ms == pytest.approx(10.0)
    assert stats.max_ms == pytest.approx(30.0)
    assert stats.fps == pytest.approx(100.0)


@pytest.mark.parametrize('durations_ms', [[-1.0], [math.nan], [math.inf]])
def test_summarize_durations_ms_rejects_invalid_values(durations_ms: list[float]) -> None:
    with pytest.raises(ValueError, match='finite|non-negative'):
        summarize_durations_ms(durations_ms=durations_ms, batch_size=2)


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


def test_build_benchmark_batches_keeps_tensors_on_cpu_before_measurement(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_0001"
    sample_dir.mkdir()
    _write_rgb_image(sample_dir / "img0.png", width=4, height=3, value=10)
    _write_rgb_image(sample_dir / "img2.png", width=4, height=3, value=20)
    samples = [
        BenchmarkSample(
            name="sample_0001",
            img0_path=sample_dir / "img0.png",
            img2_path=sample_dir / "img2.png",
            timestep=0.25,
        )
    ]

    batches = build_benchmark_batches(samples=samples, batch_size=1)

    assert len(batches) == 1
    assert batches[0].img0.device.type == "cpu"
    assert batches[0].img1.device.type == "cpu"
    assert batches[0].embt.device.type == "cpu"


def test_build_benchmark_batches_rejects_incompatible_image_shapes(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_0001"
    sample_dir.mkdir()
    _write_rgb_image(sample_dir / "img0.png", width=4, height=3, value=10)
    _write_rgb_image(sample_dir / "img2.png", width=5, height=3, value=20)
    samples = [
        BenchmarkSample(
            name="sample_0001",
            img0_path=sample_dir / "img0.png",
            img2_path=sample_dir / "img2.png",
            timestep=0.25,
        )
    ]

    with pytest.raises(ValueError, match="share shape"):
        build_benchmark_batches(samples=samples, batch_size=1)


def test_measure_batch_ms_uses_run_inference_batch_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_inference_config(model_name="UPRNet")
    batch = BenchmarkBatch(
        sample_names=["sample_0001"],
        img0=torch.ones((1, 3, 2, 4), dtype=torch.float32),
        img1=torch.full((1, 3, 2, 4), 2.0, dtype=torch.float32),
        embt=torch.full((1, 1, 1, 1), 0.25, dtype=torch.float32),
        image_shape=(2, 4),
    )

    class ModelWithoutDirectInference:
        pass

    calls: list[dict[str, Any]] = []

    def fake_run_inference_batch(
        config: InferenceRunConfig,
        model: Any,
        batch: tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> dict[str, str]:
        img0, imgt, img1, bmv, fmv, embt, info = batch
        calls.append(
            {
                "device": str(device),
                "img0_device": img0.device.type,
                "imgt_sum": float(imgt.sum().item()),
                "img1_device": img1.device.type,
                "bmv_shape": tuple(int(size) for size in bmv.shape),
                "fmv_shape": tuple(int(size) for size in fmv.shape),
                "embt_device": embt.device.type,
                "info": info,
            }
        )
        return {"status": "ok"}

    monkeypatch.setattr("benchmarks.common.run_inference_batch_wrapper", fake_run_inference_batch)

    durations_ms = measure_batch_ms(
        model=ModelWithoutDirectInference(),
        config=config,
        batch=batch,
        warmup=1,
        repeat=2,
        device=torch.device("cpu"),
    )

    assert len(durations_ms) == 2
    assert len(calls) == 3
    assert all(call["device"] == "cpu" for call in calls)
    assert all(call["img0_device"] == "cpu" for call in calls)
    assert all(call["img1_device"] == "cpu" for call in calls)
    assert all(call["embt_device"] == "cpu" for call in calls)
    assert all(call["imgt_sum"] == 0.0 for call in calls)
    assert all(call["bmv_shape"] == (1, 2, 2, 4) for call in calls)
    assert all(call["fmv_shape"] == (1, 2, 2, 4) for call in calls)
    assert all(call["info"] == {} for call in calls)


def test_summarize_benchmark_calls_uses_actual_processed_sample_count() -> None:
    stats = summarize_benchmark_calls(
        durations_ms=[10.0, 30.0, 20.0],
        sample_counts=[2, 2, 1],
    )

    assert stats.mean_ms == pytest.approx(20.0)
    assert stats.median_ms == pytest.approx(20.0)
    assert stats.p90_ms == pytest.approx(30.0)
    assert stats.min_ms == pytest.approx(10.0)
    assert stats.max_ms == pytest.approx(30.0)
    assert stats.fps == pytest.approx((5 * 1000.0) / 60.0)
