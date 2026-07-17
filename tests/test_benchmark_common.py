from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.common import build_report_paths
from benchmarks.common import discover_benchmark_samples
from benchmarks.common import format_timing_summary
from benchmarks.common import summarize_durations_ms


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
