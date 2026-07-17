from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from benchmarks.common import BenchmarkBatch
from benchmarks.common import BenchmarkSample
from benchmarks.common import build_batch_report_metadata
from benchmarks.common import build_inference_batch
from benchmarks.common import build_report_paths
from benchmarks.common import build_batch_report_row
from benchmarks.common import build_benchmark_batches
from benchmarks.common import discover_benchmark_samples
from benchmarks.common import format_timing_summary
from benchmarks.common import measure_batch_phases
from benchmarks.common import resolve_rgb_path
from benchmarks.common import summarize_phase_rows
from benchmarks.common import summarize_benchmark_calls
from benchmarks.common import summarize_durations_ms
from src.engine.run_config import build_flow_approx_config
from src.engine.run_config import InferenceRunConfig
from src.engine.run_config import MetricRunConfig
from src.engine.run_config import ModelRunConfig
from src.engine.run_config import RgbSequenceConfig


def test_model_benchmark_entrypoints_import() -> None:
    for module_name in (
        "benchmarks.benchmark_rife",
        "benchmarks.benchmark_uprnet",
        "benchmarks.benchmark_emavfi",
        "benchmarks.benchmark_sgmvfi",
        "benchmarks.benchmark_flowapprox",
    ):
        module = importlib.import_module(module_name)
        assert callable(module.main)


@pytest.mark.parametrize(
    "script_path",
    [
        Path("benchmarks/benchmark_rife.py"),
        Path("benchmarks/benchmark_uprnet.py"),
        Path("benchmarks/benchmark_emavfi.py"),
        Path("benchmarks/benchmark_sgmvfi.py"),
        Path("benchmarks/benchmark_flowapprox.py"),
    ],
)
def test_model_benchmark_entrypoints_support_file_script_help(script_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Benchmark" in completed.stdout


def _build_inference_config(model_name: str) -> InferenceRunConfig:
    return _build_inference_config_with_flow_approx(model_name=model_name, flow_approx_values={})


def _build_inference_config_with_flow_approx(
    model_name: str,
    flow_approx_values: dict[str, object],
) -> InferenceRunConfig:
    return InferenceRunConfig(
        mode="inference",
        model=ModelRunConfig(
            model_name=model_name,
            model_init_args={},
            eval_convex_upsampling=None,
        ),
        flow_approx=build_flow_approx_config(model_name=model_name, config_values=flow_approx_values),
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
    (sample_dir / 'meta.json').write_text(json.dumps({'frame_60_2_idx': 246}), encoding='utf-8')

    with pytest.raises(FileNotFoundError, match='img2.png'):
        discover_benchmark_samples(input_dir=tmp_path)


def test_resolve_rgb_path_raises_key_error_when_dataset_style_frame_key_is_missing(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_0001"
    sample_dir.mkdir()

    with pytest.raises(KeyError, match="frame_60_2_idx"):
        resolve_rgb_path(sample_dir=sample_dir, alias_name="img2.png", frame_key="frame_60_2_idx", meta={})


def test_discover_benchmark_samples_accepts_dataset_style_rgb_and_motion_paths(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_0001"
    sample_dir.mkdir()
    (sample_dir / "fps_30").mkdir()
    (sample_dir / "fps_60").mkdir()
    _write_rgb_image(sample_dir / "colorNoScreenUI_244.png", width=4, height=3, value=10)
    _write_rgb_image(sample_dir / "colorNoScreenUI_246.png", width=4, height=3, value=20)
    (sample_dir / "fps_30" / "backwardVel_Depth_123.exr").write_bytes(b"fake-exr")
    (sample_dir / "fps_30" / "forwardVel_Depth_122.exr").write_bytes(b"fake-exr")
    (sample_dir / "fps_60" / "backwardVel_Depth_245.exr").write_bytes(b"fake-exr")
    (sample_dir / "fps_60" / "forwardVel_Depth_245.exr").write_bytes(b"fake-exr")
    (sample_dir / "meta.json").write_text(
        json.dumps(
            {
                "timestep": 0.5,
                "frame_60_0_idx": 244,
                "frame_60_1_idx": 245,
                "frame_60_2_idx": 246,
                "frame_30_0_idx": 122,
                "frame_30_1_idx": 123,
            }
        ),
        encoding="utf-8",
    )

    samples = discover_benchmark_samples(input_dir=tmp_path)

    assert samples[0].img0_path.name == "colorNoScreenUI_244.png"
    assert samples[0].img2_path.name == "colorNoScreenUI_246.png"
    assert samples[0].bmv_30_path is not None
    assert samples[0].bmv_30_path.name == "backwardVel_Depth_123.exr"
    assert samples[0].fmv_30_path is not None
    assert samples[0].fmv_30_path.name == "forwardVel_Depth_122.exr"
    assert samples[0].bmv_60_path is not None
    assert samples[0].bmv_60_path.name == "backwardVel_Depth_245.exr"
    assert samples[0].fmv_60_path is not None
    assert samples[0].fmv_60_path.name == "forwardVel_Depth_245.exr"


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
        phase_summary={
            "transfer_mean_ms": 1.0,
            "flow_approx_mean_ms": 0.0,
            "model_mean_ms": 19.0,
            "total_mean_ms": 20.0,
            "flow_approx_percent": 0.0,
            "model_percent": 95.0,
            "fps_total": 50.0,
            "fps_model_only": 52.63157894736842,
        },
        report_metadata={},
    )

    assert summary["model_label"] == "UPRNet"
    assert summary["config_path"] == "configs/run/inference_uprnet_official.yaml"
    assert summary["checkpoint_path"] == "src/models/external/UPR-Net/checkpoints/upr-base.pkl"
    assert summary["mean_ms"] == 20.0
    assert summary["fps"] == 50.0
    assert summary["transfer_mean_ms"] == pytest.approx(1.0)
    assert summary["fps_model_only"] == pytest.approx(52.63157894736842)


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
            bmv_30_path=None,
            fmv_30_path=None,
            bmv_60_path=None,
            fmv_60_path=None,
            bmv_30_expected_path=None,
            fmv_30_expected_path=None,
            bmv_60_expected_path=None,
            fmv_60_expected_path=None,
        )
    ]

    batches = build_benchmark_batches(samples=samples, batch_size=1, config=_build_inference_config(model_name="UPRNet"))

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
            bmv_30_path=None,
            fmv_30_path=None,
            bmv_60_path=None,
            fmv_60_path=None,
            bmv_30_expected_path=None,
            fmv_30_expected_path=None,
            bmv_60_expected_path=None,
            fmv_60_expected_path=None,
        )
    ]

    with pytest.raises(ValueError, match="share shape"):
        build_benchmark_batches(samples=samples, batch_size=1, config=_build_inference_config(model_name="UPRNet"))


def test_build_benchmark_batches_rejects_mixed_shapes_across_run(tmp_path: Path) -> None:
    first_sample_dir = tmp_path / "sample_0001"
    second_sample_dir = tmp_path / "sample_0002"
    first_sample_dir.mkdir()
    second_sample_dir.mkdir()
    _write_rgb_image(first_sample_dir / "img0.png", width=4, height=3, value=10)
    _write_rgb_image(first_sample_dir / "img2.png", width=4, height=3, value=20)
    _write_rgb_image(second_sample_dir / "img0.png", width=5, height=3, value=30)
    _write_rgb_image(second_sample_dir / "img2.png", width=5, height=3, value=40)
    samples = [
        BenchmarkSample(
            name="sample_0001",
            img0_path=first_sample_dir / "img0.png",
            img2_path=first_sample_dir / "img2.png",
            timestep=0.25,
            bmv_30_path=None,
            fmv_30_path=None,
            bmv_60_path=None,
            fmv_60_path=None,
            bmv_30_expected_path=None,
            fmv_30_expected_path=None,
            bmv_60_expected_path=None,
            fmv_60_expected_path=None,
        ),
        BenchmarkSample(
            name="sample_0002",
            img0_path=second_sample_dir / "img0.png",
            img2_path=second_sample_dir / "img2.png",
            timestep=0.5,
            bmv_30_path=None,
            fmv_30_path=None,
            bmv_60_path=None,
            fmv_60_path=None,
            bmv_30_expected_path=None,
            fmv_30_expected_path=None,
            bmv_60_expected_path=None,
            fmv_60_expected_path=None,
        ),
    ]

    with pytest.raises(ValueError, match="entire run must share shape"):
        build_benchmark_batches(samples=samples, batch_size=1, config=_build_inference_config(model_name="UPRNet"))


def test_build_benchmark_batches_loads_motion_and_depth_with_existing_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_dir = tmp_path / "sample_0001"
    sample_dir.mkdir()
    (sample_dir / "fps_30").mkdir()
    (sample_dir / "fps_60").mkdir()
    _write_rgb_image(sample_dir / "img0.png", width=4, height=3, value=10)
    _write_rgb_image(sample_dir / "img2.png", width=4, height=3, value=20)
    for motion_path in (
        sample_dir / "fps_30" / "backwardVel_Depth_1.exr",
        sample_dir / "fps_30" / "forwardVel_Depth_0.exr",
    ):
        motion_path.write_bytes(b"fake-exr")
    (sample_dir / "meta.json").write_text(
        json.dumps({"frame_30_0_idx": 0, "frame_30_1_idx": 1}),
        encoding="utf-8",
    )

    def fake_load_backward_velocity(velocity_path: Path) -> tuple[Any, Any]:
        motion = torch.zeros((3, 4, 2), dtype=torch.float32).numpy()
        depth = torch.ones((3, 4), dtype=torch.float32).numpy()
        return motion, depth

    monkeypatch.setattr("benchmarks.common.load_backward_velocity", fake_load_backward_velocity)

    samples = discover_benchmark_samples(input_dir=tmp_path)
    batches = build_benchmark_batches(
        samples=samples,
        batch_size=1,
        config=_build_inference_config_with_flow_approx(
            model_name="IFRNet_Residual_FlowApprox",
            flow_approx_values={"flow_approx_method": "linear_splatting"},
        ),
    )

    assert batches[0].bmv_30 is not None
    assert tuple(batches[0].bmv_30.shape) == (1, 2, 3, 4)
    assert batches[0].source_depth1 is not None
    assert tuple(batches[0].source_depth1.shape) == (1, 1, 3, 4)


def test_build_benchmark_batches_rejects_motion_and_depth_shape_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_dir = tmp_path / "sample_0001"
    sample_dir.mkdir()
    (sample_dir / "fps_30").mkdir()
    _write_rgb_image(sample_dir / "img0.png", width=4, height=3, value=10)
    _write_rgb_image(sample_dir / "img2.png", width=4, height=3, value=20)
    for motion_path in (
        sample_dir / "fps_30" / "backwardVel_Depth_1.exr",
        sample_dir / "fps_30" / "forwardVel_Depth_0.exr",
    ):
        motion_path.write_bytes(b"fake-exr")
    (sample_dir / "meta.json").write_text(
        json.dumps({"frame_30_0_idx": 0, "frame_30_1_idx": 1}),
        encoding="utf-8",
    )

    def fake_load_backward_velocity(velocity_path: Path) -> tuple[Any, Any]:
        motion = torch.zeros((2, 5, 2), dtype=torch.float32).numpy()
        depth = torch.ones((2, 5), dtype=torch.float32).numpy()
        return motion, depth

    monkeypatch.setattr("benchmarks.common.load_backward_velocity", fake_load_backward_velocity)

    samples = discover_benchmark_samples(input_dir=tmp_path)

    with pytest.raises(ValueError, match="sample_0001.*motion_shape=.*depth_shape=.*expected_image_shape=\\(3, 4\\)"):
        build_benchmark_batches(
            samples=samples,
            batch_size=1,
            config=_build_inference_config_with_flow_approx(
                model_name="IFRNet_Residual_FlowApprox",
                flow_approx_values={"flow_approx_method": "linear_splatting"},
            ),
        )


def test_build_benchmark_batches_skips_optional_exr_loading_for_image_only_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_sample_dir = tmp_path / "sample_0001"
    second_sample_dir = tmp_path / "sample_0002"
    first_sample_dir.mkdir()
    second_sample_dir.mkdir()
    (second_sample_dir / "fps_30").mkdir()
    for sample_dir, img0_value, img2_value in (
        (first_sample_dir, 10, 20),
        (second_sample_dir, 30, 40),
    ):
        _write_rgb_image(sample_dir / "img0.png", width=4, height=3, value=img0_value)
        _write_rgb_image(sample_dir / "img2.png", width=4, height=3, value=img2_value)
    (second_sample_dir / "fps_30" / "backwardVel_Depth_1.exr").write_bytes(b"fake-exr")
    (second_sample_dir / "meta.json").write_text(
        json.dumps({"frame_30_1_idx": 1}),
        encoding="utf-8",
    )

    def fail_if_optional_exr_is_loaded(_velocity_path: Path) -> tuple[Any, Any]:
        raise AssertionError("image-only benchmark batches must not load optional EXR tensors")

    monkeypatch.setattr("benchmarks.common.load_backward_velocity", fail_if_optional_exr_is_loaded)

    samples = discover_benchmark_samples(input_dir=tmp_path)

    batches = build_benchmark_batches(samples=samples, batch_size=2, config=_build_inference_config(model_name="UPRNet"))

    assert len(batches) == 1
    assert batches[0].bmv_30 is None
    assert batches[0].fmv_30 is None
    assert batches[0].bmv_60 is None
    assert batches[0].fmv_60 is None
    assert batches[0].source_depth0 is None
    assert batches[0].source_depth1 is None


def test_build_inference_batch_returns_flow_aware_tuple_for_flow_approx_model() -> None:
    config = _build_inference_config(model_name="IFRNet_Residual_FlowApprox")
    batch = BenchmarkBatch(
        sample_names=["sample_0001"],
        img0=torch.ones((1, 3, 2, 4), dtype=torch.float32),
        img1=torch.full((1, 3, 2, 4), 2.0, dtype=torch.float32),
        embt=torch.full((1, 1, 1, 1), 0.25, dtype=torch.float32),
        image_shape=(2, 4),
        bmv_30=torch.full((1, 2, 2, 4), 3.0, dtype=torch.float32),
        fmv_30=torch.full((1, 2, 2, 4), 4.0, dtype=torch.float32),
        bmv_60=torch.full((1, 2, 2, 4), 5.0, dtype=torch.float32),
        fmv_60=torch.full((1, 2, 2, 4), 6.0, dtype=torch.float32),
        source_depth0=torch.full((1, 1, 2, 4), 7.0, dtype=torch.float32),
        source_depth1=torch.full((1, 1, 2, 4), 8.0, dtype=torch.float32),
        required_tensor_contexts={},
    )

    inference_batch = build_inference_batch(config=config, batch=batch)
    img0, imgt, img1, bmv_60, fmv_60, bmv_30, fmv_30, embt, info = inference_batch

    assert torch.equal(img0, batch.img0)
    assert torch.equal(img1, batch.img1)
    assert float(imgt.sum().item()) == 0.0
    assert torch.equal(bmv_60, batch.bmv_60)
    assert torch.equal(fmv_60, batch.fmv_60)
    assert torch.equal(bmv_30, batch.bmv_30)
    assert torch.equal(fmv_30, batch.fmv_30)
    assert torch.equal(embt, batch.embt)
    assert info == {
        "source_depth0": batch.source_depth0,
        "source_depth1": batch.source_depth1,
    }


def test_build_inference_batch_reports_missing_required_exr_context_for_flow_aware_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_dir = tmp_path / "sample_0001"
    sample_dir.mkdir()
    (sample_dir / "fps_30").mkdir()
    _write_rgb_image(sample_dir / "img0.png", width=4, height=3, value=10)
    _write_rgb_image(sample_dir / "img2.png", width=4, height=3, value=20)
    for motion_path in (
        sample_dir / "fps_30" / "backwardVel_Depth_123.exr",
        sample_dir / "fps_30" / "forwardVel_Depth_122.exr",
    ):
        motion_path.write_bytes(b"fake-exr")
    (sample_dir / "meta.json").write_text(
        json.dumps(
            {
                "frame_60_1_idx": 245,
                "frame_30_0_idx": 122,
                "frame_30_1_idx": 123,
            }
        ),
        encoding="utf-8",
    )

    def fake_load_backward_velocity(_velocity_path: Path) -> tuple[Any, Any]:
        motion = torch.zeros((3, 4, 2), dtype=torch.float32).numpy()
        depth = torch.ones((3, 4), dtype=torch.float32).numpy()
        return motion, depth

    monkeypatch.setattr("benchmarks.common.load_backward_velocity", fake_load_backward_velocity)

    samples = discover_benchmark_samples(input_dir=tmp_path)
    config = _build_inference_config_with_flow_approx(
        model_name="IFRNet_Residual_FlowApprox",
        flow_approx_values={"flow_approx_method": "linear_splatting", "splatting_fill_strategy": "ground_truth"},
    )
    batches = build_benchmark_batches(samples=samples, batch_size=1, config=config)

    with pytest.raises(
        ValueError,
        match="sample_0001.*bmv_60.*fps_60.*backwardVel_Depth_245\\.exr",
    ):
        build_inference_batch(config=config, batch=batches[0])


def test_build_inference_batch_uses_only_30fps_source_motion_for_non_ground_truth_fill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_dir = tmp_path / "sample_0001"
    sample_dir.mkdir()
    (sample_dir / "fps_30").mkdir()
    _write_rgb_image(sample_dir / "img0.png", width=4, height=3, value=10)
    _write_rgb_image(sample_dir / "img2.png", width=4, height=3, value=20)
    for motion_path in (
        sample_dir / "fps_30" / "backwardVel_Depth_1.exr",
        sample_dir / "fps_30" / "forwardVel_Depth_0.exr",
    ):
        motion_path.write_bytes(b"fake-exr")
    (sample_dir / "meta.json").write_text(
        json.dumps({"frame_30_0_idx": 0, "frame_30_1_idx": 1}),
        encoding="utf-8",
    )

    def fake_load_backward_velocity(_velocity_path: Path) -> tuple[Any, Any]:
        motion = torch.zeros((3, 4, 2), dtype=torch.float32).numpy()
        depth = torch.ones((3, 4), dtype=torch.float32).numpy()
        return motion, depth

    monkeypatch.setattr("benchmarks.common.load_backward_velocity", fake_load_backward_velocity)

    config = _build_inference_config(model_name="IFRNet_Residual_FlowApprox")
    samples = discover_benchmark_samples(input_dir=tmp_path)
    batches = build_benchmark_batches(samples=samples, batch_size=1, config=config)

    inference_batch = build_inference_batch(config=config, batch=batches[0])
    _img0, _imgt, _img1, bmv_60, fmv_60, bmv_30, fmv_30, _embt, info = inference_batch

    assert tuple(bmv_60.shape) == (1, 2, 3, 4)
    assert tuple(fmv_60.shape) == (1, 2, 3, 4)
    assert float(bmv_60.sum().item()) == 0.0
    assert float(fmv_60.sum().item()) == 0.0
    assert torch.equal(bmv_30, batches[0].bmv_30)
    assert torch.equal(fmv_30, batches[0].fmv_30)
    assert info == {"source_depth0": None, "source_depth1": None}


def test_measure_batch_phases_reports_zero_flow_for_image_only_models(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_inference_config(model_name="UPRNet")
    batch = BenchmarkBatch(
        sample_names=["sample_0001"],
        img0=torch.ones((1, 3, 2, 4), dtype=torch.float32),
        img1=torch.full((1, 3, 2, 4), 2.0, dtype=torch.float32),
        embt=torch.full((1, 1, 1, 1), 0.25, dtype=torch.float32),
        image_shape=(2, 4),
        bmv_30=None,
        fmv_30=None,
        bmv_60=None,
        fmv_60=None,
        source_depth0=None,
        source_depth1=None,
        required_tensor_contexts={},
    )

    class ModelWithoutDirectInference:
        pass

    calls: list[dict[str, Any]] = []

    def fake_prepare_benchmark_batch_inputs(
        config: InferenceRunConfig,
        batch: tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> dict[str, Any]:
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
        return {"prepared": True}

    def fake_build_benchmark_init_flow(config: InferenceRunConfig, phase_inputs: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("image-only models should not measure flow approximation")

    def fake_run_benchmark_model_phase(
        config: InferenceRunConfig,
        model: Any,
        batch_inputs: dict[str, Any],
        init_flow: Any,
    ) -> dict[str, str]:
        assert batch_inputs == {"prepared": True}
        assert init_flow is None
        return {"status": "ok"}

    monkeypatch.setattr("benchmarks.common.prepare_benchmark_batch_inputs", fake_prepare_benchmark_batch_inputs)
    monkeypatch.setattr("benchmarks.common.build_benchmark_init_flow", fake_build_benchmark_init_flow)
    monkeypatch.setattr("benchmarks.common.run_benchmark_model_phase", fake_run_benchmark_model_phase)

    durations = measure_batch_phases(
        model=ModelWithoutDirectInference(),
        config=config,
        batch=batch,
        warmup=1,
        repeat=2,
        device=torch.device("cpu"),
    )

    assert len(durations) == 2
    assert len(calls) == 3
    assert all(call["device"] == "cpu" for call in calls)
    assert all(call["img0_device"] == "cpu" for call in calls)
    assert all(call["img1_device"] == "cpu" for call in calls)
    assert all(call["embt_device"] == "cpu" for call in calls)
    assert all(call["imgt_sum"] == 0.0 for call in calls)
    assert all(call["bmv_shape"] == (1, 2, 2, 4) for call in calls)
    assert all(call["fmv_shape"] == (1, 2, 2, 4) for call in calls)
    assert all(call["info"] == {} for call in calls)
    assert all(duration.flow_approx_ms == 0.0 for duration in durations)


def test_measure_batch_phases_skips_flow_phase_for_ifrnet_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_inference_config(model_name="IFRNet")
    batch = BenchmarkBatch(
        sample_names=["sample_0001"],
        img0=torch.ones((1, 3, 2, 4), dtype=torch.float32),
        img1=torch.full((1, 3, 2, 4), 2.0, dtype=torch.float32),
        embt=torch.full((1, 1, 1, 1), 0.25, dtype=torch.float32),
        image_shape=(2, 4),
        bmv_30=None,
        fmv_30=None,
        bmv_60=None,
        fmv_60=None,
        source_depth0=None,
        source_depth1=None,
        required_tensor_contexts={},
    )

    def fake_prepare_benchmark_batch_inputs(
        config: InferenceRunConfig,
        batch: tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> dict[str, str]:
        return {"prepared": "baseline"}

    def fake_build_benchmark_init_flow(config: InferenceRunConfig, phase_inputs: dict[str, str]) -> dict[str, str]:
        raise AssertionError("baseline IFRNet should not enter the benchmark flow-approx phase")

    model_phase_calls: list[Any] = []

    def fake_run_benchmark_model_phase(
        config: InferenceRunConfig,
        model: Any,
        batch_inputs: dict[str, str],
        init_flow: Any,
    ) -> dict[str, str]:
        model_phase_calls.append(init_flow)
        assert batch_inputs == {"prepared": "baseline"}
        assert init_flow is None
        return {"status": "ok"}

    monkeypatch.setattr("benchmarks.common.prepare_benchmark_batch_inputs", fake_prepare_benchmark_batch_inputs)
    monkeypatch.setattr("benchmarks.common.build_benchmark_init_flow", fake_build_benchmark_init_flow)
    monkeypatch.setattr("benchmarks.common.run_benchmark_model_phase", fake_run_benchmark_model_phase)

    durations = measure_batch_phases(
        model=object(),
        config=config,
        batch=batch,
        warmup=1,
        repeat=2,
        device=torch.device("cpu"),
    )

    assert len(model_phase_calls) == 3
    assert all(init_flow is None for init_flow in model_phase_calls)
    assert all(duration.flow_approx_ms == 0.0 for duration in durations)


def test_measure_batch_phases_keeps_flow_approx_work_out_of_model_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_inference_config(model_name="IFRNet_Residual_FlowApprox")
    batch = BenchmarkBatch(
        sample_names=["sample_0001"],
        img0=torch.ones((1, 3, 2, 4), dtype=torch.float32),
        img1=torch.full((1, 3, 2, 4), 2.0, dtype=torch.float32),
        embt=torch.full((1, 1, 1, 1), 0.25, dtype=torch.float32),
        image_shape=(2, 4),
        bmv_30=torch.full((1, 2, 2, 4), 3.0, dtype=torch.float32),
        fmv_30=torch.full((1, 2, 2, 4), 4.0, dtype=torch.float32),
        bmv_60=torch.full((1, 2, 2, 4), 5.0, dtype=torch.float32),
        fmv_60=torch.full((1, 2, 2, 4), 6.0, dtype=torch.float32),
        source_depth0=torch.full((1, 1, 2, 4), 7.0, dtype=torch.float32),
        source_depth1=torch.full((1, 1, 2, 4), 8.0, dtype=torch.float32),
        required_tensor_contexts={},
    )

    build_calls: list[dict[str, Any]] = []
    model_phase_calls: list[dict[str, Any]] = []

    def fake_prepare_benchmark_batch_inputs(
        config: InferenceRunConfig,
        batch: tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> dict[str, str]:
        return {"prepared": "flow-approx"}

    def fake_build_benchmark_init_flow(config: InferenceRunConfig, phase_inputs: dict[str, str]) -> dict[str, Any]:
        build_index = len(build_calls)
        init_flow = {"init_flow_id": build_index}
        build_calls.append({"phase_inputs": phase_inputs, "init_flow": init_flow})
        return init_flow

    def fake_run_benchmark_model_phase(
        config: InferenceRunConfig,
        model: Any,
        batch_inputs: dict[str, str],
        init_flow: Any,
    ) -> dict[str, str]:
        model_phase_calls.append({"batch_inputs": batch_inputs, "init_flow": init_flow})
        assert batch_inputs == {"prepared": "flow-approx"}
        assert init_flow == build_calls[len(model_phase_calls) - 1]["init_flow"]
        return {"status": "ok"}

    monkeypatch.setattr("benchmarks.common.prepare_benchmark_batch_inputs", fake_prepare_benchmark_batch_inputs)
    monkeypatch.setattr("benchmarks.common.build_benchmark_init_flow", fake_build_benchmark_init_flow)
    monkeypatch.setattr("benchmarks.common.run_benchmark_model_phase", fake_run_benchmark_model_phase)

    durations = measure_batch_phases(
        model=object(),
        config=config,
        batch=batch,
        warmup=1,
        repeat=2,
        device=torch.device("cpu"),
    )

    assert len(build_calls) == 3
    assert len(model_phase_calls) == 3
    assert all(duration.flow_approx_ms >= 0.0 for duration in durations)
    assert all(call["init_flow"] is not None for call in model_phase_calls)


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


def test_phase_report_rows_include_transfer_flow_and_model_fields() -> None:
    rows = [
        {
            "transfer_ms": 1.0,
            "flow_approx_ms": 2.0,
            "model_ms": 7.0,
            "total_ms": 10.0,
            "batch_size": 1,
        }
    ]

    summary = summarize_phase_rows(rows=rows)

    assert summary["transfer_mean_ms"] == pytest.approx(1.0)
    assert summary["flow_approx_mean_ms"] == pytest.approx(2.0)
    assert summary["model_mean_ms"] == pytest.approx(7.0)
    assert summary["total_mean_ms"] == pytest.approx(10.0)
    assert summary["flow_approx_percent"] == pytest.approx(20.0)
    assert summary["model_percent"] == pytest.approx(70.0)


def test_report_metadata_includes_motion_and_depth_shapes_when_present() -> None:
    batch = BenchmarkBatch(
        sample_names=["sample_0001"],
        img0=torch.ones((1, 3, 2, 4), dtype=torch.float32),
        img1=torch.full((1, 3, 2, 4), 2.0, dtype=torch.float32),
        embt=torch.full((1, 1, 1, 1), 0.25, dtype=torch.float32),
        image_shape=(2, 4),
        bmv_30=torch.full((1, 2, 2, 4), 3.0, dtype=torch.float32),
        fmv_30=torch.full((1, 2, 2, 4), 4.0, dtype=torch.float32),
        bmv_60=None,
        fmv_60=None,
        source_depth0=torch.full((1, 1, 2, 4), 7.0, dtype=torch.float32),
        source_depth1=torch.full((1, 1, 2, 4), 8.0, dtype=torch.float32),
        required_tensor_contexts={},
    )
    phase_summary = {
        "transfer_mean_ms": 1.0,
        "flow_approx_mean_ms": 2.0,
        "model_mean_ms": 7.0,
        "total_mean_ms": 10.0,
        "flow_approx_percent": 20.0,
        "model_percent": 70.0,
        "fps_total": 100.0,
        "fps_model_only": 125.0,
    }
    stats = summarize_durations_ms(durations_ms=[10.0, 10.0], batch_size=1)

    row = build_batch_report_row(batch_index=0, batch=batch, phase_summary=phase_summary)
    summary = format_timing_summary(
        model_label="IFRNet_Residual_FlowApprox",
        config_path=Path("configs/run/inference_ifrnet_residual_flowapprox.yaml"),
        checkpoint_path=Path("checkpoint.pt"),
        device_name="cpu",
        gpu_name="",
        input_shape=batch.image_shape,
        warmup=1,
        repeat=2,
        batch_size=1,
        stats=stats,
        phase_summary=phase_summary,
        report_metadata=build_batch_report_metadata(batch=batch),
    )

    assert row["motion_shape"] == "(2, 2, 4)"
    assert row["depth_shape"] == "(1, 2, 4)"
    assert summary["motion_shape"] == "(2, 2, 4)"
    assert summary["depth_shape"] == "(1, 2, 4)"


def test_benchmark_materialized_inputs_are_git_ignored() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "check-ignore", "benchmarks/inputs/sample_0001/img0.png"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
