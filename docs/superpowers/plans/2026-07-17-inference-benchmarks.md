# Inference Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing benchmark scripts so image-only baselines and project flow-approximation models can be timed from real materialized samples, with flow approximation measured separately from model inference.

**Architecture:** Keep model-specific scripts thin and put shared loading, validation, phase timing, and report writing in `benchmarks/common.py`. Reuse `src.data.dataset_loader` and `src.engine.interpolation_batch` conventions instead of duplicating the training/inference pipeline logic. Materialized benchmark inputs live under `benchmarks/inputs/` and are ignored by Git.

**Tech Stack:** Python, PyTorch, PIL, OpenCV EXR loading through existing `src.data.image_ops`, existing `src.engine.run_config`, existing `src.engine.model_registry`, existing `src.engine.interpolation_batch`, pytest smoke tests.

## Global Constraints

- The current branch already contains `benchmarks/common.py`, image-only benchmark entrypoints, and `tests/test_benchmark_common.py`.
- Ignore materialized benchmark inputs under `benchmarks/inputs/` so real RGB/EXR samples are never staged by accident.
- Timing includes tensor transfer to the selected device and model inference, including wrapper padding, TTA, timestep handling, and flow-approximation setup when the selected model uses it.
- Timing excludes dataset DataFrame creation, metric calculation, image saving, video export, RGB sequence export, and top-k artifact export.
- RGB inputs may use benchmark aliases `img0.png` and `img2.png` or dataset-style `colorNoScreenUI_{frame_idx}.png` resolved through `meta.json`.
- Game motion inputs use the training dataloader EXR naming: `backwardVel_Depth_{frame_idx}.exr` and `forwardVel_Depth_{frame_idx}.exr`.
- `IFRNet_Residual_FlowApprox` source motion follows `FlowEstimationTrainDataset`: `bmv_30` from `backwardVel_Depth_{frame_30_1_idx}.exr` and `fmv_30` from `forwardVel_Depth_{frame_30_0_idx}.exr`.
- Splatting flow-approximation modes read source depth from the same 30fps EXR files through `load_backward_velocity`.
- Fail fast for missing config, missing input directory, empty input directory, missing sample images, missing required EXR files, incompatible image/motion/depth shapes, unavailable CUDA when requested, and checkpoint/model loading errors.
- Use CUDA synchronization before and after each measured phase when the selected device is CUDA.

---

## File Structure

- Modify `.gitignore`: add benchmark input/output artifact directories.
- Modify `benchmarks/common.py`: sample metadata parsing, EXR motion/depth loading, flow-aware batch construction, phase timing, and phase report fields.
- Create `benchmarks/benchmark_flowapprox.py`: benchmark entrypoint for `IFRNet_Residual_FlowApprox`.
- Modify `tests/test_benchmark_common.py`: smoke tests for Git ignore behavior, metadata path resolution, EXR loader conversion through monkeypatching, flow-aware batch contracts, and phase report aggregation.
- Modify `docs/superpowers/plans/2026-07-17-inference-benchmarks.md`: track this remaining implementation sequence.

---

### Task 1: Ignore Materialized Benchmark Inputs

**Files:**
- Modify: `.gitignore`
- Modify: `tests/test_benchmark_common.py`

**Interfaces:**
- Produces: Git ignore rules for `benchmarks/inputs/` and `benchmarks/outputs/`.

- [ ] **Step 1: Add a smoke test for ignore behavior**

Append to `tests/test_benchmark_common.py`:

```python
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
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
pytest tests/test_benchmark_common.py::test_benchmark_materialized_inputs_are_git_ignored -q
```

Expected: FAIL because `.gitignore` does not yet ignore `benchmarks/inputs/`.

- [ ] **Step 3: Add benchmark artifact ignore rules**

Add this block to `.gitignore` under `# Generated artifacts`:

```gitignore
benchmarks/inputs/
benchmarks/outputs/
```

- [ ] **Step 4: Run the ignore test and verify it passes**

Run:

```powershell
pytest tests/test_benchmark_common.py::test_benchmark_materialized_inputs_are_git_ignored -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add .gitignore tests/test_benchmark_common.py
git commit -m "chore: ignore benchmark input artifacts"
```

---

### Task 2: Resolve RGB And EXR Sample Paths

**Files:**
- Modify: `benchmarks/common.py`
- Modify: `tests/test_benchmark_common.py`

**Interfaces:**
- Modifies: `BenchmarkSample`
- Produces: `read_sample_meta(sample_dir: Path) -> dict[str, object]`
- Produces: `resolve_rgb_path(sample_dir: Path, alias_name: str, frame_key: str, meta: dict[str, object]) -> Path`
- Produces: `resolve_motion_path(sample_dir: Path, fps_dir: str, modality_name: str, frame_key: str, meta: dict[str, object]) -> Path | None`

- [ ] **Step 1: Add tests for dataset-style RGB and EXR path resolution**

Append to `tests/test_benchmark_common.py`:

```python
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
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```powershell
pytest tests/test_benchmark_common.py::test_discover_benchmark_samples_accepts_dataset_style_rgb_and_motion_paths -q
```

Expected: FAIL because `BenchmarkSample` does not expose EXR paths and discovery only accepts `img0.png` / `img2.png`.

- [ ] **Step 3: Extend sample metadata and path resolution**

In `benchmarks/common.py`, replace `BenchmarkSample` with:

```python
@dataclass(frozen=True)
class BenchmarkSample:
    name: str
    img0_path: Path
    img2_path: Path
    timestep: float
    bmv_30_path: Path | None
    fmv_30_path: Path | None
    bmv_60_path: Path | None
    fmv_60_path: Path | None
```

Add these helpers:

```python
def read_sample_meta(sample_dir: Path) -> dict[str, object]:
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        return {}
    raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(raw_meta, dict):
        raise TypeError(f"Benchmark sample meta.json must contain a mapping: path={meta_path}")
    return raw_meta


def resolve_frame_index(meta: dict[str, object], frame_key: str, meta_path_context: Path) -> int:
    if frame_key not in meta:
        raise KeyError(f"Benchmark sample meta is missing {frame_key}: sample={meta_path_context}")
    return int(meta[frame_key])


def resolve_rgb_path(sample_dir: Path, alias_name: str, frame_key: str, meta: dict[str, object]) -> Path:
    alias_path = sample_dir / alias_name
    if alias_path.is_file():
        return alias_path
    frame_index = resolve_frame_index(meta=meta, frame_key=frame_key, meta_path_context=sample_dir)
    dataset_style_path = sample_dir / f"colorNoScreenUI_{frame_index}.png"
    if not dataset_style_path.is_file():
        raise FileNotFoundError(
            f"Benchmark sample is missing RGB input: alias={alias_path}, dataset_style={dataset_style_path}"
        )
    return dataset_style_path


def resolve_motion_path(
    sample_dir: Path,
    fps_dir: str,
    modality_name: str,
    frame_key: str,
    meta: dict[str, object],
) -> Path | None:
    if frame_key not in meta:
        return None
    frame_index = int(meta[frame_key])
    motion_path = sample_dir / fps_dir / f"{modality_name}_{frame_index}.exr"
    return motion_path if motion_path.is_file() else None
```

Update `discover_benchmark_samples` so each sample uses one parsed `meta` dict, resolves RGB paths through `resolve_rgb_path`, and stores the four optional EXR paths.

- [ ] **Step 4: Run benchmark helper tests**

Run:

```powershell
pytest tests/test_benchmark_common.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add benchmarks/common.py tests/test_benchmark_common.py
git commit -m "Add benchmark EXR sample path resolution"
```

---

### Task 3: Load Game Motion And Build Flow-Aware Batches

**Files:**
- Modify: `benchmarks/common.py`
- Modify: `tests/test_benchmark_common.py`

**Interfaces:**
- Modifies: `BenchmarkBatch`
- Produces: `load_motion_and_depth_tensors(motion_path: Path) -> tuple[torch.Tensor, torch.Tensor]`
- Produces: `build_inference_batch(config: InferenceRunConfig, batch: BenchmarkBatch) -> tuple[torch.Tensor, ...]`

- [ ] **Step 1: Add tests for EXR conversion without reading real EXR files**

Append to `tests/test_benchmark_common.py`:

```python
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
    batches = build_benchmark_batches(samples=samples, batch_size=1)

    assert batches[0].bmv_30 is not None
    assert tuple(batches[0].bmv_30.shape) == (1, 2, 3, 4)
    assert batches[0].source_depth1 is not None
    assert tuple(batches[0].source_depth1.shape) == (1, 1, 3, 4)
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```powershell
pytest tests/test_benchmark_common.py::test_build_benchmark_batches_loads_motion_and_depth_with_existing_loader -q
```

Expected: FAIL because `BenchmarkBatch` does not carry motion/depth tensors.

- [ ] **Step 3: Extend batch data and loaders**

In `benchmarks/common.py`, import the existing loader:

```python
from src.data.image_ops import load_backward_velocity
from src.data.dataset_loader import depth_to_tensor
from src.data.dataset_loader import flow_to_tensor
```

Replace `BenchmarkBatch` with fields for optional motion/depth:

```python
@dataclass(frozen=True)
class BenchmarkBatch:
    sample_names: list[str]
    img0: torch.Tensor
    img1: torch.Tensor
    embt: torch.Tensor
    image_shape: tuple[int, int]
    bmv_30: torch.Tensor | None
    fmv_30: torch.Tensor | None
    bmv_60: torch.Tensor | None
    fmv_60: torch.Tensor | None
    source_depth0: torch.Tensor | None
    source_depth1: torch.Tensor | None
```

Add:

```python
def load_motion_and_depth_tensors(motion_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    motion, depth = load_backward_velocity(motion_path)
    return flow_to_tensor(motion), depth_to_tensor(depth)
```

Update `build_benchmark_batches` to stack each optional tensor only when every sample in the batch has that tensor. If some samples have a required tensor and others do not, raise `ValueError` with the sample names and tensor name.

Update `build_inference_batch(config, batch)` to return the flow-aware tuple when `uses_flow_approx_model(config.model.model_name)` is true:

```python
return (
    batch.img0,
    torch.zeros_like(batch.img0),
    batch.img1,
    require_batch_tensor(batch.bmv_60, "bmv_60", batch.sample_names),
    require_batch_tensor(batch.fmv_60, "fmv_60", batch.sample_names),
    require_batch_tensor(batch.bmv_30, "bmv_30", batch.sample_names),
    require_batch_tensor(batch.fmv_30, "fmv_30", batch.sample_names),
    batch.embt,
    {"source_depth0": batch.source_depth0, "source_depth1": batch.source_depth1}
)
```

For image-only models, keep the current RGB tuple with zero placeholder flows.

- [ ] **Step 4: Run benchmark helper tests**

Run:

```powershell
pytest tests/test_benchmark_common.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add benchmarks/common.py tests/test_benchmark_common.py
git commit -m "Add benchmark game-motion batch loading"
```

---

### Task 4: Measure Transfer, Flow Approximation, And Model Phases Separately

**Files:**
- Modify: `src/engine/interpolation_batch.py`
- Modify: `benchmarks/common.py`
- Modify: `tests/test_benchmark_common.py`

**Interfaces:**
- Produces: `BenchmarkPhaseBatchInputs`
- Produces: `prepare_benchmark_batch_inputs(config: InferenceRunConfig, batch: Any, device: Any) -> BenchmarkPhaseBatchInputs`
- Produces: `build_benchmark_init_flow(config: InferenceRunConfig, phase_inputs: BenchmarkPhaseBatchInputs) -> BenchmarkPhaseBatchInputs`
- Produces: `run_benchmark_model_phase(config: InferenceRunConfig, model: Any, batch_inputs: BenchmarkPhaseBatchInputs, init_flow: Any) -> Any`
- Produces: `PhaseDurations(transfer_ms: float, flow_approx_ms: float, model_ms: float, total_ms: float)`

- [ ] **Step 1: Add a phase timing aggregation test**

Append to `tests/test_benchmark_common.py`:

```python
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
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```powershell
pytest tests/test_benchmark_common.py::test_phase_report_rows_include_transfer_flow_and_model_fields -q
```

Expected: FAIL because `summarize_phase_rows` does not exist.

- [ ] **Step 3: Expose benchmark phase helpers in `src/engine/interpolation_batch.py`**

Add public wrappers that reuse existing private logic without changing normal training/inference:

```python
@dataclass(frozen=True)
class BenchmarkPhaseBatchInputs:
    inputs: _PreparedBatchInputs
    init_flow: _InitFlowState | None


def prepare_benchmark_batch_inputs(config: InferenceRunConfig, batch: Any, device: Any) -> BenchmarkPhaseBatchInputs:
    inputs = _prepare_batch_inputs(
        model_name=config.model.model_name,
        batch=batch,
        device=device,
        flow_approx_method=config.flow_approx.method,
    )
    return BenchmarkPhaseBatchInputs(inputs=inputs, init_flow=None)


def build_benchmark_init_flow(config: InferenceRunConfig, phase_inputs: BenchmarkPhaseBatchInputs) -> BenchmarkPhaseBatchInputs:
    if uses_image_only_vfi_model(config.model.model_name):
        return phase_inputs
    init_flow = _build_init_flow(
        model_name=config.model.model_name,
        source_bmv=phase_inputs.inputs.source_bmv,
        source_fmv=phase_inputs.inputs.source_fmv,
        embt=phase_inputs.inputs.embt,
        flow_approx=config.flow_approx,
        source_depth0=phase_inputs.inputs.source_depth0,
        source_depth1=phase_inputs.inputs.source_depth1,
        ground_truth_bmv=phase_inputs.inputs.bmv,
        ground_truth_fmv=phase_inputs.inputs.fmv,
    )
    return BenchmarkPhaseBatchInputs(inputs=phase_inputs.inputs, init_flow=init_flow)
```

Add a model-phase helper that mirrors `_run_model_inference` but uses the prebuilt `init_flow` for residual flow-approx models.

- [ ] **Step 4: Update benchmark timing to measure each phase**

In `benchmarks/common.py`, add:

```python
@dataclass(frozen=True)
class PhaseDurations:
    transfer_ms: float
    flow_approx_ms: float
    model_ms: float
    total_ms: float
```

Replace `measure_batch_ms` with `measure_batch_phases`, which:

1. Builds the CPU inference batch.
2. Synchronizes and times `prepare_benchmark_batch_inputs`.
3. Synchronizes and times `build_benchmark_init_flow`.
4. Synchronizes and times `run_benchmark_model_phase`.
5. Returns `PhaseDurations`.

Add `summarize_phase_rows(rows: list[dict[str, object]]) -> dict[str, float]`.

- [ ] **Step 5: Run benchmark helper tests**

Run:

```powershell
pytest tests/test_benchmark_common.py -q
```

Expected: PASS.

- [ ] **Step 6: Run compile verification**

Run:

```powershell
python -m compileall benchmarks src/engine/interpolation_batch.py tests
```

Expected: exit code 0.

- [ ] **Step 7: Commit Task 4**

Run:

```powershell
git add benchmarks/common.py src/engine/interpolation_batch.py tests/test_benchmark_common.py
git commit -m "Add benchmark phase timing"
```

---

### Task 5: Add Flow-Approximation Benchmark Entrypoint

**Files:**
- Create: `benchmarks/benchmark_flowapprox.py`
- Modify: `tests/test_benchmark_common.py`

**Interfaces:**
- Consumes: `run_benchmark(default_config: str, model_label: str, argv: list[str] | None) -> None`
- Produces: CLI entrypoint for `IFRNet_Residual_FlowApprox`.

- [ ] **Step 1: Add entrypoint import and help tests**

Extend the existing entrypoint test module list with:

```python
"benchmarks.benchmark_flowapprox",
```

Extend the existing script help parametrization with:

```python
Path("benchmarks/benchmark_flowapprox.py"),
```

- [ ] **Step 2: Run the entrypoint tests and verify they fail**

Run:

```powershell
pytest tests/test_benchmark_common.py::test_model_benchmark_entrypoints_import tests/test_benchmark_common.py::test_model_benchmark_entrypoints_support_file_script_help -q
```

Expected: FAIL because `benchmarks/benchmark_flowapprox.py` does not exist.

- [ ] **Step 3: Add `benchmarks/benchmark_flowapprox.py`**

```python
from __future__ import annotations

from benchmarks.common import run_benchmark


def main(argv: list[str] | None = None) -> None:
    run_benchmark(
        default_config="configs/run/inference_flowaprox_layer_1_0618.yaml",
        model_label="IFRNet_Residual_FlowApprox",
        argv=argv,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the entrypoint tests and verify they pass**

Run:

```powershell
pytest tests/test_benchmark_common.py::test_model_benchmark_entrypoints_import tests/test_benchmark_common.py::test_model_benchmark_entrypoints_support_file_script_help -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

Run:

```powershell
git add benchmarks/benchmark_flowapprox.py tests/test_benchmark_common.py
git commit -m "Add flow-approx benchmark entrypoint"
```

---

### Task 6: Final Verification

**Files:**
- No source changes unless verification exposes a root-cause issue.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: final verified branch state.

- [ ] **Step 1: Run benchmark smoke tests**

Run:

```powershell
pytest tests/test_benchmark_common.py -q
```

Expected: PASS.

- [ ] **Step 2: Run compile verification**

Run:

```powershell
python -m compileall benchmarks src/engine/interpolation_batch.py tests
```

Expected: exit code 0.

- [ ] **Step 3: Verify ignored input samples**

Run:

```powershell
git check-ignore benchmarks/inputs/sample_0001/img0.png
```

Expected output:

```text
benchmarks/inputs/sample_0001/img0.png
```

- [ ] **Step 4: Inspect final diff**

Run:

```powershell
git --no-pager diff --stat origin/codex/repo-architecture-review...HEAD
git status --short --branch
```

Expected: branch ahead with only planned commits and no unstaged changes.

---

## Self-Review Notes

- Spec coverage: `.gitignore`, EXR input naming, source depth from EXR, flow-approx model support, separate transfer/flow/model timing, CSV/JSON phase fields, and fail-fast validation are covered.
- Placeholder scan: no placeholder markers or open-ended implementation steps remain.
- Type consistency: `BenchmarkSample`, `BenchmarkBatch`, `PhaseDurations`, and interpolation phase helpers are introduced before later tasks consume them.
