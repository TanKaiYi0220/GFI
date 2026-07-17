What you implemented
- Extended `BenchmarkBatch` in `benchmarks/common.py` to carry optional `bmv_30`, `fmv_30`, `bmv_60`, `fmv_60`, `source_depth0`, and `source_depth1` tensors.
- Added `load_motion_and_depth_tensors(motion_path: Path) -> tuple[torch.Tensor, torch.Tensor]` that reuses `src.data.image_ops.load_backward_velocity` plus `src.data.dataset_loader.flow_to_tensor` and `depth_to_tensor`.
- Updated `build_benchmark_batches` to load and stack optional motion/depth tensors only when every sample in the batch has that tensor, and to raise `ValueError` for mixed availability.
- Updated `build_inference_batch(config, batch)` to keep image-only behavior unchanged and return the flow-aware tuple order expected by `FlowEstimationTrainDataset` / `src.engine.interpolation_batch._prepare_batch_inputs` for flow-approx models.
- Added tests covering EXR-backed motion/depth loading into benchmark batches and flow-aware inference tuple construction. Updated the existing `BenchmarkBatch` test fixture to match the expanded dataclass signature.

What you tested and exact results
- Focused red/green tests:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark_common.py::test_build_benchmark_batches_loads_motion_and_depth_with_existing_loader -q`
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark_common.py::test_build_inference_batch_returns_flow_aware_tuple_for_flow_approx_model -q`
- Full benchmark helper test suite:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark_common.py -q`
- Exact latest full-suite result:
  - `24 passed, 1 warning in 13.94s`

TDD Evidence: RED command/output and GREEN command/output
- RED command:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark_common.py::test_build_benchmark_batches_loads_motion_and_depth_with_existing_loader -q`
- RED output:
```text
F                                                                        [100%]
================================== FAILURES ===================================
__ test_build_benchmark_batches_loads_motion_and_depth_with_existing_loader ___

obj = <module 'benchmarks.common' from 'C:\\Users\\User\\AppData\\Local\\Temp\\gfi-inference-benchmarks\\benchmarks\\common.py'>
name = 'load_backward_velocity', ann = 'benchmarks.common'

    def annotated_getattr(obj: object, name: str, ann: str) -> object:
        try:
>           obj = getattr(obj, name)
E           AttributeError: module 'benchmarks.common' has no attribute 'load_backward_velocity'

tests\test_benchmark_common.py:361:
E       AttributeError: 'module' object at benchmarks.common has no attribute 'load_backward_velocity'

1 failed in 1.53s
```
- GREEN command:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark_common.py::test_build_benchmark_batches_loads_motion_and_depth_with_existing_loader tests/test_benchmark_common.py::test_build_inference_batch_returns_flow_aware_tuple_for_flow_approx_model -q`
- GREEN output:
```text
..                                                                       [100%]
============================== warnings summary ===============================
tests/test_benchmark_common.py::test_build_benchmark_batches_loads_motion_and_depth_with_existing_loader
  C:\Users\User\AppData\Local\Temp\gfi-inference-benchmarks\benchmarks\common.py:111: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
    image_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
2 passed, 1 warning in 21.52s
```

Files changed
- `benchmarks/common.py`
- `tests/test_benchmark_common.py`

Self-review findings
- The flow-aware tuple order now matches `src.engine.interpolation_batch._prepare_batch_inputs`.
- Image-only benchmark batching still returns zero placeholder flows and empty info.
- Mixed optional tensor presence fails early during batch building, and missing required flow-aware tensors fail during inference batch construction.
- Motion and source depth loading currently decodes the same 30fps EXR twice per sample when both flow and depth are needed; this is correct and minimal for the task, but not optimized.

Any concerns
- `tests/test_benchmark_common.py` still emits one pre-existing `TypedStorage` deprecation warning from `load_rgb_tensor`; this task did not change that path.
- The shell environment did not expose `pytest` on PATH, so verification used `.\\.venv\\Scripts\\python.exe -m pytest`.

---

Review fix report

What I fixed
- Added fail-fast motion/depth tensor shape validation in `benchmarks/common.py` so every loaded EXR-derived tensor must match `[2, H, W]` for motion and `[1, H, W]` for depth, with `H` and `W` matching the sample RGB shape.
- Extended `BenchmarkSample` with expected EXR path metadata so flow-aware missing-input failures can report the sample name and expected EXR path even when discovery keeps the motion paths optional.
- Added `required_tensor_contexts` to `BenchmarkBatch` and upgraded `require_batch_tensor(...)` error reporting to include sample-specific missing-EXR context.
- Added focused tests for incompatible RGB/motion/depth shapes and missing required EXRs in flow-aware batch building.

What I tested and exact results
- Focused missing-required-EXR test:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark_common.py::test_build_inference_batch_reports_missing_required_exr_context_for_flow_aware_model -q`
  - Result: `1 passed, 1 warning in 3.28s`
- Focused shape-mismatch test:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark_common.py::test_build_benchmark_batches_rejects_motion_and_depth_shape_mismatch -q`
  - Result: `1 passed, 1 warning in 3.28s`
- Full benchmark helper suite:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark_common.py -q`
  - Result: `26 passed, 1 warning in 14.82s`

TDD Evidence: RED command/output and GREEN command/output
- RED command 1:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark_common.py::test_build_benchmark_batches_rejects_motion_and_depth_shape_mismatch -q`
- RED output 1:
```text
F                                                                        [100%]
================================== FAILURES ===================================
____ test_build_benchmark_batches_rejects_motion_and_depth_shape_mismatch _____

>       with pytest.raises(ValueError, match="sample_0001.*motion_shape=.*depth_shape=.*expected_image_shape=\\(3, 4\\)"):
E       Failed: DID NOT RAISE ValueError

1 failed, 1 warning in 3.80s
```
- GREEN command 1:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark_common.py::test_build_benchmark_batches_rejects_motion_and_depth_shape_mismatch -q`
- GREEN output 1:
```text
.                                                                        [100%]
============================== warnings summary ===============================
tests/test_benchmark_common.py::test_build_benchmark_batches_rejects_motion_and_depth_shape_mismatch
  C:\Users\User\AppData\Local\Temp\gfi-inference-benchmarks\benchmarks\common.py:116: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
    image_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 1 warning in 2.81s
```
- RED command 2:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark_common.py::test_build_inference_batch_reports_missing_required_exr_context_for_flow_aware_model -q`
- RED output 2:
```text
F                                                                        [100%]
================================== FAILURES ===================================
_ test_build_inference_batch_reports_missing_required_exr_context_for_flow_aware_model _

E       AssertionError: Regex pattern did not match.
E         Expected regex: 'sample_0001.*bmv_60.*backwardVel_Depth_245\\.exr.*fps_60'
E         Actual message: "Benchmark batch is missing required tensor for flow-aware inference: tensor_name=bmv_60, sample_names=['sample_0001']"

1 failed, 1 warning in 3.80s
```
- GREEN command 2:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark_common.py::test_build_inference_batch_reports_missing_required_exr_context_for_flow_aware_model -q`
- GREEN output 2:
```text
.                                                                        [100%]
============================== warnings summary ===============================
tests/test_benchmark_common.py::test_build_inference_batch_reports_missing_required_exr_context_for_flow_aware_model
  C:\Users\User\AppData\Local\Temp\gfi-inference-benchmarks\benchmarks\common.py:116: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
    image_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 1 warning in 3.28s
```

Files changed for review fixes
- `benchmarks/common.py`
- `tests/test_benchmark_common.py`

Self-review findings for review fixes
- Shape validation now fails during batch construction before any flow-aware inference call can use malformed tensors.
- Missing required flow-aware EXRs now report sample-specific expected paths, while discovery still leaves motion paths optional for image-only benchmark scenarios.
- The new expected-path metadata changed `BenchmarkSample` and `BenchmarkBatch` construction in tests only; no unrelated files were touched.

Any concerns for review fixes
- The pre-existing `TypedStorage` deprecation warning still appears in these test runs.
