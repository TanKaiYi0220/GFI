What you implemented

- Added benchmark-facing phase helpers in `src/engine/interpolation_batch.py`:
  - `BenchmarkPhaseBatchInputs`
  - `prepare_benchmark_batch_inputs(...)`
  - `build_benchmark_init_flow(...)`
  - `run_benchmark_model_phase(...)`
- Reused existing `_prepare_batch_inputs`, `_build_init_flow`, and inference internals so benchmark code does not duplicate flow-approximation math.
- Added `PhaseDurations` and replaced single-duration measurement with `measure_batch_phases(...)` in `benchmarks/common.py`.
- Added `summarize_phase_rows(...)` and updated benchmark report rows and run summary to include:
  - `transfer_mean_ms`
  - `flow_approx_mean_ms`
  - `model_mean_ms`
  - `total_mean_ms`
  - `flow_approx_percent`
  - `model_percent`
  - `fps_total`
  - `fps_model_only`
- Preserved image-only behavior by reporting `flow_approx_ms = 0.0` and skipping the flow-approx timing phase for image-only models.
- Updated tests to cover phase aggregation, image-only phase timing behavior, and summary formatting.

What you tested and exact results

- Focused benchmark helper suite:
  - Command: `.\.venv\Scripts\python.exe -m pytest tests/test_benchmark_common.py -q`
  - Result: `27 passed, 1 warning in 15.25s`
- Full pytest suite:
  - Command: `.\.venv\Scripts\python.exe -m pytest -q`
  - Result: `27 passed, 1 warning in 15.25s`
- Compile verification:
  - Command: `.\.venv\Scripts\python.exe -m compileall benchmarks src\engine\interpolation_batch.py tests`
  - Result: exit code `0`

TDD Evidence: RED command/output and GREEN command/output

RED

- Command: `.\.venv\Scripts\python.exe -m pytest tests/test_benchmark_common.py::test_phase_report_rows_include_transfer_flow_and_model_fields -q`
- Output:

```text
=================================== ERRORS ====================================
_______________ ERROR collecting tests/test_benchmark_common.py _______________
ImportError while importing test module 'C:\Users\User\AppData\Local\Temp\gfi-inference-benchmarks\tests\test_benchmark_common.py'.
Traceback:
tests\test_benchmark_common.py:24: in <module>
    from benchmarks.common import summarize_phase_rows
E   ImportError: cannot import name 'summarize_phase_rows' from 'benchmarks.common' (C:\Users\User\AppData\Local\Temp\gfi-inference-benchmarks\benchmarks\common.py)
=========================== short test summary info ============================
ERROR tests/test_benchmark_common.py
1 error in 2.64s
ERROR: found no collectors for C:\Users\User\AppData\Local\Temp\gfi-inference-benchmarks\tests\test_benchmark_common.py::test_phase_report_rows_include_transfer_flow_and_model_fields
```

GREEN

- Command: `.\.venv\Scripts\python.exe -m pytest tests/test_benchmark_common.py -q`
- Output:

```text
...........................                                              [100%]
============================== warnings summary ===============================
tests/test_benchmark_common.py::test_build_benchmark_batches_keeps_tensors_on_cpu_before_measurement
  C:\Users\User\AppData\Local\Temp\gfi-inference-benchmarks\benchmarks\common.py:128: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
    image_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
27 passed, 1 warning in 15.25s
```

Files changed

- `src/engine/interpolation_batch.py`
- `benchmarks/common.py`
- `tests/test_benchmark_common.py`

Self-review findings

- The benchmark now times transfer, flow approximation, and model execution separately while keeping the inference/training paths on the shared internal logic.
- The flow-approximation benchmark path reuses existing init-flow construction instead of reimplementing any math in `benchmarks/common.py`.
- Image-only models explicitly skip the flow-approx timing phase and report `0.0` as required.
- Existing benchmark summary fields (`mean_ms`, `fps`, etc.) remain in the run-level summary, with the new phase fields added alongside them.

Any concerns

- The test suite still emits a pre-existing `TypedStorage` deprecation warning from `load_rgb_tensor` in `benchmarks/common.py`. This task did not change that code path.

Review fix report (Friday, July 17, 2026)

What I fixed

- Fixed `measure_batch_phases(...)` so the separate flow-approx timing phase runs only for real flow-approx models.
- Baseline `IFRNet` now skips `build_benchmark_init_flow(...)` in benchmark timing and reports `flow_approx_ms = 0.0`.
- Tightened `build_benchmark_init_flow(...)` so it is a no-op for non-flow-approx models in the public benchmark-facing helper.
- Removed the dead per-batch `stats = summarize_durations_ms(...)` value from `run_benchmark(...)`.
- Added focused regression coverage for:
  - baseline `IFRNet` zero-flow benchmark phase behavior
  - flow-approx phase separation, proving init-flow work is built separately and passed into the model phase

TDD Evidence for review fixes

RED

- Command: `.\.venv\Scripts\python.exe -m pytest tests/test_benchmark_common.py::test_measure_batch_phases_skips_flow_phase_for_ifrnet_baseline -q`
- Output:

```text
F                                                                        [100%]
================================== FAILURES ===================================
_______ test_measure_batch_phases_skips_flow_phase_for_ifrnet_baseline ________

...
>       raise AssertionError("baseline IFRNet should not enter the benchmark flow-approx phase")
E       AssertionError: baseline IFRNet should not enter the benchmark flow-approx phase

tests\test_benchmark_common.py:601: AssertionError
=========================== short test summary info ============================
FAILED tests/test_benchmark_common.py::test_measure_batch_phases_skips_flow_phase_for_ifrnet_baseline
1 failed in 3.57s
```

GREEN

- Command: `.\.venv\Scripts\python.exe -m pytest tests/test_benchmark_common.py::test_measure_batch_phases_skips_flow_phase_for_ifrnet_baseline -q`
- Output:

```text
.                                                                        [100%]
1 passed in 3.25s
```

Focused review-fix tests and exact results

- Baseline IFRNet zero-flow-phase test:
  - Command: `.\.venv\Scripts\python.exe -m pytest tests/test_benchmark_common.py::test_measure_batch_phases_skips_flow_phase_for_ifrnet_baseline -q`
  - Result: `1 passed in 3.25s`
- Flow-approx phase separation test:
  - Command: `.\.venv\Scripts\python.exe -m pytest tests/test_benchmark_common.py::test_measure_batch_phases_keeps_flow_approx_work_out_of_model_phase -q`
  - Result: `1 passed in 3.25s`
- Focused benchmark helper suite:
  - Command: `.\.venv\Scripts\python.exe -m pytest tests/test_benchmark_common.py -q`
  - Result: `29 passed, 1 warning in 14.62s`
- Compile verification:
  - Command: `.\.venv\Scripts\python.exe -m compileall benchmarks src\engine\interpolation_batch.py tests`
  - Result: exit code `0`

Files changed for review fixes

- `benchmarks/common.py`
- `src/engine/interpolation_batch.py`
- `tests/test_benchmark_common.py`

Self-review findings for review fixes

- The benchmark phase gate now matches the actual design boundary: only `IFRNet_Residual_FlowApprox` pays a separate flow-approx phase.
- Baseline `IFRNet` still transfers its motion tensors as part of the transfer/model path, but no longer performs or times extra init-flow work that real inference does not use.
- The new focused tests pin both sides of the contract: zero flow-phase for baseline, and no flow rebuild inside the model phase for flow-approx models.

Concerns after review fixes

- The pre-existing `TypedStorage` deprecation warning is still present during pytest. This review pass did not touch `load_rgb_tensor(...)`.
