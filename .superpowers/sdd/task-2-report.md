# Task 2 Report

- Status: DONE_WITH_CONCERNS
- Date: 2026-07-17

## Files Changed

- `benchmarks/common.py`
- `tests/test_benchmark_common.py`
- `.superpowers/sdd/task-2-report.md`

## Commits Made

- `367d0c8` `Add benchmark inference runner`

## Test Commands And Outputs

1. Requested command:

   Command:
   ```powershell
   pytest tests/test_benchmark_common.py -q
   ```

   Output:
   ```text
   pytest : The term 'pytest' is not recognized as the name of a cmdlet, function, script file, or operable program.
   Check the spelling of the name, or if a path was included, verify that the path is correct and try again.
   At line:2 char:1
   + pytest tests/test_benchmark_common.py -q
   + ~~~~~~
       + CategoryInfo          : ObjectNotFound: (pytest:String) [], CommandNotFoundException
       + FullyQualifiedErrorId : CommandNotFoundException
   ```

2. Fallback check with system Python:

   Command:
   ```powershell
   python -m pytest tests/test_benchmark_common.py -q
   ```

   Output:
   ```text
   C:\Users\User\miniconda3\python.exe: No module named pytest
   ```

3. Red step in an available Conda environment with `pytest` and `torch`:

   Command:
   ```powershell
   conda run -n ABME python -m pytest tests/test_benchmark_common.py -q
   ```

   Output:
   ```text
   =================================== ERRORS ====================================
   _______________ ERROR collecting tests/test_benchmark_common.py _______________
   ImportError while importing test module 'C:\Users\User\AppData\Local\Temp\gfi-inference-benchmarks\tests\test_benchmark_common.py'.
   Hint: make sure your test modules/packages have valid Python names.
   Traceback:
   ..\..\..\..\miniconda3\envs\ABME\lib\importlib\__init__.py:127: in import_module
       return _bootstrap._gcd_import(name[level:], package, level)
   tests\test_benchmark_common.py:9: in <module>
       from benchmarks.common import build_report_paths
   E   ImportError: cannot import name 'build_report_paths' from 'benchmarks.common' (C:\Users\User\AppData\Local\Temp\gfi-inference-benchmarks\benchmarks\common.py)
   =========================== short test summary info ============================
   ERROR tests/test_benchmark_common.py
   !!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
   1 error in 0.10s

   ERROR conda.cli.main_run:execute(125): `conda run python -m pytest tests/test_benchmark_common.py -q` failed. (See above for error)
   ```

4. First green attempt after implementation:

   Command:
   ```powershell
   conda run -n ABME python -m pytest tests/test_benchmark_common.py -q
   ```

   Output:
   ```text
   .......F.                                                                [100%]
   ================================== FAILURES ===================================
   __________ test_format_timing_summary_includes_config_and_checkpoint __________

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
   >       assert summary["config_path"] == "configs/run/inference_uprnet_official.yaml"
   E       AssertionError: assert 'configs\\run...official.yaml' == 'configs/run/...official.yaml'
   E         - configs/run/inference_uprnet_official.yaml
   E         ?        ^   ^
   E         + configs\run\inference_uprnet_official.yaml
   E         ?        ^   ^

   tests\test_benchmark_common.py:87: AssertionError
   =========================== short test summary info ============================
   FAILED tests/test_benchmark_common.py::test_format_timing_summary_includes_config_and_checkpoint
   1 failed, 8 passed in 0.70s

   ERROR conda.cli.main_run:execute(125): `conda run python -m pytest tests/test_benchmark_common.py -q` failed. (See above for error)
   ```

5. Final test pass:

   Command:
   ```powershell
   conda run -n ABME python -m pytest tests/test_benchmark_common.py -q
   ```

   Output:
   ```text
   .........                                                                [100%]
   9 passed in 0.55s
   ```

6. Compile verification:

   Command:
   ```powershell
   python -m compileall benchmarks tests
   ```

   Output:
   ```text
   Listing 'benchmarks'...
   Listing 'tests'...
   ```

## Self-Review Notes

- Followed the requested TDD order: tests added first, observed red before implementation, then fixed one Windows-specific formatting issue exposed by the new test.
- Kept edits inside the assigned task scope and did not touch entrypoint scripts.
- `format_timing_summary()` uses `Path.as_posix()` so the report JSON remains stable across Windows and Unix path separators.
- Concern: the worktree-local `pytest` command and the default `python` interpreter were not usable for the requested test command, so verification used `conda run -n ABME python -m pytest ...` instead.

---

## Review Fixes - 2026-07-17

- Status: DONE

### Files Changed

- `benchmarks/common.py`
- `tests/test_benchmark_common.py`
- `.superpowers/sdd/task-2-report.md`

### Root Cause Summary

- `build_benchmark_batches()` moved tensors to the selected device before measurement, so host-to-device transfer time was excluded.
- `measure_batch_ms()` timed a direct `model.inference()` call instead of the repository inference wrapper, so tuple-returning and model-specific inference contracts were bypassed.
- Overall summary FPS used the requested batch size instead of the actual sample count processed by each measured call.

### Fix Summary

- Kept `BenchmarkBatch` tensors on CPU and removed the `device` parameter from `build_benchmark_batches()`.
- Added `build_inference_batch()` so benchmark batches are converted into the CPU tuple contract expected by `run_inference_batch()`, including explicit dummy `imgt`, `bmv`, `fmv`, and `info` values for image-only models.
- Routed warmup and timed measurement through `run_inference_batch_wrapper()` so the timed path matches repository inference behavior.
- Added `summarize_benchmark_calls()` so aggregate FPS uses total processed samples over total measured duration, including partial final batches.
- Added focused tests for CPU residency before measurement, wrapper-based timing, incompatible image-shape rejection, and aggregate FPS with variable batch sizes.
- Added an `inference_context()` compatibility helper because the available `ABME` environment does not provide `torch.inference_mode()`.

### Test Commands And Outputs

1. Red step after adding the new tests:

   Command:
   ```powershell
   conda run -n ABME python -m pytest tests/test_benchmark_common.py -q
   ```

   Output:
   ```text
   =================================== ERRORS ====================================
   _______________ ERROR collecting tests/test_benchmark_common.py _______________
   ImportError while importing test module 'C:\Users\User\AppData\Local\Temp\gfi-inference-benchmarks\tests\test_benchmark_common.py'.
   Hint: make sure your test modules/packages have valid Python names.
   Traceback:
   ..\..\..\..\miniconda3\envs\ABME\lib\importlib\__init__.py:127: in import_module
       return _bootstrap._gcd_import(name[level:], package, level)
   tests\test_benchmark_common.py:19: in <module>
       from benchmarks.common import summarize_benchmark_calls
   E   ImportError: cannot import name 'summarize_benchmark_calls' from 'benchmarks.common' (C:\Users\User\AppData\Local\Temp\gfi-inference-benchmarks\benchmarks\common.py)
   =========================== short test summary info ===========================
   ERROR tests/test_benchmark_common.py
   !!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
   1 error in 0.86s

   ERROR conda.cli.main_run:execute(125): `conda run python -m pytest tests/test_benchmark_common.py -q` failed. (See above for error)
   ```

2. Intermediate red step after the first implementation pass:

   Command:
   ```powershell
   conda run -n ABME python -m pytest tests/test_benchmark_common.py -q
   ```

   Output:
   ```text
   =================================== ERRORS ====================================
   _______________ ERROR collecting tests/test_benchmark_common.py _______________
   tests\test_benchmark_common.py:12: in <module>
       from benchmarks.common import BenchmarkBatch
   benchmarks\common.py:23: in <module>
       from src.engine.interpolation_batch import run_inference_batch
   src\engine\interpolation_batch.py:17: in <module>
       SplattingRegionMaps = dict[str, Any]
   E   TypeError: 'type' object is not subscriptable
   =========================== short test summary info ===========================
   ERROR tests/test_benchmark_common.py - TypeError: 'type' object is not subscr...
   !!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
   1 error in 1.70s

   ERROR conda.cli.main_run:execute(125): `conda run python -m pytest tests/test_benchmark_common.py -q` failed. (See above for error)
   ```

3. Second intermediate red step after lazy wrapper import:

   Command:
   ```powershell
   conda run -n ABME python -m pytest tests/test_benchmark_common.py -q
   ```

   Output:
   ```text
   ...........F.                                                            [100%]
   ================================== FAILURES ===================================
   ___________ test_measure_batch_ms_uses_run_inference_batch_wrapper ____________

   monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x0000024DD20EB3D0>

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

   >       durations_ms = measure_batch_ms(
               model=ModelWithoutDirectInference(),
               config=config,
               batch=batch,
               warmup=1,
               repeat=2,
               device=torch.device("cpu"),
           )

   tests\test_benchmark_common.py:245:
   _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

   model = <test_benchmark_common.test_measure_batch_ms_uses_run_inference_batch_wrapper.<locals>.ModelWithoutDirectInference object at 0x0000024DD214E0D0>
   config = InferenceRunConfig(mode='inference', model=ModelRunConfig(model_name='UPRNet', model_init_args={}, eval_convex_upsampl...d_filter=None, mode_filter=None, dedupe_endpoints=True, export_target=False, export_prediction=False), input_config={})
   batch = BenchmarkBatch(sample_names=['sample_0001'], img0=tensor([[[[1., 1., 1., 1.],
             [1., 1., 1., 1.]],

            [...2., 2.]],

            [[2., 2., 2., 2.],
             [2., 2., 2., 2.]]]]), embt=tensor([[[[0.2500]]]]), image_shape=(2, 4))
   warmup = 1, repeat = 2, device = device(type='cpu')

       def measure_batch_ms(
           model: torch.nn.Module,
           config: InferenceRunConfig,
           batch: BenchmarkBatch,
           warmup: int,
           repeat: int,
           device: torch.device,
       ) -> list[float]:
   >       with torch.inference_mode():
   E       AttributeError: module 'torch' has no attribute 'inference_mode'

   benchmarks\common.py:236: AttributeError
   =========================== short test summary info ===========================
   FAILED tests/test_benchmark_common.py::test_measure_batch_ms_uses_run_inference_batch_wrapper
   1 failed, 12 passed in 0.79s

   ERROR conda.cli.main_run:execute(125): `conda run python -m pytest tests/test_benchmark_common.py -q` failed. (See above for error)
   ```

4. Final green test pass:

   Command:
   ```powershell
   conda run -n ABME python -m pytest tests/test_benchmark_common.py -q
   ```

   Output:
   ```text
   .............                                                            [100%]
   13 passed in 0.58s
   ```

5. Compile verification:

   Command:
   ```powershell
   python -m compileall benchmarks tests
   ```

   Output:
   ```text
   Listing 'benchmarks'...
   Compiling 'benchmarks\\common.py'...
   Listing 'tests'...
   Compiling 'tests\\test_benchmark_common.py'...
   ```

### Concerns

- The requested `pytest` entrypoint is still unavailable in this worktree, and the local `.venv` also does not contain `pytest`, so the red/green test loop used `conda run -n ABME python -m pytest ...`.
- The available `ABME` environment uses a `torch` version without `torch.inference_mode()` and a Python version that cannot import `dict[str, Any]` at module load time from `src.engine.interpolation_batch`; the benchmark helper now handles that by using a lazy wrapper import and a compatible inference context.
