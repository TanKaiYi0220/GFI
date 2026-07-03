# Runtime Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor training and inference runtime code around typed run configuration objects and shared interpolation batch execution while preserving existing config keys, CLI behavior, model behavior, checkpoint behavior, and output schemas.

**Architecture:** Keep `scripts/train.py` and `scripts/inference.py` as entrypoints, but move reusable runtime seams into `src/engine/`. Use dataclasses for resolved config, a model registry for model-specific branching, focused dataset/checkpoint helpers, and one interpolation batch module that owns dataset tuple unpacking and model calls.

**Tech Stack:** Python 3, PyTorch, pandas, argparse, JSON-compatible `.yaml` config files, existing `src.engine.flow_approx`, existing `src.engine.evaluation`.

## Global Constraints

- Preserve existing YAML-compatible config files under `configs/run/*.yaml`; do not migrate them to a different file format in this refactor.
- Preserve existing YAML keys, CLI behavior, model behavior, checkpoint behavior, dry-run JSON shape, metrics CSV shape, and selected artifact columns.
- Avoid model architecture changes, loss behavior changes, and external model source changes.
- Use single-owner validation: config validation in `run_config.py`, model-name validation in `model_registry.py`, checkpoint shape validation in `checkpoints.py`, batch contract validation in `interpolation_batch.py`.
- Do not repeat the same validation through intermediate pipeline helpers after a seam returns a validated object.
- Keep metric accumulation in the scripts unless a tiny move is required by the selected seams.
- Do not add broad unit-test suites. Use import smoke checks, compile checks, and dry-run JSON comparisons.
- Make one commit per task.

---

## File Structure

- Create `src/engine/model_registry.py`
  - Own model names, flow-approx model classification, model class resolution, and convex upsampling toggling.
- Create `src/engine/run_config.py`
  - Own parsing and validation of raw config payloads into typed dataclasses.
  - Own dry-run summary builders for values that only depend on resolved run configuration.
- Create `src/engine/dataset_runs.py`
  - Own merged dataset CSV loading, valid-row filtering, dataset class selection, and train/inference dataset construction.
- Create `src/engine/checkpoints.py`
  - Own checkpoint save/load helpers and `TrainingState`.
- Create `src/engine/interpolation_batch.py`
  - Own one training or inference batch step for baseline, residual, and residual-flow-approx variants.
- Modify `scripts/train.py`
  - Keep CLI entrypoint, logger, train/evaluate loops, sample saving, and metric accumulation.
  - Replace moved helpers with imports and typed config access.
- Modify `scripts/inference.py`
  - Keep CLI entrypoint, metric accumulation, artifact saving, and CSV export.
  - Replace moved helpers with imports and typed config access.

## Baseline Capture

Run these commands before Task 1 and keep the files under `$env:TEMP` for comparison after Task 6 and Task 7.

- [ ] **Step 1: Capture train dry-run outputs**

```powershell
$baselineDir = Join-Path $env:TEMP "gfi-runtime-refactor-baseline"
New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
python scripts/train.py --config configs/run/train_baseline_0618.yaml --mode dry-run > (Join-Path $baselineDir "train_baseline_0618.before.json")
python scripts/train.py --config configs/run/train_residual_0618.yaml --mode dry-run > (Join-Path $baselineDir "train_residual_0618.before.json")
python scripts/train.py --config configs/run/train_flow_approx_0618_layer_1.yaml --mode dry-run > (Join-Path $baselineDir "train_flow_approx_0618_layer_1.before.json")
```

Expected: each command exits `0` and writes one JSON object.

- [ ] **Step 2: Build inference dry-run config copies**

```powershell
$baselineDir = Join-Path $env:TEMP "gfi-runtime-refactor-baseline"
python -c "import json, pathlib, sys; src=pathlib.Path(sys.argv[1]); dst=pathlib.Path(sys.argv[2]); data=json.loads(src.read_text(encoding='utf-8')); data['mode']='dry-run'; dst.write_text(json.dumps(data, indent=2), encoding='utf-8')" configs/run/inference_finetuning_0618.yaml (Join-Path $baselineDir "inference_finetuning_0618.dry-run.json")
python -c "import json, pathlib, sys; src=pathlib.Path(sys.argv[1]); dst=pathlib.Path(sys.argv[2]); data=json.loads(src.read_text(encoding='utf-8')); data['mode']='dry-run'; dst.write_text(json.dumps(data, indent=2), encoding='utf-8')" configs/run/inference_residual_layer_4_0618.yaml (Join-Path $baselineDir "inference_residual_layer_4_0618.dry-run.json")
python -c "import json, pathlib, sys; src=pathlib.Path(sys.argv[1]); dst=pathlib.Path(sys.argv[2]); data=json.loads(src.read_text(encoding='utf-8')); data['mode']='dry-run'; dst.write_text(json.dumps(data, indent=2), encoding='utf-8')" configs/run/inference_flowaprox_layer_1_0618.yaml (Join-Path $baselineDir "inference_flowaprox_layer_1_0618.dry-run.json")
```

Expected: each command exits `0` and writes a JSON-compatible config with `"mode": "dry-run"`.

- [ ] **Step 3: Capture inference dry-run outputs**

```powershell
$baselineDir = Join-Path $env:TEMP "gfi-runtime-refactor-baseline"
python scripts/inference.py --config (Join-Path $baselineDir "inference_finetuning_0618.dry-run.json") > (Join-Path $baselineDir "inference_finetuning_0618.before.json")
python scripts/inference.py --config (Join-Path $baselineDir "inference_residual_layer_4_0618.dry-run.json") > (Join-Path $baselineDir "inference_residual_layer_4_0618.before.json")
python scripts/inference.py --config (Join-Path $baselineDir "inference_flowaprox_layer_1_0618.dry-run.json") > (Join-Path $baselineDir "inference_flowaprox_layer_1_0618.before.json")
```

Expected: each command exits `0` and writes one JSON object.

---

### Task 1: Model Registry

**Files:**
- Create: `src/engine/model_registry.py`
- Modify: `scripts/train.py`
- Modify: `scripts/inference.py`

**Interfaces:**
- Consumes: existing model classes in `src.models.IFRNet` and `src.models.IFRNet_Residual`.
- Produces:
  - `BASELINE_MODEL_NAME: str`
  - `RESIDUAL_MODEL_NAME: str`
  - `RESIDUAL_FLOW_APPROX_MODEL_NAME: str`
  - `MODEL_NAMES: tuple[str, ...]`
  - `FLOW_APPROX_MODEL_NAMES: tuple[str, ...]`
  - `uses_flow_approx_model(model_name: str) -> bool`
  - `resolve_model_class(model_name: str) -> type[Any]`
  - `set_model_convex_upsampling(model: Any, enabled: bool, context: str) -> bool`

- [ ] **Step 1: Add the registry module**

```python
from __future__ import annotations

from typing import Any

BASELINE_MODEL_NAME: str = "IFRNet"
RESIDUAL_MODEL_NAME: str = "IFRNet_Residual"
RESIDUAL_FLOW_APPROX_MODEL_NAME: str = "IFRNet_Residual_FlowApprox"
MODEL_NAMES: tuple[str, ...] = (BASELINE_MODEL_NAME, RESIDUAL_MODEL_NAME, RESIDUAL_FLOW_APPROX_MODEL_NAME)
FLOW_APPROX_MODEL_NAMES: tuple[str, ...] = (RESIDUAL_FLOW_APPROX_MODEL_NAME,)


def uses_flow_approx_model(model_name: str) -> bool:
    return model_name in FLOW_APPROX_MODEL_NAMES


def resolve_model_class(model_name: str) -> type[Any]:
    if model_name == BASELINE_MODEL_NAME:
        from src.models.IFRNet import Model as IFRNetModel

        return IFRNetModel
    if model_name == RESIDUAL_MODEL_NAME:
        from src.models.IFRNet_Residual import Model as IFRNetResidualModel

        return IFRNetResidualModel
    if model_name == RESIDUAL_FLOW_APPROX_MODEL_NAME:
        from src.models.IFRNet_Residual import Model as IFRNetResidualModel

        return IFRNetResidualModel

    available_models = ", ".join(MODEL_NAMES)
    raise KeyError(f"Unknown model '{model_name}'. Available models: {available_models}")


def set_model_convex_upsampling(model: Any, enabled: bool, context: str) -> bool:
    setter = getattr(model, "set_convex_upsampling", None)
    if not callable(setter):
        raise TypeError(
            f"{context} requested eval_convex_upsampling={enabled}, "
            f"but model type {type(model).__name__} does not support it."
        )
    previous_value = bool(getattr(model, "convex_upsampling"))
    setter(enabled)
    return previous_value
```

- [ ] **Step 2: Replace script-local model registry symbols**

In `scripts/train.py`, import the produced symbols from `src.engine.model_registry`, remove local definitions of `MODEL_NAMES`, `FLOW_APPROX_MODEL_NAMES`, `uses_flow_approx_model`, `resolve_model_class`, and `set_model_convex_upsampling`, and keep all call sites unchanged.

In `scripts/inference.py`, import model names, `resolve_model_class`, and `set_model_convex_upsampling` from `src.engine.model_registry`; remove script-local model-name constants.

- [ ] **Step 3: Verify imports and model resolution**

```powershell
python -c "from src.engine.model_registry import MODEL_NAMES, resolve_model_class; print(MODEL_NAMES); print(resolve_model_class('IFRNet').__name__)"
```

Expected output contains:

```text
('IFRNet', 'IFRNet_Residual', 'IFRNet_Residual_FlowApprox')
Model
```

- [ ] **Step 4: Verify compile**

```powershell
python -m compileall src scripts tests
```

Expected: command exits `0`.

- [ ] **Step 5: Commit**

```powershell
git add src/engine/model_registry.py scripts/train.py scripts/inference.py
git commit -m "refactor: move model registry helpers"
```

---

### Task 2: Typed Run Configuration

**Files:**
- Create: `src/engine/run_config.py`
- Modify: `scripts/train.py`
- Modify: `scripts/inference.py`

**Interfaces:**
- Consumes:
  - `src.utils.config.load_yaml_file(config_path: Path) -> dict[str, Any]`
  - `src.engine.evaluation.read_metric_config`
  - metric requirement helpers from `src.engine.evaluation`
  - flow approximation constants and helpers from `src.engine.flow_approx`
  - model constants and classifiers from `src.engine.model_registry`
- Produces:
  - `ModelRunConfig`
  - `FlowApproxConfig`
  - `MetricRunConfig`
  - `TrainRunConfig`
  - `InferenceRunConfig`
  - `parse_bool_value(value: Any, key: str) -> bool`
  - `parse_eval_convex_upsampling_arg(value: str) -> bool`
  - `read_model_init_args(config_values: dict[str, Any]) -> dict[str, Any]`
  - `read_optional_bool(config_values: dict[str, Any], key: str) -> bool | None`
  - `build_train_run_config(args: argparse.Namespace, config_defaults: dict[str, Any]) -> TrainRunConfig`
  - `build_inference_run_config(config_path: Path, project_root: Path) -> InferenceRunConfig`
  - `build_train_dry_run_summary(config: TrainRunConfig) -> dict[str, object]`
  - `build_inference_dry_run_summary(config: InferenceRunConfig) -> dict[str, object]`

- [ ] **Step 1: Add dataclasses and parsing helpers**

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.dataset_config import ACTIVE_DATASET_ROOT_KEY
from src.engine.evaluation import read_metric_config
from src.engine.evaluation import require_psnr_div_disabled
from src.engine.evaluation import require_psnr_enabled
from src.engine.evaluation import require_vfips_disabled
from src.engine.flow_approx import DEFAULT_SPLATTING_FILL_STRATEGY
from src.engine.flow_approx import FLOW_APPROX_METHOD_CHOICES
from src.engine.flow_approx import SPLATTING_FILL_STRATEGIES
from src.engine.flow_approx import is_splatting_flow_approx_method
from src.engine.flow_approx import resolve_splatting_fill_strategy
from src.engine.model_registry import MODEL_NAMES
from src.engine.model_registry import uses_flow_approx_model
from src.utils.config import load_yaml_file

DEFAULT_INIT_FLOW_DOWNSCALE_STRATEGY: str = "bilinear"
DEFAULT_INIT_FLOW_MASK_EPSILON: float = 1e-6
INIT_FLOW_DOWNSCALE_STRATEGIES: tuple[str, ...] = ("bilinear", "masked_area")


@dataclass(frozen=True)
class ModelRunConfig:
    model_name: str
    model_init_args: dict[str, Any]
    eval_convex_upsampling: bool | None


@dataclass(frozen=True)
class FlowApproxConfig:
    method: str
    splatting_fill_strategy: str
    effective_splatting_fill_strategy: str
    init_flow_downscale_strategy: str
    effective_init_flow_downscale_strategy: str
    init_flow_mask_epsilon: float


@dataclass(frozen=True)
class MetricRunConfig:
    values: dict[str, object]


@dataclass(frozen=True)
class TrainRunConfig:
    mode: str
    model: ModelRunConfig
    flow_approx: FlowApproxConfig
    metrics: MetricRunConfig
    root_dir: Path
    dataset_root_dir: str
    paths_config: str | None
    train_preset: str
    test_preset: str
    epochs: int
    resume_path: str | None
    pretrained_checkpoint_path: str | None
    eval_interval: int
    lr_start: float
    lr_end: float
    seed: int
    batch_size: int
    output_dir: str
    only_fps: int
    input_fps: int
    sample_train_frames: list[str]
    sample_test_frames: list[str]
    sample_interval_epoch: int
    input_config: dict[str, Any]


@dataclass(frozen=True)
class InferenceRunConfig:
    mode: str
    model: ModelRunConfig
    flow_approx: FlowApproxConfig
    metrics: MetricRunConfig
    inference_presets: list[str]
    root_dir: Path
    dataset_root_dir: Path
    checkpoint_path: Path
    output_dir: Path
    seed: int
    batch_size: int
    only_fps: int
    input_fps: int
    scale_factor: float
    flow_diff_threshold: float
    flow_diff_percentile: float
    save_topk_worst_psnr: int
    save_topk_best_psnr: int
    save_topk_largest_flow_diff: int
    input_config: dict[str, Any]


def parse_bool_value(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized_value = value.strip().lower()
        if normalized_value in ("true", "1", "yes", "on"):
            return True
        if normalized_value in ("false", "0", "no", "off"):
            return False
    raise TypeError(f"{key} must be a boolean, got {value!r}")
```

- [ ] **Step 2: Add single-owner flow/config validation**

Implement `build_flow_approx_config(model_name: str, config_values: dict[str, Any]) -> FlowApproxConfig` in `run_config.py`.

Rules:
- `method` defaults to `"combination"`.
- `splatting_fill_strategy` defaults to `DEFAULT_SPLATTING_FILL_STRATEGY`.
- `init_flow_downscale_strategy` defaults to `DEFAULT_INIT_FLOW_DOWNSCALE_STRATEGY`.
- `init_flow_mask_epsilon` defaults to `DEFAULT_INIT_FLOW_MASK_EPSILON`.
- Reject unknown model names by checking `model_name in MODEL_NAMES`.
- Reject flow approx methods not in `FLOW_APPROX_METHOD_CHOICES`.
- Reject fill strategies not in `SPLATTING_FILL_STRATEGIES`.
- Reject init-flow downscale strategies not in `INIT_FLOW_DOWNSCALE_STRATEGIES`.
- Reject non-positive `init_flow_mask_epsilon`.
- Reject `masked_area` unless the model is `IFRNet_Residual_FlowApprox` and the method is splatting-based.
- For non-flow-approx models, set both effective strategy strings to `""`.
- For flow-approx non-splatting methods, set both effective strategy strings to `""`.

- [ ] **Step 3: Add train config builder**

`build_train_run_config` should accept the parsed `argparse.Namespace` and the raw config payload, enforce metric requirements for training, and return `TrainRunConfig`. Keep CLI defaults in `build_train_arg_parser`; this task moves resolved values after argparse, not argparse itself.

`build_train_dry_run_summary` must preserve the current keys and conditional keys produced by `scripts/train.py::build_dry_run_summary`.

- [ ] **Step 4: Add inference config builder**

`build_inference_run_config` should load the config file, resolve relative paths against `project_root`, enforce metric requirements for inference, and return `InferenceRunConfig`.

`build_inference_dry_run_summary` must preserve the current keys and conditional keys produced by `scripts/inference.py::main` before it returns in dry-run mode.

- [ ] **Step 5: Wire train parsing to typed config without changing the training loop**

In `scripts/train.py`, keep `parse_train_args(argv)` returning `argparse.Namespace` for this task. Add:

```python
def parse_train_config(argv: list[str] | None) -> TrainRunConfig:
    args = parse_train_args(argv)
    return build_train_run_config(args=args, config_defaults=args.input_config)
```

Use `build_train_dry_run_summary(config=run_config)` in `main` for dry-run. Training can still call `run_training(args)` until Task 6.

- [ ] **Step 6: Wire inference dry-run to typed config without changing the inference loop**

In `scripts/inference.py`, build `InferenceRunConfig` at the top of `main`. For `mode == "dry-run"`, print `build_inference_dry_run_summary(config=run_config)` and return. Keep the existing non-dry-run local variables for this task if that keeps the diff small.

- [ ] **Step 7: Verify imports and dry-run summaries**

```powershell
python -c "from src.engine.run_config import TrainRunConfig, InferenceRunConfig, build_flow_approx_config; print(TrainRunConfig.__name__, InferenceRunConfig.__name__, build_flow_approx_config('IFRNet', {}).method)"
python scripts/train.py --config configs/run/train_baseline_0618.yaml --mode dry-run
$baselineDir = Join-Path $env:TEMP "gfi-runtime-refactor-baseline"
python scripts/inference.py --config (Join-Path $baselineDir "inference_finetuning_0618.dry-run.json")
python -m compileall src scripts tests
```

Expected: import command prints `TrainRunConfig InferenceRunConfig combination`; dry-runs print JSON; compile exits `0`.

- [ ] **Step 8: Commit**

```powershell
git add src/engine/run_config.py scripts/train.py scripts/inference.py
git commit -m "refactor: add typed run config seam"
```

---

### Task 3: Dataset Run Helpers

**Files:**
- Create: `src/engine/dataset_runs.py`
- Modify: `scripts/train.py`
- Modify: `scripts/inference.py`

**Interfaces:**
- Consumes:
  - dataset presets from `src.data.dataset_config`
  - `VFITrainDataset` and `FlowEstimationTrainDataset` from `src.data.dataset_loader`
  - `uses_flow_approx_model(model_name: str) -> bool`
  - `is_splatting_flow_approx_method(flow_approx_method: str) -> bool`
- Produces:
  - `build_merged_dataframe(root_dir: Path, checkpoints_dir: Path, dataset_preset_name: str, only_fps: int, logger: logging.Logger) -> Any`
  - `filter_valid_dataframe(dataframe: Any) -> Any`
  - `build_training_dataset(dataframe: Any, dataset_root_dir: str, augment: bool, input_fps: int, model_name: str, flow_approx_method: str) -> Any`
  - `build_inference_dataset(dataframe: Any, dataset_root_dir: Path, input_fps: int, model_name: str, flow_approx_method: str) -> Any`
  - `resolve_dataset_class_name(model_name: str) -> str`

- [ ] **Step 1: Add dataset helper module**

Move the current `build_merged_dataframe`, `build_training_dataset`, and `resolve_dataset_class_name` bodies from `scripts/train.py` into `src/engine/dataset_runs.py`.

Add:

```python
def filter_valid_dataframe(dataframe: Any) -> Any:
    if "valid" not in dataframe.columns:
        return dataframe
    return dataframe[dataframe["valid"] == True].reset_index(drop=True)


def build_inference_dataset(
    dataframe: Any,
    dataset_root_dir: Path,
    input_fps: int,
    model_name: str,
    flow_approx_method: str,
) -> Any:
    if uses_flow_approx_model(model_name):
        from src.data.dataset_loader import FlowEstimationTrainDataset

        include_source_depths = is_splatting_flow_approx_method(flow_approx_method=flow_approx_method)
        return FlowEstimationTrainDataset(
            dataframe,
            str(dataset_root_dir),
            input_fps,
            False,
            include_source_depths,
        )

    from src.data.dataset_loader import VFITrainDataset

    return VFITrainDataset(dataframe, str(dataset_root_dir), False, input_fps)
```

- [ ] **Step 2: Replace train imports**

In `scripts/train.py`, import `build_merged_dataframe`, `build_training_dataset`, and `resolve_dataset_class_name` from `src.engine.dataset_runs`; remove local copies.

- [ ] **Step 3: Replace inference imports**

In `scripts/inference.py`, import `build_merged_dataframe`, `build_inference_dataset`, and `filter_valid_dataframe` from `src.engine.dataset_runs`; remove the `from scripts.train import build_merged_dataframe` import.

Replace the inference dataset branch inside the grouped loop with:

```python
dataset = build_inference_dataset(
    dataframe=group_dataframe,
    dataset_root_dir=run_config.dataset_root_dir,
    input_fps=run_config.input_fps,
    model_name=run_config.model.model_name,
    flow_approx_method=run_config.flow_approx.method,
)
```

- [ ] **Step 4: Verify compile and import**

```powershell
python -c "from src.engine.dataset_runs import resolve_dataset_class_name; print(resolve_dataset_class_name('IFRNet_Residual_FlowApprox'))"
python -m compileall src scripts tests
```

Expected output contains `FlowEstimationTrainDataset`, and compile exits `0`.

- [ ] **Step 5: Commit**

```powershell
git add src/engine/dataset_runs.py scripts/train.py scripts/inference.py
git commit -m "refactor: move dataset run helpers"
```

---

### Task 4: Checkpoint Helpers

**Files:**
- Create: `src/engine/checkpoints.py`
- Modify: `scripts/train.py`
- Modify: `scripts/inference.py`

**Interfaces:**
- Consumes: model, optimizer, torch module, device, logger.
- Produces:
  - `TrainingState`
  - `save_checkpoint(checkpoint_path: Path, model: Any, optimizer: Any, epoch: int, best_psnr: float) -> None`
  - `is_raw_model_state_dict(checkpoint: Any, torch_module: Any) -> bool`
  - `extract_pretrained_state_dict(checkpoint: Any, checkpoint_path: Path, torch_module: Any) -> Any`
  - `load_training_state(resume_path: str | None, pretrained_checkpoint_path: str | None, model: Any, optimizer: Any, device: Any, logger: logging.Logger, iters_per_epoch: int, model_name: str) -> TrainingState`
  - `load_inference_state_dict(checkpoint_path: Path, device: Any) -> Any`

- [ ] **Step 1: Add checkpoint module**

Move `TrainingState`, `save_checkpoint`, `is_raw_model_state_dict`, `extract_pretrained_state_dict`, and the body of `load_training_state` into `src/engine/checkpoints.py`.

Change `load_training_state` so it accepts explicit values instead of `argparse.Namespace`:

```python
def load_training_state(
    resume_path: str | None,
    pretrained_checkpoint_path: str | None,
    model: Any,
    optimizer: Any,
    device: Any,
    logger: logging.Logger,
    iters_per_epoch: int,
    model_name: str,
) -> TrainingState:
    # Move the current resume/pretrained/scratch branches from scripts/train.py.
    # Use resume_path, pretrained_checkpoint_path, iters_per_epoch, and model_name
    # instead of reading from argparse.Namespace.
```

Add:

```python
def load_inference_state_dict(checkpoint_path: Path, device: Any) -> Any:
    import torch

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if is_raw_model_state_dict(checkpoint=checkpoint, torch_module=torch):
        return checkpoint

    available_keys = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    raise KeyError(
        "checkpoint_path must point to either a raw model state_dict or a full training checkpoint "
        f"containing a 'model' key: path={checkpoint_path}, keys={available_keys}"
    )
```

- [ ] **Step 2: Replace train checkpoint helpers**

In `scripts/train.py`, import `TrainingState`, `save_checkpoint`, and `load_training_state` from `src.engine.checkpoints`. Remove local copies.

Update the call site:

```python
training_state = load_training_state(
    resume_path=config.resume_path,
    pretrained_checkpoint_path=config.pretrained_checkpoint_path,
    model=model,
    optimizer=optimizer,
    device=device,
    logger=logger,
    iters_per_epoch=len(train_loader),
    model_name=config.model.model_name,
)
```

If Task 6 has not migrated `run_training` to `TrainRunConfig` yet, use `args.resume_path`, `args.pretrained_checkpoint_path`, `args.model_name`, and `args.iters_per_epoch` for this task.

- [ ] **Step 3: Replace inference checkpoint loading**

In `scripts/inference.py`, import `load_inference_state_dict` and replace:

```python
checkpoint = torch.load(str(checkpoint_path), map_location=device)
state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
model.load_state_dict(state_dict)
```

with:

```python
model.load_state_dict(load_inference_state_dict(checkpoint_path=checkpoint_path, device=device))
```

- [ ] **Step 4: Verify compile and import**

```powershell
python -c "from src.engine.checkpoints import TrainingState, load_inference_state_dict; print(TrainingState.__name__, callable(load_inference_state_dict))"
python -m compileall src scripts tests
```

Expected output contains `TrainingState True`, and compile exits `0`.

- [ ] **Step 5: Commit**

```powershell
git add src/engine/checkpoints.py scripts/train.py scripts/inference.py
git commit -m "refactor: move checkpoint helpers"
```

---

### Task 5: Interpolation Batch Execution

**Files:**
- Create: `src/engine/interpolation_batch.py`

**Interfaces:**
- Consumes:
  - `TrainRunConfig`, `InferenceRunConfig`, and `FlowApproxConfig`
  - model registry constants and classifiers
  - `build_flow_init_result_with_fill_strategy`
  - `make_source_grid` and `flatten_target_index`
- Produces:
  - `InterpolationBatchResult`
  - `run_training_batch(config: TrainRunConfig, model: Any, batch: Any, device: Any) -> InterpolationBatchResult`
  - `run_inference_batch(config: InferenceRunConfig, model: Any, batch: Any, device: Any) -> InterpolationBatchResult`

- [ ] **Step 1: Add result dataclass and shared tensor helpers**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.engine.flow_approx import build_flow_init_result_with_fill_strategy
from src.engine.flow_approx import flatten_target_index
from src.engine.flow_approx import is_splatting_flow_approx_method
from src.engine.flow_approx import make_source_grid
from src.engine.model_registry import BASELINE_MODEL_NAME
from src.engine.model_registry import RESIDUAL_FLOW_APPROX_MODEL_NAME
from src.engine.model_registry import RESIDUAL_MODEL_NAME
from src.engine.model_registry import uses_flow_approx_model
from src.engine.run_config import FlowApproxConfig
from src.engine.run_config import InferenceRunConfig
from src.engine.run_config import TrainRunConfig

SplattingRegionMaps = dict[str, Any]


@dataclass(frozen=True)
class InterpolationBatchResult:
    img0: Any
    img1: Any
    imgt: Any
    imgt_pred: Any
    embt: Any
    info: dict[str, Any] | None
    bmv: Any | None
    fmv: Any | None
    init_bmv: Any | None
    init_fmv: Any | None
    init_masks: Any | None
    up_flow0_1: Any | None
    up_flow1_1: Any | None
    up_mask_1: Any | None
    imgt_merge: Any | None
    loss_rec: Any | None
    loss_geo: Any | None
    loss_dis: Any | None
    splatting_region_maps: SplattingRegionMaps | None
```

- [ ] **Step 2: Move flow-init and model-call logic**

Move `forward_model` from `scripts/train.py` into `interpolation_batch.py` as a private helper:

```python
def run_model_forward(
    model_name: str,
    model: Any,
    img0: Any,
    img1: Any,
    embt: Any,
    imgt: Any,
    source_bmv: Any,
    source_fmv: Any,
    flow_approx: FlowApproxConfig,
    source_depth0: Any | None,
    source_depth1: Any | None,
    ground_truth_bmv: Any | None,
    ground_truth_fmv: Any | None,
) -> tuple[Any, Any | None, Any | None]:
    # Move the current scripts/train.py forward_model body here.
    # Return the raw model output plus the init_bmv/init_fmv tensors used by the call.
```

Return `(model_output, init_bmv, init_fmv)` and keep the current model-call behavior unchanged.

- [ ] **Step 3: Move splatting region-map construction**

Move `build_nearest_splat_hit_count` and `build_splatting_region_maps` from `scripts/inference.py` into `interpolation_batch.py`. Keep visualization/colorization helpers in `scripts/inference.py`.

- [ ] **Step 4: Implement `run_training_batch`**

Use the existing tuple unpacking from `scripts/train.py::run_training_batch`. Return `InterpolationBatchResult` with training losses populated and inference-only fields set to `None`.

Unsupported tuple/model combinations should raise:

```python
raise ValueError(f"Unsupported training batch contract for model_name={config.model.model_name}")
```

- [ ] **Step 5: Implement `run_inference_batch`**

Use the existing model branches from `scripts/inference.py::run_inference_batch_with_fill_strategy`. Return `InterpolationBatchResult` for all three model variants.

Unsupported model names should raise:

```python
raise ValueError(f"Unsupported model_name: {config.model.model_name}")
```

- [ ] **Step 6: Verify compile and import**

```powershell
python -c "from src.engine.interpolation_batch import InterpolationBatchResult; print(InterpolationBatchResult.__name__)"
python -m compileall src scripts tests
```

Expected output contains `InterpolationBatchResult`, and compile exits `0`.

- [ ] **Step 7: Commit**

```powershell
git add src/engine/interpolation_batch.py
git commit -m "refactor: add shared interpolation batch execution"
```

---

### Task 6: Thin Training Runtime Around Engine Modules

**Files:**
- Modify: `scripts/train.py`
- Modify: `src/engine/run_config.py`
- Modify: `src/engine/interpolation_batch.py`
- Modify: `src/engine/checkpoints.py`
- Modify: `src/engine/dataset_runs.py`

**Interfaces:**
- Consumes all modules created in Tasks 1-5.
- Produces `scripts/train.py` using `TrainRunConfig` through the runtime path instead of ad hoc config values on `argparse.Namespace`.

- [ ] **Step 1: Change train runtime signatures**

Change signatures that currently take `args: argparse.Namespace` to take `config: TrainRunConfig`:

```python
def evaluate(
    config: TrainRunConfig,
    model: Any,
    loader: Any,
    device: Any,
    lpips_model: Any | None,
    flolpips_model: Any | None,
) -> tuple[float, Any, Any, dict[str, float]]:
    # Existing evaluate body with args.* reads replaced by config.* reads.


def train(
    config: TrainRunConfig,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    logger: logging.Logger,
    training_state: TrainingState,
    sample_dataframes: dict[str, Any],
    lpips_model: Any | None,
    flolpips_model: Any | None,
) -> None:
    # Existing train body with args.* reads replaced by config.* reads.


def run_training(config: TrainRunConfig) -> None:
    # Existing run_training body with args.* reads replaced by config.* reads.
```

- [ ] **Step 2: Replace namespace field access**

Use these field replacements:

```text
args.model_name -> config.model.model_name
args.model_init_args -> config.model.model_init_args
args.eval_convex_upsampling -> config.model.eval_convex_upsampling
args.metric_config -> config.metrics.values
args.flow_approx_method -> config.flow_approx.method
args.splatting_fill_strategy -> config.flow_approx.splatting_fill_strategy
args.init_flow_downscale_strategy -> config.flow_approx.init_flow_downscale_strategy
args.init_flow_mask_epsilon -> config.flow_approx.init_flow_mask_epsilon
args.root_dir -> config.root_dir
args.dataset_root_dir -> config.dataset_root_dir
args.output_dir -> config.output_dir
```

Keep scalar training settings as direct config fields, for example `args.epochs -> config.epochs`.

- [ ] **Step 3: Move dry-run summary ownership**

Remove `build_dry_run_summary` from `scripts/train.py`. `main` should be:

```python
def main(argv: list[str] | None = None) -> None:
    config = parse_train_config(argv)

    if config.mode == "dry-run":
        print(json.dumps(build_train_dry_run_summary(config=config), indent=2))
        return

    run_training(config=config)
```

- [ ] **Step 4: Preserve sample saving**

Update `save_epoch_samples` to accept `config: TrainRunConfig`; keep its output artifact behavior unchanged. It should call `build_training_dataset` and `run_training_batch` with the typed config.

- [ ] **Step 5: Capture after dry-run outputs**

```powershell
$baselineDir = Join-Path $env:TEMP "gfi-runtime-refactor-baseline"
python scripts/train.py --config configs/run/train_baseline_0618.yaml --mode dry-run > (Join-Path $baselineDir "train_baseline_0618.after.json")
python scripts/train.py --config configs/run/train_residual_0618.yaml --mode dry-run > (Join-Path $baselineDir "train_residual_0618.after.json")
python scripts/train.py --config configs/run/train_flow_approx_0618_layer_1.yaml --mode dry-run > (Join-Path $baselineDir "train_flow_approx_0618_layer_1.after.json")
```

Expected: each command exits `0`.

- [ ] **Step 6: Compare train dry-run JSON**

```powershell
$baselineDir = Join-Path $env:TEMP "gfi-runtime-refactor-baseline"
python -c "import json, pathlib, sys; before=json.loads(pathlib.Path(sys.argv[1]).read_text()); after=json.loads(pathlib.Path(sys.argv[2]).read_text()); assert before == after, {'before': before, 'after': after}" (Join-Path $baselineDir "train_baseline_0618.before.json") (Join-Path $baselineDir "train_baseline_0618.after.json")
python -c "import json, pathlib, sys; before=json.loads(pathlib.Path(sys.argv[1]).read_text()); after=json.loads(pathlib.Path(sys.argv[2]).read_text()); assert before == after, {'before': before, 'after': after}" (Join-Path $baselineDir "train_residual_0618.before.json") (Join-Path $baselineDir "train_residual_0618.after.json")
python -c "import json, pathlib, sys; before=json.loads(pathlib.Path(sys.argv[1]).read_text()); after=json.loads(pathlib.Path(sys.argv[2]).read_text()); assert before == after, {'before': before, 'after': after}" (Join-Path $baselineDir "train_flow_approx_0618_layer_1.before.json") (Join-Path $baselineDir "train_flow_approx_0618_layer_1.after.json")
```

Expected: each command exits `0`.

- [ ] **Step 7: Verify compile**

```powershell
python -m compileall src scripts tests
```

Expected: command exits `0`.

- [ ] **Step 8: Commit**

```powershell
git add scripts/train.py src/engine/run_config.py src/engine/interpolation_batch.py src/engine/checkpoints.py src/engine/dataset_runs.py
git commit -m "refactor: thin training runtime"
```

---

### Task 7: Thin Inference Runtime Around Engine Modules

**Files:**
- Modify: `scripts/inference.py`
- Modify: `src/engine/run_config.py`
- Modify: `src/engine/interpolation_batch.py`
- Modify: `src/engine/dataset_runs.py`
- Modify: `src/engine/checkpoints.py`

**Interfaces:**
- Consumes all modules created in Tasks 1-5.
- Produces `scripts/inference.py` using `InferenceRunConfig`, `build_inference_dataset`, `load_inference_state_dict`, and `run_inference_batch` through the non-dry-run path.

- [ ] **Step 1: Use `InferenceRunConfig` in `main`**

At the top of `main`, replace local raw config extraction with:

```python
args = parse_args(argv)
config_path = Path(args.config)
if not config_path.is_absolute():
    config_path = PROJECT_ROOT / config_path

run_config = build_inference_run_config(config_path=config_path, project_root=PROJECT_ROOT)
if run_config.mode == "dry-run":
    print(json.dumps(build_inference_dry_run_summary(config=run_config), indent=2))
    return
```

- [ ] **Step 2: Replace local inference variables**

Use these field replacements:

```text
model_name -> run_config.model.model_name
model_init_args -> run_config.model.model_init_args
eval_convex_upsampling -> run_config.model.eval_convex_upsampling
metric_config -> run_config.metrics.values
flow_approx_method -> run_config.flow_approx.method
splatting_fill_strategy -> run_config.flow_approx.splatting_fill_strategy
init_flow_downscale_strategy -> run_config.flow_approx.init_flow_downscale_strategy
init_flow_mask_epsilon -> run_config.flow_approx.init_flow_mask_epsilon
inference_presets -> run_config.inference_presets
root_dir -> run_config.root_dir
dataset_root_dir -> run_config.dataset_root_dir
checkpoint_path -> run_config.checkpoint_path
output_dir -> run_config.output_dir
scale_factor -> run_config.scale_factor
```

- [ ] **Step 3: Replace dataset construction**

Inside the grouped inference loop, replace inline dataset branching with `build_inference_dataset`.

- [ ] **Step 4: Replace inference batch result dictionary access**

Use `InterpolationBatchResult` attributes in metric calculations, VFIPS segment construction, flow-diff calculations, and selected artifact saving.

`save_selected_sample_artifacts` should start like this:

```python
def save_selected_sample_artifacts(
    cv2: Any,
    flow_diff_percentile: float,
    flow_diff_threshold: float,
    flow_to_image: Any,
    inference_result: InterpolationBatchResult,
    np: Any,
    save_dir: Path,
    save_image: Any,
) -> dict[str, str]:
    img0 = inference_result.img0
    img1 = inference_result.img1
    imgt = inference_result.imgt
    bmv = inference_result.bmv
    fmv = inference_result.fmv
    imgt_pred = inference_result.imgt_pred
    imgt_merge = inference_result.imgt_merge
    init_bmv = inference_result.init_bmv
    init_fmv = inference_result.init_fmv
    up_flow0_1 = inference_result.up_flow0_1
    up_flow1_1 = inference_result.up_flow1_1
    up_mask_1 = inference_result.up_mask_1
    splatting_region_maps = inference_result.splatting_region_maps
```

- [ ] **Step 5: Capture after inference dry-run outputs**

```powershell
$baselineDir = Join-Path $env:TEMP "gfi-runtime-refactor-baseline"
python scripts/inference.py --config (Join-Path $baselineDir "inference_finetuning_0618.dry-run.json") > (Join-Path $baselineDir "inference_finetuning_0618.after.json")
python scripts/inference.py --config (Join-Path $baselineDir "inference_residual_layer_4_0618.dry-run.json") > (Join-Path $baselineDir "inference_residual_layer_4_0618.after.json")
python scripts/inference.py --config (Join-Path $baselineDir "inference_flowaprox_layer_1_0618.dry-run.json") > (Join-Path $baselineDir "inference_flowaprox_layer_1_0618.after.json")
```

Expected: each command exits `0`.

- [ ] **Step 6: Compare inference dry-run JSON**

```powershell
$baselineDir = Join-Path $env:TEMP "gfi-runtime-refactor-baseline"
python -c "import json, pathlib, sys; before=json.loads(pathlib.Path(sys.argv[1]).read_text()); after=json.loads(pathlib.Path(sys.argv[2]).read_text()); assert before == after, {'before': before, 'after': after}" (Join-Path $baselineDir "inference_finetuning_0618.before.json") (Join-Path $baselineDir "inference_finetuning_0618.after.json")
python -c "import json, pathlib, sys; before=json.loads(pathlib.Path(sys.argv[1]).read_text()); after=json.loads(pathlib.Path(sys.argv[2]).read_text()); assert before == after, {'before': before, 'after': after}" (Join-Path $baselineDir "inference_residual_layer_4_0618.before.json") (Join-Path $baselineDir "inference_residual_layer_4_0618.after.json")
python -c "import json, pathlib, sys; before=json.loads(pathlib.Path(sys.argv[1]).read_text()); after=json.loads(pathlib.Path(sys.argv[2]).read_text()); assert before == after, {'before': before, 'after': after}" (Join-Path $baselineDir "inference_flowaprox_layer_1_0618.before.json") (Join-Path $baselineDir "inference_flowaprox_layer_1_0618.after.json")
```

Expected: each comparison exits `0`.

- [ ] **Step 7: Verify compile**

```powershell
python -m compileall src scripts tests
```

Expected: command exits `0`.

- [ ] **Step 8: Commit**

```powershell
git add scripts/inference.py src/engine/run_config.py src/engine/interpolation_batch.py src/engine/dataset_runs.py src/engine/checkpoints.py
git commit -m "refactor: thin inference runtime"
```

---

### Task 8: Final Verification And Review

**Files:**
- Modify only if a previous task needs a small correction found by verification.

**Interfaces:**
- Consumes complete refactor from Tasks 1-7.
- Produces verified branch ready for code review.

- [ ] **Step 1: Run compile**

```powershell
python -m compileall src scripts tests
```

Expected: command exits `0`.

- [ ] **Step 2: Run repository unittest discovery**

```powershell
python -m unittest discover -s tests -v
```

Expected for the current repository: exits `1` with `NO TESTS RAN`. This is a known baseline, not a refactor failure.

- [ ] **Step 3: Run representative train dry-runs**

```powershell
python scripts/train.py --config configs/run/train_baseline_0618.yaml --mode dry-run
python scripts/train.py --config configs/run/train_residual_0618.yaml --mode dry-run
python scripts/train.py --config configs/run/train_flow_approx_0618_layer_1.yaml --mode dry-run
```

Expected: each command exits `0` and prints JSON.

- [ ] **Step 4: Run representative inference dry-runs**

```powershell
$baselineDir = Join-Path $env:TEMP "gfi-runtime-refactor-baseline"
python scripts/inference.py --config (Join-Path $baselineDir "inference_finetuning_0618.dry-run.json")
python scripts/inference.py --config (Join-Path $baselineDir "inference_residual_layer_4_0618.dry-run.json")
python scripts/inference.py --config (Join-Path $baselineDir "inference_flowaprox_layer_1_0618.dry-run.json")
```

Expected: each command exits `0` and prints JSON.

- [ ] **Step 5: Inspect diff size and moved helpers**

```powershell
git --no-pager diff --stat HEAD~7..HEAD
git --no-pager diff --check
rg -n "from scripts.train import|def forward_model|def run_inference_batch_with_fill_strategy|MODEL_NAMES: tuple|FLOW_APPROX_MODEL_NAMES" scripts src
```

Expected:
- `git --no-pager diff --check` exits `0`.
- `rg` should not find `from scripts.train import` in `scripts/inference.py`.
- `rg` should not find script-local `MODEL_NAMES` or `FLOW_APPROX_MODEL_NAMES`.
- Any remaining `run_inference_batch_with_fill_strategy` match should be in history only, not in the working tree.

- [ ] **Step 6: Commit verification notes if code changed during verification**

If verification required a correction:

```powershell
git add scripts src
git commit -m "fix: align runtime refactor verification"
```

If no correction was needed, do not create an empty commit.
