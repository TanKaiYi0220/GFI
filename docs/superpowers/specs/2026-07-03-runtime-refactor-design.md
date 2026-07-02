# Runtime Refactor Design

## Summary

Refactor the training and inference runtime around two deep modules:

- `src/engine/run_config.py`: owns run configuration parsing, defaults, and validation.
- `src/engine/interpolation_batch.py`: owns one interpolation batch execution across train, inference, and later video export.

The refactor must preserve existing YAML keys, CLI behavior, model behavior, checkpoint behavior, and output schemas. It may move larger helper clusters out of scripts when the move supports these two modules and reduces script size immediately.

## Goals

- Keep `scripts/train.py` and `scripts/inference.py` as entrypoints.
- Reduce script size by moving reusable runtime behavior into `src/engine/`.
- Concentrate model variant, flow approximation, init-flow mask, and batch tuple knowledge at one seam.
- Concentrate config defaults and validation at one seam.
- Preserve current run configs under `configs/run/*.yaml`.
- Avoid model architecture changes.

## Non-Goals

- No YAML schema redesign.
- No model layer or loss behavior changes.
- No broad metric accumulation refactor in this slice unless required by the two selected seams.
- No GPU training or full dataset run requirement for this refactor.
- No changes to external model source code.

## Architecture

### Run Configuration Module

Add `src/engine/run_config.py`.

This module should expose typed dataclasses for resolved run settings:

- `ModelRunConfig`
- `FlowApproxConfig`
- `MetricRunConfig`
- `TrainRunConfig`
- `InferenceRunConfig`

It should absorb repeated config helpers currently living in scripts:

- optional boolean parsing
- `model_init_args` parsing
- init-flow downscale defaults and validation
- flow approximation defaults and validation
- train and inference dry-run summary helpers when they depend only on resolved run configuration

Scripts may still use `argparse`, but they should convert raw config values into typed run configuration objects before entering the runtime pipeline.

### Interpolation Batch Module

Add `src/engine/interpolation_batch.py`.

This module owns one model step. Its interface should accept:

- model name or model variant metadata
- model object
- dataset batch
- device
- flow approximation configuration
- init-flow downscale configuration
- execution mode, such as train or inference
- scale factor for inference

It should return named data instead of loose tuples or ad hoc dictionaries. The result should include:

- `img0`
- `img1`
- `imgt`
- `imgt_pred`
- `init_bmv`
- `init_fmv`
- `up_flow0_1`
- `up_flow1_1`
- `up_mask_1`
- training losses when the execution mode requires them
- optional splatting region or debug data when inference needs it

The module owns fragile knowledge that is currently repeated:

- which dataset tuple shape belongs to each model variant
- how source 30fps flow becomes init flow
- how init-flow coverage masks pass into `IFRNet_Residual`
- which model output positions are meaningful
- which output keys are available to metrics and artifacts

### Supporting Modules For Immediate Script Shrink

Add supporting modules when they reduce script size and serve the two deep modules:

- `src/engine/model_registry.py`
  - Move `MODEL_NAMES`, `FLOW_APPROX_MODEL_NAMES`, `uses_flow_approx_model`, `resolve_model_class`, and `set_model_convex_upsampling`.
- `src/engine/dataset_runs.py`
  - Move `build_merged_dataframe`, `build_training_dataset`, `resolve_dataset_class_name`, and inference dataset selection for `VFITrainDataset` and `FlowEstimationTrainDataset`.
- `src/engine/checkpoints.py`
  - Move `save_checkpoint`, `is_raw_model_state_dict`, `extract_pretrained_state_dict`, and `load_training_state`. Keep the `TrainingState` dataclass with the checkpoint helpers to avoid circular imports.

Metric accumulation remains a later refactor unless extracting the selected seams requires a small move.

## Validation Ownership

Use single-owner validation. Do not repeat the same checks throughout intermediate pipeline code.

- `run_config.py` validates config values and incompatible config combinations once.
- `model_registry.py` validates unknown model names once.
- `checkpoints.py` validates checkpoint structure once.
- `interpolation_batch.py` validates unsupported model/batch contracts once.

After a seam returns a validated object, downstream code should trust it. This keeps fail-fast behavior without defensive checks scattered through every helper.

## Data Flow

### Training

1. `scripts/train.py` parses CLI and YAML as it does today.
2. Raw values are converted into `TrainRunConfig`.
3. Datasets and model are built through engine modules.
4. The training loop calls `run_training_batch` or equivalent from `interpolation_batch.py`.
5. Existing metric and checkpoint behavior is preserved.

### Inference

1. `scripts/inference.py` parses the same config keys as today.
2. Raw values are converted into `InferenceRunConfig`.
3. Dataset groups and model are built through engine modules.
4. The inference loop calls `run_inference_batch` or equivalent from `interpolation_batch.py`.
5. Existing metrics, selected sample saving, and CSV schemas are preserved.

## Error Handling

- Raise specific `ValueError`, `TypeError`, `KeyError`, or `RuntimeError` at the owning seam.
- Error messages should include the offending value and available options when relevant.
- Do not silently fall back to default behavior when the config explicitly asks for an invalid mode.
- Do not catch broad exceptions in the refactor modules.

## Testing And Verification

Use lightweight verification in this worktree:

- `python -m compileall src scripts tests`
- import smoke checks for new modules
- train dry-run configs for representative model variants:
  - baseline
  - residual
  - residual flow approximation
- inference dry-run configs for representative model variants
- compare dry-run JSON summaries before and after for each representative train and inference config

Full GPU training and dataset inference are outside this refactor unless explicitly requested.

## Implementation Order

1. Move model registry behavior to `src/engine/model_registry.py`.
2. Add `src/engine/run_config.py` and migrate config parsing and validation without changing YAML keys.
3. Move dataset run helpers to `src/engine/dataset_runs.py`.
4. Move checkpoint helpers to `src/engine/checkpoints.py` if the import graph stays simple.
5. Add `src/engine/interpolation_batch.py`.
6. Thin `scripts/train.py` around the new modules.
7. Thin `scripts/inference.py` around the new modules.
8. Run compile and dry-run verification.

## Review Notes

This design intentionally creates internal seams before changing behavior. If script size does not shrink enough after the first pass, the next candidate is metric accumulation, not a wider rewrite of model or dataset behavior.
