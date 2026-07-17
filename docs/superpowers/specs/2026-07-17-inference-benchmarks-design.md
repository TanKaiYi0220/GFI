# Inference Benchmarks Design

## Goal

Add a `benchmarks/` folder for measuring per-model inference speed on real frame samples without including dataset construction, metrics, artifact export, or video generation. For project models that consume game motion, benchmark inputs include those motion tensors and reports separate flow-approximation time from model inference time.

## Scope

The benchmark uses existing inference YAML files to construct the model, load checkpoints, and reuse model initialization settings. It reads already-materialized RGB input pairs and, when the selected model/config requires them, game motion tensors from a benchmark input folder. Timing includes tensor preparation on the selected device and the model inference call, including wrapper behavior such as padding, TTA, and timestep handling. Timing excludes dataset DataFrame creation, metric calculation, output image saving, video export, RGB sequence export, and top-k sample artifact export.

For flow-approximation models, the benchmark reports flow approximation and model inference as separate phases:

- `transfer_ms`: CPU tensors moved to the selected device.
- `flow_approx_ms`: 30fps game motion converted to initial 60fps motion or masks.
- `model_ms`: model forward/inference after initial flow is available.
- `total_ms`: `transfer_ms + flow_approx_ms + model_ms`.

For image-only baselines, `flow_approx_ms` is `0.0`.

## Folder Layout

```text
benchmarks/
  common.py
  benchmark_rife.py
  benchmark_uprnet.py
  benchmark_emavfi.py
  benchmark_sgmvfi.py
  inputs/
    sample_0001/
      colorNoScreenUI_244.png
      colorNoScreenUI_246.png
      meta.json
      fps_30/
        backwardVel_Depth_123.exr
        forwardVel_Depth_122.exr
      fps_60/
        backwardVel_Depth_245.exr
        forwardVel_Depth_245.exr
  outputs/
```

Each `benchmark_xxx.py` is a thin model-specific entrypoint. Shared argument parsing, input loading, model loading, timing, and report writing live in `benchmarks/common.py`.

The `fps_30/` and `fps_60/` folders are benchmark-local aliases for materialized samples. When samples are referenced directly from the dataset root, `meta.json` may instead point to the original record/mode directories; the EXR basenames and frame-index mapping remain the same as the training dataloader.

## Input Contract

Each sample directory must contain two RGB input images. The default loader accepts either benchmark aliases (`img0.png` and `img2.png`) or the dataset naming style (`colorNoScreenUI_{frame_idx}.png`) declared in `meta.json`. Optional `meta.json` may define:

```json
{
  "timestep": 0.5,
  "frame_60_0_idx": 244,
  "frame_60_1_idx": 245,
  "frame_60_2_idx": 246,
  "frame_30_0_idx": 122,
  "frame_30_1_idx": 123
}
```

If `meta.json` is missing, the timestep is `0.5`. Inputs are loaded as RGB tensors in `[0, 1]`.

Motion files are required only for models/configs that need them:

- `RIFE`, `UPRNet`, `EMAVFI`, `SGMVFI`: RGB-only benchmark samples are valid.
- `IFRNet` and `IFRNet_Residual`: RGB samples are valid; `fps_60/backwardVel_Depth_{frame_60_1_idx}.exr` and `fps_60/forwardVel_Depth_{frame_60_1_idx}.exr` may be loaded when comparing against ground-truth game motion.
- `IFRNet_Residual_FlowApprox`: source 30fps game motion is required from `fps_30/backwardVel_Depth_{frame_30_1_idx}.exr` and `fps_30/forwardVel_Depth_{frame_30_0_idx}.exr`, matching `FlowEstimationTrainDataset`.
- Splatting flow approximation modes that need depth read source depths from the same 30fps EXR files; no separate depth files are part of the default benchmark contract.

The benchmark should reuse the existing EXR loading convention from `src.data.dataset_loader`: `backwardVel_Depth_{frame_idx}.exr` and `forwardVel_Depth_{frame_idx}.exr` are decoded through `load_backward_velocity`, which returns a `[H, W, 2]` game-motion field plus the matching depth channel. The benchmark then converts motion to `[2, H, W]` tensors and depth to `[1, H, W]`, matching the training pipeline. All samples in one benchmark run must share the same RGB and motion/depth shapes.

## CLI

Example:

```powershell
python benchmarks/benchmark_uprnet.py `
  --config configs/run/inference_uprnet_official.yaml `
  --input-dir benchmarks/inputs `
  --warmup 5 `
  --repeat 30 `
  --batch-size 1
```

Shared options:

- `--config`: existing inference YAML.
- `--input-dir`: folder containing pre-stored sample folders.
- `--output-dir`: report folder, default `benchmarks/outputs`.
- `--warmup`: untimed warmup iterations.
- `--repeat`: timed iterations per sample.
- `--batch-size`: number of samples per timed forward pass.
- `--device`: `cuda` by default.

Flow-approximation model example:

```powershell
python benchmarks/benchmark_flowapprox.py `
  --config configs/run/inference_flowaprox_layer_1_0618.yaml `
  --input-dir benchmarks/inputs `
  --warmup 5 `
  --repeat 30 `
  --batch-size 1
```

## Timing Method

For CUDA, the benchmark synchronizes before and after each measured phase using `torch.cuda.synchronize()`. It records per-iteration milliseconds and reports `mean_ms`, `median_ms`, `p90_ms`, `min_ms`, `max_ms`, and `fps`.

The benchmark should not duplicate flow-approximation logic. Instead, `src.engine.interpolation_batch` exposes a small benchmark-facing phase API that reuses the same batch preparation, flow approximation, and model inference functions as normal inference. The benchmark calls those phase helpers with synchronization around each phase.

## Outputs

The benchmark writes one CSV and one JSON report per run. Reports include model name, config path, checkpoint path, device, GPU name when available, input shape, motion shape when used, warmup count, repeat count, batch size, and timing statistics.

For each batch row, reports include:

- `transfer_mean_ms`
- `flow_approx_mean_ms`
- `model_mean_ms`
- `total_mean_ms`
- `flow_approx_percent`
- `model_percent`
- `fps_total`
- `fps_model_only`

Run-level summary includes the same phase fields aggregated over all measured calls.

## Errors

The benchmark fails fast when the config is missing, the input folder is empty, required sample images are missing, required motion/depth files are missing for the selected model/config, checkpoint loading fails, CUDA is requested but unavailable, RGB/motion/depth shapes do not match, or sample image shapes in a batch do not match.

## Testing

Add focused smoke tests for pure benchmark helpers: sample discovery, meta parsing, motion file validation, summary statistics, phase timing aggregation, and report row formatting. Avoid requiring real checkpoints or CUDA in tests. Add integration-style smoke tests with fake models/batches to prove flow-approximation timing is reported separately from model timing without requiring real checkpoints.
