# Inference Benchmarks Design

## Goal

Add a `benchmarks/` folder for measuring per-model inference speed on real frame samples without including dataset construction, metrics, artifact export, or video generation.

## Scope

The benchmark uses existing inference YAML files to construct the model, load checkpoints, and reuse model initialization settings. It reads already-materialized RGB input pairs from a benchmark input folder. Timing includes tensor preparation on the selected device and the model inference call, including wrapper behavior such as padding, TTA, and timestep handling. Timing excludes dataset DataFrame creation, metric calculation, output image saving, video export, RGB sequence export, and top-k sample artifact export.

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
      img0.png
      img2.png
      meta.json
  outputs/
```

Each `benchmark_xxx.py` is a thin model-specific entrypoint. Shared argument parsing, input loading, model loading, timing, and report writing live in `benchmarks/common.py`.

## Input Contract

Each sample directory must contain `img0.png` and `img2.png`. Optional `meta.json` may define:

```json
{
  "timestep": 0.5
}
```

If `meta.json` is missing, the timestep is `0.5`. Inputs are loaded as RGB tensors in `[0, 1]`.

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

## Timing Method

For CUDA, the benchmark synchronizes before and after each measured call using `torch.cuda.synchronize()`. It records per-iteration milliseconds and reports `mean_ms`, `median_ms`, `p90_ms`, `min_ms`, `max_ms`, and `fps`.

## Outputs

The benchmark writes one CSV and one JSON report per run. Reports include model name, config path, checkpoint path, device, GPU name when available, input shape, warmup count, repeat count, batch size, and timing statistics.

## Errors

The benchmark fails fast when the config is missing, the input folder is empty, required sample images are missing, checkpoint loading fails, CUDA is requested but unavailable, or sample image shapes in a batch do not match.

## Testing

Add focused smoke tests for pure benchmark helpers: sample discovery, meta parsing, summary statistics, and report row formatting. Avoid requiring real checkpoints or CUDA in tests.
