# GFI Research Scaffold

This repository provides a simplified academic project structure for IFRNet-based architecture research with custom datasets.

## Layout

```text
configs/      Experiment, data, model, training, and path configuration
src/          Reusable Python source code
scripts/      Command-line entrypoints for the formal pipeline
experiments/  Short-lived analysis, ablations, and scratch work
outputs/      Generated checkpoints, logs, predictions, and figures
tests/        Validation and smoke-test area
```

## Source Layout

```text
src/
  data/
    dataset.py      Dataset sample schema and directory-based sample collection
    dataset_config.py  Dataset preset definitions and global dataset-root selection
    dataset_loader.py  Project-specific dataframe-to-sample dataset loader
    preprocess.py   Reusable path-rewrite helpers
    analysis.py     Lightweight dataset analysis helpers
  engine/
    evaluation.py   Lightweight metric collection used by training and validation
    pipeline.py     Shared training, inference, and evaluation runtime helpers
  models/
    external/       External or paper-derived model adapters
    registry.py     Model registration and default training-path metadata
    custom/         Private research variants
    components/     Reusable model blocks
    losses/         Loss definitions
  utils/            Config, logging, seed, and filesystem helpers
```

## Dataset Convention

The default templates assume a simple one-directory-per-sample layout:

```text
dataset_root/
  train/
    sample_0001/
      frame_000.png
      frame_001.png
      frame_002.png
    sample_0002/
      frame_000.png
      frame_001.png
      frame_002.png
  val/
  test/
```

Edit the `INPUT_FRAME_NAMES` and `TARGET_FRAME_NAME` constants in the scripts if your filenames differ.
Edit `ACTIVE_DATASET_ROOT_KEY` in `src/data/dataset_config.py` to switch between local and docker dataset roots.

## Commands

### Project Execution

Validate the integrated IFRNet-style training entrypoint and inspect resolved presets, paths, and output directories.

```powershell
python scripts/train.py --mode dry-run --model-name IFRNet --train-preset train_vfx_0416 --test-preset test_vfx_0416 --root-dir C:\dataset_indexes
```

Run the integrated training pipeline after registering your model class in `src/models/registry.py` and implementing `src/data/dataset_loader.py`.

```powershell
python scripts/train.py --mode train --model-name IFRNet --train-preset train_vfx_0416 --test-preset test_vfx_0416 --root-dir C:\dataset_indexes --dataset-root-dir C:\datasets\VFI
```

Validate the full inference entrypoint before filling in the checkpoint-loading and forward-pass hooks.

```powershell
python scripts/inference.py --config configs/experiment/exp001.yaml --checkpoint outputs/checkpoints/latest.pt --output-path outputs/predictions/predictions.json --mode dry-run
```

Run the formal inference template after implementing the model, checkpoint, dataloader, and prediction-save hooks.

```powershell
python scripts/inference.py --config configs/experiment/exp001.yaml --checkpoint outputs/checkpoints/latest.pt --output-path outputs/predictions/predictions.json --mode infer
```

### Component Validation

Check whether one split-based dataset layout can be scanned correctly.

```powershell
python scripts/preprocess_dataset.py --dataset-root C:\dataset --split train --output-dir C:\dataset_processed --mode scan
```

Run split-based preprocessing after implementing `preprocess_sample()`.

```powershell
python scripts/preprocess_dataset.py --dataset-root C:\dataset --split train --output-dir C:\dataset_processed --mode run
```

Check whether one dataset preset expands into the expected sequence directories.

```powershell
python scripts/preprocess_dataset.py --dataset-preset train_vfx_0416 --output-dir C:\dataset_processed --mode dry-run
```

Summarize one split-based dataset to verify sample count and frame-count distribution.

```powershell
python scripts/analyze_dataset.py --dataset-root C:\dataset --split train --mode summary
```

Summarize one preset-based dataset collection to verify sequence count, FPS values, and difficulty coverage.

```powershell
python scripts/analyze_dataset.py --dataset-preset train_vfx_0416 --mode summary
```

Run a direct smoke check for `src/data/dataset_config.py` to verify preset lookup, config expansion, and path generation.

```powershell
python -m src.data.dataset_config --preset train_vfx_0416 --limit 3
```

## Notes

- Keep the formal runtime helpers under `src/engine/pipeline.py`.
- Keep reusable dataset logic under the flat `src/data/` files.
- Keep reusable ARPG-style dataset presets in `src/data/dataset_config.py`.
- Keep dataset root paths in `configs/paths/default.yaml` and switch environments with `ACTIVE_DATASET_ROOT_KEY` in `src/data/dataset_config.py`.
- Register concrete model classes in `src/models/registry.py` before running `scripts/train.py` or `scripts/inference.py`.
- Implement the row-to-tensor logic in `src/data/dataset_loader.py` for your actual CSV schema and frame layout.
- Keep disposable exploratory work under `experiments/` so it does not leak into the formal pipeline.
- Store dataset files outside the repository whenever possible.
- Use the files under `scripts/` as templates.
- Keep `train` and `inference` as the more formal runtime entrypoints.
- Keep `preprocess` and `analyze` lightweight and direct.
- Prefer direct directory scanning over manifest files unless your dataset truly needs a separate index.
