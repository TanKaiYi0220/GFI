# GFI Research Scaffold

This repository provides a simplified academic project structure for IFRNet-based architecture research with custom datasets.

## Docker

This project can run in Docker on top of `nvcr.io/nvidia/pytorch:24.02-py3`.

Before the first build, log in to NGC so Docker can pull `nvcr.io` images.

```powershell
docker login nvcr.io
```

Use username `$oauthtoken` and your NGC API key as the password.

Included in the Docker setup:

- Base image: `nvcr.io/nvidia/pytorch:24.02-py3`
- Python runtime aligned to the container's Python 3.10 environment
- Python packages from `pyproject.toml`: `pandas`, `scikit-image`, `scikit-learn`
- Default interactive shell: `zsh`
- Runtime layout aligned with your manual workflow:
  - host dataset: `/datasets/VFI/datasets/VFI_0326`
  - host project: `$HOME/Desktop/VFI`
  - container dataset path: `/workspace/datasets`
  - container project path: `/workspace/src`
  - container name: `GFI`
  - IPC mode: `host`

Build the image:

```powershell
docker compose build
```

Open an interactive container with GPU access:

```powershell
docker compose run --rm gfi
```

Equivalent `docker run` command:

```powershell
docker run \
  -v /datasets/VFI/datasets/VFI_0326:/workspace/datasets \
  -v $HOME/Desktop/VFI:/workspace/src \
  --name GFI \
  --gpus all \
  --ipc=host \
  -it gfi:24.02-pytorch zsh
```

Run a project command inside the container:

```powershell
docker compose run --rm gfi python scripts/train.py --mode dry-run --model-name IFRNet --train-preset train_vfx_0416 --test-preset test_vfx_0416 --root-dir /workspace/datasets
```

### zsh Customization

The container starts with `zsh` and automatically sources `docker/zshrc`.

If you want to keep your own aliases or prompt customizations, create a project-local override file:

```powershell
New-Item -ItemType File .zshrc.local
```

Anything in `.zshrc.local` will be sourced automatically when the container starts.

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

Edit the global variables at the top of `scripts/preprocess_dataset.py` to choose the preset, paths, merge strategy, and which preprocessing steps should run.

```powershell
python scripts/preprocess_dataset.py
```

Set `REMOVE_IDENTICAL = True` to generate one raw frame-index CSV per mode and mark frames identical to the previous frame as invalid.

```powershell
python scripts/preprocess_dataset.py
```

Set `CHECK_IDENTICAL_CROSS_FPS = True` to compare the generated 30fps and 60fps frame-index CSV files for cross-FPS image consistency.

```powershell
python scripts/preprocess_dataset.py
```

Set `MANUAL_LABELING = True` to launch the Easy-vs-Medium review loop after implementing `review_images()` in `src/data/manual_labeling.py`.

```powershell
python scripts/preprocess_dataset.py
```

Set `MERGE_DATASETS = True`. Use `MERGE_STRATEGY = "only-difficult"` for the VFX-style workflow or switch to `merge-easy-medium` when you need one shared validity mask.

```powershell
python scripts/preprocess_dataset.py
```

Set `RAW_SEQUENCE = True` to generate `raw_sequence_frame_index.csv` files from the preprocessed 30fps and 60fps validity CSV files.

```powershell
python scripts/preprocess_dataset.py
```

Set `LINEARITY_CHECK = True` to append linearity statistics after implementing `load_backward_velocity()` in `src/data/image_ops.py`.

```powershell
python scripts/preprocess_dataset.py
```

Enable several global step flags at once when you want one end-to-end pass instead of one command per stage.

```powershell
python scripts/preprocess_dataset.py
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
