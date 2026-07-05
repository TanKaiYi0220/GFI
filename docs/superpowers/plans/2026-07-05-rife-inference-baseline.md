# RIFE Inference Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add official RIFE as an image-only inference/evaluation baseline while keeping external source and checkpoint files out of git.

**Architecture:** Keep the official RIFE clone under ignored `src/models/external/RIFE`, and commit only the GFI-side adapter, registry/runtime seams, setup helper, and representative config. Treat RIFE as image interpolation only: it produces `imgt_pred` for metrics and selected image artifacts, but does not provide GFI flow, masks, residual merge images, or flow-diff artifacts.

**Tech Stack:** Python 3.10+, PyTorch, existing JSON-compatible `.yaml` configs, existing `scripts/inference.py`, official `hzwer/ECCV2022-RIFE` HD model package files under `src/models/external/RIFE/train_log`.

## Global Constraints

- Use the official RIFE repository: `https://github.com/hzwer/ECCV2022-RIFE`.
- Clone target: `src/models/external/RIFE`.
- Checkpoint target: `src/models/external/RIFE/train_log/flownet.pkl`.
- `src/models/external/` remains gitignored.
- Do not commit the official RIFE repo or checkpoint into GFI.
- Do not add RIFE fine-tuning in this slice.
- Fail fast for missing external repo, missing RIFE model-package files, missing checkpoint directory, missing `flownet.pkl`, unsupported timestep shape, and flow-specific requests that RIFE cannot satisfy.
- Keep config files JSON-compatible because `src.utils.config.load_yaml_file` uses `json.load`.
- Preserve existing IFRNet checkpoint loading and inference behavior.
- Use smoke checks, import checks, dry-run JSON checks, and compile checks; do not add a broad unit-test suite.
- Make one commit per task.

---

## File Structure

- Create `scripts/setup_external_rife.py`
  - Owns cloning the official RIFE repository into the ignored external directory.
  - Refuses to overwrite an existing non-empty directory.
  - Prints the exact manual checkpoint/model-package placement expected by the adapter.
- Create `src/models/RIFE.py`
  - Owns the GFI adapter around official RIFE HD v3 model-package files.
  - Imports `train_log.IFNet_HDv3.IFNet` from the ignored external clone.
  - Provides `load_external_checkpoint(checkpoint_path: Path, device: Any) -> None`.
  - Provides `inference(img0: Any, img1: Any, embt: Any, scale_factor: float) -> Any`.
- Modify `src/engine/model_registry.py`
  - Adds `RIFE_MODEL_NAME`, `TRAIN_MODEL_NAMES`, `INFERENCE_MODEL_NAMES`, and `IMAGE_ONLY_VFI_MODEL_NAMES`.
  - Resolves `RIFE` to `src.models.RIFE.Model`.
  - Keeps RIFE out of flow-approx models and training model choices.
- Modify `scripts/train.py`
  - Uses `TRAIN_MODEL_NAMES` for training CLI choices so RIFE remains inference-only.
- Modify `src/engine/run_config.py`
  - Rejects RIFE in training config resolution even when it arrives through config-backed parser defaults.
- Modify `src/engine/checkpoints.py`
  - Adds a checkpoint-loading seam that delegates to model-owned external checkpoint loaders when present.
- Modify `src/engine/interpolation_batch.py`
  - Adds an image-only inference branch for RIFE and returns `InterpolationBatchResult` with flow-related fields set to `None`.
- Modify `scripts/inference.py`
  - Uses the new checkpoint-loading seam.
  - Allows selected-sample image artifacts for RIFE while leaving flow-specific artifact columns blank.
  - Raises a clear error when `save_topk_largest_flow_diff > 0` is used with an image-only model.
- Create `configs/run/inference_rife_official.yaml`
  - Provides a representative RIFE inference config with PSNR and SSIM enabled and flow-specific selection disabled.

---

### Task 1: External RIFE Setup Helper

**Files:**
- Create: `scripts/setup_external_rife.py`

**Interfaces:**
- Consumes: `git` CLI and the official repository URL.
- Produces:
  - `RIFE_REPO_URL: str`
  - `DEFAULT_TARGET_DIR: Path`
  - `RifeSetupError`
  - `resolve_target_dir(path_value: str) -> Path`
  - `clone_rife_repo(repo_url: str, target_dir: Path) -> None`
  - `main(argv: list[str]) -> None`

- [ ] **Step 1: Create the setup helper**

```python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RIFE_REPO_URL: str = "https://github.com/hzwer/ECCV2022-RIFE.git"
DEFAULT_TARGET_DIR: Path = PROJECT_ROOT / "src" / "models" / "external" / "RIFE"


class RifeSetupError(RuntimeError):
    """Raised when the external RIFE repository cannot be prepared safely."""


def resolve_target_dir(path_value: str) -> Path:
    target_dir = Path(path_value)
    if target_dir.is_absolute():
        return target_dir
    return PROJECT_ROOT / target_dir


def clone_rife_repo(repo_url: str, target_dir: Path) -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        raise RifeSetupError(
            "RIFE target directory already exists and is not empty: "
            f"path={target_dir}. Move it manually or choose --target-dir; this helper will not overwrite it."
        )

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "clone", "--depth", "1", repo_url, str(target_dir)]
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clone the official RIFE repository into src/models/external/RIFE.")
    parser.add_argument("--repo-url", type=str, default=RIFE_REPO_URL)
    parser.add_argument("--target-dir", type=str, default=str(DEFAULT_TARGET_DIR))
    return parser


def main(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    target_dir = resolve_target_dir(path_value=str(args.target_dir))
    clone_rife_repo(repo_url=str(args.repo_url), target_dir=target_dir)
    print(f"Cloned official RIFE into {target_dir}")
    print("Download the official HD model package and place its files under:")
    print(f"  {target_dir / 'train_log'}")
    print("Required adapter checkpoint:")
    print(f"  {target_dir / 'train_log' / 'flownet.pkl'}")


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 2: Verify the helper imports without cloning**

Run:

```powershell
python -c "import scripts.setup_external_rife as s; print(s.RIFE_REPO_URL); print(s.DEFAULT_TARGET_DIR.name)"
```

Expected output contains:

```text
https://github.com/hzwer/ECCV2022-RIFE.git
RIFE
```

- [ ] **Step 3: Verify the no-overwrite error path**

Run:

```powershell
python -c "from pathlib import Path; from scripts.setup_external_rife import clone_rife_repo, RifeSetupError; p=Path('src/models/external'); p.mkdir(parents=True, exist_ok=True); open(p / 'rife_setup_probe.txt', 'w', encoding='utf-8').write('probe'); clone_rife_repo('https://github.com/hzwer/ECCV2022-RIFE.git', p)"
```

Expected: command exits non-zero with `RifeSetupError` and a message containing `will not overwrite it`.

- [ ] **Step 4: Remove the single probe file**

Run:

```powershell
Remove-Item "src\models\external\rife_setup_probe.txt"
```

Expected: command exits `0`.

- [ ] **Step 5: Verify compile**

Run:

```powershell
python -m compileall scripts/setup_external_rife.py
```

Expected: command exits `0`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add scripts/setup_external_rife.py
git commit -m "chore: add RIFE external setup helper"
```

---

### Task 2: RIFE Adapter

**Files:**
- Create: `src/models/RIFE.py`

**Interfaces:**
- Consumes:
  - `src/models/external/RIFE/model/warplayer.py`
  - `src/models/external/RIFE/train_log/IFNet_HDv3.py`
  - `src/models/external/RIFE/train_log/RIFE_HDv3.py`
  - `src/models/external/RIFE/train_log/flownet.pkl`
- Produces:
  - `RIFEExternalFilesError`
  - `extract_scalar_timestep(embt: torch.Tensor) -> float`
  - `class Model(nn.Module)`
  - `Model.load_external_checkpoint(checkpoint_path: Path, device: Any) -> None`
  - `Model.inference(img0: Any, img1: Any, embt: Any, scale_factor: float) -> Any`

- [ ] **Step 1: Create the adapter module**

```python
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Iterator

import torch
from torch import nn

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
REQUIRED_EXTERNAL_FILES: tuple[Path, ...] = (
    Path("model/warplayer.py"),
    Path("train_log/IFNet_HDv3.py"),
    Path("train_log/RIFE_HDv3.py"),
)
CHECKPOINT_FILENAME: str = "flownet.pkl"


class RIFEExternalFilesError(RuntimeError):
    """Raised when the ignored official RIFE files are not available."""


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    path_value = str(path)
    inserted = False
    if path_value not in sys.path:
        sys.path.insert(0, path_value)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(path_value)


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _require_external_files(external_root: Path) -> None:
    if not external_root.exists():
        raise FileNotFoundError(
            "Official RIFE repository is missing. "
            f"Expected clone at {external_root}. Run: python scripts/setup_external_rife.py"
        )
    if not external_root.is_dir():
        raise NotADirectoryError(f"Official RIFE path must be a directory: path={external_root}")

    missing_files = [str(external_root / relative_path) for relative_path in REQUIRED_EXTERNAL_FILES if not (external_root / relative_path).is_file()]
    if len(missing_files) > 0:
        raise RIFEExternalFilesError(
            "Official RIFE HD model-package files are missing. "
            "Download the official HD model package, unzip it under src/models/external/RIFE/train_log, "
            f"and ensure these files exist: missing_files={missing_files}"
        )


def _load_ifnet_class(external_root: Path) -> type[nn.Module]:
    _require_external_files(external_root=external_root)
    with _temporary_sys_path(path=external_root):
        importlib.invalidate_caches()
        module = importlib.import_module("train_log.IFNet_HDv3")
    ifnet_class = getattr(module, "IFNet", None)
    if ifnet_class is None:
        raise ImportError(f"train_log.IFNet_HDv3 does not define IFNet: external_root={external_root}")
    return ifnet_class


def _convert_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def extract_scalar_timestep(embt: torch.Tensor) -> float:
    if not torch.is_tensor(embt):
        raise TypeError(f"RIFE embt must be a torch.Tensor, got {type(embt).__name__}")
    if embt.numel() == 0:
        raise ValueError("RIFE embt must contain at least one timestep value.")
    if embt.dim() == 0:
        return float(embt.detach().cpu().item())

    batch_values = embt.detach().float().reshape(int(embt.shape[0]), -1)
    first_value_per_sample = batch_values[:, 0]
    if not torch.allclose(batch_values, first_value_per_sample[:, None].expand_as(batch_values)):
        raise ValueError(f"RIFE supports one scalar timestep per sample, got embt_shape={tuple(embt.shape)}")
    if not torch.allclose(first_value_per_sample, first_value_per_sample[:1].expand_as(first_value_per_sample)):
        values = first_value_per_sample.detach().cpu().tolist()
        raise ValueError(f"RIFE supports one shared timestep per batch, got per_sample_values={values}")

    return float(first_value_per_sample[0].detach().cpu().item())


class Model(nn.Module):
    def __init__(self, external_root: str) -> None:
        super().__init__()
        self.external_root: Path = _resolve_project_path(path_value=external_root)
        ifnet_class = _load_ifnet_class(external_root=self.external_root)
        self.flownet: nn.Module = ifnet_class()
        self._checkpoint_loaded: bool = False

    def load_external_checkpoint(self, checkpoint_path: Path, device: Any) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "RIFE checkpoint directory is missing. "
                f"Expected directory containing {CHECKPOINT_FILENAME}: path={checkpoint_path}"
            )
        if not checkpoint_path.is_dir():
            raise NotADirectoryError(
                "RIFE checkpoint_path must be a directory containing flownet.pkl, "
                f"got path={checkpoint_path}"
            )

        flownet_path = checkpoint_path / CHECKPOINT_FILENAME
        if not flownet_path.is_file():
            raise FileNotFoundError(
                "RIFE checkpoint file is missing. "
                f"Expected official checkpoint at {flownet_path}"
            )

        state_dict = torch.load(str(flownet_path), map_location=device)
        if not isinstance(state_dict, dict):
            raise TypeError(f"RIFE flownet checkpoint must be a state_dict mapping, got {type(state_dict).__name__}")

        converted_state_dict = _convert_state_dict_keys(state_dict=state_dict)
        try:
            self.flownet.load_state_dict(converted_state_dict, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "RIFE checkpoint is incompatible with train_log.IFNet_HDv3.IFNet. "
                f"checkpoint_path={flownet_path}"
            ) from error
        self.flownet.to(device)
        self._checkpoint_loaded = True

    def inference(self, img0: Any, img1: Any, embt: Any, scale_factor: float) -> Any:
        if not self._checkpoint_loaded:
            raise RuntimeError("RIFE checkpoint is not loaded. Call load_external_checkpoint before inference.")
        timestep = extract_scalar_timestep(embt=embt)
        imgs = torch.cat((img0, img1), dim=1)
        scale_list = [8.0 / scale_factor, 4.0 / scale_factor, 2.0 / scale_factor, 1.0 / scale_factor]
        _flow, _mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[3].clamp(0.0, 1.0)
```

- [ ] **Step 2: Verify module import without external files**

Run:

```powershell
python -c "import src.models.RIFE as rife; print(rife.CHECKPOINT_FILENAME)"
```

Expected output:

```text
flownet.pkl
```

- [ ] **Step 3: Verify missing external repo fails fast**

Run:

```powershell
python -c "from src.models.RIFE import Model; Model(external_root='src/models/external/RIFE_DOES_NOT_EXIST')"
```

Expected: command exits non-zero with `FileNotFoundError` and a message containing `Run: python scripts/setup_external_rife.py`.

- [ ] **Step 4: Verify timestep validation**

Run:

```powershell
python -c "import torch; from src.models.RIFE import extract_scalar_timestep; print(extract_scalar_timestep(torch.tensor([[0.5], [0.5]]))); extract_scalar_timestep(torch.tensor([[0.25], [0.75]]))"
```

Expected: command exits non-zero after printing `0.5`, with `ValueError` containing `one shared timestep per batch`.

- [ ] **Step 5: Verify compile**

Run:

```powershell
python -m compileall src/models/RIFE.py
```

Expected: command exits `0`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src/models/RIFE.py
git commit -m "feat: add RIFE inference adapter"
```

---

### Task 3: Runtime Registry, Checkpoint, and Batch Seams

**Files:**
- Modify: `src/engine/model_registry.py`
- Modify: `scripts/train.py`
- Modify: `src/engine/run_config.py`
- Modify: `src/engine/checkpoints.py`
- Modify: `src/engine/interpolation_batch.py`
- Modify: `scripts/inference.py`

**Interfaces:**
- Consumes:
  - `src.models.RIFE.Model`
  - `Model.load_external_checkpoint(checkpoint_path: Path, device: Any) -> None` from Task 2
- Produces:
  - `RIFE_MODEL_NAME: str`
  - `TRAIN_MODEL_NAMES: tuple[str, ...]`
  - `INFERENCE_MODEL_NAMES: tuple[str, ...]`
  - `IMAGE_ONLY_VFI_MODEL_NAMES: tuple[str, ...]`
  - `uses_image_only_vfi_model(model_name: str) -> bool`
  - `load_inference_checkpoint(model: Any, checkpoint_path: Path, device: Any) -> None`
  - RIFE-compatible `run_inference_batch(...) -> InterpolationBatchResult`

- [ ] **Step 1: Extend the model registry**

Replace `src/engine/model_registry.py` with:

```python
from __future__ import annotations

from typing import Any

BASELINE_MODEL_NAME: str = "IFRNet"
RESIDUAL_MODEL_NAME: str = "IFRNet_Residual"
RESIDUAL_FLOW_APPROX_MODEL_NAME: str = "IFRNet_Residual_FlowApprox"
RIFE_MODEL_NAME: str = "RIFE"

TRAIN_MODEL_NAMES: tuple[str, ...] = (BASELINE_MODEL_NAME, RESIDUAL_MODEL_NAME, RESIDUAL_FLOW_APPROX_MODEL_NAME)
INFERENCE_MODEL_NAMES: tuple[str, ...] = (*TRAIN_MODEL_NAMES, RIFE_MODEL_NAME)
MODEL_NAMES: tuple[str, ...] = INFERENCE_MODEL_NAMES
FLOW_APPROX_MODEL_NAMES: tuple[str, ...] = (RESIDUAL_FLOW_APPROX_MODEL_NAME,)
IMAGE_ONLY_VFI_MODEL_NAMES: tuple[str, ...] = (RIFE_MODEL_NAME,)


def uses_flow_approx_model(model_name: str) -> bool:
    return model_name in FLOW_APPROX_MODEL_NAMES


def uses_image_only_vfi_model(model_name: str) -> bool:
    return model_name in IMAGE_ONLY_VFI_MODEL_NAMES


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
    if model_name == RIFE_MODEL_NAME:
        from src.models.RIFE import Model as RIFEModel

        return RIFEModel

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

- [ ] **Step 2: Keep RIFE out of training CLI choices**

In `scripts/train.py`, change the import and parser choice:

```python
from src.engine.model_registry import TRAIN_MODEL_NAMES
```

and:

```python
parser.add_argument("--model-name", default=config_defaults.get("model_name", "IFRNet"), choices=TRAIN_MODEL_NAMES)
```

Keep the existing `resolve_model_class`, `set_model_convex_upsampling`, and `uses_flow_approx_model` imports.

- [ ] **Step 3: Reject inference-only models in training config resolution**

In `src/engine/run_config.py`, change the model-registry import block to include `TRAIN_MODEL_NAMES`:

```python
from src.engine.model_registry import TRAIN_MODEL_NAMES
from src.engine.model_registry import uses_flow_approx_model
```

Add this helper near `read_optional_bool`:

```python
def require_train_model_name(model_name: str) -> None:
    if model_name in TRAIN_MODEL_NAMES:
        return

    available_models = ", ".join(TRAIN_MODEL_NAMES)
    raise ValueError(
        f"Training does not support model_name={model_name}. "
        f"Available training models: {available_models}. RIFE is inference-only in this baseline slice."
    )
```

In `build_train_run_config`, add this check immediately after `metric_config = read_metric_config(config_defaults)`:

```python
    model_name = str(args.model_name)
    require_train_model_name(model_name=model_name)
```

Then reuse that `model_name` local when creating `ModelRunConfig` instead of calling `str(args.model_name)` again.

- [ ] **Step 4: Add the checkpoint-loading seam**

In `src/engine/checkpoints.py`, add this function after `load_inference_state_dict`:

```python
def load_inference_checkpoint(model: Any, checkpoint_path: Path, device: Any) -> None:
    external_loader = getattr(model, "load_external_checkpoint", None)
    if callable(external_loader):
        external_loader(checkpoint_path=checkpoint_path, device=device)
        return

model.load_state_dict(load_inference_state_dict(checkpoint_path=checkpoint_path, device=device))
```

- [ ] **Step 5: Use the checkpoint-loading seam in inference**

In `scripts/inference.py`, replace:

```python
from src.engine.checkpoints import load_inference_state_dict
```

with:

```python
from src.engine.checkpoints import load_inference_checkpoint
```

Replace:

```python
model.load_state_dict(load_inference_state_dict(checkpoint_path=run_config.checkpoint_path, device=device))
```

with:

```python
load_inference_checkpoint(model=model, checkpoint_path=run_config.checkpoint_path, device=device)
```

- [ ] **Step 6: Add image-only inference support in interpolation batches**

In `src/engine/interpolation_batch.py`, add this import:

```python
from src.engine.model_registry import uses_image_only_vfi_model
```

In `_run_model_inference`, add this branch before the `if model_name == BASELINE_MODEL_NAME:` branch:

```python
    if uses_image_only_vfi_model(model_name):
        imgt_pred = model.inference(
            batch_inputs.img0,
            batch_inputs.img1,
            batch_inputs.embt,
            scale_factor,
        )
        return InterpolationBatchResult(
            img0=batch_inputs.img0,
            img1=batch_inputs.img1,
            imgt=batch_inputs.imgt,
            imgt_pred=imgt_pred,
            embt=batch_inputs.embt,
            info=batch_inputs.info,
            bmv=None,
            fmv=None,
            init_bmv=None,
            init_fmv=None,
            init_masks=None,
            up_flow0_1=None,
            up_flow1_1=None,
            up_mask_1=None,
            imgt_merge=None,
            loss_rec=None,
            loss_geo=None,
            loss_dis=None,
            splatting_region_maps=None,
        )
```

- [ ] **Step 7: Verify registry and imports**

Run:

```powershell
python -c "from src.engine.model_registry import INFERENCE_MODEL_NAMES, TRAIN_MODEL_NAMES, resolve_model_class, uses_image_only_vfi_model; print(INFERENCE_MODEL_NAMES); print(TRAIN_MODEL_NAMES); print(resolve_model_class('RIFE').__name__); print(uses_image_only_vfi_model('RIFE'))"
```

Expected output contains:

```text
('IFRNet', 'IFRNet_Residual', 'IFRNet_Residual_FlowApprox', 'RIFE')
('IFRNet', 'IFRNet_Residual', 'IFRNet_Residual_FlowApprox')
Model
True
```

- [ ] **Step 8: Verify training rejects RIFE**

Run:

```powershell
python -c "from types import SimpleNamespace; from src.engine.run_config import build_train_run_config; args=SimpleNamespace(model_name='RIFE', mode='dry-run', train_preset='train_vfx_0416', test_preset='test_vfx_0416', root_dir='data', dataset_root_dir='data', output_dir='outputs/rife_train_probe', seed=1234, epochs=1, batch_size=1, eval_interval=1, input_fps=30, only_fps=60, resume_path=None, pretrained_checkpoint_path=None, flow_approx_method='combination', splatting_fill_strategy='none', init_flow_downscale_strategy='bilinear', init_flow_mask_epsilon=1e-6, eval_convex_upsampling=None, sample_train_frames=[], sample_test_frames=[], sample_interval_epoch=1); build_train_run_config(args, {'metrics': {'enable_psnr': True, 'enable_ssim': False, 'enable_lpips': False, 'enable_flip': False, 'enable_psnr_div': False, 'enable_flolpips': False, 'enable_vfips': False}})"
```

Expected: command exits non-zero with `ValueError` containing `RIFE is inference-only`.

- [ ] **Step 9: Verify missing checkpoint fails through the new seam**

Run:

```powershell
python -c "from pathlib import Path; from src.engine.checkpoints import load_inference_checkpoint; model=type('PlainModel', (), {})(); load_inference_checkpoint(model, Path('missing_rife_checkpoint.pth'), 'cpu')"
```

Expected: command exits non-zero with the existing IFRNet checkpoint error from `torch.load`, proving non-RIFE behavior still uses `load_inference_state_dict`.

- [ ] **Step 10: Verify compile**

Run:

```powershell
python -m compileall src/engine/model_registry.py src/engine/run_config.py src/engine/checkpoints.py src/engine/interpolation_batch.py scripts/train.py scripts/inference.py
```

Expected: command exits `0`.

- [ ] **Step 11: Commit**

Run:

```powershell
git add src/engine/model_registry.py src/engine/run_config.py src/engine/checkpoints.py src/engine/interpolation_batch.py scripts/train.py scripts/inference.py
git commit -m "feat: route RIFE through inference runtime seams"
```

---

### Task 4: RIFE-Compatible Selected Artifacts

**Files:**
- Modify: `scripts/inference.py`

**Interfaces:**
- Consumes:
  - `InterpolationBatchResult` from `src.engine.interpolation_batch`
  - `uses_image_only_vfi_model(model_name: str) -> bool`
- Produces:
  - `SELECTED_ARTIFACT_COLUMNS: tuple[str, ...]`
  - `build_blank_selected_artifact_paths() -> dict[str, str]`
  - `has_flow_artifacts(inference_result: InterpolationBatchResult) -> bool`
  - `validate_flow_specific_requests(model_name: str, save_topk_largest_flow_diff: int) -> None`
  - `save_selected_sample_artifacts(...) -> dict[str, str]` that saves base images for RIFE and leaves flow columns blank.

- [ ] **Step 1: Add registry import and artifact constants**

In `scripts/inference.py`, add this import:

```python
from src.engine.model_registry import uses_image_only_vfi_model
```

Add these constants after `VfipsSegment = dict[str, Any]`:

```python
SELECTED_ARTIFACT_COLUMNS: tuple[str, ...] = (
    "image_0_path",
    "image_1_path",
    "image_gt_path",
    "image_pred_path",
    "image_merge_path",
    "bmv_path",
    "fmv_path",
    "flow_1_to_0_path",
    "flow_1_to_2_path",
    "flow_mask_path",
    "image_0_warped_path",
    "image_1_warped_path",
    "image_0_bmv_warped_path",
    "image_1_fmv_warped_path",
    "image_0_init_warped_path",
    "image_1_init_warped_path",
    "image_init_warped_merge_path",
    "splatting_region_label_bmv_color_path",
    "splatting_region_label_fmv_color_path",
    "splatting_hit_count_bmv_path",
    "splatting_hit_count_fmv_path",
)
```

- [ ] **Step 2: Add artifact helper functions**

Add these functions before `save_selected_sample_artifacts`:

```python
def build_blank_selected_artifact_paths() -> dict[str, str]:
    return {column_name: "" for column_name in SELECTED_ARTIFACT_COLUMNS}


def has_flow_artifacts(inference_result: InterpolationBatchResult) -> bool:
    return (
        inference_result.bmv is not None
        and inference_result.fmv is not None
        and inference_result.up_flow0_1 is not None
        and inference_result.up_flow1_1 is not None
        and inference_result.up_mask_1 is not None
    )


def validate_flow_specific_requests(model_name: str, save_topk_largest_flow_diff: int) -> None:
    if uses_image_only_vfi_model(model_name=model_name) and save_topk_largest_flow_diff > 0:
        raise ValueError(
            "save_topk_largest_flow_diff requires flow artifacts, "
            f"but model_name={model_name} returns image-only predictions. Set save_topk_largest_flow_diff=0."
        )
```

- [ ] **Step 3: Refactor selected artifact saving**

Inside `save_selected_sample_artifacts`, remove the early flow-artifact `RuntimeError` block. Build and save base images first, then return early when flow artifacts are absent:

```python
    img0_np = np.round(img0[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img1_np = np.round(img1[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    imgt_np = np.round(imgt[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img_pred_np = np.round(imgt_pred[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    imgt_merge_np = None if imgt_merge is None else np.round(imgt_merge[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    image_paths = build_blank_selected_artifact_paths()
    image_paths.update(
        {
            "image_0_path": str(save_dir / "image_0.png"),
            "image_1_path": str(save_dir / "image_1.png"),
            "image_gt_path": str(save_dir / "image_gt.png"),
            "image_pred_path": str(save_dir / "image_pred.png"),
            "image_merge_path": str(save_dir / "image_merge.png") if imgt_merge_np is not None else "",
        }
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    save_image(Path(image_paths["image_0_path"]), img0_np)
    save_image(Path(image_paths["image_1_path"]), img1_np)
    save_image(Path(image_paths["image_gt_path"]), imgt_np)
    save_image(Path(image_paths["image_pred_path"]), img_pred_np)
    if imgt_merge_np is not None:
        save_image(Path(image_paths["image_merge_path"]), imgt_merge_np)

    if not has_flow_artifacts(inference_result=inference_result):
        return image_paths

    from src.models.external.IFRNet.utils import warp
```

Keep the existing flow visualization code after the `warp` import. Remove the later duplicate `image_paths = { ... }` block and keep the flow-specific key assignments by changing them to `image_paths.update({ ... })`.

- [ ] **Step 4: Use the artifact column constant**

Replace the long inline artifact column tuple in `main` with:

```python
            for column_name in SELECTED_ARTIFACT_COLUMNS:
                group_metrics_df[column_name] = ""
```

- [ ] **Step 5: Validate flow-specific requests once**

After `metric_config = dict(run_config.metrics.values)` in `main`, add:

```python
    validate_flow_specific_requests(
        model_name=run_config.model.model_name,
        save_topk_largest_flow_diff=run_config.save_topk_largest_flow_diff,
    )
```

- [ ] **Step 6: Verify image-only artifact helpers**

Run:

```powershell
python -c "from scripts.inference import SELECTED_ARTIFACT_COLUMNS, build_blank_selected_artifact_paths, validate_flow_specific_requests; print(len(SELECTED_ARTIFACT_COLUMNS)); print(all(v == '' for v in build_blank_selected_artifact_paths().values())); validate_flow_specific_requests('RIFE', 0)"
```

Expected output contains:

```text
21
True
```

- [ ] **Step 7: Verify incompatible RIFE flow selection fails fast**

Run:

```powershell
python -c "from scripts.inference import validate_flow_specific_requests; validate_flow_specific_requests('RIFE', 1)"
```

Expected: command exits non-zero with `ValueError` containing `Set save_topk_largest_flow_diff=0`.

- [ ] **Step 8: Verify compile**

Run:

```powershell
python -m compileall scripts/inference.py
```

Expected: command exits `0`.

- [ ] **Step 9: Commit**

Run:

```powershell
git add scripts/inference.py
git commit -m "feat: allow image-only RIFE inference artifacts"
```

---

### Task 5: RIFE Config and End-to-End Smoke Checks

**Files:**
- Create: `configs/run/inference_rife_official.yaml`

**Interfaces:**
- Consumes:
  - `src.models.RIFE.Model(external_root: str)`
  - `checkpoint_path` directory contract from Task 2
  - `scripts/inference.py --config <path>`
- Produces:
  - A representative RIFE inference config with `model_name: "RIFE"`, `model_init_args.external_root`, and `checkpoint_path: "src/models/external/RIFE/train_log"`.

- [ ] **Step 1: Create the representative RIFE inference config**

```json
{
  "mode": "infer",
  "model_name": "RIFE",
  "model_init_args": {
    "external_root": "src/models/external/RIFE"
  },
  "root_dir": "data/",
  "dataset_root_dir": "/workspace/datasets/",
  "inference_preset": "test_vfx_0416",
  "checkpoint_path": "src/models/external/RIFE/train_log",
  "output_dir": "inference_outputs/RIFE_Official",
  "seed": 1234,
  "batch_size": 1,
  "only_fps": 60,
  "input_fps": 30,
  "metrics": {
    "enable_psnr": true,
    "enable_ssim": true,
    "enable_lpips": false,
    "enable_flip": false,
    "enable_psnr_div": false,
    "enable_flolpips": false,
    "enable_vfips": false,
    "lpips_net": "alex",
    "flip_dynamic_range": "LDR",
    "flip_pixels_per_degree": 67.0,
    "flip_tonemapper": "ACES",
    "psnr_div_divergence_threshold": 0.01,
    "flolpips_repo_path": null,
    "vfips_repo_path": null,
    "vfips_checkpoint_path": null,
    "vfips_clip_length": 12,
    "vfips_stride": 12
  },
  "scale_factor": 1.0,
  "flow_approx_method": "combination",
  "flow_diff_threshold": 1.0,
  "flow_diff_percentile": 99.0,
  "save_topk_worst_psnr": 3,
  "save_topk_best_psnr": 0,
  "save_topk_largest_flow_diff": 0,
  "video": {
    "output_dir": "inference_outputs/RIFE_Official/videos",
    "fps": 60,
    "ignore_valid": true,
    "record_filter": null,
    "mode_filter": null,
    "export_grid": true,
    "tile_scale": 0.5,
    "pad": 8,
    "export_all": false,
    "export_vfi60": true,
    "single_files": []
  }
}
```

- [ ] **Step 2: Verify config is JSON-compatible**

Run:

```powershell
python -m json.tool configs/run/inference_rife_official.yaml > $null
```

Expected: command exits `0`.

- [ ] **Step 3: Verify dry-run JSON**

Run:

```powershell
$dryRunConfig = Join-Path $env:TEMP "inference_rife_official.dry-run.yaml"
python -c "import json, pathlib, sys; src=pathlib.Path(sys.argv[1]); dst=pathlib.Path(sys.argv[2]); data=json.loads(src.read_text(encoding='utf-8')); data['mode']='dry-run'; dst.write_text(json.dumps(data, indent=2), encoding='utf-8')" configs/run/inference_rife_official.yaml $dryRunConfig
python scripts/inference.py --config $dryRunConfig | python -m json.tool
```

Expected: command exits `0`, and output contains:

```json
"model_name": "RIFE"
```

- [ ] **Step 4: Verify import smoke**

Run:

```powershell
python -c "import src.models.RIFE; import scripts.inference; print('ok')"
```

Expected output:

```text
ok
```

- [ ] **Step 5: Verify missing setup error is actionable**

Run:

```powershell
python -c "from src.models.RIFE import Model; Model(external_root='src/models/external/RIFE_DOES_NOT_EXIST')"
```

Expected: command exits non-zero with `FileNotFoundError` and a message containing `python scripts/setup_external_rife.py`.

- [ ] **Step 6: Verify optional tiny RIFE inference when the official files are present**

Run:

```powershell
if (Test-Path "src\models\external\RIFE\train_log\flownet.pkl") {
  python -c "import torch; from pathlib import Path; from src.models.RIFE import Model; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=Model(external_root='src/models/external/RIFE').to(device); model.load_external_checkpoint(Path('src/models/external/RIFE/train_log'), device); img0=torch.zeros(1,3,64,64,device=device); img1=torch.ones(1,3,64,64,device=device); embt=torch.tensor([[0.5]],device=device); out=model.inference(img0,img1,embt,1.0); print(tuple(out.shape), float(out.min()), float(out.max()))"
}
```

Expected when checkpoint exists: command exits `0`, prints `(1, 3, 64, 64)`, and min/max values are in `[0.0, 1.0]`.

Expected when checkpoint is absent: PowerShell exits `0` without running Python.

- [ ] **Step 7: Verify compile**

Run:

```powershell
python -m compileall src scripts tests
```

Expected: command exits `0`.

- [ ] **Step 8: Commit**

Run:

```powershell
git add configs/run/inference_rife_official.yaml
git commit -m "config: add official RIFE inference baseline"
```

---

## Final Verification

- [ ] **Step 1: Verify branch and status**

Run:

```powershell
git status --short --branch
```

Expected: branch is `codex/repo-architecture-review` and status is clean except for being ahead of origin.

- [ ] **Step 2: Verify all planned smoke checks together**

Run:

```powershell
python -c "import src.models.RIFE; import scripts.inference; print('imports-ok')"
python -m json.tool configs/run/inference_rife_official.yaml > $null
$dryRunConfig = Join-Path $env:TEMP "inference_rife_official.final.dry-run.yaml"
python -c "import json, pathlib, sys; src=pathlib.Path(sys.argv[1]); dst=pathlib.Path(sys.argv[2]); data=json.loads(src.read_text(encoding='utf-8')); data['mode']='dry-run'; dst.write_text(json.dumps(data, indent=2), encoding='utf-8')" configs/run/inference_rife_official.yaml $dryRunConfig
python scripts/inference.py --config $dryRunConfig | python -m json.tool
python -m compileall src scripts tests
```

Expected: each command exits `0`; dry-run JSON contains `"model_name": "RIFE"`.

- [ ] **Step 3: Inspect committed diff**

Run:

```powershell
git --no-pager diff origin/codex/repo-architecture-review...HEAD --stat
```

Expected: committed changes include the approved RIFE design spec plus this implementation's adapter, runtime seams, setup helper, and config. No files under `src/models/external/` are tracked.
