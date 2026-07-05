from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
REQUIRED_EXTERNAL_FILES: tuple[Path, ...] = (
    Path("core/pipeline.py"),
    Path("core/models/upr_base.py"),
    Path("core/models/upr_large.py"),
    Path("core/models/upr_llarge.py"),
    Path("core/utils/correlation.py"),
    Path("core/models/softsplat/softsplat.py"),
)
MODEL_SIZE_CHOICES: tuple[str, ...] = ("base", "large", "LARGE")


class UPRNetExternalFilesError(RuntimeError):
    """Raised when ignored official UPR-Net files or dependencies are unavailable."""


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


@contextmanager
def _temporary_external_modules(prefixes: tuple[str, ...]) -> Iterator[None]:
    previous_modules = {
        module_name: module
        for module_name, module in sys.modules.items()
        if any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    for module_name in previous_modules:
        sys.modules.pop(module_name, None)

    try:
        yield
    finally:
        for module_name in list(sys.modules):
            if any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in prefixes):
                sys.modules.pop(module_name, None)
        sys.modules.update(previous_modules)


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _require_external_files(external_root: Path) -> None:
    if not external_root.exists():
        raise FileNotFoundError(
            "Official UPR-Net repository is missing. "
            f"Expected clone at {external_root}. "
            "Clone https://github.com/srcn-ivl/UPR-Net into src/models/external/UPR-Net."
        )
    if not external_root.is_dir():
        raise NotADirectoryError(f"Official UPR-Net path must be a directory: path={external_root}")

    missing_files = [
        str(external_root / relative_path)
        for relative_path in REQUIRED_EXTERNAL_FILES
        if not (external_root / relative_path).is_file()
    ]
    if len(missing_files) > 0:
        raise UPRNetExternalFilesError(
            "Official UPR-Net source files are missing. "
            "Clone https://github.com/srcn-ivl/UPR-Net into src/models/external/UPR-Net, "
            f"then verify these files exist: missing_files={missing_files}"
        )


def _load_pipeline_class(external_root: Path) -> type[Any]:
    _require_external_files(external_root=external_root)
    with _temporary_sys_path(path=external_root), _temporary_external_modules(prefixes=("core",)):
        try:
            module = importlib.import_module("core.pipeline")
        except ModuleNotFoundError as error:
            if error.name == "cupy":
                raise UPRNetExternalFilesError(
                    "UPR-Net requires cupy for official correlation and softsplat kernels. "
                    "Install a CUDA-compatible cupy package in this project environment."
                ) from error
            raise

    pipeline_class = getattr(module, "Pipeline", None)
    if pipeline_class is None:
        raise ImportError(f"core.pipeline does not define Pipeline: external_root={external_root}")
    return pipeline_class


def _validate_model_size(model_size: str) -> None:
    if model_size in MODEL_SIZE_CHOICES:
        return
    available_sizes = ", ".join(MODEL_SIZE_CHOICES)
    raise ValueError(f"Unsupported UPR-Net model_size={model_size}. Available sizes: {available_sizes}")


def _validate_positive_int(value: int, name: str) -> None:
    if value > 0:
        return
    raise ValueError(f"UPR-Net {name} must be positive, got {value}")


def _next_multiple(value: int, multiple: int) -> int:
    return ((value - 1) // multiple + 1) * multiple


def _pad_to_size(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    image_height = int(image.shape[-2])
    image_width = int(image.shape[-1])
    padding = (0, width - image_width, 0, height - image_height)
    if padding == (0, 0, 0, 0):
        return image
    return F.pad(image, padding, "constant", 0.5)


def _convert_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def extract_scalar_timestep(embt: torch.Tensor) -> float:
    if not torch.is_tensor(embt):
        raise TypeError(f"UPR-Net embt must be a torch.Tensor, got {type(embt).__name__}")
    if embt.numel() == 0:
        raise ValueError("UPR-Net embt must contain at least one timestep value.")
    if embt.dim() == 0:
        return float(embt.detach().cpu().item())

    batch_values = embt.detach().float().reshape(int(embt.shape[0]), -1)
    first_value_per_sample = batch_values[:, 0]
    if not torch.allclose(batch_values, first_value_per_sample[:, None].expand_as(batch_values)):
        raise ValueError(f"UPR-Net supports one scalar timestep per sample, got embt_shape={tuple(embt.shape)}")
    if not torch.allclose(first_value_per_sample, first_value_per_sample[:1].expand_as(first_value_per_sample)):
        values = first_value_per_sample.detach().cpu().tolist()
        raise ValueError(f"UPR-Net supports one shared timestep per batch, got per_sample_values={values}")

    return float(first_value_per_sample[0].detach().cpu().item())


def _load_strict_model_state(model: nn.Module, checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    converted_checkpoint = _convert_state_dict_keys(state_dict=checkpoint)
    model_state = model.state_dict()
    missing_keys = [key for key in model_state if key not in converted_checkpoint]
    unexpected_keys = [key for key in converted_checkpoint if key not in model_state]
    shape_mismatch_keys = [
        key
        for key in model_state
        if key in converted_checkpoint and model_state[key].shape != converted_checkpoint[key].shape
    ]
    if len(missing_keys) > 0 or len(unexpected_keys) > 0 or len(shape_mismatch_keys) > 0:
        raise RuntimeError(
            "UPR-Net checkpoint is incompatible with the selected official model. "
            f"checkpoint_path={checkpoint_path}, "
            f"missing_keys={missing_keys[:8]}, unexpected_keys={unexpected_keys[:8]}, "
            f"shape_mismatch_keys={shape_mismatch_keys[:8]}"
        )
    model.load_state_dict(converted_checkpoint, strict=True)


class Model(nn.Module):
    def __init__(self, external_root: str, model_size: str, pyr_level: int, nr_lvl_skipped: int) -> None:
        super().__init__()
        _validate_model_size(model_size=model_size)
        _validate_positive_int(value=pyr_level, name="pyr_level")
        if nr_lvl_skipped < 0:
            raise ValueError(f"UPR-Net nr_lvl_skipped must be non-negative, got {nr_lvl_skipped}")

        self.external_root: Path = _resolve_project_path(path_value=external_root)
        _require_external_files(external_root=self.external_root)
        self.model_size: str = model_size
        self.pyr_level: int = pyr_level
        self.nr_lvl_skipped: int = nr_lvl_skipped
        self.pipeline: Any | None = None
        self._checkpoint_loaded: bool = False

    def eval(self) -> Model:
        super().eval()
        if self.pipeline is not None:
            self.pipeline.eval()
        return self

    def load_external_checkpoint(self, checkpoint_path: Path, device: Any) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"UPR-Net checkpoint file is missing: path={checkpoint_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"UPR-Net checkpoint_path must be a .pkl file, got path={checkpoint_path}")

        pipeline_class = _load_pipeline_class(external_root=self.external_root)
        model_config = {
            "load_pretrain": False,
            "model_size": self.model_size,
            "pyr_level": self.pyr_level,
            "nr_lvl_skipped": self.nr_lvl_skipped,
        }
        pipeline = pipeline_class(model_config)
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"UPR-Net checkpoint must be a state_dict mapping, got {type(checkpoint).__name__}")

        _load_strict_model_state(model=pipeline.model, checkpoint=checkpoint, checkpoint_path=checkpoint_path)
        pipeline.model.to(device)
        pipeline.eval()
        self.pipeline = pipeline
        self._checkpoint_loaded = True

    def inference(self, img0: Any, img1: Any, embt: Any, scale_factor: float) -> Any:
        if not self._checkpoint_loaded or self.pipeline is None:
            raise RuntimeError("UPR-Net checkpoint is not loaded. Call load_external_checkpoint before inference.")
        if scale_factor != 1.0:
            raise ValueError(f"UPR-Net does not support scale_factor={scale_factor}; use scale_factor=1.0.")
        if img0.shape[-2:] != img1.shape[-2:]:
            raise ValueError(f"UPR-Net input frames must share spatial shape, got img0={img0.shape} img1={img1.shape}")

        timestep = extract_scalar_timestep(embt=embt)
        height = int(img0.shape[-2])
        width = int(img0.shape[-1])
        divisor = 2 ** (self.pyr_level - 1 + 2)
        padded_height = _next_multiple(value=height, multiple=divisor)
        padded_width = _next_multiple(value=width, multiple=divisor)
        padded_img0 = _pad_to_size(image=img0, height=padded_height, width=padded_width)
        padded_img1 = _pad_to_size(image=img1, height=padded_height, width=padded_width)
        imgt_pred, _bi_flow = self.pipeline.inference(
            padded_img0,
            padded_img1,
            time_period=timestep,
            pyr_level=self.pyr_level,
            nr_lvl_skipped=self.nr_lvl_skipped,
        )
        return imgt_pred[:, :, :height, :width].clamp(0.0, 1.0)


__all__ = ["Model", "UPRNetExternalFilesError", "extract_scalar_timestep"]
