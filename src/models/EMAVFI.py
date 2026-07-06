from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
REQUIRED_EXTERNAL_FILES: tuple[Path, ...] = (
    Path("config.py"),
    Path("model/__init__.py"),
    Path("model/feature_extractor.py"),
    Path("model/flow_estimation.py"),
    Path("model/refine.py"),
    Path("model/warplayer.py"),
)
MODEL_VARIANT_CHOICES: tuple[str, ...] = ("ours", "ours_small", "ours_t", "ours_small_t")


@dataclass(frozen=True)
class _ModelVariantConfig:
    feature_channels: int
    depths: tuple[int, int, int, int, int]


MODEL_VARIANT_CONFIGS: dict[str, _ModelVariantConfig] = {
    "ours": _ModelVariantConfig(feature_channels=32, depths=(2, 2, 2, 4, 4)),
    "ours_t": _ModelVariantConfig(feature_channels=32, depths=(2, 2, 2, 4, 4)),
    "ours_small": _ModelVariantConfig(feature_channels=16, depths=(2, 2, 2, 2, 2)),
    "ours_small_t": _ModelVariantConfig(feature_channels=16, depths=(2, 2, 2, 2, 2)),
}


class EMAVFIExternalFilesError(RuntimeError):
    """Raised when ignored official EMA-VFI files or dependencies are unavailable."""


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
            "Official EMA-VFI repository is missing. "
            f"Expected clone at {external_root}. "
            "Clone https://github.com/MCG-NJU/EMA-VFI into src/models/external/EMA-VFI."
        )
    if not external_root.is_dir():
        raise NotADirectoryError(f"Official EMA-VFI path must be a directory: path={external_root}")

    missing_files = [
        str(external_root / relative_path)
        for relative_path in REQUIRED_EXTERNAL_FILES
        if not (external_root / relative_path).is_file()
    ]
    if len(missing_files) > 0:
        raise EMAVFIExternalFilesError(
            "Official EMA-VFI source files are missing. "
            "Clone https://github.com/MCG-NJU/EMA-VFI into src/models/external/EMA-VFI, "
            f"then verify these files exist: missing_files={missing_files}"
        )


def _validate_model_variant(model_variant: str) -> None:
    if model_variant in MODEL_VARIANT_CHOICES:
        return
    available_variants = ", ".join(MODEL_VARIANT_CHOICES)
    raise ValueError(f"Unsupported EMA-VFI model_variant={model_variant}. Available variants: {available_variants}")


def _validate_bool(value: bool, name: str) -> None:
    if isinstance(value, bool):
        return
    raise TypeError(f"EMA-VFI {name} must be a boolean, got {type(value).__name__}")


def _validate_positive_int(value: int, name: str) -> None:
    if value > 0:
        return
    raise ValueError(f"EMA-VFI {name} must be positive, got {value}")


def _load_external_config_module(external_root: Path) -> Any:
    _require_external_files(external_root=external_root)
    with _temporary_sys_path(path=external_root), _temporary_external_modules(prefixes=("config", "model")):
        try:
            return importlib.import_module("config")
        except ModuleNotFoundError as error:
            if error.name == "timm":
                raise EMAVFIExternalFilesError(
                    "EMA-VFI requires timm for the official feature extractor. "
                    "Install timm in this project environment before loading the checkpoint."
                ) from error
            raise
        except RuntimeError as error:
            if "torchvision::nms" in str(error):
                raise EMAVFIExternalFilesError(
                    "EMA-VFI official import failed because timm imported torchvision, but this environment's "
                    "torch/torchvision installation is incompatible: missing operator torchvision::nms. "
                    "Install mutually compatible torch, torchvision, and timm versions before loading EMA-VFI. "
                    f"external_root={external_root}, original_error={error}"
                ) from error
            raise


def _build_official_network(external_root: Path, model_variant: str) -> nn.Module:
    config_module = _load_external_config_module(external_root=external_root)
    init_model_config = getattr(config_module, "init_model_config", None)
    feature_extractor_class = getattr(config_module, "feature_extractor", None)
    flow_estimation_class = getattr(config_module, "flow_estimation", None)
    if not callable(init_model_config):
        raise ImportError(f"EMA-VFI config.py does not define init_model_config: external_root={external_root}")
    if not callable(feature_extractor_class):
        raise ImportError(f"EMA-VFI config.py does not expose feature_extractor: external_root={external_root}")
    if not callable(flow_estimation_class):
        raise ImportError(f"EMA-VFI config.py does not expose flow_estimation: external_root={external_root}")

    variant_config = MODEL_VARIANT_CONFIGS[model_variant]
    backbone_config, flow_config = init_model_config(
        F=variant_config.feature_channels,
        W=7,
        depth=list(variant_config.depths),
    )
    return flow_estimation_class(feature_extractor_class(**backbone_config), **flow_config)


def _convert_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
        if "attn_mask" not in key and "HW" not in key
    }


def _load_strict_model_state(model: nn.Module, checkpoint: Any, checkpoint_path: Path) -> None:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"EMA-VFI checkpoint must be a state_dict mapping, got {type(checkpoint).__name__}")

    converted_checkpoint = _convert_state_dict_keys(state_dict=checkpoint)
    try:
        model.load_state_dict(converted_checkpoint, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "EMA-VFI checkpoint is incompatible with the selected official model. "
            f"checkpoint_path={checkpoint_path}"
        ) from error


def _next_multiple(value: int, multiple: int) -> int:
    return ((value - 1) // multiple + 1) * multiple


def _center_replicate_pad_to_size(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    image_height = int(image.shape[-2])
    image_width = int(image.shape[-1])
    pad_height = height - image_height
    pad_width = width - image_width
    padding = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)
    if padding == (0, 0, 0, 0):
        return image
    return F.pad(image, padding, mode="replicate")


def _center_crop_to_size(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    image_height = int(image.shape[-2])
    image_width = int(image.shape[-1])
    top = (image_height - height) // 2
    left = (image_width - width) // 2
    return image[:, :, top : top + height, left : left + width]


def extract_scalar_timestep(embt: torch.Tensor) -> float:
    if not torch.is_tensor(embt):
        raise TypeError(f"EMA-VFI embt must be a torch.Tensor, got {type(embt).__name__}")
    if embt.numel() == 0:
        raise ValueError("EMA-VFI embt must contain at least one timestep value.")
    if embt.dim() == 0:
        return float(embt.detach().cpu().item())

    batch_values = embt.detach().float().reshape(int(embt.shape[0]), -1)
    first_value_per_sample = batch_values[:, 0]
    if not torch.allclose(batch_values, first_value_per_sample[:, None].expand_as(batch_values)):
        raise ValueError(f"EMA-VFI supports one scalar timestep per sample, got embt_shape={tuple(embt.shape)}")
    if not torch.allclose(first_value_per_sample, first_value_per_sample[:1].expand_as(first_value_per_sample)):
        values = first_value_per_sample.detach().cpu().tolist()
        raise ValueError(f"EMA-VFI supports one shared timestep per batch, got per_sample_values={values}")

    return float(first_value_per_sample[0].detach().cpu().item())


class Model(nn.Module):
    def __init__(
        self,
        external_root: str,
        model_variant: str,
        tta: bool,
        fast_tta: bool,
        pad_divisor: int,
    ) -> None:
        super().__init__()
        _validate_model_variant(model_variant=model_variant)
        _validate_bool(value=tta, name="tta")
        _validate_bool(value=fast_tta, name="fast_tta")
        _validate_positive_int(value=pad_divisor, name="pad_divisor")

        self.external_root: Path = _resolve_project_path(path_value=external_root)
        _require_external_files(external_root=self.external_root)
        self.model_variant: str = model_variant
        self.tta: bool = tta
        self.fast_tta: bool = fast_tta
        self.pad_divisor: int = pad_divisor
        self.net: nn.Module | None = None
        self._checkpoint_loaded: bool = False

    def eval(self) -> Model:
        super().eval()
        if self.net is not None:
            self.net.eval()
        return self

    def load_external_checkpoint(self, checkpoint_path: Path, device: Any) -> None:
        torch_device = torch.device(device)
        if torch_device.type != "cuda":
            raise RuntimeError(
                "Official EMA-VFI inference requires CUDA because the official model creates CUDA tensors internally. "
                f"Got device={torch_device}."
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"EMA-VFI checkpoint file is missing: path={checkpoint_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"EMA-VFI checkpoint_path must be a .pkl file, got path={checkpoint_path}")

        net = _build_official_network(external_root=self.external_root, model_variant=self.model_variant)
        checkpoint = torch.load(str(checkpoint_path), map_location=torch_device)
        _load_strict_model_state(model=net, checkpoint=checkpoint, checkpoint_path=checkpoint_path)
        net.to(torch_device)
        net.eval()
        self.net = net
        self._checkpoint_loaded = True

    def _infer_padded(self, imgs: torch.Tensor, timestep: float) -> torch.Tensor:
        if self.net is None:
            raise RuntimeError("EMA-VFI checkpoint is not loaded. Call load_external_checkpoint before inference.")
        if self.fast_tta:
            flipped_imgs = imgs.flip(2).flip(3)
            input_images = torch.cat((imgs, flipped_imgs), dim=0)
            _flow, _mask, _merged, preds = self.net(input_images, timestep=timestep)
            batch_size = int(imgs.shape[0])
            return (preds[:batch_size] + preds[batch_size:].flip(2).flip(3)) / 2.0

        _flow, _mask, _merged, pred = self.net(imgs, timestep=timestep)
        if not self.tta:
            return pred

        _flow2, _mask2, _merged2, pred2 = self.net(imgs.flip(2).flip(3), timestep=timestep)
        return (pred + pred2.flip(2).flip(3)) / 2.0

    def inference(self, img0: Any, img1: Any, embt: Any, scale_factor: float) -> Any:
        if not self._checkpoint_loaded or self.net is None:
            raise RuntimeError("EMA-VFI checkpoint is not loaded. Call load_external_checkpoint before inference.")
        if scale_factor != 1.0:
            raise ValueError(f"EMA-VFI does not support scale_factor={scale_factor}; use scale_factor=1.0.")
        if img0.device.type != "cuda" or img1.device.type != "cuda":
            raise RuntimeError("Official EMA-VFI inference requires CUDA input tensors.")
        if img0.shape[-2:] != img1.shape[-2:]:
            raise ValueError(f"EMA-VFI input frames must share spatial shape, got img0={img0.shape} img1={img1.shape}")

        timestep = extract_scalar_timestep(embt=embt)
        height = int(img0.shape[-2])
        width = int(img0.shape[-1])
        padded_height = _next_multiple(value=height, multiple=self.pad_divisor)
        padded_width = _next_multiple(value=width, multiple=self.pad_divisor)
        padded_img0 = _center_replicate_pad_to_size(image=img0, height=padded_height, width=padded_width)
        padded_img1 = _center_replicate_pad_to_size(image=img1, height=padded_height, width=padded_width)
        padded_prediction = self._infer_padded(imgs=torch.cat((padded_img0, padded_img1), dim=1), timestep=timestep)
        return _center_crop_to_size(image=padded_prediction, height=height, width=width).clamp(0.0, 1.0)


__all__ = ["Model", "EMAVFIExternalFilesError", "extract_scalar_timestep"]
