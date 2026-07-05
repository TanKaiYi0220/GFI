from __future__ import annotations

import importlib.util
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
            "Official RIFE repository is missing. "
            f"Expected clone at {external_root}. "
            "Clone https://github.com/hzwer/ECCV2022-RIFE into src/models/external/RIFE "
            "and place the official HD model package under train_log."
        )
    if not external_root.is_dir():
        raise NotADirectoryError(f"Official RIFE path must be a directory: path={external_root}")

    missing_files = [
        str(external_root / relative_path)
        for relative_path in REQUIRED_EXTERNAL_FILES
        if not (external_root / relative_path).is_file()
    ]
    if len(missing_files) > 0:
        raise RIFEExternalFilesError(
            "Official RIFE HD model-package files are missing. "
            "Download the official HD model package, unzip it under src/models/external/RIFE/train_log, "
            f"and ensure these files exist: missing_files={missing_files}"
        )


def _load_ifnet_class(external_root: Path) -> type[nn.Module]:
    _require_external_files(external_root=external_root)
    module_path = external_root / "train_log" / "IFNet_HDv3.py"
    module_name = f"_gfi_rife_ifnet_{abs(hash(str(module_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load train_log.IFNet_HDv3 from {module_path}")

    module = importlib.util.module_from_spec(spec)
    with _temporary_sys_path(path=external_root), _temporary_external_modules(prefixes=("model", "train_log")):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

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


__all__ = ["Model", "RIFEExternalFilesError", "extract_scalar_timestep"]
