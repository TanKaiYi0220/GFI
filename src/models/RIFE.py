from __future__ import annotations

import inspect
import importlib.util
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
    Path("model/warplayer.py"),
    Path("train_log/IFNet_HDv3.py"),
    Path("train_log/RIFE_HDv3.py"),
)
CHECKPOINT_FILENAME: str = "flownet.pkl"
PAD_MULTIPLE: int = 32
DISTILLATION_LOSS_WEIGHT: float = 0.01


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


def _convert_wrapper_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def _uses_wrapper_state_dict_keys(state_dict: dict[str, Any]) -> bool:
    return any(key.replace("module.", "", 1).startswith("flownet.") for key in state_dict)


def _next_multiple(value: int, multiple: int) -> int:
    return ((value - 1) // multiple + 1) * multiple


def _resolve_pad_multiple(scale_factor: float) -> int:
    if scale_factor <= 0.0:
        raise ValueError(f"RIFE scale_factor must be positive, got {scale_factor}")
    return max(PAD_MULTIPLE, int(PAD_MULTIPLE / scale_factor))


def _pad_to_size(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    image_height = int(image.shape[-2])
    image_width = int(image.shape[-1])
    padding = (0, width - image_width, 0, height - image_height)
    if padding == (0, 0, 0, 0):
        return image
    return F.pad(image, padding)


def _build_laplacian_kernel(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernel_values = [
        [1.0, 4.0, 6.0, 4.0, 1.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [6.0, 24.0, 36.0, 24.0, 6.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [1.0, 4.0, 6.0, 4.0, 1.0],
    ]
    kernel = torch.tensor(kernel_values, device=device, dtype=dtype) / 256.0
    return kernel.reshape(1, 1, 5, 5).repeat(channels, 1, 1, 1)


def _laplacian_downsample(image: torch.Tensor) -> torch.Tensor:
    return image[:, :, ::2, ::2]


def _laplacian_upsample(image: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(image)
    width_expanded = torch.stack((image, zeros), dim=-1).flatten(start_dim=-2)
    height_expanded = torch.stack(
        (width_expanded, torch.zeros_like(width_expanded)),
        dim=-2,
    ).flatten(start_dim=-3, end_dim=-2)
    kernel = 4.0 * _build_laplacian_kernel(
        channels=int(image.shape[1]),
        device=image.device,
        dtype=image.dtype,
    )
    return F.conv2d(F.pad(height_expanded, (2, 2, 2, 2), mode="reflect"), kernel, groups=int(image.shape[1]))


def _laplacian_pyramid(image: torch.Tensor, max_levels: int) -> list[torch.Tensor]:
    current = image
    pyramid = []
    kernel = _build_laplacian_kernel(
        channels=int(image.shape[1]),
        device=image.device,
        dtype=image.dtype,
    )
    for _level in range(max_levels):
        filtered = F.conv2d(F.pad(current, (2, 2, 2, 2), mode="reflect"), kernel, groups=int(image.shape[1]))
        downsampled = _laplacian_downsample(image=filtered)
        upsampled = _laplacian_upsample(image=downsampled)
        upsampled = upsampled[:, :, : current.shape[-2], : current.shape[-1]]
        pyramid.append(current - upsampled)
        current = downsampled
    return pyramid


class LapLoss(nn.Module):
    def __init__(self, max_levels: int, channels: int) -> None:
        super().__init__()
        self.max_levels: int = max_levels
        self.channels: int = channels

    def forward(self, input_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        if input_image.shape != target_image.shape:
            raise ValueError(
                "LapLoss input and target shapes must match, "
                f"got input_shape={tuple(input_image.shape)} target_shape={tuple(target_image.shape)}"
            )
        if int(input_image.shape[1]) != self.channels:
            raise ValueError(
                "LapLoss channel count mismatch, "
                f"expected={self.channels} actual={int(input_image.shape[1])}"
            )

        input_pyramid = _laplacian_pyramid(image=input_image, max_levels=self.max_levels)
        target_pyramid = _laplacian_pyramid(image=target_image, max_levels=self.max_levels)
        loss = input_image.new_zeros(())
        for input_level, target_level in zip(input_pyramid, target_pyramid):
            loss = loss + F.l1_loss(input_level, target_level)
        return loss


def _select_merged_prediction(merged: Any) -> torch.Tensor:
    if not isinstance(merged, (list, tuple)):
        raise TypeError(f"RIFE flownet merged output must be a list or tuple, got {type(merged).__name__}")
    if len(merged) == 0:
        raise ValueError("RIFE flownet merged output must contain at least one prediction.")
    if len(merged) > 3:
        return merged[3]
    return merged[-1]


def _extract_prediction_teacher_and_distillation_loss(output: Any) -> tuple[torch.Tensor, torch.Tensor | None, Any | None]:
    if not isinstance(output, (list, tuple)):
        raise TypeError(f"RIFE flownet output must be a tuple or list, got {type(output).__name__}")
    if len(output) == 3:
        _flow, _mask, merged = output
        return _select_merged_prediction(merged=merged), None, None
    if len(output) == 6:
        _flow, _mask, merged, _flow_teacher, merged_teacher, loss_distill = output
        if merged_teacher is not None and not torch.is_tensor(merged_teacher):
            raise TypeError(
                "RIFE flownet merged_teacher output must be a torch.Tensor or None, "
                f"got {type(merged_teacher).__name__}"
            )
        return _select_merged_prediction(merged=merged), merged_teacher, loss_distill
    raise RuntimeError(f"Unsupported RIFE flownet output length={len(output)}.")


def _scale_optional_loss(reference_loss: torch.Tensor, loss_value: Any | None, weight: float) -> torch.Tensor:
    if loss_value is None:
        return reference_loss.new_zeros(())
    if torch.is_tensor(loss_value):
        return weight * loss_value
    return reference_loss.new_tensor(weight * float(loss_value))


def _get_flownet_forward_parameter_names(flownet: nn.Module) -> tuple[str, ...]:
    signature = inspect.signature(flownet.forward)
    return tuple(signature.parameters)


def _call_flownet_inference(
    flownet: nn.Module,
    imgs: torch.Tensor,
    timestep: float,
    scale_list: list[float],
) -> Any:
    parameter_names = _get_flownet_forward_parameter_names(flownet=flownet)
    if "scale" in parameter_names:
        if "timestep" in parameter_names:
            return flownet(imgs, scale=scale_list, timestep=timestep)
        return flownet(imgs, scale=scale_list)
    if "scale_list" in parameter_names:
        if "timestep" in parameter_names:
            return flownet(imgs, scale_list=scale_list, timestep=timestep)
        return flownet(imgs, scale_list=scale_list)
    raise TypeError(f"Unsupported RIFE flownet forward signature: parameters={parameter_names}")


def _call_flownet_training(
    flownet: nn.Module,
    imgs: torch.Tensor,
    timestep: float,
    scale_list: list[float],
) -> Any:
    parameter_names = _get_flownet_forward_parameter_names(flownet=flownet)
    if "scale" in parameter_names:
        if "timestep" in parameter_names:
            return flownet(imgs, scale=scale_list, timestep=timestep)
        if "training" in parameter_names:
            return flownet(imgs, scale=scale_list, training=True)
        return flownet(imgs, scale=scale_list)
    if "scale_list" in parameter_names:
        if "timestep" in parameter_names:
            return flownet(imgs, scale_list=scale_list, timestep=timestep)
        if "training" in parameter_names:
            return flownet(imgs, scale_list=scale_list, training=True)
        return flownet(imgs, scale_list=scale_list)
    raise TypeError(f"Unsupported RIFE flownet forward signature: parameters={parameter_names}")


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
        self.reconstruction_loss: nn.Module = LapLoss(max_levels=5, channels=3)
        self._checkpoint_loaded: bool = False

    def load_state_dict(self, state_dict: Any, strict: bool = True) -> Any:
        result = super().load_state_dict(state_dict, strict=strict)
        self._checkpoint_loaded = True
        return result

    def load_external_checkpoint(self, checkpoint_path: Path, device: Any) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"RIFE checkpoint path is missing: path={checkpoint_path}")
        if checkpoint_path.is_dir():
            flownet_path = checkpoint_path / CHECKPOINT_FILENAME
            if not flownet_path.is_file():
                raise FileNotFoundError(
                    "RIFE checkpoint file is missing. "
                    f"Expected official checkpoint at {flownet_path}"
                )
            state_dict = torch.load(str(flownet_path), map_location=device)
            self._load_flownet_state_dict(state_dict=state_dict, checkpoint_path=flownet_path, device=device)
            return

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"RIFE checkpoint_path must be a file or directory, got path={checkpoint_path}")

        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            if not isinstance(checkpoint["model"], dict):
                raise TypeError(
                    "RIFE repo-native checkpoint 'model' value must be a state_dict mapping, "
                    f"got {type(checkpoint['model']).__name__}: checkpoint_path={checkpoint_path}"
                )
            wrapper_state_dict = _convert_wrapper_state_dict_keys(state_dict=checkpoint["model"])
            try:
                self.load_state_dict(wrapper_state_dict, strict=True)
            except RuntimeError as error:
                raise RuntimeError(
                    "RIFE repo-native checkpoint is incompatible with src.models.RIFE.Model. "
                    f"checkpoint_path={checkpoint_path}"
                ) from error
            self.to(device)
            self._checkpoint_loaded = True
            return

        if not isinstance(checkpoint, dict):
            raise TypeError(f"RIFE checkpoint must be a state_dict mapping, got {type(checkpoint).__name__}")
        if _uses_wrapper_state_dict_keys(state_dict=checkpoint):
            wrapper_state_dict = _convert_wrapper_state_dict_keys(state_dict=checkpoint)
            try:
                self.load_state_dict(wrapper_state_dict, strict=True)
            except RuntimeError as error:
                raise RuntimeError(
                    "RIFE wrapper state_dict is incompatible with src.models.RIFE.Model. "
                    f"checkpoint_path={checkpoint_path}"
                ) from error
            self.to(device)
            self._checkpoint_loaded = True
            return

        self._load_flownet_state_dict(state_dict=checkpoint, checkpoint_path=checkpoint_path, device=device)

    def _load_flownet_state_dict(self, state_dict: Any, checkpoint_path: Path, device: Any) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError(f"RIFE flownet checkpoint must be a state_dict mapping, got {type(state_dict).__name__}")

        converted_state_dict = _convert_state_dict_keys(state_dict=state_dict)
        try:
            self.flownet.load_state_dict(converted_state_dict, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "RIFE checkpoint is incompatible with train_log.IFNet_HDv3.IFNet. "
                f"checkpoint_path={checkpoint_path}"
            ) from error
        self.flownet.to(device)
        self._checkpoint_loaded = True

    def _run_flownet_inference(self, img0: torch.Tensor, img1: torch.Tensor, timestep: float, scale_factor: float) -> Any:
        imgs = torch.cat((img0, img1), dim=1)
        scale_list = [8.0 / scale_factor, 4.0 / scale_factor, 2.0 / scale_factor, 1.0 / scale_factor]
        return _call_flownet_inference(
            flownet=self.flownet,
            imgs=imgs,
            timestep=timestep,
            scale_list=scale_list,
        )

    def _run_flownet_training(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        imgt: torch.Tensor,
        timestep: float,
        scale_factor: float,
    ) -> Any:
        imgs = torch.cat((img0, img1, imgt), dim=1)
        scale_list = [8.0 / scale_factor, 4.0 / scale_factor, 2.0 / scale_factor, 1.0 / scale_factor]
        return _call_flownet_training(
            flownet=self.flownet,
            imgs=imgs,
            timestep=timestep,
            scale_list=scale_list,
        )

    def inference(self, img0: Any, img1: Any, embt: Any, scale_factor: float) -> Any:
        if not self._checkpoint_loaded:
            raise RuntimeError("RIFE checkpoint is not loaded. Call load_external_checkpoint before inference.")
        if img0.shape[-2:] != img1.shape[-2:]:
            raise ValueError(f"RIFE input frames must share spatial shape, got img0={img0.shape} img1={img1.shape}")
        timestep = extract_scalar_timestep(embt=embt)
        height = int(img0.shape[-2])
        width = int(img0.shape[-1])
        pad_multiple = _resolve_pad_multiple(scale_factor=scale_factor)
        padded_height = _next_multiple(value=height, multiple=pad_multiple)
        padded_width = _next_multiple(value=width, multiple=pad_multiple)
        padded_img0 = _pad_to_size(image=img0, height=padded_height, width=padded_width)
        padded_img1 = _pad_to_size(image=img1, height=padded_height, width=padded_width)
        output = self._run_flownet_inference(
            img0=padded_img0,
            img1=padded_img1,
            timestep=timestep,
            scale_factor=scale_factor,
        )
        prediction, _teacher_prediction, _loss_distill = _extract_prediction_teacher_and_distillation_loss(output=output)
        return prediction[:, :, :height, :width].clamp(0.0, 1.0)

    def forward(self, img0: Any, img1: Any, embt: Any, imgt: Any) -> Any:
        if img0.shape[-2:] != img1.shape[-2:] or img0.shape[-2:] != imgt.shape[-2:]:
            raise ValueError(
                "RIFE training frames must share spatial shape, "
                f"got img0={img0.shape} img1={img1.shape} imgt={imgt.shape}"
            )
        timestep = extract_scalar_timestep(embt=embt)
        height = int(img0.shape[-2])
        width = int(img0.shape[-1])
        padded_height = _next_multiple(value=height, multiple=PAD_MULTIPLE)
        padded_width = _next_multiple(value=width, multiple=PAD_MULTIPLE)
        padded_img0 = _pad_to_size(image=img0, height=padded_height, width=padded_width)
        padded_img1 = _pad_to_size(image=img1, height=padded_height, width=padded_width)
        padded_imgt = _pad_to_size(image=imgt, height=padded_height, width=padded_width)
        output = self._run_flownet_training(
            img0=padded_img0,
            img1=padded_img1,
            imgt=padded_imgt,
            timestep=timestep,
            scale_factor=1.0,
        )
        padded_prediction, padded_teacher_prediction, loss_distill = _extract_prediction_teacher_and_distillation_loss(
            output=output
        )
        imgt_pred = padded_prediction[:, :, :height, :width].clamp(0.0, 1.0)
        loss_l1 = self.reconstruction_loss(imgt_pred, imgt)
        if padded_teacher_prediction is None:
            loss_tea = loss_l1.new_zeros(())
        else:
            teacher_prediction = padded_teacher_prediction[:, :, :height, :width].clamp(0.0, 1.0)
            loss_tea = self.reconstruction_loss(teacher_prediction, imgt)
        loss_rec = loss_l1 + loss_tea
        loss_geo = loss_rec.new_zeros(())
        loss_dis = _scale_optional_loss(
            reference_loss=loss_rec,
            loss_value=loss_distill,
            weight=DISTILLATION_LOSS_WEIGHT,
        )
        return imgt_pred, loss_rec, loss_geo, loss_dis, None, None, None


__all__ = ["Model", "RIFEExternalFilesError", "extract_scalar_timestep"]
