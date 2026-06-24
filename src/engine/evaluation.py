from __future__ import annotations

import sys
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

LPIPS_NET_CHOICES: tuple[str, ...] = ("alex", "vgg", "squeeze")
FLIP_DYNAMIC_RANGE_CHOICES: tuple[str, ...] = ("LDR", "HDR")
FLIP_TONEMAPPER_CHOICES: tuple[str, ...] = ("ACES", "Hable", "Reinhard")
PSNR_DIV_METRIC_NAME: str = "psnr_div"
FLOLPIPS_METRIC_NAME: str = "flolpips"


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    psnr = -10 * torch.log10(((img1 - img2) * (img1 - img2)).mean())
    return psnr


def build_gaussian_vector(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = [exp(-((value - window_size // 2) ** 2) / float(2 * sigma**2)) for value in range(window_size)]
    vector = torch.tensor(values, device=device, dtype=dtype)
    return vector / vector.sum()


def build_ssim_window(
    window_size: int,
    channel_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    gaussian = build_gaussian_vector(window_size, 1.5, device, dtype).unsqueeze(1)
    window_2d = gaussian.mm(gaussian.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel_count, 1, window_size, window_size).contiguous()


def normalize_metric_batch(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        return image.unsqueeze(0)
    if image.dim() == 4:
        return image

    raise ValueError(f"Expected image tensor with shape C,H,W or B,C,H,W, got shape={tuple(image.shape)}")


def infer_dynamic_range(reference: torch.Tensor) -> float:
    max_value = 255.0 if float(reference.max().detach().cpu().item()) > 128.0 else 1.0
    min_value = -1.0 if float(reference.min().detach().cpu().item()) < -0.5 else 0.0
    return max_value - min_value


def choose_ssim_window_size(height: int, width: int) -> int:
    window_size = min(11, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size <= 0:
        raise ValueError(f"SSIM window size must be positive, got height={height} width={width}")
    return window_size


def calculate_ssim_values(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    image1 = normalize_metric_batch(img1).float()
    image2 = normalize_metric_batch(img2).float()
    if tuple(image1.shape) != tuple(image2.shape):
        raise ValueError(f"SSIM input shapes must match, got img1={tuple(image1.shape)} img2={tuple(image2.shape)}")

    _batch_size, channel_count, height, width = image1.shape
    window_size = choose_ssim_window_size(height, width)
    padding = window_size // 2
    window = build_ssim_window(window_size, channel_count, image1.device, image1.dtype)
    dynamic_range = infer_dynamic_range(image1)

    mu1 = F.conv2d(F.pad(image1, (padding, padding, padding, padding), mode="replicate"), window, groups=channel_count)
    mu2 = F.conv2d(F.pad(image2, (padding, padding, padding, padding), mode="replicate"), window, groups=channel_count)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(F.pad(image1 * image1, (padding, padding, padding, padding), mode="replicate"), window, groups=channel_count) - mu1_sq
    sigma2_sq = F.conv2d(F.pad(image2 * image2, (padding, padding, padding, padding), mode="replicate"), window, groups=channel_count) - mu2_sq
    sigma12 = F.conv2d(F.pad(image1 * image2, (padding, padding, padding, padding), mode="replicate"), window, groups=channel_count) - mu1_mu2

    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator
    return ssim_map.flatten(start_dim=1).mean(dim=1)


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    return calculate_ssim_values(img1, img2).mean()


def parse_metric_bool(raw_config: dict[str, object], key: str, fallback: bool) -> bool:
    if key not in raw_config:
        return fallback

    value = raw_config[key]
    if not isinstance(value, bool):
        raise TypeError(f"metrics.{key} must be a boolean, got {type(value).__name__}")
    return value


def parse_metric_string(raw_config: dict[str, object], key: str, fallback: str) -> str:
    if key not in raw_config:
        return fallback

    value = raw_config[key]
    if not isinstance(value, str):
        raise TypeError(f"metrics.{key} must be a string, got {type(value).__name__}")
    return value


def parse_metric_optional_string(raw_config: dict[str, object], key: str) -> str | None:
    if key not in raw_config or raw_config[key] is None:
        return None

    value = raw_config[key]
    if not isinstance(value, str):
        raise TypeError(f"metrics.{key} must be a string or null, got {type(value).__name__}")
    return value


def parse_metric_float(raw_config: dict[str, object], key: str, fallback: float) -> float:
    if key not in raw_config:
        return fallback

    value = raw_config[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"metrics.{key} must be a number, got {type(value).__name__}")
    return float(value)


def parse_flip_dynamic_range(raw_config: dict[str, object]) -> str:
    dynamic_range = parse_metric_string(raw_config, "flip_dynamic_range", "LDR").upper()
    if dynamic_range not in FLIP_DYNAMIC_RANGE_CHOICES:
        raise ValueError(
            f"metrics.flip_dynamic_range must be one of {FLIP_DYNAMIC_RANGE_CHOICES}, got {dynamic_range}"
        )
    return dynamic_range


def parse_flip_tonemapper(raw_config: dict[str, object]) -> str:
    raw_tonemapper = parse_metric_string(raw_config, "flip_tonemapper", "ACES").lower()
    tonemapper_by_key = {tonemapper.lower(): tonemapper for tonemapper in FLIP_TONEMAPPER_CHOICES}
    if raw_tonemapper not in tonemapper_by_key:
        raise ValueError(
            f"metrics.flip_tonemapper must be one of {FLIP_TONEMAPPER_CHOICES}, got {raw_tonemapper}"
        )
    return tonemapper_by_key[raw_tonemapper]


def read_metric_config(config_values: dict[str, Any]) -> dict[str, object]:
    if "metrics" not in config_values or config_values["metrics"] is None:
        raw_config: dict[str, object] = {}
    elif isinstance(config_values["metrics"], dict):
        raw_config = dict(config_values["metrics"])
    else:
        raise TypeError(f"metrics must be a mapping, got {type(config_values['metrics']).__name__}")

    lpips_net = parse_metric_string(raw_config, "lpips_net", "alex")
    if lpips_net not in LPIPS_NET_CHOICES:
        raise ValueError(f"metrics.lpips_net must be one of {LPIPS_NET_CHOICES}, got {lpips_net}")

    flip_pixels_per_degree = parse_metric_float(raw_config, "flip_pixels_per_degree", 67.0)
    if flip_pixels_per_degree <= 0.0:
        raise ValueError(f"metrics.flip_pixels_per_degree must be positive, got {flip_pixels_per_degree}")

    psnr_divergence_threshold = parse_metric_float(raw_config, "psnr_div_divergence_threshold", 0.01)
    if psnr_divergence_threshold <= 0.0:
        raise ValueError(
            f"metrics.psnr_div_divergence_threshold must be positive, got {psnr_divergence_threshold}"
        )

    return {
        "enable_psnr": parse_metric_bool(raw_config, "enable_psnr", True),
        "enable_ssim": parse_metric_bool(raw_config, "enable_ssim", True),
        "enable_lpips": parse_metric_bool(raw_config, "enable_lpips", False),
        "enable_flip": parse_metric_bool(raw_config, "enable_flip", False),
        "enable_psnr_div": parse_metric_bool(raw_config, "enable_psnr_div", False),
        "enable_flolpips": parse_metric_bool(raw_config, "enable_flolpips", False),
        "lpips_net": lpips_net,
        "flip_dynamic_range": parse_flip_dynamic_range(raw_config),
        "flip_pixels_per_degree": flip_pixels_per_degree,
        "flip_tonemapper": parse_flip_tonemapper(raw_config),
        "psnr_div_divergence_threshold": psnr_divergence_threshold,
        "flolpips_repo_path": parse_metric_optional_string(raw_config, "flolpips_repo_path"),
    }


def require_psnr_enabled(metric_config: dict[str, object], pipeline_name: str) -> None:
    if bool(metric_config["enable_psnr"]):
        return

    raise ValueError(
        f"{pipeline_name} requires metrics.enable_psnr=true because checkpoint selection and top-k sample selection use PSNR."
    )


def require_psnr_div_disabled(metric_config: dict[str, object], pipeline_name: str) -> None:
    if not bool(metric_config["enable_psnr_div"]):
        return

    raise ValueError(
        f"{pipeline_name} does not support metrics.enable_psnr_div=true. "
        "PSNR-DIV is a temporal sequence metric and must be computed on ordered inference clips."
    )


def require_flolpips_disabled(metric_config: dict[str, object], pipeline_name: str) -> None:
    if not bool(metric_config["enable_flolpips"]):
        return

    raise ValueError(
        f"{pipeline_name} does not support metrics.enable_flolpips=true. "
        "FloLPIPS requires image triplets: previous frame, interpolated prediction, and next frame."
    )


def get_enabled_metric_names(metric_config: dict[str, object]) -> tuple[str, ...]:
    metric_names = []
    if bool(metric_config["enable_psnr"]):
        metric_names.append("psnr")
    if bool(metric_config["enable_ssim"]):
        metric_names.append("ssim")
    if bool(metric_config["enable_lpips"]):
        metric_names.append("lpips")
    if bool(metric_config["enable_flip"]):
        metric_names.append("flip")
    if bool(metric_config["enable_psnr_div"]):
        metric_names.append(PSNR_DIV_METRIC_NAME)
    if bool(metric_config["enable_flolpips"]):
        metric_names.append(FLOLPIPS_METRIC_NAME)
    return tuple(metric_names)


def build_metric_meters(metric_config: dict[str, object]) -> dict[str, "AverageMeter"]:
    return {metric_name: AverageMeter() for metric_name in get_enabled_metric_names(metric_config)}


def build_lpips_model(metric_config: dict[str, object], device: torch.device) -> Any | None:
    if not bool(metric_config["enable_lpips"]):
        return None

    try:
        import lpips
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "LPIPS metric is enabled, but the 'lpips' package is not installed in this environment. "
            "Install it in the project conda environment first, then rerun with metrics.enable_lpips=true."
        ) from error

    model = lpips.LPIPS(net=str(metric_config["lpips_net"])).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def import_flip_evaluator() -> Any:
    try:
        import flip_evaluator
    except ImportError as error:
        raise ImportError(
            "NVIDIA FLIP metric is enabled, but the 'flip-evaluator' package is not installed or cannot be imported. "
            "Install it in the project environment first, then rerun with metrics.enable_flip=true."
        ) from error

    return flip_evaluator


def build_flip_evaluator(metric_config: dict[str, object]) -> Any | None:
    if not bool(metric_config["enable_flip"]):
        return None

    return import_flip_evaluator()


def import_flolpips_module(metric_config: dict[str, object]) -> Any:
    repo_path = metric_config["flolpips_repo_path"]
    if repo_path is not None:
        resolved_path = Path(str(repo_path)).expanduser().resolve()
        if not resolved_path.is_dir():
            raise FileNotFoundError(f"metrics.flolpips_repo_path does not exist or is not a directory: {resolved_path}")
        resolved_path_text = str(resolved_path)
        if resolved_path_text not in sys.path:
            sys.path.insert(0, resolved_path_text)

    try:
        import flolpips
    except ImportError as error:
        raise ImportError(
            "FloLPIPS metric is enabled, but the official FloLPIPS module could not be imported. "
            "Clone https://github.com/danier97/FloLPIPS and set metrics.flolpips_repo_path to that directory, "
            "or make its flolpips.py importable through PYTHONPATH."
        ) from error

    return flolpips


def build_flolpips_model(metric_config: dict[str, object], device: torch.device) -> Any | None:
    if not bool(metric_config["enable_flolpips"]):
        return None
    if device.type != "cuda":
        raise RuntimeError(
            "metrics.enable_flolpips=true requires a CUDA device because the official FloLPIPS PWCNet "
            "implementation uses CUDA/CuPy correlation kernels."
        )

    flolpips_module = import_flolpips_module(metric_config=metric_config)
    try:
        model = flolpips_module.Flolpips().to(device)
    except Exception as error:
        raise RuntimeError(
            "Failed to initialize FloLPIPS. Ensure the official FloLPIPS dependencies are installed "
            "(torchvision, cupy for your CUDA version, opencv-python) and that PWCNet weights can be loaded."
        ) from error

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def calculate_psnr_batch(target: torch.Tensor, prediction: torch.Tensor) -> list[float]:
    batch_target = normalize_metric_batch(target)
    batch_prediction = normalize_metric_batch(prediction)
    if tuple(batch_target.shape) != tuple(batch_prediction.shape):
        raise ValueError(f"PSNR input shapes must match, got target={tuple(batch_target.shape)} prediction={tuple(batch_prediction.shape)}")

    values: list[float] = []
    for batch_index in range(int(batch_prediction.shape[0])):
        value = float(calculate_psnr(batch_target[batch_index], batch_prediction[batch_index]).detach().cpu().item())
        values.append(value)
    return values


def calculate_ssim_batch(target: torch.Tensor, prediction: torch.Tensor) -> list[float]:
    values = calculate_ssim_values(target, prediction).detach().cpu().tolist()
    return [float(value) for value in values]


def normalize_lpips_batch(image: torch.Tensor) -> torch.Tensor:
    return normalize_metric_batch(image).detach().float().clamp(0.0, 1.0) * 2.0 - 1.0


def calculate_lpips_batch(target: torch.Tensor, prediction: torch.Tensor, lpips_model: Any) -> list[float]:
    normalized_target = normalize_lpips_batch(target)
    normalized_prediction = normalize_lpips_batch(prediction)
    values = lpips_model(normalized_prediction, normalized_target)
    return [float(value) for value in values.detach().cpu().reshape(-1).tolist()]


def normalize_flolpips_batch(image: torch.Tensor) -> torch.Tensor:
    batch = normalize_metric_batch(image).detach().float()
    if int(batch.shape[1]) != 3:
        raise ValueError(f"FloLPIPS expects RGB images with 3 channels, got shape={tuple(batch.shape)}")
    return batch.clamp(0.0, 1.0)


def calculate_flolpips_batch(
    img0: torch.Tensor,
    img1: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    flolpips_model: Any,
) -> list[float]:
    normalized_img0 = normalize_flolpips_batch(img0)
    normalized_img1 = normalize_flolpips_batch(img1)
    normalized_target = normalize_flolpips_batch(target)
    normalized_prediction = normalize_flolpips_batch(prediction)
    if tuple(normalized_img0.shape) != tuple(normalized_img1.shape):
        raise ValueError(
            f"FloLPIPS endpoint shapes must match, got img0={tuple(normalized_img0.shape)} "
            f"img1={tuple(normalized_img1.shape)}"
        )
    if tuple(normalized_target.shape) != tuple(normalized_prediction.shape):
        raise ValueError(
            f"FloLPIPS target and prediction shapes must match, got target={tuple(normalized_target.shape)} "
            f"prediction={tuple(normalized_prediction.shape)}"
        )
    if tuple(normalized_img0.shape) != tuple(normalized_target.shape):
        raise ValueError(
            f"FloLPIPS endpoint and target shapes must match, got endpoint={tuple(normalized_img0.shape)} "
            f"target={tuple(normalized_target.shape)}"
        )

    values = flolpips_model(
        normalized_img0,
        normalized_img1,
        normalized_prediction,
        normalized_target,
    )
    return [float(value) for value in values.detach().cpu().reshape(-1).tolist()]


def normalize_flip_batch(image: torch.Tensor, dynamic_range: str) -> np.ndarray:
    batch = normalize_metric_batch(image).detach().float()
    if int(batch.shape[1]) != 3:
        raise ValueError(f"NVIDIA FLIP expects RGB images with 3 channels, got shape={tuple(batch.shape)}")

    if dynamic_range == "LDR":
        batch = batch.clamp(0.0, 1.0)
    elif dynamic_range == "HDR":
        batch = batch.clamp_min(0.0)
    else:
        raise ValueError(f"Unsupported NVIDIA FLIP dynamic range: {dynamic_range}")

    return batch.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32, copy=False)


def build_flip_parameters(metric_config: dict[str, object]) -> dict[str, object]:
    parameters: dict[str, object] = {
        "ppd": float(metric_config["flip_pixels_per_degree"]),
    }
    if str(metric_config["flip_dynamic_range"]) == "HDR":
        parameters["tonemapper"] = str(metric_config["flip_tonemapper"])
    return parameters


def calculate_flip_batch(
    target: torch.Tensor,
    prediction: torch.Tensor,
    metric_config: dict[str, object],
) -> list[float]:
    dynamic_range = str(metric_config["flip_dynamic_range"])
    batch_target = normalize_flip_batch(target, dynamic_range)
    batch_prediction = normalize_flip_batch(prediction, dynamic_range)
    if tuple(batch_target.shape) != tuple(batch_prediction.shape):
        raise ValueError(
            f"NVIDIA FLIP input shapes must match, got target={tuple(batch_target.shape)} "
            f"prediction={tuple(batch_prediction.shape)}"
        )

    flip_evaluator = import_flip_evaluator()
    parameters = build_flip_parameters(metric_config)
    values: list[float] = []
    for batch_index in range(int(batch_prediction.shape[0])):
        try:
            _error_map, mean_flip_error, _used_parameters = flip_evaluator.evaluate(
                batch_target[batch_index],
                batch_prediction[batch_index],
                dynamic_range,
                inputsRGB=True,
                applyMagma=False,
                computeMeanError=True,
                parameters=dict(parameters),
            )
        except SystemExit as error:
            raise RuntimeError(
                f"NVIDIA FLIP evaluator rejected inputs or parameters: dynamic_range={dynamic_range} "
                f"parameters={parameters}"
            ) from error

        values.append(float(mean_flip_error))
    return values


def tensor_to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    batch = normalize_metric_batch(image).detach().float()
    if int(batch.shape[0]) != 1:
        raise ValueError(f"Expected a single image for PSNR-DIV conversion, got shape={tuple(batch.shape)}")
    if int(batch.shape[1]) != 3:
        raise ValueError(f"PSNR-DIV expects RGB images with 3 channels, got shape={tuple(batch.shape)}")

    image_rgb = batch[0].clamp(0.0, 1.0).permute(1, 2, 0).contiguous().cpu().numpy()
    return np.round(image_rgb * 255.0).astype(np.uint8)


def convert_rgb_to_y_bt709(rgb_image: np.ndarray) -> np.ndarray:
    red = rgb_image[:, :, 0].astype(np.float32)
    green = rgb_image[:, :, 1].astype(np.float32)
    blue = rgb_image[:, :, 2].astype(np.float32)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return luminance.astype(np.uint8)


def prepare_psnr_div_sample(target: torch.Tensor, prediction: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_rgb = tensor_to_uint8_rgb(target)
    prediction_rgb = tensor_to_uint8_rgb(prediction)
    return convert_rgb_to_y_bt709(target_rgb), convert_rgb_to_y_bt709(prediction_rgb), prediction_rgb


def calculate_psnr_divergence(motion_field: np.ndarray) -> np.ndarray:
    horizontal_motion = motion_field[..., 0]
    vertical_motion = motion_field[..., 1]
    horizontal_axis0_gradient, _horizontal_axis1_gradient = np.gradient(horizontal_motion)
    _vertical_axis0_gradient, vertical_axis1_gradient = np.gradient(vertical_motion)
    divergence = np.abs(horizontal_axis0_gradient + vertical_axis1_gradient)
    max_divergence = float(np.max(divergence))
    if max_divergence <= 0.0:
        return np.zeros_like(divergence, dtype=np.float32)
    return (divergence / max_divergence).astype(np.float32)


def compute_psnr_div_farneback_flow(previous_rgb: np.ndarray, next_rgb: np.ndarray) -> np.ndarray:
    import cv2

    previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        previous_gray,
        next_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )


def calculate_psnr_div_value(
    target_y: np.ndarray,
    prediction_y: np.ndarray,
    motion_field: np.ndarray,
    divergence_threshold: float,
) -> float:
    if tuple(target_y.shape) != tuple(prediction_y.shape):
        raise ValueError(
            f"PSNR-DIV luminance input shapes must match, got target={tuple(target_y.shape)} "
            f"prediction={tuple(prediction_y.shape)}"
        )
    if tuple(motion_field.shape[:2]) != tuple(target_y.shape):
        raise ValueError(
            f"PSNR-DIV motion field shape must match image shape, got motion={tuple(motion_field.shape)} "
            f"image={tuple(target_y.shape)}"
        )

    divergence = calculate_psnr_divergence(motion_field=motion_field)
    mask = divergence > divergence_threshold
    if int(mask.sum()) == 0:
        return float("nan")

    error = target_y.astype(np.float32) - prediction_y.astype(np.float32)
    weighted_mse = float(np.mean((error[mask]) ** 2))
    if weighted_mse <= 0.0:
        return float("nan")
    return float(20.0 * np.log10(255.0 / np.sqrt(weighted_mse)))


def calculate_psnr_div_sample(
    target_y: np.ndarray,
    prediction_y: np.ndarray,
    flow_start_prediction_rgb: np.ndarray,
    flow_end_prediction_rgb: np.ndarray,
    divergence_threshold: float,
) -> float:
    motion_field = compute_psnr_div_farneback_flow(
        previous_rgb=flow_start_prediction_rgb,
        next_rgb=flow_end_prediction_rgb,
    )
    return calculate_psnr_div_value(
        target_y=target_y,
        prediction_y=prediction_y,
        motion_field=motion_field,
        divergence_threshold=divergence_threshold,
    )


def calculate_batch_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    metric_config: dict[str, object],
    lpips_model: Any | None,
    flolpips_model: Any | None,
    img0: torch.Tensor | None,
    img1: torch.Tensor | None,
) -> dict[str, list[float]]:
    batch_values: dict[str, list[float]] = {}

    if bool(metric_config["enable_psnr"]):
        batch_values["psnr"] = calculate_psnr_batch(target, prediction)

    if bool(metric_config["enable_ssim"]):
        batch_values["ssim"] = calculate_ssim_batch(target, prediction)

    if bool(metric_config["enable_lpips"]):
        if lpips_model is None:
            raise RuntimeError("metrics.enable_lpips=true requires a loaded LPIPS model.")
        batch_values["lpips"] = calculate_lpips_batch(target, prediction, lpips_model)

    if bool(metric_config["enable_flolpips"]):
        if flolpips_model is None:
            raise RuntimeError("metrics.enable_flolpips=true requires a loaded FloLPIPS model.")
        if img0 is None or img1 is None:
            raise RuntimeError("metrics.enable_flolpips=true requires img0 and img1 endpoint tensors.")
        batch_values[FLOLPIPS_METRIC_NAME] = calculate_flolpips_batch(
            img0=img0,
            img1=img1,
            target=target,
            prediction=prediction,
            flolpips_model=flolpips_model,
        )

    if bool(metric_config["enable_flip"]):
        batch_values["flip"] = calculate_flip_batch(target, prediction, metric_config)

    if len(batch_values) == 0:
        raise ValueError("At least one metric must be enabled.")

    expected_count = len(next(iter(batch_values.values())))
    for metric_name, values in batch_values.items():
        if len(values) != expected_count:
            raise ValueError(f"Metric {metric_name} returned {len(values)} values, expected {expected_count}")
    return batch_values


def average_metric_values(metric_meters: dict[str, "AverageMeter"]) -> dict[str, float]:
    return {metric_name: float(metric_meter.avg) for metric_name, metric_meter in metric_meters.items()}


def format_metric_averages(metric_meters: dict[str, "AverageMeter"]) -> str:
    return " ".join(f"{metric_name}={metric_meter.avg:.6f}" for metric_name, metric_meter in metric_meters.items())


def format_metric_values(metric_values: dict[str, float]) -> str:
    return " ".join(f"{metric_name}={metric_value:.6f}" for metric_name, metric_value in metric_values.items())


@dataclass
class AverageMeter:
    sum: float = 0.0
    count: int = 0

    def update(self, value: float, n: int) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0

        return self.sum / self.count
