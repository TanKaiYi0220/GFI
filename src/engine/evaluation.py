from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

import torch
import torch.nn.functional as F

LPIPS_NET_CHOICES: tuple[str, ...] = ("alex", "vgg", "squeeze")


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

    return {
        "enable_psnr": parse_metric_bool(raw_config, "enable_psnr", True),
        "enable_ssim": parse_metric_bool(raw_config, "enable_ssim", True),
        "enable_lpips": parse_metric_bool(raw_config, "enable_lpips", False),
        "lpips_net": lpips_net,
    }


def require_psnr_enabled(metric_config: dict[str, object], pipeline_name: str) -> None:
    if bool(metric_config["enable_psnr"]):
        return

    raise ValueError(
        f"{pipeline_name} requires metrics.enable_psnr=true because checkpoint selection and top-k sample selection use PSNR."
    )


def get_enabled_metric_names(metric_config: dict[str, object]) -> tuple[str, ...]:
    metric_names = []
    if bool(metric_config["enable_psnr"]):
        metric_names.append("psnr")
    if bool(metric_config["enable_ssim"]):
        metric_names.append("ssim")
    if bool(metric_config["enable_lpips"]):
        metric_names.append("lpips")
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


def calculate_batch_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    metric_config: dict[str, object],
    lpips_model: Any | None,
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
