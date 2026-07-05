from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.dataset_config import ACTIVE_DATASET_ROOT_KEY
from src.engine.evaluation import read_metric_config
from src.engine.evaluation import require_psnr_div_disabled
from src.engine.evaluation import require_psnr_enabled
from src.engine.evaluation import require_vfips_disabled
from src.engine.flow_approx import DEFAULT_SPLATTING_FILL_STRATEGY
from src.engine.flow_approx import FLOW_APPROX_METHOD_CHOICES
from src.engine.flow_approx import SPLATTING_FILL_STRATEGIES
from src.engine.flow_approx import is_splatting_flow_approx_method
from src.engine.flow_approx import resolve_splatting_fill_strategy
from src.engine.model_registry import TRAIN_MODEL_NAMES
from src.engine.model_registry import uses_flow_approx_model
from src.engine.model_registry import uses_image_only_vfi_model
from src.utils.config import load_yaml_file

DEFAULT_INIT_FLOW_DOWNSCALE_STRATEGY: str = "bilinear"
DEFAULT_INIT_FLOW_MASK_EPSILON: float = 1e-6
INIT_FLOW_DOWNSCALE_STRATEGIES: tuple[str, ...] = ("bilinear", "masked_area")


@dataclass(frozen=True)
class ModelRunConfig:
    model_name: str
    model_init_args: dict[str, Any]
    eval_convex_upsampling: bool | None


@dataclass(frozen=True)
class FlowApproxConfig:
    method: str
    splatting_fill_strategy: str
    effective_splatting_fill_strategy: str
    init_flow_downscale_strategy: str
    effective_init_flow_downscale_strategy: str
    init_flow_mask_epsilon: float


@dataclass(frozen=True)
class MetricRunConfig:
    values: dict[str, object]


@dataclass(frozen=True)
class TrainRunConfig:
    mode: str
    model: ModelRunConfig
    flow_approx: FlowApproxConfig
    metrics: MetricRunConfig
    root_dir: Path
    dataset_root_dir: str
    paths_config: str | None
    train_preset: str
    test_preset: str
    epochs: int
    resume_path: str | None
    pretrained_checkpoint_path: str | None
    eval_interval: int
    lr_start: float
    lr_end: float
    seed: int
    batch_size: int
    output_dir: str
    only_fps: int
    input_fps: int
    sample_train_frames: list[str]
    sample_test_frames: list[str]
    sample_interval_epoch: int
    input_config: dict[str, Any]


@dataclass(frozen=True)
class InferenceRunConfig:
    mode: str
    model: ModelRunConfig
    flow_approx: FlowApproxConfig
    metrics: MetricRunConfig
    inference_presets: list[str]
    root_dir: Path
    dataset_root_dir: Path
    checkpoint_path: Path
    output_dir: Path
    seed: int
    batch_size: int
    only_fps: int
    input_fps: int
    scale_factor: float
    flow_diff_threshold: float
    flow_diff_percentile: float
    save_topk_worst_psnr: int
    save_topk_best_psnr: int
    save_topk_largest_flow_diff: int
    input_config: dict[str, Any]


def parse_bool_value(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized_value = value.strip().lower()
        if normalized_value in ("true", "1", "yes", "on"):
            return True
        if normalized_value in ("false", "0", "no", "off"):
            return False
    raise TypeError(f"{key} must be a boolean, got {value!r}")


def parse_eval_convex_upsampling_arg(value: str) -> bool:
    return parse_bool_value(value, "eval_convex_upsampling")


def read_model_init_args(config_values: dict[str, Any]) -> dict[str, Any]:
    raw_model_init_args = config_values.get("model_init_args", {})
    if raw_model_init_args is None:
        return {}
    if not isinstance(raw_model_init_args, dict):
        raise TypeError(f"model_init_args must be a mapping, got {type(raw_model_init_args).__name__}")

    return dict(raw_model_init_args)


def read_optional_bool(config_values: dict[str, Any], key: str) -> bool | None:
    if key not in config_values or config_values[key] is None:
        return None
    return parse_bool_value(config_values[key], key)


def require_train_model_name(model_name: str) -> None:
    if model_name in TRAIN_MODEL_NAMES:
        return

    available_models = ", ".join(TRAIN_MODEL_NAMES)
    raise ValueError(
        f"Training does not support model_name={model_name}. "
        f"Available training models: {available_models}. External baselines are inference-only in this slice."
    )


def require_image_only_flow_approx_defaults(
    model_name: str,
    method: str,
    splatting_fill_strategy: str,
    init_flow_downscale_strategy: str,
    init_flow_mask_epsilon: float,
) -> None:
    if not uses_image_only_vfi_model(model_name):
        return

    if (
        method == "combination"
        and splatting_fill_strategy == DEFAULT_SPLATTING_FILL_STRATEGY
        and init_flow_downscale_strategy == DEFAULT_INIT_FLOW_DOWNSCALE_STRATEGY
        and init_flow_mask_epsilon == DEFAULT_INIT_FLOW_MASK_EPSILON
    ):
        return

    raise ValueError(
        f"model_name={model_name} returns image-only predictions and does not support flow approximation settings. "
        "Use default flow_approx_method=combination, "
        f"splatting_fill_strategy={DEFAULT_SPLATTING_FILL_STRATEGY}, "
        f"init_flow_downscale_strategy={DEFAULT_INIT_FLOW_DOWNSCALE_STRATEGY}, "
        f"and init_flow_mask_epsilon={DEFAULT_INIT_FLOW_MASK_EPSILON}."
    )


def build_flow_approx_config(model_name: str, config_values: dict[str, Any]) -> FlowApproxConfig:
    method = str(config_values.get("flow_approx_method", "combination"))
    if method not in FLOW_APPROX_METHOD_CHOICES:
        available_methods = ", ".join(FLOW_APPROX_METHOD_CHOICES)
        raise ValueError(f"Unsupported flow_approx_method '{method}'. Available methods: {available_methods}")

    splatting_fill_strategy = str(config_values.get("splatting_fill_strategy", DEFAULT_SPLATTING_FILL_STRATEGY))
    if splatting_fill_strategy not in SPLATTING_FILL_STRATEGIES:
        available_strategies = ", ".join(SPLATTING_FILL_STRATEGIES)
        raise ValueError(
            f"Unsupported splatting_fill_strategy '{splatting_fill_strategy}'. Available strategies: {available_strategies}"
        )

    init_flow_downscale_strategy = str(
        config_values.get("init_flow_downscale_strategy", DEFAULT_INIT_FLOW_DOWNSCALE_STRATEGY)
    )
    if init_flow_downscale_strategy not in INIT_FLOW_DOWNSCALE_STRATEGIES:
        available_strategies = ", ".join(INIT_FLOW_DOWNSCALE_STRATEGIES)
        raise ValueError(
            "Unsupported init_flow_downscale_strategy "
            f"'{init_flow_downscale_strategy}'. Available strategies: {available_strategies}"
        )

    init_flow_mask_epsilon = float(config_values.get("init_flow_mask_epsilon", DEFAULT_INIT_FLOW_MASK_EPSILON))
    if init_flow_mask_epsilon <= 0:
        raise ValueError(f"init_flow_mask_epsilon must be positive, got {init_flow_mask_epsilon}")

    require_image_only_flow_approx_defaults(
        model_name=model_name,
        method=method,
        splatting_fill_strategy=splatting_fill_strategy,
        init_flow_downscale_strategy=init_flow_downscale_strategy,
        init_flow_mask_epsilon=init_flow_mask_epsilon,
    )

    if init_flow_downscale_strategy == "masked_area":
        if model_name != "IFRNet_Residual_FlowApprox":
            raise ValueError("init_flow_downscale_strategy=masked_area requires model_name=IFRNet_Residual_FlowApprox.")
        if not is_splatting_flow_approx_method(flow_approx_method=method):
            raise ValueError(
                "init_flow_downscale_strategy=masked_area requires a splatting flow approximation method."
            )

    if not uses_flow_approx_model(model_name):
        return FlowApproxConfig(
            method=method,
            splatting_fill_strategy=splatting_fill_strategy,
            effective_splatting_fill_strategy="",
            init_flow_downscale_strategy=init_flow_downscale_strategy,
            effective_init_flow_downscale_strategy="",
            init_flow_mask_epsilon=init_flow_mask_epsilon,
        )

    if not is_splatting_flow_approx_method(flow_approx_method=method):
        return FlowApproxConfig(
            method=method,
            splatting_fill_strategy=splatting_fill_strategy,
            effective_splatting_fill_strategy="",
            init_flow_downscale_strategy=init_flow_downscale_strategy,
            effective_init_flow_downscale_strategy="",
            init_flow_mask_epsilon=init_flow_mask_epsilon,
        )

    return FlowApproxConfig(
        method=method,
        splatting_fill_strategy=splatting_fill_strategy,
        effective_splatting_fill_strategy=resolve_splatting_fill_strategy(
            flow_approx_method=method,
            splatting_fill_strategy=splatting_fill_strategy,
        ),
        init_flow_downscale_strategy=init_flow_downscale_strategy,
        effective_init_flow_downscale_strategy=init_flow_downscale_strategy,
        init_flow_mask_epsilon=init_flow_mask_epsilon,
    )


def build_train_run_config(args: argparse.Namespace, config_defaults: dict[str, Any]) -> TrainRunConfig:
    metric_config = read_metric_config(config_defaults)
    model_name = str(args.model_name)
    require_train_model_name(model_name=model_name)
    require_psnr_enabled(metric_config, "training")
    require_psnr_div_disabled(metric_config, "training")
    require_vfips_disabled(metric_config, "training")

    model_init_args = read_model_init_args(config_defaults)
    eval_convex_upsampling = (
        None
        if args.eval_convex_upsampling is None
        else parse_bool_value(args.eval_convex_upsampling, "eval_convex_upsampling")
    )
    model_config = ModelRunConfig(
        model_name=model_name,
        model_init_args=model_init_args,
        eval_convex_upsampling=eval_convex_upsampling,
    )

    flow_approx_config = build_flow_approx_config(model_name=model_config.model_name, config_values=vars(args))

    return TrainRunConfig(
        mode=str(args.mode),
        model=model_config,
        flow_approx=flow_approx_config,
        metrics=MetricRunConfig(values=dict(metric_config)),
        root_dir=Path(args.root_dir),
        dataset_root_dir=args.dataset_root_dir,
        paths_config=args.paths_config,
        train_preset=str(args.train_preset),
        test_preset=str(args.test_preset),
        epochs=int(args.epochs),
        resume_path=args.resume_path,
        pretrained_checkpoint_path=args.pretrained_checkpoint_path,
        eval_interval=int(args.eval_interval),
        lr_start=float(args.lr_start),
        lr_end=float(args.lr_end),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        output_dir=args.output_dir,
        only_fps=int(args.only_fps),
        input_fps=int(args.input_fps),
        sample_train_frames=list(args.sample_train_frames),
        sample_test_frames=list(args.sample_test_frames),
        sample_interval_epoch=int(args.sample_interval_epoch),
        input_config=dict(config_defaults),
    )


def build_inference_run_config(config_path: Path, project_root: Path) -> InferenceRunConfig:
    resolved_config_path = config_path if config_path.is_absolute() else project_root / config_path
    config = load_yaml_file(resolved_config_path)

    model_name = str(config["model_name"])
    model_config = ModelRunConfig(
        model_name=model_name,
        model_init_args=read_model_init_args(config),
        eval_convex_upsampling=read_optional_bool(config, "eval_convex_upsampling"),
    )
    flow_approx_config = build_flow_approx_config(model_name=model_name, config_values=config)

    metric_config = read_metric_config(config)
    require_psnr_enabled(metric_config, "inference")

    return InferenceRunConfig(
        mode=str(config["mode"]),
        model=model_config,
        flow_approx=flow_approx_config,
        metrics=MetricRunConfig(values=dict(metric_config)),
        inference_presets=_read_inference_presets(config),
        root_dir=_resolve_project_path(project_root=project_root, path_value=config["root_dir"]),
        dataset_root_dir=_resolve_project_path(project_root=project_root, path_value=config["dataset_root_dir"]),
        checkpoint_path=_resolve_project_path(project_root=project_root, path_value=config["checkpoint_path"]),
        output_dir=_resolve_project_path(project_root=project_root, path_value=config["output_dir"]),
        seed=int(config["seed"]),
        batch_size=int(config["batch_size"]),
        only_fps=int(config["only_fps"]),
        input_fps=int(config["input_fps"]),
        scale_factor=float(config["scale_factor"]),
        flow_diff_threshold=float(config.get("flow_diff_threshold", 1.0)),
        flow_diff_percentile=float(config.get("flow_diff_percentile", 99.0)),
        save_topk_worst_psnr=int(config.get("save_topk_worst_psnr", 3)),
        save_topk_best_psnr=int(config.get("save_topk_best_psnr", 0)),
        save_topk_largest_flow_diff=int(config.get("save_topk_largest_flow_diff", 3)),
        input_config=dict(config),
    )


def build_train_dry_run_summary(config: TrainRunConfig) -> dict[str, object]:
    summary: dict[str, object] = {
        "mode": config.mode,
        "model_name": config.model.model_name,
        "train_preset": config.train_preset,
        "test_preset": config.test_preset,
        "active_root_key": ACTIVE_DATASET_ROOT_KEY,
        "dataset_root_dir": config.dataset_root_dir,
        "csv_root_dir": str(config.root_dir),
        "output_dir": config.output_dir,
        "resume_path": config.resume_path,
        "pretrained_checkpoint_path": config.pretrained_checkpoint_path,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "eval_interval": config.eval_interval,
        "input_fps": config.input_fps,
        "only_fps": config.only_fps,
        "dataset_class": _resolve_dataset_class_name(config.model.model_name),
        "metrics": dict(config.metrics.values),
    }
    if config.model.eval_convex_upsampling is not None:
        summary["eval_convex_upsampling"] = config.model.eval_convex_upsampling

    if len(config.model.model_init_args) > 0:
        summary["model_init_args"] = dict(config.model.model_init_args)

    if uses_flow_approx_model(config.model.model_name):
        summary["flow_approx_method"] = config.flow_approx.method
        summary["splatting_fill_strategy"] = config.flow_approx.splatting_fill_strategy
        summary["effective_splatting_fill_strategy"] = config.flow_approx.effective_splatting_fill_strategy
        summary["init_flow_downscale_strategy"] = config.flow_approx.init_flow_downscale_strategy
        summary["effective_init_flow_downscale_strategy"] = config.flow_approx.effective_init_flow_downscale_strategy
        summary["init_flow_mask_epsilon"] = config.flow_approx.init_flow_mask_epsilon

    return summary


def build_inference_dry_run_summary(config: InferenceRunConfig) -> dict[str, object]:
    summary: dict[str, object] = {
        "mode": config.mode,
        "model_name": config.model.model_name,
        "inference_presets": list(config.inference_presets),
        "root_dir": str(config.root_dir),
        "dataset_root_dir": str(config.dataset_root_dir),
        "checkpoint_path": str(config.checkpoint_path),
        "output_dir": str(config.output_dir),
        "batch_size": config.batch_size,
        "only_fps": config.only_fps,
        "input_fps": config.input_fps,
        "scale_factor": config.scale_factor,
        "flow_approx_method": config.flow_approx.method,
        "splatting_fill_strategy": config.flow_approx.splatting_fill_strategy,
        "init_flow_downscale_strategy": config.flow_approx.init_flow_downscale_strategy,
        "init_flow_mask_epsilon": config.flow_approx.init_flow_mask_epsilon,
        "flow_diff_threshold": config.flow_diff_threshold,
        "flow_diff_percentile": config.flow_diff_percentile,
        "save_topk_worst_psnr": config.save_topk_worst_psnr,
        "save_topk_best_psnr": config.save_topk_best_psnr,
        "save_topk_largest_flow_diff": config.save_topk_largest_flow_diff,
        "metrics": dict(config.metrics.values),
    }
    if config.model.eval_convex_upsampling is not None:
        summary["eval_convex_upsampling"] = config.model.eval_convex_upsampling
    if len(config.model.model_init_args) > 0:
        summary["model_init_args"] = dict(config.model.model_init_args)

    return summary


def _resolve_project_path(project_root: Path, path_value: Any) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return project_root / path


def _read_inference_presets(config: dict[str, Any]) -> list[str]:
    if "inference_presets" in config:
        value = config["inference_presets"]
    elif "inference_preset" in config:
        value = config["inference_preset"]
    else:
        raise KeyError("Config must contain inference_preset or inference_presets.")

    if isinstance(value, str):
        return [value]

    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if len(value) == 0:
            raise ValueError("inference_presets must contain at least one preset name.")
        return list(value)

    raise TypeError("inference_preset must be a string, or inference_presets must be a list of strings.")


def _resolve_dataset_class_name(model_name: str) -> str:
    if uses_flow_approx_model(model_name):
        return "FlowEstimationTrainDataset"
    return "VFITrainDataset"
