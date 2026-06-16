from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scratch.train_oracle_effective_time import CachedOracleEffectiveTimeTrainDataset
from experiments.scratch.train_oracle_effective_time import DEFAULT_ORACLE_EFFECTIVE_TIME_CACHE_DTYPE
from experiments.scratch.train_oracle_effective_time import DEFAULT_ORACLE_EFFECTIVE_TIME_EPSILON
from experiments.scratch.train_oracle_effective_time import DEFAULT_ORACLE_EFFECTIVE_TIME_INVALID_FLOW_SCALE
from experiments.scratch.train_oracle_effective_time import DEFAULT_ORACLE_EFFECTIVE_TIME_MIN_MOTION
from experiments.scratch.train_oracle_effective_time import DEFAULT_ORACLE_EFFECTIVE_TIME_SOURCE
from experiments.scratch.train_oracle_effective_time import ORACLE_EFFECTIVE_TIME_CACHE_DTYPES
from experiments.scratch.train_oracle_effective_time import ORACLE_EFFECTIVE_TIME_SOURCES
from experiments.scratch.train_oracle_effective_time import build_oracle_t_eff_stats
from experiments.scratch.train_oracle_effective_time import ensure_oracle_effective_time_cache
from scripts.inference import DEFAULT_INIT_FLOW_DOWNSCALE_STRATEGY
from scripts.inference import DEFAULT_INIT_FLOW_MASK_EPSILON
from scripts.inference import DEFAULT_SPLATTING_FILL_STRATEGY
from scripts.inference import FLOW_APPROX_METHOD_CHOICES
from scripts.inference import INIT_FLOW_DOWNSCALE_STRATEGIES
from scripts.inference import RESIDUAL_FLOW_APPROX_MODEL_NAME
from scripts.inference import SPLATTING_FILL_STRATEGIES
from scripts.inference import build_splatting_region_maps
from scripts.inference import read_inference_presets
from scripts.inference import save_selected_sample_artifacts
from scripts.train import build_merged_dataframe
from scripts.train import load_pretrained_model_state
from scripts.train import read_effective_time_config
from scripts.train import read_model_init_args
from scripts.train import resolve_model_class
from scripts.train import set_seed
from src.engine.evaluation import average_metric_values
from src.engine.evaluation import build_lpips_model
from src.engine.evaluation import build_metric_meters
from src.engine.evaluation import calculate_batch_metrics
from src.engine.evaluation import format_metric_averages
from src.engine.evaluation import read_metric_config
from src.engine.evaluation import require_psnr_enabled
from src.engine.flow_approx import attach_effective_time_estimator
from src.engine.flow_approx import build_flow_init_result_with_fill_strategy
from src.engine.flow_approx import is_splatting_flow_approx_method
from src.utils.config import load_yaml_file
from src.utils.logger import build_logger

OracleInferenceBatchResult = dict[str, Any]


def resolve_config_path(config_text: str) -> Path:
    config_path = Path(config_text)
    if config_path.is_absolute():
        return config_path
    return PROJECT_ROOT / config_path


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_default_oracle_output_dir(output_dir_text: Any) -> str | None:
    if not isinstance(output_dir_text, str) or output_dir_text == "":
        return None
    output_dir = Path(output_dir_text)
    return str(output_dir.parent / f"{output_dir.name}_oracle_t_eff")


def load_config_defaults(argv: list[str] | None) -> tuple[dict[str, Any], Path]:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", required=True, type=str)
    bootstrap_args, _remaining_argv = bootstrap_parser.parse_known_args(argv)
    config_path = resolve_config_path(bootstrap_args.config)
    config_defaults = load_yaml_file(config_path)
    if not isinstance(config_defaults, dict):
        raise TypeError(f"Inference config must contain a JSON object: path={config_path}")
    return config_defaults, config_path


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    config_defaults, config_path = load_config_defaults(argv)
    parser = argparse.ArgumentParser(
        description="Run oracle t_eff inference with offline cached oracle effective-time flow initialization."
    )
    parser.add_argument("--config", required=True, type=str, help="Path to one inference config file.")
    parser.add_argument("--mode", default=config_defaults.get("mode", "run"), type=str)
    parser.add_argument(
        "--output-dir",
        default=build_default_oracle_output_dir(config_defaults.get("output_dir")),
        type=str,
    )
    parser.add_argument(
        "--oracle-effective-time-source",
        default=config_defaults.get("oracle_effective_time_source", DEFAULT_ORACLE_EFFECTIVE_TIME_SOURCE),
        choices=ORACLE_EFFECTIVE_TIME_SOURCES,
    )
    parser.add_argument(
        "--oracle-effective-time-epsilon",
        default=config_defaults.get("oracle_effective_time_epsilon", DEFAULT_ORACLE_EFFECTIVE_TIME_EPSILON),
        type=float,
    )
    parser.add_argument(
        "--oracle-effective-time-min-motion",
        default=config_defaults.get("oracle_effective_time_min_motion", DEFAULT_ORACLE_EFFECTIVE_TIME_MIN_MOTION),
        type=float,
    )
    parser.add_argument(
        "--oracle-effective-time-invalid-flow-scale",
        default=config_defaults.get(
            "oracle_effective_time_invalid_flow_scale",
            DEFAULT_ORACLE_EFFECTIVE_TIME_INVALID_FLOW_SCALE,
        ),
        type=float,
    )
    parser.add_argument(
        "--oracle-effective-time-cache-dir",
        default=config_defaults.get("oracle_effective_time_cache_dir"),
        type=str,
    )
    parser.add_argument(
        "--oracle-effective-time-cache-dtype",
        default=config_defaults.get("oracle_effective_time_cache_dtype", DEFAULT_ORACLE_EFFECTIVE_TIME_CACHE_DTYPE),
        choices=ORACLE_EFFECTIVE_TIME_CACHE_DTYPES,
    )
    parser.set_defaults(
        rebuild_oracle_effective_time_cache=bool(
            config_defaults.get("rebuild_oracle_effective_time_cache", False)
        )
    )
    parser.add_argument(
        "--rebuild-oracle-effective-time-cache",
        dest="rebuild_oracle_effective_time_cache",
        action="store_true",
        help="Regenerate oracle t_eff cache files even when they already exist.",
    )
    args = parser.parse_args(argv)
    args.config_path = config_path
    if args.output_dir is None:
        raise ValueError("output_dir must be provided by config or CLI for oracle inference.")
    if args.oracle_effective_time_cache_dir is None:
        args.oracle_effective_time_cache_dir = str(resolve_path(args.output_dir) / "oracle_t_eff_cache")
    return args


def validate_oracle_inference_config(
    args: argparse.Namespace,
    model_name: str,
    flow_approx_method: str,
    splatting_fill_strategy: str,
    init_flow_downscale_strategy: str,
    init_flow_mask_epsilon: float,
) -> None:
    if model_name != RESIDUAL_FLOW_APPROX_MODEL_NAME:
        raise ValueError(
            "Oracle effective-time inference requires "
            f"model_name={RESIDUAL_FLOW_APPROX_MODEL_NAME}, got {model_name}."
        )
    if flow_approx_method not in FLOW_APPROX_METHOD_CHOICES:
        raise ValueError(f"Unsupported flow_approx_method: {flow_approx_method}")
    if splatting_fill_strategy not in SPLATTING_FILL_STRATEGIES:
        available_strategies = ", ".join(SPLATTING_FILL_STRATEGIES)
        raise ValueError(
            f"Unsupported splatting_fill_strategy: {splatting_fill_strategy}. "
            f"Available strategies: {available_strategies}"
        )
    if init_flow_downscale_strategy not in INIT_FLOW_DOWNSCALE_STRATEGIES:
        available_strategies = ", ".join(INIT_FLOW_DOWNSCALE_STRATEGIES)
        raise ValueError(
            f"Unsupported init_flow_downscale_strategy: {init_flow_downscale_strategy}. "
            f"Available strategies: {available_strategies}"
        )
    if init_flow_mask_epsilon <= 0.0:
        raise ValueError(f"init_flow_mask_epsilon must be positive, got {init_flow_mask_epsilon}")
    if args.oracle_effective_time_source not in ORACLE_EFFECTIVE_TIME_SOURCES:
        available_sources = ", ".join(ORACLE_EFFECTIVE_TIME_SOURCES)
        raise ValueError(
            f"Unsupported oracle_effective_time_source={args.oracle_effective_time_source}. "
            f"Available sources: {available_sources}"
        )
    if args.oracle_effective_time_epsilon <= 0.0:
        raise ValueError(
            f"oracle_effective_time_epsilon must be positive, got {args.oracle_effective_time_epsilon}"
        )
    if args.oracle_effective_time_min_motion < 0.0:
        raise ValueError(
            f"oracle_effective_time_min_motion must be non-negative, got {args.oracle_effective_time_min_motion}"
        )
    if args.oracle_effective_time_invalid_flow_scale < 0.0 or args.oracle_effective_time_invalid_flow_scale > 1.0:
        raise ValueError(
            "oracle_effective_time_invalid_flow_scale must be in [0, 1], "
            f"got {args.oracle_effective_time_invalid_flow_scale}"
        )
    if init_flow_downscale_strategy == "masked_area" and not is_splatting_flow_approx_method(flow_approx_method):
        raise ValueError("init_flow_downscale_strategy=masked_area requires a splatting flow approximation method.")


def extract_model_state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def strip_effective_time_estimator_state_dict(state_dict: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict dictionary, got {type(state_dict)}")
    removed_keys = sorted(
        key for key in state_dict.keys() if isinstance(key, str) and key.startswith("effective_time_estimator.")
    )
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not (isinstance(key, str) and key.startswith("effective_time_estimator."))
    }
    return filtered_state_dict, removed_keys


def build_dry_run_summary(
    args: argparse.Namespace,
    config: dict[str, Any],
    model_name: str,
    model_init_args: dict[str, Any],
    inference_presets: list[str],
    flow_approx_method: str,
    splatting_fill_strategy: str,
    init_flow_downscale_strategy: str,
    init_flow_mask_epsilon: float,
    config_effective_time_mode: str,
    metric_config: dict[str, object],
) -> dict[str, object]:
    return {
        "mode": args.mode,
        "script": "experiments/scratch/inference_oracle_effective_time.py",
        "config_path": str(args.config_path),
        "model_name": model_name,
        "model_init_args": dict(model_init_args),
        "inference_presets": list(inference_presets),
        "root_dir": str(resolve_path(str(config["root_dir"]))),
        "dataset_root_dir": str(resolve_path(str(config["dataset_root_dir"]))),
        "checkpoint_path": str(resolve_path(str(config["checkpoint_path"]))),
        "output_dir": str(resolve_path(args.output_dir)),
        "batch_size": int(config["batch_size"]),
        "only_fps": int(config["only_fps"]),
        "input_fps": int(config["input_fps"]),
        "scale_factor": float(config["scale_factor"]),
        "flow_approx_method": flow_approx_method,
        "splatting_fill_strategy": splatting_fill_strategy,
        "init_flow_downscale_strategy": init_flow_downscale_strategy,
        "init_flow_mask_epsilon": init_flow_mask_epsilon,
        "config_effective_time_mode": config_effective_time_mode,
        "oracle_effective_time_mode": "oracle_cache",
        "oracle_effective_time_source": args.oracle_effective_time_source,
        "oracle_effective_time_epsilon": args.oracle_effective_time_epsilon,
        "oracle_effective_time_min_motion": args.oracle_effective_time_min_motion,
        "oracle_effective_time_invalid_flow_scale": args.oracle_effective_time_invalid_flow_scale,
        "oracle_effective_time_cache_dir": args.oracle_effective_time_cache_dir,
        "oracle_effective_time_cache_dtype": args.oracle_effective_time_cache_dtype,
        "rebuild_oracle_effective_time_cache": args.rebuild_oracle_effective_time_cache,
        "metrics": dict(metric_config),
    }


def run_oracle_inference_batch_with_fill_strategy(
    batch: Any,
    device: Any,
    flow_approx_method: str,
    splatting_fill_strategy: str,
    init_flow_downscale_strategy: str,
    init_flow_mask_epsilon: float,
    model: Any,
    scale_factor: float,
) -> OracleInferenceBatchResult:
    img0, imgt, img1, bmv, fmv, bmv_30, fmv_30, oracle_t_eff, embt, info = batch
    img0 = img0.to(device)
    imgt = imgt.to(device)
    img1 = img1.to(device)
    bmv = bmv.to(device)
    fmv = fmv.to(device)
    bmv_30 = bmv_30.to(device)
    fmv_30 = fmv_30.to(device)
    oracle_t_eff = oracle_t_eff.to(device)
    embt = embt.to(device)

    source_depth0 = None
    source_depth1 = None
    if is_splatting_flow_approx_method(flow_approx_method=flow_approx_method):
        source_depth0 = info["source_depth0"].to(device)
        source_depth1 = info["source_depth1"].to(device)

    flow_init = build_flow_init_result_with_fill_strategy(
        fmv_30=fmv_30,
        bmv_30=bmv_30,
        embt=embt,
        flow_approx_method=flow_approx_method,
        source_depth0=source_depth0,
        source_depth1=source_depth1,
        splatting_fill_strategy=splatting_fill_strategy,
        ground_truth_bmv=bmv,
        ground_truth_fmv=fmv,
        effective_time=oracle_t_eff,
    )
    init_bmv = flow_init.bmv
    init_fmv = flow_init.fmv
    init_masks = flow_init.masks
    splatting_region_maps = build_splatting_region_maps(fmv_30, bmv_30, embt, init_masks, oracle_t_eff)

    init_bmv_mask = None
    init_fmv_mask = None
    if init_flow_downscale_strategy == "masked_area":
        if init_masks is None:
            raise RuntimeError(
                "init_flow_downscale_strategy=masked_area requires splatting coverage masks, but none were produced."
            )
        init_bmv_mask = init_masks[:, 0:1]
        init_fmv_mask = init_masks[:, 1:2]

    imgt_pred, up_flow0_1, up_flow1_1, up_mask_1, _up_res_1, imgt_merge = model.inference(
        img0,
        img1,
        embt,
        scale_factor,
        init_flow0=init_bmv,
        init_flow1=init_fmv,
        init_flow0_mask=init_bmv_mask,
        init_flow1_mask=init_fmv_mask,
        init_flow_mask_epsilon=init_flow_mask_epsilon,
    )
    return {
        "bmv": bmv,
        "embt": embt,
        "effective_time": oracle_t_eff,
        "fmv": fmv,
        "img0": img0,
        "img1": img1,
        "imgt": imgt,
        "imgt_merge": imgt_merge,
        "imgt_pred": imgt_pred,
        "init_bmv": init_bmv,
        "init_fmv": init_fmv,
        "init_masks": init_masks,
        "oracle_t_eff": oracle_t_eff,
        "splatting_region_maps": splatting_region_maps,
        "up_flow0_1": up_flow0_1,
        "up_flow1_1": up_flow1_1,
        "up_mask_1": up_mask_1,
    }


def main(argv: list[str] | None) -> None:
    args = parse_args(argv)
    config = load_yaml_file(args.config_path)
    if not isinstance(config, dict):
        raise TypeError(f"Inference config must contain a JSON object: path={args.config_path}")

    config["mode"] = args.mode
    config["output_dir"] = args.output_dir
    config_effective_time_mode, _effective_time_hidden_channels, _effective_time_radius = read_effective_time_config(
        config
    )
    model_name = str(config["model_name"])
    model_init_args = read_model_init_args(config)
    inference_presets = read_inference_presets(config)
    flow_approx_method = str(config["flow_approx_method"])
    splatting_fill_strategy = str(config.get("splatting_fill_strategy", DEFAULT_SPLATTING_FILL_STRATEGY))
    init_flow_downscale_strategy = str(
        config.get("init_flow_downscale_strategy", DEFAULT_INIT_FLOW_DOWNSCALE_STRATEGY)
    )
    init_flow_mask_epsilon = float(config.get("init_flow_mask_epsilon", DEFAULT_INIT_FLOW_MASK_EPSILON))
    metric_config = read_metric_config(config)
    require_psnr_enabled(metric_config, "oracle effective-time inference")
    validate_oracle_inference_config(
        args=args,
        model_name=model_name,
        flow_approx_method=flow_approx_method,
        splatting_fill_strategy=splatting_fill_strategy,
        init_flow_downscale_strategy=init_flow_downscale_strategy,
        init_flow_mask_epsilon=init_flow_mask_epsilon,
    )

    summary = build_dry_run_summary(
        args=args,
        config=config,
        model_name=model_name,
        model_init_args=model_init_args,
        inference_presets=inference_presets,
        flow_approx_method=flow_approx_method,
        splatting_fill_strategy=splatting_fill_strategy,
        init_flow_downscale_strategy=init_flow_downscale_strategy,
        init_flow_mask_epsilon=init_flow_mask_epsilon,
        config_effective_time_mode=config_effective_time_mode,
        metric_config=metric_config,
    )
    if args.mode == "dry-run":
        print(json.dumps(summary, indent=2))
        return

    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import Subset
    from tqdm import tqdm

    from src.data.image_ops import flow_to_image
    from src.data.image_ops import save_image

    root_dir = resolve_path(str(config["root_dir"]))
    dataset_root_dir = resolve_path(str(config["dataset_root_dir"]))
    checkpoint_path = resolve_path(str(config["checkpoint_path"]))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    only_fps = int(config["only_fps"])
    input_fps = int(config["input_fps"])
    scale_factor = float(config["scale_factor"])
    flow_diff_threshold = float(config.get("flow_diff_threshold", 1.0))
    flow_diff_percentile = float(config.get("flow_diff_percentile", 99.0))
    save_topk_worst_psnr = int(config.get("save_topk_worst_psnr", 3))
    save_topk_best_psnr = int(config.get("save_topk_best_psnr", 0))
    save_topk_largest_flow_diff = int(config.get("save_topk_largest_flow_diff", 3))

    logger = build_logger("experiments.scratch.inference_oracle_effective_time")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = build_lpips_model(metric_config, device)
    logger.info("device=%s model=%s", device, model_name)
    logger.info("metrics=%s", metric_config)
    logger.info(
        "flow_approx_method=%s splatting_fill_strategy=%s init_flow_downscale_strategy=%s init_flow_mask_epsilon=%s",
        flow_approx_method,
        splatting_fill_strategy,
        init_flow_downscale_strategy,
        init_flow_mask_epsilon,
    )
    logger.info(
        "config_effective_time_mode=%s oracle_effective_time_mode=%s",
        config_effective_time_mode,
        "oracle_cache",
    )
    logger.info(
        "oracle_effective_time_source=%s epsilon=%s min_motion=%s invalid_flow_scale=%s",
        args.oracle_effective_time_source,
        args.oracle_effective_time_epsilon,
        args.oracle_effective_time_min_motion,
        args.oracle_effective_time_invalid_flow_scale,
    )
    logger.info(
        "oracle_effective_time_cache_dir=%s oracle_effective_time_cache_dtype=%s rebuild_oracle_effective_time_cache=%s",
        args.oracle_effective_time_cache_dir,
        args.oracle_effective_time_cache_dtype,
        args.rebuild_oracle_effective_time_cache,
    )

    dataframe_list: list[Any] = []
    for inference_preset in inference_presets:
        preset_dataframe = build_merged_dataframe(root_dir, output_dir, inference_preset, only_fps, logger)
        preset_dataframe["inference_preset"] = inference_preset
        dataframe_list.append(preset_dataframe)

    dataframe = pd.concat(dataframe_list, ignore_index=True)
    if "valid" in dataframe.columns:
        dataframe = dataframe[dataframe["valid"] == True].reset_index(drop=True)

    oracle_cache_dir = Path(args.oracle_effective_time_cache_dir)
    ensure_oracle_effective_time_cache(
        dataframe=dataframe,
        dataset_root_dir=dataset_root_dir,
        cache_dir=oracle_cache_dir,
        args=args,
        logger=logger,
        split_name="inference",
    )

    include_source_depths = is_splatting_flow_approx_method(flow_approx_method=flow_approx_method)
    model_class = resolve_model_class(model_name)
    model = model_class(**model_init_args)
    attach_effective_time_estimator(model=model, effective_time_mode="disabled", effective_time_hidden_channels=32)
    model = model.to(device)
    if hasattr(model, "init_flow_layer"):
        logger.info("model_init_flow_layer=%s", model.init_flow_layer)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = extract_model_state_dict(checkpoint)
    filtered_state_dict, removed_effective_time_keys = strip_effective_time_estimator_state_dict(state_dict)
    if len(removed_effective_time_keys) > 0:
        logger.info(
            "Dropped checkpoint effective_time_estimator keys because oracle inference uses cached oracle_t_eff: count=%s",
            len(removed_effective_time_keys),
        )
    load_pretrained_model_state(model=model, state_dict=filtered_state_dict, logger=logger)
    model.eval()

    metric_meters = build_metric_meters(metric_config)
    rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for (inference_preset, record, mode_name), group_dataframe in dataframe.groupby(
            ["inference_preset", "record", "mode"],
            sort=False,
        ):
            group_dataframe = group_dataframe.reset_index(drop=True)
            dataset = CachedOracleEffectiveTimeTrainDataset(
                dataframe=group_dataframe,
                dataset_root_dir=str(dataset_root_dir),
                input_fps=input_fps,
                augment=False,
                include_source_depths=include_source_depths,
                oracle_cache_dir=oracle_cache_dir,
                oracle_args=args,
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            record_metric_meters = build_metric_meters(metric_config)
            progress = tqdm(loader, desc=f"{inference_preset}_{record}_{mode_name}", leave=True)
            sample_offset = 0
            group_rows: list[dict[str, object]] = []

            for batch in progress:
                inference_result = run_oracle_inference_batch_with_fill_strategy(
                    batch=batch,
                    device=device,
                    flow_approx_method=flow_approx_method,
                    splatting_fill_strategy=splatting_fill_strategy,
                    init_flow_downscale_strategy=init_flow_downscale_strategy,
                    init_flow_mask_epsilon=init_flow_mask_epsilon,
                    model=model,
                    scale_factor=scale_factor,
                )
                imgt = inference_result["imgt"]
                imgt_pred = inference_result["imgt_pred"]
                init_bmv = inference_result["init_bmv"]
                init_fmv = inference_result["init_fmv"]
                up_flow0_1 = inference_result["up_flow0_1"]
                up_flow1_1 = inference_result["up_flow1_1"]
                batch_metric_values = calculate_batch_metrics(imgt.detach(), imgt_pred.detach(), metric_config, lpips_model)
                oracle_t_eff_stats = build_oracle_t_eff_stats(inference_result["oracle_t_eff"])

                for batch_index in range(int(imgt_pred.shape[0])):
                    row = group_dataframe.iloc[sample_offset + batch_index]
                    frame_range = f"frame_{int(row['img0']):04d}_{int(row['img2']):04d}"
                    sample_metric_values = {
                        metric_name: float(metric_values[batch_index])
                        for metric_name, metric_values in batch_metric_values.items()
                    }
                    for metric_name, metric_value in sample_metric_values.items():
                        metric_meters[metric_name].update(metric_value, 1)
                        record_metric_meters[metric_name].update(metric_value, 1)

                    init_flow_1_to_0_np = init_bmv[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                    init_flow_1_to_2_np = init_fmv[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                    final_flow_1_to_0_np = up_flow0_1[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                    final_flow_1_to_2_np = up_flow1_1[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                    diff_mag_1_to_0_np = np.linalg.norm(final_flow_1_to_0_np - init_flow_1_to_0_np, axis=2)
                    diff_mag_1_to_2_np = np.linalg.norm(final_flow_1_to_2_np - init_flow_1_to_2_np, axis=2)
                    diff_1_to_0 = {
                        "diff_mag_mean": float(diff_mag_1_to_0_np.mean()),
                        "diff_mag_max": float(diff_mag_1_to_0_np.max()),
                        "diff_changed_ratio": float((diff_mag_1_to_0_np > flow_diff_threshold).mean()),
                        "diff_percentile_value": float(
                            max(np.percentile(diff_mag_1_to_0_np, flow_diff_percentile), 1e-6)
                        ),
                    }
                    diff_1_to_2 = {
                        "diff_mag_mean": float(diff_mag_1_to_2_np.mean()),
                        "diff_mag_max": float(diff_mag_1_to_2_np.max()),
                        "diff_changed_ratio": float((diff_mag_1_to_2_np > flow_diff_threshold).mean()),
                        "diff_percentile_value": float(
                            max(np.percentile(diff_mag_1_to_2_np, flow_diff_percentile), 1e-6)
                        ),
                    }

                    group_rows.append(
                        {
                            "sample_index": int(sample_offset + batch_index),
                            "inference_preset": str(inference_preset),
                            "record": str(record),
                            "mode": str(mode_name),
                            "record_name": f"{record}_{mode_name}",
                            "frame_range": frame_range,
                            "valid": bool(row["valid"]) if "valid" in row.index else True,
                            "distance_index_mean": float(row["D_index Mean"]) if "D_index Mean" in row.index else -1.0,
                            "distance_index_median": float(row["D_index Median"]) if "D_index Median" in row.index else -1.0,
                            **sample_metric_values,
                            "oracle_t_eff_mean": oracle_t_eff_stats["oracle_t_eff_mean"][batch_index],
                            "oracle_t_eff_std": oracle_t_eff_stats["oracle_t_eff_std"][batch_index],
                            "oracle_t_eff_min": oracle_t_eff_stats["oracle_t_eff_min"][batch_index],
                            "oracle_t_eff_max": oracle_t_eff_stats["oracle_t_eff_max"][batch_index],
                            "flow_diff_1_to_0_mean": diff_1_to_0["diff_mag_mean"],
                            "flow_diff_1_to_0_max": diff_1_to_0["diff_mag_max"],
                            "flow_diff_1_to_0_changed_ratio": diff_1_to_0["diff_changed_ratio"],
                            "flow_diff_1_to_0_percentile_value": diff_1_to_0["diff_percentile_value"],
                            "flow_diff_1_to_2_mean": diff_1_to_2["diff_mag_mean"],
                            "flow_diff_1_to_2_max": diff_1_to_2["diff_mag_max"],
                            "flow_diff_1_to_2_changed_ratio": diff_1_to_2["diff_changed_ratio"],
                            "flow_diff_1_to_2_percentile_value": diff_1_to_2["diff_percentile_value"],
                        }
                    )

                sample_offset += int(imgt_pred.shape[0])
                progress.set_postfix({"mean_psnr": f"{record_metric_meters['psnr'].avg:.6f}"})

            group_metrics_df = pd.DataFrame(group_rows)
            record_metric_values = average_metric_values(record_metric_meters)
            record_rows.append(
                {
                    "record": str(record),
                    "inference_preset": str(inference_preset),
                    "mode": str(mode_name),
                    "record_name": f"{record}_{mode_name}",
                    "samples": int(len(group_dataframe)),
                    **{f"mean_{metric_name}": metric_value for metric_name, metric_value in record_metric_values.items()},
                    "mean_oracle_t_eff_mean": float(group_metrics_df["oracle_t_eff_mean"].mean()),
                    "mean_oracle_t_eff_std": float(group_metrics_df["oracle_t_eff_std"].mean()),
                }
            )

            selected_sample_reasons: dict[int, list[str]] = {}
            if save_topk_worst_psnr > 0:
                for sample_index in group_metrics_df.nsmallest(save_topk_worst_psnr, "psnr")["sample_index"].tolist():
                    selected_sample_reasons.setdefault(int(sample_index), []).append("worst_psnr")
            if save_topk_best_psnr > 0:
                for sample_index in group_metrics_df.nlargest(save_topk_best_psnr, "psnr")["sample_index"].tolist():
                    selected_sample_reasons.setdefault(int(sample_index), []).append("best_psnr")
            if save_topk_largest_flow_diff > 0:
                for sample_index in group_metrics_df.nlargest(
                    save_topk_largest_flow_diff,
                    "flow_diff_1_to_0_changed_ratio",
                )["sample_index"].tolist():
                    selected_sample_reasons.setdefault(int(sample_index), []).append("largest_flow_diff_1_to_0")
                for sample_index in group_metrics_df.nlargest(
                    save_topk_largest_flow_diff,
                    "flow_diff_1_to_2_changed_ratio",
                )["sample_index"].tolist():
                    selected_sample_reasons.setdefault(int(sample_index), []).append("largest_flow_diff_1_to_2")

            group_metrics_df["selected_for_save"] = group_metrics_df["sample_index"].map(
                lambda sample_index: int(sample_index) in selected_sample_reasons
            )
            group_metrics_df["save_reason"] = group_metrics_df["sample_index"].map(
                lambda sample_index: ";".join(selected_sample_reasons.get(int(sample_index), []))
            )
            for column_name in (
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
            ):
                group_metrics_df[column_name] = ""

            selected_indices = sorted(selected_sample_reasons.keys())
            if len(selected_indices) > 0:
                selected_dataset = Subset(dataset, selected_indices)
                selected_loader = DataLoader(selected_dataset, batch_size=1, shuffle=False)
                selected_progress = tqdm(selected_loader, desc=f"save_{inference_preset}_{record}_{mode_name}", leave=True)

                for selected_batch_index, batch in enumerate(selected_progress):
                    selected_row = group_metrics_df[
                        group_metrics_df["sample_index"] == selected_indices[selected_batch_index]
                    ].iloc[0]
                    frame_range = str(selected_row["frame_range"])
                    save_dir = output_dir / str(record) / str(mode_name) / frame_range
                    inference_result = run_oracle_inference_batch_with_fill_strategy(
                        batch=batch,
                        device=device,
                        flow_approx_method=flow_approx_method,
                        splatting_fill_strategy=splatting_fill_strategy,
                        init_flow_downscale_strategy=init_flow_downscale_strategy,
                        init_flow_mask_epsilon=init_flow_mask_epsilon,
                        model=model,
                        scale_factor=scale_factor,
                    )
                    image_paths = save_selected_sample_artifacts(
                        cv2=cv2,
                        flow_diff_percentile=flow_diff_percentile,
                        flow_diff_threshold=flow_diff_threshold,
                        flow_to_image=flow_to_image,
                        inference_result=inference_result,
                        np=np,
                        save_dir=save_dir,
                        save_image=save_image,
                    )
                    for column_name, path_value in image_paths.items():
                        group_metrics_df.loc[
                            group_metrics_df["sample_index"] == selected_indices[selected_batch_index],
                            column_name,
                        ] = path_value

            rows.extend(group_metrics_df.to_dict("records"))
            logger.info(
                "record=%s mode=%s samples=%s metrics=%s",
                record,
                mode_name,
                len(group_dataframe),
                format_metric_averages(record_metric_meters),
            )

    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(record_rows).to_csv(output_dir / "record_metrics.csv", index=False)
    logger.info("samples=%s metrics=%s output_dir=%s", len(rows), format_metric_averages(metric_meters), output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
