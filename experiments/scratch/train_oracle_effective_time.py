from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_logger
from scripts.train import build_merged_dataframe
from scripts.train import build_record_name_summary
from scripts.train import build_train_arg_parser
from scripts.train import get_lr
from scripts.train import load_train_run_config
from scripts.train import load_training_state
from scripts.train import read_model_init_args
from scripts.train import resolve_effective_init_flow_downscale_strategy
from scripts.train import resolve_effective_splatting_fill_strategy
from scripts.train import resolve_model_class
from scripts.train import save_checkpoint
from scripts.train import save_input_config
from scripts.train import set_lr
from scripts.train import set_seed
from scripts.train import uses_flow_approx_model
from src.data.augment import shared_random_crop
from src.data.augment import shared_random_horizontal_flip
from src.data.augment import shared_random_reverse_channel
from src.data.augment import shared_random_rotate
from src.data.augment import shared_random_vertical_flip
from src.data.dataset_loader import BaseDataset
from src.data.dataset_loader import DEFAULT_MODALITY_CONFIG
from src.data.dataset_loader import build_distance_indexing
from src.data.dataset_loader import build_embedding_tensor
from src.data.dataset_loader import depth_to_tensor
from src.data.dataset_loader import flow_to_tensor
from src.data.dataset_loader import image_to_tensor
from src.data.image_ops import load_backward_velocity
from src.engine.evaluation import AverageMeter
from src.engine.evaluation import average_metric_values
from src.engine.evaluation import build_lpips_model
from src.engine.evaluation import build_metric_meters
from src.engine.evaluation import calculate_batch_metrics
from src.engine.evaluation import format_metric_averages
from src.engine.evaluation import format_metric_values
from src.engine.evaluation import read_metric_config
from src.engine.evaluation import require_psnr_enabled
from src.engine.flow_approx import attach_effective_time_estimator
from src.engine.flow_approx import build_flow_init_result_with_fill_strategy
from src.engine.flow_approx import is_splatting_flow_approx_method

ORACLE_EFFECTIVE_TIME_SOURCES: tuple[str, ...] = ("average", "fmv", "bmv")
DEFAULT_ORACLE_EFFECTIVE_TIME_SOURCE: str = "average"
DEFAULT_ORACLE_EFFECTIVE_TIME_EPSILON: float = 1.0e-6
DEFAULT_ORACLE_EFFECTIVE_TIME_MIN_MOTION: float = 1.0e-3
DEFAULT_ORACLE_EFFECTIVE_TIME_INVALID_FLOW_SCALE: float = 0.5
ORACLE_EFFECTIVE_TIME_CACHE_DTYPES: tuple[str, ...] = ("float16", "float32")
DEFAULT_ORACLE_EFFECTIVE_TIME_CACHE_DTYPE: str = "float16"


@dataclass(frozen=True)
class ProjectionScaleResult:
    scale: Any
    valid: Any
    raw_scale: Any


@dataclass(frozen=True)
class OracleBatchStepOutput:
    imgt: Any
    imgt_pred: Any
    info: dict[str, Any]
    loss_rec: Any
    loss_geo: Any
    loss_dis: Any
    oracle_t_eff: Any


def validate_oracle_args(args: argparse.Namespace) -> None:
    if not uses_flow_approx_model(args.model_name):
        raise ValueError("Oracle effective-time training requires model_name=IFRNet_Residual_FlowApprox.")
    if args.oracle_effective_time_source not in ORACLE_EFFECTIVE_TIME_SOURCES:
        available_sources = ", ".join(ORACLE_EFFECTIVE_TIME_SOURCES)
        raise ValueError(
            f"Unsupported oracle_effective_time_source={args.oracle_effective_time_source}. "
            f"Available sources: {available_sources}"
        )
    if args.oracle_effective_time_epsilon <= 0.0:
        raise ValueError(f"oracle_effective_time_epsilon must be positive, got {args.oracle_effective_time_epsilon}")
    if args.oracle_effective_time_min_motion < 0.0:
        raise ValueError(
            f"oracle_effective_time_min_motion must be non-negative, got {args.oracle_effective_time_min_motion}"
        )
    if args.oracle_effective_time_invalid_flow_scale < 0.0 or args.oracle_effective_time_invalid_flow_scale > 1.0:
        raise ValueError(
            "oracle_effective_time_invalid_flow_scale must be in [0, 1], "
            f"got {args.oracle_effective_time_invalid_flow_scale}"
        )
    if args.init_flow_downscale_strategy == "masked_area" and not is_splatting_flow_approx_method(args.flow_approx_method):
        raise ValueError("init_flow_downscale_strategy=masked_area requires a splatting flow approximation method.")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default=None, type=str)
    bootstrap_args, _remaining_argv = bootstrap_parser.parse_known_args(argv)

    config_path = None if bootstrap_args.config is None else Path(bootstrap_args.config)
    config_defaults = load_train_run_config(config_path)
    parser = build_train_arg_parser(config_defaults)
    parser.description = "Train IFRNet_Residual_FlowApprox with oracle t_eff injected into flow initialization."
    parser.add_argument(
        "--oracle-effective-time-source",
        default=config_defaults.get("oracle_effective_time_source", DEFAULT_ORACLE_EFFECTIVE_TIME_SOURCE),
        choices=ORACLE_EFFECTIVE_TIME_SOURCES,
        help="Which oracle t_eff direction to inject into the shared effective-time map.",
    )
    parser.add_argument(
        "--oracle-effective-time-epsilon",
        default=config_defaults.get("oracle_effective_time_epsilon", DEFAULT_ORACLE_EFFECTIVE_TIME_EPSILON),
        type=float,
        help="Denominator epsilon for projecting rendered 60fps flow onto 30fps endpoint flow.",
    )
    parser.add_argument(
        "--oracle-effective-time-min-motion",
        default=config_defaults.get("oracle_effective_time_min_motion", DEFAULT_ORACLE_EFFECTIVE_TIME_MIN_MOTION),
        type=float,
        help="Minimum 30fps endpoint-flow magnitude required before oracle projection is trusted.",
    )
    parser.add_argument(
        "--oracle-effective-time-invalid-flow-scale",
        default=config_defaults.get(
            "oracle_effective_time_invalid_flow_scale",
            DEFAULT_ORACLE_EFFECTIVE_TIME_INVALID_FLOW_SCALE,
        ),
        type=float,
        help="Fallback scale for invalid oracle projection. 0.5 means fixed midpoint fallback.",
    )
    parser.add_argument(
        "--oracle-effective-time-cache-dir",
        default=config_defaults.get("oracle_effective_time_cache_dir"),
        type=str,
        help="Directory for offline oracle t_eff .npy cache files. Defaults to output_dir/oracle_t_eff_cache.",
    )
    parser.add_argument(
        "--oracle-effective-time-cache-dtype",
        default=config_defaults.get("oracle_effective_time_cache_dtype", DEFAULT_ORACLE_EFFECTIVE_TIME_CACHE_DTYPE),
        choices=ORACLE_EFFECTIVE_TIME_CACHE_DTYPES,
        help="Storage dtype for cached oracle t_eff maps.",
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
    if args.oracle_effective_time_cache_dir is None:
        args.oracle_effective_time_cache_dir = str(Path(args.output_dir) / "oracle_t_eff_cache")
    args.model_init_args = read_model_init_args(config_defaults)
    args.metric_config = read_metric_config(config_defaults)
    args.input_config = dict(config_defaults)
    args.input_config["oracle_effective_time_source"] = args.oracle_effective_time_source
    args.input_config["oracle_effective_time_epsilon"] = args.oracle_effective_time_epsilon
    args.input_config["oracle_effective_time_min_motion"] = args.oracle_effective_time_min_motion
    args.input_config["oracle_effective_time_invalid_flow_scale"] = args.oracle_effective_time_invalid_flow_scale
    args.input_config["oracle_effective_time_cache_dir"] = args.oracle_effective_time_cache_dir
    args.input_config["oracle_effective_time_cache_dtype"] = args.oracle_effective_time_cache_dtype
    args.input_config["rebuild_oracle_effective_time_cache"] = args.rebuild_oracle_effective_time_cache
    require_psnr_enabled(args.metric_config, "oracle effective-time training")
    validate_oracle_args(args)
    return args


def validate_flow_pair(endpoint_flow: Any, target_flow: Any) -> None:
    if endpoint_flow.ndim != 3 or endpoint_flow.shape[2] != 2:
        raise ValueError(f"endpoint_flow must have shape [H, W, 2], got {tuple(endpoint_flow.shape)}")
    if tuple(endpoint_flow.shape) != tuple(target_flow.shape):
        raise ValueError(
            f"Flow shape mismatch: endpoint_flow={tuple(endpoint_flow.shape)} target_flow={tuple(target_flow.shape)}"
        )


def projection_scale_numpy(
    endpoint_flow: np.ndarray,
    target_flow: np.ndarray,
    epsilon: float,
    min_motion: float,
    invalid_flow_scale: float,
) -> ProjectionScaleResult:
    validate_flow_pair(endpoint_flow, target_flow)
    denominator = np.sum(endpoint_flow * endpoint_flow, axis=2, keepdims=True)
    raw_scale = np.sum(endpoint_flow * target_flow, axis=2, keepdims=True) / (denominator + epsilon)
    valid = np.sqrt(denominator) > min_motion
    scale = np.where(valid, np.clip(raw_scale, 0.0, 1.0), invalid_flow_scale).astype(np.float32)
    return ProjectionScaleResult(scale=scale, valid=valid, raw_scale=raw_scale)


def build_oracle_cache_key(args: argparse.Namespace, row: Any) -> str:
    key_payload = {
        "version": 1,
        "record": str(row["record"]),
        "mode": str(row["mode"]),
        "fps": int(row["fps"]),
        "img0": int(row["img0"]),
        "img1": int(row["img1"]),
        "img2": int(row["img2"]),
        "source": str(args.oracle_effective_time_source),
        "epsilon": float(args.oracle_effective_time_epsilon),
        "min_motion": float(args.oracle_effective_time_min_motion),
        "invalid_flow_scale": float(args.oracle_effective_time_invalid_flow_scale),
        "dtype": str(args.oracle_effective_time_cache_dtype),
    }
    serialized = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def build_oracle_cache_path(cache_dir: Path, args: argparse.Namespace, row: Any) -> Path:
    return cache_dir / f"{build_oracle_cache_key(args, row)}.npy"


def build_modality_path(
    dataset_root_dir: Path,
    record: str,
    mode: str,
    frame_idx: int,
    modality_name: str,
) -> Path:
    modality_spec = DEFAULT_MODALITY_CONFIG[modality_name]
    base_dir = dataset_root_dir / record / mode
    subdir = str(modality_spec.get("subdir", ""))
    if subdir != "":
        base_dir = base_dir / subdir

    filename = f"{modality_spec['prefix']}{frame_idx}{modality_spec['ext']}"
    return base_dir / filename


def load_motion_for_oracle_cache(row: Any, dataset_root_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_30_0_idx = int(row["img0"]) // 2
    frame_30_1_idx = int(row["img2"]) // 2
    frame_60_1_idx = int(row["img1"])
    record = str(row["record"])
    mode = str(row["mode"])
    mode_30 = mode.replace("fps_60", "fps_30")

    bmv_60_path = build_modality_path(dataset_root_dir, record, mode, frame_60_1_idx, "backwardVel_Depth")
    fmv_60_path = build_modality_path(dataset_root_dir, record, mode, frame_60_1_idx, "forwardVel_Depth")
    bmv_30_path = build_modality_path(dataset_root_dir, record, mode_30, frame_30_1_idx, "backwardVel_Depth")
    fmv_30_path = build_modality_path(dataset_root_dir, record, mode_30, frame_30_0_idx, "forwardVel_Depth")
    bmv_60, _bmv_60_depth = load_backward_velocity(bmv_60_path)
    fmv_60, _fmv_60_depth = load_backward_velocity(fmv_60_path)
    bmv_30, _bmv_30_depth = load_backward_velocity(bmv_30_path)
    fmv_30, _fmv_30_depth = load_backward_velocity(fmv_30_path)
    return bmv_60, fmv_60, bmv_30, fmv_30


def combine_oracle_t_eff_numpy(
    fmv_t_eff: np.ndarray,
    fmv_valid: np.ndarray,
    bmv_t_eff: np.ndarray,
    bmv_valid: np.ndarray,
    oracle_effective_time_source: str,
) -> np.ndarray:
    fixed_time = np.full_like(fmv_t_eff, 0.5, dtype=np.float32)
    if oracle_effective_time_source == "fmv":
        return np.where(fmv_valid, fmv_t_eff, fixed_time).astype(np.float32)
    if oracle_effective_time_source == "bmv":
        return np.where(bmv_valid, bmv_t_eff, fixed_time).astype(np.float32)
    if oracle_effective_time_source == "average":
        both_valid = fmv_valid & bmv_valid
        fmv_only = fmv_valid & ~bmv_valid
        bmv_only = bmv_valid & ~fmv_valid
        average_time = 0.5 * (fmv_t_eff + bmv_t_eff)
        oracle_time = np.where(both_valid, average_time, fixed_time)
        oracle_time = np.where(fmv_only, fmv_t_eff, oracle_time)
        return np.where(bmv_only, bmv_t_eff, oracle_time).astype(np.float32)

    raise ValueError(f"Unsupported oracle_effective_time_source={oracle_effective_time_source}")


def calculate_oracle_t_eff_numpy(
    args: argparse.Namespace,
    bmv_30: np.ndarray,
    fmv_30: np.ndarray,
    bmv_60: np.ndarray,
    fmv_60: np.ndarray,
) -> np.ndarray:
    bmv_scale = projection_scale_numpy(
        endpoint_flow=bmv_30,
        target_flow=bmv_60,
        epsilon=args.oracle_effective_time_epsilon,
        min_motion=args.oracle_effective_time_min_motion,
        invalid_flow_scale=args.oracle_effective_time_invalid_flow_scale,
    )
    fmv_scale = projection_scale_numpy(
        endpoint_flow=fmv_30,
        target_flow=fmv_60,
        epsilon=args.oracle_effective_time_epsilon,
        min_motion=args.oracle_effective_time_min_motion,
        invalid_flow_scale=args.oracle_effective_time_invalid_flow_scale,
    )
    fmv_t_eff = fmv_scale.scale
    bmv_t_eff = 1.0 - bmv_scale.scale
    return combine_oracle_t_eff_numpy(
        fmv_t_eff=fmv_t_eff,
        fmv_valid=fmv_scale.valid,
        bmv_t_eff=bmv_t_eff,
        bmv_valid=bmv_scale.valid,
        oracle_effective_time_source=args.oracle_effective_time_source,
    ).clip(0.0, 1.0)


def save_oracle_t_eff_cache(cache_path: Path, oracle_t_eff: np.ndarray, cache_dtype: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    typed_oracle_t_eff = oracle_t_eff.astype(np.float16 if cache_dtype == "float16" else np.float32)
    temp_path = cache_path.with_suffix(".tmp.npy")
    np.save(temp_path, typed_oracle_t_eff)
    temp_path.replace(cache_path)


def ensure_oracle_effective_time_cache(
    dataframe: Any,
    dataset_root_dir: Path,
    cache_dir: Path,
    args: argparse.Namespace,
    logger: Any,
    split_name: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    written_count = 0
    skipped_count = 0
    for row_index in range(len(dataframe)):
        row = dataframe.iloc[row_index]
        cache_path = build_oracle_cache_path(cache_dir, args, row)
        if cache_path.is_file() and not args.rebuild_oracle_effective_time_cache:
            skipped_count += 1
            continue

        bmv_60, fmv_60, bmv_30, fmv_30 = load_motion_for_oracle_cache(row, dataset_root_dir)
        oracle_t_eff = calculate_oracle_t_eff_numpy(
            args=args,
            bmv_30=bmv_30,
            fmv_30=fmv_30,
            bmv_60=bmv_60,
            fmv_60=fmv_60,
        )
        save_oracle_t_eff_cache(cache_path, oracle_t_eff, args.oracle_effective_time_cache_dtype)
        written_count += 1
        if written_count % 100 == 0:
            logger.info(
                "oracle_t_eff_cache split=%s written=%s skipped=%s current=%s",
                split_name,
                written_count,
                skipped_count,
                cache_path,
            )

    logger.info(
        "oracle_t_eff_cache_ready split=%s cache_dir=%s written=%s skipped=%s total=%s",
        split_name,
        cache_dir,
        written_count,
        skipped_count,
        len(dataframe),
    )


class CachedOracleEffectiveTimeTrainDataset(BaseDataset):
    def __init__(
        self,
        dataframe: Any,
        dataset_root_dir: str,
        input_fps: int,
        augment: bool,
        include_source_depths: bool,
        oracle_cache_dir: Path,
        oracle_args: argparse.Namespace,
    ) -> None:
        super().__init__(dataframe, dataset_root_dir, input_fps, DEFAULT_MODALITY_CONFIG, None, None, None)
        self.augment = augment
        self.include_source_depths = include_source_depths
        self.oracle_cache_dir = oracle_cache_dir
        self.oracle_args = oracle_args

    def __len__(self) -> int:
        return len(self.dataframe)

    def _load_oracle_t_eff(self, row: Any) -> np.ndarray:
        cache_path = build_oracle_cache_path(self.oracle_cache_dir, self.oracle_args, row)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing offline oracle t_eff cache: path={cache_path} "
                f"record={row['record']} mode={row['mode']} img0={row['img0']} img1={row['img1']} img2={row['img2']}"
            )

        oracle_t_eff = np.load(cache_path, mmap_mode="r")
        if oracle_t_eff.ndim == 2:
            oracle_t_eff = oracle_t_eff[:, :, None]
        if oracle_t_eff.ndim != 3 or oracle_t_eff.shape[2] != 1:
            raise ValueError(f"oracle_t_eff cache must have shape [H, W, 1], got {tuple(oracle_t_eff.shape)}")
        return oracle_t_eff

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        if self.df_fps == self.input_fps:
            raise NotImplementedError("CachedOracleEffectiveTimeTrainDataset expects 30fps input against 60fps targets.")

        row = self.dataframe.iloc[index]
        frame_30_0_idx = int(row["img0"]) // 2
        frame_30_1_idx = int(row["img2"]) // 2
        frame_60_0_idx = int(row["img0"])
        frame_60_1_idx = int(row["img1"])
        frame_60_2_idx = int(row["img2"])
        record = str(row["record"])
        mode = str(row["mode"])
        mode_30 = mode.replace("fps_60", "fps_30")

        info = {
            "record_name": f"{record}_{mode}",
            "frame_range": f"frame_{frame_60_0_idx:04d}_{frame_60_2_idx:04d}",
            "valid": bool(row["valid"]) if "valid" in row.index else True,
            "distance_indexing": build_distance_indexing(row),
        }

        img_60_0_path = self._build_modality_path(record, mode, frame_60_0_idx, "colorNoScreenUI")
        img_60_1_path = self._build_modality_path(record, mode, frame_60_1_idx, "colorNoScreenUI")
        img_60_2_path = self._build_modality_path(record, mode, frame_60_2_idx, "colorNoScreenUI")
        bmv_60_path = self._build_modality_path(record, mode, frame_60_1_idx, "backwardVel_Depth")
        fmv_60_path = self._build_modality_path(record, mode, frame_60_1_idx, "forwardVel_Depth")
        bmv_30_path = self._build_modality_path(record, mode_30, frame_30_1_idx, "backwardVel_Depth")
        fmv_30_path = self._build_modality_path(record, mode_30, frame_30_0_idx, "forwardVel_Depth")

        img0 = self._load_image(img_60_0_path)
        imgt = self._load_image(img_60_1_path)
        img1 = self._load_image(img_60_2_path)
        bmv_60 = self._load_game_motion(bmv_60_path)
        fmv_60 = self._load_game_motion(fmv_60_path)
        oracle_t_eff = self._load_oracle_t_eff(row)
        if self.include_source_depths:
            bmv_30, source_depth1 = self._load_game_motion_and_depth(bmv_30_path)
            fmv_30, source_depth0 = self._load_game_motion_and_depth(fmv_30_path)
        else:
            bmv_30 = self._load_game_motion(bmv_30_path)
            fmv_30 = self._load_game_motion(fmv_30_path)
            source_depth0 = None
            source_depth1 = None

        if self.augment:
            flow_fields = (bmv_60, fmv_60, bmv_30, fmv_30, oracle_t_eff)
            if self.include_source_depths:
                flow_fields = (bmv_60, fmv_60, bmv_30, fmv_30, oracle_t_eff, source_depth0, source_depth1)

            img0, imgt, img1, flow_fields = shared_random_crop(img0, imgt, img1, flow_fields, (224, 224))
            img0, imgt, img1 = shared_random_reverse_channel(img0, imgt, img1, 0.5)
            img0, imgt, img1, flow_fields = shared_random_vertical_flip(img0, imgt, img1, flow_fields, 0.3)
            img0, imgt, img1, flow_fields = shared_random_horizontal_flip(img0, imgt, img1, flow_fields, 0.5)
            img0, imgt, img1, flow_fields = shared_random_rotate(img0, imgt, img1, flow_fields, 0.05)
            if self.include_source_depths:
                bmv_60, fmv_60, bmv_30, fmv_30, oracle_t_eff, source_depth0, source_depth1 = flow_fields
            else:
                bmv_60, fmv_60, bmv_30, fmv_30, oracle_t_eff = flow_fields

        img0_tensor = image_to_tensor(img0)
        imgt_tensor = image_to_tensor(imgt)
        img1_tensor = image_to_tensor(img1)
        bmv_60_tensor = flow_to_tensor(bmv_60)
        fmv_60_tensor = flow_to_tensor(fmv_60)
        bmv_30_tensor = flow_to_tensor(bmv_30)
        fmv_30_tensor = flow_to_tensor(fmv_30)
        oracle_t_eff_tensor = depth_to_tensor(oracle_t_eff)
        embt_tensor = build_embedding_tensor()

        if self.include_source_depths:
            info["source_depth0"] = depth_to_tensor(source_depth0)
            info["source_depth1"] = depth_to_tensor(source_depth1)

        return (
            img0_tensor,
            imgt_tensor,
            img1_tensor,
            bmv_60_tensor,
            fmv_60_tensor,
            bmv_30_tensor,
            fmv_30_tensor,
            oracle_t_eff_tensor,
            embt_tensor,
            info,
        )


def build_oracle_t_eff_stats(oracle_t_eff: Any) -> dict[str, list[float]]:
    flat_t_eff = oracle_t_eff.detach().float().flatten(start_dim=1)
    return {
        "oracle_t_eff_mean": [float(value) for value in flat_t_eff.mean(dim=1).cpu().tolist()],
        "oracle_t_eff_std": [float(value) for value in flat_t_eff.std(dim=1, unbiased=False).cpu().tolist()],
        "oracle_t_eff_min": [float(value) for value in flat_t_eff.min(dim=1).values.cpu().tolist()],
        "oracle_t_eff_max": [float(value) for value in flat_t_eff.max(dim=1).values.cpu().tolist()],
    }


def run_oracle_training_batch(
    args: argparse.Namespace,
    model: Any,
    batch: Any,
    device: Any,
) -> OracleBatchStepOutput:
    img0, imgt, img1, bmv_60, fmv_60, bmv_30, fmv_30, oracle_t_eff, embt, info = batch
    img0 = img0.to(device)
    imgt = imgt.to(device)
    img1 = img1.to(device)
    bmv_60 = bmv_60.to(device)
    fmv_60 = fmv_60.to(device)
    bmv_30 = bmv_30.to(device)
    fmv_30 = fmv_30.to(device)
    oracle_t_eff = oracle_t_eff.to(device)
    embt = embt.to(device)

    source_depth0 = None
    source_depth1 = None
    if is_splatting_flow_approx_method(flow_approx_method=args.flow_approx_method):
        source_depth0 = info["source_depth0"].to(device)
        source_depth1 = info["source_depth1"].to(device)

    flow_init = build_flow_init_result_with_fill_strategy(
        fmv_30=fmv_30,
        bmv_30=bmv_30,
        embt=embt,
        flow_approx_method=args.flow_approx_method,
        source_depth0=source_depth0,
        source_depth1=source_depth1,
        splatting_fill_strategy=args.splatting_fill_strategy,
        ground_truth_bmv=bmv_60,
        ground_truth_fmv=fmv_60,
        effective_time=oracle_t_eff,
    )

    init_bmv_mask = None
    init_fmv_mask = None
    if args.init_flow_downscale_strategy == "masked_area":
        if flow_init.masks is None:
            raise RuntimeError("init_flow_downscale_strategy=masked_area requires splatting coverage masks.")
        init_bmv_mask = flow_init.masks[:, 0:1]
        init_fmv_mask = flow_init.masks[:, 1:2]

    model_output = model(
        img0,
        img1,
        embt,
        imgt,
        init_flow0=flow_init.bmv,
        init_flow1=flow_init.fmv,
        init_flow0_mask=init_bmv_mask,
        init_flow1_mask=init_fmv_mask,
        init_flow_mask_epsilon=args.init_flow_mask_epsilon,
    )
    imgt_pred, loss_rec, loss_geo, loss_dis, _up_flow0_1, _up_flow1_1, _up_mask_1 = model_output
    return OracleBatchStepOutput(
        imgt=imgt,
        imgt_pred=imgt_pred,
        info=info,
        loss_rec=loss_rec,
        loss_geo=loss_geo,
        loss_dis=loss_dis,
        oracle_t_eff=oracle_t_eff,
    )


def build_loss_record(loss_rec: Any, loss_geo: Any, loss_dis: Any, total_loss: Any) -> dict[str, float]:
    return {
        "loss_rec": float(loss_rec.detach().cpu()),
        "loss_geo": float(loss_geo.detach().cpu()),
        "loss_dis": float(loss_dis.detach().cpu()),
        "loss_total": float(total_loss.detach().cpu()),
    }


def append_oracle_batch_metric_records(
    target_records: list[dict[str, object]],
    metric_meters: dict[str, AverageMeter],
    batch_output: OracleBatchStepOutput,
    loss_record: dict[str, float],
    metric_config: dict[str, object],
    lpips_model: Any | None,
) -> None:
    batch_size = int(batch_output.imgt_pred.shape[0])
    batch_metric_values = calculate_batch_metrics(
        batch_output.imgt.detach(),
        batch_output.imgt_pred.detach(),
        metric_config,
        lpips_model,
    )
    oracle_t_eff_stats = build_oracle_t_eff_stats(batch_output.oracle_t_eff)
    normalized_loss_record = {metric_name: float(metric_value) for metric_name, metric_value in loss_record.items()}

    for batch_index in range(batch_size):
        sample_metric_values = {
            metric_name: float(metric_values[batch_index])
            for metric_name, metric_values in batch_metric_values.items()
        }
        for metric_name, metric_value in sample_metric_values.items():
            metric_meters[metric_name].update(metric_value, 1)
        target_records.append(
            {
                "record_name": batch_output.info["record_name"][batch_index],
                "frame_range": batch_output.info["frame_range"][batch_index],
                **sample_metric_values,
                "oracle_t_eff_mean": oracle_t_eff_stats["oracle_t_eff_mean"][batch_index],
                "oracle_t_eff_std": oracle_t_eff_stats["oracle_t_eff_std"][batch_index],
                "oracle_t_eff_min": oracle_t_eff_stats["oracle_t_eff_min"][batch_index],
                "oracle_t_eff_max": oracle_t_eff_stats["oracle_t_eff_max"][batch_index],
                **normalized_loss_record,
            }
        )


def evaluate_oracle(args: argparse.Namespace, model: Any, loader: Any, device: Any, lpips_model: Any | None) -> tuple[float, Any, Any, dict[str, float]]:
    import pandas as pd
    import torch
    from tqdm import tqdm

    model.eval()
    metric_meters = build_metric_meters(args.metric_config)
    eval_records: list[dict[str, object]] = []

    with torch.no_grad():
        progress = tqdm(loader, desc="Evaluating oracle t_eff")
        for batch in progress:
            batch_output = run_oracle_training_batch(args, model, batch, device)
            total_loss = batch_output.loss_rec + batch_output.loss_geo + batch_output.loss_dis
            loss_record = build_loss_record(batch_output.loss_rec, batch_output.loss_geo, batch_output.loss_dis, total_loss)
            append_oracle_batch_metric_records(
                eval_records,
                metric_meters,
                batch_output,
                loss_record,
                args.metric_config,
                lpips_model,
            )
            progress.set_postfix({"eval_psnr": f"{metric_meters['psnr'].avg:.6f}"})

    eval_df = pd.DataFrame(eval_records)
    record_name_df = build_record_name_summary(eval_df, args.metric_config)
    return metric_meters["psnr"].avg, eval_df, record_name_df, average_metric_values(metric_meters)


def train_oracle(
    args: argparse.Namespace,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    logger: Any,
    training_state: Any,
    lpips_model: Any | None,
) -> None:
    import pandas as pd
    from tqdm import tqdm

    best_psnr = training_state.best_psnr
    global_step = training_state.global_step
    checkpoints_dir = Path(args.output_dir) / "checkpoints"

    for epoch in range(training_state.start_epoch, args.epochs):
        model.train()
        train_metric_meters = build_metric_meters(args.metric_config)
        train_loss_total_meter = AverageMeter()
        train_loss_rec_meter = AverageMeter()
        train_loss_geo_meter = AverageMeter()
        train_loss_dis_meter = AverageMeter()
        train_records: list[dict[str, object]] = []
        progress = tqdm(train_loader, desc=f"Oracle Epoch {epoch + 1}/{args.epochs}")

        for batch in progress:
            lr = get_lr(args, global_step)
            set_lr(optimizer, lr)
            optimizer.zero_grad()

            batch_output = run_oracle_training_batch(args, model, batch, device)
            total_loss = batch_output.loss_rec + batch_output.loss_geo + batch_output.loss_dis
            total_loss.backward()
            optimizer.step()

            batch_size = int(batch_output.imgt.shape[0])
            train_loss_total_meter.update(float(total_loss.detach().cpu()), batch_size)
            train_loss_rec_meter.update(float(batch_output.loss_rec.detach().cpu()), batch_size)
            train_loss_geo_meter.update(float(batch_output.loss_geo.detach().cpu()), batch_size)
            train_loss_dis_meter.update(float(batch_output.loss_dis.detach().cpu()), batch_size)

            global_step += 1
            progress.set_postfix(
                {
                    "train_loss": f"{float(total_loss.detach().cpu()):.6f}",
                    "avg_train_loss": f"{train_loss_total_meter.avg:.6f}",
                    "lr": f"{lr:.6f}",
                }
            )
            if (epoch + 1) % args.eval_interval != 0:
                continue

            loss_record = build_loss_record(batch_output.loss_rec, batch_output.loss_geo, batch_output.loss_dis, total_loss)
            append_oracle_batch_metric_records(
                train_records,
                train_metric_meters,
                batch_output,
                loss_record,
                args.metric_config,
                lpips_model,
            )

        if (epoch + 1) % args.eval_interval == 0:
            train_df = pd.DataFrame(train_records)
            train_record_name_df = build_record_name_summary(train_df, args.metric_config)
            train_df.to_csv(checkpoints_dir / f"train_epoch_{epoch + 1}.csv", index=False)
            train_record_name_df.to_csv(checkpoints_dir / f"train_epoch_{epoch + 1}_record_name.csv", index=False)
            logger.info(
                "Epoch %s train_loss_total=%.6f train_loss_rec=%.6f train_loss_geo=%.6f train_loss_dis=%.6f train_metrics=%s",
                epoch + 1,
                train_loss_total_meter.avg,
                train_loss_rec_meter.avg,
                train_loss_geo_meter.avg,
                train_loss_dis_meter.avg,
                format_metric_averages(train_metric_meters),
            )
        else:
            logger.info(
                "Epoch %s train_loss_total=%.6f train_loss_rec=%.6f train_loss_geo=%.6f train_loss_dis=%.6f",
                epoch + 1,
                train_loss_total_meter.avg,
                train_loss_rec_meter.avg,
                train_loss_geo_meter.avg,
                train_loss_dis_meter.avg,
            )

        if (epoch + 1) % args.eval_interval == 0:
            test_psnr, test_df, test_record_name_df, test_metric_values = evaluate_oracle(
                args,
                model,
                test_loader,
                device,
                lpips_model,
            )
            test_df.to_csv(checkpoints_dir / f"test_epoch_{epoch + 1}.csv", index=False)
            test_record_name_df.to_csv(checkpoints_dir / f"test_epoch_{epoch + 1}_record_name.csv", index=False)
            logger.info("Epoch %s test_metrics=%s", epoch + 1, format_metric_values(test_metric_values))

            if test_psnr > best_psnr:
                best_psnr = test_psnr
                save_checkpoint(checkpoints_dir / "best.pth", model, optimizer, epoch, best_psnr)
                logger.info("New Best PSNR - Epoch %s test_psnr=%.6f", epoch + 1, test_psnr)

        save_checkpoint(checkpoints_dir / f"epoch_{epoch + 1}.pth", model, optimizer, epoch, best_psnr)
        save_checkpoint(checkpoints_dir / "latest.pth", model, optimizer, epoch, best_psnr)


def build_dry_run_summary(args: argparse.Namespace) -> dict[str, object]:
    return {
        "mode": args.mode,
        "script": "experiments/scratch/train_oracle_effective_time.py",
        "model_name": args.model_name,
        "model_init_args": dict(args.model_init_args),
        "train_preset": args.train_preset,
        "test_preset": args.test_preset,
        "root_dir": args.root_dir,
        "dataset_root_dir": args.dataset_root_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_interval": args.eval_interval,
        "flow_approx_method": args.flow_approx_method,
        "splatting_fill_strategy": args.splatting_fill_strategy,
        "effective_splatting_fill_strategy": resolve_effective_splatting_fill_strategy(args),
        "init_flow_downscale_strategy": args.init_flow_downscale_strategy,
        "effective_init_flow_downscale_strategy": resolve_effective_init_flow_downscale_strategy(args),
        "oracle_effective_time_source": args.oracle_effective_time_source,
        "oracle_effective_time_epsilon": args.oracle_effective_time_epsilon,
        "oracle_effective_time_min_motion": args.oracle_effective_time_min_motion,
        "oracle_effective_time_invalid_flow_scale": args.oracle_effective_time_invalid_flow_scale,
        "oracle_effective_time_cache_dir": args.oracle_effective_time_cache_dir,
        "oracle_effective_time_cache_dtype": args.oracle_effective_time_cache_dtype,
        "rebuild_oracle_effective_time_cache": args.rebuild_oracle_effective_time_cache,
        "pretrained_checkpoint_path": args.pretrained_checkpoint_path,
        "resume_path": args.resume_path,
        "metrics": dict(args.metric_config),
    }


def run_training(args: argparse.Namespace) -> None:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger, run_dir = build_logger(output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = build_lpips_model(args.metric_config, device)
    logger.info("device=%s", device)
    logger.info("oracle_effective_time_source=%s", args.oracle_effective_time_source)
    logger.info(
        "oracle_effective_time_epsilon=%s oracle_effective_time_min_motion=%s oracle_effective_time_invalid_flow_scale=%s",
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

    root_dir = Path(args.root_dir)
    train_df = build_merged_dataframe(root_dir, checkpoints_dir, args.train_preset, args.only_fps, logger)
    test_df = build_merged_dataframe(root_dir, checkpoints_dir, args.test_preset, args.only_fps, logger)

    if "valid" in train_df.columns:
        logger.info("Valid Count %s in %s", train_df["valid"].value_counts().to_dict(), args.train_preset)
        train_df = train_df[train_df["valid"] == True]
    if "valid" in test_df.columns:
        logger.info("Valid Count %s in %s", test_df["valid"].value_counts().to_dict(), args.test_preset)
        test_df = test_df[test_df["valid"] == True]

    oracle_cache_dir = Path(args.oracle_effective_time_cache_dir)
    ensure_oracle_effective_time_cache(
        train_df,
        Path(args.dataset_root_dir),
        oracle_cache_dir,
        args,
        logger,
        "train",
    )
    ensure_oracle_effective_time_cache(
        test_df,
        Path(args.dataset_root_dir),
        oracle_cache_dir,
        args,
        logger,
        "test",
    )

    include_source_depths = is_splatting_flow_approx_method(flow_approx_method=args.flow_approx_method)
    train_dataset = CachedOracleEffectiveTimeTrainDataset(
        train_df.reset_index(drop=True),
        args.dataset_root_dir,
        args.input_fps,
        True,
        include_source_depths,
        oracle_cache_dir,
        args,
    )
    test_dataset = CachedOracleEffectiveTimeTrainDataset(
        test_df.reset_index(drop=True),
        args.dataset_root_dir,
        args.input_fps,
        False,
        include_source_depths,
        oracle_cache_dir,
        args,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    args.iters_per_epoch = len(train_loader)
    model_class = resolve_model_class(args.model_name)
    model = model_class(**dict(args.model_init_args))
    attach_effective_time_estimator(model=model, effective_time_mode="disabled", effective_time_hidden_channels=32)
    model = model.to(device)
    if hasattr(model, "init_flow_layer"):
        logger.info("model_init_flow_layer=%s", model.init_flow_layer)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0)
    training_state = load_training_state(args, model, optimizer, device, logger)
    logger.info(
        "model=%s mode=%s train_samples=%s test_samples=%s start_epoch=%s epochs=%s batch_size=%s",
        args.model_name,
        training_state.mode,
        len(train_dataset),
        len(test_dataset),
        training_state.start_epoch,
        args.epochs,
        args.batch_size,
    )
    logger.info("run_log_dir=%s", run_dir)
    input_config_path = save_input_config(target_dir=run_dir, input_config=args.input_config)
    logger.info("input_config_path=%s", input_config_path)

    train_oracle(args, model, optimizer, train_loader, test_loader, device, logger, training_state, lpips_model)


def main(argv: list[str] | None) -> None:
    args = parse_args(argv)
    if args.mode == "dry-run":
        print(json.dumps(build_dry_run_summary(args), indent=2))
        return

    run_training(args)


if __name__ == "__main__":
    main(sys.argv[1:])
