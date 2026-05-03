from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_logger
from scripts.train import build_merged_dataframe
from src.data.dataset_config import list_dataset_presets
from src.data.dataset_config import resolve_active_dataset_root
from src.data.dataset_loader import DEFAULT_MODALITY_CONFIG
from src.data.dataset_loader import build_distance_indexing
from src.data.image_ops import load_backward_velocity
from src.data.image_ops import load_png


def parse_bool_argument(value: str) -> bool:
    normalized_value = value.strip().lower()
    if normalized_value in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean-like value, got: {value}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack VFI samples into .npz files and validate round-trip precision.")
    parser.add_argument("--root-dir", default="./datasets/data", help="Directory containing preprocessed CSV indexes.")
    parser.add_argument("--dataset-root-dir", default=None, type=str, help="Root directory containing frame and velocity assets.")
    parser.add_argument("--paths-config", default=None, type=str, help="Optional path to configs/paths/default.yaml.")
    parser.add_argument("--train-preset", default="train_vfx_0416", choices=list_dataset_presets())
    parser.add_argument("--test-preset", default="test_vfx_0416", choices=list_dataset_presets())
    parser.add_argument("--only-fps", default=60, type=int, help="Use CSV entries for this FPS only.")
    parser.add_argument("--cache-target", default="both", choices=["train", "test", "both"])
    parser.add_argument("--cache-dir", default=None, type=str, help="Directory where .npz cache files and reports will be written.")
    parser.add_argument("--limit", default=None, type=int, help="Optional maximum sample count per preset.")
    parser.add_argument("--overwrite", default=False, type=parse_bool_argument, help="Whether existing .npz files should be replaced.")
    parser.add_argument(
        "--npz-compression",
        default="stored",
        choices=["stored", "deflated"],
        help="Use plain .npz for maximum load speed or compressed .npz for smaller files.",
    )
    parser.add_argument(
        "--validate-roundtrip",
        default=True,
        type=parse_bool_argument,
        help="Reload each written .npz file and compare every modality against the source arrays.",
    )
    return parser


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.dataset_root_dir is None:
        paths_config_path = None if args.paths_config is None else Path(args.paths_config)
        args.dataset_root_dir = str(resolve_active_dataset_root(paths_config_path))

    if args.cache_dir is None:
        args.cache_dir = str(PROJECT_ROOT / "outputs" / "npz_cache")

    return args


def build_cache_output_dir(cache_root: Path, preset_name: str) -> Path:
    return cache_root / preset_name


def build_manifest_output_path(cache_output_dir: Path, preset_name: str) -> Path:
    return cache_output_dir / f"{preset_name}_manifest.csv"


def build_validation_output_path(cache_output_dir: Path, preset_name: str) -> Path:
    return cache_output_dir / f"{preset_name}_validation.csv"


def build_summary_output_path(cache_output_dir: Path, preset_name: str) -> Path:
    return cache_output_dir / f"{preset_name}_summary.json"


def build_cache_file_path(
    cache_output_dir: Path,
    record: str,
    mode: str,
    frame_0_idx: int,
    frame_1_idx: int,
    frame_2_idx: int,
) -> Path:
    return cache_output_dir / record / mode / f"frame_{frame_0_idx:04d}_{frame_1_idx:04d}_{frame_2_idx:04d}.npz"


def build_modality_path(
    dataset_root_dir: str,
    record: str,
    mode: str,
    frame_idx: int,
    modality_name: str,
) -> Path:
    modality_spec = DEFAULT_MODALITY_CONFIG[modality_name]
    base_dir = Path(dataset_root_dir) / record / mode
    subdir = str(modality_spec.get("subdir", ""))
    if subdir != "":
        base_dir = base_dir / subdir

    filename = f"{modality_spec['prefix']}{frame_idx}{modality_spec['ext']}"
    return base_dir / filename


def load_image_rgb(image_path: Path) -> np.ndarray:
    image = load_png(image_path)
    return np.ascontiguousarray(image[:, :, :3])


def load_motion(velocity_path: Path) -> np.ndarray:
    motion, _depth = load_backward_velocity(velocity_path)
    return np.ascontiguousarray(motion)


def build_sample_info(row: pd.Series, frame_0_idx: int, frame_2_idx: int) -> dict[str, Any]:
    return {
        "frame_range": f"frame_{frame_0_idx:04d}_{frame_2_idx:04d}",
        "valid": bool(row["valid"]) if "valid" in row.index else True,
        "distance_indexing": build_distance_indexing(row),
    }


def load_sample_arrays(row: pd.Series, dataset_root_dir: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frame_0_idx = int(row["img0"])
    frame_1_idx = int(row["img1"])
    frame_2_idx = int(row["img2"])
    record = str(row["record"])
    mode = str(row["mode"])

    img_0_path = build_modality_path(dataset_root_dir, record, mode, frame_0_idx, "colorNoScreenUI")
    img_1_path = build_modality_path(dataset_root_dir, record, mode, frame_1_idx, "colorNoScreenUI")
    img_2_path = build_modality_path(dataset_root_dir, record, mode, frame_2_idx, "colorNoScreenUI")
    backward_path = build_modality_path(dataset_root_dir, record, mode, frame_1_idx, "backwardVel_Depth")
    forward_path = build_modality_path(dataset_root_dir, record, mode, frame_1_idx, "forwardVel_Depth")

    arrays = {
        "img0": load_image_rgb(img_0_path),
        "imgt": load_image_rgb(img_1_path),
        "img1": load_image_rgb(img_2_path),
        "bmv": load_motion(backward_path),
        "fmv": load_motion(forward_path),
    }
    info = build_sample_info(row, frame_0_idx, frame_2_idx)
    return arrays, info


def save_npz_file(
    cache_path: Path,
    arrays: dict[str, np.ndarray],
    npz_compression: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if npz_compression == "stored":
        np.savez(cache_path, **arrays)
        return

    np.savez_compressed(cache_path, **arrays)


def load_npz_file(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as payload:
        return {key: np.ascontiguousarray(payload[key]) for key in payload.files}


def compute_array_diff_metrics(source_array: np.ndarray, cached_array: np.ndarray) -> dict[str, Any]:
    shape_match = source_array.shape == cached_array.shape
    dtype_match = source_array.dtype == cached_array.dtype

    if not shape_match:
        return {
            "shape_match": False,
            "dtype_match": dtype_match,
            "exact_match": False,
            "max_abs_diff": float("inf"),
            "mean_abs_diff": float("inf"),
        }

    exact_match = bool(np.array_equal(source_array, cached_array))
    source_float = source_array.astype(np.float64, copy=False)
    cached_float = cached_array.astype(np.float64, copy=False)
    diff = np.abs(source_float - cached_float)
    max_abs_diff = float(diff.max()) if diff.size > 0 else 0.0
    mean_abs_diff = float(diff.mean()) if diff.size > 0 else 0.0

    return {
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "exact_match": exact_match,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def build_validation_record(
    cache_path: Path,
    arrays: dict[str, np.ndarray],
    cached_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    validation_record: dict[str, Any] = {
        "cache_path": str(cache_path),
        "raw_bytes": int(sum(array.nbytes for array in arrays.values())),
        "cache_bytes": int(cache_path.stat().st_size),
    }
    validation_record["cache_to_raw_ratio"] = (
        float(validation_record["cache_bytes"]) / float(validation_record["raw_bytes"])
        if int(validation_record["raw_bytes"]) > 0
        else 0.0
    )

    has_mismatch = False
    for array_name, source_array in arrays.items():
        cached_array = cached_arrays[array_name]
        metrics = compute_array_diff_metrics(source_array, cached_array)
        for metric_name, metric_value in metrics.items():
            validation_record[f"{array_name}_{metric_name}"] = metric_value
        has_mismatch = has_mismatch or not bool(metrics["exact_match"])

    validation_record["all_modalities_exact_match"] = not has_mismatch
    return validation_record


def build_manifest_record(
    cache_path: Path,
    row: pd.Series,
    info: dict[str, Any],
) -> dict[str, Any]:
    distance_indexing = list(info["distance_indexing"])
    return {
        "cache_path": str(cache_path),
        "record": str(row["record"]),
        "mode": str(row["mode"]),
        "img0": int(row["img0"]),
        "img1": int(row["img1"]),
        "img2": int(row["img2"]),
        "fps": int(row["fps"]) if "fps" in row.index else -1,
        "valid": bool(info["valid"]),
        "frame_range": str(info["frame_range"]),
        "distance_index_mean": float(distance_indexing[0]),
        "distance_index_median": float(distance_indexing[1]),
    }


def limit_dataframe(dataframe: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return dataframe.reset_index(drop=True)

    return dataframe.iloc[:limit].reset_index(drop=True)


def summarize_validation(validation_df: pd.DataFrame, preset_name: str, npz_compression: str) -> dict[str, Any]:
    if len(validation_df) == 0:
        return {
            "preset_name": preset_name,
            "npz_compression": npz_compression,
            "sample_count": 0,
        }

    summary: dict[str, Any] = {
        "preset_name": preset_name,
        "npz_compression": npz_compression,
        "sample_count": int(len(validation_df)),
        "exact_sample_count": int(validation_df["all_modalities_exact_match"].sum()),
        "exact_sample_rate": float(validation_df["all_modalities_exact_match"].mean()),
        "total_raw_bytes": int(validation_df["raw_bytes"].sum()),
        "total_cache_bytes": int(validation_df["cache_bytes"].sum()),
        "cache_to_raw_ratio_mean": float(validation_df["cache_to_raw_ratio"].mean()),
        "cache_to_raw_ratio_total": (
            float(validation_df["cache_bytes"].sum()) / float(validation_df["raw_bytes"].sum())
            if int(validation_df["raw_bytes"].sum()) > 0
            else 0.0
        ),
        "modalities": {},
    }

    for array_name in ("img0", "imgt", "img1", "bmv", "fmv"):
        exact_column = f"{array_name}_exact_match"
        shape_column = f"{array_name}_shape_match"
        dtype_column = f"{array_name}_dtype_match"
        max_diff_column = f"{array_name}_max_abs_diff"
        mean_diff_column = f"{array_name}_mean_abs_diff"

        summary["modalities"][array_name] = {
            "shape_match_rate": float(validation_df[shape_column].mean()),
            "dtype_match_rate": float(validation_df[dtype_column].mean()),
            "exact_match_rate": float(validation_df[exact_column].mean()),
            "max_abs_diff_max": float(validation_df[max_diff_column].max()),
            "max_abs_diff_mean": float(validation_df[max_diff_column].mean()),
            "mean_abs_diff_mean": float(validation_df[mean_diff_column].mean()),
        }

    return summary


def write_summary(summary_path: Path, summary_payload: dict[str, Any]) -> None:
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def select_preset_names(args: argparse.Namespace) -> list[str]:
    if args.cache_target == "train":
        return [str(args.train_preset)]
    if args.cache_target == "test":
        return [str(args.test_preset)]

    return [str(args.train_preset), str(args.test_preset)]


def pack_preset(
    preset_name: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    cache_root = Path(args.cache_dir)
    cache_output_dir = build_cache_output_dir(cache_root, preset_name)
    cache_output_dir.mkdir(parents=True, exist_ok=True)

    root_dir = Path(args.root_dir)
    dataframe = build_merged_dataframe(root_dir, cache_output_dir, preset_name, args.only_fps, logger)
    if "valid" in dataframe.columns:
        logger.info("Valid Count %s in %s", dataframe["valid"].value_counts().to_dict(), preset_name)
        dataframe = dataframe[dataframe["valid"] == True]
    dataframe = limit_dataframe(dataframe, args.limit)

    manifest_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []

    logger.info("Packing preset=%s samples=%s cache_dir=%s", preset_name, len(dataframe), cache_output_dir)
    for _, row_series in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"pack:{preset_name}"):
        arrays, info = load_sample_arrays(row_series, args.dataset_root_dir)
        cache_path = build_cache_file_path(
            cache_output_dir=cache_output_dir,
            record=str(row_series["record"]),
            mode=str(row_series["mode"]),
            frame_0_idx=int(row_series["img0"]),
            frame_1_idx=int(row_series["img1"]),
            frame_2_idx=int(row_series["img2"]),
        )

        if args.overwrite or not cache_path.is_file():
            save_npz_file(cache_path, arrays, args.npz_compression)

        manifest_records.append(build_manifest_record(cache_path, row_series, info))

        if args.validate_roundtrip:
            cached_arrays = load_npz_file(cache_path)
            validation_records.append(build_validation_record(cache_path, arrays, cached_arrays))

    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = build_manifest_output_path(cache_output_dir, preset_name)
    manifest_df.to_csv(manifest_path, index=False)
    logger.info("Wrote manifest %s rows=%s", manifest_path, len(manifest_df))

    if args.validate_roundtrip:
        validation_df = pd.DataFrame(validation_records)
        validation_path = build_validation_output_path(cache_output_dir, preset_name)
        validation_df.to_csv(validation_path, index=False)
        logger.info("Wrote validation report %s rows=%s", validation_path, len(validation_df))

        summary_payload = summarize_validation(validation_df, preset_name, args.npz_compression)
        summary_path = build_summary_output_path(cache_output_dir, preset_name)
        write_summary(summary_path, summary_payload)
        logger.info("Wrote summary report %s", summary_path)
        logger.info(
            "preset=%s exact_sample_rate=%.6f total_cache_bytes=%s total_raw_bytes=%s cache_to_raw_ratio_total=%.6f",
            preset_name,
            summary_payload["exact_sample_rate"],
            summary_payload["total_cache_bytes"],
            summary_payload["total_raw_bytes"],
            summary_payload["cache_to_raw_ratio_total"],
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args = prepare_args(args)

    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    logger, run_dir = build_logger(cache_root)
    logger.info("cache_root=%s dataset_root_dir=%s run_log_dir=%s", cache_root, args.dataset_root_dir, run_dir)
    logger.info(
        "cache_target=%s npz_compression=%s validate_roundtrip=%s limit=%s overwrite=%s",
        args.cache_target,
        args.npz_compression,
        args.validate_roundtrip,
        args.limit,
        args.overwrite,
    )

    for preset_name in select_preset_names(args):
        pack_preset(preset_name, args, logger)


if __name__ == "__main__":
    main()
