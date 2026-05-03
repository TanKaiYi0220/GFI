from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import get_dataset_preset
from src.data.dataset_config import iter_dataset_configs
from src.data.dataset_config import resolve_active_dataset_root
from src.data.dataset_loader import build_distance_indexing
from src.data.dataset_loader import build_embedding_tensor
from src.data.dataset_loader import flow_to_tensor
from src.data.dataset_loader import image_to_tensor
from src.data.image_ops import load_backward_velocity
from src.data.image_ops import load_png


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache VFI training samples into one .pt file per sample.")
    parser.add_argument("--root-dir", required=True, type=str, help="Directory containing preprocessed CSV indexes.")
    parser.add_argument("--dataset-root-dir", default=None, type=str, help="Root directory containing frame and velocity assets.")
    parser.add_argument("--paths-config", default=None, type=str, help="Optional path to configs/paths/default.yaml.")
    parser.add_argument("--preset", required=True, type=str, help="Dataset preset name, such as train_vfx_0416.")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory where cached samples and manifest will be written.")
    parser.add_argument("--only-fps", default=60, type=int, help="Use CSV entries for this FPS only.")
    parser.add_argument("--limit", default=None, type=int, help="Optional number of samples to cache for a quick benchmark.")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16", help="Tensor dtype used for cached sample tensors.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files.")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip rows where valid is False.")
    return parser


def build_logger() -> logging.Logger:
    logger = logging.getLogger("GFICacheSamples")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def resolve_dataset_root_dir(dataset_root_dir: str | None, paths_config: str | None) -> str:
    if dataset_root_dir is not None:
        return dataset_root_dir

    paths_config_path = None if paths_config is None else Path(paths_config)
    return str(resolve_active_dataset_root(paths_config_path))


def build_merged_dataframe(root_dir: Path, preset_name: str, only_fps: int, logger: logging.Logger) -> pd.DataFrame:
    dataset_preset = get_dataset_preset(preset_name)
    dataframe_list: list[pd.DataFrame] = []

    for dataset_config in iter_dataset_configs(dataset_preset):
        if dataset_config.fps != only_fps:
            continue

        csv_path = root_dir / f"{dataset_config.record_name}_preprocessed" / f"{dataset_config.mode_index}_raw_sequence_frame_index.csv"
        if not csv_path.is_file():
            logger.warning("Dataset CSV missing: %s", csv_path)
            continue

        dataframe = pd.read_csv(csv_path)
        dataframe["record"] = dataset_config.record
        dataframe["mode"] = dataset_config.mode_path
        dataframe_list.append(dataframe)
        logger.info("Loaded dataset CSV %s rows=%s", csv_path, len(dataframe))

    if len(dataframe_list) == 0:
        raise RuntimeError(f"No dataset CSV found under root_dir={root_dir} for preset={preset_name}")

    merged_dataframe = pd.concat(dataframe_list, ignore_index=True)
    logger.info("Merged dataset size=%s preset=%s", len(merged_dataframe), preset_name)
    return merged_dataframe


def build_modality_path(
    dataset_root_dir: str,
    record: str,
    mode: str,
    frame_index: int,
    prefix: str,
    extension: str,
) -> Path:
    return Path(dataset_root_dir) / record / mode / f"{prefix}{frame_index}{extension}"


def sanitize_cache_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_cache_path(
    output_dir: Path,
    record: str,
    mode: str,
    frame_0_idx: int,
    frame_1_idx: int,
    frame_2_idx: int,
) -> Path:
    return output_dir / "samples" / record / mode / f"{frame_0_idx:04d}_{frame_1_idx:04d}_{frame_2_idx:04d}.pt"


def load_sample_tensors(
    dataset_root_dir: str,
    record: str,
    mode: str,
    frame_0_idx: int,
    frame_1_idx: int,
    frame_2_idx: int,
    cache_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    img_0_path = build_modality_path(dataset_root_dir, record, mode, frame_0_idx, "colorNoScreenUI_", ".png")
    img_1_path = build_modality_path(dataset_root_dir, record, mode, frame_1_idx, "colorNoScreenUI_", ".png")
    img_2_path = build_modality_path(dataset_root_dir, record, mode, frame_2_idx, "colorNoScreenUI_", ".png")
    backward_path = build_modality_path(dataset_root_dir, record, mode, frame_1_idx, "backwardVel_Depth_", ".exr")
    forward_path = build_modality_path(dataset_root_dir, record, mode, frame_1_idx, "forwardVel_Depth_", ".exr")

    img0 = load_png(img_0_path)[:, :, :3]
    imgt = load_png(img_1_path)[:, :, :3]
    img1 = load_png(img_2_path)[:, :, :3]
    bmv, _backward_depth = load_backward_velocity(backward_path)
    fmv, _forward_depth = load_backward_velocity(forward_path)

    return {
        "img0": image_to_tensor(img0).to(dtype=cache_dtype),
        "imgt": image_to_tensor(imgt).to(dtype=cache_dtype),
        "img1": image_to_tensor(img1).to(dtype=cache_dtype),
        "bmv": flow_to_tensor(bmv).to(dtype=cache_dtype),
        "fmv": flow_to_tensor(fmv).to(dtype=cache_dtype),
        "embt": build_embedding_tensor(),
    }


def build_sample_payload(
    row: pd.Series,
    dataset_root_dir: str,
    cache_dtype: torch.dtype,
) -> dict[str, Any]:
    frame_0_idx = int(row["img0"])
    frame_1_idx = int(row["img1"])
    frame_2_idx = int(row["img2"])
    record = str(row["record"])
    mode = str(row["mode"])

    tensors = load_sample_tensors(
        dataset_root_dir=dataset_root_dir,
        record=record,
        mode=mode,
        frame_0_idx=frame_0_idx,
        frame_1_idx=frame_1_idx,
        frame_2_idx=frame_2_idx,
        cache_dtype=cache_dtype,
    )

    return {
        "tensors": tensors,
        "info": {
            "record": record,
            "mode": mode,
            "frame_range": f"frame_{frame_0_idx:04d}_{frame_2_idx:04d}",
            "img0": frame_0_idx,
            "img1": frame_1_idx,
            "img2": frame_2_idx,
            "valid": bool(row["valid"]) if "valid" in row.index else True,
            "distance_indexing": build_distance_indexing(row),
            "cache_dtype": str(cache_dtype),
        },
    }


def write_cache_manifest(manifest_records: list[dict[str, Any]], output_dir: Path) -> Path:
    manifest_path = output_dir / "manifest.csv"
    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_path


def main() -> None:
    args = build_arg_parser().parse_args()
    logger = build_logger()
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root_dir = resolve_dataset_root_dir(args.dataset_root_dir, args.paths_config)
    cache_dtype = sanitize_cache_dtype(args.dtype)
    dataframe = build_merged_dataframe(root_dir, args.preset, args.only_fps, logger)

    if args.skip_invalid and "valid" in dataframe.columns:
        dataframe = dataframe[dataframe["valid"] == True].reset_index(drop=True)
        logger.info("Filtered valid rows size=%s", len(dataframe))

    if args.limit is not None:
        dataframe = dataframe.iloc[: args.limit].reset_index(drop=True)
        logger.info("Limited rows size=%s", len(dataframe))

    manifest_records: list[dict[str, Any]] = []
    written_count = 0
    skipped_count = 0

    for row_index in tqdm(range(len(dataframe)), total=len(dataframe), desc="Caching samples"):
        row_series = dataframe.iloc[row_index]
        frame_0_idx = int(row_series["img0"])
        frame_1_idx = int(row_series["img1"])
        frame_2_idx = int(row_series["img2"])
        record = str(row_series["record"])
        mode = str(row_series["mode"])
        cache_path = build_cache_path(output_dir, record, mode, frame_0_idx, frame_1_idx, frame_2_idx)

        if cache_path.is_file() and not args.overwrite:
            skipped_count += 1
        else:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = build_sample_payload(
                row=row_series,
                dataset_root_dir=dataset_root_dir,
                cache_dtype=cache_dtype,
            )
            torch.save(payload, cache_path)
            written_count += 1

        manifest_records.append(
            {
                "row_index": row_index,
                "record": record,
                "mode": mode,
                "img0": frame_0_idx,
                "img1": frame_1_idx,
                "img2": frame_2_idx,
                "valid": bool(row_series["valid"]) if "valid" in row_series.index else True,
                "cache_path": str(cache_path),
            }
        )

    manifest_path = write_cache_manifest(manifest_records, output_dir)
    logger.info("Caching complete written=%s skipped=%s manifest=%s", written_count, skipped_count, manifest_path)


if __name__ == "__main__":
    main()
