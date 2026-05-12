from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import ACTIVE_DATASET_ROOT_KEY
from src.data.dataset_config import DatasetConfig
from src.data.dataset_config import get_dataset_preset
from src.data.dataset_config import iter_dataset_configs
from src.data.dataset_config import resolve_active_dataset_root
from src.data.manual_labeling import review_images
from src.data.preprocess import apply_linearity_check
from src.data.preprocess import build_difficult_only_dataframe
from src.data.preprocess import build_frame_index_csv_path
from src.data.preprocess import build_frame_index_for_mode
from src.data.preprocess import build_preprocessed_csv_path
from src.data.preprocess import build_raw_sequence_dataframe
from src.data.preprocess import check_identical_images_cross_fps
from src.data.preprocess import ensure_modality_validity_columns
from src.data.preprocess import mark_non_finite_frames_invalid
from src.data.preprocess import merge_easy_medium_dataframes
from src.data.preprocess import remove_identical_frames
from src.utils.io import ensure_directory


DATASET_PRESET_NAME: str = "train_vfx_0416"
TEST_DATASET_PRESET_NAME: str = "test_vfx_0416"
DATASET_ROOT_DIR_OVERRIDE: str | None = None
PATHS_CONFIG_PATH: str | None = None
DATA_DIR: str = "./data"
MERGE_STRATEGY: str = "only-difficult"
ONLY_FPS: int = 60

DRY_RUN: bool = True
REMOVE_IDENTICAL: bool = False
CHECK_NON_FINITE_EXR: bool = False
CHECK_IDENTICAL_CROSS_FPS: bool = False
MANUAL_LABELING: bool = False
MERGE_DATASETS: bool = True
RAW_SEQUENCE: bool = True
LINEARITY_CHECK: bool = True


def resolve_dataset_root_dir(dataset_root_dir: str | None, paths_config: str | None) -> Path:
    if dataset_root_dir is not None:
        return Path(dataset_root_dir)

    paths_config_path = Path(paths_config) if paths_config is not None else None
    return resolve_active_dataset_root(paths_config_path)


def get_target_configs(dataset_preset_name: str) -> list[DatasetConfig]:
    dataset_preset = get_dataset_preset(dataset_preset_name)
    return list(iter_dataset_configs(dataset_preset))


def should_use_sequence_config(dataset_config: DatasetConfig, merge_strategy: str) -> bool:
    if merge_strategy == "only-difficult":
        return dataset_config.difficulty == "Difficult"

    return dataset_config.difficulty == "Medium"


def print_run_summary(
    dataset_root_dir: Path,
    data_dir: Path,
    dataset_configs: list[DatasetConfig],
) -> None:
    print(f"dataset_preset={DATASET_PRESET_NAME}")
    print(f"active_root_key={ACTIVE_DATASET_ROOT_KEY}")
    print(f"dataset_root_dir={dataset_root_dir}")
    print(f"data_dir={data_dir}")
    print(f"config_count={len(dataset_configs)}")
    print(f"merge_strategy={MERGE_STRATEGY}")
    print(f"only_fps={ONLY_FPS}")


def load_or_build_frame_index_dataframe(dataset_root_dir: Path, data_dir: Path, dataset_config: DatasetConfig) -> Any:
    frame_index_path = build_frame_index_csv_path(data_dir, dataset_config.record_name, dataset_config.mode_name)
    if frame_index_path.is_file():
        dataframe = pd.read_csv(frame_index_path, dtype={"reason": "string"})
        dataframe = ensure_modality_validity_columns(dataframe)
        return dataframe.sort_values(by="frame_idx").reset_index(drop=True)

    dataframe = build_frame_index_for_mode(dataset_root_dir, dataset_config.record, dataset_config.mode_path)
    dataframe = ensure_modality_validity_columns(dataframe)
    return dataframe.sort_values(by="frame_idx").reset_index(drop=True)


def run_remove_identical(dataset_root_dir: Path, data_dir: Path, dataset_configs: list[DatasetConfig]) -> None:
    for dataset_config in dataset_configs:
        print(f"Processing record={dataset_config.record_name} mode={dataset_config.mode_path}")
        raw_df = build_frame_index_for_mode(dataset_root_dir, dataset_config.record, dataset_config.mode_path)
        raw_df = raw_df.sort_values(by="frame_idx").reset_index(drop=True)
        raw_df = remove_identical_frames(raw_df, dataset_root_dir)
        output_path = build_frame_index_csv_path(data_dir, dataset_config.record_name, dataset_config.mode_name)
        ensure_directory(output_path.parent)
        raw_df.to_csv(output_path, index=False)
        print(output_path)


def run_check_non_finite_exr(dataset_root_dir: Path, data_dir: Path, dataset_configs: list[DatasetConfig]) -> None:
    for dataset_config in dataset_configs:
        print(f"Checking EXR finite values record={dataset_config.record_name} mode={dataset_config.mode_path}")
        raw_df = load_or_build_frame_index_dataframe(dataset_root_dir, data_dir, dataset_config)
        raw_df = mark_non_finite_frames_invalid(raw_df, dataset_root_dir)
        output_path = build_frame_index_csv_path(data_dir, dataset_config.record_name, dataset_config.mode_name)
        ensure_directory(output_path.parent)
        raw_df.to_csv(output_path, index=False)
        print(output_path)


def run_check_cross_fps(dataset_root_dir: Path, data_dir: Path, dataset_configs: list[DatasetConfig]) -> None:
    for dataset_config in dataset_configs:
        if dataset_config.fps != 30:
            continue

        fps_60_name = dataset_config.mode_name.replace("fps_30", "fps_60")
        fps_30_path = build_frame_index_csv_path(data_dir, dataset_config.record_name, dataset_config.mode_name)
        fps_60_path = build_frame_index_csv_path(data_dir, dataset_config.record_name, fps_60_name)
        fps_30_df = pd.read_csv(fps_30_path, dtype={"reason": "string"})
        fps_60_df = pd.read_csv(fps_60_path, dtype={"reason": "string"})

        print(f"Checking record={dataset_config.record_name} mode={dataset_config.mode_name}")
        check_identical_images_cross_fps(fps_30_df, fps_60_df, dataset_root_dir)


def run_manual_labeling(dataset_root_dir: Path, data_dir: Path, dataset_configs: list[DatasetConfig]) -> None:
    for dataset_config in dataset_configs:
        if dataset_config.difficulty != "Medium":
            continue

        easy_mode_name = dataset_config.mode_name.replace("Medium", "Easy")
        easy_path = build_frame_index_csv_path(data_dir, dataset_config.record_name, easy_mode_name)
        medium_path = build_frame_index_csv_path(data_dir, dataset_config.record_name, dataset_config.mode_name)
        easy_df = pd.read_csv(easy_path, dtype={"reason": "string"})
        medium_df = pd.read_csv(medium_path, dtype={"reason": "string"})

        print(f"Manual labeling record={dataset_config.record_name} mode={dataset_config.mode_name}")
        review_images(easy_df, medium_df, dataset_root_dir)
        easy_df.to_csv(easy_path, index=False)
        medium_df.to_csv(medium_path, index=False)


def run_merge(data_dir: Path, dataset_configs: list[DatasetConfig], merge_strategy: str) -> None:
    for dataset_config in dataset_configs:
        if merge_strategy == "only-difficult":
            if dataset_config.difficulty != "Difficult":
                continue

            input_path = build_frame_index_csv_path(data_dir, dataset_config.record_name, dataset_config.mode_name)
            dataframe = pd.read_csv(input_path, dtype={"reason": "string"})
            merged_df = build_difficult_only_dataframe(dataframe)
        else:
            if dataset_config.difficulty != "Medium":
                continue

            medium_path = build_frame_index_csv_path(data_dir, dataset_config.record_name, dataset_config.mode_name)
            medium_df = pd.read_csv(medium_path, dtype={"reason": "string"})

            if merge_strategy == "ignore-easy":
                merged_df = medium_df.copy()
                merged_df["global_is_valid"] = medium_df["is_valid"]
            else:
                easy_mode_name = dataset_config.mode_name.replace("Medium", "Easy")
                easy_path = build_frame_index_csv_path(data_dir, dataset_config.record_name, easy_mode_name)
                easy_df = pd.read_csv(easy_path, dtype={"reason": "string"})
                merged_df = merge_easy_medium_dataframes(easy_df, medium_df)

        output_path = build_preprocessed_csv_path(
            data_dir,
            dataset_config.record_name,
            dataset_config.mode_index,
            "merged_frame_index",
        )
        ensure_directory(output_path.parent)
        merged_df.to_csv(output_path, index=False)
        print(output_path)


def run_raw_sequence(data_dir: Path, dataset_configs: list[DatasetConfig], merge_strategy: str, only_fps: int) -> None:
    for dataset_config in dataset_configs:
        if not should_use_sequence_config(dataset_config, merge_strategy):
            continue
        if dataset_config.fps != only_fps:
            continue

        fps_30_index = dataset_config.mode_index.replace("fps_60", "fps_30")
        df_30_path = build_preprocessed_csv_path(data_dir, dataset_config.record_name, fps_30_index, "merged_frame_index")
        df_60_path = build_preprocessed_csv_path(data_dir, dataset_config.record_name, dataset_config.mode_index, "merged_frame_index")
        df_30 = pd.read_csv(df_30_path, dtype={"reason_easy": "string", "reason_medium": "string"})
        df_60 = pd.read_csv(df_60_path, dtype={"reason_easy": "string", "reason_medium": "string"})

        raw_seq_df = build_raw_sequence_dataframe(df_30, df_60, dataset_config)
        output_path = build_preprocessed_csv_path(
            data_dir,
            dataset_config.record_name,
            dataset_config.mode_index,
            "raw_sequence_frame_index",
        )
        ensure_directory(output_path.parent)
        raw_seq_df.to_csv(output_path, index=False)
        print(output_path)


def run_linearity_check(dataset_root_dir: Path, data_dir: Path, dataset_configs: list[DatasetConfig], merge_strategy: str, only_fps: int) -> None:
    for dataset_config in dataset_configs:
        if not should_use_sequence_config(dataset_config, merge_strategy):
            continue
        if dataset_config.fps != only_fps:
            continue

        raw_sequence_path = build_preprocessed_csv_path(
            data_dir,
            dataset_config.record_name,
            dataset_config.mode_index,
            "raw_sequence_frame_index",
        )
        raw_seq_df = pd.read_csv(raw_sequence_path)
        raw_seq_df = apply_linearity_check(raw_seq_df, dataset_root_dir, dataset_config)
        raw_seq_df.to_csv(raw_sequence_path, index=False)
        print(raw_sequence_path)


def main() -> None:
    dataset_root_dir = resolve_dataset_root_dir(DATASET_ROOT_DIR_OVERRIDE, PATHS_CONFIG_PATH)
    data_dir = Path(DATA_DIR)
    dataset_configs = get_target_configs(DATASET_PRESET_NAME)

    if DRY_RUN:
        print_run_summary(dataset_root_dir, data_dir, dataset_configs)

    if REMOVE_IDENTICAL:
        run_remove_identical(dataset_root_dir, data_dir, dataset_configs)

    if CHECK_NON_FINITE_EXR:
        run_check_non_finite_exr(dataset_root_dir, data_dir, dataset_configs)

    if CHECK_IDENTICAL_CROSS_FPS:
        run_check_cross_fps(dataset_root_dir, data_dir, dataset_configs)

    if MANUAL_LABELING:
        run_manual_labeling(dataset_root_dir, data_dir, dataset_configs)

    if MERGE_DATASETS:
        run_merge(data_dir, dataset_configs, MERGE_STRATEGY)

    if RAW_SEQUENCE:
        run_raw_sequence(data_dir, dataset_configs, MERGE_STRATEGY, ONLY_FPS)

    if LINEARITY_CHECK:
        run_linearity_check(dataset_root_dir, data_dir, dataset_configs, MERGE_STRATEGY, ONLY_FPS)

    dataset_root_dir = resolve_dataset_root_dir(DATASET_ROOT_DIR_OVERRIDE, PATHS_CONFIG_PATH)
    data_dir = Path(DATA_DIR)
    dataset_configs = get_target_configs(TEST_DATASET_PRESET_NAME)

    if DRY_RUN:
        print_run_summary(dataset_root_dir, data_dir, dataset_configs)

    if REMOVE_IDENTICAL:
        run_remove_identical(dataset_root_dir, data_dir, dataset_configs)

    if CHECK_NON_FINITE_EXR:
        run_check_non_finite_exr(dataset_root_dir, data_dir, dataset_configs)

    if CHECK_IDENTICAL_CROSS_FPS:
        run_check_cross_fps(dataset_root_dir, data_dir, dataset_configs)

    if MANUAL_LABELING:
        run_manual_labeling(dataset_root_dir, data_dir, dataset_configs)

    if MERGE_DATASETS:
        run_merge(data_dir, dataset_configs, MERGE_STRATEGY)

    if RAW_SEQUENCE:
        run_raw_sequence(data_dir, dataset_configs, MERGE_STRATEGY, ONLY_FPS)

    if LINEARITY_CHECK:
        run_linearity_check(dataset_root_dir, data_dir, dataset_configs, MERGE_STRATEGY, ONLY_FPS)


if __name__ == "__main__":
    main()
