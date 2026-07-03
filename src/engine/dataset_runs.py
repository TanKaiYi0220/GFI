from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.data.dataset_config import get_dataset_preset
from src.data.dataset_config import iter_dataset_configs
from src.engine.flow_approx import is_splatting_flow_approx_method
from src.engine.model_registry import uses_flow_approx_model


def build_merged_dataframe(
    root_dir: Path,
    checkpoints_dir: Path,
    dataset_preset_name: str,
    only_fps: int,
    logger: logging.Logger,
) -> Any:
    import pandas as pd

    dataset_preset = get_dataset_preset(dataset_preset_name)
    dataframe_list: list[Any] = []

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
        raise RuntimeError(f"No dataset CSV found under root_dir={root_dir} for preset={dataset_preset_name}")

    merged = pd.concat(dataframe_list, ignore_index=True)
    merged.to_csv(checkpoints_dir / f"{dataset_preset_name}_merged.csv", index=False)
    logger.info("Merged dataset size=%s preset=%s", len(merged), dataset_preset_name)
    return merged


def filter_valid_dataframe(dataframe: Any) -> Any:
    if "valid" not in dataframe.columns:
        return dataframe
    return dataframe[dataframe["valid"] == True].reset_index(drop=True)


def build_training_dataset(
    dataframe: Any,
    dataset_root_dir: str,
    augment: bool,
    input_fps: int,
    model_name: str,
    flow_approx_method: str,
) -> Any:
    normalized_dataframe = dataframe.reset_index(drop=True)

    if uses_flow_approx_model(model_name):
        from src.data.dataset_loader import FlowEstimationTrainDataset

        include_source_depths = is_splatting_flow_approx_method(flow_approx_method=flow_approx_method)
        return FlowEstimationTrainDataset(normalized_dataframe, dataset_root_dir, input_fps, augment, include_source_depths)

    from src.data.dataset_loader import VFITrainDataset

    return VFITrainDataset(normalized_dataframe, dataset_root_dir, augment, input_fps)


def build_inference_dataset(
    dataframe: Any,
    dataset_root_dir: Path,
    input_fps: int,
    model_name: str,
    flow_approx_method: str,
) -> Any:
    if uses_flow_approx_model(model_name):
        from src.data.dataset_loader import FlowEstimationTrainDataset

        include_source_depths = is_splatting_flow_approx_method(flow_approx_method=flow_approx_method)
        return FlowEstimationTrainDataset(
            dataframe,
            str(dataset_root_dir),
            input_fps,
            False,
            include_source_depths,
        )

    from src.data.dataset_loader import VFITrainDataset

    return VFITrainDataset(dataframe, str(dataset_root_dir), False, input_fps)


def resolve_dataset_class_name(model_name: str) -> str:
    if uses_flow_approx_model(model_name):
        return "FlowEstimationTrainDataset"

    return "VFITrainDataset"
