from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.clipping import get_valid_continuous_segments
from src.data.dataset import DatasetSample
from src.data.dataset_config import DatasetConfig
from src.data.image_ops import identical_images
from src.data.image_ops import load_backward_velocity
from src.data.image_ops import load_exr
from src.data.image_ops import load_png

FRAME_GLOB_PATTERN: str = "colorNoScreenUI_*.exr"
IMAGE_NAME_TEMPLATE: str = "colorNoScreenUI_{frame_idx}.png"
COLOR_EXR_TEMPLATE: str = "colorNoScreenUI_{frame_idx}.exr"
BACKWARD_VELOCITY_TEMPLATE: str = "backwardVel_Depth_{frame_idx}.exr"
FORWARD_VELOCITY_TEMPLATE: str = "forwardVel_Depth_{frame_idx}.exr"
NON_FINITE_REASON_PREFIX: str = "Non-finite EXR values"
LINEARITY_REASON_COLUMN: str = "linearity_reason"
LINEARITY_REASON_NO_VALID_RATIO: str = "No valid ratio pixels"
LINEARITY_REASON_OUT_OF_RANGE: str = "Linearity ratio out of range"
RATIO_DENOMINATOR_EPSILON: float = 1e-12


def rewrite_sample_root(samples: list[DatasetSample], source_root: Path, target_root: Path) -> list[DatasetSample]:
    """Rewrite sample paths from one root directory to another."""
    normalized_samples: list[DatasetSample] = []
    for sample in samples:
        normalized_input_frames: list[str] = [
            str(rewrite_path_root(path=Path(frame_path), source_root=source_root, target_root=target_root))
            for frame_path in sample["input_frames"]
        ]
        normalized_target_frame: str = str(
            rewrite_path_root(path=Path(sample["target_frame"]), source_root=source_root, target_root=target_root),
        )
        normalized_sample: DatasetSample = DatasetSample(
            sample_id=sample["sample_id"],
            input_frames=normalized_input_frames,
            target_frame=normalized_target_frame,
            split=sample["split"],
            metadata=dict(sample["metadata"]),
        )
        normalized_samples.append(normalized_sample)

    return normalized_samples


def rewrite_path_root(path: Path, source_root: Path, target_root: Path) -> Path:
    """Rewrite one path from a source root to a target root."""
    relative_path: Path = path.relative_to(source_root)
    return target_root / relative_path


def build_frame_index_for_mode(root_dir: Path, record: str, mode: str) -> Any:
    """Build one raw frame-index dataframe for a record and mode directory."""
    rows: list[dict[str, object]] = []
    mode_root = root_dir / record / mode
    frame_files = sorted(glob(str(mode_root / FRAME_GLOB_PATTERN)))

    for frame_file in frame_files:
        name = Path(frame_file).name
        idx_str = name.split("_")[-1].split(".")[0]
        frame_idx = int(idx_str)
        rows.append(
            {
                "record": record,
                "mode": mode,
                "frame_idx": frame_idx,
                "is_valid": True,
                "reason": "",
            },
        )

    dataframe = pd.DataFrame(rows)
    if "reason" in dataframe.columns:
        dataframe["reason"] = dataframe["reason"].astype(str)

    return dataframe


def normalize_reason_column(dataframe: Any) -> Any:
    """Ensure the reason column is always present and string-typed."""
    if "reason" not in dataframe.columns:
        dataframe["reason"] = ""
    else:
        dataframe["reason"] = dataframe["reason"].fillna("").astype(str)

    return dataframe


def append_reason(existing_reason: str, new_reason: str) -> str:
    """Append one reason without duplicating the same message."""
    if existing_reason == "":
        return new_reason
    if new_reason in existing_reason:
        return existing_reason

    return f"{existing_reason}; {new_reason}"


def build_exr_paths_for_frame(mode_dir: Path, frame_idx: int) -> dict[str, Path]:
    """Build one EXR path mapping for the frame modalities checked during preprocessing."""
    return {
        "colorNoScreenUI": mode_dir / COLOR_EXR_TEMPLATE.format(frame_idx=frame_idx),
        "backwardVel_Depth": mode_dir / BACKWARD_VELOCITY_TEMPLATE.format(frame_idx=frame_idx),
        "forwardVel_Depth": mode_dir / FORWARD_VELOCITY_TEMPLATE.format(frame_idx=frame_idx),
    }


def find_non_finite_exr_modalities(mode_dir: Path, frame_idx: int) -> list[str]:
    """Return modality names whose EXR content contains NaN or Inf values."""
    invalid_modalities: list[str] = []
    exr_paths = build_exr_paths_for_frame(mode_dir, frame_idx)

    for modality_name, exr_path in exr_paths.items():
        exr_data = load_exr(exr_path)
        if not np.isfinite(exr_data).all():
            invalid_modalities.append(modality_name)

    return invalid_modalities


def mark_non_finite_frames_invalid(raw_df: Any, root_dir: Path) -> Any:
    """Mark frames invalid when any required EXR modality contains NaN or Inf values."""
    raw_df = normalize_reason_column(raw_df)
    invalid_count = 0
    progress_desc = "check_non_finite_exr"
    if len(raw_df) > 0:
        progress_desc = f"{raw_df.iloc[0]['record']}:{raw_df.iloc[0]['mode']}"
    progress = tqdm(range(len(raw_df)), desc=progress_desc)

    for row_index in progress:
        row = raw_df.iloc[row_index]
        frame_idx = int(row["frame_idx"])
        mode_dir = root_dir / str(row["record"]) / str(row["mode"])
        invalid_modalities = find_non_finite_exr_modalities(mode_dir, frame_idx)

        if len(invalid_modalities) > 0:
            reason = f"{NON_FINITE_REASON_PREFIX}: {', '.join(invalid_modalities)}"
            raw_df.at[row_index, "is_valid"] = False
            raw_df.at[row_index, "reason"] = append_reason(str(raw_df.at[row_index, "reason"]), reason)
            invalid_count += 1

        progress.set_postfix({"current_frame": frame_idx, "invalid_count": invalid_count})

    return raw_df


def remove_identical_frames(raw_df: Any, root_dir: Path) -> Any:
    """Mark frames as invalid when they are identical to the previous frame."""
    invalid_count = 0
    progress = tqdm(range(1, len(raw_df)))
    for row_index in progress:
        current_row = raw_df.iloc[row_index]
        previous_row = raw_df.iloc[row_index - 1]

        current_img_name = IMAGE_NAME_TEMPLATE.format(frame_idx=current_row["frame_idx"])
        previous_img_name = IMAGE_NAME_TEMPLATE.format(frame_idx=previous_row["frame_idx"])
        dir_path = root_dir / str(current_row["record"]) / str(current_row["mode"])

        img1_path = dir_path / previous_img_name
        img2_path = dir_path / current_img_name
        img1 = load_png(img1_path).astype("uint8")
        img2 = load_png(img2_path).astype("uint8")

        if identical_images(img1, img2):
            raw_df.at[row_index, "is_valid"] = False
            raw_df.at[row_index, "reason"] = "Identical to previous frame"
            invalid_count += 1

        progress.set_postfix({"current_frame": current_row["frame_idx"], "invalid_count": invalid_count})

    return raw_df


def check_identical_images_cross_fps(fps_30_df: Any, fps_60_df: Any, root_dir: Path) -> None:
    """Print mismatched 30fps and 60fps frames for one mode pair."""
    progress = tqdm(range(1, len(fps_30_df)))
    for row_index in progress:
        current_row_30 = fps_30_df.iloc[row_index]
        current_row_60 = fps_60_df.iloc[2 * row_index]

        img_name_30 = IMAGE_NAME_TEMPLATE.format(frame_idx=current_row_30["frame_idx"])
        img_name_60 = IMAGE_NAME_TEMPLATE.format(frame_idx=current_row_60["frame_idx"])
        dir_path_30 = root_dir / str(current_row_30["record"]) / str(current_row_30["mode"])
        dir_path_60 = root_dir / str(current_row_60["record"]) / str(current_row_60["mode"])

        img_30 = load_png(dir_path_30 / img_name_30).astype("uint8")
        img_60 = load_png(dir_path_60 / img_name_60).astype("uint8")

        if not identical_images(img_30, img_60):
            print(
                "Non-identical frames found "
                f"fps30={current_row_30['frame_idx']} fps60={current_row_60['frame_idx']}",
            )

        progress.set_postfix({"frame_30": current_row_30["frame_idx"], "frame_60": current_row_60["frame_idx"]})


def cosine_project_ratio(array1: Any, array2: Any) -> Any:
    """Project one flow field onto another and return the normalized ratio."""
    array_inner = array1[..., 0] * array2[..., 0] + array1[..., 1] * array2[..., 1]
    array2_mag = np.linalg.norm(array2, axis=-1)
    denominator = array2_mag ** 2
    valid_mask = (
        np.isfinite(array_inner)
        & np.isfinite(denominator)
        & (denominator > RATIO_DENOMINATOR_EPSILON)
    )
    ratio = np.full(array_inner.shape, np.nan, dtype=np.float32)
    ratio[valid_mask] = array_inner[valid_mask] / denominator[valid_mask]
    return ratio


def build_valid_clip_windows(dataframe: Any, target_frames_count: int) -> list[dict[str, int]]:
    """Expand valid contiguous segments into fixed-length sliding windows."""
    segments = get_valid_continuous_segments(dataframe)
    clip_windows: list[dict[str, int]] = []

    for segment in segments:
        if segment["length"] < target_frames_count:
            continue

        last_start = segment["end"] - target_frames_count + 1
        for start_frame in range(segment["start"], last_start + 1):
            clip_windows.append({"start": start_frame, "end": start_frame + target_frames_count - 1})

    return clip_windows


def merge_easy_medium_dataframes(easy_df: Any, medium_df: Any) -> Any:
    """Merge easy and medium dataframes with one shared validity column."""
    merged_df = easy_df.merge(
        medium_df,
        on=["record", "frame_idx"],
        how="inner",
        suffixes=("_easy", "_medium"),
    )
    merged_df["global_is_valid"] = merged_df["is_valid_easy"] & merged_df["is_valid_medium"]
    return merged_df


def build_difficult_only_dataframe(dataframe: Any) -> Any:
    """Copy one difficulty dataframe and expose the shared validity column."""
    difficult_df = dataframe.copy()
    difficult_df["global_is_valid"] = difficult_df["is_valid"]
    return difficult_df


def build_raw_sequence_dataframe(df_30: Any, df_60: Any, dataset_config: DatasetConfig) -> Any:
    """Build one raw sequence dataframe from preprocessed 30fps and 60fps frame flags."""
    rows: list[dict[str, object]] = []
    for frame_idx in range(0, dataset_config.max_index - 1, 2):
        fps_30_img_2_flag = df_30.at[frame_idx // 2 + 1, "global_is_valid"]
        fps_60_img_1_flag = df_60.at[frame_idx + 1, "global_is_valid"]
        fps_60_img_2_flag = df_60.at[frame_idx + 2, "global_is_valid"]
        rows.append(
            {
                "record": dataset_config.record,
                "fps": dataset_config.fps,
                "img0": frame_idx,
                "img1": frame_idx + 1,
                "img2": frame_idx + 2,
                "valid": bool(fps_30_img_2_flag and fps_60_img_1_flag and fps_60_img_2_flag),
            },
        )

    return pd.DataFrame(rows)


def apply_linearity_check(raw_seq_df: Any, root_dir: Path, dataset_config: DatasetConfig) -> Any:
    """Append linearity statistics to one raw sequence dataframe."""
    raw_seq_df["D_index Mean"] = [-1.0] * len(raw_seq_df)
    raw_seq_df["D_index Median"] = [-1.0] * len(raw_seq_df)
    raw_seq_df[LINEARITY_REASON_COLUMN] = [""] * len(raw_seq_df)

    invalid_count = 0
    progress = tqdm(range(len(raw_seq_df)))
    for row_index in progress:
        row = raw_seq_df.iloc[row_index]
        if not bool(row["valid"]):
            progress.set_postfix({"skipped_invalid": row_index + 1, "invalid_count": invalid_count})
            continue

        img_1_idx = int(row["img1"])
        img_2_idx = int(row["img2"])

        fps_30_mode_path = dataset_config.mode_path.replace("fps_60", "fps_30")
        fps_30_dir = root_dir / dataset_config.record_name / fps_30_mode_path
        fps_60_dir = root_dir / dataset_config.record_name / dataset_config.mode_path

        backward_vel_2_0, _ = load_backward_velocity(
            fps_30_dir / BACKWARD_VELOCITY_TEMPLATE.format(frame_idx=img_2_idx // 2),
        )
        backward_vel_1_0, _ = load_backward_velocity(
            fps_60_dir / BACKWARD_VELOCITY_TEMPLATE.format(frame_idx=img_1_idx),
        )

        dis_index = cosine_project_ratio(backward_vel_1_0, backward_vel_2_0)
        finite_dis_index = dis_index[np.isfinite(dis_index)]
        if finite_dis_index.size == 0:
            raw_seq_df.at[row_index, "valid"] = False
            raw_seq_df.at[row_index, LINEARITY_REASON_COLUMN] = LINEARITY_REASON_NO_VALID_RATIO
            invalid_count += 1
            progress.set_postfix(
                {
                    "valid_ratio_pixels": 0,
                    "invalid_count": invalid_count,
                },
            )
            continue

        dis_index_mean = float(np.mean(finite_dis_index))
        dis_index_median = float(np.median(finite_dis_index))

        raw_seq_df.at[row_index, "D_index Mean"] = dis_index_mean
        raw_seq_df.at[row_index, "D_index Median"] = dis_index_median

        if is_out_of_ratio_range(dis_index_mean) or is_out_of_ratio_range(dis_index_median):
            raw_seq_df.at[row_index, "valid"] = False
            raw_seq_df.at[row_index, LINEARITY_REASON_COLUMN] = LINEARITY_REASON_OUT_OF_RANGE
            invalid_count += 1

        progress.set_postfix(
            {
                "D_index Mean": dis_index_mean,
                "D_index Median": dis_index_median,
                "valid_ratio_pixels": int(finite_dis_index.size),
                "invalid_count": invalid_count,
            },
        )

    return raw_seq_df


def is_out_of_ratio_range(value: float) -> bool:
    """Check whether one ratio is outside the expected [0, 1] range after rounding."""
    rounded_value = round(value, 2)
    return rounded_value < 0.0 or rounded_value > 1.0


def build_frame_index_csv_path(data_root: Path, record_name: str, mode_name: str) -> Path:
    """Build one CSV path for frame-index data."""
    return data_root / record_name / f"{mode_name}_frame_index.csv"


def build_preprocessed_csv_path(data_root: Path, record_name: str, mode_index: str, suffix: str) -> Path:
    """Build one CSV path under the record-level preprocessed directory."""
    return data_root / f"{record_name}_preprocessed" / f"{mode_index}_{suffix}.csv"
