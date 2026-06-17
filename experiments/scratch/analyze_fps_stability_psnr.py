from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CONFIG_DIR: Path = PROJECT_ROOT / "analysis_outputs" / ".matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_METRICS_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\CGV_inference_outputs\IFRNet_FineTuning_0611\metrics.csv",
)
DEFAULT_DATASET_JSON_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\Dataset",
)
DEFAULT_RAW_SEQUENCE_ROOT: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\Dataset\Minor_0507",
)
DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\fps_stability_psnr\ifrnet_finetuning_0611",
)
DEFAULT_TARGET_FPS: float = 60.0
DEFAULT_MOTION_BIN_WIDTH: float = 1.0
DEFAULT_MIN_BIN_SAMPLES: int = 20
DEFAULT_SCATTER_THRESHOLD_FPS: float = 2.0
DEFAULT_SCATTER_SYMMETRY_THRESHOLD_MS: float = 0.5
DEFAULT_HIGHLIGHT_SYMMETRY_THRESHOLD_MS: float = 0.5
DEFAULT_SCATTER_ORACLE_WINDOW_RADIUS: float = 0.025
DEFAULT_HIGHLIGHT_ORACLE_WINDOW_RADIUS: float = 0.025
DEFAULT_ORACLE_CENTER: float = 0.5
DEFAULT_FPS_THRESHOLD_VALUES: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0)
DEFAULT_SYMMETRY_THRESHOLD_MS_VALUES: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0)
DEFAULT_ORACLE_WINDOW_RADII: tuple[float, ...] = (0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25)
KEY_COLUMNS: tuple[str, ...] = ("record", "major_mode_id", "frame_range")
PSNR_COLUMN: str = "psnr"
MOTION_COLUMN: str = "motion_magnitude_mean"
SYMMETRY_COLUMN: str = "delta_time_symmetry_abs_ms"
ORACLE_COLUMN: str = "oracle_fmv_t_eff_mean"
ORACLE_DISTANCE_COLUMN: str = "oracle_fmv_t_eff_distance"
FRAME_RANGE_PATTERN = re.compile(r"frame_(\d+)_(\d+)")


@dataclass(frozen=True)
class FpsThresholdSpec:
    label: str
    tolerance_fps: float | None


@dataclass(frozen=True)
class SymmetryThresholdSpec:
    label: str
    tolerance_ms: float | None


@dataclass(frozen=True)
class OracleWindowSpec:
    label: str
    radius: float | None
    lower: float | None
    upper: float | None


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join per-frame delta time JSON with inference metrics and analyze FPS stability vs PSNR."
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=DEFAULT_METRICS_CSV,
        help="Path to model inference metrics.csv.",
    )
    parser.add_argument(
        "--dataset-json-dir",
        type=Path,
        default=DEFAULT_DATASET_JSON_DIR,
        help="Directory containing per-frame deltaSecond JSON files.",
    )
    parser.add_argument(
        "--raw-sequence-root",
        type=Path,
        default=DEFAULT_RAW_SEQUENCE_ROOT,
        help="Root containing *_raw_sequence_frame_index.csv files with motion statistics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for joined CSVs and figures.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=DEFAULT_TARGET_FPS,
        help="Target FPS used to compute stability deviation.",
    )
    parser.add_argument(
        "--motion-bin-width",
        type=float,
        default=DEFAULT_MOTION_BIN_WIDTH,
        help="Uniform motion bin width for threshold dashboard trend plots.",
    )
    parser.add_argument(
        "--min-bin-samples",
        type=int,
        default=DEFAULT_MIN_BIN_SAMPLES,
        help="Minimum samples required for a motion bin to appear in trend plots.",
    )
    parser.add_argument(
        "--scatter-threshold-fps",
        type=float,
        default=DEFAULT_SCATTER_THRESHOLD_FPS,
        help="FPS tolerance used to label stable vs unstable samples in scatter plots.",
    )
    parser.add_argument(
        "--fps-threshold-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_FPS_THRESHOLD_VALUES),
        help="FPS tolerance values swept by the dashboard.",
    )
    parser.add_argument(
        "--scatter-symmetry-threshold-ms",
        type=float,
        default=DEFAULT_SCATTER_SYMMETRY_THRESHOLD_MS,
        help="Delta-time symmetry tolerance, in milliseconds, used to label symmetric vs non-symmetric samples.",
    )
    parser.add_argument(
        "--highlight-symmetry-threshold-ms",
        type=float,
        default=DEFAULT_HIGHLIGHT_SYMMETRY_THRESHOLD_MS,
        help="Delta-time symmetry threshold highlighted in the gain dashboard.",
    )
    parser.add_argument(
        "--symmetry-threshold-ms-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_SYMMETRY_THRESHOLD_MS_VALUES),
        help="Delta-time symmetry tolerance values, in milliseconds, swept by the dashboard.",
    )
    parser.add_argument(
        "--oracle-center",
        type=float,
        default=DEFAULT_ORACLE_CENTER,
        help="Center value used for oracle_fmv_t_eff_mean clean-window filtering.",
    )
    parser.add_argument(
        "--scatter-oracle-window-radius",
        type=float,
        default=DEFAULT_SCATTER_ORACLE_WINDOW_RADIUS,
        help="Oracle window radius used to label clean vs non-clean samples in oracle scatter plots.",
    )
    parser.add_argument(
        "--highlight-oracle-window-radius",
        type=float,
        default=DEFAULT_HIGHLIGHT_ORACLE_WINDOW_RADIUS,
        help="Oracle window radius highlighted in the oracle clean-dataset dashboard.",
    )
    parser.add_argument(
        "--oracle-window-radii",
        type=float,
        nargs="+",
        default=list(DEFAULT_ORACLE_WINDOW_RADII),
        help="Oracle window radii around oracle-center swept by the dashboard.",
    )
    return parser.parse_args(argv)


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_columns(dataframe: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"{label} is missing columns: {missing_columns}")


def parse_major_mode_id(mode: str) -> str:
    mode_prefix = str(mode).replace("\\", "/").split("/")[0]
    mode_parts = mode_prefix.split("_")
    if len(mode_parts) < 2:
        raise ValueError(f"Unexpected mode format: mode={mode}")
    return mode_parts[0]


def parse_frame_range(frame_range: str) -> tuple[int, int]:
    match = FRAME_RANGE_PATTERN.fullmatch(str(frame_range))
    if match is None:
        raise ValueError(f"Unexpected frame_range format: frame_range={frame_range}")
    return int(match.group(1)), int(match.group(2))


def parse_sequence_json_name(json_path: Path) -> tuple[str, str, str]:
    stem_parts = json_path.stem.split("_")
    if len(stem_parts) < 3:
        raise ValueError(f"Unexpected JSON filename format: path={json_path}")
    record = "_".join(stem_parts[:-2])
    major_mode_id = stem_parts[-2]
    difficulty = stem_parts[-1]
    if record == "":
        raise ValueError(f"Could not parse record from JSON filename: path={json_path}")
    return record, major_mode_id, difficulty


def parse_raw_sequence_name(csv_path: Path) -> tuple[str, str]:
    parent_name = csv_path.parent.name
    if not parent_name.endswith("_preprocessed"):
        raise ValueError(f"Expected *_preprocessed parent directory: path={csv_path}")
    record = parent_name.replace("_preprocessed", "")

    stem = csv_path.name.replace("_raw_sequence_frame_index.csv", "")
    stem_parts = stem.split("_")
    if len(stem_parts) != 4 or stem_parts[2] != "fps":
        raise ValueError(f"Unexpected raw-sequence filename format: path={csv_path}")
    return record, stem_parts[0]


def format_frame_range(frame_start: int, frame_end: int) -> str:
    return f"frame_{frame_start:04d}_{frame_end:04d}"


def load_metrics(metrics_csv: Path) -> pd.DataFrame:
    require_existing_path(metrics_csv, "metrics CSV")
    metrics = pd.read_csv(metrics_csv)
    if len(metrics) == 0:
        raise ValueError(f"Metrics CSV is empty: {metrics_csv}")
    require_columns(
        dataframe=metrics,
        columns=("record", "mode", "frame_range", PSNR_COLUMN),
        label="metrics CSV",
    )
    metrics = metrics.copy()
    metrics["major_mode_id"] = metrics["mode"].map(parse_major_mode_id)

    duplicate_count = int(metrics.duplicated(subset=["record", "mode", "frame_range"]).sum())
    if duplicate_count > 0:
        raise ValueError(f"Metrics CSV contains duplicate sample keys: duplicate_count={duplicate_count}")
    return metrics


def load_sequence_delta_seconds(json_path: Path) -> dict[int, float]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if "frameContentList" not in payload:
        raise ValueError(f"JSON is missing frameContentList: path={json_path}")
    frame_rows = payload["frameContentList"]
    if not isinstance(frame_rows, list) or len(frame_rows) == 0:
        raise ValueError(f"frameContentList must be a non-empty list: path={json_path}")

    frame_delta_seconds: dict[int, float] = {}
    for frame_row in frame_rows:
        frame_index = int(frame_row["frameIdx"])
        delta_second = float(frame_row["deltaSecond"])
        if delta_second <= 0.0:
            raise ValueError(f"deltaSecond must be positive: path={json_path} frameIdx={frame_index} deltaSecond={delta_second}")
        frame_delta_seconds[frame_index] = delta_second
    return frame_delta_seconds


def build_required_sample_windows(metrics: pd.DataFrame) -> pd.DataFrame:
    windows = metrics[list(KEY_COLUMNS)].drop_duplicates().reset_index(drop=True)
    windows[["frame_start", "frame_end"]] = windows["frame_range"].apply(
        lambda value: pd.Series(parse_frame_range(str(value))),
    )
    return windows


def build_window_timing_row(
    record: str,
    major_mode_id: str,
    difficulty: str,
    frame_range: str,
    frame_start: int,
    frame_end: int,
    frame_delta_seconds: dict[int, float],
    target_fps: float,
) -> dict[str, object]:
    if frame_end <= frame_start:
        raise ValueError(f"frame_end must be greater than frame_start: frame_range={frame_range}")

    frame_indices = list(range(frame_start + 1, frame_end + 1))
    missing_indices = [frame_index for frame_index in frame_indices if frame_index not in frame_delta_seconds]
    if len(missing_indices) > 0:
        raise ValueError(
            f"JSON timing is missing sample frame indices: record={record} major_mode_id={major_mode_id} "
            f"frame_range={frame_range} missing_indices={missing_indices}"
        )
    if len(frame_indices) != 2:
        raise ValueError(
            f"Symmetric delta-time analysis expects exactly two frame intervals: "
            f"record={record} major_mode_id={major_mode_id} frame_range={frame_range} frame_indices={frame_indices}"
        )

    delta_seconds = np.asarray([frame_delta_seconds[frame_index] for frame_index in frame_indices], dtype=np.float64)
    fps_values = 1.0 / delta_seconds
    target_delta_second = 1.0 / target_fps
    delta_abs_error_ms = np.abs(delta_seconds - target_delta_second) * 1000.0
    fps_abs_error = np.abs(fps_values - target_fps)
    delta_second_1_to_0 = float(delta_seconds[0])
    delta_second_2_to_1 = float(delta_seconds[1])
    delta_time_sum = delta_second_1_to_0 + delta_second_2_to_1
    delta_time_position = delta_second_1_to_0 / delta_time_sum
    delta_time_symmetry_abs = abs(delta_second_1_to_0 - delta_second_2_to_1)
    delta_time_symmetry_ratio = max(delta_second_1_to_0, delta_second_2_to_1) / min(delta_second_1_to_0, delta_second_2_to_1)
    return {
        "record": record,
        "major_mode_id": major_mode_id,
        "json_difficulty": difficulty,
        "frame_range": frame_range,
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "timing_frame_count": int(len(frame_indices)),
        "delta_second_mean": float(delta_seconds.mean()),
        "delta_second_std": float(delta_seconds.std()),
        "delta_second_min": float(delta_seconds.min()),
        "delta_second_max": float(delta_seconds.max()),
        "delta_abs_error_mean_ms": float(delta_abs_error_ms.mean()),
        "delta_abs_error_max_ms": float(delta_abs_error_ms.max()),
        "fps_mean": float(fps_values.mean()),
        "fps_median": float(np.median(fps_values)),
        "fps_std": float(fps_values.std()),
        "fps_min": float(fps_values.min()),
        "fps_max": float(fps_values.max()),
        "fps_abs_error_mean": float(fps_abs_error.mean()),
        "fps_abs_error_max": float(fps_abs_error.max()),
        "delta_second_1_to_0": delta_second_1_to_0,
        "delta_second_2_to_1": delta_second_2_to_1,
        "delta_time_position": float(delta_time_position),
        "delta_time_position_error": float(abs(delta_time_position - 0.5)),
        "delta_time_symmetry_abs": float(delta_time_symmetry_abs),
        "delta_time_symmetry_abs_ms": float(delta_time_symmetry_abs * 1000.0),
        "delta_time_symmetry_ratio": float(delta_time_symmetry_ratio),
    }


def load_timing_rows(dataset_json_dir: Path, metrics: pd.DataFrame, target_fps: float) -> pd.DataFrame:
    require_existing_path(dataset_json_dir, "dataset JSON directory")
    required_windows = build_required_sample_windows(metrics)
    rows: list[dict[str, object]] = []
    matched_json_keys: set[tuple[str, str]] = set()

    for json_path in sorted(dataset_json_dir.glob("*.json")):
        record, major_mode_id, difficulty = parse_sequence_json_name(json_path)
        sample_windows = required_windows[
            (required_windows["record"] == record)
            & (required_windows["major_mode_id"] == major_mode_id)
        ]
        if len(sample_windows) == 0:
            continue

        matched_json_keys.add((record, major_mode_id))
        frame_delta_seconds = load_sequence_delta_seconds(json_path)
        for row in sample_windows.itertuples(index=False):
            rows.append(
                build_window_timing_row(
                    record=record,
                    major_mode_id=major_mode_id,
                    difficulty=difficulty,
                    frame_range=str(row.frame_range),
                    frame_start=int(row.frame_start),
                    frame_end=int(row.frame_end),
                    frame_delta_seconds=frame_delta_seconds,
                    target_fps=target_fps,
                )
            )

    if len(rows) == 0:
        raise RuntimeError(f"No matching JSON timing files found under: {dataset_json_dir}")

    timing = pd.DataFrame(rows)
    duplicate_count = int(timing.duplicated(subset=list(KEY_COLUMNS)).sum())
    if duplicate_count > 0:
        raise ValueError(f"Timing rows contain duplicate sample keys: duplicate_count={duplicate_count}")

    missing_keys = sorted(
        set(required_windows[["record", "major_mode_id"]].itertuples(index=False, name=None)) - matched_json_keys
    )
    if len(missing_keys) > 0:
        raise ValueError(f"Missing JSON timing files for record/mode keys: missing_keys={missing_keys}")
    return timing


def load_raw_sequence_stats(raw_sequence_root: Path) -> pd.DataFrame:
    require_existing_path(raw_sequence_root, "raw sequence root")
    rows: list[pd.DataFrame] = []
    for csv_path in sorted(raw_sequence_root.rglob("*_raw_sequence_frame_index.csv")):
        record, major_mode_id = parse_raw_sequence_name(csv_path)
        dataframe = pd.read_csv(csv_path)
        require_columns(
            dataframe=dataframe,
            columns=("img0", "img2", "valid", MOTION_COLUMN, ORACLE_COLUMN),
            label=f"raw sequence CSV: {csv_path}",
        )
        dataframe = dataframe.copy()
        dataframe["record"] = record
        dataframe["major_mode_id"] = major_mode_id
        dataframe["frame_range"] = dataframe.apply(
            lambda row: format_frame_range(int(row["img0"]), int(row["img2"])),
            axis=1,
        )
        selected_columns = [
            "record",
            "major_mode_id",
            "frame_range",
            "img0",
            "img1",
            "img2",
            "valid",
            "D_index Mean",
            "D_index Median",
            MOTION_COLUMN,
            "motion_magnitude_p95",
            "motion_magnitude_max",
            "oracle_fmv_t_eff_mean",
            "oracle_bmv_t_eff_mean",
            "oracle_t_eff_gap_mean",
        ]
        available_columns = [column for column in selected_columns if column in dataframe.columns]
        rows.append(dataframe[available_columns])

    if len(rows) == 0:
        raise RuntimeError(f"No raw sequence CSV files found under: {raw_sequence_root}")

    raw_stats = pd.concat(rows, ignore_index=True)
    duplicate_count = int(raw_stats.duplicated(subset=list(KEY_COLUMNS)).sum())
    if duplicate_count > 0:
        raise ValueError(f"Raw sequence stats contain duplicate sample keys: duplicate_count={duplicate_count}")
    return raw_stats.rename(
        columns={
            "valid": "raw_sequence_valid",
            "D_index Mean": "raw_distance_index_mean",
            "D_index Median": "raw_distance_index_median",
        }
    )


def build_joined_dataframe(metrics: pd.DataFrame, timing: pd.DataFrame, raw_stats: pd.DataFrame) -> pd.DataFrame:
    joined = metrics.merge(timing, on=list(KEY_COLUMNS), how="left", indicator="timing_merge")
    unmatched_timing = joined[joined["timing_merge"] != "both"]
    if len(unmatched_timing) > 0:
        preview = unmatched_timing[["record", "mode", "frame_range"]].head(10).to_dict("records")
        raise ValueError(f"Failed to match metrics rows to JSON timing rows: preview={preview}")
    joined = joined.drop(columns=["timing_merge"])

    joined = joined.merge(raw_stats, on=list(KEY_COLUMNS), how="left", indicator="raw_merge")
    unmatched_raw = joined[joined["raw_merge"] != "both"]
    if len(unmatched_raw) > 0:
        preview = unmatched_raw[["record", "mode", "frame_range"]].head(10).to_dict("records")
        raise ValueError(f"Failed to match metrics rows to raw sequence stats: preview={preview}")
    return joined.drop(columns=["raw_merge"])


def build_threshold_specs(fps_threshold_values: Sequence[float]) -> list[FpsThresholdSpec]:
    thresholds = sorted(set(float(value) for value in fps_threshold_values))
    invalid_thresholds = [value for value in thresholds if value <= 0.0]
    if len(invalid_thresholds) > 0:
        raise ValueError(f"FPS thresholds must be positive: invalid_thresholds={invalid_thresholds}")
    return [FpsThresholdSpec(label="all", tolerance_fps=None)] + [
        FpsThresholdSpec(label=f"+/-{threshold:g} fps", tolerance_fps=threshold)
        for threshold in thresholds
    ]


def filter_by_fps_threshold(dataframe: pd.DataFrame, threshold: FpsThresholdSpec) -> pd.DataFrame:
    if threshold.tolerance_fps is None:
        return dataframe.copy()
    return dataframe[dataframe["fps_abs_error_max"] <= threshold.tolerance_fps].copy()


def build_motion_bins(dataframe: pd.DataFrame, motion_bin_width: float) -> list[float]:
    max_motion = float(dataframe[MOTION_COLUMN].max())
    max_edge = math.ceil(max_motion / motion_bin_width) * motion_bin_width
    bin_count = int(max_edge / motion_bin_width)
    bin_edges = [round(index * motion_bin_width, 6) for index in range(bin_count + 1)]
    if len(bin_edges) < 2:
        bin_edges = [0.0, motion_bin_width]
    if bin_edges[-1] <= max_motion:
        bin_edges.append(round(bin_edges[-1] + motion_bin_width, 6))
    return bin_edges


def summarize_threshold(
    dataframe: pd.DataFrame,
    threshold: FpsThresholdSpec,
    all_sample_count: int,
) -> dict[str, object]:
    return {
        "threshold_label": threshold.label,
        "tolerance_fps": math.nan if threshold.tolerance_fps is None else float(threshold.tolerance_fps),
        "samples": int(len(dataframe)),
        "retention_ratio": float(len(dataframe) / all_sample_count),
        "mean_psnr": float(dataframe[PSNR_COLUMN].mean()),
        "median_psnr": float(dataframe[PSNR_COLUMN].median()),
        "mean_motion_magnitude": float(dataframe[MOTION_COLUMN].mean()),
        "mean_fps": float(dataframe["fps_mean"].mean()),
        "median_fps": float(dataframe["fps_mean"].median()),
        "mean_fps_abs_error_max": float(dataframe["fps_abs_error_max"].mean()),
        "median_fps_abs_error_max": float(dataframe["fps_abs_error_max"].median()),
        "mean_delta_abs_error_max_ms": float(dataframe["delta_abs_error_max_ms"].mean()),
    }


def build_motion_bin_summary(
    dataframe: pd.DataFrame,
    threshold: FpsThresholdSpec,
    motion_bin_edges: Sequence[float],
    min_bin_samples: int,
) -> pd.DataFrame:
    working = dataframe.copy()
    working["motion_bin"] = pd.cut(
        working[MOTION_COLUMN],
        bins=list(motion_bin_edges),
        include_lowest=True,
        right=False,
    )
    summary = (
        working.groupby("motion_bin", observed=True)
        .agg(
            samples=(PSNR_COLUMN, "size"),
            motion_mean=(MOTION_COLUMN, "mean"),
            psnr_mean=(PSNR_COLUMN, "mean"),
            psnr_median=(PSNR_COLUMN, "median"),
            fps_mean=("fps_mean", "mean"),
            fps_abs_error_max_mean=("fps_abs_error_max", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["samples"] >= min_bin_samples].copy()
    summary["threshold_label"] = threshold.label
    summary["tolerance_fps"] = math.nan if threshold.tolerance_fps is None else float(threshold.tolerance_fps)
    summary["motion_bin_label"] = summary["motion_bin"].astype(str)
    return summary.drop(columns=["motion_bin"])


def build_threshold_outputs(
    joined: pd.DataFrame,
    fps_threshold_values: Sequence[float],
    motion_bin_width: float,
    min_bin_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = build_threshold_specs(fps_threshold_values)
    motion_bin_edges = build_motion_bins(dataframe=joined, motion_bin_width=motion_bin_width)
    threshold_rows: list[dict[str, object]] = []
    motion_bin_frames: list[pd.DataFrame] = []
    all_sample_count = int(len(joined))

    for threshold in thresholds:
        filtered = filter_by_fps_threshold(dataframe=joined, threshold=threshold)
        if len(filtered) == 0:
            continue
        threshold_rows.append(
            summarize_threshold(
                dataframe=filtered,
                threshold=threshold,
                all_sample_count=all_sample_count,
            )
        )
        motion_bin_frames.append(
            build_motion_bin_summary(
                dataframe=filtered,
                threshold=threshold,
                motion_bin_edges=motion_bin_edges,
                min_bin_samples=min_bin_samples,
            )
        )

    return pd.DataFrame(threshold_rows), pd.concat(motion_bin_frames, ignore_index=True)


def build_symmetry_threshold_specs(symmetry_threshold_ms_values: Sequence[float]) -> list[SymmetryThresholdSpec]:
    thresholds = sorted(set(float(value) for value in symmetry_threshold_ms_values))
    invalid_thresholds = [value for value in thresholds if value <= 0.0]
    if len(invalid_thresholds) > 0:
        raise ValueError(f"Symmetry thresholds must be positive: invalid_thresholds={invalid_thresholds}")
    return [SymmetryThresholdSpec(label="all", tolerance_ms=None)] + [
        SymmetryThresholdSpec(label=f"<={threshold:g} ms", tolerance_ms=threshold)
        for threshold in thresholds
    ]


def filter_by_symmetry_threshold(dataframe: pd.DataFrame, threshold: SymmetryThresholdSpec) -> pd.DataFrame:
    if threshold.tolerance_ms is None:
        return dataframe.copy()
    return dataframe[dataframe[SYMMETRY_COLUMN] <= threshold.tolerance_ms].copy()


def summarize_symmetry_threshold(
    dataframe: pd.DataFrame,
    threshold: SymmetryThresholdSpec,
    all_sample_count: int,
) -> dict[str, object]:
    return {
        "threshold_label": threshold.label,
        "tolerance_ms": math.nan if threshold.tolerance_ms is None else float(threshold.tolerance_ms),
        "samples": int(len(dataframe)),
        "retention_ratio": float(len(dataframe) / all_sample_count),
        "mean_psnr": float(dataframe[PSNR_COLUMN].mean()),
        "median_psnr": float(dataframe[PSNR_COLUMN].median()),
        "mean_motion_magnitude": float(dataframe[MOTION_COLUMN].mean()),
        "mean_delta_time_symmetry_abs_ms": float(dataframe[SYMMETRY_COLUMN].mean()),
        "median_delta_time_symmetry_abs_ms": float(dataframe[SYMMETRY_COLUMN].median()),
        "mean_delta_time_position": float(dataframe["delta_time_position"].mean()),
        "mean_delta_time_position_error": float(dataframe["delta_time_position_error"].mean()),
        "mean_fps": float(dataframe["fps_mean"].mean()),
    }


def build_symmetry_motion_bin_summary(
    dataframe: pd.DataFrame,
    threshold: SymmetryThresholdSpec,
    motion_bin_edges: Sequence[float],
    min_bin_samples: int,
) -> pd.DataFrame:
    working = dataframe.copy()
    working["motion_bin"] = pd.cut(
        working[MOTION_COLUMN],
        bins=list(motion_bin_edges),
        include_lowest=True,
        right=False,
    )
    summary = (
        working.groupby("motion_bin", observed=True)
        .agg(
            samples=(PSNR_COLUMN, "size"),
            motion_mean=(MOTION_COLUMN, "mean"),
            psnr_mean=(PSNR_COLUMN, "mean"),
            psnr_median=(PSNR_COLUMN, "median"),
            delta_time_symmetry_abs_ms_mean=(SYMMETRY_COLUMN, "mean"),
            delta_time_position_error_mean=("delta_time_position_error", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["samples"] >= min_bin_samples].copy()
    summary["threshold_label"] = threshold.label
    summary["tolerance_ms"] = math.nan if threshold.tolerance_ms is None else float(threshold.tolerance_ms)
    summary["motion_bin_label"] = summary["motion_bin"].astype(str)
    return summary.drop(columns=["motion_bin"])


def build_symmetry_threshold_outputs(
    joined: pd.DataFrame,
    symmetry_threshold_ms_values: Sequence[float],
    motion_bin_width: float,
    min_bin_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = build_symmetry_threshold_specs(symmetry_threshold_ms_values)
    motion_bin_edges = build_motion_bins(dataframe=joined, motion_bin_width=motion_bin_width)
    threshold_rows: list[dict[str, object]] = []
    motion_bin_frames: list[pd.DataFrame] = []
    all_sample_count = int(len(joined))

    for threshold in thresholds:
        filtered = filter_by_symmetry_threshold(dataframe=joined, threshold=threshold)
        if len(filtered) == 0:
            continue
        threshold_rows.append(
            summarize_symmetry_threshold(
                dataframe=filtered,
                threshold=threshold,
                all_sample_count=all_sample_count,
            )
        )
        motion_bin_frames.append(
            build_symmetry_motion_bin_summary(
                dataframe=filtered,
                threshold=threshold,
                motion_bin_edges=motion_bin_edges,
                min_bin_samples=min_bin_samples,
            )
        )

    return pd.DataFrame(threshold_rows), pd.concat(motion_bin_frames, ignore_index=True)


def add_oracle_distance_columns(dataframe: pd.DataFrame, oracle_center: float) -> pd.DataFrame:
    result = dataframe.copy()
    result[ORACLE_DISTANCE_COLUMN] = (result[ORACLE_COLUMN] - oracle_center).abs()
    result["oracle_fmv_t_eff_side"] = np.where(result[ORACLE_COLUMN] < oracle_center, "early", "late")
    return result


def build_oracle_window_specs(oracle_center: float, oracle_window_radii: Sequence[float]) -> list[OracleWindowSpec]:
    radii = sorted(set(float(radius) for radius in oracle_window_radii))
    invalid_radii = [radius for radius in radii if radius <= 0.0]
    if len(invalid_radii) > 0:
        raise ValueError(f"Oracle window radii must be positive: invalid_radii={invalid_radii}")
    windows = [OracleWindowSpec(label="all", radius=None, lower=None, upper=None)]
    for radius in radii:
        lower = oracle_center - radius
        upper = oracle_center + radius
        windows.append(
            OracleWindowSpec(
                label=f"{lower:.3f}-{upper:.3f}",
                radius=radius,
                lower=lower,
                upper=upper,
            )
        )
    return windows


def filter_by_oracle_window(dataframe: pd.DataFrame, window: OracleWindowSpec) -> pd.DataFrame:
    if window.radius is None:
        return dataframe.copy()
    return dataframe[dataframe[ORACLE_COLUMN].between(window.lower, window.upper, inclusive="both")].copy()


def summarize_oracle_window(
    dataframe: pd.DataFrame,
    window: OracleWindowSpec,
    all_sample_count: int,
) -> dict[str, object]:
    return {
        "window_label": window.label,
        "radius": math.nan if window.radius is None else float(window.radius),
        "lower": math.nan if window.lower is None else float(window.lower),
        "upper": math.nan if window.upper is None else float(window.upper),
        "samples": int(len(dataframe)),
        "retention_ratio": float(len(dataframe) / all_sample_count),
        "mean_psnr": float(dataframe[PSNR_COLUMN].mean()),
        "median_psnr": float(dataframe[PSNR_COLUMN].median()),
        "mean_motion_magnitude": float(dataframe[MOTION_COLUMN].mean()),
        "mean_oracle_fmv_t_eff": float(dataframe[ORACLE_COLUMN].mean()),
        "median_oracle_fmv_t_eff": float(dataframe[ORACLE_COLUMN].median()),
        "mean_oracle_distance": float(dataframe[ORACLE_DISTANCE_COLUMN].mean()),
        "median_oracle_distance": float(dataframe[ORACLE_DISTANCE_COLUMN].median()),
    }


def build_oracle_motion_bin_summary(
    dataframe: pd.DataFrame,
    window: OracleWindowSpec,
    motion_bin_edges: Sequence[float],
    min_bin_samples: int,
) -> pd.DataFrame:
    working = dataframe.copy()
    working["motion_bin"] = pd.cut(
        working[MOTION_COLUMN],
        bins=list(motion_bin_edges),
        include_lowest=True,
        right=False,
    )
    summary = (
        working.groupby("motion_bin", observed=True)
        .agg(
            samples=(PSNR_COLUMN, "size"),
            motion_mean=(MOTION_COLUMN, "mean"),
            psnr_mean=(PSNR_COLUMN, "mean"),
            psnr_median=(PSNR_COLUMN, "median"),
            oracle_fmv_t_eff_mean=(ORACLE_COLUMN, "mean"),
            oracle_distance_mean=(ORACLE_DISTANCE_COLUMN, "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["samples"] >= min_bin_samples].copy()
    summary["window_label"] = window.label
    summary["radius"] = math.nan if window.radius is None else float(window.radius)
    summary["motion_bin_label"] = summary["motion_bin"].astype(str)
    return summary.drop(columns=["motion_bin"])


def build_oracle_window_outputs(
    joined: pd.DataFrame,
    oracle_center: float,
    oracle_window_radii: Sequence[float],
    motion_bin_width: float,
    min_bin_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = build_oracle_window_specs(oracle_center=oracle_center, oracle_window_radii=oracle_window_radii)
    motion_bin_edges = build_motion_bins(dataframe=joined, motion_bin_width=motion_bin_width)
    window_rows: list[dict[str, object]] = []
    motion_bin_frames: list[pd.DataFrame] = []
    all_sample_count = int(len(joined))

    for window in windows:
        filtered = filter_by_oracle_window(dataframe=joined, window=window)
        if len(filtered) == 0:
            continue
        window_rows.append(
            summarize_oracle_window(
                dataframe=filtered,
                window=window,
                all_sample_count=all_sample_count,
            )
        )
        motion_bin_frames.append(
            build_oracle_motion_bin_summary(
                dataframe=filtered,
                window=window,
                motion_bin_edges=motion_bin_edges,
                min_bin_samples=min_bin_samples,
            )
        )

    return pd.DataFrame(window_rows), pd.concat(motion_bin_frames, ignore_index=True)


def build_threshold_colors(threshold_order: Sequence[str]) -> dict[str, object]:
    color_values = plt.cm.viridis_r([index / max(len(threshold_order) - 1, 1) for index in range(len(threshold_order))])
    colors = {label: color_values[index] for index, label in enumerate(threshold_order)}
    colors["all"] = "#4c78a8"
    return colors


def write_threshold_dashboard(
    threshold_summary: pd.DataFrame,
    motion_bin_summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "fps_stability_threshold_dashboard.png"
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=160)
    threshold_order = threshold_summary["threshold_label"].tolist()
    colors = build_threshold_colors(threshold_order)

    axis = axes[0, 0]
    for threshold_label in threshold_order:
        line = motion_bin_summary[motion_bin_summary["threshold_label"] == threshold_label]
        if len(line) == 0:
            continue
        sample_count = int(threshold_summary.loc[threshold_summary["threshold_label"] == threshold_label, "samples"].iloc[0])
        axis.plot(
            line["motion_mean"],
            line["psnr_mean"],
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label=f"{threshold_label} (n={sample_count})",
            color=colors.get(threshold_label),
        )
    axis.set_title("Mean PSNR by motion bin")
    axis.set_xlabel(MOTION_COLUMN)
    axis.set_ylabel("mean psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    for threshold_label in threshold_order:
        line = motion_bin_summary[motion_bin_summary["threshold_label"] == threshold_label]
        if len(line) == 0:
            continue
        axis.plot(
            line["motion_mean"],
            line["psnr_median"],
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label=threshold_label,
            color=colors.get(threshold_label),
        )
    axis.set_title("Median PSNR by motion bin")
    axis.set_xlabel(MOTION_COLUMN)
    axis.set_ylabel("median psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    axis.bar(
        threshold_summary["threshold_label"],
        threshold_summary["mean_psnr"],
        color=[colors.get(label, "#777777") for label in threshold_summary["threshold_label"]],
    )
    axis.set_title("Overall mean PSNR after stable-FPS filtering")
    axis.set_xlabel("max FPS deviation threshold")
    axis.set_ylabel("mean psnr (dB)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(True, axis="y", alpha=0.25)

    axis = axes[1, 1]
    axis.bar(
        threshold_summary["threshold_label"],
        threshold_summary["retention_ratio"] * 100.0,
        color=[colors.get(label, "#777777") for label in threshold_summary["threshold_label"]],
    )
    axis.set_title("Stable sample retention after filtering")
    axis.set_xlabel("max FPS deviation threshold")
    axis.set_ylabel("retained samples (%)")
    axis.set_ylim(0.0, 105.0)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(True, axis="y", alpha=0.25)

    figure.suptitle("IFRNet FineTuning: FPS-stability threshold sweep", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_symmetry_threshold_dashboard(
    threshold_summary: pd.DataFrame,
    motion_bin_summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "delta_time_symmetry_threshold_dashboard.png"
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=160)
    threshold_order = threshold_summary["threshold_label"].tolist()
    colors = build_threshold_colors(threshold_order)

    axis = axes[0, 0]
    for threshold_label in threshold_order:
        line = motion_bin_summary[motion_bin_summary["threshold_label"] == threshold_label]
        if len(line) == 0:
            continue
        sample_count = int(threshold_summary.loc[threshold_summary["threshold_label"] == threshold_label, "samples"].iloc[0])
        axis.plot(
            line["motion_mean"],
            line["psnr_mean"],
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label=f"{threshold_label} (n={sample_count})",
            color=colors.get(threshold_label),
        )
    axis.set_title("Mean PSNR by motion bin")
    axis.set_xlabel(MOTION_COLUMN)
    axis.set_ylabel("mean psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    for threshold_label in threshold_order:
        line = motion_bin_summary[motion_bin_summary["threshold_label"] == threshold_label]
        if len(line) == 0:
            continue
        axis.plot(
            line["motion_mean"],
            line["psnr_median"],
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label=threshold_label,
            color=colors.get(threshold_label),
        )
    axis.set_title("Median PSNR by motion bin")
    axis.set_xlabel(MOTION_COLUMN)
    axis.set_ylabel("median psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    axis.bar(
        threshold_summary["threshold_label"],
        threshold_summary["mean_psnr"],
        color=[colors.get(label, "#777777") for label in threshold_summary["threshold_label"]],
    )
    axis.set_title("Overall mean PSNR after symmetry filtering")
    axis.set_xlabel("max delta-time difference threshold")
    axis.set_ylabel("mean psnr (dB)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(True, axis="y", alpha=0.25)

    axis = axes[1, 1]
    axis.bar(
        threshold_summary["threshold_label"],
        threshold_summary["retention_ratio"] * 100.0,
        color=[colors.get(label, "#777777") for label in threshold_summary["threshold_label"]],
    )
    axis.set_title("Symmetric sample retention after filtering")
    axis.set_xlabel("max delta-time difference threshold")
    axis.set_ylabel("retained samples (%)")
    axis.set_ylim(0.0, 105.0)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(True, axis="y", alpha=0.25)

    figure.suptitle("IFRNet FineTuning: delta-time symmetry threshold sweep", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def format_symmetry_threshold_label(threshold_ms: float) -> str:
    return f"<={threshold_ms:g} ms"


def write_symmetry_gain_dashboard(
    threshold_summary: pd.DataFrame,
    highlight_threshold_ms: float,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "delta_time_symmetry_gain_dashboard.png"
    highlight_label = format_symmetry_threshold_label(highlight_threshold_ms)
    if "all" not in set(threshold_summary["threshold_label"]):
        raise ValueError("Symmetry threshold summary is missing the all row.")
    if highlight_label not in set(threshold_summary["threshold_label"]):
        available_labels = threshold_summary["threshold_label"].tolist()
        raise ValueError(f"Highlight threshold label not found: label={highlight_label} available_labels={available_labels}")

    baseline = threshold_summary[threshold_summary["threshold_label"] == "all"].iloc[0]
    highlight = threshold_summary[threshold_summary["threshold_label"] == highlight_label].iloc[0]
    filtered_summary = threshold_summary[threshold_summary["threshold_label"] != "all"].copy()
    filtered_summary["mean_psnr_gain"] = filtered_summary["mean_psnr"] - float(baseline["mean_psnr"])
    filtered_summary["retention_percent"] = filtered_summary["retention_ratio"] * 100.0

    kept_samples = int(highlight["samples"])
    total_samples = int(baseline["samples"])
    removed_samples = total_samples - kept_samples
    mean_psnr_gain = float(highlight["mean_psnr"] - baseline["mean_psnr"])
    median_psnr_gain = float(highlight["median_psnr"] - baseline["median_psnr"])

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160)
    figure.suptitle("Delta-time symmetry clean-filter effect", fontsize=16)

    axis = axes[0, 0]
    labels = ["all samples", highlight_label]
    mean_values = [float(baseline["mean_psnr"]), float(highlight["mean_psnr"])]
    median_values = [float(baseline["median_psnr"]), float(highlight["median_psnr"])]
    x_positions = np.arange(len(labels))
    width = 0.36
    axis.bar(x_positions - width / 2.0, mean_values, width=width, color="#4c78a8", label="mean PSNR")
    axis.bar(x_positions + width / 2.0, median_values, width=width, color="#59a14f", label="median PSNR")
    y_min = min(mean_values + median_values) - 0.25
    y_max = max(mean_values + median_values) + 0.25
    axis.set_ylim(y_min, y_max)
    axis.set_xticks(x_positions, labels)
    axis.set_ylabel("psnr (dB)")
    axis.set_title("Zoomed PSNR comparison")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    for x_position, mean_value, median_value in zip(x_positions, mean_values, median_values):
        axis.text(x_position - width / 2.0, mean_value + 0.03, f"{mean_value:.2f}", ha="center", va="bottom", fontsize=9)
        axis.text(x_position + width / 2.0, median_value + 0.03, f"{median_value:.2f}", ha="center", va="bottom", fontsize=9)
    axis.annotate(
        f"mean +{mean_psnr_gain:.2f} dB\nmedian +{median_psnr_gain:.2f} dB",
        xy=(1.0, float(highlight["mean_psnr"])),
        xytext=(0.62, y_max - 0.05),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        ha="left",
        va="top",
        fontsize=10,
    )

    axis = axes[0, 1]
    axis.bar(["kept", "removed"], [kept_samples, removed_samples], color=["#2ca02c", "#d62728"], alpha=0.85)
    axis.set_title("Sample retention at highlighted threshold")
    axis.set_ylabel("samples")
    axis.set_ylim(0.0, total_samples * 1.12)
    axis.grid(True, axis="y", alpha=0.25)
    for index, value in enumerate([kept_samples, removed_samples]):
        percent = value / total_samples * 100.0
        axis.text(
            index,
            value - total_samples * 0.035,
            f"{value}\n{percent:.1f}%",
            ha="center",
            va="top",
            fontsize=11,
            color="white",
            fontweight="bold",
        )

    axis = axes[1, 0]
    colors = ["#2ca02c" if label == highlight_label else "#4c78a8" for label in filtered_summary["threshold_label"]]
    axis.bar(filtered_summary["threshold_label"], filtered_summary["mean_psnr_gain"], color=colors, alpha=0.9)
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set_title("Mean PSNR gain over all samples")
    axis.set_xlabel("delta-time symmetry threshold")
    axis.set_ylabel("mean PSNR gain (dB)")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(True, axis="y", alpha=0.25)
    for index, row in enumerate(filtered_summary.itertuples(index=False)):
        value = float(row.mean_psnr_gain)
        axis.text(index, value + 0.015, f"{value:+.2f}", ha="center", va="bottom", fontsize=8)

    axis = axes[1, 1]
    colors = ["#2ca02c" if label == highlight_label else "#4c78a8" for label in filtered_summary["threshold_label"]]
    axis.scatter(
        filtered_summary["retention_percent"],
        filtered_summary["mean_psnr"],
        s=70,
        color=colors,
        alpha=0.9,
    )
    y_axis_min = float(min(filtered_summary["mean_psnr"].min(), baseline["mean_psnr"])) - 0.05
    y_axis_max = float(filtered_summary["mean_psnr"].max()) + 0.12
    axis.set_ylim(y_axis_min, y_axis_max)
    for index, row in enumerate(filtered_summary.itertuples(index=False)):
        y_offset = -14 if float(row.mean_psnr) > y_axis_max - 0.16 else 5 + (index % 3) * 7
        axis.annotate(
            str(row.threshold_label),
            (float(row.retention_percent), float(row.mean_psnr)),
            textcoords="offset points",
            xytext=(4, y_offset),
            fontsize=8,
        )
    axis.axhline(float(baseline["mean_psnr"]), color="#777777", linestyle="--", linewidth=1.1, label=f"all mean = {float(baseline['mean_psnr']):.2f}")
    axis.set_title("Retention vs mean PSNR tradeoff")
    axis.set_xlabel("retained samples (%)")
    axis.set_ylabel("mean psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_oracle_window_dashboard(
    window_summary: pd.DataFrame,
    motion_bin_summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "oracle_fmv_t_eff_window_dashboard.png"
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=160)
    window_order = window_summary["window_label"].tolist()
    colors = build_threshold_colors(window_order)

    axis = axes[0, 0]
    for window_label in window_order:
        line = motion_bin_summary[motion_bin_summary["window_label"] == window_label]
        if len(line) == 0:
            continue
        sample_count = int(window_summary.loc[window_summary["window_label"] == window_label, "samples"].iloc[0])
        axis.plot(
            line["motion_mean"],
            line["psnr_mean"],
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label=f"{window_label} (n={sample_count})",
            color=colors.get(window_label),
        )
    axis.set_title("Mean PSNR by motion bin")
    axis.set_xlabel(MOTION_COLUMN)
    axis.set_ylabel("mean psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    for window_label in window_order:
        line = motion_bin_summary[motion_bin_summary["window_label"] == window_label]
        if len(line) == 0:
            continue
        axis.plot(
            line["motion_mean"],
            line["psnr_median"],
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label=window_label,
            color=colors.get(window_label),
        )
    axis.set_title("Median PSNR by motion bin")
    axis.set_xlabel(MOTION_COLUMN)
    axis.set_ylabel("median psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    axis.bar(
        window_summary["window_label"],
        window_summary["mean_psnr"],
        color=[colors.get(label, "#777777") for label in window_summary["window_label"]],
    )
    axis.set_title("Overall mean PSNR after oracle-window filtering")
    axis.set_xlabel("oracle_fmv_t_eff_mean window")
    axis.set_ylabel("mean psnr (dB)")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(True, axis="y", alpha=0.25)

    axis = axes[1, 1]
    axis.bar(
        window_summary["window_label"],
        window_summary["retention_ratio"] * 100.0,
        color=[colors.get(label, "#777777") for label in window_summary["window_label"]],
    )
    axis.set_title("Clean sample retention after filtering")
    axis.set_xlabel("oracle_fmv_t_eff_mean window")
    axis.set_ylabel("retained samples (%)")
    axis.set_ylim(0.0, 105.0)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(True, axis="y", alpha=0.25)

    figure.suptitle("IFRNet FineTuning: oracle_fmv_t_eff clean-window sweep", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def format_oracle_window_label(oracle_center: float, window_radius: float) -> str:
    return f"{oracle_center - window_radius:.3f}-{oracle_center + window_radius:.3f}"


def write_oracle_gain_dashboard(
    window_summary: pd.DataFrame,
    oracle_center: float,
    highlight_window_radius: float,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "oracle_fmv_t_eff_gain_dashboard.png"
    highlight_label = format_oracle_window_label(
        oracle_center=oracle_center,
        window_radius=highlight_window_radius,
    )
    if "all" not in set(window_summary["window_label"]):
        raise ValueError("Oracle window summary is missing the all row.")
    if highlight_label not in set(window_summary["window_label"]):
        available_labels = window_summary["window_label"].tolist()
        raise ValueError(f"Highlight oracle window not found: label={highlight_label} available_labels={available_labels}")

    baseline = window_summary[window_summary["window_label"] == "all"].iloc[0]
    highlight = window_summary[window_summary["window_label"] == highlight_label].iloc[0]
    filtered_summary = window_summary[window_summary["window_label"] != "all"].copy()
    filtered_summary["mean_psnr_gain"] = filtered_summary["mean_psnr"] - float(baseline["mean_psnr"])
    filtered_summary["retention_percent"] = filtered_summary["retention_ratio"] * 100.0

    kept_samples = int(highlight["samples"])
    total_samples = int(baseline["samples"])
    removed_samples = total_samples - kept_samples
    mean_psnr_gain = float(highlight["mean_psnr"] - baseline["mean_psnr"])
    median_psnr_gain = float(highlight["median_psnr"] - baseline["median_psnr"])

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160)
    figure.suptitle("Oracle effective-time clean-filter effect", fontsize=16)

    axis = axes[0, 0]
    labels = ["all samples", highlight_label]
    mean_values = [float(baseline["mean_psnr"]), float(highlight["mean_psnr"])]
    median_values = [float(baseline["median_psnr"]), float(highlight["median_psnr"])]
    x_positions = np.arange(len(labels))
    width = 0.36
    axis.bar(x_positions - width / 2.0, mean_values, width=width, color="#4c78a8", label="mean PSNR")
    axis.bar(x_positions + width / 2.0, median_values, width=width, color="#59a14f", label="median PSNR")
    y_min = min(mean_values + median_values) - 0.25
    y_max = max(mean_values + median_values) + 0.25
    axis.set_ylim(y_min, y_max)
    axis.set_xticks(x_positions, labels)
    axis.set_ylabel("psnr (dB)")
    axis.set_title("Zoomed PSNR comparison")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    for x_position, mean_value, median_value in zip(x_positions, mean_values, median_values):
        axis.text(x_position - width / 2.0, mean_value + 0.03, f"{mean_value:.2f}", ha="center", va="bottom", fontsize=9)
        axis.text(x_position + width / 2.0, median_value + 0.03, f"{median_value:.2f}", ha="center", va="bottom", fontsize=9)
    axis.annotate(
        f"mean +{mean_psnr_gain:.2f} dB\nmedian +{median_psnr_gain:.2f} dB",
        xy=(1.0, float(highlight["mean_psnr"])),
        xytext=(0.62, y_max - 0.05),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        ha="left",
        va="top",
        fontsize=10,
    )

    axis = axes[0, 1]
    axis.bar(["kept", "removed"], [kept_samples, removed_samples], color=["#2ca02c", "#d62728"], alpha=0.85)
    axis.set_title("Sample retention at highlighted window")
    axis.set_ylabel("samples")
    axis.set_ylim(0.0, total_samples * 1.12)
    axis.grid(True, axis="y", alpha=0.25)
    for index, value in enumerate([kept_samples, removed_samples]):
        percent = value / total_samples * 100.0
        axis.text(
            index,
            value - total_samples * 0.035,
            f"{value}\n{percent:.1f}%",
            ha="center",
            va="top",
            fontsize=11,
            color="white",
            fontweight="bold",
        )

    axis = axes[1, 0]
    colors = ["#2ca02c" if label == highlight_label else "#4c78a8" for label in filtered_summary["window_label"]]
    axis.bar(filtered_summary["window_label"], filtered_summary["mean_psnr_gain"], color=colors, alpha=0.9)
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set_title("Mean PSNR gain over all samples")
    axis.set_xlabel("oracle_fmv_t_eff_mean window")
    axis.set_ylabel("mean PSNR gain (dB)")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(True, axis="y", alpha=0.25)
    for index, row in enumerate(filtered_summary.itertuples(index=False)):
        value = float(row.mean_psnr_gain)
        axis.text(index, value + 0.015, f"{value:+.2f}", ha="center", va="bottom", fontsize=8)

    axis = axes[1, 1]
    colors = ["#2ca02c" if label == highlight_label else "#4c78a8" for label in filtered_summary["window_label"]]
    axis.scatter(
        filtered_summary["retention_percent"],
        filtered_summary["mean_psnr"],
        s=70,
        color=colors,
        alpha=0.9,
    )
    y_axis_min = float(min(filtered_summary["mean_psnr"].min(), baseline["mean_psnr"])) - 0.05
    y_axis_max = float(filtered_summary["mean_psnr"].max()) + 0.12
    axis.set_ylim(y_axis_min, y_axis_max)
    labeled_windows = {
        highlight_label,
        format_oracle_window_label(oracle_center=oracle_center, window_radius=0.05),
        format_oracle_window_label(oracle_center=oracle_center, window_radius=0.15),
        format_oracle_window_label(oracle_center=oracle_center, window_radius=0.175),
        format_oracle_window_label(oracle_center=oracle_center, window_radius=0.25),
    }
    label_rows = filtered_summary[filtered_summary["window_label"].isin(labeled_windows)].reset_index(drop=True)
    for index, row in enumerate(label_rows.itertuples(index=False)):
        y_offset = -14 if float(row.mean_psnr) > y_axis_max - 0.16 else 5 + (index % 3) * 7
        axis.annotate(
            str(row.window_label),
            (float(row.retention_percent), float(row.mean_psnr)),
            textcoords="offset points",
            xytext=(4, y_offset),
            fontsize=8,
        )
    axis.axhline(float(baseline["mean_psnr"]), color="#777777", linestyle="--", linewidth=1.1, label=f"all mean = {float(baseline['mean_psnr']):.2f}")
    axis.set_title("Retention vs mean PSNR tradeoff")
    axis.set_xlabel("retained samples (%)")
    axis.set_ylabel("mean psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_distribution_labels(dataframe: pd.DataFrame, scatter_threshold_fps: float) -> pd.Series:
    stable_mask = dataframe["fps_abs_error_max"] <= scatter_threshold_fps
    psnr_threshold = float(dataframe[PSNR_COLUMN].median())
    high_psnr_mask = dataframe[PSNR_COLUMN] >= psnr_threshold
    labels = pd.Series(index=dataframe.index, dtype="string")
    labels[stable_mask & high_psnr_mask] = "stable_fps_high_psnr"
    labels[stable_mask & ~high_psnr_mask] = "stable_fps_low_psnr"
    labels[~stable_mask & high_psnr_mask] = "unstable_fps_high_psnr"
    labels[~stable_mask & ~high_psnr_mask] = "unstable_fps_low_psnr"
    return labels


def write_fps_mean_distribution_plot(
    joined: pd.DataFrame,
    scatter_threshold_fps: float,
    target_fps: float,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "fps_mean_psnr_distribution.png"
    dataframe = joined.copy()
    dataframe["fps_psnr_label"] = build_distribution_labels(
        dataframe=dataframe,
        scatter_threshold_fps=scatter_threshold_fps,
    )
    psnr_threshold = float(dataframe[PSNR_COLUMN].median())
    label_order = [
        "stable_fps_high_psnr",
        "stable_fps_low_psnr",
        "unstable_fps_high_psnr",
        "unstable_fps_low_psnr",
    ]
    label_display_names = {
        "stable_fps_high_psnr": "stable fps, high psnr",
        "stable_fps_low_psnr": "stable fps, low psnr",
        "unstable_fps_high_psnr": "unstable fps, high psnr",
        "unstable_fps_low_psnr": "unstable fps, low psnr",
    }
    colors = {
        "stable_fps_high_psnr": "#2ca02c",
        "stable_fps_low_psnr": "#ff7f0e",
        "unstable_fps_high_psnr": "#1f77b4",
        "unstable_fps_low_psnr": "#d62728",
    }

    figure, axis = plt.subplots(figsize=(10.5, 6.4), dpi=160)
    for label in label_order:
        label_dataframe = dataframe[dataframe["fps_psnr_label"] == label]
        if len(label_dataframe) == 0:
            continue
        axis.scatter(
            label_dataframe["fps_mean"],
            label_dataframe[PSNR_COLUMN],
            s=16,
            alpha=0.42,
            edgecolors="none",
            color=colors[label],
            label=f"{label_display_names[label]} (n={len(label_dataframe)})",
        )

    axis.axvspan(
        target_fps - scatter_threshold_fps,
        target_fps + scatter_threshold_fps,
        color="#2ca02c",
        alpha=0.08,
        label=f"target FPS reference band +/-{scatter_threshold_fps:g}",
    )
    axis.axvline(target_fps, color="black", linestyle="--", linewidth=1.1, label=f"target fps = {target_fps:g}")
    axis.axhline(psnr_threshold, color="#777777", linestyle="--", linewidth=1.1, label=f"median psnr = {psnr_threshold:.2f}")
    axis.set_title("Per-sample mean FPS vs PSNR")
    axis.set_xlabel("window mean FPS")
    axis.set_ylabel("psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_fps_error_distribution_plot(
    joined: pd.DataFrame,
    scatter_threshold_fps: float,
    target_fps: float,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "fps_error_psnr_distribution.png"
    dataframe = joined.copy()
    dataframe["fps_psnr_label"] = build_distribution_labels(
        dataframe=dataframe,
        scatter_threshold_fps=scatter_threshold_fps,
    )
    psnr_threshold = float(dataframe[PSNR_COLUMN].median())
    label_order = [
        "stable_fps_high_psnr",
        "stable_fps_low_psnr",
        "unstable_fps_high_psnr",
        "unstable_fps_low_psnr",
    ]
    label_display_names = {
        "stable_fps_high_psnr": "stable fps, high psnr",
        "stable_fps_low_psnr": "stable fps, low psnr",
        "unstable_fps_high_psnr": "unstable fps, high psnr",
        "unstable_fps_low_psnr": "unstable fps, low psnr",
    }
    colors = {
        "stable_fps_high_psnr": "#2ca02c",
        "stable_fps_low_psnr": "#ff7f0e",
        "unstable_fps_high_psnr": "#1f77b4",
        "unstable_fps_low_psnr": "#d62728",
    }

    figure, axis = plt.subplots(figsize=(10.5, 6.4), dpi=160)
    for label in label_order:
        label_dataframe = dataframe[dataframe["fps_psnr_label"] == label]
        if len(label_dataframe) == 0:
            continue
        axis.scatter(
            label_dataframe["fps_abs_error_max"],
            label_dataframe[PSNR_COLUMN],
            s=16,
            alpha=0.42,
            edgecolors="none",
            color=colors[label],
            label=f"{label_display_names[label]} (n={len(label_dataframe)})",
        )

    axis.axvline(
        scatter_threshold_fps,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=f"stable threshold = +/-{scatter_threshold_fps:g} fps",
    )
    axis.axhline(psnr_threshold, color="#777777", linestyle="--", linewidth=1.1, label=f"median psnr = {psnr_threshold:.2f}")
    axis.set_title("Per-sample FPS stability error vs PSNR")
    axis.set_xlabel(f"max absolute FPS deviation from {target_fps:g}")
    axis.set_ylabel("psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_symmetry_distribution_labels(dataframe: pd.DataFrame, scatter_symmetry_threshold_ms: float) -> pd.Series:
    symmetric_mask = dataframe[SYMMETRY_COLUMN] <= scatter_symmetry_threshold_ms
    psnr_threshold = float(dataframe[PSNR_COLUMN].median())
    high_psnr_mask = dataframe[PSNR_COLUMN] >= psnr_threshold
    labels = pd.Series(index=dataframe.index, dtype="string")
    labels[symmetric_mask & high_psnr_mask] = "symmetric_high_psnr"
    labels[symmetric_mask & ~high_psnr_mask] = "symmetric_low_psnr"
    labels[~symmetric_mask & high_psnr_mask] = "non_symmetric_high_psnr"
    labels[~symmetric_mask & ~high_psnr_mask] = "non_symmetric_low_psnr"
    return labels


def build_symmetry_label_metadata() -> tuple[list[str], dict[str, str], dict[str, str]]:
    label_order = [
        "symmetric_high_psnr",
        "symmetric_low_psnr",
        "non_symmetric_high_psnr",
        "non_symmetric_low_psnr",
    ]
    label_display_names = {
        "symmetric_high_psnr": "symmetric, high psnr",
        "symmetric_low_psnr": "symmetric, low psnr",
        "non_symmetric_high_psnr": "non-symmetric, high psnr",
        "non_symmetric_low_psnr": "non-symmetric, low psnr",
    }
    colors = {
        "symmetric_high_psnr": "#2ca02c",
        "symmetric_low_psnr": "#ff7f0e",
        "non_symmetric_high_psnr": "#1f77b4",
        "non_symmetric_low_psnr": "#d62728",
    }
    return label_order, label_display_names, colors


def write_symmetry_error_distribution_plot(
    joined: pd.DataFrame,
    scatter_symmetry_threshold_ms: float,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "delta_time_symmetry_psnr_distribution.png"
    dataframe = joined.copy()
    dataframe["symmetry_psnr_label"] = build_symmetry_distribution_labels(
        dataframe=dataframe,
        scatter_symmetry_threshold_ms=scatter_symmetry_threshold_ms,
    )
    psnr_threshold = float(dataframe[PSNR_COLUMN].median())
    label_order, label_display_names, colors = build_symmetry_label_metadata()

    figure, axis = plt.subplots(figsize=(10.5, 6.4), dpi=160)
    for label in label_order:
        label_dataframe = dataframe[dataframe["symmetry_psnr_label"] == label]
        if len(label_dataframe) == 0:
            continue
        axis.scatter(
            label_dataframe[SYMMETRY_COLUMN],
            label_dataframe[PSNR_COLUMN],
            s=16,
            alpha=0.42,
            edgecolors="none",
            color=colors[label],
            label=f"{label_display_names[label]} (n={len(label_dataframe)})",
        )

    axis.axvline(
        scatter_symmetry_threshold_ms,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=f"symmetric threshold <= {scatter_symmetry_threshold_ms:g} ms",
    )
    axis.axhline(psnr_threshold, color="#777777", linestyle="--", linewidth=1.1, label=f"median psnr = {psnr_threshold:.2f}")
    axis.set_title("Per-sample delta-time symmetry vs PSNR")
    axis.set_xlabel("|delta_second_1_to_0 - delta_second_2_to_1| (ms)")
    axis.set_ylabel("psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_delta_time_position_distribution_plot(
    joined: pd.DataFrame,
    scatter_symmetry_threshold_ms: float,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "delta_time_position_psnr_distribution.png"
    dataframe = joined.copy()
    dataframe["symmetry_psnr_label"] = build_symmetry_distribution_labels(
        dataframe=dataframe,
        scatter_symmetry_threshold_ms=scatter_symmetry_threshold_ms,
    )
    psnr_threshold = float(dataframe[PSNR_COLUMN].median())
    label_order, label_display_names, colors = build_symmetry_label_metadata()

    figure, axis = plt.subplots(figsize=(10.5, 6.4), dpi=160)
    for label in label_order:
        label_dataframe = dataframe[dataframe["symmetry_psnr_label"] == label]
        if len(label_dataframe) == 0:
            continue
        axis.scatter(
            label_dataframe["delta_time_position"],
            label_dataframe[PSNR_COLUMN],
            s=16,
            alpha=0.42,
            edgecolors="none",
            color=colors[label],
            label=f"{label_display_names[label]} (n={len(label_dataframe)})",
        )

    axis.axvline(0.5, color="black", linestyle="--", linewidth=1.1, label="symmetric position = 0.5")
    axis.axhline(psnr_threshold, color="#777777", linestyle="--", linewidth=1.1, label=f"median psnr = {psnr_threshold:.2f}")
    axis.set_title("Per-sample middle-frame time position vs PSNR")
    axis.set_xlabel("delta_second_1_to_0 / (delta_second_1_to_0 + delta_second_2_to_1)")
    axis.set_ylabel("psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_oracle_distribution_labels(dataframe: pd.DataFrame, scatter_oracle_window_radius: float) -> pd.Series:
    clean_mask = dataframe[ORACLE_DISTANCE_COLUMN] <= scatter_oracle_window_radius
    psnr_threshold = float(dataframe[PSNR_COLUMN].median())
    high_psnr_mask = dataframe[PSNR_COLUMN] >= psnr_threshold
    labels = pd.Series(index=dataframe.index, dtype="string")
    labels[clean_mask & high_psnr_mask] = "oracle_clean_high_psnr"
    labels[clean_mask & ~high_psnr_mask] = "oracle_clean_low_psnr"
    labels[~clean_mask & high_psnr_mask] = "oracle_outside_high_psnr"
    labels[~clean_mask & ~high_psnr_mask] = "oracle_outside_low_psnr"
    return labels


def build_oracle_label_metadata() -> tuple[list[str], dict[str, str], dict[str, str]]:
    label_order = [
        "oracle_clean_high_psnr",
        "oracle_clean_low_psnr",
        "oracle_outside_high_psnr",
        "oracle_outside_low_psnr",
    ]
    label_display_names = {
        "oracle_clean_high_psnr": "clean oracle, high psnr",
        "oracle_clean_low_psnr": "clean oracle, low psnr",
        "oracle_outside_high_psnr": "outside oracle, high psnr",
        "oracle_outside_low_psnr": "outside oracle, low psnr",
    }
    colors = {
        "oracle_clean_high_psnr": "#2ca02c",
        "oracle_clean_low_psnr": "#ff7f0e",
        "oracle_outside_high_psnr": "#1f77b4",
        "oracle_outside_low_psnr": "#d62728",
    }
    return label_order, label_display_names, colors


def write_oracle_fmv_t_eff_distribution_plot(
    joined: pd.DataFrame,
    oracle_center: float,
    scatter_oracle_window_radius: float,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "oracle_fmv_t_eff_psnr_distribution.png"
    dataframe = joined.copy()
    dataframe["oracle_psnr_label"] = build_oracle_distribution_labels(
        dataframe=dataframe,
        scatter_oracle_window_radius=scatter_oracle_window_radius,
    )
    psnr_threshold = float(dataframe[PSNR_COLUMN].median())
    label_order, label_display_names, colors = build_oracle_label_metadata()
    lower = oracle_center - scatter_oracle_window_radius
    upper = oracle_center + scatter_oracle_window_radius

    figure, axis = plt.subplots(figsize=(10.5, 6.4), dpi=160)
    for label in label_order:
        label_dataframe = dataframe[dataframe["oracle_psnr_label"] == label]
        if len(label_dataframe) == 0:
            continue
        axis.scatter(
            label_dataframe[ORACLE_COLUMN],
            label_dataframe[PSNR_COLUMN],
            s=16,
            alpha=0.42,
            edgecolors="none",
            color=colors[label],
            label=f"{label_display_names[label]} (n={len(label_dataframe)})",
        )

    axis.axvspan(lower, upper, color="#2ca02c", alpha=0.08, label=f"clean oracle window {lower:.3f}-{upper:.3f}")
    axis.axvline(oracle_center, color="black", linestyle="--", linewidth=1.1, label=f"oracle center = {oracle_center:g}")
    axis.axhline(psnr_threshold, color="#777777", linestyle="--", linewidth=1.1, label=f"median psnr = {psnr_threshold:.2f}")
    axis.set_title("Per-sample oracle_fmv_t_eff_mean vs PSNR")
    axis.set_xlabel(ORACLE_COLUMN)
    axis.set_ylabel("psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_oracle_distance_distribution_plot(
    joined: pd.DataFrame,
    scatter_oracle_window_radius: float,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "oracle_fmv_t_eff_distance_psnr_distribution.png"
    dataframe = joined.copy()
    dataframe["oracle_psnr_label"] = build_oracle_distribution_labels(
        dataframe=dataframe,
        scatter_oracle_window_radius=scatter_oracle_window_radius,
    )
    psnr_threshold = float(dataframe[PSNR_COLUMN].median())
    label_order, label_display_names, colors = build_oracle_label_metadata()

    figure, axis = plt.subplots(figsize=(10.5, 6.4), dpi=160)
    for label in label_order:
        label_dataframe = dataframe[dataframe["oracle_psnr_label"] == label]
        if len(label_dataframe) == 0:
            continue
        axis.scatter(
            label_dataframe[ORACLE_DISTANCE_COLUMN],
            label_dataframe[PSNR_COLUMN],
            s=16,
            alpha=0.42,
            edgecolors="none",
            color=colors[label],
            label=f"{label_display_names[label]} (n={len(label_dataframe)})",
        )

    axis.axvline(
        scatter_oracle_window_radius,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=f"clean threshold <= {scatter_oracle_window_radius:g}",
    )
    axis.axhline(psnr_threshold, color="#777777", linestyle="--", linewidth=1.1, label=f"median psnr = {psnr_threshold:.2f}")
    axis.set_title("Per-sample oracle_fmv_t_eff distance vs PSNR")
    axis.set_xlabel(f"|{ORACLE_COLUMN} - 0.5|")
    axis.set_ylabel("psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_record_summary(joined: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    record_summary = (
        joined.groupby(["inference_preset", "record", "mode"], as_index=False)
        .agg(
            samples=(PSNR_COLUMN, "size"),
            mean_psnr=(PSNR_COLUMN, "mean"),
            median_psnr=(PSNR_COLUMN, "median"),
            mean_fps=("fps_mean", "mean"),
            mean_fps_abs_error_max=("fps_abs_error_max", "mean"),
            mean_delta_time_symmetry_abs_ms=(SYMMETRY_COLUMN, "mean"),
            median_delta_time_symmetry_abs_ms=(SYMMETRY_COLUMN, "median"),
            mean_delta_time_position=("delta_time_position", "mean"),
            mean_delta_time_position_error=("delta_time_position_error", "mean"),
            mean_oracle_fmv_t_eff=(ORACLE_COLUMN, "mean"),
            median_oracle_fmv_t_eff=(ORACLE_COLUMN, "median"),
            mean_oracle_distance=(ORACLE_DISTANCE_COLUMN, "mean"),
            median_oracle_distance=(ORACLE_DISTANCE_COLUMN, "median"),
            mean_motion=(MOTION_COLUMN, "mean"),
        )
        .sort_values(["inference_preset", "record", "mode"])
        .reset_index(drop=True)
    )
    record_summary.to_csv(output_dir / "fps_stability_record_summary.csv", index=False)
    return record_summary


def write_csv_outputs(
    joined: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    motion_bin_summary: pd.DataFrame,
    symmetry_threshold_summary: pd.DataFrame,
    symmetry_motion_bin_summary: pd.DataFrame,
    oracle_window_summary: pd.DataFrame,
    oracle_motion_bin_summary: pd.DataFrame,
    scatter_oracle_window_radius: float,
    output_dir: Path,
) -> None:
    joined.to_csv(output_dir / "fps_stability_sample_comparison.csv", index=False)
    threshold_summary.to_csv(output_dir / "fps_stability_threshold_summary.csv", index=False)
    motion_bin_summary.to_csv(output_dir / "fps_stability_motion_bin_summary.csv", index=False)
    symmetry_threshold_summary.to_csv(output_dir / "delta_time_symmetry_threshold_summary.csv", index=False)
    symmetry_motion_bin_summary.to_csv(output_dir / "delta_time_symmetry_motion_bin_summary.csv", index=False)
    oracle_window_summary.to_csv(output_dir / "oracle_fmv_t_eff_window_summary.csv", index=False)
    oracle_motion_bin_summary.to_csv(output_dir / "oracle_fmv_t_eff_motion_bin_summary.csv", index=False)
    oracle_clean_samples = joined[joined[ORACLE_DISTANCE_COLUMN] <= scatter_oracle_window_radius].copy()
    oracle_clean_samples.to_csv(output_dir / "oracle_fmv_t_eff_clean_samples.csv", index=False)


def run(args: argparse.Namespace) -> None:
    if args.target_fps <= 0.0:
        raise ValueError(f"target-fps must be positive: target_fps={args.target_fps}")
    if args.motion_bin_width <= 0.0:
        raise ValueError(f"motion-bin-width must be positive: motion_bin_width={args.motion_bin_width}")
    if args.min_bin_samples <= 0:
        raise ValueError(f"min-bin-samples must be positive: min_bin_samples={args.min_bin_samples}")
    if args.scatter_threshold_fps <= 0.0:
        raise ValueError(f"scatter-threshold-fps must be positive: scatter_threshold_fps={args.scatter_threshold_fps}")
    if args.scatter_symmetry_threshold_ms <= 0.0:
        raise ValueError(
            f"scatter-symmetry-threshold-ms must be positive: "
            f"scatter_symmetry_threshold_ms={args.scatter_symmetry_threshold_ms}"
        )
    if args.highlight_symmetry_threshold_ms <= 0.0:
        raise ValueError(
            f"highlight-symmetry-threshold-ms must be positive: "
            f"highlight_symmetry_threshold_ms={args.highlight_symmetry_threshold_ms}"
        )
    if args.oracle_center <= 0.0 or args.oracle_center >= 1.0:
        raise ValueError(f"oracle-center must be in (0, 1): oracle_center={args.oracle_center}")
    if args.scatter_oracle_window_radius <= 0.0:
        raise ValueError(
            f"scatter-oracle-window-radius must be positive: "
            f"scatter_oracle_window_radius={args.scatter_oracle_window_radius}"
        )
    if args.highlight_oracle_window_radius <= 0.0:
        raise ValueError(
            f"highlight-oracle-window-radius must be positive: "
            f"highlight_oracle_window_radius={args.highlight_oracle_window_radius}"
        )

    metrics = load_metrics(metrics_csv=args.metrics_csv)
    timing = load_timing_rows(
        dataset_json_dir=args.dataset_json_dir,
        metrics=metrics,
        target_fps=float(args.target_fps),
    )
    raw_stats = load_raw_sequence_stats(raw_sequence_root=args.raw_sequence_root)
    joined = build_joined_dataframe(metrics=metrics, timing=timing, raw_stats=raw_stats)
    joined = add_oracle_distance_columns(dataframe=joined, oracle_center=float(args.oracle_center))
    threshold_summary, motion_bin_summary = build_threshold_outputs(
        joined=joined,
        fps_threshold_values=args.fps_threshold_values,
        motion_bin_width=float(args.motion_bin_width),
        min_bin_samples=int(args.min_bin_samples),
    )
    symmetry_threshold_summary, symmetry_motion_bin_summary = build_symmetry_threshold_outputs(
        joined=joined,
        symmetry_threshold_ms_values=args.symmetry_threshold_ms_values,
        motion_bin_width=float(args.motion_bin_width),
        min_bin_samples=int(args.min_bin_samples),
    )
    oracle_window_summary, oracle_motion_bin_summary = build_oracle_window_outputs(
        joined=joined,
        oracle_center=float(args.oracle_center),
        oracle_window_radii=args.oracle_window_radii,
        motion_bin_width=float(args.motion_bin_width),
        min_bin_samples=int(args.min_bin_samples),
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_outputs(
        joined=joined,
        threshold_summary=threshold_summary,
        motion_bin_summary=motion_bin_summary,
        symmetry_threshold_summary=symmetry_threshold_summary,
        symmetry_motion_bin_summary=symmetry_motion_bin_summary,
        oracle_window_summary=oracle_window_summary,
        oracle_motion_bin_summary=oracle_motion_bin_summary,
        scatter_oracle_window_radius=float(args.scatter_oracle_window_radius),
        output_dir=output_dir,
    )
    record_summary = write_record_summary(joined=joined, output_dir=output_dir)
    threshold_dashboard_path = write_threshold_dashboard(
        threshold_summary=threshold_summary,
        motion_bin_summary=motion_bin_summary,
        output_dir=output_dir,
    )
    symmetry_threshold_dashboard_path = write_symmetry_threshold_dashboard(
        threshold_summary=symmetry_threshold_summary,
        motion_bin_summary=symmetry_motion_bin_summary,
        output_dir=output_dir,
    )
    symmetry_gain_dashboard_path = write_symmetry_gain_dashboard(
        threshold_summary=symmetry_threshold_summary,
        highlight_threshold_ms=float(args.highlight_symmetry_threshold_ms),
        output_dir=output_dir,
    )
    oracle_window_dashboard_path = write_oracle_window_dashboard(
        window_summary=oracle_window_summary,
        motion_bin_summary=oracle_motion_bin_summary,
        output_dir=output_dir,
    )
    oracle_gain_dashboard_path = write_oracle_gain_dashboard(
        window_summary=oracle_window_summary,
        oracle_center=float(args.oracle_center),
        highlight_window_radius=float(args.highlight_oracle_window_radius),
        output_dir=output_dir,
    )
    fps_mean_distribution_path = write_fps_mean_distribution_plot(
        joined=joined,
        scatter_threshold_fps=float(args.scatter_threshold_fps),
        target_fps=float(args.target_fps),
        output_dir=output_dir,
    )
    fps_error_distribution_path = write_fps_error_distribution_plot(
        joined=joined,
        scatter_threshold_fps=float(args.scatter_threshold_fps),
        target_fps=float(args.target_fps),
        output_dir=output_dir,
    )
    symmetry_distribution_path = write_symmetry_error_distribution_plot(
        joined=joined,
        scatter_symmetry_threshold_ms=float(args.scatter_symmetry_threshold_ms),
        output_dir=output_dir,
    )
    delta_time_position_distribution_path = write_delta_time_position_distribution_plot(
        joined=joined,
        scatter_symmetry_threshold_ms=float(args.scatter_symmetry_threshold_ms),
        output_dir=output_dir,
    )
    oracle_distribution_path = write_oracle_fmv_t_eff_distribution_plot(
        joined=joined,
        oracle_center=float(args.oracle_center),
        scatter_oracle_window_radius=float(args.scatter_oracle_window_radius),
        output_dir=output_dir,
    )
    oracle_distance_distribution_path = write_oracle_distance_distribution_plot(
        joined=joined,
        scatter_oracle_window_radius=float(args.scatter_oracle_window_radius),
        output_dir=output_dir,
    )

    print(f"metrics_csv={args.metrics_csv}")
    print(f"dataset_json_dir={args.dataset_json_dir}")
    print(f"raw_sequence_root={args.raw_sequence_root}")
    print(f"output_dir={output_dir}")
    print(f"sample_count={len(joined)}")
    print(f"threshold_dashboard={threshold_dashboard_path}")
    print(f"symmetry_threshold_dashboard={symmetry_threshold_dashboard_path}")
    print(f"symmetry_gain_dashboard={symmetry_gain_dashboard_path}")
    print(f"oracle_window_dashboard={oracle_window_dashboard_path}")
    print(f"oracle_gain_dashboard={oracle_gain_dashboard_path}")
    print(f"fps_mean_distribution={fps_mean_distribution_path}")
    print(f"fps_error_distribution={fps_error_distribution_path}")
    print(f"symmetry_distribution={symmetry_distribution_path}")
    print(f"delta_time_position_distribution={delta_time_position_distribution_path}")
    print(f"oracle_distribution={oracle_distribution_path}")
    print(f"oracle_distance_distribution={oracle_distance_distribution_path}")
    print(threshold_summary.to_string(index=False))
    print(symmetry_threshold_summary.to_string(index=False))
    print(oracle_window_summary.to_string(index=False))
    print(record_summary.to_string(index=False))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
