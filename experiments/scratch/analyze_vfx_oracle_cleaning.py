from __future__ import annotations

import argparse
import os
import re
import sys
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

from experiments.scratch.split_motion_psnr_buckets import BUCKET_COLORS
from experiments.scratch.split_motion_psnr_buckets import BUCKET_ORDER
from experiments.scratch.split_motion_psnr_buckets import build_bucket_summary
from experiments.scratch.split_motion_psnr_buckets import classify_bucket

DEFAULT_METRICS_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\DGX_inference_outputs\IFRNet_FineTuning_0611\metrics.csv",
)
DEFAULT_RAW_SEQUENCE_ROOT: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\Dataset\VFX_0416",
)
DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\dgx_vfx_0416_oracle_cleaning\ifrnet_finetuning_0611",
)
DEFAULT_ORACLE_CENTER: float = 0.5
DEFAULT_HIGHLIGHT_ORACLE_WINDOW_RADIUS: float = 0.025
DEFAULT_ORACLE_WINDOW_RADII: tuple[float, ...] = (0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25)
DEFAULT_MOTION_BIN_WIDTH: float = 1.0
DEFAULT_MIN_BIN_SAMPLES: int = 40
DEFAULT_MOTION_THRESHOLD_QUANTILE: float = 0.9
DEFAULT_PSNR_THRESHOLD_QUANTILE: float = 0.15
DEFAULT_LOW_PSNR_THRESHOLD: float = 25.0
KEY_COLUMNS: tuple[str, ...] = ("record", "major_mode_id", "minor_mode_id", "fps_value", "frame_range")
PSNR_COLUMN: str = "psnr"
MOTION_COLUMN: str = "motion_magnitude_mean"
ORACLE_COLUMN: str = "oracle_fmv_t_eff_mean"
ORACLE_DISTANCE_COLUMN: str = "oracle_fmv_t_eff_distance"
FRAME_RANGE_PATTERN = re.compile(r"frame_(\d+)_(\d+)")
BUCKET_DISPLAY_LABELS: dict[str, str] = {
    "large_motion_artifacts": "large motion\nartifacts",
    "not_large_motion_high_psnr": "not large motion\nhigh psnr",
    "other_artifacts": "other\nartifacts",
    "large_motion_high_psnr": "large motion\nhigh psnr",
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze oracle effective-time cleaning for the DGX VFX_0416 IFRNet FineTuning output."
    )
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--raw-sequence-root", type=Path, default=DEFAULT_RAW_SEQUENCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oracle-center", type=float, default=DEFAULT_ORACLE_CENTER)
    parser.add_argument("--highlight-oracle-window-radius", type=float, default=DEFAULT_HIGHLIGHT_ORACLE_WINDOW_RADIUS)
    parser.add_argument("--oracle-window-radii", type=float, nargs="+", default=list(DEFAULT_ORACLE_WINDOW_RADII))
    parser.add_argument("--motion-bin-width", type=float, default=DEFAULT_MOTION_BIN_WIDTH)
    parser.add_argument("--min-bin-samples", type=int, default=DEFAULT_MIN_BIN_SAMPLES)
    parser.add_argument("--motion-threshold-quantile", type=float, default=DEFAULT_MOTION_THRESHOLD_QUANTILE)
    parser.add_argument("--psnr-threshold-quantile", type=float, default=DEFAULT_PSNR_THRESHOLD_QUANTILE)
    parser.add_argument("--low-psnr-threshold", type=float, default=DEFAULT_LOW_PSNR_THRESHOLD)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.oracle_center <= 0.0 or args.oracle_center >= 1.0:
        raise ValueError(f"oracle-center must be in (0, 1): oracle_center={args.oracle_center}")
    if args.highlight_oracle_window_radius <= 0.0:
        raise ValueError(
            f"highlight-oracle-window-radius must be positive: radius={args.highlight_oracle_window_radius}"
        )
    invalid_radii = [float(radius) for radius in args.oracle_window_radii if float(radius) <= 0.0]
    if len(invalid_radii) > 0:
        raise ValueError(f"oracle-window-radii must be positive: invalid_radii={invalid_radii}")
    if args.motion_bin_width <= 0.0:
        raise ValueError(f"motion-bin-width must be positive: motion_bin_width={args.motion_bin_width}")
    if args.min_bin_samples <= 0:
        raise ValueError(f"min-bin-samples must be positive: min_bin_samples={args.min_bin_samples}")
    if args.motion_threshold_quantile <= 0.0 or args.motion_threshold_quantile >= 1.0:
        raise ValueError(f"motion-threshold-quantile must be in (0, 1): value={args.motion_threshold_quantile}")
    if args.psnr_threshold_quantile <= 0.0 or args.psnr_threshold_quantile >= 1.0:
        raise ValueError(f"psnr-threshold-quantile must be in (0, 1): value={args.psnr_threshold_quantile}")
    if args.low_psnr_threshold <= 0.0:
        raise ValueError(f"low-psnr-threshold must be positive: value={args.low_psnr_threshold}")


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_columns(dataframe: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"{label} is missing required columns: {missing_columns}")


def parse_mode_components(mode: str) -> tuple[str, str, int]:
    mode_parts = str(mode).replace("\\", "/").split("/")
    if len(mode_parts) != 3:
        raise ValueError(f"Unexpected mode format: mode={mode}")
    major_parts = mode_parts[0].split("_")
    minor_parts = mode_parts[1].split("_")
    if len(major_parts) < 2 or len(minor_parts) < 3:
        raise ValueError(f"Unexpected mode hierarchy: mode={mode}")
    if not mode_parts[2].startswith("fps_"):
        raise ValueError(f"Unexpected fps mode component: mode={mode}")
    return major_parts[0], minor_parts[-1], int(mode_parts[2].replace("fps_", ""))


def parse_frame_range(frame_range: str) -> tuple[int, int]:
    match = FRAME_RANGE_PATTERN.fullmatch(str(frame_range))
    if match is None:
        raise ValueError(f"Unexpected frame_range format: frame_range={frame_range}")
    return int(match.group(1)), int(match.group(2))


def format_frame_range(img0: int, img2: int) -> str:
    return f"frame_{img0:04d}_{img2:04d}"


def parse_raw_sequence_name(csv_path: Path) -> tuple[str, str, str, int]:
    parent_name = csv_path.parent.name
    if not parent_name.endswith("_preprocessed"):
        raise ValueError(f"Expected *_preprocessed parent directory: path={csv_path}")
    record = parent_name.replace("_preprocessed", "")
    stem = csv_path.name.replace("_raw_sequence_frame_index.csv", "")
    stem_parts = stem.split("_")
    if len(stem_parts) != 4 or stem_parts[2] != "fps":
        raise ValueError(f"Unexpected raw-sequence filename format: path={csv_path}")
    return record, stem_parts[0], stem_parts[1], int(stem_parts[3])


def load_metrics(metrics_csv: Path) -> pd.DataFrame:
    require_existing_path(metrics_csv, "metrics CSV")
    metrics = pd.read_csv(metrics_csv)
    if len(metrics) == 0:
        raise ValueError(f"Metrics CSV is empty: {metrics_csv}")
    require_columns(
        dataframe=metrics,
        columns=("record", "mode", "frame_range", PSNR_COLUMN, "distance_index_mean", "distance_index_median"),
        label="metrics CSV",
    )
    metrics = metrics.copy()
    mode_components = metrics["mode"].map(parse_mode_components)
    metrics["major_mode_id"] = [component[0] for component in mode_components]
    metrics["minor_mode_id"] = [component[1] for component in mode_components]
    metrics["fps_value"] = [component[2] for component in mode_components]
    frame_components = metrics["frame_range"].map(parse_frame_range)
    metrics["img0"] = [component[0] for component in frame_components]
    metrics["img2"] = [component[1] for component in frame_components]

    duplicate_count = int(metrics.duplicated(subset=list(KEY_COLUMNS)).sum())
    if duplicate_count > 0:
        raise ValueError(f"Metrics CSV contains duplicate sample keys: duplicate_count={duplicate_count}")
    return metrics


def load_raw_sequence_stats(raw_sequence_root: Path) -> pd.DataFrame:
    require_existing_path(raw_sequence_root, "raw sequence root")
    rows: list[pd.DataFrame] = []
    for csv_path in sorted(raw_sequence_root.rglob("*_raw_sequence_frame_index.csv")):
        record, major_mode_id, minor_mode_id, fps_value = parse_raw_sequence_name(csv_path)
        dataframe = pd.read_csv(csv_path)
        require_columns(
            dataframe=dataframe,
            columns=("img0", "img1", "img2", "valid", MOTION_COLUMN, ORACLE_COLUMN),
            label=f"raw sequence CSV: {csv_path}",
        )
        dataframe = dataframe.copy()
        dataframe["record"] = record
        dataframe["major_mode_id"] = major_mode_id
        dataframe["minor_mode_id"] = minor_mode_id
        dataframe["fps_value"] = fps_value
        dataframe["frame_range"] = [
            format_frame_range(img0=int(img0), img2=int(img2))
            for img0, img2 in zip(dataframe["img0"], dataframe["img2"])
        ]
        selected_columns = [
            "record",
            "major_mode_id",
            "minor_mode_id",
            "fps_value",
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
            ORACLE_COLUMN,
            "oracle_bmv_t_eff_mean",
            "oracle_t_eff_gap_mean",
            "oracle_fmv_valid_ratio",
            "oracle_fmv_clamped_ratio",
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
            "img0": "raw_img0",
            "img1": "raw_img1",
            "img2": "raw_img2",
        }
    )


def build_joined_dataframe(metrics: pd.DataFrame, raw_stats: pd.DataFrame, oracle_center: float) -> pd.DataFrame:
    joined = metrics.merge(raw_stats, on=list(KEY_COLUMNS), how="left", indicator="raw_merge")
    unmatched = joined[joined["raw_merge"] != "both"]
    if len(unmatched) > 0:
        preview = unmatched[["record", "mode", "frame_range"]].head(10).to_dict("records")
        raise ValueError(f"Failed to match metrics rows to VFX raw sequence stats: preview={preview}")
    joined = joined.drop(columns=["raw_merge"])
    joined[ORACLE_DISTANCE_COLUMN] = (joined[ORACLE_COLUMN].astype(float) - oracle_center).abs()
    joined["oracle_fmv_t_eff_side"] = np.where(joined[ORACLE_COLUMN] < oracle_center, "early", "late")
    return joined


def build_oracle_window_summary(dataframe: pd.DataFrame, oracle_center: float, radii: Sequence[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    all_mean = float(dataframe[PSNR_COLUMN].mean())
    all_median = float(dataframe[PSNR_COLUMN].median())
    all_samples = int(len(dataframe))
    rows.append(
        {
            "window_label": "all",
            "radius": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "samples": all_samples,
            "retention_ratio": 1.0,
            "mean_psnr": all_mean,
            "median_psnr": all_median,
            "mean_psnr_gain": 0.0,
            "median_psnr_gain": 0.0,
        }
    )
    for radius in sorted(set(float(value) for value in radii)):
        lower = oracle_center - radius
        upper = oracle_center + radius
        clean = dataframe[dataframe[ORACLE_DISTANCE_COLUMN] <= radius]
        rows.append(
            {
                "window_label": f"{lower:.3f}-{upper:.3f}",
                "radius": radius,
                "lower": lower,
                "upper": upper,
                "samples": int(len(clean)),
                "retention_ratio": float(len(clean) / all_samples),
                "mean_psnr": float(clean[PSNR_COLUMN].mean()),
                "median_psnr": float(clean[PSNR_COLUMN].median()),
                "mean_psnr_gain": float(clean[PSNR_COLUMN].mean() - all_mean),
                "median_psnr_gain": float(clean[PSNR_COLUMN].median() - all_median),
            }
        )
    return pd.DataFrame(rows)


def build_motion_bin_summary(
    dataframe: pd.DataFrame,
    label: str,
    motion_bin_width: float,
    min_bin_samples: int,
) -> pd.DataFrame:
    binned = dataframe.copy()
    binned["motion_bin"] = np.floor(binned[MOTION_COLUMN] / motion_bin_width) * motion_bin_width
    summary = (
        binned.groupby("motion_bin")
        .agg(
            samples=(PSNR_COLUMN, "size"),
            mean_psnr=(PSNR_COLUMN, "mean"),
            median_psnr=(PSNR_COLUMN, "median"),
            motion_mean=(MOTION_COLUMN, "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["samples"] >= min_bin_samples].copy()
    summary["window_label"] = label
    return summary


def write_oracle_window_dashboard(
    dataframe: pd.DataFrame,
    summary: pd.DataFrame,
    oracle_center: float,
    motion_bin_width: float,
    min_bin_samples: int,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "oracle_t_eff_window_dashboard.png"
    plot_rows = summary.copy()
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=160)
    figure.suptitle("DGX VFX oracle_fmv_t_eff clean-window sweep", fontsize=16)

    window_rows = plot_rows[plot_rows["window_label"] != "all"].copy()
    selected_windows = ["all"] + window_rows["window_label"].tolist()
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(selected_windows)))

    axis = axes[0, 0]
    for color, window_label in zip(colors, selected_windows):
        if window_label == "all":
            selected = dataframe
        else:
            radius = float(window_rows[window_rows["window_label"] == window_label]["radius"].iloc[0])
            selected = dataframe[dataframe[ORACLE_DISTANCE_COLUMN] <= radius]
        bin_summary = build_motion_bin_summary(
            dataframe=selected,
            label=window_label,
            motion_bin_width=motion_bin_width,
            min_bin_samples=min_bin_samples,
        )
        axis.plot(bin_summary["motion_mean"], bin_summary["mean_psnr"], marker="o", linewidth=1.4, markersize=3.5, color=color, label=window_label)
    axis.set_title("Mean PSNR by motion bin")
    axis.set_xlabel(MOTION_COLUMN)
    axis.set_ylabel("mean psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    for color, window_label in zip(colors, selected_windows):
        if window_label == "all":
            selected = dataframe
        else:
            radius = float(window_rows[window_rows["window_label"] == window_label]["radius"].iloc[0])
            selected = dataframe[dataframe[ORACLE_DISTANCE_COLUMN] <= radius]
        bin_summary = build_motion_bin_summary(
            dataframe=selected,
            label=window_label,
            motion_bin_width=motion_bin_width,
            min_bin_samples=min_bin_samples,
        )
        axis.plot(bin_summary["motion_mean"], bin_summary["median_psnr"], marker="o", linewidth=1.4, markersize=3.5, color=color, label=window_label)
    axis.set_title("Median PSNR by motion bin")
    axis.set_xlabel(MOTION_COLUMN)
    axis.set_ylabel("median psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    bars = axis.bar(plot_rows["window_label"], plot_rows["mean_psnr"], color=colors, alpha=0.9)
    axis.set_title("Overall mean PSNR after filtering")
    axis.set_xlabel("oracle_fmv_t_eff_mean window")
    axis.set_ylabel("mean psnr (dB)")
    axis.tick_params(axis="x", rotation=35, labelsize=8)
    axis.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, plot_rows["mean_psnr"]):
        axis.text(bar.get_x() + bar.get_width() / 2.0, float(value) + 0.03, f"{float(value):.2f}", ha="center", va="bottom", fontsize=8)

    axis = axes[1, 1]
    bars = axis.bar(plot_rows["window_label"], plot_rows["retention_ratio"] * 100.0, color=colors, alpha=0.9)
    axis.set_title("Sample retention after filtering")
    axis.set_xlabel("oracle_fmv_t_eff_mean window")
    axis.set_ylabel("retained samples (%)")
    axis.set_ylim(0.0, 105.0)
    axis.tick_params(axis="x", rotation=35, labelsize=8)
    axis.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, plot_rows["retention_ratio"]):
        axis.text(bar.get_x() + bar.get_width() / 2.0, float(value) * 100.0 + 1.0, f"{float(value) * 100.0:.1f}%", ha="center", va="bottom", fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_bucketed_dataframe(
    dataframe: pd.DataFrame,
    motion_threshold: float,
    psnr_threshold: float,
) -> pd.DataFrame:
    bucketed = dataframe.copy()
    bucketed["psnr_summary"] = bucketed[PSNR_COLUMN].astype(float)
    bucketed["motion_threshold"] = motion_threshold
    bucketed["psnr_threshold"] = psnr_threshold
    bucketed["bucket"] = [
        classify_bucket(
            motion_value=float(motion_value),
            psnr_value=float(psnr_value),
            motion_threshold=motion_threshold,
            psnr_threshold=psnr_threshold,
        )
        for motion_value, psnr_value in zip(bucketed[MOTION_COLUMN], bucketed["psnr_summary"])
    ]
    bucketed["bucket"] = pd.Categorical(bucketed["bucket"], categories=BUCKET_ORDER, ordered=True)
    return bucketed.sort_values(["bucket", MOTION_COLUMN, "psnr_summary"], ascending=[True, False, True]).reset_index(drop=True)


def build_complete_bucket_summary(bucketed: pd.DataFrame) -> pd.DataFrame:
    summary = build_bucket_summary(dataframe=bucketed, motion_column=MOTION_COLUMN)
    summary = summary.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    summary["samples"] = summary["samples"].fillna(0).astype(int)
    summary["ratio"] = summary["ratio"].fillna(0.0).astype(float)
    return summary


def write_distribution_dashboard(
    bucketed: pd.DataFrame,
    summary: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9), dpi=160)
    motion_axis = axes[0, 0]
    scatter_axis = axes[0, 1]
    psnr_axis = axes[1, 0]
    coverage_axis = axes[1, 1]
    motion_threshold = float(bucketed["motion_threshold"].iloc[0])
    psnr_threshold = float(bucketed["psnr_threshold"].iloc[0])

    for bucket_name in BUCKET_ORDER:
        bucket_dataframe = bucketed[bucketed["bucket"] == bucket_name]
        if len(bucket_dataframe) == 0:
            continue
        label = f"{bucket_name} (n={len(bucket_dataframe)})"
        color = BUCKET_COLORS[bucket_name]
        motion_axis.hist(bucket_dataframe[MOTION_COLUMN], bins=45, alpha=0.38, color=color, label=label)
        scatter_axis.scatter(
            bucket_dataframe[MOTION_COLUMN],
            bucket_dataframe["psnr_summary"],
            s=10,
            alpha=0.35,
            edgecolors="none",
            color=color,
            label=label,
        )
        psnr_axis.hist(bucket_dataframe["psnr_summary"], bins=45, alpha=0.38, color=color, label=label)

    motion_axis.axvline(motion_threshold, color="black", linestyle="--", linewidth=1.1, label=f"motion threshold = {motion_threshold:.2f}")
    motion_axis.set_xlabel(MOTION_COLUMN)
    motion_axis.set_ylabel("samples")
    motion_axis.set_title("Per-sample motion distribution")
    motion_axis.grid(True, axis="y", alpha=0.25)
    motion_axis.legend(fontsize=8)

    scatter_axis.axvline(motion_threshold, color="black", linestyle="--", linewidth=1.1, label=f"motion threshold = {motion_threshold:.2f}")
    scatter_axis.axhline(psnr_threshold, color="gray", linestyle="--", linewidth=1.1, label=f"psnr threshold = {psnr_threshold:.2f}")
    scatter_axis.set_xlabel(MOTION_COLUMN)
    scatter_axis.set_ylabel("psnr_summary")
    scatter_axis.set_title("Per-sample motion vs PSNR")
    scatter_axis.grid(True, alpha=0.25)
    scatter_axis.legend(fontsize=8, markerscale=1.3)

    psnr_axis.axvline(psnr_threshold, color="gray", linestyle="--", linewidth=1.1, label=f"psnr threshold = {psnr_threshold:.2f}")
    psnr_axis.set_xlabel("psnr_summary")
    psnr_axis.set_ylabel("samples")
    psnr_axis.set_title("Per-sample PSNR distribution")
    psnr_axis.grid(True, axis="y", alpha=0.25)
    psnr_axis.legend(fontsize=8)

    summary = summary.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    coverage_axis.barh(
        [BUCKET_DISPLAY_LABELS[str(bucket_name)] for bucket_name in summary["bucket"]],
        summary["ratio"] * 100.0,
        color=[BUCKET_COLORS[str(bucket_name)] for bucket_name in summary["bucket"]],
    )
    for index, row in summary.iterrows():
        coverage_axis.text(
            float(row["ratio"]) * 100.0 + 0.6,
            index,
            f"{float(row['ratio']) * 100.0:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
        )
    coverage_axis.set_xlabel("dataset percentage")
    coverage_axis.set_title("Bucket coverage")
    coverage_axis.grid(True, axis="x", alpha=0.25)

    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def write_low_psnr_outputs(clean: pd.DataFrame, low_psnr_threshold: float, output_dir: Path) -> tuple[Path, Path]:
    low = clean[clean[PSNR_COLUMN] < low_psnr_threshold].copy()
    selected_columns = [
        "record",
        "mode",
        "frame_range",
        "img0",
        "img2",
        PSNR_COLUMN,
        MOTION_COLUMN,
        ORACLE_COLUMN,
        ORACLE_DISTANCE_COLUMN,
        "distance_index_mean",
        "raw_distance_index_mean",
        "bucket",
        "image_0_path",
        "image_1_path",
        "image_gt_path",
        "image_pred_path",
    ]
    available_columns = [column for column in selected_columns if column in low.columns]
    frames_path = output_dir / f"low_psnr_below_{low_psnr_threshold:g}_after_oracle_cleaning.csv"
    summary_path = output_dir / f"low_psnr_below_{low_psnr_threshold:g}_after_oracle_cleaning_summary.csv"
    low[available_columns].sort_values([PSNR_COLUMN, "record", "mode", "frame_range"]).to_csv(frames_path, index=False)
    pd.DataFrame(
        [
            {
                "low_psnr_threshold": low_psnr_threshold,
                "low_psnr_samples": int(len(low)),
                "clean_samples": int(len(clean)),
                "low_psnr_ratio_of_clean": float(len(low) / len(clean)),
            }
        ]
    ).to_csv(summary_path, index=False)
    return frames_path, summary_path


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(args.metrics_csv)
    raw_stats = load_raw_sequence_stats(args.raw_sequence_root)
    joined = build_joined_dataframe(metrics=metrics, raw_stats=raw_stats, oracle_center=float(args.oracle_center))
    summary = build_oracle_window_summary(
        dataframe=joined,
        oracle_center=float(args.oracle_center),
        radii=[float(value) for value in args.oracle_window_radii],
    )
    joined.to_csv(output_dir / "vfx_oracle_sample_comparison.csv", index=False)
    summary.to_csv(output_dir / "oracle_t_eff_window_summary.csv", index=False)

    dashboard_path = write_oracle_window_dashboard(
        dataframe=joined,
        summary=summary,
        oracle_center=float(args.oracle_center),
        motion_bin_width=float(args.motion_bin_width),
        min_bin_samples=int(args.min_bin_samples),
        output_dir=output_dir,
    )

    highlight_radius = float(args.highlight_oracle_window_radius)
    clean = joined[joined[ORACLE_DISTANCE_COLUMN] <= highlight_radius].copy()
    clean.to_csv(output_dir / "oracle_t_eff_clean_samples.csv", index=False)

    motion_threshold = float(joined[MOTION_COLUMN].quantile(float(args.motion_threshold_quantile)))
    psnr_threshold = float(joined[PSNR_COLUMN].quantile(float(args.psnr_threshold_quantile)))
    thresholds = pd.DataFrame(
        [
            {
                "motion_column": MOTION_COLUMN,
                "motion_threshold_quantile": float(args.motion_threshold_quantile),
                "motion_threshold": motion_threshold,
                "psnr_column": PSNR_COLUMN,
                "psnr_threshold_quantile": float(args.psnr_threshold_quantile),
                "psnr_threshold": psnr_threshold,
                "oracle_center": float(args.oracle_center),
                "highlight_oracle_window_radius": highlight_radius,
                "oracle_lower": float(args.oracle_center) - highlight_radius,
                "oracle_upper": float(args.oracle_center) + highlight_radius,
            }
        ]
    )
    thresholds.to_csv(output_dir / "motion_psnr_thresholds.csv", index=False)

    all_bucketed = build_bucketed_dataframe(joined, motion_threshold=motion_threshold, psnr_threshold=psnr_threshold)
    clean_bucketed = build_bucketed_dataframe(clean, motion_threshold=motion_threshold, psnr_threshold=psnr_threshold)
    all_bucketed.to_csv(output_dir / "all_samples_bucketed.csv", index=False)
    clean_bucketed.to_csv(output_dir / "oracle_t_eff_clean_bucketed.csv", index=False)
    all_bucket_summary = build_complete_bucket_summary(all_bucketed)
    clean_bucket_summary = build_complete_bucket_summary(clean_bucketed)
    all_bucket_summary.to_csv(output_dir / "all_samples_bucket_summary.csv", index=False)
    clean_bucket_summary.to_csv(output_dir / "oracle_t_eff_clean_bucket_summary.csv", index=False)

    write_distribution_dashboard(
        bucketed=all_bucketed,
        summary=all_bucket_summary,
        title=(
            "DGX VFX all samples motion/PSNR bucket distribution\n"
            f"motion >= {motion_threshold:.2f}, psnr < {psnr_threshold:.2f}"
        ),
        output_path=output_dir / "all_samples_motion_distribution.png",
    )
    write_distribution_dashboard(
        bucketed=clean_bucketed,
        summary=clean_bucket_summary,
        title=(
            "DGX VFX oracle_fmv_t_eff cleaned motion/PSNR bucket distribution\n"
            f"retained {len(clean_bucketed)}/{len(joined)} ({len(clean_bucketed) / len(joined) * 100.0:.1f}%), "
            f"motion >= {motion_threshold:.2f}, psnr < {psnr_threshold:.2f}"
        ),
        output_path=output_dir / "oracle_t_eff_clean_motion_distribution.png",
    )
    low_frames_path, low_summary_path = write_low_psnr_outputs(
        clean=clean_bucketed,
        low_psnr_threshold=float(args.low_psnr_threshold),
        output_dir=output_dir,
    )

    print(f"output_dir={output_dir}")
    print(f"joined_rows={len(joined)}")
    print(f"dashboard={dashboard_path}")
    print(f"all_distribution={output_dir / 'all_samples_motion_distribution.png'}")
    print(f"clean_distribution={output_dir / 'oracle_t_eff_clean_motion_distribution.png'}")
    print(f"low_frames={low_frames_path}")
    print(f"low_summary={low_summary_path}")
    print(summary.to_string(index=False))
    print(clean_bucket_summary.to_string(index=False))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
