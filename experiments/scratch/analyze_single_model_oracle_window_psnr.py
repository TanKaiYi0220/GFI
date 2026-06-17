from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_METRICS_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\DGX_outputs\IFRNet_Residual_FlowApprox_1_Layer_TwoStage_Splat_4Direction_MaskedArea_0611\checkpoints\test_epoch_50.csv",
)
DEFAULT_DATASET_ROOT: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\Dataset\VFX_0416",
)
DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\oracle_window_psnr\vfx_0416_masked_area_0611",
)
DEFAULT_ONLY_FPS: int = 60
DEFAULT_DIFFICULTY_NAME: str = "Difficult"
DEFAULT_ORACLE_CENTER: float = 0.5
DEFAULT_MOTION_BIN_WIDTH: float = 1.0
DEFAULT_MIN_BIN_SAMPLES: int = 20
DEFAULT_WINDOW_RADII: tuple[float, ...] = (0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25)
KEY_COLUMNS: tuple[str, ...] = ("record_name", "frame_range")
MOTION_COLUMN: str = "motion_magnitude_mean"
ORACLE_COLUMN: str = "oracle_fmv_t_eff_mean"
PSNR_COLUMN: str = "psnr"


@dataclass(frozen=True)
class WindowSpec:
    label: str
    radius: float | None
    lower: float | None
    upper: float | None


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one model's PSNR-vs-motion trend under oracle_fmv_t_eff_mean window filtering."
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=DEFAULT_METRICS_CSV,
        help="Path to one sample-level metrics CSV with record_name and frame_range.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to dataset root containing *_preprocessed/*_raw_sequence_frame_index.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for joined CSV, summaries, and figures.",
    )
    parser.add_argument(
        "--only-fps",
        type=int,
        default=DEFAULT_ONLY_FPS,
        help="FPS suffix for selecting raw_sequence CSV files.",
    )
    parser.add_argument(
        "--difficulty-name",
        type=str,
        default=DEFAULT_DIFFICULTY_NAME,
        help="Difficulty token used when reconstructing mode names from preprocessed filenames.",
    )
    parser.add_argument(
        "--oracle-center",
        type=float,
        default=DEFAULT_ORACLE_CENTER,
        help="Center value for oracle_fmv_t_eff_mean filtering.",
    )
    parser.add_argument(
        "--motion-bin-width",
        type=float,
        default=DEFAULT_MOTION_BIN_WIDTH,
        help="Uniform motion bin width for trend plots.",
    )
    parser.add_argument(
        "--min-bin-samples",
        type=int,
        default=DEFAULT_MIN_BIN_SAMPLES,
        help="Minimum samples required for a motion bin to appear in the trend plot.",
    )
    parser.add_argument(
        "--window-radii",
        type=float,
        nargs="+",
        default=list(DEFAULT_WINDOW_RADII),
        help="Oracle window radii around oracle-center.",
    )
    return parser.parse_args(argv)


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_columns(dataframe: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"{label} is missing columns: {missing_columns}")


def build_window_specs(oracle_center: float, window_radii: Sequence[float]) -> list[WindowSpec]:
    windows = [WindowSpec(label="all", radius=None, lower=None, upper=None)]
    for radius in sorted(set(float(radius) for radius in window_radii)):
        if radius <= 0.0:
            raise ValueError(f"window radius must be positive: {radius}")
        lower = oracle_center - radius
        upper = oracle_center + radius
        windows.append(
            WindowSpec(
                label=f"{lower:.3f}-{upper:.3f}",
                radius=radius,
                lower=lower,
                upper=upper,
            )
        )
    return windows


def parse_preprocessed_mode(csv_path: Path, difficulty_name: str) -> tuple[str, str]:
    parent_name = csv_path.parent.name
    if not parent_name.endswith("_preprocessed"):
        raise ValueError(f"Expected *_preprocessed parent directory: {csv_path}")
    record = parent_name.replace("_preprocessed", "")

    stem = csv_path.name.replace("_raw_sequence_frame_index.csv", "")
    stem_parts = stem.split("_")
    if len(stem_parts) != 4:
        raise ValueError(f"Unexpected preprocessed filename format: {csv_path.name}")

    major_mode_id = stem_parts[0]
    minor_mode_id = stem_parts[1]
    fps = stem_parts[3]
    mode = f"{major_mode_id}_{difficulty_name}/{major_mode_id}_{difficulty_name}_{minor_mode_id}/fps_{fps}"
    return record, mode


def load_raw_sample_stats(dataset_root: Path, only_fps: int, difficulty_name: str) -> pd.DataFrame:
    require_existing_path(dataset_root, "dataset root")
    pattern = f"*_fps_{only_fps}_raw_sequence_frame_index.csv"
    rows: list[pd.DataFrame] = []
    for csv_path in dataset_root.rglob(pattern):
        if not csv_path.parent.name.endswith("_preprocessed"):
            continue
        record, mode = parse_preprocessed_mode(csv_path=csv_path, difficulty_name=difficulty_name)
        dataframe = pd.read_csv(csv_path)
        if len(dataframe) == 0:
            continue
        dataframe = dataframe.copy()
        dataframe["mode"] = mode
        dataframe["record_name"] = record + "_" + dataframe["mode"].astype(str)
        dataframe["frame_range"] = dataframe.apply(
            lambda row: f"frame_{int(row['img0']):04d}_{int(row['img2']):04d}",
            axis=1,
        )
        rows.append(dataframe)

    if len(rows) == 0:
        raise RuntimeError(f"No raw sequence CSV files found for only_fps={only_fps} under {dataset_root}")

    result = pd.concat(rows, ignore_index=True)
    require_columns(
        dataframe=result,
        columns=[*KEY_COLUMNS, MOTION_COLUMN, ORACLE_COLUMN, "oracle_t_eff_gap_mean"],
        label="raw sample stats",
    )
    duplicate_count = int(result.duplicated(subset=list(KEY_COLUMNS)).sum())
    if duplicate_count > 0:
        raise ValueError(f"Raw sample stats contain duplicate sample keys: duplicate_count={duplicate_count}")
    return result


def load_metrics(metrics_csv: Path) -> pd.DataFrame:
    require_existing_path(metrics_csv, "metrics CSV")
    dataframe = pd.read_csv(metrics_csv)
    if len(dataframe) == 0:
        raise ValueError(f"Metrics CSV is empty: {metrics_csv}")
    require_columns(dataframe=dataframe, columns=[*KEY_COLUMNS, PSNR_COLUMN], label="metrics CSV")
    duplicate_count = int(dataframe.duplicated(subset=list(KEY_COLUMNS)).sum())
    if duplicate_count > 0:
        raise ValueError(f"Metrics CSV contains duplicate sample keys: duplicate_count={duplicate_count}")
    return dataframe


def build_joined_dataframe(metrics: pd.DataFrame, raw_stats: pd.DataFrame) -> pd.DataFrame:
    joined = metrics.merge(raw_stats, on=list(KEY_COLUMNS), how="left", indicator=True)
    unmatched = joined[joined["_merge"] != "both"].copy()
    if len(unmatched) > 0:
        preview = unmatched[list(KEY_COLUMNS)].head(10).to_dict("records")
        raise ValueError(f"Failed to match some metric rows to raw sample stats: preview={preview}")
    joined = joined.drop(columns=["_merge"])
    return joined


def filter_by_window(dataframe: pd.DataFrame, window: WindowSpec) -> pd.DataFrame:
    if window.radius is None:
        return dataframe.copy()
    mask = dataframe[ORACLE_COLUMN].between(window.lower, window.upper, inclusive="both")
    return dataframe.loc[mask].copy()


def build_motion_bins(dataframe: pd.DataFrame, motion_bin_width: float) -> list[float]:
    max_motion = float(dataframe[MOTION_COLUMN].max())
    max_edge = math.ceil(max_motion / motion_bin_width) * motion_bin_width
    bin_edges = [round(index * motion_bin_width, 6) for index in range(int(max_edge / motion_bin_width) + 1)]
    if bin_edges[-1] < max_motion:
        bin_edges.append(round(bin_edges[-1] + motion_bin_width, 6))
    return bin_edges


def summarize_window(
    dataframe: pd.DataFrame,
    window: WindowSpec,
    oracle_center: float,
    all_sample_count: int,
) -> dict[str, float | str]:
    oracle_distance = (dataframe[ORACLE_COLUMN] - oracle_center).abs()
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
        "mean_oracle_distance": float(oracle_distance.mean()),
    }


def build_motion_bin_summary(
    dataframe: pd.DataFrame,
    window: WindowSpec,
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
        )
        .reset_index()
    )
    summary = summary[summary["samples"] >= min_bin_samples].copy()
    summary["window_label"] = window.label
    summary["radius"] = math.nan if window.radius is None else float(window.radius)
    summary["motion_bin_label"] = summary["motion_bin"].astype(str)
    return summary.drop(columns=["motion_bin"])


def write_csv_outputs(joined: pd.DataFrame, window_summary: pd.DataFrame, motion_bin_summary: pd.DataFrame, output_dir: Path) -> None:
    joined.to_csv(output_dir / "motion_psnr_sample_comparison.csv", index=False)
    window_summary.to_csv(output_dir / "oracle_window_summary.csv", index=False)
    motion_bin_summary.to_csv(output_dir / "oracle_window_motion_bin_summary.csv", index=False)


def write_dashboard(window_summary: pd.DataFrame, motion_bin_summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=160)
    window_order = window_summary["window_label"].tolist()
    color_values = plt.cm.viridis_r([index / max(len(window_order) - 1, 1) for index in range(len(window_order))])
    colors = {label: color_values[index] for index, label in enumerate(window_order)}
    colors["all"] = "#4c78a8"

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
    axis.set_title("Overall mean PSNR after filtering")
    axis.set_xlabel("oracle_fmv_t_eff_mean window")
    axis.set_ylabel("mean psnr (dB)")
    axis.grid(True, axis="y", alpha=0.25)

    axis = axes[1, 1]
    axis.bar(
        window_summary["window_label"],
        window_summary["retention_ratio"] * 100.0,
        color=[colors.get(label, "#777777") for label in window_summary["window_label"]],
    )
    axis.set_title("Sample retention after filtering")
    axis.set_xlabel("oracle_fmv_t_eff_mean window")
    axis.set_ylabel("retained samples (%)")
    axis.set_ylim(0.0, 105.0)
    axis.grid(True, axis="y", alpha=0.25)

    figure.suptitle("Single-model oracle effective-time window sweep", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / "oracle_window_psnr_dashboard.png", bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    if args.only_fps <= 0:
        raise ValueError(f"only-fps must be positive: {args.only_fps}")
    if args.motion_bin_width <= 0.0:
        raise ValueError(f"motion-bin-width must be positive: {args.motion_bin_width}")
    if args.min_bin_samples <= 0:
        raise ValueError(f"min-bin-samples must be positive: {args.min_bin_samples}")

    metrics = load_metrics(args.metrics_csv)
    raw_stats = load_raw_sample_stats(
        dataset_root=args.dataset_root,
        only_fps=args.only_fps,
        difficulty_name=args.difficulty_name,
    )
    joined = build_joined_dataframe(metrics=metrics, raw_stats=raw_stats)
    motion_bin_edges = build_motion_bins(joined, args.motion_bin_width)
    windows = build_window_specs(args.oracle_center, args.window_radii)

    all_sample_count = int(len(joined))
    window_summary_rows: list[dict[str, float | str]] = []
    motion_bin_frames: list[pd.DataFrame] = []
    for window in windows:
        filtered = filter_by_window(joined, window)
        if len(filtered) == 0:
            continue
        window_summary_rows.append(
            summarize_window(
                dataframe=filtered,
                window=window,
                oracle_center=args.oracle_center,
                all_sample_count=all_sample_count,
            )
        )
        motion_bin_frames.append(
            build_motion_bin_summary(
                dataframe=filtered,
                window=window,
                motion_bin_edges=motion_bin_edges,
                min_bin_samples=args.min_bin_samples,
            )
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    window_summary = pd.DataFrame(window_summary_rows)
    motion_bin_summary = pd.concat(motion_bin_frames, ignore_index=True)
    write_csv_outputs(joined=joined, window_summary=window_summary, motion_bin_summary=motion_bin_summary, output_dir=output_dir)
    write_dashboard(window_summary=window_summary, motion_bin_summary=motion_bin_summary, output_dir=output_dir)

    print(f"metrics_csv={args.metrics_csv}")
    print(f"dataset_root={args.dataset_root}")
    print(f"output_dir={output_dir}")
    print(window_summary.to_string(index=False))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
