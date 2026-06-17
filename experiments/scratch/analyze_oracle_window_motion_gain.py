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

DEFAULT_INPUT_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\CGV_inference_outputs\cross_model_Bilinear_vs_OracleAdaptiveTimeBilinear_Minor_0611\motion_magnitude_mean\motion_psnr_sample_comparison.csv",
)
DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\oracle_window_motion_gain\bilinear_vs_oracle_adaptive_time_bilinear_minor_0611",
)
DEFAULT_ORACLE_CENTER: float = 0.5
DEFAULT_MOTION_BIN_WIDTH: float = 1.0
DEFAULT_MIN_BIN_SAMPLES: int = 20
DEFAULT_WINDOW_RADII: tuple[float, ...] = (0.025, 0.05, 0.075, 0.1)
KEY_COLUMNS: tuple[str, ...] = ("record", "major_mode_id", "minor_mode_id", "img0", "img2", "img1")
MOTION_COLUMN: str = "motion_magnitude_mean"
ORACLE_COLUMN: str = "oracle_fmv_t_eff_mean"


@dataclass(frozen=True)
class WindowSpec:
    label: str
    radius: float | None
    lower: float | None
    upper: float | None


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter cross-model comparison samples by oracle_fmv_t_eff_mean windows and compare motion-vs-improvement trends."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to motion_psnr_sample_comparison.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for threshold sweep outputs.",
    )
    parser.add_argument(
        "--delta-column",
        type=str,
        default="",
        help="Delta-PSNR column. Leave empty to auto-detect a single delta_psnr_* column.",
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
        help="Oracle window radii around oracle-center. Example: 0.025 0.05 0.075 0.1",
    )
    return parser.parse_args(argv)


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    require_existing_path(csv_path, "input CSV")
    dataframe = pd.read_csv(csv_path)
    if len(dataframe) == 0:
        raise ValueError(f"input CSV is empty: {csv_path}")
    return dataframe


def detect_delta_column(dataframe: pd.DataFrame, configured_delta_column: str) -> str:
    if configured_delta_column != "":
        if configured_delta_column not in dataframe.columns:
            raise ValueError(f"Configured delta column is missing: {configured_delta_column}")
        return configured_delta_column

    delta_columns = [column for column in dataframe.columns if column.startswith("delta_psnr_")]
    if len(delta_columns) != 1:
        raise ValueError(
            "Could not auto-detect a single delta-PSNR column. "
            f"Found columns: {delta_columns}"
        )
    return delta_columns[0]


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


def filter_by_window(dataframe: pd.DataFrame, window: WindowSpec, oracle_column: str) -> pd.DataFrame:
    if window.radius is None:
        return dataframe.copy()
    mask = dataframe[oracle_column].between(window.lower, window.upper, inclusive="both")
    return dataframe.loc[mask].copy()


def parse_delta_names(delta_column: str) -> tuple[str, str]:
    prefix = "delta_psnr_"
    suffix = "_minus_"
    if not delta_column.startswith(prefix) or suffix not in delta_column:
        return "Candidate", "Baseline"
    payload = delta_column[len(prefix):]
    candidate_name, baseline_name = payload.split(suffix, maxsplit=1)
    return candidate_name, baseline_name


def build_motion_bins(dataframe: pd.DataFrame, motion_column: str, motion_bin_width: float) -> list[float]:
    max_motion = float(dataframe[motion_column].max())
    max_edge = math.ceil(max_motion / motion_bin_width) * motion_bin_width
    bin_edges = [round(index * motion_bin_width, 6) for index in range(int(max_edge / motion_bin_width) + 1)]
    if bin_edges[-1] < max_motion:
        bin_edges.append(round(bin_edges[-1] + motion_bin_width, 6))
    return bin_edges


def summarize_window(
    dataframe: pd.DataFrame,
    window: WindowSpec,
    oracle_center: float,
    delta_column: str,
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
        "mean_delta_psnr": float(dataframe[delta_column].mean()),
        "median_delta_psnr": float(dataframe[delta_column].median()),
        "win_rate": float((dataframe[delta_column] > 0.0).mean()),
        "mean_motion_magnitude": float(dataframe[MOTION_COLUMN].mean()),
        "mean_oracle_fmv_t_eff": float(dataframe[ORACLE_COLUMN].mean()),
        "mean_oracle_distance": float(oracle_distance.mean()),
    }


def build_motion_bin_summary(
    dataframe: pd.DataFrame,
    window: WindowSpec,
    delta_column: str,
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
            samples=(delta_column, "size"),
            motion_mean=(MOTION_COLUMN, "mean"),
            delta_psnr_mean=(delta_column, "mean"),
            delta_psnr_median=(delta_column, "median"),
            win_rate=(delta_column, lambda values: float((values > 0.0).mean())),
            oracle_fmv_t_eff_mean=(ORACLE_COLUMN, "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["samples"] >= min_bin_samples].copy()
    summary["window_label"] = window.label
    summary["radius"] = math.nan if window.radius is None else float(window.radius)
    summary["motion_bin_label"] = summary["motion_bin"].astype(str)
    return summary.drop(columns=["motion_bin"])


def write_csv_outputs(
    window_summary: pd.DataFrame,
    motion_bin_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    window_summary.to_csv(output_dir / "oracle_window_summary.csv", index=False)
    motion_bin_summary.to_csv(output_dir / "oracle_window_motion_bin_summary.csv", index=False)


def write_dashboard(
    window_summary: pd.DataFrame,
    motion_bin_summary: pd.DataFrame,
    candidate_name: str,
    baseline_name: str,
    output_dir: Path,
) -> None:
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
        axis.plot(
            line["motion_mean"],
            line["delta_psnr_mean"],
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label=f"{window_label} (n={int(window_summary.loc[window_summary['window_label'] == window_label, 'samples'].iloc[0])})",
            color=colors.get(window_label),
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_title(f"Mean delta PSNR by motion bin\n{candidate_name} - {baseline_name}")
    axis.set_xlabel(MOTION_COLUMN)
    axis.set_ylabel("mean delta psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    for window_label in window_order:
        line = motion_bin_summary[motion_bin_summary["window_label"] == window_label]
        if len(line) == 0:
            continue
        axis.plot(
            line["motion_mean"],
            line["win_rate"],
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label=window_label,
            color=colors.get(window_label),
        )
    axis.set_title("Win rate by motion bin")
    axis.set_xlabel(MOTION_COLUMN)
    axis.set_ylabel("fraction with delta_psnr > 0")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    axis.bar(
        window_summary["window_label"],
        window_summary["mean_delta_psnr"],
        color=[colors.get(label, "#777777") for label in window_summary["window_label"]],
    )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_title("Overall mean delta PSNR after filtering")
    axis.set_xlabel("oracle_fmv_t_eff_mean window")
    axis.set_ylabel("mean delta psnr (dB)")
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

    figure.suptitle("Oracle effective-time window sweep", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / "oracle_window_motion_gain_dashboard.png", bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    if args.motion_bin_width <= 0.0:
        raise ValueError(f"motion-bin-width must be positive: {args.motion_bin_width}")
    if args.min_bin_samples <= 0:
        raise ValueError(f"min-bin-samples must be positive: {args.min_bin_samples}")

    dataframe = load_dataframe(args.input_csv)
    delta_column = detect_delta_column(dataframe, args.delta_column)
    require_columns(
        dataframe=dataframe,
        columns=[*KEY_COLUMNS, MOTION_COLUMN, ORACLE_COLUMN, delta_column],
        label="input CSV",
    )
    working = dataframe[[*KEY_COLUMNS, MOTION_COLUMN, ORACLE_COLUMN, delta_column]].copy()

    candidate_name, baseline_name = parse_delta_names(delta_column)
    motion_bin_edges = build_motion_bins(working, MOTION_COLUMN, args.motion_bin_width)
    windows = build_window_specs(args.oracle_center, args.window_radii)

    all_sample_count = int(len(working))
    window_summary_rows: list[dict[str, float | str]] = []
    motion_bin_frames: list[pd.DataFrame] = []
    for window in windows:
        filtered = filter_by_window(working, window, ORACLE_COLUMN)
        if len(filtered) == 0:
            continue
        window_summary_rows.append(
            summarize_window(
                dataframe=filtered,
                window=window,
                oracle_center=args.oracle_center,
                delta_column=delta_column,
                all_sample_count=all_sample_count,
            )
        )
        motion_bin_frames.append(
            build_motion_bin_summary(
                dataframe=filtered,
                window=window,
                delta_column=delta_column,
                motion_bin_edges=motion_bin_edges,
                min_bin_samples=args.min_bin_samples,
            )
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    window_summary = pd.DataFrame(window_summary_rows)
    motion_bin_summary = pd.concat(motion_bin_frames, ignore_index=True)
    write_csv_outputs(window_summary=window_summary, motion_bin_summary=motion_bin_summary, output_dir=output_dir)
    write_dashboard(
        window_summary=window_summary,
        motion_bin_summary=motion_bin_summary,
        candidate_name=candidate_name,
        baseline_name=baseline_name,
        output_dir=output_dir,
    )

    print(f"input_csv={args.input_csv}")
    print(f"delta_column={delta_column}")
    print(f"output_dir={output_dir}")
    print(window_summary.to_string(index=False))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
