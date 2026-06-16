from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

KEY_COLUMNS: tuple[str, ...] = (
    "record",
    "major_mode_id",
    "minor_mode_id",
    "img0",
    "img2",
    "img1",
)
FEATURE_COLUMNS: tuple[str, ...] = (
    "motion_magnitude_mean",
    "motion_magnitude_p95",
    "motion_magnitude_max",
    "oracle_fmv_t_eff_mean",
    "oracle_bmv_t_eff_mean",
    "oracle_fmv_scale_mean",
    "oracle_bmv_scale_mean",
    "oracle_t_eff_gap_mean",
    "oracle_fmv_valid_ratio",
    "oracle_bmv_valid_ratio",
    "oracle_fmv_clamped_ratio",
    "oracle_bmv_clamped_ratio",
)
DEFAULT_SAMPLE_COMPARISON_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\CGV_inference_outputs\cross_model_MaskedArea_vs_AdaptiveTimeMaskedArea_Minor_0611\motion_magnitude_mean\motion_psnr_sample_comparison.csv",
)
DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\motion_psnr_buckets\cross_model_MaskedArea_vs_AdaptiveTimeMaskedArea_Minor_0611",
)
DEFAULT_MOTION_COLUMN: str = "motion_magnitude_mean"
DEFAULT_MOTION_QUANTILE: float = 0.90
DEFAULT_PSNR_QUANTILE: float = 0.15
DEFAULT_PSNR_AGGREGATION: str = "mean"
PSNR_AGGREGATIONS: tuple[str, ...] = ("mean", "min", "max")
BUCKET_ORDER: tuple[str, ...] = (
    "large_motion_artifacts",
    "not_large_motion_high_psnr",
    "other_artifacts",
    "large_motion_high_psnr",
)
BUCKET_COLORS: dict[str, str] = {
    "large_motion_artifacts": "#d62728",
    "not_large_motion_high_psnr": "#2ca02c",
    "other_artifacts": "#ff7f0e",
    "large_motion_high_psnr": "#1f77b4",
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split motion-vs-PSNR sample comparison rows into interpretable buckets."
    )
    parser.add_argument(
        "--sample-comparison-csv",
        type=Path,
        default=DEFAULT_SAMPLE_COMPARISON_CSV,
        help="Path to motion_psnr_sample_comparison.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for bucket CSVs and plots.",
    )
    parser.add_argument(
        "--motion-column",
        type=str,
        default=DEFAULT_MOTION_COLUMN,
        help="Motion feature column used for the x-axis split.",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=None,
        help="Explicit large-motion threshold. If omitted, use motion-threshold-quantile.",
    )
    parser.add_argument(
        "--motion-threshold-quantile",
        type=float,
        default=DEFAULT_MOTION_QUANTILE,
        help="Quantile used when motion-threshold is omitted.",
    )
    parser.add_argument(
        "--psnr-threshold",
        type=float,
        default=None,
        help="Explicit low-PSNR threshold. If omitted, use psnr-threshold-quantile.",
    )
    parser.add_argument(
        "--psnr-threshold-quantile",
        type=float,
        default=DEFAULT_PSNR_QUANTILE,
        help="Quantile used when psnr-threshold is omitted.",
    )
    parser.add_argument(
        "--psnr-aggregation",
        type=str,
        choices=PSNR_AGGREGATIONS,
        default=DEFAULT_PSNR_AGGREGATION,
        help="How to aggregate multiple experiment PSNR columns into one sample PSNR score.",
    )
    parser.add_argument(
        "--psnr-column",
        action="append",
        default=None,
        help="Explicit PSNR experiment column. Repeat to use multiple columns. If omitted, auto-detect.",
    )
    return parser.parse_args(argv)


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_quantile(value: float, label: str) -> None:
    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{label} must be in (0, 1), got {value}")


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    require_existing_path(csv_path, "sample comparison CSV")
    dataframe = pd.read_csv(csv_path)
    if len(dataframe) == 0:
        raise ValueError(f"Sample comparison CSV is empty: {csv_path}")
    return dataframe


def detect_experiment_columns(dataframe: pd.DataFrame) -> list[str]:
    excluded_columns = set(KEY_COLUMNS) | set(FEATURE_COLUMNS)
    experiment_columns: list[str] = []
    for column in dataframe.columns:
        if column in excluded_columns:
            continue
        if column.startswith("delta_psnr_"):
            continue
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            experiment_columns.append(column)

    if len(experiment_columns) == 0:
        raise ValueError("No experiment PSNR columns were detected in sample comparison CSV.")
    return experiment_columns


def resolve_experiment_columns(dataframe: pd.DataFrame, requested_columns: list[str] | None) -> list[str]:
    if requested_columns is None or len(requested_columns) == 0:
        return detect_experiment_columns(dataframe)

    missing_columns = [column for column in requested_columns if column not in dataframe.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"Requested PSNR columns are missing: {missing_columns}")
    return requested_columns


def build_psnr_summary_column(
    dataframe: pd.DataFrame,
    experiment_columns: list[str],
    aggregation: str,
) -> pd.Series:
    experiment_frame = dataframe[experiment_columns]
    if aggregation == "mean":
        return experiment_frame.mean(axis=1)
    if aggregation == "min":
        return experiment_frame.min(axis=1)
    if aggregation == "max":
        return experiment_frame.max(axis=1)
    raise ValueError(f"Unsupported psnr aggregation: {aggregation}")


def resolve_motion_threshold(dataframe: pd.DataFrame, motion_column: str, threshold: float | None, quantile: float) -> float:
    if motion_column not in dataframe.columns:
        raise ValueError(f"motion column was not found: {motion_column}")
    if threshold is not None:
        return float(threshold)

    require_quantile(quantile, "motion-threshold-quantile")
    return float(dataframe[motion_column].quantile(quantile))


def resolve_psnr_threshold(psnr_summary: pd.Series, threshold: float | None, quantile: float) -> float:
    if threshold is not None:
        return float(threshold)

    require_quantile(quantile, "psnr-threshold-quantile")
    return float(psnr_summary.quantile(quantile))


def classify_bucket(motion_value: float, psnr_value: float, motion_threshold: float, psnr_threshold: float) -> str:
    if motion_value >= motion_threshold and psnr_value < psnr_threshold:
        return "large_motion_artifacts"
    if motion_value < motion_threshold and psnr_value >= psnr_threshold:
        return "not_large_motion_high_psnr"
    if motion_value < motion_threshold and psnr_value < psnr_threshold:
        return "other_artifacts"
    return "large_motion_high_psnr"


def build_bucketed_dataframe(
    dataframe: pd.DataFrame,
    motion_column: str,
    motion_threshold: float,
    psnr_threshold: float,
    psnr_summary: pd.Series,
    experiment_columns: list[str],
    aggregation: str,
) -> pd.DataFrame:
    bucketed = dataframe.copy()
    bucketed["psnr_summary"] = psnr_summary.astype(float)
    bucketed["psnr_summary_source"] = aggregation
    bucketed["psnr_experiment_columns"] = ",".join(experiment_columns)
    bucketed["motion_threshold"] = float(motion_threshold)
    bucketed["psnr_threshold"] = float(psnr_threshold)
    bucketed["bucket"] = [
        classify_bucket(
            motion_value=float(motion_value),
            psnr_value=float(psnr_value),
            motion_threshold=motion_threshold,
            psnr_threshold=psnr_threshold,
        )
        for motion_value, psnr_value in zip(bucketed[motion_column], bucketed["psnr_summary"])
    ]
    bucketed["bucket"] = pd.Categorical(bucketed["bucket"], categories=BUCKET_ORDER, ordered=True)
    return bucketed.sort_values(["bucket", motion_column, "psnr_summary"], ascending=[True, False, True]).reset_index(drop=True)


def build_bucket_summary(dataframe: pd.DataFrame, motion_column: str) -> pd.DataFrame:
    summary = (
        dataframe.groupby("bucket", observed=True)
        .agg(
            samples=("bucket", "size"),
            motion_min=(motion_column, "min"),
            motion_mean=(motion_column, "mean"),
            motion_max=(motion_column, "max"),
            psnr_min=("psnr_summary", "min"),
            psnr_mean=("psnr_summary", "mean"),
            psnr_max=("psnr_summary", "max"),
        )
        .reset_index()
    )
    summary["ratio"] = summary["samples"] / len(dataframe)
    return summary


def write_bucket_csvs(dataframe: pd.DataFrame, output_dir: Path) -> None:
    dataframe.to_csv(output_dir / "bucketed_samples.csv", index=False)
    for bucket_name in BUCKET_ORDER:
        bucket_dataframe = dataframe[dataframe["bucket"] == bucket_name].copy()
        bucket_dataframe.to_csv(output_dir / f"{bucket_name}.csv", index=False)


def write_bucket_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    summary.to_csv(output_dir / "bucket_summary.csv", index=False)


def write_bucket_scatter(
    dataframe: pd.DataFrame,
    motion_column: str,
    motion_threshold: float,
    psnr_threshold: float,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(8, 5), dpi=160)
    for bucket_name in BUCKET_ORDER:
        bucket_dataframe = dataframe[dataframe["bucket"] == bucket_name]
        if len(bucket_dataframe) == 0:
            continue
        plt.scatter(
            bucket_dataframe[motion_column],
            bucket_dataframe["psnr_summary"],
            s=12,
            alpha=0.45,
            edgecolors="none",
            label=f"{bucket_name} (n={len(bucket_dataframe)})",
            color=BUCKET_COLORS[bucket_name],
        )

    plt.axvline(
        motion_threshold,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"motion threshold = {motion_threshold:.2f}",
    )
    plt.axhline(
        psnr_threshold,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=f"psnr threshold = {psnr_threshold:.2f}",
    )
    plt.xlabel(motion_column)
    plt.ylabel("psnr_summary")
    plt.title(
        "Motion vs PSNR bucket split\n"
        f"{motion_column} >= {motion_threshold:.2f}, psnr_summary < {psnr_threshold:.2f}"
    )
    plt.grid(True, alpha=0.25)
    plt.legend(markerscale=1.5)
    plt.tight_layout()
    plt.savefig(output_dir / "bucket_scatter.png")
    plt.close()


def write_bucket_distribution_dashboard(
    dataframe: pd.DataFrame,
    motion_column: str,
    motion_threshold: float,
    psnr_threshold: float,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9), dpi=160)
    motion_axis = axes[0, 0]
    scatter_axis = axes[0, 1]
    psnr_axis = axes[1, 0]
    coverage_axis = axes[1, 1]

    motion_bins = 40
    psnr_bins = 40
    for bucket_name in BUCKET_ORDER:
        bucket_dataframe = dataframe[dataframe["bucket"] == bucket_name]
        if len(bucket_dataframe) == 0:
            continue
        label = f"{bucket_name} (n={len(bucket_dataframe)})"
        color = BUCKET_COLORS[bucket_name]
        motion_axis.hist(
            bucket_dataframe[motion_column],
            bins=motion_bins,
            alpha=0.38,
            color=color,
            label=label,
        )
        scatter_axis.scatter(
            bucket_dataframe[motion_column],
            bucket_dataframe["psnr_summary"],
            s=12,
            alpha=0.4,
            edgecolors="none",
            color=color,
            label=label,
        )
        psnr_axis.hist(
            bucket_dataframe["psnr_summary"],
            bins=psnr_bins,
            alpha=0.38,
            color=color,
            label=label,
        )

    motion_axis.axvline(
        motion_threshold,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=f"motion threshold = {motion_threshold:.2f}",
    )
    motion_axis.set_xlabel(motion_column)
    motion_axis.set_ylabel("samples")
    motion_axis.set_title("Per-sample motion distribution")
    motion_axis.grid(True, axis="y", alpha=0.25)
    motion_axis.legend(fontsize=8)

    scatter_axis.axvline(
        motion_threshold,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=f"motion threshold = {motion_threshold:.2f}",
    )
    scatter_axis.axhline(
        psnr_threshold,
        color="gray",
        linestyle="--",
        linewidth=1.1,
        label=f"psnr threshold = {psnr_threshold:.2f}",
    )
    scatter_axis.set_xlabel(motion_column)
    scatter_axis.set_ylabel("psnr_summary")
    scatter_axis.set_title("Per-sample motion vs PSNR")
    scatter_axis.grid(True, alpha=0.25)
    scatter_axis.legend(fontsize=8, markerscale=1.3)

    psnr_axis.axvline(
        psnr_threshold,
        color="gray",
        linestyle="--",
        linewidth=1.1,
        label=f"psnr threshold = {psnr_threshold:.2f}",
    )
    psnr_axis.set_xlabel("psnr_summary")
    psnr_axis.set_ylabel("samples")
    psnr_axis.set_title("Per-sample PSNR distribution")
    psnr_axis.grid(True, axis="y", alpha=0.25)
    psnr_axis.legend(fontsize=8)

    coverage_summary = build_bucket_summary(dataframe, motion_column)
    coverage_summary = coverage_summary.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    coverage_labels = [
        "large motion\nartifacts",
        "not large motion\nhigh psnr",
        "other\nartifacts",
        "large motion\nhigh psnr",
    ]
    coverage_axis.barh(
        coverage_labels,
        coverage_summary["ratio"] * 100.0,
        color=[BUCKET_COLORS[bucket_name] for bucket_name in coverage_summary["bucket"]],
    )
    for index, row in coverage_summary.iterrows():
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

    figure.suptitle("Motion/PSNR raw sample distribution by artifact bucket", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / "bucket_distribution_dashboard.png", bbox_inches="tight")
    plt.close(figure)


def write_thresholds_file(
    motion_column: str,
    motion_threshold: float,
    psnr_threshold: float,
    aggregation: str,
    experiment_columns: list[str],
    output_dir: Path,
) -> None:
    rows = [
        {
            "motion_column": motion_column,
            "motion_threshold": motion_threshold,
            "psnr_threshold": psnr_threshold,
            "psnr_summary_source": aggregation,
            "psnr_experiment_columns": ",".join(experiment_columns),
        }
    ]
    pd.DataFrame(rows).to_csv(output_dir / "thresholds.csv", index=False)


def run(args: argparse.Namespace) -> None:
    dataframe = load_dataframe(args.sample_comparison_csv)
    experiment_columns = resolve_experiment_columns(dataframe, args.psnr_column)
    psnr_summary = build_psnr_summary_column(dataframe, experiment_columns, args.psnr_aggregation)
    motion_threshold = resolve_motion_threshold(
        dataframe=dataframe,
        motion_column=args.motion_column,
        threshold=args.motion_threshold,
        quantile=args.motion_threshold_quantile,
    )
    psnr_threshold = resolve_psnr_threshold(
        psnr_summary=psnr_summary,
        threshold=args.psnr_threshold,
        quantile=args.psnr_threshold_quantile,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    bucketed = build_bucketed_dataframe(
        dataframe=dataframe,
        motion_column=args.motion_column,
        motion_threshold=motion_threshold,
        psnr_threshold=psnr_threshold,
        psnr_summary=psnr_summary,
        experiment_columns=experiment_columns,
        aggregation=args.psnr_aggregation,
    )
    summary = build_bucket_summary(bucketed, args.motion_column)
    write_bucket_csvs(bucketed, output_dir)
    write_bucket_summary(summary, output_dir)
    write_thresholds_file(
        motion_column=args.motion_column,
        motion_threshold=motion_threshold,
        psnr_threshold=psnr_threshold,
        aggregation=args.psnr_aggregation,
        experiment_columns=experiment_columns,
        output_dir=output_dir,
    )
    write_bucket_scatter(
        dataframe=bucketed,
        motion_column=args.motion_column,
        motion_threshold=motion_threshold,
        psnr_threshold=psnr_threshold,
        output_dir=output_dir,
    )
    write_bucket_distribution_dashboard(
        dataframe=bucketed,
        motion_column=args.motion_column,
        motion_threshold=motion_threshold,
        psnr_threshold=psnr_threshold,
        output_dir=output_dir,
    )

    print(f"input={args.sample_comparison_csv}")
    print(f"output_dir={output_dir}")
    print(f"motion_column={args.motion_column}")
    print(f"motion_threshold={motion_threshold:.6f}")
    print(f"psnr_threshold={psnr_threshold:.6f}")
    print(f"psnr_summary_source={args.psnr_aggregation}")
    print(f"psnr_experiment_columns={experiment_columns}")
    print(summary.to_string(index=False))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(argv=sys.argv[1:])
