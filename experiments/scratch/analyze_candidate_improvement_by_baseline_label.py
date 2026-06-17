from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from experiments.scratch.split_motion_psnr_buckets import BUCKET_COLORS
from experiments.scratch.split_motion_psnr_buckets import BUCKET_ORDER
from experiments.scratch.split_motion_psnr_buckets import KEY_COLUMNS

DEFAULT_BASELINE_LABELED_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\baseline_finetuning_motion_psnr\buckets\bucketed_samples.csv",
)
DEFAULT_BASELINE_JOINED_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\baseline_finetuning_motion_psnr\motion_psnr_joined.csv",
)
DEFAULT_CANDIDATE_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\DGX_inference_outputs\IFRNet_Residual_FlowApprox_1_Layer_TwoStage_Splat_4Direction_MaskedArea_0611\motion_psnr_analysis\motion_psnr_joined.csv",
)
DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\candidate_improvement_by_baseline_label\masked_area_0611",
)
DEFAULT_ORACLE_THRESHOLDS_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\baseline_finetuning_oracle_t_eff_psnr\artifact_overlap\thresholds.csv",
)
DEFAULT_BASELINE_PSNR_COLUMN: str = "psnr_summary"
DEFAULT_BASELINE_LABEL_COLUMN: str = "bucket"
DEFAULT_CANDIDATE_PSNR_COLUMN: str = "psnr"
DEFAULT_CANDIDATE_NAME: str = "Masked Area"
MOTION_COLUMN: str = "motion_magnitude_mean"
CANDIDATE_MOTION_COLUMN: str = "candidate_motion_magnitude_mean"
ORACLE_FMV_T_EFF_COLUMN: str = "oracle_fmv_t_eff_mean"
ORACLE_BMV_T_EFF_COLUMN: str = "oracle_bmv_t_eff_mean"
ORACLE_T_EFF_GAP_COLUMN: str = "oracle_t_eff_gap_mean"
BASELINE_EXTRA_COLUMNS: tuple[str, ...] = (
    MOTION_COLUMN,
    "motion_magnitude_p95",
    "motion_magnitude_max",
    ORACLE_FMV_T_EFF_COLUMN,
    ORACLE_BMV_T_EFF_COLUMN,
    "oracle_fmv_scale_mean",
    "oracle_bmv_scale_mean",
    ORACLE_T_EFF_GAP_COLUMN,
    "oracle_fmv_valid_ratio",
    "oracle_bmv_valid_ratio",
    "oracle_fmv_clamped_ratio",
    "oracle_bmv_clamped_ratio",
    "motion_threshold",
    "psnr_threshold",
)
DISPLAY_LABELS: dict[str, str] = {
    "large_motion_artifacts": "large motion\nartifacts",
    "not_large_motion_high_psnr": "not large motion\nhigh psnr",
    "other_artifacts": "other\nartifacts",
    "large_motion_high_psnr": "large motion\nhigh psnr",
}
BASELINE_METRIC_RENAME_MAP: dict[str, str] = {
    "ssim": "baseline_ssim",
    "lpips": "baseline_lpips",
}
CANDIDATE_METRIC_RENAME_MAP: dict[str, str] = {
    "ssim": "candidate_ssim",
    "lpips": "candidate_lpips",
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize candidate-model improvement by baseline-defined artifact labels."
    )
    parser.add_argument(
        "--baseline-labeled-csv",
        type=Path,
        default=DEFAULT_BASELINE_LABELED_CSV,
        help="Path to baseline bucketed_samples.csv.",
    )
    parser.add_argument(
        "--baseline-joined-csv",
        type=Path,
        default=DEFAULT_BASELINE_JOINED_CSV,
        help="Path to baseline motion_psnr_joined.csv for optional SSIM/LPIPS columns.",
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=DEFAULT_CANDIDATE_CSV,
        help="Path to candidate motion_psnr_joined.csv or compatible CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for merged CSV, summary CSV, and plots.",
    )
    parser.add_argument(
        "--oracle-thresholds-csv",
        type=Path,
        default=DEFAULT_ORACLE_THRESHOLDS_CSV,
        help="Path to baseline oracle threshold CSV.",
    )
    parser.add_argument(
        "--baseline-psnr-column",
        type=str,
        default=DEFAULT_BASELINE_PSNR_COLUMN,
        help="Baseline PSNR column in baseline-labeled CSV.",
    )
    parser.add_argument(
        "--baseline-label-column",
        type=str,
        default=DEFAULT_BASELINE_LABEL_COLUMN,
        help="Baseline label column in baseline-labeled CSV.",
    )
    parser.add_argument(
        "--candidate-psnr-column",
        type=str,
        default=DEFAULT_CANDIDATE_PSNR_COLUMN,
        help="Candidate PSNR column in candidate CSV.",
    )
    parser.add_argument(
        "--candidate-name",
        type=str,
        default=DEFAULT_CANDIDATE_NAME,
        help="Display name for candidate model.",
    )
    return parser.parse_args(argv)


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def load_csv(csv_path: Path, label: str) -> pd.DataFrame:
    require_existing_path(csv_path, label)
    dataframe = pd.read_csv(csv_path)
    if len(dataframe) == 0:
        raise ValueError(f"{label} is empty: {csv_path}")
    return dataframe


def require_columns(dataframe: pd.DataFrame, required_columns: Sequence[str], label: str) -> None:
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"{label} is missing columns: {missing_columns}")


def build_baseline_frame(
    dataframe: pd.DataFrame,
    baseline_psnr_column: str,
    baseline_label_column: str,
) -> pd.DataFrame:
    require_columns(
        dataframe=dataframe,
        required_columns=[*KEY_COLUMNS, baseline_psnr_column, baseline_label_column, *BASELINE_EXTRA_COLUMNS],
        label="baseline labeled CSV",
    )
    baseline = dataframe[[*KEY_COLUMNS, baseline_psnr_column, baseline_label_column, *BASELINE_EXTRA_COLUMNS]].copy()
    baseline = baseline.rename(
        columns={
            baseline_psnr_column: "baseline_psnr",
            baseline_label_column: "baseline_label",
        }
    )
    duplicate_mask = baseline.duplicated(subset=list(KEY_COLUMNS), keep=False)
    if bool(duplicate_mask.any()):
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(f"baseline labeled CSV has duplicate key rows: {duplicate_count}")
    return baseline


def build_metric_sidecar_frame(
    dataframe: pd.DataFrame,
    rename_map: dict[str, str],
    label: str,
) -> pd.DataFrame:
    require_columns(dataframe=dataframe, required_columns=KEY_COLUMNS, label=label)
    available_columns = [column for column in rename_map if column in dataframe.columns]
    if len(available_columns) == 0:
        return pd.DataFrame(columns=[*KEY_COLUMNS, *rename_map.values()])

    metric_frame = dataframe[[*KEY_COLUMNS, *available_columns]].copy()
    duplicate_mask = metric_frame.duplicated(subset=list(KEY_COLUMNS), keep=False)
    if bool(duplicate_mask.any()):
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(f"{label} has duplicate key rows: {duplicate_count}")
    return metric_frame.rename(columns={column: rename_map[column] for column in available_columns})


def merge_baseline_metric_sidecar(
    baseline: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
) -> pd.DataFrame:
    merged = baseline.merge(baseline_metrics, on=list(KEY_COLUMNS), how="left")
    for metric_column in BASELINE_METRIC_RENAME_MAP.values():
        if metric_column not in merged.columns:
            merged[metric_column] = pd.NA
    return merged


def extract_thresholds(dataframe: pd.DataFrame) -> tuple[float | None, float | None]:
    motion_threshold = None
    psnr_threshold = None
    if "motion_threshold" in dataframe.columns:
        motion_values = dataframe["motion_threshold"].dropna().unique().tolist()
        if len(motion_values) == 1:
            motion_threshold = float(motion_values[0])
    if "psnr_threshold" in dataframe.columns:
        psnr_values = dataframe["psnr_threshold"].dropna().unique().tolist()
        if len(psnr_values) == 1:
            psnr_threshold = float(psnr_values[0])
    return motion_threshold, psnr_threshold


def load_oracle_thresholds(csv_path: Path) -> tuple[float, float]:
    require_existing_path(csv_path, "oracle thresholds CSV")
    dataframe = pd.read_csv(csv_path)
    required_columns = ("oracle_left_threshold", "oracle_right_threshold")
    require_columns(dataframe=dataframe, required_columns=required_columns, label="oracle thresholds CSV")
    if len(dataframe) != 1:
        raise ValueError(f"oracle thresholds CSV must have exactly one row: {csv_path}")
    return float(dataframe["oracle_left_threshold"].iloc[0]), float(dataframe["oracle_right_threshold"].iloc[0])


def build_candidate_frame(dataframe: pd.DataFrame, candidate_psnr_column: str) -> pd.DataFrame:
    require_columns(
        dataframe=dataframe,
        required_columns=[*KEY_COLUMNS, candidate_psnr_column, MOTION_COLUMN],
        label="candidate CSV",
    )
    available_metric_columns = [column for column in CANDIDATE_METRIC_RENAME_MAP if column in dataframe.columns]
    candidate = dataframe[[*KEY_COLUMNS, candidate_psnr_column, MOTION_COLUMN, *available_metric_columns]].copy()
    candidate = candidate.rename(
        columns={
            candidate_psnr_column: "candidate_psnr",
            MOTION_COLUMN: CANDIDATE_MOTION_COLUMN,
            **{column: CANDIDATE_METRIC_RENAME_MAP[column] for column in available_metric_columns},
        }
    )
    duplicate_mask = candidate.duplicated(subset=list(KEY_COLUMNS), keep=False)
    if bool(duplicate_mask.any()):
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(f"candidate CSV has duplicate key rows: {duplicate_count}")
    return candidate


def merge_baseline_and_candidate(baseline: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = baseline.merge(candidate, on=list(KEY_COLUMNS), how="left", indicator=True)
    unmatched = merged[merged["_merge"] != "both"].copy()
    matched = merged[merged["_merge"] == "both"].copy()
    if len(matched) == 0:
        raise ValueError("candidate CSV does not overlap with baseline-labeled samples.")

    matched = matched.drop(columns=["_merge"])
    matched["delta_psnr"] = matched["candidate_psnr"] - matched["baseline_psnr"]
    matched["candidate_better"] = matched["delta_psnr"] > 0.0
    matched["candidate_much_better"] = matched["delta_psnr"] > 0.5
    matched["candidate_much_worse"] = matched["delta_psnr"] < -0.5
    matched["baseline_label"] = pd.Categorical(matched["baseline_label"], categories=BUCKET_ORDER, ordered=True)
    unmatched = unmatched.drop(columns=["candidate_psnr", "_merge"], errors="ignore")
    return (
        matched.sort_values(["baseline_label", "delta_psnr"], ascending=[True, False]).reset_index(drop=True),
        unmatched.reset_index(drop=True),
    )


def build_label_summary(merged: pd.DataFrame) -> pd.DataFrame:
    summary = (
        merged.groupby("baseline_label", observed=True)
        .agg(
            samples=("baseline_label", "size"),
            baseline_psnr_mean=("baseline_psnr", "mean"),
            candidate_psnr_mean=("candidate_psnr", "mean"),
            delta_psnr_mean=("delta_psnr", "mean"),
            delta_psnr_median=("delta_psnr", "median"),
            delta_psnr_min=("delta_psnr", "min"),
            delta_psnr_max=("delta_psnr", "max"),
            win_rate=("candidate_better", "mean"),
            much_better_rate=("candidate_much_better", "mean"),
            much_worse_rate=("candidate_much_worse", "mean"),
            oracle_fmv_t_eff_mean=(ORACLE_FMV_T_EFF_COLUMN, "mean"),
            oracle_bmv_t_eff_mean=(ORACLE_BMV_T_EFF_COLUMN, "mean"),
            oracle_t_eff_gap_mean=(ORACLE_T_EFF_GAP_COLUMN, "mean"),
        )
        .reset_index()
    )
    summary["display_label"] = [DISPLAY_LABELS[str(label)] for label in summary["baseline_label"]]
    return summary


def build_improvement_flags(
    merged: pd.DataFrame,
    oracle_left_threshold: float,
    oracle_right_threshold: float,
    psnr_threshold: float | None,
) -> pd.DataFrame:
    oracle_fmv_label = (
        (merged[ORACLE_FMV_T_EFF_COLUMN] <= oracle_left_threshold)
        | (merged[ORACLE_FMV_T_EFF_COLUMN] >= oracle_right_threshold)
    )
    flags = merged[
        [
            *KEY_COLUMNS,
            "baseline_label",
            "candidate_better",
            "delta_psnr",
            "baseline_psnr",
            "candidate_psnr",
            *[
                column
                for column in (
                    "baseline_ssim",
                    "candidate_ssim",
                    "baseline_lpips",
                    "candidate_lpips",
                )
                if column in merged.columns
            ],
            MOTION_COLUMN,
            ORACLE_FMV_T_EFF_COLUMN,
            ORACLE_BMV_T_EFF_COLUMN,
            ORACLE_T_EFF_GAP_COLUMN,
        ]
    ].copy()
    if "baseline_ssim" in flags.columns and "candidate_ssim" in flags.columns:
        flags["delta_ssim"] = flags["candidate_ssim"] - flags["baseline_ssim"]
    if "baseline_lpips" in flags.columns and "candidate_lpips" in flags.columns:
        flags["delta_lpips"] = flags["candidate_lpips"] - flags["baseline_lpips"]
    flags["baseline_motion_label"] = flags["baseline_label"]
    flags["baseline_oracle_fmv_t_eff_label"] = oracle_fmv_label.astype(bool)
    if psnr_threshold is None:
        flags["baseline_is_low_psnr"] = False
    else:
        flags["baseline_is_low_psnr"] = (flags["baseline_psnr"] < psnr_threshold).astype(bool)
    flags["baseline_oracle_t_eff_artifact"] = (
        flags["baseline_oracle_fmv_t_eff_label"] & flags["baseline_is_low_psnr"]
    )
    flags = flags.rename(columns={"candidate_better": "improve"})
    ordered_columns = [
        *KEY_COLUMNS,
        "baseline_motion_label",
        "improve",
        "baseline_oracle_fmv_t_eff_label",
        "baseline_is_low_psnr",
        "baseline_oracle_t_eff_artifact",
        "delta_psnr",
        "baseline_psnr",
        "candidate_psnr",
        *[
            column
            for column in (
                "baseline_ssim",
                "candidate_ssim",
                "delta_ssim",
                "baseline_lpips",
                "candidate_lpips",
                "delta_lpips",
            )
            if column in flags.columns
        ],
        MOTION_COLUMN,
        ORACLE_FMV_T_EFF_COLUMN,
        ORACLE_BMV_T_EFF_COLUMN,
        ORACLE_T_EFF_GAP_COLUMN,
        "baseline_label",
    ]
    return flags[ordered_columns].copy()


def write_csv_outputs(merged: pd.DataFrame, unmatched: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    merged.to_csv(output_dir / "candidate_vs_baseline_labeled_samples.csv", index=False)
    unmatched.to_csv(output_dir / "unmatched_baseline_samples.csv", index=False)
    summary.to_csv(output_dir / "candidate_vs_baseline_label_summary.csv", index=False)


def write_improvement_flags_csv(improvement_flags: pd.DataFrame, output_dir: Path) -> None:
    improvement_flags.to_csv(output_dir / "baseline_label_improvement_flags.csv", index=False)


def write_plot(merged: pd.DataFrame, summary: pd.DataFrame, candidate_name: str, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160)
    mean_delta_axis = axes[0, 0]
    win_rate_axis = axes[0, 1]
    boxplot_axis = axes[1, 0]
    count_axis = axes[1, 1]

    x_positions = list(range(len(summary)))
    colors = [BUCKET_COLORS[str(label)] for label in summary["baseline_label"]]

    mean_delta_axis.bar(x_positions, summary["delta_psnr_mean"], color=colors, alpha=0.9)
    mean_delta_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    mean_delta_axis.set_xticks(x_positions, summary["display_label"])
    mean_delta_axis.set_ylabel("mean delta psnr (dB)")
    mean_delta_axis.set_title(f"{candidate_name} - baseline mean delta by baseline label")
    mean_delta_axis.grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(summary["delta_psnr_mean"]):
        offset = 0.03 if value >= 0.0 else -0.05
        vertical_alignment = "bottom" if value >= 0.0 else "top"
        mean_delta_axis.text(index, float(value) + offset, f"{float(value):.2f}", ha="center", va=vertical_alignment, fontsize=9)

    win_rate_axis.bar(x_positions, summary["win_rate"] * 100.0, color=colors, alpha=0.9)
    win_rate_axis.axhline(50.0, color="black", linestyle="--", linewidth=1.0)
    win_rate_axis.set_xticks(x_positions, summary["display_label"])
    win_rate_axis.set_ylabel("candidate better rate (%)")
    win_rate_axis.set_title(f"{candidate_name} win rate by baseline label")
    win_rate_axis.grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(summary["win_rate"] * 100.0):
        win_rate_axis.text(index, float(value) + 1.0, f"{float(value):.1f}%", ha="center", va="bottom", fontsize=9)

    boxplot_data = [
        merged.loc[merged["baseline_label"] == bucket_name, "delta_psnr"].tolist()
        for bucket_name in BUCKET_ORDER
    ]
    boxplot = boxplot_axis.boxplot(
        boxplot_data,
        patch_artist=True,
        labels=[DISPLAY_LABELS[bucket_name] for bucket_name in BUCKET_ORDER],
    )
    for patch, bucket_name in zip(boxplot["boxes"], BUCKET_ORDER):
        patch.set_facecolor(BUCKET_COLORS[bucket_name])
        patch.set_alpha(0.55)
    boxplot_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    boxplot_axis.set_ylabel("delta psnr (dB)")
    boxplot_axis.set_title("Per-sample delta distribution by baseline label")
    boxplot_axis.grid(True, axis="y", alpha=0.25)

    count_summary = (
        merged.groupby(["baseline_label", "candidate_better"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(BUCKET_ORDER)
    )
    worse_counts = count_summary.get(False, pd.Series([0] * len(BUCKET_ORDER), index=BUCKET_ORDER))
    better_counts = count_summary.get(True, pd.Series([0] * len(BUCKET_ORDER), index=BUCKET_ORDER))
    count_axis.barh(
        [DISPLAY_LABELS[bucket_name] for bucket_name in BUCKET_ORDER],
        worse_counts.tolist(),
        color="#d62728",
        alpha=0.85,
        label="candidate <= baseline",
    )
    count_axis.barh(
        [DISPLAY_LABELS[bucket_name] for bucket_name in BUCKET_ORDER],
        better_counts.tolist(),
        left=worse_counts.tolist(),
        color="#2ca02c",
        alpha=0.85,
        label="candidate > baseline",
    )
    count_axis.set_xlabel("samples")
    count_axis.set_title("Improved vs not improved sample counts")
    count_axis.grid(True, axis="x", alpha=0.25)
    count_axis.legend()

    figure.suptitle(f"Candidate improvement by baseline label: {candidate_name}", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / "candidate_improvement_by_baseline_label.png", bbox_inches="tight")
    plt.close(figure)


def write_candidate_distribution_by_baseline_label(
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    candidate_name: str,
    motion_threshold: float | None,
    psnr_threshold: float | None,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9), dpi=160)
    motion_axis = axes[0, 0]
    scatter_axis = axes[0, 1]
    psnr_axis = axes[1, 0]
    coverage_axis = axes[1, 1]

    for bucket_name in BUCKET_ORDER:
        label_dataframe = merged[merged["baseline_label"] == bucket_name]
        if len(label_dataframe) == 0:
            continue
        legend_label = f"{bucket_name} (n={len(label_dataframe)})"
        color = BUCKET_COLORS[bucket_name]
        motion_axis.hist(
            label_dataframe[MOTION_COLUMN],
            bins=40,
            alpha=0.38,
            color=color,
            label=legend_label,
        )
        scatter_axis.scatter(
            label_dataframe[MOTION_COLUMN],
            label_dataframe["candidate_psnr"],
            s=12,
            alpha=0.4,
            edgecolors="none",
            color=color,
            label=legend_label,
        )
        psnr_axis.hist(
            label_dataframe["candidate_psnr"],
            bins=40,
            alpha=0.38,
            color=color,
            label=legend_label,
        )

    if motion_threshold is not None:
        motion_axis.axvline(
            motion_threshold,
            color="black",
            linestyle="--",
            linewidth=1.1,
            label=f"motion threshold = {motion_threshold:.2f}",
        )
        scatter_axis.axvline(
            motion_threshold,
            color="black",
            linestyle="--",
            linewidth=1.1,
            label=f"motion threshold = {motion_threshold:.2f}",
        )
    if psnr_threshold is not None:
        scatter_axis.axhline(
            psnr_threshold,
            color="gray",
            linestyle="--",
            linewidth=1.1,
            label=f"psnr threshold = {psnr_threshold:.2f}",
        )
        psnr_axis.axvline(
            psnr_threshold,
            color="gray",
            linestyle="--",
            linewidth=1.1,
            label=f"psnr threshold = {psnr_threshold:.2f}",
        )

    motion_axis.set_xlabel(MOTION_COLUMN)
    motion_axis.set_ylabel("samples")
    motion_axis.set_title("Per-sample motion distribution")
    motion_axis.grid(True, axis="y", alpha=0.25)
    motion_axis.legend(fontsize=8)

    scatter_axis.set_xlabel(MOTION_COLUMN)
    scatter_axis.set_ylabel("candidate psnr")
    scatter_axis.set_title("Candidate motion vs PSNR")
    scatter_axis.grid(True, alpha=0.25)
    scatter_axis.legend(fontsize=8, markerscale=1.3)

    psnr_axis.set_xlabel("candidate_psnr")
    psnr_axis.set_ylabel("samples")
    psnr_axis.set_title("Candidate PSNR distribution")
    psnr_axis.grid(True, axis="y", alpha=0.25)
    psnr_axis.legend(fontsize=8)

    coverage_dataframe = summary.copy()
    coverage_dataframe["dataset_percentage"] = coverage_dataframe["samples"] / len(merged) * 100.0
    coverage_axis.barh(
        coverage_dataframe["display_label"],
        coverage_dataframe["dataset_percentage"],
        color=[BUCKET_COLORS[str(label)] for label in coverage_dataframe["baseline_label"]],
    )
    for index, value in enumerate(coverage_dataframe["dataset_percentage"]):
        coverage_axis.text(float(value) + 0.6, index, f"{float(value):.1f}%", ha="left", va="center", fontsize=9)
    coverage_axis.set_xlabel("dataset percentage")
    coverage_axis.set_title("Baseline-label coverage")
    coverage_axis.grid(True, axis="x", alpha=0.25)

    figure.suptitle(f"Candidate distribution colored by baseline label: {candidate_name}", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / "candidate_distribution_by_baseline_label.png", bbox_inches="tight")
    plt.close(figure)


def write_oracle_distribution_by_baseline_label(
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    candidate_name: str,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9), dpi=160)
    oracle_axis = axes[0, 0]
    scatter_axis = axes[0, 1]
    gap_axis = axes[1, 0]
    improve_axis = axes[1, 1]

    for bucket_name in BUCKET_ORDER:
        label_dataframe = merged[merged["baseline_label"] == bucket_name]
        if len(label_dataframe) == 0:
            continue
        legend_label = f"{bucket_name} (n={len(label_dataframe)})"
        color = BUCKET_COLORS[bucket_name]
        oracle_axis.hist(
            label_dataframe[ORACLE_FMV_T_EFF_COLUMN],
            bins=40,
            alpha=0.38,
            color=color,
            label=legend_label,
        )
        scatter_axis.scatter(
            label_dataframe[ORACLE_FMV_T_EFF_COLUMN],
            label_dataframe["candidate_psnr"],
            s=12,
            alpha=0.4,
            edgecolors="none",
            color=color,
            label=legend_label,
        )
        gap_axis.hist(
            label_dataframe[ORACLE_T_EFF_GAP_COLUMN],
            bins=40,
            alpha=0.38,
            color=color,
            label=legend_label,
        )

    oracle_axis.axvline(0.5, color="black", linestyle="--", linewidth=1.1, label="oracle center = 0.50")
    oracle_axis.set_xlabel(ORACLE_FMV_T_EFF_COLUMN)
    oracle_axis.set_ylabel("samples")
    oracle_axis.set_title("Oracle forward t_eff distribution")
    oracle_axis.grid(True, axis="y", alpha=0.25)
    oracle_axis.legend(fontsize=8)

    scatter_axis.axvline(0.5, color="black", linestyle="--", linewidth=1.1, label="oracle center = 0.50")
    scatter_axis.set_xlabel(ORACLE_FMV_T_EFF_COLUMN)
    scatter_axis.set_ylabel("candidate psnr")
    scatter_axis.set_title("Oracle forward t_eff vs candidate PSNR")
    scatter_axis.grid(True, alpha=0.25)
    scatter_axis.legend(fontsize=8, markerscale=1.3)

    gap_axis.set_xlabel(ORACLE_T_EFF_GAP_COLUMN)
    gap_axis.set_ylabel("samples")
    gap_axis.set_title("Oracle forward/backward t_eff gap distribution")
    gap_axis.grid(True, axis="y", alpha=0.25)
    gap_axis.legend(fontsize=8)

    improve_axis.bar(
        summary["display_label"],
        summary["win_rate"] * 100.0,
        color=[BUCKET_COLORS[str(label)] for label in summary["baseline_label"]],
        alpha=0.9,
    )
    improve_axis.axhline(50.0, color="black", linestyle="--", linewidth=1.0)
    improve_axis.set_ylabel("candidate better rate (%)")
    improve_axis.set_title("Improvement rate by baseline label")
    improve_axis.grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(summary["win_rate"] * 100.0):
        improve_axis.text(index, float(value) + 1.0, f"{float(value):.1f}%", ha="center", va="bottom", fontsize=9)

    figure.suptitle(f"Oracle t_eff analysis by baseline label: {candidate_name}", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / "oracle_t_eff_by_baseline_label.png", bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    baseline_dataframe = load_csv(args.baseline_labeled_csv, "baseline labeled CSV")
    baseline_joined_dataframe = load_csv(args.baseline_joined_csv, "baseline joined CSV")
    candidate_dataframe = load_csv(args.candidate_csv, "candidate CSV")
    motion_threshold, psnr_threshold = extract_thresholds(baseline_dataframe)
    oracle_left_threshold, oracle_right_threshold = load_oracle_thresholds(args.oracle_thresholds_csv)
    baseline = build_baseline_frame(
        dataframe=baseline_dataframe,
        baseline_psnr_column=args.baseline_psnr_column,
        baseline_label_column=args.baseline_label_column,
    )
    baseline_metric_sidecar = build_metric_sidecar_frame(
        dataframe=baseline_joined_dataframe,
        rename_map=BASELINE_METRIC_RENAME_MAP,
        label="baseline joined CSV",
    )
    baseline = merge_baseline_metric_sidecar(
        baseline=baseline,
        baseline_metrics=baseline_metric_sidecar,
    )
    candidate = build_candidate_frame(
        dataframe=candidate_dataframe,
        candidate_psnr_column=args.candidate_psnr_column,
    )
    merged, unmatched = merge_baseline_and_candidate(baseline=baseline, candidate=candidate)
    summary = build_label_summary(merged)
    improvement_flags = build_improvement_flags(
        merged=merged,
        oracle_left_threshold=oracle_left_threshold,
        oracle_right_threshold=oracle_right_threshold,
        psnr_threshold=psnr_threshold,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_outputs(merged=merged, unmatched=unmatched, summary=summary, output_dir=output_dir)
    write_improvement_flags_csv(improvement_flags=improvement_flags, output_dir=output_dir)
    write_plot(merged=merged, summary=summary, candidate_name=args.candidate_name, output_dir=output_dir)
    write_candidate_distribution_by_baseline_label(
        merged=merged,
        summary=summary,
        candidate_name=args.candidate_name,
        motion_threshold=motion_threshold,
        psnr_threshold=psnr_threshold,
        output_dir=output_dir,
    )
    write_oracle_distribution_by_baseline_label(
        merged=merged,
        summary=summary,
        candidate_name=args.candidate_name,
        output_dir=output_dir,
    )

    print(f"baseline_labeled_csv={args.baseline_labeled_csv}")
    print(f"candidate_csv={args.candidate_csv}")
    print(f"output_dir={output_dir}")
    print(f"matched_samples={len(merged)}")
    print(f"unmatched_baseline_samples={len(unmatched)}")
    print(f"motion_threshold={motion_threshold}")
    print(f"psnr_threshold={psnr_threshold}")
    print(f"oracle_left_threshold={oracle_left_threshold}")
    print(f"oracle_right_threshold={oracle_right_threshold}")
    print(summary.to_string(index=False))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
