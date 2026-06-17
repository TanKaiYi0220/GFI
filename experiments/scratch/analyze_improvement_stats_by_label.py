from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

DEFAULT_INPUT_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\candidate_improvement_by_baseline_label\masked_downscale_minor_0611\baseline_label_improvement_flags.csv",
)
DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\candidate_improvement_by_baseline_label\masked_downscale_minor_0611\improve_statistics",
)
LABEL_ORDER: tuple[str, ...] = (
    "large_motion_artifacts",
    "not_large_motion_high_psnr",
    "other_artifacts",
    "large_motion_high_psnr",
)
LABEL_DISPLAY_NAMES: dict[str, str] = {
    "large_motion_artifacts": "large motion\nartifacts",
    "not_large_motion_high_psnr": "not large motion\nhigh psnr",
    "other_artifacts": "other\nartifacts",
    "large_motion_high_psnr": "large motion\nhigh psnr",
}
IMPROVE_LABELS: dict[bool, str] = {
    True: "Improve=True",
    False: "Improve=False",
}
BOXPLOT_FEATURES: tuple[tuple[str, str], ...] = (
    ("motion_magnitude_mean", "motion_magnitude_mean"),
    ("baseline_psnr", "baseline_psnr"),
    ("delta_psnr", "delta_psnr"),
    ("delta_ssim", "delta_ssim"),
)
DISTRIBUTION_FEATURES: tuple[tuple[str, str], ...] = (
    ("motion_magnitude_mean", "motion_magnitude_mean"),
    ("baseline_psnr", "baseline_psnr"),
)
STAT_FEATURES: tuple[str, ...] = (
    "motion_magnitude_mean",
    "baseline_psnr",
    "baseline_ssim",
    "candidate_psnr",
    "candidate_ssim",
    "delta_psnr",
    "delta_ssim",
    "candidate_lpips",
    "oracle_fmv_t_eff_mean",
    "oracle_t_eff_gap_mean",
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run descriptive and statistical analysis for improve=True/False inside each baseline label."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to baseline_label_improvement_flags.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV summaries and plots.",
    )
    return parser.parse_args(argv)


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    require_existing_path(csv_path, "input CSV")
    dataframe = pd.read_csv(csv_path)
    if len(dataframe) == 0:
        raise ValueError(f"Input CSV is empty: {csv_path}")
    required_columns = {"baseline_label", "improve", "delta_psnr"}
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if len(missing_columns) > 0:
        raise ValueError(f"Input CSV is missing required columns: {missing_columns}")
    dataframe = dataframe.copy()
    dataframe["baseline_label"] = pd.Categorical(dataframe["baseline_label"], categories=LABEL_ORDER, ordered=True)
    dataframe["improve"] = dataframe["improve"].astype(bool)
    dataframe["improve_display"] = dataframe["improve"].map(IMPROVE_LABELS)
    return dataframe.sort_values(["baseline_label", "improve", "delta_psnr"], ascending=[True, False, False]).reset_index(drop=True)


def summarize_value_series(values: pd.Series) -> dict[str, float]:
    clean_values = values.dropna()
    if len(clean_values) == 0:
        return {
            "count": 0.0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "q1": math.nan,
            "median": math.nan,
            "q3": math.nan,
            "max": math.nan,
        }
    return {
        "count": float(len(clean_values)),
        "mean": float(clean_values.mean()),
        "std": float(clean_values.std(ddof=0)),
        "min": float(clean_values.min()),
        "q1": float(clean_values.quantile(0.25)),
        "median": float(clean_values.median()),
        "q3": float(clean_values.quantile(0.75)),
        "max": float(clean_values.max()),
    }


def build_count_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    summary = (
        dataframe.groupby(["baseline_label", "improve"], observed=True)
        .size()
        .reset_index(name="samples")
    )
    total_per_label = (
        dataframe.groupby("baseline_label", observed=True)
        .size()
        .rename("label_total")
        .reset_index()
    )
    summary = summary.merge(total_per_label, on="baseline_label", how="left")
    summary["label_ratio"] = summary["samples"] / summary["label_total"]
    summary["improve_display"] = summary["improve"].map(IMPROVE_LABELS)
    summary["baseline_label_display"] = [LABEL_DISPLAY_NAMES[str(label)] for label in summary["baseline_label"]]
    return summary


def build_descriptive_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    available_features = [feature for feature in STAT_FEATURES if feature in dataframe.columns]
    for baseline_label in LABEL_ORDER:
        label_dataframe = dataframe[dataframe["baseline_label"] == baseline_label]
        if len(label_dataframe) == 0:
            continue
        for improve_value in (True, False):
            group_dataframe = label_dataframe[label_dataframe["improve"] == improve_value]
            for feature_name in available_features:
                stats = summarize_value_series(group_dataframe[feature_name])
                rows.append(
                    {
                        "baseline_label": baseline_label,
                        "baseline_label_display": LABEL_DISPLAY_NAMES[baseline_label],
                        "improve": improve_value,
                        "improve_display": IMPROVE_LABELS[improve_value],
                        "feature": feature_name,
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def build_mannwhitney_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    available_features = [feature for feature in STAT_FEATURES if feature in dataframe.columns]
    for baseline_label in LABEL_ORDER:
        label_dataframe = dataframe[dataframe["baseline_label"] == baseline_label]
        if len(label_dataframe) == 0:
            continue
        improved = label_dataframe[label_dataframe["improve"] == True]
        not_improved = label_dataframe[label_dataframe["improve"] == False]
        for feature_name in available_features:
            improved_values = improved[feature_name].dropna()
            not_improved_values = not_improved[feature_name].dropna()
            if len(improved_values) == 0 or len(not_improved_values) == 0:
                rows.append(
                    {
                        "baseline_label": baseline_label,
                        "baseline_label_display": LABEL_DISPLAY_NAMES[baseline_label],
                        "feature": feature_name,
                        "improved_samples": int(len(improved_values)),
                        "not_improved_samples": int(len(not_improved_values)),
                        "improved_median": math.nan if len(improved_values) == 0 else float(improved_values.median()),
                        "not_improved_median": math.nan if len(not_improved_values) == 0 else float(not_improved_values.median()),
                        "median_gap_improved_minus_not_improved": math.nan,
                        "mannwhitney_u": math.nan,
                        "p_value": math.nan,
                        "rank_biserial_correlation": math.nan,
                    }
                )
                continue

            mannwhitney = mannwhitneyu(improved_values, not_improved_values, alternative="two-sided")
            sample_pairs = len(improved_values) * len(not_improved_values)
            rank_biserial = (2.0 * float(mannwhitney.statistic) / float(sample_pairs)) - 1.0
            rows.append(
                {
                    "baseline_label": baseline_label,
                    "baseline_label_display": LABEL_DISPLAY_NAMES[baseline_label],
                    "feature": feature_name,
                    "improved_samples": int(len(improved_values)),
                    "not_improved_samples": int(len(not_improved_values)),
                    "improved_median": float(improved_values.median()),
                    "not_improved_median": float(not_improved_values.median()),
                    "median_gap_improved_minus_not_improved": float(improved_values.median() - not_improved_values.median()),
                    "mannwhitney_u": float(mannwhitney.statistic),
                    "p_value": float(mannwhitney.pvalue),
                    "rank_biserial_correlation": rank_biserial,
                }
            )
    return pd.DataFrame(rows)


def write_csv_outputs(
    count_summary: pd.DataFrame,
    descriptive_summary: pd.DataFrame,
    mannwhitney_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    count_summary.to_csv(output_dir / "label_improve_counts.csv", index=False)
    descriptive_summary.to_csv(output_dir / "label_improve_descriptive_stats.csv", index=False)
    mannwhitney_summary.to_csv(output_dir / "label_improve_mannwhitney.csv", index=False)

def write_boxplot_dashboard(dataframe: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=160)
    for axis, (feature_name, title) in zip(axes.flatten(), BOXPLOT_FEATURES):
        if feature_name not in dataframe.columns:
            axis.axis("off")
            continue
        sns.boxplot(
            data=dataframe,
            x="baseline_label",
            y=feature_name,
            hue="improve_display",
            order=list(LABEL_ORDER),
            hue_order=[IMPROVE_LABELS[True], IMPROVE_LABELS[False]],
            showfliers=False,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("")
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_xticks(range(len(LABEL_ORDER)))
        axis.set_xticklabels([LABEL_DISPLAY_NAMES[label] for label in LABEL_ORDER])
        if axis.legend_ is not None:
            axis.legend_.set_title("")
    figure.suptitle("Per-label improve=True/False boxplots", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / "label_improve_boxplots.png", bbox_inches="tight")
    plt.close(figure)


def write_distribution_dashboard(dataframe: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=160)
    for row_index, (feature_name, title) in enumerate(DISTRIBUTION_FEATURES):
        for column_index, baseline_label in enumerate(LABEL_ORDER):
            axis = axes[row_index, column_index]
            label_dataframe = dataframe[dataframe["baseline_label"] == baseline_label]
            if feature_name not in label_dataframe.columns or len(label_dataframe) == 0:
                axis.axis("off")
                continue
            for improve_value, color in ((True, "#2ca02c"), (False, "#d62728")):
                group_dataframe = label_dataframe[label_dataframe["improve"] == improve_value]
                if len(group_dataframe) == 0:
                    continue
                sns.histplot(
                    group_dataframe[feature_name].dropna(),
                    bins=24,
                    stat="density",
                    element="step",
                    fill=False,
                    common_norm=False,
                    color=color,
                    label=IMPROVE_LABELS[improve_value],
                    ax=axis,
                )
            axis.set_title(f"{LABEL_DISPLAY_NAMES[baseline_label]}\n{title}")
            axis.grid(True, axis="y", alpha=0.25)
            if row_index == len(DISTRIBUTION_FEATURES) - 1:
                axis.set_xlabel(feature_name)
            else:
                axis.set_xlabel("")
            if column_index == 0:
                axis.set_ylabel("density")
            else:
                axis.set_ylabel("")
            if axis.legend_ is None:
                axis.legend(fontsize=8, title="")
    figure.suptitle("Per-label improve=True/False distributions", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / "label_improve_distributions.png", bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    dataframe = load_dataframe(args.input_csv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    count_summary = build_count_summary(dataframe)
    descriptive_summary = build_descriptive_summary(dataframe)
    mannwhitney_summary = build_mannwhitney_summary(dataframe)
    write_csv_outputs(
        count_summary=count_summary,
        descriptive_summary=descriptive_summary,
        mannwhitney_summary=mannwhitney_summary,
        output_dir=output_dir,
    )
    write_boxplot_dashboard(dataframe=dataframe, output_dir=output_dir)
    write_distribution_dashboard(dataframe=dataframe, output_dir=output_dir)

    print(f"input_csv={args.input_csv}")
    print(f"output_dir={output_dir}")
    print(count_summary.to_string(index=False))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
