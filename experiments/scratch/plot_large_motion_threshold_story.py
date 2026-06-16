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
from sklearn.tree import DecisionTreeClassifier

from experiments.scratch.analyze_artifact_overlap_clustered import fit_threshold_tree
from experiments.scratch.analyze_artifact_overlap_clustered import run_kmeans_artifact_clustering

DEFAULT_SAMPLE_COMPARISON_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\baseline_finetuning_oracle_t_eff_psnr\motion_psnr_sample_comparison.csv",
)
DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\baseline_finetuning_oracle_t_eff_psnr\large_motion_story",
)
DEFAULT_PSNR_COLUMN: str = "IFRNet FineTuning"
DEFAULT_RANDOM_STATE: int = 1234
DEFAULT_TREE_MAX_DEPTH: int = 2
DEFAULT_TREE_MIN_SAMPLES_LEAF: int = 20
DEFAULT_ONSET_MIN_SAMPLES_LEAF: int = 50


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a slide-friendly large-motion threshold story plot."
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
        help="Directory for story plot and summary CSVs.",
    )
    parser.add_argument(
        "--psnr-column",
        type=str,
        default=DEFAULT_PSNR_COLUMN,
        help="PSNR column used for analysis.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for clustering and trees.",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=DEFAULT_TREE_MAX_DEPTH,
        help="Max depth for the cluster-core threshold tree.",
    )
    parser.add_argument(
        "--tree-min-samples-leaf",
        type=int,
        default=DEFAULT_TREE_MIN_SAMPLES_LEAF,
        help="Minimum leaf size for the cluster-core threshold tree.",
    )
    parser.add_argument(
        "--onset-min-samples-leaf",
        type=int,
        default=DEFAULT_ONSET_MIN_SAMPLES_LEAF,
        help="Minimum leaf size for the onset threshold tree.",
    )
    return parser.parse_args(argv)


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    require_existing_path(csv_path, "sample comparison CSV")
    dataframe = pd.read_csv(csv_path)
    if len(dataframe) == 0:
        raise ValueError(f"Sample comparison CSV is empty: {csv_path}")
    return dataframe


def validate_columns(dataframe: pd.DataFrame, psnr_column: str) -> None:
    required_columns = ("motion_magnitude_mean", psnr_column)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"Sample comparison CSV is missing columns: {missing_columns}")


def extract_threshold_by_feature(
    threshold_rows: list[dict[str, object]],
    feature_name: str,
    operator: str | None,
) -> float | None:
    for row in threshold_rows:
        if str(row["feature_name"]) != feature_name:
            continue
        if operator is not None and str(row["operator"]) != operator:
            continue
        return float(row["threshold"])
    return None


def build_story_summary_rows(
    psnr_q1: float,
    core_motion_threshold: float,
    core_psnr_threshold: float,
    core_mask: pd.Series,
    onset_motion_threshold: float,
    onset_mask: pd.Series,
    onset_low_q1_mask: pd.Series,
) -> list[dict[str, object]]:
    return [
        {
            "metric": "psnr_q1_threshold",
            "value": float(psnr_q1),
        },
        {
            "metric": "cluster_core_motion_threshold",
            "value": float(core_motion_threshold),
        },
        {
            "metric": "cluster_core_psnr_threshold",
            "value": float(core_psnr_threshold),
        },
        {
            "metric": "cluster_core_dataset_ratio",
            "value": float(core_mask.mean()),
        },
        {
            "metric": "onset_motion_threshold",
            "value": float(onset_motion_threshold),
        },
        {
            "metric": "onset_zone_dataset_ratio",
            "value": float(onset_mask.mean()),
        },
        {
            "metric": "low_q1_inside_onset_dataset_ratio",
            "value": float(onset_low_q1_mask.mean()),
        },
        {
            "metric": "low_q1_inside_onset_zone_ratio",
            "value": float(onset_low_q1_mask.sum() / onset_mask.sum()),
        },
    ]


def write_story_summary_csv(rows: list[dict[str, object]], output_dir: Path) -> None:
    pd.DataFrame(rows).to_csv(output_dir / "large_motion_story_summary.csv", index=False)


def write_story_plot(
    dataframe: pd.DataFrame,
    psnr_column: str,
    psnr_q1: float,
    core_motion_threshold: float,
    core_psnr_threshold: float,
    core_mask: pd.Series,
    onset_motion_threshold: float,
    onset_mask: pd.Series,
    onset_low_q1_mask: pd.Series,
    output_dir: Path,
) -> None:
    figure, (scatter_axis, bar_axis) = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        dpi=160,
        gridspec_kw={"width_ratios": [2.6, 1.2]},
    )

    scatter_axis.scatter(
        dataframe["motion_magnitude_mean"],
        dataframe[psnr_column],
        s=10,
        alpha=0.18,
        edgecolors="none",
        color="#9e9e9e",
        label=f"all samples (n={len(dataframe)})",
    )
    scatter_axis.scatter(
        dataframe.loc[onset_mask, "motion_magnitude_mean"],
        dataframe.loc[onset_mask, psnr_column],
        s=16,
        alpha=0.35,
        edgecolors="none",
        color="#ff7f0e",
        label=f"practical onset zone (n={int(onset_mask.sum())})",
    )
    scatter_axis.scatter(
        dataframe.loc[core_mask, "motion_magnitude_mean"],
        dataframe.loc[core_mask, psnr_column],
        s=16,
        alpha=0.55,
        edgecolors="none",
        color="#9467bd",
        label=f"cluster core artifact (n={int(core_mask.sum())})",
    )

    scatter_axis.axvline(
        core_motion_threshold,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=f"core motion > {core_motion_threshold:.2f}",
    )
    scatter_axis.axhline(
        core_psnr_threshold,
        color="#666666",
        linestyle="--",
        linewidth=1.1,
        label=f"core psnr <= {core_psnr_threshold:.2f}",
    )
    scatter_axis.axvline(
        onset_motion_threshold,
        color="#ff7f0e",
        linestyle="-.",
        linewidth=1.2,
        label=f"onset motion > {onset_motion_threshold:.2f}",
    )
    scatter_axis.axhline(
        psnr_q1,
        color="#1f77b4",
        linestyle="-.",
        linewidth=1.2,
        label=f"Q1 psnr = {psnr_q1:.2f}",
    )
    scatter_axis.set_xlabel("motion_magnitude_mean")
    scatter_axis.set_ylabel("psnr")
    scatter_axis.set_title("Large motion threshold story")
    scatter_axis.grid(True, alpha=0.25)
    scatter_axis.legend(loc="upper right", markerscale=1.4)

    onset_zone_ratio = float(onset_mask.mean() * 100.0)
    onset_low_dataset_ratio = float(onset_low_q1_mask.mean() * 100.0)
    onset_low_zone_ratio = float(onset_low_q1_mask.sum() / onset_mask.sum() * 100.0)
    core_ratio = float(core_mask.mean() * 100.0)

    bar_labels = [
        "core artifact\n(dataset)",
        "onset zone\n(dataset)",
        "low-Q1 inside onset\n(dataset)",
    ]
    bar_values = [core_ratio, onset_zone_ratio, onset_low_dataset_ratio]
    bar_colors = ["#9467bd", "#ff7f0e", "#1f77b4"]
    bar_axis.barh(bar_labels, bar_values, color=bar_colors)
    for index, value in enumerate(bar_values):
        bar_axis.text(value + 0.8, index, f"{value:.1f}%", va="center", ha="left", fontsize=9)
    bar_axis.set_xlim(0.0, max(20.0, max(bar_values) * 1.35))
    bar_axis.set_xlabel("dataset percentage")
    bar_axis.set_title("Coverage summary")
    bar_axis.grid(True, axis="x", alpha=0.2)
    bar_axis.text(
        0.02,
        -0.42,
        f"{onset_low_zone_ratio:.1f}% of onset-zone samples are below Q1 PSNR",
        transform=bar_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    figure.suptitle(
        "Baseline IFRNet FineTuning: core large-motion artifact vs practical onset",
        fontsize=15,
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "large_motion_threshold_story.png", bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    dataframe = load_dataframe(args.sample_comparison_csv)
    validate_columns(dataframe, args.psnr_column)
    psnr_q1 = float(dataframe[args.psnr_column].quantile(0.25))

    large_motion_cluster = run_kmeans_artifact_clustering(
        dataframe=dataframe,
        feature_columns=["motion_magnitude_mean", args.psnr_column],
        primary_feature_column="motion_magnitude_mean",
        psnr_column=args.psnr_column,
        cluster_count=3,
        random_state=args.random_state,
    )
    large_motion_tree = fit_threshold_tree(
        dataframe=dataframe,
        feature_columns=["motion_magnitude_mean", args.psnr_column],
        cluster_labels=large_motion_cluster.cluster_labels,
        artifact_cluster_id=large_motion_cluster.artifact_cluster_id,
        max_depth=args.tree_max_depth,
        min_samples_leaf=args.tree_min_samples_leaf,
        random_state=args.random_state,
        artifact_label="large_motion_artifact",
    )
    core_motion_threshold = extract_threshold_by_feature(
        large_motion_tree.threshold_rows,
        feature_name="motion_magnitude_mean",
        operator=">",
    )
    core_psnr_threshold = extract_threshold_by_feature(
        large_motion_tree.threshold_rows,
        feature_name=args.psnr_column,
        operator="<=",
    )
    if core_motion_threshold is None or core_psnr_threshold is None:
        raise ValueError("Failed to extract cluster-core motion/psnr thresholds from tree rule.")

    low_q1_targets = (dataframe[args.psnr_column] <= psnr_q1).astype(int)
    onset_tree = DecisionTreeClassifier(
        max_depth=1,
        min_samples_leaf=args.onset_min_samples_leaf,
        random_state=args.random_state,
    )
    onset_tree.fit(dataframe[["motion_magnitude_mean"]], low_q1_targets)
    onset_motion_threshold = float(onset_tree.tree_.threshold[0])

    core_mask = large_motion_tree.threshold_mask.astype(bool)
    onset_mask = dataframe["motion_magnitude_mean"] > onset_motion_threshold
    onset_low_q1_mask = onset_mask & (dataframe[args.psnr_column] <= psnr_q1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_story_summary_csv(
        rows=build_story_summary_rows(
            psnr_q1=psnr_q1,
            core_motion_threshold=core_motion_threshold,
            core_psnr_threshold=core_psnr_threshold,
            core_mask=core_mask,
            onset_motion_threshold=onset_motion_threshold,
            onset_mask=onset_mask,
            onset_low_q1_mask=onset_low_q1_mask,
        ),
        output_dir=output_dir,
    )
    write_story_plot(
        dataframe=dataframe,
        psnr_column=args.psnr_column,
        psnr_q1=psnr_q1,
        core_motion_threshold=core_motion_threshold,
        core_psnr_threshold=core_psnr_threshold,
        core_mask=core_mask,
        onset_motion_threshold=onset_motion_threshold,
        onset_mask=onset_mask,
        onset_low_q1_mask=onset_low_q1_mask,
        output_dir=output_dir,
    )

    print(f"input={args.sample_comparison_csv}")
    print(f"output_dir={output_dir}")
    print(f"cluster_core_rule={large_motion_tree.positive_rule_text}")
    print(f"cluster_core_rule_fidelity={large_motion_tree.fidelity_to_cluster:.6f}")
    print(f"q1_psnr={psnr_q1:.6f}")
    print(f"practical_onset_motion_threshold={onset_motion_threshold:.6f}")
    print(f"core_artifact_dataset_ratio={float(core_mask.mean()):.6f}")
    print(f"onset_zone_dataset_ratio={float(onset_mask.mean()):.6f}")
    print(f"low_q1_inside_onset_dataset_ratio={float(onset_low_q1_mask.mean()):.6f}")
    print(f"low_q1_inside_onset_zone_ratio={float(onset_low_q1_mask.sum() / onset_mask.sum()):.6f}")


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
