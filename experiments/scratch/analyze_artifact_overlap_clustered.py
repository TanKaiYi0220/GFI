from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

DEFAULT_SAMPLE_COMPARISON_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\baseline_finetuning_oracle_t_eff_psnr\motion_psnr_sample_comparison.csv",
)
DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\baseline_finetuning_oracle_t_eff_psnr\artifact_overlap_clustered",
)
DEFAULT_PSNR_COLUMN: str = "IFRNet FineTuning"
DEFAULT_CLUSTER_COUNT: int = 3
DEFAULT_RANDOM_STATE: int = 1234
DEFAULT_TREE_MAX_DEPTH: int = 2
DEFAULT_TREE_MIN_SAMPLES_LEAF: int = 20
ORACLE_CENTER: float = 0.5
OVERLAP_ORDER: tuple[str, ...] = (
    "both_artifacts",
    "large_motion_only",
    "oracle_t_eff_only",
    "neither",
)
OVERLAP_COLORS: dict[str, str] = {
    "both_artifacts": "#9467bd",
    "large_motion_only": "#d62728",
    "oracle_t_eff_only": "#1f77b4",
    "neither": "#bdbdbd",
}


@dataclass(frozen=True)
class ClusterResult:
    summary: pd.DataFrame
    artifact_cluster_id: int
    cluster_labels: pd.Series


@dataclass(frozen=True)
class ThresholdModelResult:
    threshold_mask: pd.Series
    fidelity_to_cluster: float
    positive_rule_text: str
    threshold_rows: list[dict[str, object]]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find artifact thresholds from clustering instead of fixed quantiles."
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
        help="Directory for overlap CSVs and plots.",
    )
    parser.add_argument(
        "--psnr-column",
        type=str,
        default=DEFAULT_PSNR_COLUMN,
        help="PSNR column used for artifact scoring.",
    )
    parser.add_argument(
        "--motion-cluster-count",
        type=int,
        default=DEFAULT_CLUSTER_COUNT,
        help="KMeans cluster count for motion-vs-PSNR clustering.",
    )
    parser.add_argument(
        "--oracle-cluster-count",
        type=int,
        default=DEFAULT_CLUSTER_COUNT,
        help="KMeans cluster count for oracle-distance-vs-PSNR clustering.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for clustering and tree fitting.",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=DEFAULT_TREE_MAX_DEPTH,
        help="Max depth for the readable threshold decision tree.",
    )
    parser.add_argument(
        "--tree-min-samples-leaf",
        type=int,
        default=DEFAULT_TREE_MIN_SAMPLES_LEAF,
        help="Minimum samples per leaf for the threshold decision tree.",
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
    required_columns = ("motion_magnitude_mean", "oracle_fmv_t_eff_mean", psnr_column)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"Sample comparison CSV is missing columns: {missing_columns}")


def run_kmeans_artifact_clustering(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    primary_feature_column: str,
    psnr_column: str,
    cluster_count: int,
    random_state: int,
) -> ClusterResult:
    if cluster_count < 2:
        raise ValueError(f"cluster_count must be at least 2, got {cluster_count}")
    if cluster_count > len(dataframe):
        raise ValueError(f"cluster_count={cluster_count} cannot exceed sample count={len(dataframe)}")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(dataframe[feature_columns].to_numpy())
    cluster_model = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=50)
    cluster_labels = pd.Series(cluster_model.fit_predict(scaled_features), index=dataframe.index, name="cluster_id")

    summary = (
        dataframe.assign(cluster_id=cluster_labels)
        .groupby("cluster_id", as_index=False)
        .agg(
            samples=("motion_magnitude_mean", "size"),
            primary_feature_mean=(primary_feature_column, "mean"),
            psnr_mean=(psnr_column, "mean"),
        )
    )
    summary["primary_feature_z"] = (
        summary["primary_feature_mean"] - summary["primary_feature_mean"].mean()
    ) / summary["primary_feature_mean"].std(ddof=0)
    summary["psnr_z"] = (
        summary["psnr_mean"] - summary["psnr_mean"].mean()
    ) / summary["psnr_mean"].std(ddof=0)
    summary["artifact_score"] = summary["primary_feature_z"] - summary["psnr_z"]
    artifact_cluster_id = int(summary.sort_values("artifact_score", ascending=False).iloc[0]["cluster_id"])
    return ClusterResult(summary=summary, artifact_cluster_id=artifact_cluster_id, cluster_labels=cluster_labels)


def extract_positive_rules(
    tree_model: DecisionTreeClassifier,
    feature_names: list[str],
) -> list[list[tuple[str, str, float]]]:
    tree = tree_model.tree_
    rules: list[list[tuple[str, str, float]]] = []

    def walk(node_id: int, path_rules: list[tuple[str, str, float]]) -> None:
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            predicted_class_id = int(tree.value[node_id][0].argmax())
            if predicted_class_id == 1:
                rules.append(list(path_rules))
            return

        feature_name = feature_names[int(tree.feature[node_id])]
        threshold = float(tree.threshold[node_id])

        walk(
            tree.children_left[node_id],
            path_rules + [(feature_name, "<=", threshold)],
        )
        walk(
            tree.children_right[node_id],
            path_rules + [(feature_name, ">", threshold)],
        )

    walk(0, [])
    return rules


def normalize_rule_path(path_rules: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    return sorted(path_rules, key=lambda row: (row[0], row[1], row[2]))


def simplify_positive_rules(
    rules: list[list[tuple[str, str, float]]],
) -> list[list[tuple[str, str, float]]]:
    simplified_rules = [normalize_rule_path(path_rules) for path_rules in rules]
    changed = True
    while changed:
        changed = False
        next_rules: list[list[tuple[str, str, float]]] = []
        used_indices: set[int] = set()
        for left_index, left_rules in enumerate(simplified_rules):
            if left_index in used_indices:
                continue

            merged = False
            for right_index in range(left_index + 1, len(simplified_rules)):
                if right_index in used_indices:
                    continue

                right_rules = simplified_rules[right_index]
                if len(left_rules) != len(right_rules):
                    continue

                differing_positions: list[int] = []
                for position, (left_rule, right_rule) in enumerate(zip(left_rules, right_rules)):
                    if left_rule != right_rule:
                        differing_positions.append(position)

                if len(differing_positions) != 1:
                    continue

                differing_position = differing_positions[0]
                left_rule = left_rules[differing_position]
                right_rule = right_rules[differing_position]
                if (
                    left_rule[0] != right_rule[0]
                    or left_rule[2] != right_rule[2]
                    or {left_rule[1], right_rule[1]} != {"<=", ">"}
                ):
                    continue

                merged_rules = [rule for position, rule in enumerate(left_rules) if position != differing_position]
                next_rules.append(normalize_rule_path(merged_rules))
                used_indices.add(left_index)
                used_indices.add(right_index)
                changed = True
                merged = True
                break

            if not merged and left_index not in used_indices:
                next_rules.append(left_rules)
                used_indices.add(left_index)

        simplified_rules = [normalize_rule_path(path_rules) for path_rules in next_rules]

    deduplicated_rules: list[list[tuple[str, str, float]]] = []
    seen_keys: set[tuple[tuple[str, str, float], ...]] = set()
    for path_rules in simplified_rules:
        key = tuple(path_rules)
        if key in seen_keys:
            continue
        deduplicated_rules.append(path_rules)
        seen_keys.add(key)
    return deduplicated_rules


def rules_to_text(rules: list[list[tuple[str, str, float]]]) -> str:
    if len(rules) == 0:
        return "no positive rule found"

    rendered_paths: list[str] = []
    for path_rules in rules:
        rendered_paths.append(
            " and ".join(f"{feature_name} {operator} {threshold:.6f}" for feature_name, operator, threshold in path_rules)
        )
    return " OR ".join(rendered_paths)


def rules_to_rows(label: str, rules: list[list[tuple[str, str, float]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path_index, path_rules in enumerate(rules):
        for feature_name, operator, threshold in path_rules:
            rows.append(
                {
                    "artifact_label": label,
                    "path_index": path_index,
                    "feature_name": feature_name,
                    "operator": operator,
                    "threshold": float(threshold),
                }
            )
    return rows


def fit_threshold_tree(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    cluster_labels: pd.Series,
    artifact_cluster_id: int,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
    artifact_label: str,
) -> ThresholdModelResult:
    binary_targets = (cluster_labels == artifact_cluster_id).astype(int)
    tree_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree_model.fit(dataframe[feature_columns], binary_targets)
    threshold_mask = pd.Series(
        tree_model.predict(dataframe[feature_columns]).astype(bool),
        index=dataframe.index,
        name=f"is_{artifact_label}",
    )
    fidelity_to_cluster = float((threshold_mask.astype(int) == binary_targets).mean())
    positive_rules = simplify_positive_rules(extract_positive_rules(tree_model, feature_columns))
    return ThresholdModelResult(
        threshold_mask=threshold_mask,
        fidelity_to_cluster=fidelity_to_cluster,
        positive_rule_text=rules_to_text(positive_rules),
        threshold_rows=rules_to_rows(artifact_label, positive_rules),
    )


def classify_overlap_label(is_large_motion_artifact: bool, is_oracle_t_eff_artifact: bool) -> str:
    if is_large_motion_artifact and is_oracle_t_eff_artifact:
        return "both_artifacts"
    if is_large_motion_artifact:
        return "large_motion_only"
    if is_oracle_t_eff_artifact:
        return "oracle_t_eff_only"
    return "neither"


def build_overlap_dataframe(
    dataframe: pd.DataFrame,
    large_motion_mask: pd.Series,
    oracle_mask: pd.Series,
) -> pd.DataFrame:
    overlap = dataframe.copy()
    overlap["psnr_summary"] = overlap["IFRNet FineTuning"].astype(float)
    overlap["oracle_t_eff_distance"] = (overlap["oracle_fmv_t_eff_mean"] - ORACLE_CENTER).abs()
    overlap["is_large_motion_artifact"] = large_motion_mask.astype(bool)
    overlap["is_oracle_t_eff_artifact"] = oracle_mask.astype(bool)
    overlap["overlap_label"] = [
        classify_overlap_label(bool(is_large), bool(is_oracle))
        for is_large, is_oracle in zip(
            overlap["is_large_motion_artifact"],
            overlap["is_oracle_t_eff_artifact"],
        )
    ]
    overlap["overlap_label"] = pd.Categorical(overlap["overlap_label"], categories=OVERLAP_ORDER, ordered=True)
    return overlap


def build_overlap_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_samples = len(dataframe)
    for overlap_label in OVERLAP_ORDER:
        label_dataframe = dataframe[dataframe["overlap_label"] == overlap_label]
        rows.append(
            {
                "group": overlap_label,
                "samples": int(len(label_dataframe)),
                "ratio": float(len(label_dataframe) / total_samples),
                "motion_mean": float(label_dataframe["motion_magnitude_mean"].mean()) if len(label_dataframe) > 0 else 0.0,
                "oracle_fmv_t_eff_mean": float(label_dataframe["oracle_fmv_t_eff_mean"].mean()) if len(label_dataframe) > 0 else 0.0,
                "psnr_mean": float(label_dataframe["psnr_summary"].mean()) if len(label_dataframe) > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_threshold_summary_rows(
    large_motion_cluster: ClusterResult,
    large_motion_tree: ThresholdModelResult,
    oracle_cluster: ClusterResult,
    oracle_tree: ThresholdModelResult,
) -> list[dict[str, object]]:
    return [
        {
            "artifact_label": "large_motion_artifact",
            "artifact_cluster_id": large_motion_cluster.artifact_cluster_id,
            "cluster_samples": int(large_motion_cluster.summary.loc[
                large_motion_cluster.summary["cluster_id"] == large_motion_cluster.artifact_cluster_id,
                "samples",
            ].iloc[0]),
            "cluster_primary_feature_mean": float(large_motion_cluster.summary.loc[
                large_motion_cluster.summary["cluster_id"] == large_motion_cluster.artifact_cluster_id,
                "primary_feature_mean",
            ].iloc[0]),
            "cluster_psnr_mean": float(large_motion_cluster.summary.loc[
                large_motion_cluster.summary["cluster_id"] == large_motion_cluster.artifact_cluster_id,
                "psnr_mean",
            ].iloc[0]),
            "threshold_rule": large_motion_tree.positive_rule_text,
            "threshold_rule_fidelity": large_motion_tree.fidelity_to_cluster,
        },
        {
            "artifact_label": "oracle_t_eff_artifact",
            "artifact_cluster_id": oracle_cluster.artifact_cluster_id,
            "cluster_samples": int(oracle_cluster.summary.loc[
                oracle_cluster.summary["cluster_id"] == oracle_cluster.artifact_cluster_id,
                "samples",
            ].iloc[0]),
            "cluster_primary_feature_mean": float(oracle_cluster.summary.loc[
                oracle_cluster.summary["cluster_id"] == oracle_cluster.artifact_cluster_id,
                "primary_feature_mean",
            ].iloc[0]),
            "cluster_psnr_mean": float(oracle_cluster.summary.loc[
                oracle_cluster.summary["cluster_id"] == oracle_cluster.artifact_cluster_id,
                "psnr_mean",
            ].iloc[0]),
            "threshold_rule": oracle_tree.positive_rule_text,
            "threshold_rule_fidelity": oracle_tree.fidelity_to_cluster,
        },
    ]


def write_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    dataframe.to_csv(output_path, index=False)


def plot_motion_threshold_scatter(
    dataframe: pd.DataFrame,
    threshold_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5), dpi=160)
    for overlap_label in OVERLAP_ORDER:
        label_dataframe = dataframe[dataframe["overlap_label"] == overlap_label]
        plt.scatter(
            label_dataframe["motion_magnitude_mean"],
            label_dataframe["psnr_summary"],
            s=12,
            alpha=0.45,
            edgecolors="none",
            label=f"{overlap_label} (n={len(label_dataframe)})",
            color=OVERLAP_COLORS[overlap_label],
        )

    for row in threshold_rows:
        if row["artifact_label"] != "large_motion_artifact":
            continue
        feature_name = str(row["feature_name"])
        operator = str(row["operator"])
        threshold = float(row["threshold"])
        if feature_name == "motion_magnitude_mean":
            plt.axvline(threshold, color="black", linestyle="--", linewidth=1.0, label=f"motion {operator} {threshold:.2f}")
        if feature_name == "IFRNet FineTuning":
            plt.axhline(threshold, color="gray", linestyle="--", linewidth=1.0, label=f"psnr {operator} {threshold:.2f}")

    plt.xlabel("motion_magnitude_mean")
    plt.ylabel("psnr_summary")
    plt.title("Cluster-derived large motion artifact threshold")
    plt.grid(True, alpha=0.25)
    plt.legend(markerscale=1.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_oracle_threshold_scatter(
    dataframe: pd.DataFrame,
    threshold_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5), dpi=160)
    for overlap_label in OVERLAP_ORDER:
        label_dataframe = dataframe[dataframe["overlap_label"] == overlap_label]
        plt.scatter(
            label_dataframe["oracle_fmv_t_eff_mean"],
            label_dataframe["psnr_summary"],
            s=12,
            alpha=0.45,
            edgecolors="none",
            label=f"{overlap_label} (n={len(label_dataframe)})",
            color=OVERLAP_COLORS[overlap_label],
        )

    for row in threshold_rows:
        if row["artifact_label"] != "oracle_t_eff_artifact":
            continue
        feature_name = str(row["feature_name"])
        operator = str(row["operator"])
        threshold = float(row["threshold"])
        if feature_name == "oracle_t_eff_distance":
            left_threshold = ORACLE_CENTER - threshold
            right_threshold = ORACLE_CENTER + threshold
            plt.axvline(left_threshold, color="black", linestyle="--", linewidth=1.0, label=f"t_eff <= {left_threshold:.3f}")
            plt.axvline(right_threshold, color="black", linestyle="--", linewidth=1.0, label=f"t_eff >= {right_threshold:.3f}")
        if feature_name == "IFRNet FineTuning":
            plt.axhline(threshold, color="gray", linestyle="--", linewidth=1.0, label=f"psnr {operator} {threshold:.2f}")

    plt.xlabel("oracle_fmv_t_eff_mean")
    plt.ylabel("psnr_summary")
    plt.title("Cluster-derived oracle_t_eff artifact threshold")
    plt.grid(True, alpha=0.25)
    plt.legend(markerscale=1.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_overlap_coverage(summary: pd.DataFrame, output_path: Path) -> None:
    ordered_summary = summary.copy()
    ordered_summary["percentage"] = ordered_summary["ratio"] * 100.0

    plt.figure(figsize=(9, 2.2), dpi=160)
    left = 0.0
    for overlap_label in OVERLAP_ORDER:
        row = ordered_summary[ordered_summary["group"] == overlap_label].iloc[0]
        width = float(row["percentage"])
        plt.barh(
            y=["dataset coverage"],
            width=[width],
            left=[left],
            color=OVERLAP_COLORS[overlap_label],
            label=f"{overlap_label} ({width:.1f}%)",
        )
        if width >= 3.0:
            plt.text(left + width / 2.0, 0, f"{width:.1f}%", ha="center", va="center", color="white", fontsize=9)
        left += width

    plt.xlim(0.0, 100.0)
    plt.xlabel("dataset percentage")
    plt.title("Cluster-derived artifact coverage with overlap")
    plt.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run(args: argparse.Namespace) -> None:
    dataframe = load_dataframe(args.sample_comparison_csv)
    validate_columns(dataframe, args.psnr_column)
    dataframe = dataframe.copy()
    dataframe["oracle_t_eff_distance"] = (dataframe["oracle_fmv_t_eff_mean"] - ORACLE_CENTER).abs()

    large_motion_cluster = run_kmeans_artifact_clustering(
        dataframe=dataframe,
        feature_columns=["motion_magnitude_mean", args.psnr_column],
        primary_feature_column="motion_magnitude_mean",
        psnr_column=args.psnr_column,
        cluster_count=args.motion_cluster_count,
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

    oracle_cluster = run_kmeans_artifact_clustering(
        dataframe=dataframe,
        feature_columns=["oracle_t_eff_distance", args.psnr_column],
        primary_feature_column="oracle_t_eff_distance",
        psnr_column=args.psnr_column,
        cluster_count=args.oracle_cluster_count,
        random_state=args.random_state,
    )
    oracle_tree = fit_threshold_tree(
        dataframe=dataframe,
        feature_columns=["oracle_t_eff_distance", args.psnr_column],
        cluster_labels=oracle_cluster.cluster_labels,
        artifact_cluster_id=oracle_cluster.artifact_cluster_id,
        max_depth=args.tree_max_depth,
        min_samples_leaf=args.tree_min_samples_leaf,
        random_state=args.random_state,
        artifact_label="oracle_t_eff_artifact",
    )

    overlap = build_overlap_dataframe(
        dataframe=dataframe,
        large_motion_mask=large_motion_tree.threshold_mask,
        oracle_mask=oracle_tree.threshold_mask,
    )
    overlap_summary = build_overlap_summary(overlap)
    threshold_summary = pd.DataFrame(
        build_threshold_summary_rows(
            large_motion_cluster=large_motion_cluster,
            large_motion_tree=large_motion_tree,
            oracle_cluster=oracle_cluster,
            oracle_tree=oracle_tree,
        )
    )
    threshold_rows = pd.DataFrame(large_motion_tree.threshold_rows + oracle_tree.threshold_rows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(large_motion_cluster.summary, output_dir / "large_motion_cluster_summary.csv")
    write_dataframe(oracle_cluster.summary, output_dir / "oracle_t_eff_cluster_summary.csv")
    write_dataframe(threshold_summary, output_dir / "cluster_threshold_summary.csv")
    write_dataframe(threshold_rows, output_dir / "cluster_threshold_rows.csv")
    write_dataframe(overlap, output_dir / "cluster_artifact_overlap_samples.csv")
    write_dataframe(overlap_summary, output_dir / "cluster_artifact_overlap_summary.csv")
    plot_motion_threshold_scatter(overlap, large_motion_tree.threshold_rows, output_dir / "cluster_large_motion_threshold.png")
    plot_oracle_threshold_scatter(overlap, oracle_tree.threshold_rows, output_dir / "cluster_oracle_t_eff_threshold.png")
    plot_overlap_coverage(overlap_summary, output_dir / "cluster_artifact_overlap_coverage.png")

    print(f"input={args.sample_comparison_csv}")
    print(f"output_dir={output_dir}")
    print()
    print("large_motion_cluster_rule")
    print(large_motion_tree.positive_rule_text)
    print(f"large_motion_rule_fidelity={large_motion_tree.fidelity_to_cluster:.6f}")
    print(large_motion_cluster.summary.to_string(index=False))
    print()
    print("oracle_t_eff_cluster_rule")
    print(oracle_tree.positive_rule_text)
    print(f"oracle_t_eff_rule_fidelity={oracle_tree.fidelity_to_cluster:.6f}")
    print(oracle_cluster.summary.to_string(index=False))
    print()
    print(overlap_summary.to_string(index=False))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
