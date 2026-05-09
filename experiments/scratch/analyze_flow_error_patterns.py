from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_OUTPUT_DIR: Path = PROJECT_ROOT / "analysis_outputs" / "test_vfx_0416"
FLOW_CSV_PATH: Path = ANALYSIS_OUTPUT_DIR / "flow_approx_by_sample.csv"
OUTPUT_DIR: Path = ANALYSIS_OUTPUT_DIR / "error_pattern_analysis"
FLOW_METHOD_ORDER: tuple[str, ...] = ("single", "combination")
TOKEN_SPLIT_PATTERN: re.Pattern[str] = re.compile(r"[/_]+")
MOTION_FEATURES: tuple[str, ...] = ("motion_pooled_mean", "motion_pooled_p95")
FLOW_ERROR_FEATURES: tuple[str, ...] = (
    "approx_error_pooled_mean",
    "approx_error_pooled_p95",
)
WARPED_PSNR_FEATURES: tuple[str, ...] = (
    "warp_psnr_mean_gt60",
    "warp_psnr_mean_approx",
    "warp_psnr_delta_mean_vs_gt60_abs",
)
ANALYSIS_TARGET_FEATURES: tuple[str, ...] = FLOW_ERROR_FEATURES + WARPED_PSNR_FEATURES
CLUSTER_FEATURES: tuple[str, ...] = (
    "motion_pooled_mean",
    "motion_pooled_p95",
    "approx_error_pooled_mean",
    "approx_error_pooled_p95",
    "warp_psnr_mean_gt60",
    "warp_psnr_mean_approx",
    "warp_psnr_delta_mean_vs_gt60_abs",
)
DEFAULT_CLUSTER_COUNT: int = 4
DEFAULT_QUANTILE_COUNT: int = 5


def require_file(csv_path: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")


def load_flow_dataframe(csv_path: Path) -> pd.DataFrame:
    require_file(csv_path)
    dataframe = pd.read_csv(csv_path)
    dataframe["warp_psnr_delta_mean_vs_gt60_abs"] = dataframe["warp_psnr_delta_mean_vs_gt60"].abs()
    dataframe["record_label"] = dataframe["record_name"].astype(str).map(format_record_label)
    return dataframe


def extract_record_components(record_name: str) -> tuple[str, int, int, int] | None:
    tokens = [token for token in TOKEN_SPLIT_PATTERN.split(record_name) if token != ""]
    if len(tokens) < 4:
        return None

    prefix = tokens[0]
    numeric_tokens = [int(token) for token in tokens if token.isdigit()]
    if len(numeric_tokens) < 3:
        return None

    group_a = numeric_tokens[0]
    group_b = numeric_tokens[1]
    sub_index = numeric_tokens[-2] if tokens[-1].isdigit() and len(numeric_tokens) >= 2 else numeric_tokens[-1]
    return prefix, group_a, group_b, sub_index


def format_record_label(record_name: str) -> str:
    components = extract_record_components(record_name)
    if components is not None:
        prefix, group_a, group_b, sub_index = components
        return f"{prefix}_{group_a}_{group_b}_{sub_index}"

    return record_name


def build_label_sort_key(label: str) -> tuple[int, ...] | tuple[str]:
    parts = label.split("_")
    numeric_parts: list[int] = []

    for part in parts:
        if part.isdigit():
            numeric_parts.append(int(part))
            continue

        if len(numeric_parts) == 0:
            return (label,)

    return tuple(numeric_parts)


def build_record_sort_key(record_name: str) -> tuple[object, ...]:
    components = extract_record_components(record_name)
    if components is not None:
        prefix, group_a, group_b, sub_index = components
        return (prefix, group_a, group_b, sub_index)

    return (format_record_label(record_name),)


def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def build_correlation_rows(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    correlation_rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all", dataframe)]
    scopes.extend((method_name, dataframe[dataframe["method"] == method_name].copy()) for method_name in FLOW_METHOD_ORDER)

    for scope_name, scope_dataframe in scopes:
        if len(scope_dataframe) < 2:
            continue

        for motion_feature in MOTION_FEATURES:
            for target_feature in ANALYSIS_TARGET_FEATURES:
                pearson_value = float(scope_dataframe[motion_feature].corr(scope_dataframe[target_feature], method="pearson"))
                spearman_value = float(scope_dataframe[motion_feature].corr(scope_dataframe[target_feature], method="spearman"))
                correlation_rows.append(
                    {
                        "scope": scope_name,
                        "motion_feature": motion_feature,
                        "target_feature": target_feature,
                        "pearson_corr": pearson_value,
                        "spearman_corr": spearman_value,
                        "sample_count": int(len(scope_dataframe)),
                    }
                )

    return correlation_rows


def build_motion_quantile_rows(dataframe: pd.DataFrame, quantile_count: int) -> list[dict[str, object]]:
    quantile_rows: list[dict[str, object]] = []

    for method_name in FLOW_METHOD_ORDER:
        method_dataframe = dataframe[dataframe["method"] == method_name].copy()
        if len(method_dataframe) == 0:
            continue

        quantile_series = pd.qcut(
            method_dataframe["motion_pooled_mean"],
            q=min(quantile_count, len(method_dataframe)),
            labels=False,
            duplicates="drop",
        )
        method_dataframe["motion_quantile"] = quantile_series

        grouped = (
            method_dataframe.groupby("motion_quantile", as_index=False)[
                [
                    "motion_pooled_mean",
                    "motion_pooled_p95",
                    "approx_error_pooled_mean",
                    "approx_error_pooled_p95",
                    "warp_psnr_mean_gt60",
                    "warp_psnr_mean_approx",
                    "warp_psnr_delta_mean_vs_gt60_abs",
                ]
            ]
            .mean()
        )
        counts = method_dataframe.groupby("motion_quantile", as_index=False).size().rename(columns={"size": "sample_count"})
        grouped = grouped.merge(counts, on="motion_quantile", how="left")

        for _, row in grouped.iterrows():
            quantile_rows.append(
                {
                    "method": method_name,
                    "motion_quantile": int(row["motion_quantile"]),
                    "sample_count": int(row["sample_count"]),
                    "motion_pooled_mean": float(row["motion_pooled_mean"]),
                    "motion_pooled_p95": float(row["motion_pooled_p95"]),
                    "approx_error_pooled_mean": float(row["approx_error_pooled_mean"]),
                    "approx_error_pooled_p95": float(row["approx_error_pooled_p95"]),
                    "warp_psnr_mean_gt60": float(row["warp_psnr_mean_gt60"]),
                    "warp_psnr_mean_approx": float(row["warp_psnr_mean_approx"]),
                    "warp_psnr_delta_mean_vs_gt60_abs": float(row["warp_psnr_delta_mean_vs_gt60_abs"]),
                }
            )

    return quantile_rows


def build_record_method_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    summary = (
        dataframe.groupby(["record_name", "record_label", "method"], as_index=False)[list(CLUSTER_FEATURES)]
        .mean()
    )
    counts = dataframe.groupby(["record_name", "record_label", "method"], as_index=False).size().rename(columns={"size": "sample_count"})
    summary = summary.merge(counts, on=["record_name", "record_label", "method"], how="left")
    summary["record_sort_key"] = summary["record_name"].map(build_record_sort_key)
    summary = summary.sort_values(["method", "record_sort_key"]).reset_index(drop=True)
    return summary.drop(columns=["record_sort_key"])


def run_record_clustering(summary_dataframe: pd.DataFrame, cluster_count: int) -> pd.DataFrame:
    clustered_frames: list[pd.DataFrame] = []

    for method_name in FLOW_METHOD_ORDER:
        method_dataframe = summary_dataframe[summary_dataframe["method"] == method_name].copy()
        if len(method_dataframe) == 0:
            continue

        resolved_cluster_count = min(cluster_count, len(method_dataframe))
        if resolved_cluster_count <= 1:
            method_dataframe["cluster_id"] = 0
            clustered_frames.append(method_dataframe)
            continue

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(method_dataframe[list(CLUSTER_FEATURES)])
        cluster_model = KMeans(n_clusters=resolved_cluster_count, random_state=1234, n_init=20)
        method_dataframe["cluster_id"] = cluster_model.fit_predict(scaled_features)
        clustered_frames.append(method_dataframe)

    if len(clustered_frames) == 0:
        return summary_dataframe.assign(cluster_id=-1)

    return pd.concat(clustered_frames, ignore_index=True)


def build_cluster_summary(clustered_dataframe: pd.DataFrame) -> pd.DataFrame:
    summary = (
        clustered_dataframe.groupby(["method", "cluster_id"], as_index=False)[["sample_count", *CLUSTER_FEATURES]]
        .mean()
    )
    label_rows: list[dict[str, object]] = []

    for (method_name, cluster_id), cluster_dataframe in clustered_dataframe.groupby(["method", "cluster_id"], sort=False):
        top_labels = cluster_dataframe.sort_values("approx_error_pooled_mean", ascending=False)["record_label"].head(5).tolist()
        label_rows.append(
            {
                "method": method_name,
                "cluster_id": int(cluster_id),
                "top_record_labels": ";".join(top_labels),
            }
        )

    label_dataframe = pd.DataFrame(label_rows)
    return summary.merge(label_dataframe, on=["method", "cluster_id"], how="left")


def build_high_error_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []

    for method_name in FLOW_METHOD_ORDER:
        method_dataframe = dataframe[dataframe["method"] == method_name].copy()
        if len(method_dataframe) == 0:
            continue

        error_threshold = float(method_dataframe["approx_error_pooled_mean"].quantile(0.75))
        low_threshold = float(method_dataframe["approx_error_pooled_mean"].quantile(0.25))
        high_error_dataframe = method_dataframe[method_dataframe["approx_error_pooled_mean"] >= error_threshold]
        low_error_dataframe = method_dataframe[method_dataframe["approx_error_pooled_mean"] <= low_threshold]

        summary_rows.append(
            {
                "method": method_name,
                "group_name": "high_error_top25pct",
                "sample_count": int(len(high_error_dataframe)),
                "motion_pooled_mean": float(high_error_dataframe["motion_pooled_mean"].mean()),
                "motion_pooled_p95": float(high_error_dataframe["motion_pooled_p95"].mean()),
                "approx_error_pooled_mean": float(high_error_dataframe["approx_error_pooled_mean"].mean()),
                "approx_error_pooled_p95": float(high_error_dataframe["approx_error_pooled_p95"].mean()),
                "warp_psnr_mean_gt60": float(high_error_dataframe["warp_psnr_mean_gt60"].mean()),
                "warp_psnr_mean_approx": float(high_error_dataframe["warp_psnr_mean_approx"].mean()),
                "warp_psnr_delta_mean_vs_gt60_abs": float(high_error_dataframe["warp_psnr_delta_mean_vs_gt60_abs"].mean()),
            }
        )
        summary_rows.append(
            {
                "method": method_name,
                "group_name": "low_error_bottom25pct",
                "sample_count": int(len(low_error_dataframe)),
                "motion_pooled_mean": float(low_error_dataframe["motion_pooled_mean"].mean()),
                "motion_pooled_p95": float(low_error_dataframe["motion_pooled_p95"].mean()),
                "approx_error_pooled_mean": float(low_error_dataframe["approx_error_pooled_mean"].mean()),
                "approx_error_pooled_p95": float(low_error_dataframe["approx_error_pooled_p95"].mean()),
                "warp_psnr_mean_gt60": float(low_error_dataframe["warp_psnr_mean_gt60"].mean()),
                "warp_psnr_mean_approx": float(low_error_dataframe["warp_psnr_mean_approx"].mean()),
                "warp_psnr_delta_mean_vs_gt60_abs": float(low_error_dataframe["warp_psnr_delta_mean_vs_gt60_abs"].mean()),
            }
        )

    return pd.DataFrame(summary_rows)


def plot_motion_error_scatter(dataframe: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, len(FLOW_METHOD_ORDER), figsize=(12, 5), sharex=True, sharey=True)

    for axis, method_name in zip(axes, FLOW_METHOD_ORDER):
        method_dataframe = dataframe[dataframe["method"] == method_name]
        axis.scatter(
            method_dataframe["motion_pooled_mean"],
            method_dataframe["approx_error_pooled_mean"],
            s=16,
            alpha=0.45,
            color="#4C72B0",
        )
        axis.set_title(method_name)
        axis.set_xlabel("motion_pooled_mean")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("approx_error_pooled_mean")
    figure.suptitle("Motion Magnitude vs Flow Approximation Error")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_quantile_trend(quantile_dataframe: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, len(FLOW_METHOD_ORDER), figsize=(12, 5), sharey=True)
    colors = {"single": "#55A868", "combination": "#C44E52"}

    for axis, method_name in zip(axes, FLOW_METHOD_ORDER):
        method_dataframe = quantile_dataframe[quantile_dataframe["method"] == method_name]
        axis.plot(
            method_dataframe["motion_quantile"],
            method_dataframe["approx_error_pooled_mean"],
            marker="o",
            linewidth=1.8,
            color=colors[method_name],
        )
        axis.set_title(method_name)
        axis.set_xlabel("motion quantile")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("mean pooled approx error")
    figure.suptitle("Flow Error Trend Across Motion Quantiles")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_quantile_target_trend(
    quantile_dataframe: pd.DataFrame,
    target_column: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(FLOW_METHOD_ORDER), figsize=(12, 5), sharey=True)
    colors = {"single": "#55A868", "combination": "#C44E52"}

    for axis, method_name in zip(axes, FLOW_METHOD_ORDER):
        method_dataframe = quantile_dataframe[quantile_dataframe["method"] == method_name]
        axis.plot(
            method_dataframe["motion_quantile"],
            method_dataframe[target_column],
            marker="o",
            linewidth=1.8,
            color=colors[method_name],
        )
        axis.set_title(method_name)
        axis.set_xlabel("motion quantile")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel(y_label)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_clustered_record_scatter(clustered_dataframe: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, len(FLOW_METHOD_ORDER), figsize=(12, 5), sharex=True, sharey=True)
    cluster_color_map = plt.get_cmap("tab10")
    legend_handles: list[Line2D] = []
    seen_cluster_ids: set[int] = set()

    for axis, method_name in zip(axes, FLOW_METHOD_ORDER):
        method_dataframe = clustered_dataframe[clustered_dataframe["method"] == method_name]
        cluster_ids = method_dataframe["cluster_id"].astype(int)
        axis.scatter(
            method_dataframe["motion_pooled_mean"],
            method_dataframe["approx_error_pooled_mean"],
            c=method_dataframe["cluster_id"],
            cmap="tab10",
            s=42,
            alpha=0.85,
        )
        axis.set_title(method_name)
        axis.set_xlabel("motion_pooled_mean")
        axis.grid(True, alpha=0.3)

        for _, row in method_dataframe.iterrows():
            axis.annotate(
                str(row["record_label"]),
                (row["motion_pooled_mean"], row["approx_error_pooled_mean"]),
                fontsize=7,
                alpha=0.8,
            )

        for cluster_id in sorted(cluster_ids.unique().tolist()):
            if cluster_id in seen_cluster_ids:
                continue

            seen_cluster_ids.add(cluster_id)
            color_index = int(cluster_id) % int(cluster_color_map.N)
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"cluster {cluster_id}",
                    markerfacecolor=cluster_color_map.colors[color_index],
                    markersize=8,
                )
            )

    axes[0].set_ylabel("approx_error_pooled_mean")
    figure.suptitle("Record-Level Clusters: Motion vs Flow Error")
    figure.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, title="cluster_id")
    figure.tight_layout(rect=(0.0, 0.0, 0.93, 1.0))
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_motion_vs_target_scatter(
    dataframe: pd.DataFrame,
    target_column: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(FLOW_METHOD_ORDER), figsize=(12, 5), sharex=True, sharey=True)

    for axis, method_name in zip(axes, FLOW_METHOD_ORDER):
        method_dataframe = dataframe[dataframe["method"] == method_name]
        axis.scatter(
            method_dataframe["motion_pooled_mean"],
            method_dataframe[target_column],
            s=16,
            alpha=0.45,
            color="#4C72B0",
        )
        axis.set_title(method_name)
        axis.set_xlabel("motion_pooled_mean")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel(y_label)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flow_dataframe = load_flow_dataframe(FLOW_CSV_PATH)
    correlation_dataframe = pd.DataFrame(build_correlation_rows(flow_dataframe))
    quantile_dataframe = pd.DataFrame(build_motion_quantile_rows(flow_dataframe, DEFAULT_QUANTILE_COUNT))
    record_method_summary = build_record_method_summary(flow_dataframe)
    clustered_record_summary = run_record_clustering(record_method_summary, DEFAULT_CLUSTER_COUNT)
    cluster_summary = build_cluster_summary(clustered_record_summary)
    high_error_summary = build_high_error_summary(flow_dataframe)

    save_dataframe(correlation_dataframe, OUTPUT_DIR / "motion_error_correlation.csv")
    save_dataframe(quantile_dataframe, OUTPUT_DIR / "motion_quantile_trend.csv")
    save_dataframe(record_method_summary, OUTPUT_DIR / "record_method_summary.csv")
    save_dataframe(clustered_record_summary, OUTPUT_DIR / "record_method_clusters.csv")
    save_dataframe(cluster_summary, OUTPUT_DIR / "cluster_summary.csv")
    save_dataframe(high_error_summary, OUTPUT_DIR / "high_vs_low_error_summary.csv")

    plot_motion_error_scatter(flow_dataframe, OUTPUT_DIR / "scatter_motion_vs_error_by_method.png")
    plot_quantile_trend(quantile_dataframe, OUTPUT_DIR / "motion_quantile_error_trend.png")
    plot_motion_vs_target_scatter(
        flow_dataframe,
        target_column="warp_psnr_mean_gt60",
        title="Motion Magnitude vs Ground-Truth Warped PSNR",
        y_label="warp_psnr_mean_gt60",
        output_path=OUTPUT_DIR / "scatter_motion_vs_warp_psnr_gt60.png",
    )
    plot_motion_vs_target_scatter(
        flow_dataframe,
        target_column="warp_psnr_mean_approx",
        title="Motion Magnitude vs Approximated-Flow Warped PSNR",
        y_label="warp_psnr_mean_approx",
        output_path=OUTPUT_DIR / "scatter_motion_vs_warp_psnr_approx.png",
    )
    plot_motion_vs_target_scatter(
        flow_dataframe,
        target_column="warp_psnr_delta_mean_vs_gt60_abs",
        title="Motion Magnitude vs Absolute Warped PSNR Delta",
        y_label="warp_psnr_delta_mean_vs_gt60_abs",
        output_path=OUTPUT_DIR / "scatter_motion_vs_warp_psnr_delta_abs.png",
    )
    plot_quantile_target_trend(
        quantile_dataframe,
        target_column="warp_psnr_mean_gt60",
        title="Ground-Truth Warped PSNR Across Motion Quantiles",
        y_label="mean warp_psnr_mean_gt60",
        output_path=OUTPUT_DIR / "motion_quantile_warp_psnr_gt60_trend.png",
    )
    plot_quantile_target_trend(
        quantile_dataframe,
        target_column="warp_psnr_mean_approx",
        title="Approximated-Flow Warped PSNR Across Motion Quantiles",
        y_label="mean warp_psnr_mean_approx",
        output_path=OUTPUT_DIR / "motion_quantile_warp_psnr_approx_trend.png",
    )
    plot_quantile_target_trend(
        quantile_dataframe,
        target_column="warp_psnr_delta_mean_vs_gt60_abs",
        title="Absolute Warped PSNR Delta Across Motion Quantiles",
        y_label="mean abs warped PSNR delta",
        output_path=OUTPUT_DIR / "motion_quantile_warp_psnr_delta_abs_trend.png",
    )
    plot_clustered_record_scatter(clustered_record_summary, OUTPUT_DIR / "record_level_cluster_scatter.png")


if __name__ == "__main__":
    main()
