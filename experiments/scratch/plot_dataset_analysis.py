from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_OUTPUT_DIR: Path = PROJECT_ROOT / "analysis_outputs" / "test_vfx_0416"
MOTION_CSV_PATH: Path = ANALYSIS_OUTPUT_DIR / "motion_by_sample.csv"
FLOW_CSV_PATH: Path = ANALYSIS_OUTPUT_DIR / "flow_approx_by_sample.csv"
PLOT_OUTPUT_DIR: Path = ANALYSIS_OUTPUT_DIR / "plots"
FLOW_METHOD_ORDER: tuple[str, ...] = ("single", "combination")


def require_file(csv_path: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    require_file(csv_path)
    return pd.read_csv(csv_path)


def build_record_order(motion_dataframe: pd.DataFrame) -> list[str]:
    ordered = (
        motion_dataframe.groupby("record_name", as_index=False)["motion_pooled_mean"]
        .mean()
        .sort_values("motion_pooled_mean", ascending=False)
    )
    return ordered["record_name"].astype(str).tolist()


def build_motion_summary(motion_dataframe: pd.DataFrame, record_order: list[str]) -> pd.DataFrame:
    summary = (
        motion_dataframe.groupby("record_name", as_index=False)[
            ["motion_pooled_mean", "motion_pooled_p95", "warp_psnr_mean_gt60"]
        ]
        .mean()
    )
    summary["record_name"] = pd.Categorical(summary["record_name"], categories=record_order, ordered=True)
    return summary.sort_values("record_name").reset_index(drop=True)


def build_flow_summary(flow_dataframe: pd.DataFrame, record_order: list[str]) -> pd.DataFrame:
    summary = (
        flow_dataframe.groupby(["record_name", "method"], as_index=False)[
            [
                "motion_pooled_mean",
                "approx_error_pooled_mean",
                "approx_error_pooled_p95",
                "warp_psnr_mean_gt60",
                "warp_psnr_mean_approx",
                "warp_psnr_delta_mean_vs_gt60",
            ]
        ]
        .mean()
    )
    summary["record_name"] = pd.Categorical(summary["record_name"], categories=record_order, ordered=True)
    summary["method"] = pd.Categorical(summary["method"], categories=list(FLOW_METHOD_ORDER), ordered=True)
    return summary.sort_values(["record_name", "method"]).reset_index(drop=True)


def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def plot_motion_by_record(motion_summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(motion_summary["record_name"].astype(str), motion_summary["motion_pooled_mean"], color="#4C72B0")
    axis.set_title("Motion Magnitude by Record")
    axis.set_xlabel("record_name")
    axis.set_ylabel("mean pooled motion magnitude")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "motion_by_record.png", dpi=220)
    plt.close(figure)


def plot_grouped_metric_by_method(
    flow_summary: pd.DataFrame,
    record_order: list[str],
    metric_name: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    x_positions = list(range(len(record_order)))
    total_width = 0.72
    bar_width = total_width / len(FLOW_METHOD_ORDER)
    colors = {"single": "#55A868", "combination": "#C44E52"}

    for method_index, method_name in enumerate(FLOW_METHOD_ORDER):
        method_dataframe = flow_summary[flow_summary["method"] == method_name].copy()
        aligned = method_dataframe.set_index("record_name").reindex(record_order).reset_index()
        offsets = [
            x_position - total_width / 2.0 + bar_width / 2.0 + method_index * bar_width
            for x_position in x_positions
        ]
        axis.bar(
            offsets,
            aligned[metric_name],
            width=bar_width,
            label=method_name,
            color=colors[method_name],
        )

    axis.set_title(title)
    axis.set_xlabel("record_name")
    axis.set_ylabel(y_label)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(record_order, rotation=45, ha="right")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_scatter(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    colors = {"single": "#55A868", "combination": "#C44E52"}

    for method_name in FLOW_METHOD_ORDER:
        method_dataframe = dataframe[dataframe["method"] == method_name]
        axis.scatter(
            method_dataframe[x_column],
            method_dataframe[y_column],
            s=18,
            alpha=0.55,
            label=method_name,
            color=colors[method_name],
        )

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    motion_dataframe = load_dataframe(MOTION_CSV_PATH)
    flow_dataframe = load_dataframe(FLOW_CSV_PATH)

    record_order = build_record_order(motion_dataframe)
    motion_summary = build_motion_summary(motion_dataframe, record_order)
    flow_summary = build_flow_summary(flow_dataframe, record_order)

    save_dataframe(motion_summary, PLOT_OUTPUT_DIR / "motion_summary_for_plot.csv")
    save_dataframe(flow_summary, PLOT_OUTPUT_DIR / "flow_summary_for_plot.csv")

    plot_motion_by_record(motion_summary, PLOT_OUTPUT_DIR)
    plot_grouped_metric_by_method(
        flow_summary=flow_summary,
        record_order=record_order,
        metric_name="approx_error_pooled_mean",
        title="Flow Approximation Error by Record",
        y_label="mean pooled flow error",
        output_path=PLOT_OUTPUT_DIR / "flow_error_by_record.png",
    )
    plot_grouped_metric_by_method(
        flow_summary=flow_summary,
        record_order=record_order,
        metric_name="warp_psnr_delta_mean_vs_gt60",
        title="Warped PSNR Delta vs 60fps Motion by Record",
        y_label="mean PSNR delta",
        output_path=PLOT_OUTPUT_DIR / "warp_psnr_delta_by_record.png",
    )
    plot_scatter(
        dataframe=flow_dataframe,
        x_column="motion_pooled_mean",
        y_column="approx_error_pooled_mean",
        title="Motion Magnitude vs Flow Approximation Error",
        x_label="pooled motion magnitude",
        y_label="pooled flow approximation error",
        output_path=PLOT_OUTPUT_DIR / "scatter_motion_vs_flow_error.png",
    )
    plot_scatter(
        dataframe=flow_dataframe,
        x_column="motion_pooled_mean",
        y_column="warp_psnr_delta_mean_vs_gt60",
        title="Motion Magnitude vs Warped PSNR Delta",
        x_label="pooled motion magnitude",
        y_label="PSNR delta vs 60fps motion warp",
        output_path=PLOT_OUTPUT_DIR / "scatter_motion_vs_warp_psnr_delta.png",
    )


if __name__ == "__main__":
    main()
