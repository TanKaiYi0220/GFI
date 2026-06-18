from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
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

from experiments.scratch.analyze_vfx_oracle_cleaning import DEFAULT_RAW_SEQUENCE_ROOT
from experiments.scratch.analyze_vfx_oracle_cleaning import KEY_COLUMNS
from experiments.scratch.analyze_vfx_oracle_cleaning import MOTION_COLUMN
from experiments.scratch.analyze_vfx_oracle_cleaning import ORACLE_COLUMN
from experiments.scratch.analyze_vfx_oracle_cleaning import ORACLE_DISTANCE_COLUMN
from experiments.scratch.analyze_vfx_oracle_cleaning import PSNR_COLUMN
from experiments.scratch.analyze_vfx_oracle_cleaning import build_bucketed_dataframe
from experiments.scratch.analyze_vfx_oracle_cleaning import build_complete_bucket_summary
from experiments.scratch.analyze_vfx_oracle_cleaning import build_joined_dataframe
from experiments.scratch.analyze_vfx_oracle_cleaning import load_metrics
from experiments.scratch.analyze_vfx_oracle_cleaning import load_raw_sequence_stats
from experiments.scratch.analyze_vfx_oracle_cleaning import write_distribution_dashboard

DEFAULT_BASELINE_METRICS_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\DGX_inference_outputs\IFRNet_FineTuning_0611\metrics.csv",
)
DEFAULT_CANDIDATE_METRICS_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\DGX_inference_outputs\IFRNet_Residual_FlowApprox_1_Layer_TwoStage_Splat_4Direction_MaskedArea_0611\metrics.csv",
)
DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\dgx_vfx_0416_oracle_model_comparison\ifrnet_finetuning_vs_flowapprox_maskedarea_0611",
)
DEFAULT_ORACLE_CENTER: float = 0.5
DEFAULT_ORACLE_WINDOW_RADIUS: float = 0.025
DEFAULT_MOTION_THRESHOLD_QUANTILE: float = 0.9
DEFAULT_PSNR_THRESHOLD_QUANTILE: float = 0.15
DEFAULT_LOW_PSNR_THRESHOLD: float = 25.0
MODEL_ORDER: tuple[str, ...] = ("IFRNet FineTuning", "FlowApprox MaskedArea")
MODEL_COLORS: dict[str, str] = {
    "IFRNet FineTuning": "#4c78a8",
    "FlowApprox MaskedArea": "#f58518",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    metrics_csv: Path


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare oracle effective-time cleaning across two DGX VFX model outputs."
    )
    parser.add_argument("--baseline-metrics-csv", type=Path, default=DEFAULT_BASELINE_METRICS_CSV)
    parser.add_argument("--candidate-metrics-csv", type=Path, default=DEFAULT_CANDIDATE_METRICS_CSV)
    parser.add_argument("--baseline-name", type=str, default=MODEL_ORDER[0])
    parser.add_argument("--candidate-name", type=str, default=MODEL_ORDER[1])
    parser.add_argument("--raw-sequence-root", type=Path, default=DEFAULT_RAW_SEQUENCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oracle-center", type=float, default=DEFAULT_ORACLE_CENTER)
    parser.add_argument("--oracle-window-radius", type=float, default=DEFAULT_ORACLE_WINDOW_RADIUS)
    parser.add_argument("--motion-threshold-quantile", type=float, default=DEFAULT_MOTION_THRESHOLD_QUANTILE)
    parser.add_argument("--psnr-threshold-quantile", type=float, default=DEFAULT_PSNR_THRESHOLD_QUANTILE)
    parser.add_argument("--low-psnr-threshold", type=float, default=DEFAULT_LOW_PSNR_THRESHOLD)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.oracle_center <= 0.0 or args.oracle_center >= 1.0:
        raise ValueError(f"oracle-center must be in (0, 1): oracle_center={args.oracle_center}")
    if args.oracle_window_radius <= 0.0:
        raise ValueError(f"oracle-window-radius must be positive: radius={args.oracle_window_radius}")
    if args.motion_threshold_quantile <= 0.0 or args.motion_threshold_quantile >= 1.0:
        raise ValueError(f"motion-threshold-quantile must be in (0, 1): value={args.motion_threshold_quantile}")
    if args.psnr_threshold_quantile <= 0.0 or args.psnr_threshold_quantile >= 1.0:
        raise ValueError(f"psnr-threshold-quantile must be in (0, 1): value={args.psnr_threshold_quantile}")
    if args.low_psnr_threshold <= 0.0:
        raise ValueError(f"low-psnr-threshold must be positive: value={args.low_psnr_threshold}")


def build_model_dataframe(
    model: ModelSpec,
    raw_stats: pd.DataFrame,
    oracle_center: float,
    oracle_window_radius: float,
) -> pd.DataFrame:
    metrics = load_metrics(model.metrics_csv)
    joined = build_joined_dataframe(metrics=metrics, raw_stats=raw_stats, oracle_center=oracle_center)
    joined["model_name"] = model.name
    joined["clean_oracle_t_eff"] = joined[ORACLE_DISTANCE_COLUMN] <= oracle_window_radius
    return joined


def validate_matching_samples(dataframes: Sequence[pd.DataFrame]) -> None:
    if len(dataframes) < 2:
        return
    baseline_keys = dataframes[0][list(KEY_COLUMNS)].sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    for dataframe in dataframes[1:]:
        keys = dataframe[list(KEY_COLUMNS)].sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        if not baseline_keys.equals(keys):
            raise ValueError("Model metrics do not contain identical sample keys.")


def summarize_models(dataframe: pd.DataFrame, low_psnr_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, model_dataframe in dataframe.groupby("model_name", sort=False):
        all_samples = int(len(model_dataframe))
        all_mean = float(model_dataframe[PSNR_COLUMN].mean())
        all_median = float(model_dataframe[PSNR_COLUMN].median())
        all_low_psnr_samples = int((model_dataframe[PSNR_COLUMN] < low_psnr_threshold).sum())
        rows.append(
            {
                "model_name": str(model_name),
                "cleaning_name": "all",
                "samples": all_samples,
                "retention_ratio": 1.0,
                "mean_psnr": all_mean,
                "median_psnr": all_median,
                "mean_psnr_gain": 0.0,
                "median_psnr_gain": 0.0,
                "low_psnr_threshold": low_psnr_threshold,
                "low_psnr_samples": all_low_psnr_samples,
                "low_psnr_ratio": float(all_low_psnr_samples / all_samples),
            }
        )

        clean = model_dataframe[model_dataframe["clean_oracle_t_eff"]].copy()
        clean_low_psnr_samples = int((clean[PSNR_COLUMN] < low_psnr_threshold).sum())
        rows.append(
            {
                "model_name": str(model_name),
                "cleaning_name": "oracle_t_eff",
                "samples": int(len(clean)),
                "retention_ratio": float(len(clean) / all_samples),
                "mean_psnr": float(clean[PSNR_COLUMN].mean()),
                "median_psnr": float(clean[PSNR_COLUMN].median()),
                "mean_psnr_gain": float(clean[PSNR_COLUMN].mean() - all_mean),
                "median_psnr_gain": float(clean[PSNR_COLUMN].median() - all_median),
                "low_psnr_threshold": low_psnr_threshold,
                "low_psnr_samples": clean_low_psnr_samples,
                "low_psnr_ratio": float(clean_low_psnr_samples / len(clean)),
            }
        )
    return pd.DataFrame(rows)


def build_pairwise_delta(dataframes: Sequence[pd.DataFrame], baseline_name: str, candidate_name: str) -> pd.DataFrame:
    selected_columns = list(KEY_COLUMNS) + [
        "record_name",
        "mode",
        "img0",
        "img2",
        MOTION_COLUMN,
        ORACLE_COLUMN,
        ORACLE_DISTANCE_COLUMN,
        "clean_oracle_t_eff",
        PSNR_COLUMN,
    ]
    model_frames: list[pd.DataFrame] = []
    for dataframe in dataframes:
        model_name = str(dataframe["model_name"].iloc[0])
        selected = dataframe[[column for column in selected_columns if column in dataframe.columns]].copy()
        selected = selected.rename(columns={PSNR_COLUMN: f"{model_name}_psnr"})
        selected["model_name"] = model_name
        model_frames.append(selected)

    baseline = model_frames[0].drop(columns=["model_name"])
    candidate = model_frames[1].drop(
        columns=[
            column
            for column in ["model_name", "record_name", "mode", "img0", "img2", MOTION_COLUMN, ORACLE_COLUMN, ORACLE_DISTANCE_COLUMN, "clean_oracle_t_eff"]
            if column in model_frames[1].columns
        ]
    )
    pairwise = baseline.merge(candidate, on=list(KEY_COLUMNS), how="inner")
    pairwise["delta_psnr_candidate_minus_baseline"] = pairwise[f"{candidate_name}_psnr"] - pairwise[f"{baseline_name}_psnr"]
    return pairwise


def write_model_comparison_dashboard(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    baseline_name: str,
    candidate_name: str,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "oracle_model_comparison_dashboard.png"
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=160)
    figure.suptitle("DGX VFX oracle_fmv_t_eff cleaning across models", fontsize=16)

    axis = axes[0, 0]
    x_positions = np.arange(len(MODEL_ORDER))
    width = 0.34
    all_rows = summary[summary["cleaning_name"] == "all"].set_index("model_name").loc[list(MODEL_ORDER)].reset_index()
    clean_rows = summary[summary["cleaning_name"] == "oracle_t_eff"].set_index("model_name").loc[list(MODEL_ORDER)].reset_index()
    axis.bar(x_positions - width / 2.0, all_rows["mean_psnr"], width=width, color=[MODEL_COLORS[name] for name in all_rows["model_name"]], alpha=0.65, label="all")
    axis.bar(x_positions + width / 2.0, clean_rows["mean_psnr"], width=width, color="#2ca02c", alpha=0.9, label="oracle clean")
    axis.set_xticks(x_positions, MODEL_ORDER)
    axis.set_ylabel("mean psnr (dB)")
    axis.set_title("All vs oracle-clean mean PSNR")
    axis.set_ylim(float(summary["mean_psnr"].min()) - 0.35, float(summary["mean_psnr"].max()) + 0.35)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    for index, row in enumerate(all_rows.itertuples(index=False)):
        axis.text(index - width / 2.0, float(row.mean_psnr) + 0.04, f"{float(row.mean_psnr):.2f}", ha="center", va="bottom", fontsize=8)
    for index, row in enumerate(clean_rows.itertuples(index=False)):
        axis.text(index + width / 2.0, float(row.mean_psnr) + 0.04, f"{float(row.mean_psnr):.2f}", ha="center", va="bottom", fontsize=8)

    axis = axes[0, 1]
    axis.bar(clean_rows["model_name"], clean_rows["mean_psnr_gain"], color=[MODEL_COLORS[name] for name in clean_rows["model_name"]], alpha=0.9)
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set_title("Mean PSNR gain from oracle cleaning")
    axis.set_ylabel("gain (dB)")
    axis.grid(True, axis="y", alpha=0.25)
    for index, row in enumerate(clean_rows.itertuples(index=False)):
        axis.text(index, float(row.mean_psnr_gain) + 0.03, f"{float(row.mean_psnr_gain):+.2f}", ha="center", va="bottom", fontsize=9)

    axis = axes[1, 0]
    all_delta = pairwise["delta_psnr_candidate_minus_baseline"]
    clean_delta = pairwise.loc[pairwise["clean_oracle_t_eff"], "delta_psnr_candidate_minus_baseline"]
    bins = np.linspace(float(pairwise["delta_psnr_candidate_minus_baseline"].quantile(0.01)), float(pairwise["delta_psnr_candidate_minus_baseline"].quantile(0.99)), 60)
    axis.hist(all_delta, bins=bins, alpha=0.35, color="#777777", label=f"all mean={all_delta.mean():+.2f}")
    axis.hist(clean_delta, bins=bins, alpha=0.55, color="#2ca02c", label=f"oracle clean mean={clean_delta.mean():+.2f}")
    axis.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axis.set_title(f"{candidate_name} - {baseline_name} PSNR")
    axis.set_xlabel("delta psnr (dB)")
    axis.set_ylabel("samples")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()

    axis = axes[1, 1]
    x_positions = np.arange(len(MODEL_ORDER))
    axis.bar(x_positions - width / 2.0, all_rows["low_psnr_ratio"] * 100.0, width=width, color="#d62728", alpha=0.45, label="all")
    axis.bar(x_positions + width / 2.0, clean_rows["low_psnr_ratio"] * 100.0, width=width, color="#d62728", alpha=0.85, label="oracle clean")
    axis.set_xticks(x_positions, MODEL_ORDER)
    axis.set_title("Low-PSNR sample ratio")
    axis.set_ylabel("samples below threshold (%)")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    for index, row in enumerate(clean_rows.itertuples(index=False)):
        axis.text(
            index + width / 2.0,
            float(row.low_psnr_ratio) * 100.0 + 0.08,
            f"{int(row.low_psnr_samples)}\n{float(row.low_psnr_ratio) * 100.0:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_delta_motion_dashboard(pairwise: pd.DataFrame, baseline_name: str, candidate_name: str, output_dir: Path) -> Path:
    output_path = output_dir / "oracle_clean_delta_psnr_motion_dashboard.png"
    clean = pairwise[pairwise["clean_oracle_t_eff"]].copy()
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=160)
    figure.suptitle(f"Oracle-clean paired comparison: {candidate_name} - {baseline_name}", fontsize=14)

    axis = axes[0]
    axis.scatter(
        clean[f"{baseline_name}_psnr"],
        clean[f"{candidate_name}_psnr"],
        s=9,
        alpha=0.28,
        edgecolors="none",
        color="#4c78a8",
    )
    axis_min = float(min(clean[f"{baseline_name}_psnr"].quantile(0.005), clean[f"{candidate_name}_psnr"].quantile(0.005)))
    axis_max = float(max(clean[f"{baseline_name}_psnr"].quantile(0.995), clean[f"{candidate_name}_psnr"].quantile(0.995)))
    axis.plot([axis_min, axis_max], [axis_min, axis_max], color="#333333", linestyle="--", linewidth=1.0)
    axis.set_xlim(axis_min, axis_max)
    axis.set_ylim(axis_min, axis_max)
    axis.set_xlabel(f"{baseline_name} psnr")
    axis.set_ylabel(f"{candidate_name} psnr")
    axis.set_title("Paired sample PSNR")
    axis.grid(True, alpha=0.25)

    axis = axes[1]
    axis.scatter(
        clean[MOTION_COLUMN],
        clean["delta_psnr_candidate_minus_baseline"],
        s=9,
        alpha=0.28,
        edgecolors="none",
        color="#f58518",
    )
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axis.set_xlabel(MOTION_COLUMN)
    axis.set_ylabel("candidate - baseline PSNR (dB)")
    axis.set_title("Delta PSNR by motion")
    axis.grid(True, alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_low_psnr_outputs(
    dataframe: pd.DataFrame,
    low_psnr_threshold: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    low = dataframe[dataframe[PSNR_COLUMN] < low_psnr_threshold].copy()
    selected_columns = [
        "model_name",
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
        "image_0_path",
        "image_1_path",
        "image_gt_path",
        "image_pred_path",
    ]
    available_columns = [column for column in selected_columns if column in low.columns]
    rows_path = output_dir / f"low_psnr_below_{low_psnr_threshold:g}_after_oracle_cleaning_by_model.csv"
    low[available_columns].sort_values(["model_name", PSNR_COLUMN, "record", "mode", "frame_range"]).to_csv(rows_path, index=False)

    summary = (
        low.groupby("model_name")
        .agg(low_psnr_samples=(PSNR_COLUMN, "size"), min_psnr=(PSNR_COLUMN, "min"), mean_psnr=(PSNR_COLUMN, "mean"))
        .reset_index()
    )
    clean_counts = dataframe.groupby("model_name").size().rename("clean_samples").reset_index()
    summary = summary.merge(clean_counts, on="model_name", how="right").fillna({"low_psnr_samples": 0})
    summary["low_psnr_threshold"] = low_psnr_threshold
    summary["low_psnr_ratio_of_clean"] = summary["low_psnr_samples"] / summary["clean_samples"]
    summary_path = output_dir / f"low_psnr_below_{low_psnr_threshold:g}_after_oracle_cleaning_summary_by_model.csv"
    summary.to_csv(summary_path, index=False)
    return rows_path, summary_path


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_stats = load_raw_sequence_stats(args.raw_sequence_root)
    models = [
        ModelSpec(name=str(args.baseline_name), metrics_csv=args.baseline_metrics_csv),
        ModelSpec(name=str(args.candidate_name), metrics_csv=args.candidate_metrics_csv),
    ]
    joined_frames = [
        build_model_dataframe(
            model=model,
            raw_stats=raw_stats,
            oracle_center=float(args.oracle_center),
            oracle_window_radius=float(args.oracle_window_radius),
        )
        for model in models
    ]
    validate_matching_samples(joined_frames)
    joined = pd.concat(joined_frames, ignore_index=True)
    clean = joined[joined["clean_oracle_t_eff"]].copy()
    summary = summarize_models(dataframe=joined, low_psnr_threshold=float(args.low_psnr_threshold))
    pairwise = build_pairwise_delta(
        dataframes=joined_frames,
        baseline_name=str(args.baseline_name),
        candidate_name=str(args.candidate_name),
    )

    baseline_all = joined_frames[0]
    motion_threshold = float(baseline_all[MOTION_COLUMN].quantile(float(args.motion_threshold_quantile)))
    psnr_threshold = float(baseline_all[PSNR_COLUMN].quantile(float(args.psnr_threshold_quantile)))
    threshold_rows = pd.DataFrame(
        [
            {
                "motion_column": MOTION_COLUMN,
                "motion_threshold_quantile": float(args.motion_threshold_quantile),
                "motion_threshold": motion_threshold,
                "psnr_column": PSNR_COLUMN,
                "psnr_threshold_quantile": float(args.psnr_threshold_quantile),
                "psnr_threshold": psnr_threshold,
                "oracle_center": float(args.oracle_center),
                "oracle_window_radius": float(args.oracle_window_radius),
                "oracle_lower": float(args.oracle_center) - float(args.oracle_window_radius),
                "oracle_upper": float(args.oracle_center) + float(args.oracle_window_radius),
            }
        ]
    )

    joined.to_csv(output_dir / "vfx_oracle_model_sample_comparison.csv", index=False)
    clean.to_csv(output_dir / "vfx_oracle_clean_samples_by_model.csv", index=False)
    summary.to_csv(output_dir / "vfx_oracle_model_summary.csv", index=False)
    pairwise.to_csv(output_dir / "vfx_oracle_pairwise_delta_psnr.csv", index=False)
    threshold_rows.to_csv(output_dir / "motion_psnr_thresholds.csv", index=False)

    dashboard_path = write_model_comparison_dashboard(
        summary=summary,
        pairwise=pairwise,
        baseline_name=str(args.baseline_name),
        candidate_name=str(args.candidate_name),
        output_dir=output_dir,
    )
    delta_dashboard_path = write_delta_motion_dashboard(
        pairwise=pairwise,
        baseline_name=str(args.baseline_name),
        candidate_name=str(args.candidate_name),
        output_dir=output_dir,
    )

    bucket_summaries: list[pd.DataFrame] = []
    distribution_paths: list[Path] = []
    for model_name, model_clean in clean.groupby("model_name", sort=False):
        bucketed = build_bucketed_dataframe(model_clean, motion_threshold=motion_threshold, psnr_threshold=psnr_threshold)
        bucket_summary = build_complete_bucket_summary(bucketed)
        bucket_summary["model_name"] = str(model_name)
        bucket_summaries.append(bucket_summary)
        safe_model_name = str(model_name).lower().replace(" ", "_")
        bucketed.to_csv(output_dir / f"{safe_model_name}_oracle_clean_bucketed.csv", index=False)
        output_path = output_dir / f"{safe_model_name}_oracle_clean_motion_distribution.png"
        write_distribution_dashboard(
            bucketed=bucketed,
            summary=bucket_summary,
            title=(
                f"DGX VFX {model_name} oracle-clean motion/PSNR bucket distribution\n"
                f"retained {len(bucketed)}/{len(joined_frames[0])} ({len(bucketed) / len(joined_frames[0]) * 100.0:.1f}%), "
                f"motion >= {motion_threshold:.2f}, psnr < {psnr_threshold:.2f}"
            ),
            output_path=output_path,
        )
        distribution_paths.append(output_path)
    pd.concat(bucket_summaries, ignore_index=True).to_csv(output_dir / "oracle_clean_bucket_summary_by_model.csv", index=False)

    low_rows_path, low_summary_path = write_low_psnr_outputs(
        dataframe=clean,
        low_psnr_threshold=float(args.low_psnr_threshold),
        output_dir=output_dir,
    )

    print(f"output_dir={output_dir}")
    print(f"rows={len(joined)}")
    print(f"clean_rows={len(clean)}")
    print(f"dashboard={dashboard_path}")
    print(f"delta_dashboard={delta_dashboard_path}")
    for output_path in distribution_paths:
        print(f"distribution={output_path}")
    print(f"low_rows={low_rows_path}")
    print(f"low_summary={low_summary_path}")
    print(summary.to_string(index=False))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
