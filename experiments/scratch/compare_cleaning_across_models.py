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

from experiments.scratch.analyze_fps_stability_psnr import ORACLE_COLUMN
from experiments.scratch.analyze_fps_stability_psnr import ORACLE_DISTANCE_COLUMN
from experiments.scratch.analyze_fps_stability_psnr import PSNR_COLUMN
from experiments.scratch.analyze_fps_stability_psnr import SYMMETRY_COLUMN
from experiments.scratch.analyze_fps_stability_psnr import add_oracle_distance_columns
from experiments.scratch.analyze_fps_stability_psnr import build_joined_dataframe
from experiments.scratch.analyze_fps_stability_psnr import load_metrics
from experiments.scratch.analyze_fps_stability_psnr import load_raw_sequence_stats
from experiments.scratch.analyze_fps_stability_psnr import load_timing_rows

DEFAULT_BASELINE_METRICS_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\CGV_inference_outputs\IFRNet_FineTuning_0611\metrics.csv",
)
DEFAULT_CANDIDATE_METRICS_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\CGV_inference_outputs\IFRNet_Residual_FlowApprox_1_Layer_TwoStage_Splat_4Direction_MaskedDownscale_Minor_0611\metrics.csv",
)
DEFAULT_DATASET_JSON_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\Dataset",
)
DEFAULT_RAW_SEQUENCE_ROOT: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\Dataset\Minor_0507",
)
DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\cleaning_model_comparison\minor_0611",
)
DEFAULT_TARGET_FPS: float = 60.0
DEFAULT_FPS_THRESHOLD: float = 2.0
DEFAULT_SYMMETRY_THRESHOLD_MS: float = 0.5
DEFAULT_ORACLE_CENTER: float = 0.5
DEFAULT_ORACLE_WINDOW_RADIUS: float = 0.025
KEY_COLUMNS: tuple[str, ...] = ("inference_preset", "record", "mode", "frame_range")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    metrics_csv: Path


@dataclass(frozen=True)
class CleaningSpec:
    name: str
    display_name: str
    mask_column: str


def get_model_short_name(model_name: str) -> str:
    short_names: dict[str, str] = {
        "IFRNet FineTuning": "FineTuning",
        "FlowApprox MaskedDownscale": "FlowApprox",
    }
    return short_names.get(model_name, model_name)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare data-cleaning PSNR effects across two model metrics CSVs."
    )
    parser.add_argument("--baseline-metrics-csv", type=Path, default=DEFAULT_BASELINE_METRICS_CSV)
    parser.add_argument("--candidate-metrics-csv", type=Path, default=DEFAULT_CANDIDATE_METRICS_CSV)
    parser.add_argument("--baseline-name", type=str, default="IFRNet FineTuning")
    parser.add_argument("--candidate-name", type=str, default="FlowApprox MaskedDownscale")
    parser.add_argument("--dataset-json-dir", type=Path, default=DEFAULT_DATASET_JSON_DIR)
    parser.add_argument("--raw-sequence-root", type=Path, default=DEFAULT_RAW_SEQUENCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--fps-threshold", type=float, default=DEFAULT_FPS_THRESHOLD)
    parser.add_argument("--symmetry-threshold-ms", type=float, default=DEFAULT_SYMMETRY_THRESHOLD_MS)
    parser.add_argument("--oracle-center", type=float, default=DEFAULT_ORACLE_CENTER)
    parser.add_argument("--oracle-window-radius", type=float, default=DEFAULT_ORACLE_WINDOW_RADIUS)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.target_fps <= 0.0:
        raise ValueError(f"target-fps must be positive: target_fps={args.target_fps}")
    if args.fps_threshold <= 0.0:
        raise ValueError(f"fps-threshold must be positive: fps_threshold={args.fps_threshold}")
    if args.symmetry_threshold_ms <= 0.0:
        raise ValueError(f"symmetry-threshold-ms must be positive: symmetry_threshold_ms={args.symmetry_threshold_ms}")
    if args.oracle_center <= 0.0 or args.oracle_center >= 1.0:
        raise ValueError(f"oracle-center must be in (0, 1): oracle_center={args.oracle_center}")
    if args.oracle_window_radius <= 0.0:
        raise ValueError(f"oracle-window-radius must be positive: oracle_window_radius={args.oracle_window_radius}")


def build_model_joined_dataframe(
    model: ModelSpec,
    timing: pd.DataFrame,
    raw_stats: pd.DataFrame,
    oracle_center: float,
) -> pd.DataFrame:
    metrics = load_metrics(model.metrics_csv)
    joined = build_joined_dataframe(metrics=metrics, timing=timing, raw_stats=raw_stats)
    joined = add_oracle_distance_columns(dataframe=joined, oracle_center=oracle_center)
    joined["model_name"] = model.name
    return joined


def validate_matching_samples(dataframes: Sequence[pd.DataFrame]) -> None:
    if len(dataframes) < 2:
        return
    baseline_keys = dataframes[0][list(KEY_COLUMNS)].sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    for dataframe in dataframes[1:]:
        keys = dataframe[list(KEY_COLUMNS)].sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        if not baseline_keys.equals(keys):
            raise ValueError("Model metrics do not contain identical sample keys.")


def add_cleaning_masks(
    dataframe: pd.DataFrame,
    fps_threshold: float,
    symmetry_threshold_ms: float,
    oracle_window_radius: float,
) -> pd.DataFrame:
    result = dataframe.copy()
    result["clean_delta_time"] = result["fps_abs_error_max"] <= fps_threshold
    result["clean_symmetric_time"] = result[SYMMETRY_COLUMN] <= symmetry_threshold_ms
    result["clean_oracle_t_eff"] = result[ORACLE_DISTANCE_COLUMN] <= oracle_window_radius
    return result


def build_cleaning_specs(args: argparse.Namespace) -> list[CleaningSpec]:
    oracle_lower = args.oracle_center - args.oracle_window_radius
    oracle_upper = args.oracle_center + args.oracle_window_radius
    return [
        CleaningSpec(
            name="delta_time",
            display_name=f"delta time\nmax FPS err <= {args.fps_threshold:g}",
            mask_column="clean_delta_time",
        ),
        CleaningSpec(
            name="symmetric_time",
            display_name=f"symmetric time\n|dt1-dt2| <= {args.symmetry_threshold_ms:g} ms",
            mask_column="clean_symmetric_time",
        ),
        CleaningSpec(
            name="oracle_t_eff",
            display_name=f"oracle t_eff\n{oracle_lower:.3f}-{oracle_upper:.3f}",
            mask_column="clean_oracle_t_eff",
        ),
    ]


def summarize_cleaning_effects(dataframe: pd.DataFrame, cleaning_specs: Sequence[CleaningSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, model_dataframe in dataframe.groupby("model_name", sort=False):
        all_mean_psnr = float(model_dataframe[PSNR_COLUMN].mean())
        all_median_psnr = float(model_dataframe[PSNR_COLUMN].median())
        all_samples = int(len(model_dataframe))
        rows.append(
            {
                "model_name": str(model_name),
                "cleaning_name": "all",
                "display_name": "all samples",
                "samples": all_samples,
                "retention_ratio": 1.0,
                "mean_psnr": all_mean_psnr,
                "median_psnr": all_median_psnr,
                "mean_psnr_gain": 0.0,
                "median_psnr_gain": 0.0,
            }
        )
        for cleaning_spec in cleaning_specs:
            cleaned = model_dataframe[model_dataframe[cleaning_spec.mask_column]].copy()
            rows.append(
                {
                    "model_name": str(model_name),
                    "cleaning_name": cleaning_spec.name,
                    "display_name": cleaning_spec.display_name,
                    "samples": int(len(cleaned)),
                    "retention_ratio": float(len(cleaned) / all_samples),
                    "mean_psnr": float(cleaned[PSNR_COLUMN].mean()),
                    "median_psnr": float(cleaned[PSNR_COLUMN].median()),
                    "mean_psnr_gain": float(cleaned[PSNR_COLUMN].mean() - all_mean_psnr),
                    "median_psnr_gain": float(cleaned[PSNR_COLUMN].median() - all_median_psnr),
                }
            )
    return pd.DataFrame(rows)


def build_cleaned_sample_exports(
    dataframe: pd.DataFrame,
    cleaning_specs: Sequence[CleaningSpec],
    output_dir: Path,
) -> None:
    for cleaning_spec in cleaning_specs:
        cleaned = dataframe[dataframe[cleaning_spec.mask_column]].copy()
        cleaned.to_csv(output_dir / f"{cleaning_spec.name}_cleaned_samples_by_model.csv", index=False)


def write_model_comparison_dashboard(summary: pd.DataFrame, cleaning_specs: Sequence[CleaningSpec], output_dir: Path) -> Path:
    output_path = output_dir / "cleaning_model_comparison_dashboard.png"
    model_order = summary["model_name"].drop_duplicates().tolist()
    cleaning_order = [cleaning_spec.name for cleaning_spec in cleaning_specs]
    cleaning_labels = {
        cleaning_spec.name: cleaning_spec.display_name
        for cleaning_spec in cleaning_specs
    }
    colors = {
        model_order[0]: "#4c78a8",
        model_order[1]: "#f58518" if len(model_order) > 1 else "#f58518",
    }

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=160)
    figure.suptitle("Data-cleaning effect across models", fontsize=16)

    axis = axes[0, 0]
    x_positions = np.arange(len(cleaning_order))
    width = 0.36
    for model_index, model_name in enumerate(model_order):
        model_summary = summary[
            (summary["model_name"] == model_name)
            & (summary["cleaning_name"].isin(cleaning_order))
        ].set_index("cleaning_name").loc[cleaning_order].reset_index()
        offsets = x_positions + (model_index - (len(model_order) - 1) / 2.0) * width
        axis.bar(
            offsets,
            model_summary["mean_psnr_gain"],
            width=width,
            color=colors[model_name],
            label=model_name,
            alpha=0.9,
        )
        for x_position, value in zip(offsets, model_summary["mean_psnr_gain"]):
            axis.text(x_position, float(value) + 0.015, f"{float(value):+.2f}", ha="center", va="bottom", fontsize=8)
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set_xticks(x_positions, [cleaning_labels[name] for name in cleaning_order])
    axis.set_title("Mean PSNR gain after cleaning")
    axis.set_ylabel("mean PSNR gain (dB)")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()

    axis = axes[0, 1]
    for model_name in model_order:
        model_summary = summary[
            (summary["model_name"] == model_name)
            & (summary["cleaning_name"].isin(["all", *cleaning_order]))
        ]
        baseline = model_summary[model_summary["cleaning_name"] == "all"].iloc[0]
        cleaned = model_summary[model_summary["cleaning_name"].isin(cleaning_order)].copy()
        axis.scatter(
            cleaned["retention_ratio"] * 100.0,
            cleaned["mean_psnr"],
            s=75,
            color=colors[model_name],
            alpha=0.85,
            label=f"{model_name} cleaned",
        )
        axis.axhline(
            float(baseline["mean_psnr"]),
            color=colors[model_name],
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            label=f"{model_name} all={float(baseline['mean_psnr']):.2f}",
        )
    short_cleaning_labels: dict[str, str] = {
        "delta_time": "delta",
        "symmetric_time": "symmetric",
        "oracle_t_eff": "oracle",
    }
    for cleaning_spec in cleaning_specs:
        label_rows = summary[summary["cleaning_name"] == cleaning_spec.name]
        axis.annotate(
            short_cleaning_labels[cleaning_spec.name],
            (float(label_rows["retention_ratio"].mean()) * 100.0, float(label_rows["mean_psnr"].mean())),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    axis.set_title("Retention vs mean PSNR")
    axis.set_xlabel("retained samples (%)")
    axis.set_ylabel("mean psnr (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    x_positions = np.arange(len(cleaning_order))
    for model_index, model_name in enumerate(model_order):
        model_summary = summary[
            (summary["model_name"] == model_name)
            & (summary["cleaning_name"].isin(cleaning_order))
        ].set_index("cleaning_name").loc[cleaning_order].reset_index()
        offsets = x_positions + (model_index - (len(model_order) - 1) / 2.0) * width
        axis.bar(
            offsets,
            model_summary["retention_ratio"] * 100.0,
            width=width,
            color=colors[model_name],
            label=model_name,
            alpha=0.9,
        )
    axis.set_xticks(x_positions, [cleaning_labels[name] for name in cleaning_order])
    axis.set_title("Sample retention")
    axis.set_ylabel("retained samples (%)")
    axis.set_ylim(0.0, 105.0)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()

    axis = axes[1, 1]
    baseline_rows = summary[summary["cleaning_name"] == "all"].copy()
    oracle_rows = summary[summary["cleaning_name"] == "oracle_t_eff"].copy()
    plot_rows = pd.concat([baseline_rows, oracle_rows], ignore_index=True)
    plot_rows["cleaning_label"] = plot_rows["cleaning_name"].replace({"all": "all", "oracle_t_eff": "oracle"})
    plot_rows["x_label"] = [
        f"{get_model_short_name(str(row.model_name))}\n{str(row.cleaning_label)}"
        for row in plot_rows.itertuples(index=False)
    ]
    bars = axis.bar(
        plot_rows["x_label"],
        plot_rows["mean_psnr"],
        color=[
            colors[row.model_name] if row.cleaning_name == "all" else "#2ca02c"
            for row in plot_rows.itertuples(index=False)
        ],
        alpha=0.9,
    )
    y_min = float(plot_rows["mean_psnr"].min()) - 0.25
    y_max = float(plot_rows["mean_psnr"].max()) + 0.25
    axis.set_ylim(y_min, y_max)
    axis.set_title("Oracle clean vs all samples")
    axis.set_ylabel("mean psnr (dB)")
    axis.tick_params(axis="x", rotation=0, labelsize=8)
    axis.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, plot_rows["mean_psnr"]):
        axis.text(bar.get_x() + bar.get_width() / 2.0, float(value) + 0.03, f"{float(value):.2f}", ha="center", va="bottom", fontsize=9)

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_method_specific_gain_dashboards(summary: pd.DataFrame, cleaning_specs: Sequence[CleaningSpec], output_dir: Path) -> list[Path]:
    output_paths: list[Path] = []
    model_order = summary["model_name"].drop_duplicates().tolist()
    colors = {
        model_order[0]: "#4c78a8",
        model_order[1]: "#f58518" if len(model_order) > 1 else "#f58518",
    }

    for cleaning_spec in cleaning_specs:
        output_path = output_dir / f"{cleaning_spec.name}_model_gain_dashboard.png"
        rows = summary[summary["cleaning_name"].isin(["all", cleaning_spec.name])].copy()
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=160)
        figure.suptitle(f"{cleaning_spec.display_name.replace(chr(10), ' ')} model comparison", fontsize=14)

        axis = axes[0]
        x_positions = np.arange(len(model_order))
        width = 0.36
        for model_index, model_name in enumerate(model_order):
            model_rows = rows[rows["model_name"] == model_name]
            all_row = model_rows[model_rows["cleaning_name"] == "all"].iloc[0]
            clean_row = model_rows[model_rows["cleaning_name"] == cleaning_spec.name].iloc[0]
            values = [float(all_row["mean_psnr"]), float(clean_row["mean_psnr"])]
            local_x = np.asarray([model_index * 2.0, model_index * 2.0 + width])
            axis.bar(local_x, values, width=width, color=[colors[model_name], "#2ca02c"], alpha=0.9)
            for x_position, value in zip(local_x, values):
                axis.text(x_position, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        axis.set_xticks(
            [index * 2.0 + width / 2.0 for index in range(len(model_order))],
            model_order,
        )
        method_rows = rows[rows["cleaning_name"].isin(["all", cleaning_spec.name])]
        axis.set_ylim(float(method_rows["mean_psnr"].min()) - 0.25, float(method_rows["mean_psnr"].max()) + 0.25)
        axis.set_title("All vs cleaned mean PSNR")
        axis.set_ylabel("mean psnr (dB)")
        axis.grid(True, axis="y", alpha=0.25)

        axis = axes[1]
        clean_rows = summary[summary["cleaning_name"] == cleaning_spec.name].copy()
        axis.bar(clean_rows["model_name"], clean_rows["mean_psnr_gain"], color=[colors[name] for name in clean_rows["model_name"]], alpha=0.9)
        axis.axhline(0.0, color="#333333", linewidth=1.0)
        axis.set_title("Mean PSNR gain")
        axis.set_ylabel("gain (dB)")
        axis.grid(True, axis="y", alpha=0.25)
        for index, row in enumerate(clean_rows.itertuples(index=False)):
            axis.text(index, float(row.mean_psnr_gain) + 0.015, f"{float(row.mean_psnr_gain):+.2f}", ha="center", va="bottom", fontsize=9)

        axis = axes[2]
        axis.bar(clean_rows["model_name"], clean_rows["samples"], color=[colors[name] for name in clean_rows["model_name"]], alpha=0.9)
        axis.set_title("Cleaned samples")
        axis.set_ylabel("samples")
        axis.grid(True, axis="y", alpha=0.25)
        for index, row in enumerate(clean_rows.itertuples(index=False)):
            axis.text(
                index,
                int(row.samples) + int(rows["samples"].max()) * 0.015,
                f"{int(row.samples)}\n{float(row.retention_ratio) * 100.0:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        figure.tight_layout()
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = load_metrics(args.baseline_metrics_csv)
    timing = load_timing_rows(
        dataset_json_dir=args.dataset_json_dir,
        metrics=baseline_metrics,
        target_fps=float(args.target_fps),
    )
    raw_stats = load_raw_sequence_stats(raw_sequence_root=args.raw_sequence_root)

    models = [
        ModelSpec(name=str(args.baseline_name), metrics_csv=args.baseline_metrics_csv),
        ModelSpec(name=str(args.candidate_name), metrics_csv=args.candidate_metrics_csv),
    ]
    joined_frames = [
        build_model_joined_dataframe(
            model=model,
            timing=timing,
            raw_stats=raw_stats,
            oracle_center=float(args.oracle_center),
        )
        for model in models
    ]
    validate_matching_samples(joined_frames)
    joined = pd.concat(joined_frames, ignore_index=True)
    joined = add_cleaning_masks(
        dataframe=joined,
        fps_threshold=float(args.fps_threshold),
        symmetry_threshold_ms=float(args.symmetry_threshold_ms),
        oracle_window_radius=float(args.oracle_window_radius),
    )
    cleaning_specs = build_cleaning_specs(args)
    summary = summarize_cleaning_effects(dataframe=joined, cleaning_specs=cleaning_specs)

    joined.to_csv(output_dir / "model_cleaning_sample_comparison.csv", index=False)
    summary.to_csv(output_dir / "model_cleaning_summary.csv", index=False)
    build_cleaned_sample_exports(dataframe=joined, cleaning_specs=cleaning_specs, output_dir=output_dir)
    comparison_dashboard_path = write_model_comparison_dashboard(
        summary=summary,
        cleaning_specs=cleaning_specs,
        output_dir=output_dir,
    )
    method_dashboard_paths = write_method_specific_gain_dashboards(
        summary=summary,
        cleaning_specs=cleaning_specs,
        output_dir=output_dir,
    )

    print(f"output_dir={output_dir}")
    print(f"comparison_dashboard={comparison_dashboard_path}")
    for output_path in method_dashboard_paths:
        print(f"method_dashboard={output_path}")
    print(summary.to_string(index=False))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
