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
import pandas as pd

from experiments.scratch.split_motion_psnr_buckets import BUCKET_COLORS
from experiments.scratch.split_motion_psnr_buckets import BUCKET_ORDER
from experiments.scratch.split_motion_psnr_buckets import build_bucket_summary
from experiments.scratch.split_motion_psnr_buckets import classify_bucket

DEFAULT_SAMPLE_COMPARISON_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\cleaning_model_comparison\minor_0611\model_cleaning_sample_comparison.csv",
)
DEFAULT_THRESHOLDS_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\baseline_finetuning_motion_psnr\buckets\thresholds.csv",
)
DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\cleaning_model_comparison\minor_0611\motion_distribution_after_cleaning",
)
PSNR_COLUMN: str = "psnr"
BUCKET_DISPLAY_LABELS: dict[str, str] = {
    "large_motion_artifacts": "large motion\nartifacts",
    "not_large_motion_high_psnr": "not large motion\nhigh psnr",
    "other_artifacts": "other\nartifacts",
    "large_motion_high_psnr": "large motion\nhigh psnr",
}


@dataclass(frozen=True)
class ThresholdSpec:
    motion_column: str
    motion_threshold: float
    psnr_threshold: float


@dataclass(frozen=True)
class PlotTarget:
    model_name: str
    cleaning_name: str
    mask_column: str
    file_stem: str
    title: str


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot motion/PSNR bucket distributions after delta-time and oracle-t-eff cleaning."
    )
    parser.add_argument("--sample-comparison-csv", type=Path, default=DEFAULT_SAMPLE_COMPARISON_CSV)
    parser.add_argument("--thresholds-csv", type=Path, default=DEFAULT_THRESHOLDS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_columns(dataframe: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"{label} is missing required columns: {missing_columns}")


def load_threshold_spec(thresholds_csv: Path) -> ThresholdSpec:
    require_existing_path(thresholds_csv, "thresholds CSV")
    thresholds = pd.read_csv(thresholds_csv)
    if len(thresholds) != 1:
        raise ValueError(f"thresholds CSV must contain exactly one row: path={thresholds_csv} rows={len(thresholds)}")
    require_columns(
        dataframe=thresholds,
        columns=("motion_column", "motion_threshold", "psnr_threshold"),
        label="thresholds CSV",
    )
    row = thresholds.iloc[0]
    return ThresholdSpec(
        motion_column=str(row["motion_column"]),
        motion_threshold=float(row["motion_threshold"]),
        psnr_threshold=float(row["psnr_threshold"]),
    )


def load_sample_comparison(sample_comparison_csv: Path, threshold_spec: ThresholdSpec) -> pd.DataFrame:
    require_existing_path(sample_comparison_csv, "sample comparison CSV")
    dataframe = pd.read_csv(sample_comparison_csv)
    if len(dataframe) == 0:
        raise ValueError(f"sample comparison CSV is empty: {sample_comparison_csv}")
    require_columns(
        dataframe=dataframe,
        columns=(
            "model_name",
            PSNR_COLUMN,
            threshold_spec.motion_column,
            "clean_delta_time",
            "clean_oracle_t_eff",
        ),
        label="sample comparison CSV",
    )
    return dataframe


def build_plot_targets() -> list[PlotTarget]:
    return [
        PlotTarget(
            model_name="IFRNet FineTuning",
            cleaning_name="delta_time",
            mask_column="clean_delta_time",
            file_stem="baseline_delta_time_motion_distribution",
            title="Baseline after delta-time cleaning",
        ),
        PlotTarget(
            model_name="IFRNet FineTuning",
            cleaning_name="oracle_t_eff",
            mask_column="clean_oracle_t_eff",
            file_stem="baseline_oracle_t_eff_motion_distribution",
            title="Baseline after oracle_fmv_t_eff cleaning",
        ),
        PlotTarget(
            model_name="FlowApprox MaskedDownscale",
            cleaning_name="delta_time",
            mask_column="clean_delta_time",
            file_stem="candidate_delta_time_motion_distribution",
            title="Candidate after delta-time cleaning",
        ),
        PlotTarget(
            model_name="FlowApprox MaskedDownscale",
            cleaning_name="oracle_t_eff",
            mask_column="clean_oracle_t_eff",
            file_stem="candidate_oracle_t_eff_motion_distribution",
            title="Candidate after oracle_fmv_t_eff cleaning",
        ),
    ]


def build_bucketed_dataframe(dataframe: pd.DataFrame, target: PlotTarget, threshold_spec: ThresholdSpec) -> pd.DataFrame:
    model_dataframe = dataframe[dataframe["model_name"] == target.model_name].copy()
    if len(model_dataframe) == 0:
        raise ValueError(f"No samples found for model: model_name={target.model_name}")

    retained = model_dataframe[model_dataframe[target.mask_column].astype(bool)].copy()
    if len(retained) == 0:
        raise ValueError(f"No retained samples found for target: model_name={target.model_name} cleaning={target.cleaning_name}")

    retained["psnr_summary"] = retained[PSNR_COLUMN].astype(float)
    retained["psnr_summary_source"] = PSNR_COLUMN
    retained["psnr_experiment_columns"] = PSNR_COLUMN
    retained["motion_threshold"] = threshold_spec.motion_threshold
    retained["psnr_threshold"] = threshold_spec.psnr_threshold
    retained["bucket"] = [
        classify_bucket(
            motion_value=float(motion_value),
            psnr_value=float(psnr_value),
            motion_threshold=threshold_spec.motion_threshold,
            psnr_threshold=threshold_spec.psnr_threshold,
        )
        for motion_value, psnr_value in zip(retained[threshold_spec.motion_column], retained["psnr_summary"])
    ]
    retained["bucket"] = pd.Categorical(retained["bucket"], categories=BUCKET_ORDER, ordered=True)
    retained["cleaning_name"] = target.cleaning_name
    retained["original_model_samples"] = int(len(model_dataframe))
    retained["retained_model_samples"] = int(len(retained))
    retained["retention_ratio"] = float(len(retained) / len(model_dataframe))
    return retained.sort_values(
        ["bucket", threshold_spec.motion_column, "psnr_summary"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_complete_bucket_summary(bucketed: pd.DataFrame, target: PlotTarget, threshold_spec: ThresholdSpec) -> pd.DataFrame:
    summary = build_bucket_summary(dataframe=bucketed, motion_column=threshold_spec.motion_column)
    summary = summary.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    summary["samples"] = summary["samples"].fillna(0).astype(int)
    summary["ratio"] = summary["ratio"].fillna(0.0).astype(float)
    summary["model_name"] = target.model_name
    summary["cleaning_name"] = target.cleaning_name
    summary["motion_threshold"] = threshold_spec.motion_threshold
    summary["psnr_threshold"] = threshold_spec.psnr_threshold
    summary["total_retained_samples"] = int(len(bucketed))
    summary["retention_ratio"] = float(bucketed["retention_ratio"].iloc[0])
    return summary


def write_bucketed_csv(bucketed: pd.DataFrame, output_dir: Path, file_stem: str) -> Path:
    output_path = output_dir / f"{file_stem}_bucketed_samples.csv"
    bucketed.to_csv(output_path, index=False)
    return output_path


def write_distribution_dashboard(
    bucketed: pd.DataFrame,
    summary: pd.DataFrame,
    target: PlotTarget,
    threshold_spec: ThresholdSpec,
    output_dir: Path,
) -> Path:
    output_path = output_dir / f"{target.file_stem}.png"
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9), dpi=160)
    motion_axis = axes[0, 0]
    scatter_axis = axes[0, 1]
    psnr_axis = axes[1, 0]
    coverage_axis = axes[1, 1]

    for bucket_name in BUCKET_ORDER:
        bucket_dataframe = bucketed[bucketed["bucket"] == bucket_name]
        if len(bucket_dataframe) == 0:
            continue
        label = f"{bucket_name} (n={len(bucket_dataframe)})"
        color = BUCKET_COLORS[bucket_name]
        motion_axis.hist(
            bucket_dataframe[threshold_spec.motion_column],
            bins=40,
            alpha=0.38,
            color=color,
            label=label,
        )
        scatter_axis.scatter(
            bucket_dataframe[threshold_spec.motion_column],
            bucket_dataframe["psnr_summary"],
            s=12,
            alpha=0.4,
            edgecolors="none",
            color=color,
            label=label,
        )
        psnr_axis.hist(
            bucket_dataframe["psnr_summary"],
            bins=40,
            alpha=0.38,
            color=color,
            label=label,
        )

    motion_axis.axvline(
        threshold_spec.motion_threshold,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=f"motion threshold = {threshold_spec.motion_threshold:.2f}",
    )
    motion_axis.set_xlabel(threshold_spec.motion_column)
    motion_axis.set_ylabel("samples")
    motion_axis.set_title("Per-sample motion distribution")
    motion_axis.grid(True, axis="y", alpha=0.25)
    motion_axis.legend(fontsize=8)

    scatter_axis.axvline(
        threshold_spec.motion_threshold,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=f"motion threshold = {threshold_spec.motion_threshold:.2f}",
    )
    scatter_axis.axhline(
        threshold_spec.psnr_threshold,
        color="gray",
        linestyle="--",
        linewidth=1.1,
        label=f"psnr threshold = {threshold_spec.psnr_threshold:.2f}",
    )
    scatter_axis.set_xlabel(threshold_spec.motion_column)
    scatter_axis.set_ylabel("psnr_summary")
    scatter_axis.set_title("Per-sample motion vs PSNR")
    scatter_axis.grid(True, alpha=0.25)
    scatter_axis.legend(fontsize=8, markerscale=1.3)

    psnr_axis.axvline(
        threshold_spec.psnr_threshold,
        color="gray",
        linestyle="--",
        linewidth=1.1,
        label=f"psnr threshold = {threshold_spec.psnr_threshold:.2f}",
    )
    psnr_axis.set_xlabel("psnr_summary")
    psnr_axis.set_ylabel("samples")
    psnr_axis.set_title("Per-sample PSNR distribution")
    psnr_axis.grid(True, axis="y", alpha=0.25)
    psnr_axis.legend(fontsize=8)

    summary = summary.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    coverage_axis.barh(
        [BUCKET_DISPLAY_LABELS[str(bucket_name)] for bucket_name in summary["bucket"]],
        summary["ratio"] * 100.0,
        color=[BUCKET_COLORS[str(bucket_name)] for bucket_name in summary["bucket"]],
    )
    for index, row in summary.iterrows():
        coverage_axis.text(
            float(row["ratio"]) * 100.0 + 0.6,
            index,
            f"{float(row['ratio']) * 100.0:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
        )
    coverage_axis.set_xlabel("retained dataset percentage")
    coverage_axis.set_title("Bucket coverage after cleaning")
    coverage_axis.grid(True, axis="x", alpha=0.25)

    retained_count = int(len(bucketed))
    original_count = int(bucketed["original_model_samples"].iloc[0])
    retention_ratio = float(bucketed["retention_ratio"].iloc[0]) * 100.0
    figure.suptitle(
        f"{target.title}\n"
        f"retained {retained_count}/{original_count} ({retention_ratio:.1f}%), "
        f"motion >= {threshold_spec.motion_threshold:.2f}, psnr < {threshold_spec.psnr_threshold:.2f}",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run(args: argparse.Namespace) -> None:
    threshold_spec = load_threshold_spec(args.thresholds_csv)
    dataframe = load_sample_comparison(
        sample_comparison_csv=args.sample_comparison_csv,
        threshold_spec=threshold_spec,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_frames: list[pd.DataFrame] = []
    output_paths: list[Path] = []
    for target in build_plot_targets():
        bucketed = build_bucketed_dataframe(
            dataframe=dataframe,
            target=target,
            threshold_spec=threshold_spec,
        )
        summary = build_complete_bucket_summary(
            bucketed=bucketed,
            target=target,
            threshold_spec=threshold_spec,
        )
        write_bucketed_csv(bucketed=bucketed, output_dir=output_dir, file_stem=target.file_stem)
        output_paths.append(
            write_distribution_dashboard(
                bucketed=bucketed,
                summary=summary,
                target=target,
                threshold_spec=threshold_spec,
                output_dir=output_dir,
            )
        )
        summary_frames.append(summary)

    combined_summary = pd.concat(summary_frames, ignore_index=True)
    combined_summary.to_csv(output_dir / "motion_distribution_bucket_summary.csv", index=False)

    print(f"output_dir={output_dir}")
    print(f"motion_threshold={threshold_spec.motion_threshold:.6f}")
    print(f"psnr_threshold={threshold_spec.psnr_threshold:.6f}")
    for output_path in output_paths:
        print(f"figure={output_path}")
    print(combined_summary[["model_name", "cleaning_name", "bucket", "samples", "ratio"]].to_string(index=False))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
