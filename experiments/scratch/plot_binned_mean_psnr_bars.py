from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_SUMMARY_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\baseline_finetuning_motion_psnr\motion_psnr_summary_by_bin.csv",
)
DEFAULT_OUTPUT_PATH: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\baseline_finetuning_motion_psnr\figures\motion_magnitude_mean__binned_mean_psnr_bar.png",
)
DEFAULT_EXPERIMENT: str | None = None


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean PSNR by motion bin as a bar chart with categorical x-axis bins."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Path to motion_psnr_summary_by_bin.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the output PNG.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT,
        help="Optional experiment label filter. If omitted and only one experiment exists, use it automatically.",
    )
    return parser.parse_args(argv)


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def load_summary(summary_csv: Path) -> pd.DataFrame:
    require_existing_path(summary_csv, "summary CSV")
    summary = pd.read_csv(summary_csv)
    required_columns = ("experiment", "motion_bin", "samples", "psnr_mean")
    missing_columns = [column for column in required_columns if column not in summary.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"Summary CSV is missing required columns: {missing_columns}")
    if len(summary) == 0:
        raise ValueError(f"Summary CSV is empty: {summary_csv}")
    return summary


def resolve_experiment(summary: pd.DataFrame, requested_experiment: str | None) -> str:
    experiments = summary["experiment"].drop_duplicates().tolist()
    if requested_experiment is not None:
        if requested_experiment not in experiments:
            raise ValueError(f"experiment={requested_experiment} not found. Available: {experiments}")
        return requested_experiment
    if len(experiments) != 1:
        raise ValueError(f"Multiple experiments found, please pass --experiment. Available: {experiments}")
    return str(experiments[0])


def build_mapping_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}__bin_mapping.csv")


def write_bin_mapping(experiment_summary: pd.DataFrame, output_path: Path) -> None:
    mapping = experiment_summary[["bin_index", "motion_bin", "samples", "motion_min", "motion_mean", "motion_max", "psnr_mean"]].copy()
    mapping.to_csv(output_path, index=False)


def write_bar_plot(summary: pd.DataFrame, experiment: str, output_path: Path) -> None:
    experiment_summary = summary[summary["experiment"] == experiment].copy()
    experiment_summary["x_label"] = experiment_summary["motion_bin"].astype(str)
    experiment_summary["bin_index"] = list(range(len(experiment_summary)))
    write_bin_mapping(experiment_summary, build_mapping_output_path(output_path))

    figure, axis = plt.subplots(figsize=(10.5, 5.6), dpi=160)
    bars = axis.bar(
        experiment_summary["bin_index"],
        experiment_summary["psnr_mean"],
        color="#1f77b4",
        alpha=0.9,
        width=1.0,
        align="edge",
    )
    axis.set_xlabel("bin index")
    axis.set_ylabel("mean psnr")
    axis.set_title(f"Mean PSNR by motion_magnitude_mean quantile bin ({experiment})")
    axis.grid(True, axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.set_xlim(0.0, float(len(experiment_summary)))
    axis.set_xticks([index + 0.5 for index in experiment_summary["bin_index"].tolist()])
    axis.set_xticklabels([str(index) for index in experiment_summary["bin_index"].tolist()])
    axis.margins(x=0.0)

    for bar, sample_count in zip(bars, experiment_summary["samples"].astype(int).tolist()):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.04,
            str(sample_count),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    summary = load_summary(args.summary_csv)
    experiment = resolve_experiment(summary, args.experiment)
    write_bar_plot(summary, experiment, args.output_path)
    print(f"summary_csv={args.summary_csv}")
    print(f"experiment={experiment}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
