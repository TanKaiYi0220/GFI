from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

METRIC_NAMES: tuple[str, ...] = ("psnr", "loss_rec", "loss_geo", "loss_dis", "loss_total")
TRAIN_EPOCH_PATTERN: re.Pattern[str] = re.compile(r"train_epoch_(\d+)\.csv$")
TEST_EPOCH_PATTERN: re.Pattern[str] = re.compile(r"test_epoch_(\d+)\.csv$")


def extract_epoch(csv_path: Path, pattern: re.Pattern[str]) -> int:
    match = pattern.match(csv_path.name)
    if match is None:
        raise ValueError(f"Invalid test epoch filename: {csv_path}")

    return int(match.group(1))


def collect_metric_rows(checkpoints_dir: Path, glob_pattern: str, epoch_pattern: re.Pattern[str]) -> list[dict[str, float]]:
    csv_paths = sorted(checkpoints_dir.glob(glob_pattern), key=lambda path: extract_epoch(path, epoch_pattern))
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No {glob_pattern} found in {checkpoints_dir}")

    metric_rows: list[dict[str, float]] = []
    for csv_path in csv_paths:
        dataframe = pd.read_csv(csv_path)
        missing_columns = [metric_name for metric_name in METRIC_NAMES if metric_name not in dataframe.columns]
        if len(missing_columns) > 0:
            raise KeyError(f"Missing columns in {csv_path}: {missing_columns}")

        metric_rows.append(
            {
                "epoch": float(extract_epoch(csv_path, epoch_pattern)),
                "psnr": float(dataframe["psnr"].mean()),
                "loss_rec": float(dataframe["loss_rec"].mean()),
                "loss_geo": float(dataframe["loss_geo"].mean()),
                "loss_dis": float(dataframe["loss_dis"].mean()),
                "loss_total": float(dataframe["loss_total"].mean()),
            }
        )

    return metric_rows


def build_metrics_dataframe(checkpoints_dir: Path, glob_pattern: str, epoch_pattern: re.Pattern[str]) -> pd.DataFrame:
    return pd.DataFrame(collect_metric_rows(checkpoints_dir, glob_pattern, epoch_pattern))


def plot_metric(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame, metric_name: str, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(train_dataframe["epoch"], train_dataframe[metric_name], marker="o", linewidth=1.5, label="training")
    axis.plot(test_dataframe["epoch"], test_dataframe[metric_name], marker="o", linewidth=1.5, label="testing")
    axis.set_xlabel("epoch")
    axis.set_ylabel(metric_name)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{metric_name}.png", dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot averaged metrics from train_epoch_*.csv and test_epoch_*.csv files.")
    parser.add_argument("checkpoints_dir", type=Path, help="Directory containing test_epoch_*.csv files.")
    parser.add_argument("output_dir", type=Path, help="Output directory for metric figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dataframe = build_metrics_dataframe(args.checkpoints_dir, "train_epoch_*.csv", TRAIN_EPOCH_PATTERN)
    test_dataframe = build_metrics_dataframe(args.checkpoints_dir, "test_epoch_*.csv", TEST_EPOCH_PATTERN)

    for metric_name in METRIC_NAMES:
        plot_metric(train_dataframe, test_dataframe, metric_name, args.output_dir)


if __name__ == "__main__":
    main()
