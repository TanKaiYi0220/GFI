from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_SAMPLE_COMPARISON_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\baseline_finetuning_oracle_t_eff_psnr\motion_psnr_sample_comparison.csv",
)
DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\baseline_finetuning_oracle_t_eff_psnr\artifact_overlap",
)
DEFAULT_PSNR_COLUMN: str = "IFRNet FineTuning"
DEFAULT_MOTION_QUANTILE: float = 0.90
DEFAULT_PSNR_QUANTILE: float = 0.15
DEFAULT_ORACLE_DEVIATION_QUANTILE: float = 0.90
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


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze overlap between large-motion artifacts and oracle_t_eff artifacts."
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
        "--motion-quantile",
        type=float,
        default=DEFAULT_MOTION_QUANTILE,
        help="Quantile for large-motion threshold.",
    )
    parser.add_argument(
        "--psnr-quantile",
        type=float,
        default=DEFAULT_PSNR_QUANTILE,
        help="Quantile for low-PSNR threshold.",
    )
    parser.add_argument(
        "--oracle-deviation-quantile",
        type=float,
        default=DEFAULT_ORACLE_DEVIATION_QUANTILE,
        help="Quantile for |oracle_fmv_t_eff_mean - 0.5| threshold.",
    )
    parser.add_argument(
        "--bin-count",
        type=int,
        default=20,
        help="Bin count for oracle_fmv_t_eff_mean mean-PSNR trend plot.",
    )
    return parser.parse_args(argv)


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_quantile(value: float, label: str) -> None:
    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{label} must be in (0, 1), got {value}")


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


def build_threshold_row(
    dataframe: pd.DataFrame,
    psnr_column: str,
    motion_quantile: float,
    psnr_quantile: float,
    oracle_deviation_quantile: float,
) -> dict[str, float | str]:
    motion_threshold = float(dataframe["motion_magnitude_mean"].quantile(motion_quantile))
    psnr_threshold = float(dataframe[psnr_column].quantile(psnr_quantile))
    oracle_deviation = (dataframe["oracle_fmv_t_eff_mean"] - ORACLE_CENTER).abs()
    oracle_deviation_threshold = float(oracle_deviation.quantile(oracle_deviation_quantile))
    return {
        "psnr_column": psnr_column,
        "motion_quantile": motion_quantile,
        "motion_threshold": motion_threshold,
        "psnr_quantile": psnr_quantile,
        "psnr_threshold": psnr_threshold,
        "oracle_center": ORACLE_CENTER,
        "oracle_deviation_quantile": oracle_deviation_quantile,
        "oracle_deviation_threshold": oracle_deviation_threshold,
        "oracle_left_threshold": float(ORACLE_CENTER - oracle_deviation_threshold),
        "oracle_right_threshold": float(ORACLE_CENTER + oracle_deviation_threshold),
    }


def classify_overlap_label(is_large_motion_artifact: bool, is_oracle_t_eff_artifact: bool) -> str:
    if is_large_motion_artifact and is_oracle_t_eff_artifact:
        return "both_artifacts"
    if is_large_motion_artifact:
        return "large_motion_only"
    if is_oracle_t_eff_artifact:
        return "oracle_t_eff_only"
    return "neither"


def build_overlap_dataframe(dataframe: pd.DataFrame, threshold_row: dict[str, float | str]) -> pd.DataFrame:
    motion_threshold = float(threshold_row["motion_threshold"])
    psnr_threshold = float(threshold_row["psnr_threshold"])
    oracle_left_threshold = float(threshold_row["oracle_left_threshold"])
    oracle_right_threshold = float(threshold_row["oracle_right_threshold"])
    psnr_column = str(threshold_row["psnr_column"])

    overlap = dataframe.copy()
    overlap["psnr_summary"] = overlap[psnr_column].astype(float)
    overlap["oracle_t_eff_distance"] = (overlap["oracle_fmv_t_eff_mean"] - ORACLE_CENTER).abs()
    overlap["is_low_psnr"] = overlap["psnr_summary"] < psnr_threshold
    overlap["is_large_motion_artifact"] = (
        overlap["motion_magnitude_mean"] >= motion_threshold
    ) & overlap["is_low_psnr"]
    overlap["is_oracle_t_eff_artifact"] = (
        (
            overlap["oracle_fmv_t_eff_mean"] <= oracle_left_threshold
        ) | (
            overlap["oracle_fmv_t_eff_mean"] >= oracle_right_threshold
        )
    ) & overlap["is_low_psnr"]
    overlap["overlap_label"] = [
        classify_overlap_label(
            bool(is_large_motion_artifact),
            bool(is_oracle_t_eff_artifact),
        )
        for is_large_motion_artifact, is_oracle_t_eff_artifact in zip(
            overlap["is_large_motion_artifact"],
            overlap["is_oracle_t_eff_artifact"],
        )
    ]
    overlap["overlap_label"] = pd.Categorical(overlap["overlap_label"], categories=OVERLAP_ORDER, ordered=True)
    return overlap.sort_values(
        ["overlap_label", "psnr_summary", "motion_magnitude_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


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

    rows.extend(
        (
            {
                "group": "large_motion_artifact",
                "samples": int(dataframe["is_large_motion_artifact"].sum()),
                "ratio": float(dataframe["is_large_motion_artifact"].mean()),
                "motion_mean": float(dataframe.loc[dataframe["is_large_motion_artifact"], "motion_magnitude_mean"].mean()),
                "oracle_fmv_t_eff_mean": float(dataframe.loc[dataframe["is_large_motion_artifact"], "oracle_fmv_t_eff_mean"].mean()),
                "psnr_mean": float(dataframe.loc[dataframe["is_large_motion_artifact"], "psnr_summary"].mean()),
            },
            {
                "group": "oracle_t_eff_artifact",
                "samples": int(dataframe["is_oracle_t_eff_artifact"].sum()),
                "ratio": float(dataframe["is_oracle_t_eff_artifact"].mean()),
                "motion_mean": float(dataframe.loc[dataframe["is_oracle_t_eff_artifact"], "motion_magnitude_mean"].mean()),
                "oracle_fmv_t_eff_mean": float(dataframe.loc[dataframe["is_oracle_t_eff_artifact"], "oracle_fmv_t_eff_mean"].mean()),
                "psnr_mean": float(dataframe.loc[dataframe["is_oracle_t_eff_artifact"], "psnr_summary"].mean()),
            },
        )
    )
    return pd.DataFrame(rows)


def write_thresholds_csv(threshold_row: dict[str, float | str], output_dir: Path) -> None:
    pd.DataFrame([threshold_row]).to_csv(output_dir / "thresholds.csv", index=False)


def write_overlap_csvs(dataframe: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    dataframe.to_csv(output_dir / "artifact_overlap_samples.csv", index=False)
    summary.to_csv(output_dir / "artifact_overlap_summary.csv", index=False)


def write_oracle_scatter(
    dataframe: pd.DataFrame,
    threshold_row: dict[str, float | str],
    output_dir: Path,
) -> None:
    oracle_left_threshold = float(threshold_row["oracle_left_threshold"])
    oracle_right_threshold = float(threshold_row["oracle_right_threshold"])
    psnr_threshold = float(threshold_row["psnr_threshold"])

    plt.figure(figsize=(8, 5), dpi=160)
    for overlap_label in OVERLAP_ORDER:
        label_dataframe = dataframe[dataframe["overlap_label"] == overlap_label]
        if len(label_dataframe) == 0:
            continue
        plt.scatter(
            label_dataframe["oracle_fmv_t_eff_mean"],
            label_dataframe["psnr_summary"],
            s=12,
            alpha=0.45,
            edgecolors="none",
            label=f"{overlap_label} (n={len(label_dataframe)})",
            color=OVERLAP_COLORS[overlap_label],
        )

    plt.axvline(
        oracle_left_threshold,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"oracle left = {oracle_left_threshold:.3f}",
    )
    plt.axvline(
        oracle_right_threshold,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"oracle right = {oracle_right_threshold:.3f}",
    )
    plt.axhline(
        psnr_threshold,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=f"psnr threshold = {psnr_threshold:.2f}",
    )
    plt.xlabel("oracle_fmv_t_eff_mean")
    plt.ylabel("psnr_summary")
    plt.title(
        "Oracle t_eff vs PSNR artifact overlap\n"
        f"oracle_t_eff_artifact: t <= {oracle_left_threshold:.3f} or t >= {oracle_right_threshold:.3f}"
    )
    plt.grid(True, alpha=0.25)
    plt.legend(markerscale=1.4)
    plt.tight_layout()
    plt.savefig(output_dir / "oracle_t_eff_artifact_scatter.png")
    plt.close()


def write_oracle_binned_mean_plot(
    dataframe: pd.DataFrame,
    threshold_row: dict[str, float | str],
    bin_count: int,
    output_dir: Path,
) -> None:
    oracle_left_threshold = float(threshold_row["oracle_left_threshold"])
    oracle_right_threshold = float(threshold_row["oracle_right_threshold"])
    psnr_threshold = float(threshold_row["psnr_threshold"])

    working = dataframe.copy()
    working["oracle_bin"] = pd.qcut(working["oracle_fmv_t_eff_mean"], q=min(bin_count, working["oracle_fmv_t_eff_mean"].nunique()), duplicates="drop")
    summary = (
        working.groupby("oracle_bin", observed=True)
        .agg(
            t_eff_mean=("oracle_fmv_t_eff_mean", "mean"),
            psnr_mean=("psnr_summary", "mean"),
            samples=("psnr_summary", "size"),
        )
        .reset_index()
    )
    summary["oracle_bin"] = summary["oracle_bin"].astype(str)
    summary.to_csv(output_dir / "oracle_t_eff_binned_mean_psnr.csv", index=False)

    plt.figure(figsize=(8, 5), dpi=160)
    plt.plot(summary["t_eff_mean"], summary["psnr_mean"], marker="o", linewidth=1.6, color="#444444")
    plt.axvline(oracle_left_threshold, color="black", linestyle="--", linewidth=1.0)
    plt.axvline(oracle_right_threshold, color="black", linestyle="--", linewidth=1.0)
    plt.axhline(psnr_threshold, color="gray", linestyle="--", linewidth=1.0)
    plt.xlabel("mean oracle_fmv_t_eff_mean in bin")
    plt.ylabel("mean psnr")
    plt.title("Mean PSNR by oracle_fmv_t_eff_mean bin")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "oracle_t_eff_binned_mean_psnr.png")
    plt.close()


def write_overlap_coverage_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered_summary = summary[summary["group"].isin(OVERLAP_ORDER)].copy()
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
    plt.title("Artifact coverage with overlap")
    plt.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    plt.tight_layout()
    plt.savefig(output_dir / "artifact_overlap_coverage.png")
    plt.close()


def run(args: argparse.Namespace) -> None:
    require_quantile(args.motion_quantile, "motion-quantile")
    require_quantile(args.psnr_quantile, "psnr-quantile")
    require_quantile(args.oracle_deviation_quantile, "oracle-deviation-quantile")
    if args.bin_count < 2:
        raise ValueError(f"bin-count must be at least 2, got {args.bin_count}")

    dataframe = load_dataframe(args.sample_comparison_csv)
    validate_columns(dataframe, args.psnr_column)
    threshold_row = build_threshold_row(
        dataframe=dataframe,
        psnr_column=args.psnr_column,
        motion_quantile=args.motion_quantile,
        psnr_quantile=args.psnr_quantile,
        oracle_deviation_quantile=args.oracle_deviation_quantile,
    )
    overlap = build_overlap_dataframe(dataframe, threshold_row)
    summary = build_overlap_summary(overlap)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_thresholds_csv(threshold_row, output_dir)
    write_overlap_csvs(overlap, summary, output_dir)
    write_oracle_scatter(overlap, threshold_row, output_dir)
    write_oracle_binned_mean_plot(overlap, threshold_row, args.bin_count, output_dir)
    write_overlap_coverage_plot(summary, output_dir)

    print(f"input={args.sample_comparison_csv}")
    print(f"output_dir={output_dir}")
    print(pd.DataFrame([threshold_row]).to_string(index=False))
    print()
    print(summary.to_string(index=False))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
