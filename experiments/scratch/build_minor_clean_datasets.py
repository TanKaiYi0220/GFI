from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import TEST_MINOR_0507_DATASET_PRESET
from src.data.dataset_config import TRAIN_MINOR_0507_DATASET_PRESET
from src.data.dataset_config import DatasetConfig
from src.data.dataset_config import DatasetPreset
from src.data.dataset_config import iter_dataset_configs

DEFAULT_SOURCE_DATASET_ROOT: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\Dataset\Minor_0507",
)
DEFAULT_DELTA_TIME_OUTPUT_ROOT: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\Dataset\Minor_0507_deltaTime_clean",
)
DEFAULT_ORACLE_TIME_OUTPUT_ROOT: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260618 - Lab Meeting\Dataset\Minor_0507_oracleTime_clean",
)
DEFAULT_CLEANING_SAMPLE_COMPARISON_CSV: Path = Path(
    r"C:\Users\User\Desktop\CGVLab\GFI\code\GFI\analysis_outputs\cleaning_model_comparison\minor_0611\model_cleaning_sample_comparison.csv",
)
DEFAULT_MODEL_NAME: str = "IFRNet FineTuning"
DEFAULT_ONLY_FPS: int = 60
KEY_COLUMNS: tuple[str, ...] = ("record", "major_mode_id", "frame_range")


@dataclass(frozen=True)
class CleanVariant:
    name: str
    mask_column: str
    output_root: Path
    train_preset_name: str
    test_preset_name: str


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Minor_0507 cleaned dataset CSV folders by flipping valid=False for filtered samples."
    )
    parser.add_argument("--source-dataset-root", type=Path, default=DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--delta-time-output-root", type=Path, default=DEFAULT_DELTA_TIME_OUTPUT_ROOT)
    parser.add_argument("--oracle-time-output-root", type=Path, default=DEFAULT_ORACLE_TIME_OUTPUT_ROOT)
    parser.add_argument("--cleaning-sample-comparison-csv", type=Path, default=DEFAULT_CLEANING_SAMPLE_COMPARISON_CSV)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--only-fps", type=int, default=DEFAULT_ONLY_FPS)
    return parser.parse_args(argv)


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_columns(dataframe: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"{label} is missing required columns: {missing_columns}")


def parse_major_mode_id(mode: str) -> str:
    mode_prefix = str(mode).replace("\\", "/").split("/")[0]
    mode_parts = mode_prefix.split("_")
    if len(mode_parts) < 2:
        raise ValueError(f"Unexpected mode format: mode={mode}")
    return mode_parts[0]


def parse_raw_sequence_name(csv_path: Path) -> tuple[str, str]:
    parent_name = csv_path.parent.name
    if not parent_name.endswith("_preprocessed"):
        raise ValueError(f"Expected *_preprocessed parent directory: path={csv_path}")
    record = parent_name.replace("_preprocessed", "")
    stem = csv_path.name.replace("_raw_sequence_frame_index.csv", "")
    stem_parts = stem.split("_")
    if len(stem_parts) != 4 or stem_parts[2] != "fps":
        raise ValueError(f"Unexpected raw-sequence filename format: path={csv_path}")
    return record, stem_parts[0]


def format_frame_range(img0: int, img2: int) -> str:
    return f"frame_{img0:04d}_{img2:04d}"


def build_clean_variants(args: argparse.Namespace) -> list[CleanVariant]:
    return [
        CleanVariant(
            name="deltaTime",
            mask_column="clean_delta_time",
            output_root=args.delta_time_output_root,
            train_preset_name="train_minor_0507_deltaTime_clean",
            test_preset_name="test_minor_0507_deltaTime_clean",
        ),
        CleanVariant(
            name="oracleTime",
            mask_column="clean_oracle_t_eff",
            output_root=args.oracle_time_output_root,
            train_preset_name="train_minor_0507_oracleTime_clean",
            test_preset_name="test_minor_0507_oracleTime_clean",
        ),
    ]


def load_clean_masks(cleaning_sample_comparison_csv: Path, model_name: str) -> pd.DataFrame:
    require_existing_path(cleaning_sample_comparison_csv, "cleaning sample comparison CSV")
    comparison = pd.read_csv(cleaning_sample_comparison_csv)
    require_columns(
        dataframe=comparison,
        columns=("model_name", "record", "mode", "frame_range", "clean_delta_time", "clean_oracle_t_eff"),
        label="cleaning sample comparison CSV",
    )
    model_rows = comparison[comparison["model_name"] == model_name].copy()
    if len(model_rows) == 0:
        raise ValueError(f"No rows found for model_name={model_name}")

    model_rows["major_mode_id"] = model_rows["mode"].map(parse_major_mode_id)
    grouped = (
        model_rows[list(KEY_COLUMNS) + ["clean_delta_time", "clean_oracle_t_eff"]]
        .groupby(list(KEY_COLUMNS), as_index=False)
        .agg(
            clean_delta_time=("clean_delta_time", "min"),
            clean_oracle_t_eff=("clean_oracle_t_eff", "min"),
            clean_delta_time_max=("clean_delta_time", "max"),
            clean_oracle_t_eff_max=("clean_oracle_t_eff", "max"),
            mode_count=("frame_range", "size"),
        )
    )
    delta_conflicts = grouped[grouped["clean_delta_time"] != grouped["clean_delta_time_max"]]
    oracle_conflicts = grouped[grouped["clean_oracle_t_eff"] != grouped["clean_oracle_t_eff_max"]]
    if len(delta_conflicts) > 0 or len(oracle_conflicts) > 0:
        raise ValueError(
            "Cleaning masks conflict across modes sharing the same raw CSV row: "
            f"delta_conflicts={len(delta_conflicts)} oracle_conflicts={len(oracle_conflicts)}"
        )
    return grouped.drop(columns=["clean_delta_time_max", "clean_oracle_t_eff_max"])


def copy_non_raw_csvs(source_root: Path, output_root: Path) -> None:
    for source_path in sorted(source_root.rglob("*.csv")):
        if source_path.name.endswith("_raw_sequence_frame_index.csv"):
            continue
        relative_path = source_path.relative_to(source_root)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)


def clean_raw_sequence_csv(
    source_path: Path,
    output_path: Path,
    clean_masks: pd.DataFrame,
    mask_column: str,
) -> dict[str, object]:
    record, major_mode_id = parse_raw_sequence_name(source_path)
    dataframe = pd.read_csv(source_path)
    require_columns(
        dataframe=dataframe,
        columns=("img0", "img2", "valid"),
        label=f"raw sequence CSV: {source_path}",
    )
    cleaned = dataframe.copy()
    cleaned["frame_range"] = [
        format_frame_range(img0=int(img0), img2=int(img2))
        for img0, img2 in zip(cleaned["img0"], cleaned["img2"])
    ]
    key_masks = clean_masks[
        (clean_masks["record"] == record)
        & (clean_masks["major_mode_id"].astype(str) == str(major_mode_id))
    ][["frame_range", mask_column]]
    if key_masks["frame_range"].duplicated().any():
        raise ValueError(f"Duplicate frame masks for record={record} major_mode_id={major_mode_id}")

    merged = cleaned.merge(key_masks, on="frame_range", how="left", indicator="clean_mask_merge")
    original_valid = merged["valid"].astype(bool)
    missing_valid_masks = merged[original_valid & (merged["clean_mask_merge"] != "both")]
    if len(missing_valid_masks) > 0:
        preview = missing_valid_masks[["record", "img0", "img2", "frame_range"]].head(10).to_dict("records")
        raise ValueError(f"Missing clean masks for valid raw rows: source={source_path} preview={preview}")

    keep_mask = merged[mask_column].fillna(False).astype(bool)
    cleaned["valid"] = original_valid & keep_mask
    turned_false_count = int((original_valid & ~cleaned["valid"]).sum())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.drop(columns=["frame_range"]).to_csv(output_path, index=False)
    return {
        "source_csv": str(source_path),
        "output_csv": str(output_path),
        "rows": int(len(cleaned)),
        "original_valid": int(original_valid.sum()),
        "cleaned_valid": int(cleaned["valid"].sum()),
        "turned_false": turned_false_count,
    }


def build_merged_preset_dataframe(root_dir: Path, dataset_preset: DatasetPreset, only_fps: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset_config in iter_dataset_configs(dataset_preset):
        if dataset_config.fps != only_fps:
            continue
        csv_path = root_dir / f"{dataset_config.record_name}_preprocessed" / f"{dataset_config.mode_index}_raw_sequence_frame_index.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset CSV missing for preset merge: {csv_path}")
        dataframe = pd.read_csv(csv_path)
        dataframe["record"] = dataset_config.record
        dataframe["mode"] = dataset_config.mode_path
        frames.append(dataframe)
    if len(frames) == 0:
        raise RuntimeError(f"No dataset CSVs were loaded: root_dir={root_dir} preset={dataset_preset.name} only_fps={only_fps}")
    return pd.concat(frames, ignore_index=True)


def write_merged_preset_csvs(variant: CleanVariant, only_fps: int) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    preset_pairs: tuple[tuple[str, DatasetPreset], ...] = (
        (variant.train_preset_name, TRAIN_MINOR_0507_DATASET_PRESET),
        (variant.test_preset_name, TEST_MINOR_0507_DATASET_PRESET),
    )
    for preset_name, dataset_preset in preset_pairs:
        merged = build_merged_preset_dataframe(
            root_dir=variant.output_root,
            dataset_preset=dataset_preset,
            only_fps=only_fps,
        )
        output_path = variant.output_root / f"{preset_name}_merged.csv"
        merged.to_csv(output_path, index=False)
        outputs.append(
            {
                "variant": variant.name,
                "preset_name": preset_name,
                "output_csv": str(output_path),
                "rows": int(len(merged)),
                "valid": int(merged["valid"].astype(bool).sum()),
                "invalid": int((~merged["valid"].astype(bool)).sum()),
            }
        )
    return outputs


def build_clean_dataset(variant: CleanVariant, source_root: Path, clean_masks: pd.DataFrame, only_fps: int) -> pd.DataFrame:
    if variant.output_root.exists():
        existing_outputs = list(variant.output_root.rglob("*"))
        if len(existing_outputs) > 0:
            raise FileExistsError(f"Output root already exists and is not empty: {variant.output_root}")
    variant.output_root.mkdir(parents=True, exist_ok=True)
    copy_non_raw_csvs(source_root=source_root, output_root=variant.output_root)

    raw_summary_rows: list[dict[str, object]] = []
    for source_path in sorted(source_root.rglob("*_raw_sequence_frame_index.csv")):
        output_path = variant.output_root / source_path.relative_to(source_root)
        row = clean_raw_sequence_csv(
            source_path=source_path,
            output_path=output_path,
            clean_masks=clean_masks,
            mask_column=variant.mask_column,
        )
        row["variant"] = variant.name
        raw_summary_rows.append(row)

    merged_summary_rows = write_merged_preset_csvs(variant=variant, only_fps=only_fps)
    summary = pd.DataFrame(raw_summary_rows)
    summary.to_csv(variant.output_root / f"{variant.name}_raw_csv_cleaning_summary.csv", index=False)
    pd.DataFrame(merged_summary_rows).to_csv(variant.output_root / f"{variant.name}_preset_merged_summary.csv", index=False)
    return summary


def run(args: argparse.Namespace) -> None:
    require_existing_path(args.source_dataset_root, "source dataset root")
    clean_masks = load_clean_masks(
        cleaning_sample_comparison_csv=args.cleaning_sample_comparison_csv,
        model_name=str(args.model_name),
    )
    all_summaries: list[pd.DataFrame] = []
    for variant in build_clean_variants(args):
        summary = build_clean_dataset(
            variant=variant,
            source_root=args.source_dataset_root,
            clean_masks=clean_masks,
            only_fps=int(args.only_fps),
        )
        all_summaries.append(summary)

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    summary_output_path = PROJECT_ROOT / "analysis_outputs" / "cleaned_minor_0507_dataset_summary.csv"
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_summary.to_csv(summary_output_path, index=False)
    print(f"summary={summary_output_path}")
    print(combined_summary[["variant", "source_csv", "original_valid", "cleaned_valid", "turned_false"]].to_string(index=False))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
