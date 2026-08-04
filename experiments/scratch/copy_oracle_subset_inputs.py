from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_SOURCE_DATASET_ROOT: Path = Path("/workspace/datasets")
DEFAULT_INDEX_ROOT: Path = Path("data/VFX_0416_oracleTime_unbalanced_0p35_0p65")
DEFAULT_PRESETS: tuple[str, ...] = ("train_vfx_0416", "test_vfx_0416")
REQUIRED_RAW_SEQUENCE_COLUMNS: tuple[str, ...] = ("img0", "img1", "img2")
MANIFEST_COLUMNS: tuple[str, ...] = (
    "source_path",
    "destination_path",
    "record",
    "mode",
    "frame_index",
    "modality",
)


def find_project_root(script_path: Path) -> Path:
    for candidate_path in script_path.resolve().parents:
        if (candidate_path / "src" / "data" / "dataset_config.py").is_file():
            return candidate_path

    raise RuntimeError(f"Could not find project root from script_path={script_path}")


PROJECT_ROOT: Path = find_project_root(script_path=Path(__file__))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import DatasetConfig
from src.data.dataset_config import get_dataset_preset
from src.data.dataset_config import iter_dataset_configs


@dataclass(frozen=True)
class ModalitySpec:
    prefix: str
    extension: str


@dataclass(frozen=True)
class CopyPlan:
    source_path: Path
    destination_path: Path
    record: str
    mode: str
    frame_index: int
    modality: str


@dataclass(frozen=True)
class CopyResult:
    copied_count: int
    skipped_count: int


MODALITY_SPECS: dict[str, ModalitySpec] = {
    "colorNoScreenUI": ModalitySpec(prefix="colorNoScreenUI_", extension=".png"),
    "backwardVel_Depth": ModalitySpec(prefix="backwardVel_Depth_", extension=".exr"),
    "forwardVel_Depth": ModalitySpec(prefix="forwardVel_Depth_", extension=".exr"),
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the image/EXR inputs referenced by the oracle-time unbalanced subset into a portable "
            "handover package."
        )
    )
    parser.add_argument(
        "destination_root",
        type=Path,
        help="Destination package folder. The script creates dataset_root/ and the CSV index folder inside it.",
    )
    parser.add_argument(
        "--source-dataset-root",
        default=DEFAULT_SOURCE_DATASET_ROOT,
        type=Path,
        help="Root containing ARPG_* asset folders, equivalent to train.py --dataset-root-dir.",
    )
    parser.add_argument(
        "--root-dir",
        "--index-root",
        dest="index_root",
        type=Path,
        default=DEFAULT_INDEX_ROOT,
        help="Filtered dataloader CSV root, equivalent to train.py --root-dir.",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=list(DEFAULT_PRESETS),
        help="Dataset presets to package.",
    )
    parser.add_argument(
        "--only-fps",
        type=int,
        default=60,
        help="FPS value to read from the index root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files. Without this, matching existing files are skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the package and print counts without copying files.",
    )
    return parser.parse_args(argv)


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"{label} must be an existing directory: {path}")


def build_raw_sequence_csv_path(index_root: Path, dataset_config: DatasetConfig) -> Path:
    return (
        index_root
        / f"{dataset_config.record_name}_preprocessed"
        / f"{dataset_config.mode_index}_raw_sequence_frame_index.csv"
    )


def read_raw_sequence_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Raw sequence CSV not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_RAW_SEQUENCE_COLUMNS if column not in dataframe.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"Raw sequence CSV missing columns {missing_columns}: {csv_path}")

    return dataframe


def build_modality_relative_path(record: str, mode: str, frame_index: int, modality: str) -> Path:
    modality_spec = MODALITY_SPECS[modality]
    filename = f"{modality_spec.prefix}{frame_index}{modality_spec.extension}"
    return Path(record) / Path(mode) / filename


def build_copy_plan(
    source_dataset_root: Path,
    destination_dataset_root: Path,
    record: str,
    mode: str,
    frame_index: int,
    modality: str,
) -> CopyPlan:
    relative_path = build_modality_relative_path(
        record=record,
        mode=mode,
        frame_index=frame_index,
        modality=modality,
    )
    return CopyPlan(
        source_path=source_dataset_root / relative_path,
        destination_path=destination_dataset_root / relative_path,
        record=record,
        mode=mode,
        frame_index=frame_index,
        modality=modality,
    )


def build_row_copy_plans(
    row: pd.Series,
    source_dataset_root: Path,
    destination_dataset_root: Path,
    record: str,
    mode_60: str,
) -> list[CopyPlan]:
    frame_60_0_idx = int(row["img0"])
    frame_60_1_idx = int(row["img1"])
    frame_60_2_idx = int(row["img2"])
    frame_30_0_idx = frame_60_0_idx // 2
    frame_30_1_idx = frame_60_2_idx // 2
    mode_30 = mode_60.replace("fps_60", "fps_30")

    required_assets: tuple[tuple[str, int, str], ...] = (
        (mode_60, frame_60_0_idx, "colorNoScreenUI"),
        (mode_60, frame_60_1_idx, "colorNoScreenUI"),
        (mode_60, frame_60_2_idx, "colorNoScreenUI"),
        (mode_60, frame_60_1_idx, "backwardVel_Depth"),
        (mode_60, frame_60_1_idx, "forwardVel_Depth"),
        (mode_30, frame_30_1_idx, "backwardVel_Depth"),
        (mode_30, frame_30_0_idx, "forwardVel_Depth"),
    )

    return [
        build_copy_plan(
            source_dataset_root=source_dataset_root,
            destination_dataset_root=destination_dataset_root,
            record=record,
            mode=mode,
            frame_index=frame_index,
            modality=modality,
        )
        for mode, frame_index, modality in required_assets
    ]


def build_copy_plans_for_config(
    index_root: Path,
    source_dataset_root: Path,
    destination_dataset_root: Path,
    dataset_config: DatasetConfig,
) -> list[CopyPlan]:
    csv_path = build_raw_sequence_csv_path(index_root=index_root, dataset_config=dataset_config)
    dataframe = read_raw_sequence_dataframe(csv_path=csv_path)
    plans: list[CopyPlan] = []

    for _row_index, row in dataframe.iterrows():
        plans.extend(
            build_row_copy_plans(
                row=row,
                source_dataset_root=source_dataset_root,
                destination_dataset_root=destination_dataset_root,
                record=dataset_config.record,
                mode_60=dataset_config.mode_path,
            )
        )

    return deduplicate_copy_plans(plans=plans)


def build_copy_plans(
    index_root: Path,
    source_dataset_root: Path,
    destination_dataset_root: Path,
    preset_names: Sequence[str],
    only_fps: int,
) -> list[CopyPlan]:
    plans: list[CopyPlan] = []
    for preset_name in preset_names:
        dataset_preset = get_dataset_preset(preset_name=preset_name)
        for dataset_config in iter_dataset_configs(dataset_preset=dataset_preset):
            if dataset_config.fps != only_fps:
                continue
            plans.extend(
                build_copy_plans_for_config(
                    index_root=index_root,
                    source_dataset_root=source_dataset_root,
                    destination_dataset_root=destination_dataset_root,
                    dataset_config=dataset_config,
                )
            )

    return deduplicate_copy_plans(plans=plans)


def deduplicate_copy_plans(plans: Sequence[CopyPlan]) -> list[CopyPlan]:
    deduplicated_plans: dict[Path, CopyPlan] = {}
    for plan in plans:
        existing_plan = deduplicated_plans.get(plan.destination_path)
        if existing_plan is not None and existing_plan.source_path != plan.source_path:
            raise ValueError(
                "Two different sources map to the same destination: "
                f"destination={plan.destination_path} first_source={existing_plan.source_path} "
                f"second_source={plan.source_path}"
            )
        deduplicated_plans[plan.destination_path] = plan

    return list(deduplicated_plans.values())


def copy_file(source_path: Path, destination_path: Path, overwrite_existing: bool, dry_run: bool) -> bool:
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    if destination_path.exists():
        if not destination_path.is_file():
            raise FileExistsError(f"Destination exists and is not a file: {destination_path}")
        if not overwrite_existing:
            if source_path.stat().st_size != destination_path.stat().st_size:
                raise FileExistsError(
                    "Destination file already exists with a different size. "
                    f"Use --overwrite to replace it: source={source_path} destination={destination_path}"
                )
            return False

    if dry_run:
        return False

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return True


def copy_assets(plans: Sequence[CopyPlan], overwrite_existing: bool, dry_run: bool) -> CopyResult:
    copied_count = 0
    skipped_count = 0
    for plan in plans:
        copied = copy_file(
            source_path=plan.source_path,
            destination_path=plan.destination_path,
            overwrite_existing=overwrite_existing,
            dry_run=dry_run,
        )
        if copied:
            copied_count += 1
        else:
            skipped_count += 1

    return CopyResult(copied_count=copied_count, skipped_count=skipped_count)


def copy_index_csvs(index_root: Path, destination_index_root: Path, overwrite_existing: bool, dry_run: bool) -> CopyResult:
    require_directory(path=index_root, label="index_root")
    copied_count = 0
    skipped_count = 0

    for source_path in sorted(index_root.rglob("*.csv")):
        relative_path = source_path.relative_to(index_root)
        destination_path = destination_index_root / relative_path
        copied = copy_file(
            source_path=source_path,
            destination_path=destination_path,
            overwrite_existing=overwrite_existing,
            dry_run=dry_run,
        )
        if copied:
            copied_count += 1
        else:
            skipped_count += 1

    return CopyResult(copied_count=copied_count, skipped_count=skipped_count)


def write_manifest(manifest_path: Path, plans: Sequence[CopyPlan], dry_run: bool) -> None:
    if dry_run:
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for plan in plans:
            writer.writerow(
                {
                    "source_path": str(plan.source_path),
                    "destination_path": str(plan.destination_path),
                    "record": plan.record,
                    "mode": plan.mode,
                    "frame_index": plan.frame_index,
                    "modality": plan.modality,
                }
            )


def print_summary(
    destination_root: Path,
    destination_dataset_root: Path,
    destination_index_root: Path,
    asset_result: CopyResult,
    index_result: CopyResult,
    plans: Sequence[CopyPlan],
    dry_run: bool,
) -> None:
    print(f"destination_root={destination_root}")
    print(f"dataset_root_dir={destination_dataset_root}")
    print(f"root_dir={destination_index_root}")
    print(f"unique_asset_files={len(plans)}")
    print(f"asset_files_copied={asset_result.copied_count}")
    print(f"asset_files_skipped={asset_result.skipped_count}")
    print(f"index_csvs_copied={index_result.copied_count}")
    print(f"index_csvs_skipped={index_result.skipped_count}")
    print(f"dry_run={dry_run}")


def run(args: argparse.Namespace) -> None:
    index_root = args.index_root.resolve()
    source_dataset_root = args.source_dataset_root.resolve()
    destination_root = args.destination_root.resolve()
    destination_dataset_root = destination_root / "dataset_root"
    destination_index_root = destination_root / index_root.name

    require_directory(path=index_root, label="index_root")
    require_directory(path=source_dataset_root, label="source_dataset_root")

    plans = build_copy_plans(
        index_root=index_root,
        source_dataset_root=source_dataset_root,
        destination_dataset_root=destination_dataset_root,
        preset_names=tuple(args.presets),
        only_fps=int(args.only_fps),
    )
    asset_result = copy_assets(
        plans=plans,
        overwrite_existing=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    index_result = copy_index_csvs(
        index_root=index_root,
        destination_index_root=destination_index_root,
        overwrite_existing=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    write_manifest(manifest_path=destination_root / "copy_manifest.csv", plans=plans, dry_run=bool(args.dry_run))
    print_summary(
        destination_root=destination_root,
        destination_dataset_root=destination_dataset_root,
        destination_index_root=destination_index_root,
        asset_result=asset_result,
        index_result=index_result,
        plans=plans,
        dry_run=bool(args.dry_run),
    )


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv=argv)
    run(args=args)


if __name__ == "__main__":
    main(argv=sys.argv[1:])
