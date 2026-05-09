from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import DatasetConfig
from src.data.dataset_config import get_dataset_preset
from src.data.dataset_config import iter_dataset_configs
from src.data.dataset_config import resolve_active_dataset_root

FRAME_PATTERNS: tuple[str, ...] = (
    "colorNoScreenUI_*.exr",
    "backwardVel_Depth_*.exr",
    "forwardVel_Depth_*.exr",
)


@dataclass(frozen=True)
class ScanIssue:
    issue_type: str
    file_path: str
    record: str
    mode: str
    nan_count: int
    inf_count: int
    dtype: str
    shape: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check EXR files for NaN or Inf values before dataset preprocessing.")
    parser.add_argument("--preset", required=False, default="train_vfx_0416", help="Dataset preset key to scan.")
    parser.add_argument("--paths-config", required=False, type=str, help="Optional path to configs/paths/default.yaml.")
    parser.add_argument("--dataset-root-dir", required=False, type=str, help="Override dataset root directory.")
    parser.add_argument("--only-fps", required=False, type=int, help="Optional FPS filter.")
    parser.add_argument("--limit", required=False, default=20, type=int, help="Maximum number of issues to print in detail.")
    return parser.parse_args()


def resolve_dataset_root_dir(dataset_root_dir: str | None, paths_config: str | None) -> Path:
    if dataset_root_dir is not None:
        return Path(dataset_root_dir)

    paths_config_path = Path(paths_config) if paths_config is not None else None
    return resolve_active_dataset_root(paths_config_path)


def get_target_configs(preset_name: str, only_fps: int | None) -> list[DatasetConfig]:
    dataset_preset = get_dataset_preset(preset_name)
    dataset_configs = list(iter_dataset_configs(dataset_preset))
    if only_fps is None:
        return dataset_configs

    return [dataset_config for dataset_config in dataset_configs if dataset_config.fps == only_fps]


def build_mode_root(dataset_root_dir: Path, dataset_config: DatasetConfig) -> Path:
    return dataset_root_dir / dataset_config.record / dataset_config.mode_path


def count_non_finite_values(array: np.ndarray) -> tuple[int, int]:
    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())
    return nan_count, inf_count


def build_scan_issue(
    issue_type: str,
    file_path: Path,
    dataset_config: DatasetConfig,
    nan_count: int,
    inf_count: int,
    dtype: str,
    shape: str,
) -> ScanIssue:
    return ScanIssue(
        issue_type=issue_type,
        file_path=str(file_path),
        record=dataset_config.record,
        mode=dataset_config.mode_path,
        nan_count=nan_count,
        inf_count=inf_count,
        dtype=dtype,
        shape=shape,
    )


def scan_exr_file(file_path: Path, dataset_config: DatasetConfig) -> ScanIssue | None:
    image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return build_scan_issue(
            issue_type="load_failed",
            file_path=file_path,
            dataset_config=dataset_config,
            nan_count=-1,
            inf_count=-1,
            dtype="unknown",
            shape="unknown",
        )

    nan_count, inf_count = count_non_finite_values(image)
    if nan_count == 0 and inf_count == 0:
        return None

    return build_scan_issue(
        issue_type="non_finite",
        file_path=file_path,
        dataset_config=dataset_config,
        nan_count=nan_count,
        inf_count=inf_count,
        dtype=str(image.dtype),
        shape=str(tuple(image.shape)),
    )


def scan_mode_files(dataset_root_dir: Path, dataset_config: DatasetConfig) -> tuple[int, list[ScanIssue]]:
    mode_root = build_mode_root(dataset_root_dir, dataset_config)
    if not mode_root.is_dir():
        raise FileNotFoundError(f"Missing mode directory: {mode_root}")

    issues: list[ScanIssue] = []
    mode_file_paths: list[Path] = []

    for frame_pattern in FRAME_PATTERNS:
        file_paths = sorted(mode_root.glob(frame_pattern))
        if len(file_paths) == 0:
            issues.append(
                build_scan_issue(
                    issue_type="pattern_missing",
                    file_path=mode_root / frame_pattern,
                    dataset_config=dataset_config,
                    nan_count=0,
                    inf_count=0,
                    dtype="missing",
                    shape="missing",
                )
            )
            continue

        mode_file_paths.extend(file_paths)

    progress = tqdm(
        mode_file_paths,
        desc=f"{dataset_config.record}:{dataset_config.mode_name}",
        leave=True,
    )
    for file_path in progress:
        issue = scan_exr_file(file_path, dataset_config)
        if issue is not None:
            issues.append(issue)
        progress.set_postfix({"issues": len(issues)})

    return len(mode_file_paths), issues


def print_run_summary(dataset_root_dir: Path, dataset_configs: list[DatasetConfig]) -> None:
    print(f"dataset_root_dir={dataset_root_dir}")
    print(f"config_count={len(dataset_configs)}")


def print_issue_summary(issues: list[ScanIssue], limit: int) -> None:
    print(f"issue_count={len(issues)}")
    for issue in issues[:limit]:
        print(
            "issue "
            f"type={issue.issue_type} "
            f"record={issue.record} "
            f"mode={issue.mode} "
            f"nan_count={issue.nan_count} "
            f"inf_count={issue.inf_count} "
            f"dtype={issue.dtype} "
            f"shape={issue.shape} "
            f"path={issue.file_path}",
        )


def main() -> None:
    args = parse_args()
    dataset_root_dir = resolve_dataset_root_dir(args.dataset_root_dir, args.paths_config)
    dataset_configs = get_target_configs(args.preset, args.only_fps)

    print(f"preset={args.preset}")
    if args.only_fps is not None:
        print(f"only_fps={args.only_fps}")
    print_run_summary(dataset_root_dir, dataset_configs)

    scanned_file_count = 0
    all_issues: list[ScanIssue] = []

    for dataset_config in dataset_configs:
        mode_scanned_file_count, mode_issues = scan_mode_files(dataset_root_dir, dataset_config)
        scanned_file_count += mode_scanned_file_count
        all_issues.extend(mode_issues)
        print(
            f"scanned record={dataset_config.record} "
            f"mode={dataset_config.mode_path} "
            f"file_count={mode_scanned_file_count} "
            f"issue_count={len(mode_issues)}",
        )

    print(f"scanned_file_count={scanned_file_count}")
    print_issue_summary(all_issues, args.limit)

    if len(all_issues) > 0:
        raise RuntimeError(f"Found {len(all_issues)} EXR issues before preprocessing.")

    print("No NaN or Inf values found in scanned EXR files.")


if __name__ == "__main__":
    main()
