from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.analysis import summarize_input_distribution
from src.data.dataset_config import ACTIVE_DATASET_ROOT_KEY
from src.data.dataset_config import build_sequence_directory
from src.data.dataset_config import get_dataset_preset
from src.data.dataset_config import iter_dataset_configs
from src.data.dataset_config import list_dataset_presets
from src.data.dataset_config import resolve_active_dataset_root
from src.data.dataset import DatasetSample
from src.data.dataset import collect_samples_from_directories
from src.data.dataset import resolve_split_directory

INPUT_FRAME_NAMES: list[str] = ["frame_000.png", "frame_002.png"]
TARGET_FRAME_NAME: str = "frame_001.png"


def parse_args() -> argparse.Namespace:
    """Parse dataset analysis arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Dataset analysis template.")
    parser.add_argument("--dataset-root", required=False, help="Root directory that contains train, val, or test folders.")
    parser.add_argument("--split", required=False, help="Split name under the dataset root, such as train or val.")
    parser.add_argument("--dataset-preset", required=False, help=f"Named preset from dataset_config.py. Available presets: {', '.join(list_dataset_presets())}.")
    parser.add_argument("--paths-config", required=False, help="Optional path to the shared paths config YAML file.")
    parser.add_argument("--mode", required=True, choices=["dry-run", "summary", "custom"], help="Use 'summary' for the built-in distribution report or 'custom' after filling the template hooks.")
    return parser.parse_args()


def main() -> int:
    """Load a manifest and run the selected analysis mode."""
    args: argparse.Namespace = parse_args()
    dataset_root_arg: str | None = args.dataset_root
    split_name: str | None = args.split
    dataset_preset_name: str | None = args.dataset_preset
    paths_config_path: Path | None = Path(args.paths_config) if args.paths_config is not None else None

    if args.mode == "dry-run":
        if dataset_preset_name is not None:
            sequence_directories: list[Path] = collect_preset_directories(
                dataset_preset_name=dataset_preset_name,
                paths_config_path=paths_config_path,
            )
            print(f"dataset_preset={dataset_preset_name}")
            print(f"active_root_key={ACTIVE_DATASET_ROOT_KEY}")
            print(f"sequence_count={len(sequence_directories)}")
            print(f"first_sequence_dir={sequence_directories[0]}")
        else:
            dataset_root: Path = require_dataset_root(dataset_root_arg=dataset_root_arg)
            resolved_split_name: str = require_split_name(split_name=split_name)
            print(f"dataset_root={dataset_root}")
            print(f"split={resolved_split_name}")
        print(f"input_frame_names={INPUT_FRAME_NAMES}")
        print(f"target_frame_name={TARGET_FRAME_NAME}")
        return 0

    if dataset_preset_name is not None:
        dataset_preset = get_dataset_preset(preset_name=dataset_preset_name)
        root_dir: Path = resolve_active_dataset_root(paths_config_path=paths_config_path)
        dataset_configs = list(iter_dataset_configs(dataset_preset=dataset_preset))

        if args.mode == "summary":
            fps_values: list[int] = sorted({dataset_config.fps for dataset_config in dataset_configs})
            difficulties: list[str] = sorted({dataset_config.difficulty for dataset_config in dataset_configs})
            print(f"preset_name={dataset_preset.name}")
            print(f"active_root_key={ACTIVE_DATASET_ROOT_KEY}")
            print(f"root_dir={root_dir}")
            print(f"sequence_count={len(dataset_configs)}")
            print(f"fps_values={fps_values}")
            print(f"difficulties={difficulties}")
            return 0

        result = run_preset_analysis_template(
            dataset_preset_name=dataset_preset_name,
            paths_config_path=paths_config_path,
        )
        print(result)
        return 0

    dataset_root: Path = require_dataset_root(dataset_root_arg=dataset_root_arg)
    resolved_split_name: str = require_split_name(split_name=split_name)
    split_root: Path = resolve_split_directory(dataset_root=dataset_root, split_name=resolved_split_name)
    samples: list[DatasetSample] = collect_samples_from_directories(
        split_root=split_root,
        split_name=resolved_split_name,
        input_frame_names=INPUT_FRAME_NAMES,
        target_frame_name=TARGET_FRAME_NAME,
    )

    if args.mode == "summary":
        summary = summarize_input_distribution(samples=samples)
        print(f"sample_count={summary['sample_count']}")
        print(f"min_input_frames={summary['min_input_frames']}")
        print(f"max_input_frames={summary['max_input_frames']}")
        print(f"avg_input_frames={summary['avg_input_frames']}")
        print(f"split_counts={summary['split_counts']}")
        return 0

    result: dict[str, Any] = run_custom_analysis_template(samples=samples)
    print(result)
    return 0


def run_custom_analysis_template(samples: list[object]) -> dict[str, Any]:
    """Run the custom analysis template after the project-specific hooks are implemented."""
    analysis_records: list[object] = collect_analysis_records(samples=samples)
    analysis_result: dict[str, Any] = compute_custom_analysis(analysis_records=analysis_records)
    render_analysis_output(analysis_result=analysis_result)
    return analysis_result


def run_preset_analysis_template(dataset_preset_name: str, paths_config_path: Path | None) -> dict[str, Any]:
    """Run the custom analysis template for one dataset preset."""
    sequence_directories: list[Path] = collect_preset_directories(
        dataset_preset_name=dataset_preset_name,
        paths_config_path=paths_config_path,
    )
    analysis_records: list[object] = collect_sequence_analysis_records(sequence_directories=sequence_directories)
    analysis_result: dict[str, Any] = compute_custom_analysis(analysis_records=analysis_records)
    render_analysis_output(analysis_result=analysis_result)
    return analysis_result


def collect_analysis_records(samples: list[object]) -> list[object]:
    """Replace this hook with your record collection logic."""
    raise NotImplementedError(
        "Implement collect_analysis_records() in scripts/analyze_dataset.py with your analysis record extraction logic.",
    )


def collect_sequence_analysis_records(sequence_directories: list[Path]) -> list[object]:
    """Replace this hook with your preset-based record collection logic."""
    raise NotImplementedError(
        "Implement collect_sequence_analysis_records() in scripts/analyze_dataset.py with your preset-based analysis record extraction logic.",
    )


def compute_custom_analysis(analysis_records: list[object]) -> dict[str, Any]:
    """Replace this hook with your actual analysis logic."""
    raise NotImplementedError(
        "Implement compute_custom_analysis() in scripts/analyze_dataset.py with your custom analysis logic.",
    )


def render_analysis_output(analysis_result: dict[str, Any]) -> None:
    """Replace this hook with your custom plotting or export logic."""
    raise NotImplementedError(
        "Implement render_analysis_output() in scripts/analyze_dataset.py with your plotting or export logic.",
    )


def collect_preset_directories(dataset_preset_name: str, paths_config_path: Path | None) -> list[Path]:
    """Resolve all sequence directories defined by one dataset preset."""
    dataset_preset = get_dataset_preset(preset_name=dataset_preset_name)
    sequence_directories: list[Path] = []

    for dataset_config in iter_dataset_configs(dataset_preset=dataset_preset):
        sequence_directories.append(
            build_sequence_directory(dataset_config=dataset_config, paths_config_path=paths_config_path),
        )

    return sequence_directories


def require_dataset_root(dataset_root_arg: str | None) -> Path:
    """Resolve the dataset root for split-based scanning."""
    return Path(dataset_root_arg or "")


def require_split_name(split_name: str | None) -> str:
    """Resolve the split name for split-based scanning."""
    return split_name or ""


if __name__ == "__main__":
    raise SystemExit(main())
