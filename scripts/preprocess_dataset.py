from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import ACTIVE_DATASET_ROOT_KEY
from src.data.dataset_config import build_sequence_directory
from src.data.dataset_config import get_dataset_preset
from src.data.dataset_config import iter_dataset_configs
from src.data.dataset_config import list_dataset_presets
from src.data.dataset import DatasetSample
from src.data.dataset import collect_samples_from_directories
from src.data.dataset import resolve_split_directory

INPUT_FRAME_NAMES: list[str] = ["frame_000.png", "frame_002.png"]
TARGET_FRAME_NAME: str = "frame_001.png"


def parse_args() -> argparse.Namespace:
    """Parse dataset preprocessing arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Dataset preprocessing template.")
    parser.add_argument("--dataset-root", required=False, help="Root directory that contains train, val, or test folders.")
    parser.add_argument("--split", required=False, help="Split name under the dataset root, such as train or val.")
    parser.add_argument("--dataset-preset", required=False, help=f"Named preset from dataset_config.py. Available presets: {', '.join(list_dataset_presets())}.")
    parser.add_argument("--paths-config", required=False, help="Optional path to the shared paths config YAML file.")
    parser.add_argument("--output-dir", required=True, help="Directory where processed files should be written.")
    parser.add_argument("--mode", required=True, choices=["dry-run", "scan", "run"], help="Use 'scan' to validate dataset layout or 'run' after filling the template hooks.")
    return parser.parse_args()


def main() -> int:
    """Load preprocessing arguments and run the selected mode."""
    args: argparse.Namespace = parse_args()
    output_dir: Path = Path(args.output_dir).resolve()
    dataset_root_arg: str | None = args.dataset_root
    split_name: str | None = args.split
    dataset_preset_name: str | None = args.dataset_preset
    paths_config_path: Path | None = Path(args.paths_config).resolve() if args.paths_config is not None else None

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
        print(f"output_dir={output_dir}")
        print(f"input_frame_names={INPUT_FRAME_NAMES}")
        print(f"target_frame_name={TARGET_FRAME_NAME}")
        return 0

    if dataset_preset_name is not None:
        sequence_directories = collect_preset_directories(
            dataset_preset_name=dataset_preset_name,
            paths_config_path=paths_config_path,
        )
        if args.mode == "scan":
            print(f"sequence_count={len(sequence_directories)}")
            print(f"first_sequence_dir={sequence_directories[0]}")
            return 0

        run_preset_preprocess_template(sequence_directories=sequence_directories, output_dir=output_dir)
        return 0

    dataset_root = require_dataset_root(dataset_root_arg=dataset_root_arg)
    resolved_split_name = require_split_name(split_name=split_name)
    split_root: Path = resolve_split_directory(dataset_root=dataset_root, split_name=resolved_split_name)
    samples: list[DatasetSample] = collect_samples_from_directories(
        split_root=split_root,
        split_name=resolved_split_name,
        input_frame_names=INPUT_FRAME_NAMES,
        target_frame_name=TARGET_FRAME_NAME,
    )

    if args.mode == "scan":
        print(f"sample_count={len(samples)}")
        print(f"first_sample_id={samples[0]['sample_id']}")
        print(f"first_sample_dir={samples[0]['metadata']['sample_dir']}")
        return 0

    run_custom_preprocess_template(samples=samples, output_dir=output_dir)
    return 0


def run_custom_preprocess_template(
    samples: list[DatasetSample],
    output_dir: Path,
) -> None:
    """Run the custom preprocessing template after the project-specific hooks are implemented."""
    for sample in samples:
        preprocess_sample(sample=sample, output_dir=output_dir)


def run_preset_preprocess_template(sequence_directories: list[Path], output_dir: Path) -> None:
    """Run the custom preprocessing template for dataset presets."""
    for sequence_directory in sequence_directories:
        preprocess_sequence(sequence_directory=sequence_directory, output_dir=output_dir)


def preprocess_sample(sample: DatasetSample, output_dir: Path) -> None:
    """Replace this hook with your sample-level preprocessing logic."""
    raise NotImplementedError(
        "Implement preprocess_sample() in scripts/preprocess_dataset.py with your custom preprocessing logic.",
    )


def preprocess_sequence(sequence_directory: Path, output_dir: Path) -> None:
    """Replace this hook with your sequence-level preprocessing logic."""
    raise NotImplementedError(
        "Implement preprocess_sequence() in scripts/preprocess_dataset.py with your preset-based preprocessing logic.",
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
    return Path(dataset_root_arg or "").resolve()


def require_split_name(split_name: str | None) -> str:
    """Resolve the split name for split-based scanning."""
    return split_name or ""


if __name__ == "__main__":
    raise SystemExit(main())
