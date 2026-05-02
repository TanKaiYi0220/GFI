from __future__ import annotations

from pathlib import Path
from typing import TypedDict


class DatasetSample(TypedDict):
    sample_id: str
    input_frames: list[str]
    target_frame: str
    split: str
    metadata: dict[str, object]


class ModelInputRecord(TypedDict):
    sample_id: str
    source_paths: list[str]
    target_path: str


def build_sample(
    sample_id: str,
    input_frames: list[str],
    target_frame: str,
    split: str,
    metadata: dict[str, object],
) -> DatasetSample:
    """Build a dataset sample."""
    normalized_metadata: dict[str, object] = {}
    for key, value in metadata.items():
        normalized_metadata[key] = value

    return DatasetSample(
        sample_id=sample_id,
        input_frames=list(input_frames),
        target_frame=target_frame,
        split=split,
        metadata=normalized_metadata,
    )


def resolve_split_directory(dataset_root: Path, split_name: str) -> Path:
    """Resolve the split directory under the dataset root."""
    split_root: Path = dataset_root / split_name
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory does not exist: '{split_root}'.")

    if not split_root.is_dir():
        raise NotADirectoryError(f"Split path is not a directory: '{split_root}'.")

    return split_root


def collect_samples_from_directories(
    split_root: Path,
    split_name: str,
    input_frame_names: list[str],
    target_frame_name: str,
) -> list[DatasetSample]:
    """Collect dataset samples from one-directory-per-sample layouts."""
    samples: list[DatasetSample] = []
    for sample_dir in sorted(split_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        input_paths: list[str] = []
        for frame_name in input_frame_names:
            frame_path: Path = sample_dir / frame_name
            if not frame_path.exists():
                raise FileNotFoundError(
                    f"Missing input frame '{frame_name}' for sample '{sample_dir.name}' in '{sample_dir}'.",
                )
            input_paths.append(str(frame_path))

        target_path: Path = sample_dir / target_frame_name
        if not target_path.exists():
            raise FileNotFoundError(
                f"Missing target frame '{target_frame_name}' for sample '{sample_dir.name}' in '{sample_dir}'.",
            )

        samples.append(
            build_sample(
                sample_id=sample_dir.name,
                input_frames=input_paths,
                target_frame=str(target_path),
                split=split_name,
                metadata={"sample_dir": str(sample_dir)},
            ),
        )

    return samples


def build_model_input(sample: DatasetSample) -> ModelInputRecord:
    """Convert a dataset sample into a model-facing record."""
    return ModelInputRecord(
        sample_id=sample["sample_id"],
        source_paths=list(sample["input_frames"]),
        target_path=sample["target_frame"],
    )
