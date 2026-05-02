from __future__ import annotations

from pathlib import Path

from src.data.dataset import DatasetSample


def rewrite_sample_root(samples: list[DatasetSample], source_root: Path, target_root: Path) -> list[DatasetSample]:
    """Rewrite sample paths from one root directory to another."""
    normalized_samples: list[DatasetSample] = []
    for sample in samples:
        normalized_input_frames: list[str] = [
            str(rewrite_path_root(path=Path(frame_path), source_root=source_root, target_root=target_root))
            for frame_path in sample["input_frames"]
        ]
        normalized_target_frame: str = str(
            rewrite_path_root(path=Path(sample["target_frame"]), source_root=source_root, target_root=target_root),
        )
        normalized_sample: DatasetSample = DatasetSample(
            sample_id=sample["sample_id"],
            input_frames=normalized_input_frames,
            target_frame=normalized_target_frame,
            split=sample["split"],
            metadata=dict(sample["metadata"]),
        )
        normalized_samples.append(normalized_sample)

    return normalized_samples


def rewrite_path_root(path: Path, source_root: Path, target_root: Path) -> Path:
    """Rewrite one path from a source root to a target root."""
    resolved_path: Path = path.resolve()
    resolved_source_root: Path = source_root.resolve()
    resolved_target_root: Path = target_root.resolve()
    relative_path: Path = resolved_path.relative_to(resolved_source_root)
    return (resolved_target_root / relative_path).resolve()
