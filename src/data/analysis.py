from __future__ import annotations

from statistics import mean
from typing import TypedDict

from src.data.dataset import DatasetSample


class DistributionSummary(TypedDict):
    sample_count: int
    min_input_frames: int
    max_input_frames: int
    avg_input_frames: float
    split_counts: dict[str, int]


def summarize_input_distribution(samples: list[DatasetSample]) -> DistributionSummary:
    """Summarize the number of input frames and split distribution."""
    input_counts: list[int] = [len(sample["input_frames"]) for sample in samples]
    split_counts: dict[str, int] = {}

    for sample in samples:
        split_name: str = sample["split"]
        split_counts[split_name] = split_counts.get(split_name, 0) + 1

    return DistributionSummary(
        sample_count=len(samples),
        min_input_frames=min(input_counts),
        max_input_frames=max(input_counts),
        avg_input_frames=mean(input_counts),
        split_counts=split_counts,
    )
