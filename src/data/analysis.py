from __future__ import annotations

from statistics import mean
from typing import TypedDict

import cv2
import numpy as np

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
        split_name = sample["split"]
        split_counts[split_name] = split_counts.get(split_name, 0) + 1

    return DistributionSummary(
        sample_count=len(samples),
        min_input_frames=min(input_counts),
        max_input_frames=max(input_counts),
        avg_input_frames=mean(input_counts),
        split_counts=split_counts,
    )


def show_images_switchable(images: list[np.ndarray], titles: list[str]) -> None:
    """Display a list of images and let the user switch between them."""
    image_index = 0
    image_count = len(images)

    while True:
        preview = images[image_index].copy()
        text = f"[{image_index + 1}/{image_count}] {titles[image_index]}"
        cv2.putText(
            preview,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Overlay", preview)
        key = cv2.waitKey(0) & 0xFF

        if key in [ord("a"), 81, 82]:
            image_index = (image_index - 1) % image_count
        elif key in [ord("d"), 83, 84]:
            image_index = (image_index + 1) % image_count
        elif key in [ord("q"), 27]:
            break

    cv2.destroyAllWindows()
