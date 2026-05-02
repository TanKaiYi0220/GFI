from __future__ import annotations

from typing import Any

import numpy as np


def get_valid_continuous_segments(dataframe: Any) -> list[dict[str, int]]:
    """Return contiguous valid frame segments from one frame-index dataframe."""
    sorted_df = dataframe.sort_values(by="frame_idx").reset_index(drop=True)
    valid_column = "global_is_valid" if "global_is_valid" in sorted_df.columns else "is_valid"
    valid_df = sorted_df[sorted_df[valid_column] == True].reset_index(drop=True)
    if len(valid_df) == 0:
        return []

    frame_indices = valid_df["frame_idx"].to_numpy()
    segments: list[dict[str, int]] = []
    start_index = 0

    for row_index in range(1, len(frame_indices) + 1):
        is_break = row_index == len(frame_indices) or frame_indices[row_index] != frame_indices[row_index - 1] + 1
        if not is_break:
            continue

        start_frame = int(frame_indices[start_index])
        end_frame = int(frame_indices[row_index - 1])
        segments.append(
            {
                "start": start_frame,
                "end": end_frame,
                "length": end_frame - start_frame + 1,
            },
        )
        start_index = row_index

    return segments


def check_valid_in_high_fps(dataframe: Any) -> bool:
    """Return whether the valid rows form one fully continuous high-FPS sequence."""
    segments = get_valid_continuous_segments(dataframe)
    if len(segments) == 0:
        return False

    valid_column = "global_is_valid" if "global_is_valid" in dataframe.columns else "is_valid"
    valid_count = int(np.sum(dataframe[valid_column].astype(bool).to_numpy()))
    segment_length_sum = int(sum(segment["length"] for segment in segments))
    return segment_length_sum == valid_count and len(segments) == 1
