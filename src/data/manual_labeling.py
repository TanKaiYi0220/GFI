from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data.image_ops import flow_to_image
from src.data.image_ops import load_backward_velocity
from src.data.image_ops import load_png

FRAME_NAME_TEMPLATE: str = "colorNoScreenUI_{frame_idx}.png"
VELOCITY_NAME_TEMPLATE: str = "backwardVel_Depth_{frame_idx}.exr"
REJECT_REASON: str = "Rendering BUG (Player/Camera)"


def show_image(image: np.ndarray, image_path: Path, row_index: int, row_count: int, is_valid: bool, window_name: str) -> None:
    """Render one review frame with an overlay border and status text."""
    border_color = (0, 255, 0) if is_valid else (0, 0, 255)
    preview = image.copy()
    cv2.rectangle(preview, (5, 5), (preview.shape[1] - 5, preview.shape[0] - 5), border_color, 3)
    text = f"[{row_index + 1}/{row_count}] {image_path.name} | Status: {is_valid}"
    cv2.putText(preview, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 2)
    cv2.imshow(window_name, preview)


def review_images(reference_df: Any, target_df: Any, dataset_root_dir: Path) -> None:
    """Review paired rows from two dataframes with RGB and optical-flow toggles."""
    row_index = 0
    review_target_index = 1
    view_flow = False
    window_name = "Reviewer"
    row_count = min(len(reference_df), len(target_df))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        reference_row = reference_df.iloc[row_index]
        target_row = target_df.iloc[row_index]
        frame_idx = int(reference_row["frame_idx"])

        reference_dir = build_sequence_directory(dataset_root_dir, reference_row)
        target_dir = build_sequence_directory(dataset_root_dir, target_row)

        reference_image_path = reference_dir / FRAME_NAME_TEMPLATE.format(frame_idx=frame_idx)
        target_image_path = target_dir / FRAME_NAME_TEMPLATE.format(frame_idx=frame_idx)
        reference_velocity_path = reference_dir / VELOCITY_NAME_TEMPLATE.format(frame_idx=frame_idx)
        target_velocity_path = target_dir / VELOCITY_NAME_TEMPLATE.format(frame_idx=frame_idx)

        if review_target_index == 0:
            display_path = reference_image_path
            image = build_review_image(reference_image_path, reference_velocity_path, view_flow)
        else:
            display_path = target_image_path
            image = build_review_image(target_image_path, target_velocity_path, view_flow)

        is_valid = bool(reference_row["is_valid"] and target_row["is_valid"])
        show_image(image, display_path, row_index, row_count, is_valid, window_name)

        key = cv2.waitKey(0) & 0xFF
        if key in (82, ord("w")):
            review_target_index = (review_target_index + 1) % 2
        elif key in (84, ord("s")):
            review_target_index = (review_target_index - 1) % 2
        elif key in (83, ord("d")):
            row_index = min(row_index + 1, row_count - 1)
        elif key in (81, ord("a")):
            row_index = max(row_index - 1, 0)
        elif key in (ord("f"), ord("F")):
            view_flow = not view_flow
        elif key in (ord("y"), ord("Y")):
            set_review_status(reference_df, target_df, row_index, True, "")
            row_index = min(row_index + 1, row_count - 1)
        elif key in (ord("n"), ord("N")):
            set_review_status(reference_df, target_df, row_index, False, REJECT_REASON)
            row_index = min(row_index + 1, row_count - 1)
        elif key in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyAllWindows()


def build_sequence_directory(dataset_root_dir: Path, row: Any) -> Path:
    """Build the source directory for one review row."""
    return dataset_root_dir / str(row["record"]) / str(row["mode"])


def build_review_image(image_path: Path, velocity_path: Path, view_flow: bool) -> np.ndarray:
    """Build either the RGB frame or the flow preview for one row."""
    if not view_flow:
        return load_png(image_path)

    flow, _depth = load_backward_velocity(velocity_path)
    return flow_to_image(flow)


def set_review_status(reference_df: Any, target_df: Any, row_index: int, is_valid: bool, reason: str) -> None:
    """Apply one review decision to both paired dataframes."""
    reference_df.at[row_index, "is_valid"] = is_valid
    target_df.at[row_index, "is_valid"] = is_valid
    reference_df.at[row_index, "reason"] = reason
    target_df.at[row_index, "reason"] = reason
