from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from src.data.analysis import show_images_switchable

IDENTICAL_PSNR_THRESHOLD: float = 48.0
PNG_DATA_RANGE: float = 255.0


def identical_images(image_a: np.ndarray, image_b: np.ndarray) -> bool:
    """Return whether two images should be treated as visually identical."""
    if image_a.shape != image_b.shape:
        return False

    if np.array_equal(image_a, image_b):
        return True

    difference = calculate_psnr(image_a, image_b, PNG_DATA_RANGE)
    return difference >= IDENTICAL_PSNR_THRESHOLD


def visualize_color_difference(image_a: np.ndarray, image_b: np.ndarray) -> None:
    """Show two images and their absolute color difference."""
    difference = calculate_psnr(image_a, image_b, PNG_DATA_RANGE)
    color_difference = cv2.absdiff(image_a, image_b)
    show_images_switchable(
        [image_a, image_b, color_difference],
        ["Image 1", "Image 2", f"Color Difference (PSNR: {difference:.2f} dB)"],
    )


def load_png(image_path: Path) -> np.ndarray:
    """Load one PNG image and convert the sibling EXR file on demand when needed."""
    resolved_image_path = image_path.expanduser().resolve()
    if not resolved_image_path.exists():
        convert_exr_to_png(resolved_image_path.with_suffix(".exr"), resolved_image_path)

    image = cv2.imread(str(resolved_image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image: {resolved_image_path}")

    return image


def load_backward_velocity(velocity_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one backward velocity field and the matching depth map."""
    exr_data = load_exr(velocity_path)
    height, width, _ = exr_data.shape

    motion_1_to_0 = np.stack([exr_data[..., 2], exr_data[..., 1]], axis=-1)
    motion_1_to_0[..., 0] = -1.0 * width * motion_1_to_0[..., 0]
    motion_1_to_0[..., 1] = height * motion_1_to_0[..., 1]
    depth_0 = exr_data[..., 0]
    return motion_1_to_0, depth_0


def load_exr(image_path: Path) -> np.ndarray:
    """Load one EXR image as a float array."""
    resolved_image_path = image_path.expanduser().resolve()
    exr_image = cv2.imread(str(resolved_image_path), cv2.IMREAD_UNCHANGED)
    if exr_image is None:
        raise ValueError(f"Failed to load EXR image: {resolved_image_path}")

    return exr_image


def convert_exr_to_png(source_path: Path, target_path: Path) -> None:
    """Convert one EXR image into one PNG image."""
    exr_image = load_exr(source_path)
    png_image = np.clip(exr_image * PNG_DATA_RANGE, 0, PNG_DATA_RANGE).astype(np.uint8)
    save_image(target_path, png_image)


def save_image(image_path: Path, image: np.ndarray) -> None:
    """Write one image to disk and create the parent directory when needed."""
    resolved_image_path = image_path.expanduser().resolve()
    resolved_image_path.parent.mkdir(parents=True, exist_ok=True)
    is_written = cv2.imwrite(str(resolved_image_path), image)
    if not is_written:
        raise ValueError(f"Failed to save image: {resolved_image_path}")


def calculate_psnr(image_a: np.ndarray, image_b: np.ndarray, data_range: float) -> float:
    """Compute PSNR with an infinity result for zero-error pairs."""
    image_a_float = image_a.astype(np.float32)
    image_b_float = image_b.astype(np.float32)
    mse = float(np.mean((image_a_float - image_b_float) ** 2))
    if mse == 0.0:
        return float("inf")

    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def make_colorwheel() -> np.ndarray:
    """Generate the classic Middlebury optical-flow color wheel."""
    ry = 15
    yg = 6
    gc = 4
    cb = 11
    bm = 13
    mr = 6

    ncols = ry + yg + gc + cb + bm + mr
    colorwheel = np.zeros((ncols, 3))
    column_index = 0

    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.floor(255 * np.arange(0, ry) / ry)
    column_index += ry

    colorwheel[column_index : column_index + yg, 0] = 255 - np.floor(255 * np.arange(0, yg) / yg)
    colorwheel[column_index : column_index + yg, 1] = 255
    column_index += yg

    colorwheel[column_index : column_index + gc, 1] = 255
    colorwheel[column_index : column_index + gc, 2] = np.floor(255 * np.arange(0, gc) / gc)
    column_index += gc

    colorwheel[column_index : column_index + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    colorwheel[column_index : column_index + cb, 2] = 255
    column_index += cb

    colorwheel[column_index : column_index + bm, 2] = 255
    colorwheel[column_index : column_index + bm, 0] = np.floor(255 * np.arange(0, bm) / bm)
    column_index += bm

    colorwheel[column_index : column_index + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    colorwheel[column_index : column_index + mr, 0] = 255
    return colorwheel


def flow_uv_to_colors(u: np.ndarray, v: np.ndarray, convert_to_bgr: bool = False) -> np.ndarray:
    """Apply the optical-flow color wheel to horizontal and vertical flow channels."""
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(-v, -u) / np.pi
    fk = (angle + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    interpolation = fk - k0

    for channel_index in range(colorwheel.shape[1]):
        color_values = colorwheel[:, channel_index]
        color0 = color_values[k0] / 255.0
        color1 = color_values[k1] / 255.0
        blended_color = (1 - interpolation) * color0 + interpolation * color1
        is_in_range = rad <= 1
        blended_color[is_in_range] = 1 - rad[is_in_range] * (1 - blended_color[is_in_range])
        blended_color[~is_in_range] = blended_color[~is_in_range] * 0.75
        output_channel = 2 - channel_index if convert_to_bgr else channel_index
        flow_image[:, :, output_channel] = np.floor(255 * blended_color)

    return flow_image


def flow_to_image(flow_uv: np.ndarray, clip_flow: float | None = None, convert_to_bgr: bool = False) -> np.ndarray:
    """Convert one flow field into one RGB or BGR visualization image."""
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)
