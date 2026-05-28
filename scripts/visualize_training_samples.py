from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export training sample epoch images to videos.")
    parser.add_argument("--samples-root", required=True, type=str, help="Path to output_dir/samples.")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory for exported videos.")
    parser.add_argument("--splits", required=False, nargs="+", choices=["train", "test"], help="Optional sample splits to export.")
    parser.add_argument("--data-types", required=True, nargs="+", help="Image filenames to export, e.g. image_pred.png flow_mask.png.")
    parser.add_argument("--frame-keys", required=False, nargs="+", help="Optional sample frame keys. Defaults to every key under each split.")
    parser.add_argument("--fps", required=True, type=int, help="Output video FPS.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned videos without writing files.")
    return parser.parse_args(argv)


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def list_frame_keys(split_dir: Path, requested_frame_keys: list[str] | None) -> list[str]:
    if requested_frame_keys is not None:
        return requested_frame_keys
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split samples directory: {split_dir}")
    return sorted(path.name for path in split_dir.iterdir() if path.is_dir())


def parse_epoch_index(epoch_dir: Path) -> int:
    match = re.fullmatch(r"epoch_(\d+)", epoch_dir.name)
    if match is None:
        raise ValueError(f"Epoch directory must look like epoch_0001: {epoch_dir}")
    return int(match.group(1))


def list_epoch_dirs(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Missing frame samples directory: {frame_dir}")
    epoch_dirs = [path for path in frame_dir.iterdir() if path.is_dir() and path.name.startswith("epoch_")]
    if len(epoch_dirs) == 0:
        raise RuntimeError(f"No epoch directories found under: {frame_dir}")
    return sorted(epoch_dirs, key=parse_epoch_index)


def make_even_frame(frame: Any, target_shape: tuple[int, int]) -> Any:
    import numpy as np

    height, width = target_shape
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: frame.shape[0], : frame.shape[1]] = frame
    return canvas


def load_video_frame(image_path: Path) -> Any:
    import cv2

    frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ValueError(f"Failed to load sample image: {image_path}")
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def build_video_path(output_dir: Path, split_name: str | None, frame_key: str, data_type: str) -> Path:
    data_stem = Path(data_type).stem
    if split_name is None:
        return output_dir / frame_key / f"{data_stem}.mp4"
    return output_dir / split_name / frame_key / f"{data_stem}.mp4"


def open_video_writer(output_path: Path, fps: int, frame_shape: tuple[int, int]) -> tuple[Any, Path]:
    import cv2

    height, width = frame_shape
    ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", -1)
    gstreamer_backend = getattr(cv2, "CAP_GSTREAMER", -1)
    mjpeg_backend = getattr(cv2, "CAP_OPENCV_MJPEG", -1)
    candidates = [
        (output_path, "mp4v", ffmpeg_backend, "FFMPEG/mp4v"),
        (output_path, "avc1", ffmpeg_backend, "FFMPEG/avc1"),
        (output_path, "mp4v", gstreamer_backend, "GSTREAMER/mp4v"),
        (output_path.with_suffix(".avi"), "MJPG", mjpeg_backend, "OPENCV_MJPEG/MJPG"),
    ]
    tried_labels: list[str] = []
    for candidate_path, codec, backend, label in candidates:
        if backend < 0:
            continue
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(candidate_path), backend, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        tried_labels.append(label)
        if writer.isOpened():
            return writer, candidate_path
        writer.release()
    raise RuntimeError(f"Failed to open VideoWriter: {output_path}. tried_backends={tried_labels}")


def export_data_type_video(frame_dir: Path, output_path: Path, data_type: str, fps: int) -> Path:
    epoch_dirs = list_epoch_dirs(frame_dir)
    first_frame = load_video_frame(epoch_dirs[0] / data_type)
    height = first_frame.shape[0] if first_frame.shape[0] % 2 == 0 else first_frame.shape[0] + 1
    width = first_frame.shape[1] if first_frame.shape[1] % 2 == 0 else first_frame.shape[1] + 1
    writer, actual_output_path = open_video_writer(output_path, fps, (height, width))
    for epoch_dir in epoch_dirs:
        image_path = epoch_dir / data_type
        if not image_path.is_file():
            writer.release()
            raise FileNotFoundError(f"Missing sample image for epoch: {image_path}")
        writer.write(make_even_frame(load_video_frame(image_path), (height, width)))
    writer.release()
    return actual_output_path


def validate_data_type_video(frame_dir: Path, output_path: Path, data_type: str) -> None:
    epoch_dirs = list_epoch_dirs(frame_dir)
    missing_paths = [epoch_dir / data_type for epoch_dir in epoch_dirs if not (epoch_dir / data_type).is_file()]
    if len(missing_paths) > 0:
        missing_preview = ", ".join(str(path) for path in missing_paths[:5])
        raise FileNotFoundError(f"Missing sample images for {data_type}: {missing_preview}")
    epoch_names = ", ".join(epoch_dir.name for epoch_dir in epoch_dirs)
    print(f"DRY-RUN {output_path} frames={len(epoch_dirs)} epochs={epoch_names}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    samples_root = resolve_project_path(args.samples_root)
    output_dir = resolve_project_path(args.output_dir)
    split_names = [None] if args.splits is None else list(args.splits)
    for split_name in split_names:
        root_dir = samples_root if split_name is None else samples_root / split_name
        for frame_key in list_frame_keys(root_dir, args.frame_keys):
            frame_dir = root_dir / frame_key
            for data_type in args.data_types:
                output_path = build_video_path(output_dir, split_name, frame_key, data_type)
                if args.dry_run:
                    validate_data_type_video(frame_dir, output_path, data_type)
                    continue
                actual_output_path = export_data_type_video(frame_dir, output_path, data_type, args.fps)
                print(actual_output_path)


if __name__ == "__main__":
    main()
