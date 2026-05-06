from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.utils.config import load_yaml_file


def natural_key(text: str) -> list[object]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert saved inference frames into videos.")
    parser.add_argument("--config", required=True, type=str, help="Path to one video conversion config file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_yaml_file(Path(args.config))
    mode = str(config["mode"])
    root = Path(str(config["root"]))
    record = str(config["record"])
    target_mode = str(config["target_mode"])
    out_dir = Path(str(config["out_dir"]))
    fps = int(config.get("fps", 60))
    export_grid = bool(config.get("export_grid", False))
    layout = str(config.get("layout", "basic"))
    tile_scale = float(config.get("tile_scale", 0.5))
    pad = int(config.get("pad", 8))
    export_all = bool(config.get("export_all", False))
    export_vfi60 = bool(config.get("export_vfi60", True))
    single_files = list(config.get("single_files", []))

    summary = {
        "mode": mode,
        "root": str(root),
        "record": record,
        "target_mode": target_mode,
        "out_dir": str(out_dir),
        "fps": fps,
        "export_grid": export_grid,
        "layout": layout,
        "export_all": export_all,
        "export_vfi60": export_vfi60,
    }
    if mode == "dry-run":
        print(json.dumps(summary, indent=2))
        return

    import cv2
    import numpy as np

    seq_dir = root / record / target_mode
    if not seq_dir.exists():
        raise FileNotFoundError(f"Missing: {seq_dir}")

    frame_dirs = [path for path in seq_dir.iterdir() if path.is_dir()]
    frame_dirs.sort(key=lambda path: natural_key(path.name))
    if len(frame_dirs) == 0:
        raise RuntimeError(f"No frame dirs under: {seq_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    def read_bgr(path: Path) -> Any:
        if not path.exists():
            return None
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    def write_video(frames: list[Any], out_path: Path) -> None:
        if len(frames) == 0:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {out_path}")
        for frame in frames:
            writer.write(frame)
        writer.release()

    if export_grid:
        first_image = None
        for frame_dir in frame_dirs:
            for image_name in ("image_pred.png", "image_gt.png", "image_0.png"):
                first_image = read_bgr(frame_dir / image_name)
                if first_image is not None:
                    break
            if first_image is not None:
                break
        if first_image is None:
            raise RuntimeError("Failed to find a base image for grid export.")

        tile_width = max(16, int(first_image.shape[1] * tile_scale))
        tile_height = max(16, int(first_image.shape[0] * tile_scale))
        grid_frames = []
        for frame_dir in frame_dirs:
            if layout == "debug":
                entries = [
                    ("img0", "image_0.png"),
                    ("img1", "image_1.png"),
                    ("gt", "image_gt.png"),
                    ("pred", "image_pred.png"),
                    ("merge", "image_merge.png"),
                    ("mask", "flow_mask.png"),
                    ("flow 1->0", "flow_1_to_0.png"),
                    ("flow 1->2", "flow_1_to_2.png"),
                    ("diff", "diff_mag_overlay_1_to_0.png"),
                ]
                rows_count, cols_count = 3, 3
            else:
                entries = [
                    ("img0", "image_0.png"),
                    ("img1", "image_1.png"),
                    ("gt", "image_gt.png"),
                    ("pred", "image_pred.png"),
                    ("merge", "image_merge.png"),
                    ("diff", "diff_mag_overlay_1_to_0.png"),
                ]
                rows_count, cols_count = 2, 3

            canvas = np.zeros(
                (
                    rows_count * tile_height + (rows_count + 1) * pad,
                    cols_count * tile_width + (cols_count + 1) * pad,
                    3,
                ),
                dtype=np.uint8,
            )
            for entry_index, (label, image_name) in enumerate(entries):
                image = read_bgr(frame_dir / image_name)
                if image is None:
                    continue
                image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                row_index = entry_index // cols_count
                col_index = entry_index % cols_count
                x0 = pad + col_index * (tile_width + pad)
                y0 = pad + row_index * (tile_height + pad)
                canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = image
                cv2.rectangle(canvas, (x0, y0), (x0 + tile_width, y0 + 24), (0, 0, 0), -1)
                cv2.putText(canvas, label, (x0 + 8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            grid_frames.append(canvas)

        write_video(grid_frames, out_dir / f"{record}_{target_mode.replace('/', '__')}_{layout}_grid.mp4")

    if export_vfi60:
        pred_frames = []
        gt_frames = []
        for frame_index, frame_dir in enumerate(frame_dirs):
            image_0 = read_bgr(frame_dir / "image_0.png")
            image_pred = read_bgr(frame_dir / "image_pred.png")
            image_1 = read_bgr(frame_dir / "image_1.png")
            image_gt = read_bgr(frame_dir / "image_gt.png")
            if frame_index == 0 and image_0 is not None:
                pred_frames.append(image_0)
                gt_frames.append(image_0.copy())
            if image_pred is not None:
                pred_frames.append(image_pred)
            if image_gt is not None:
                gt_frames.append(image_gt)
            if image_1 is not None:
                pred_frames.append(image_1)
                gt_frames.append(image_1.copy())

        write_video(pred_frames, out_dir / f"{record}_{target_mode.replace('/', '__')}_vfi60_pred.mp4")
        write_video(gt_frames, out_dir / f"{record}_{target_mode.replace('/', '__')}_vfi60_gt.mp4")

    if export_all:
        files = single_files
        if len(files) == 0:
            files = [
                "image_0.png",
                "image_pred.png",
                "image_1.png",
                "image_gt.png",
                "image_merge.png",
                "image_0_warped.png",
                "image_1_warped.png",
                "flow_1_to_0.png",
                "flow_1_to_2.png",
                "flow_mask.png",
                "diff_mag_overlay_1_to_0.png",
                "diff_flow_1_to_0.png",
                "init_flow_1_to_0.png",
                "diff_changed_thr_1.00_1_to_0.png",
            ]

        single_out_dir = out_dir / "single"
        single_out_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            frames = []
            if filename == "image_0.png":
                image = read_bgr(frame_dirs[0] / filename)
                if image is not None:
                    frames.append(image)
            else:
                for frame_dir in frame_dirs:
                    image = read_bgr(frame_dir / filename)
                    if image is not None:
                        frames.append(image)
            write_video(frames, single_out_dir / f"{record}_{target_mode.replace('/', '__')}_{filename.replace('.png', '')}.mp4")


if __name__ == "__main__":
    main()
