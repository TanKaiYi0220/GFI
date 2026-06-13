from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any
from typing import TypedDict

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ProfileRow(TypedDict):
    name: str
    y: int
    x_start: int
    x_end: int
    color_bgr: tuple[int, int, int]


class FrameSequence(TypedDict):
    reference_frames: dict[int, Any]
    baseline_frames: dict[int, Any]
    candidate_frames: dict[int, Any]
    interpolated_frames: list[int]


BACKGROUND_BGR: tuple[int, int, int] = (255, 255, 255)
TEXT_BGR: tuple[int, int, int] = (32, 32, 32)
BLUE_TEXT_BGR: tuple[int, int, int] = (190, 88, 20)
GRID_BGR: tuple[int, int, int] = (230, 230, 230)
DEFAULT_ROWS: tuple[tuple[str, int, tuple[int, int, int]], ...] = (
    ("upper_body", 320, (255, 80, 40)),
    ("hands_waist", 360, (40, 150, 230)),
    ("legs", 415, (90, 180, 90)),
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-row temporal profiles from compare output folders.")
    parser.add_argument("--compare-root", required=True, type=str, help="Path to one compare winner bucket, for example .../high_psnr/baseline_win.")
    parser.add_argument("--frame-start", required=True, type=int, help="First even frame index, for example 588.")
    parser.add_argument("--frame-end", required=True, type=int, help="Last even frame index, for example 596.")
    parser.add_argument("--x-start", required=False, default=430, type=int, help="Left crop bound for temporal profile rows.")
    parser.add_argument("--x-end", required=False, default=770, type=int, help="Right crop bound for temporal profile rows.")
    parser.add_argument("--line-thickness", required=False, default=5, type=int, help="Vertical averaging thickness in pixels.")
    parser.add_argument("--time-scale", required=False, default=16, type=int, help="Output height per frame row.")
    parser.add_argument("--profile-width", required=False, default=920, type=int, help="Rendered profile width.")
    parser.add_argument("--source-scale", required=False, default=0.28, type=float, help="Scale for the reference frame preview.")
    parser.add_argument("--output-dir", required=False, type=str, help="Optional output directory. Defaults to compare_root/temporal_profile_xxxx_xxxx.")
    return parser.parse_args(argv)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_positive_int(value: int, key: str) -> int:
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


def validate_frame_range(frame_start: int, frame_end: int) -> None:
    if frame_end <= frame_start:
        raise ValueError(f"frame_end must be larger than frame_start, got start={frame_start} end={frame_end}")
    if (frame_end - frame_start) % 2 != 0:
        raise ValueError(f"frame range must span complete pairs, got start={frame_start} end={frame_end}")


def build_profile_rows(x_start: int, x_end: int) -> list[ProfileRow]:
    if x_end <= x_start:
        raise ValueError(f"x_end must be larger than x_start, got x_start={x_start} x_end={x_end}")
    rows: list[ProfileRow] = []
    for row_name, y, color_rgb in DEFAULT_ROWS:
        rows.append(
            {
                "name": row_name,
                "y": y,
                "x_start": x_start,
                "x_end": x_end,
                "color_bgr": (color_rgb[2], color_rgb[1], color_rgb[0]),
            },
        )
    return rows


def build_case_dir(compare_root: Path, frame_0: int, frame_1: int) -> Path:
    return compare_root / f"frame_{frame_0:04d}_{frame_1:04d}"


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def read_bgr_image(path: Path, cv2: Any) -> Any:
    image = cv2.imread(str(require_file(path)), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def load_frame_sequence(compare_root: Path, frame_start: int, frame_end: int, cv2: Any) -> FrameSequence:
    reference_frames: dict[int, Any] = {}
    baseline_frames: dict[int, Any] = {}
    candidate_frames: dict[int, Any] = {}
    interpolated_frames: list[int] = []

    for frame_0 in range(frame_start, frame_end, 2):
        frame_1 = frame_0 + 2
        frame_t = frame_0 + 1
        case_dir = build_case_dir(compare_root, frame_0, frame_1)
        baseline_dir = case_dir / "baseline"
        candidate_dir = case_dir / "candidate"
        image_0 = read_bgr_image(baseline_dir / "image_0.png", cv2)
        image_gt = read_bgr_image(baseline_dir / "image_gt.png", cv2)
        image_1 = read_bgr_image(baseline_dir / "image_1.png", cv2)
        baseline_pred = read_bgr_image(baseline_dir / "image_pred.png", cv2)
        candidate_pred = read_bgr_image(candidate_dir / "image_pred.png", cv2)

        reference_frames[frame_0] = image_0
        reference_frames[frame_t] = image_gt
        reference_frames[frame_1] = image_1
        baseline_frames[frame_0] = image_0
        baseline_frames[frame_t] = baseline_pred
        baseline_frames[frame_1] = image_1
        candidate_frames[frame_0] = image_0
        candidate_frames[frame_t] = candidate_pred
        candidate_frames[frame_1] = image_1
        interpolated_frames.append(frame_t)

    return {
        "reference_frames": reference_frames,
        "baseline_frames": baseline_frames,
        "candidate_frames": candidate_frames,
        "interpolated_frames": interpolated_frames,
    }


def validate_profile_rows(rows: list[ProfileRow], image_shape: tuple[int, int, int]) -> None:
    height = int(image_shape[0])
    width = int(image_shape[1])
    for row in rows:
        if row["y"] < 0 or row["y"] >= height:
            raise ValueError(f"Row {row['name']} y={row['y']} outside height={height}")
        if row["x_start"] < 0 or row["x_end"] > width:
            raise ValueError(
                f"Row {row['name']} x_range=[{row['x_start']}, {row['x_end']}] outside width={width}",
            )


def calculate_psnr(gt_image: Any, pred_image: Any) -> float:
    import numpy as np

    diff = gt_image.astype(np.float32) - pred_image.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def extract_temporal_profile(frames: dict[int, Any], frame_indices: list[int], row: ProfileRow, line_thickness: int) -> Any:
    import numpy as np

    first_frame = frames[frame_indices[0]]
    height = int(first_frame.shape[0])
    half_thickness = max(0, line_thickness // 2)
    y_start = max(0, row["y"] - half_thickness)
    y_end = min(height, row["y"] + half_thickness + 1)
    profiles = []
    for frame_index in frame_indices:
        frame = frames[frame_index]
        band = frame[y_start:y_end, row["x_start"] : row["x_end"], :].astype(np.float32)
        profiles.append(np.round(band.mean(axis=0)).astype(np.uint8))
    return np.stack(profiles, axis=0)


def scale_profile(profile: Any, profile_width: int, time_scale: int, cv2: Any) -> Any:
    target_height = int(profile.shape[0]) * time_scale
    return cv2.resize(profile, (profile_width, target_height), interpolation=cv2.INTER_NEAREST)


def draw_text(canvas: Any, text: str, x: int, y: int, scale: float, color_bgr: tuple[int, int, int], thickness: int, cv2: Any) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, thickness, cv2.LINE_AA)


def paste_image(canvas: Any, image: Any, x: int, y: int) -> None:
    canvas[y : y + image.shape[0], x : x + image.shape[1]] = image


def fit_text(text: str, max_width: int, scale: float, thickness: int, cv2: Any) -> str:
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    candidate = text
    while len(candidate) > 1:
        candidate = candidate[:-1]
        trial = f"{candidate}{suffix}"
        if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= max_width:
            return trial
    return suffix


def build_source_preview(first_frame: Any, rows: list[ProfileRow], source_scale: float, line_thickness: int, cv2: Any) -> Any:
    source_width = max(1, int(round(first_frame.shape[1] * source_scale)))
    source_height = max(1, int(round(first_frame.shape[0] * source_scale)))
    preview = cv2.resize(first_frame, (source_width, source_height), interpolation=cv2.INTER_AREA)
    scaled_thickness = max(1, int(round(line_thickness * source_scale)))
    for row in rows:
        y = int(round(row["y"] * source_scale))
        x_start = int(round(row["x_start"] * source_scale))
        x_end = int(round(row["x_end"] * source_scale))
        color = row["color_bgr"]
        cv2.line(preview, (x_start, y), (x_end, y), color, scaled_thickness, cv2.LINE_AA)
        label = fit_text(row["name"], max(30, source_width - x_start - 6), 0.45, 1, cv2)
        draw_text(preview, label, min(x_start + 5, source_width - 36), max(14, y - 5), 0.45, color, 1, cv2)
    return preview


def draw_frame_labels(
    canvas: Any,
    frame_indices: list[int],
    x: int,
    y: int,
    time_scale: int,
    interpolated_frames: list[int],
    stream_label: str,
    cv2: Any,
) -> None:
    interpolated_set = set(interpolated_frames)
    for frame_offset, frame_index in enumerate(frame_indices):
        y_mid = y + frame_offset * time_scale + int(time_scale * 0.68)
        suffix = stream_label if frame_index in interpolated_set else "input"
        draw_text(canvas, f"{frame_index} {suffix}", x, y_mid, 0.43, BLUE_TEXT_BGR, 1, cv2)


def draw_profile_grid(canvas: Any, x: int, y: int, width: int, frame_count: int, time_scale: int, cv2: Any) -> None:
    height = frame_count * time_scale
    cv2.rectangle(canvas, (x, y), (x + width, y + height), GRID_BGR, 1)
    for index in range(1, frame_count):
        line_y = y + index * time_scale
        cv2.line(canvas, (x, line_y), (x + width, line_y), (255, 255, 255), 1, cv2.LINE_AA)


def render_temporal_profile(
    rows: list[ProfileRow],
    source_frame: Any,
    panel_specs: list[tuple[str, dict[int, Any], list[int], str]],
    interpolated_frames: list[int],
    line_thickness: int,
    time_scale: int,
    profile_width: int,
    source_scale: float,
    title: str,
    output_path: Path,
    cv2: Any,
    np: Any,
) -> None:
    source_preview = build_source_preview(source_frame, rows, source_scale, line_thickness, cv2)
    margin = 18
    title_height = 62
    source_title_height = 26
    source_width = int(source_preview.shape[1])
    source_height = int(source_preview.shape[0])
    label_width = 96
    panel_gap = 20
    row_header_height = 28
    panel_title_height = 22
    max_frame_count = max(len(frame_indices) for _panel_name, _frames, frame_indices, _label in panel_specs)
    row_block_height = row_header_height
    for _panel_name, _frames, frame_indices, _label in panel_specs:
        row_block_height += panel_title_height + len(frame_indices) * time_scale + panel_gap
    row_block_height += 10
    content_x = margin * 2 + source_width + 18
    profile_x = content_x + label_width
    canvas_width = profile_x + profile_width + margin
    canvas_height = max(
        title_height + source_title_height + source_height + margin,
        title_height + len(rows) * row_block_height + margin,
    )
    canvas = np.full((canvas_height, canvas_width, 3), BACKGROUND_BGR, dtype=np.uint8)
    draw_text(canvas, fit_text(title, canvas_width - 2 * margin, 0.72, 2, cv2), margin, 36, 0.72, TEXT_BGR, 2, cv2)
    draw_text(canvas, "Sample rows", margin, title_height + 4, 0.52, TEXT_BGR, 1, cv2)
    paste_image(canvas, source_preview, margin, title_height + source_title_height)

    current_y = title_height
    for row in rows:
        draw_text(canvas, row["name"], content_x, current_y + 18, 0.56, TEXT_BGR, 1, cv2)
        current_y += row_header_height
        for panel_name, frames, frame_indices, frame_label in panel_specs:
            panel_y = current_y
            draw_text(canvas, panel_name, profile_x, panel_y + 15, 0.5, TEXT_BGR, 1, cv2)
            profile = extract_temporal_profile(frames, frame_indices, row, line_thickness)
            scaled_profile = scale_profile(profile, profile_width, time_scale, cv2)
            image_y = panel_y + panel_title_height
            paste_image(canvas, scaled_profile, profile_x, image_y)
            draw_profile_grid(canvas, profile_x, image_y, profile_width, len(frame_indices), time_scale, cv2)
            draw_frame_labels(canvas, frame_indices, content_x, image_y, time_scale, interpolated_frames, frame_label, cv2)
            current_y += panel_title_height + len(frame_indices) * time_scale + panel_gap
        current_y += 10

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise ValueError(f"Failed to write image: {output_path}")


def build_metrics_records(sequence: FrameSequence, rows: list[ProfileRow], line_thickness: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for frame_index in sequence["interpolated_frames"]:
        gt_image = sequence["reference_frames"][frame_index]
        baseline_image = sequence["baseline_frames"][frame_index]
        candidate_image = sequence["candidate_frames"][frame_index]
        records.append(
            {
                "scope": "full_frame",
                "row_name": "",
                "frame": frame_index,
                "baseline_psnr": calculate_psnr(gt_image, baseline_image),
                "candidate_psnr": calculate_psnr(gt_image, candidate_image),
                "candidate_minus_baseline": calculate_psnr(gt_image, candidate_image) - calculate_psnr(gt_image, baseline_image),
            },
        )
        for row in rows:
            gt_profile = extract_temporal_profile(sequence["reference_frames"], [frame_index], row, line_thickness)[0]
            baseline_profile = extract_temporal_profile(sequence["baseline_frames"], [frame_index], row, line_thickness)[0]
            candidate_profile = extract_temporal_profile(sequence["candidate_frames"], [frame_index], row, line_thickness)[0]
            baseline_psnr = calculate_psnr(gt_profile, baseline_profile)
            candidate_psnr = calculate_psnr(gt_profile, candidate_profile)
            records.append(
                {
                    "scope": "profile_row",
                    "row_name": row["name"],
                    "frame": frame_index,
                    "baseline_psnr": baseline_psnr,
                    "candidate_psnr": candidate_psnr,
                    "candidate_minus_baseline": candidate_psnr - baseline_psnr,
                },
            )
    return records


def write_metrics_csv(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scope", "row_name", "frame", "baseline_psnr", "candidate_psnr", "candidate_minus_baseline"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def sanitize_path_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if sanitized == "":
        raise ValueError(f"Cannot sanitize path part from value={value!r}")
    return sanitized[:120]


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    validate_frame_range(int(args.frame_start), int(args.frame_end))
    require_positive_int(int(args.line_thickness), "line_thickness")
    require_positive_int(int(args.time_scale), "time_scale")
    require_positive_int(int(args.profile_width), "profile_width")
    if float(args.source_scale) <= 0.0:
        raise ValueError(f"source_scale must be positive, got {args.source_scale}")

    import cv2
    import numpy as np

    compare_root = resolve_path(str(args.compare_root))
    frame_start = int(args.frame_start)
    frame_end = int(args.frame_end)
    rows = build_profile_rows(int(args.x_start), int(args.x_end))
    sequence = load_frame_sequence(compare_root, frame_start, frame_end, cv2)

    all_frame_indices = list(range(frame_start, frame_end + 1))
    predicted_frame_indices = list(sequence["interpolated_frames"])
    source_frame = sequence["reference_frames"][frame_start + 1]
    validate_profile_rows(rows, tuple(source_frame.shape))

    if isinstance(args.output_dir, str):
        output_dir = resolve_path(str(args.output_dir))
    else:
        output_dir = compare_root / f"temporal_profile_{frame_start:04d}_{frame_end:04d}"

    title_base = f"{compare_root.name} | frames {frame_start:04d}-{frame_end:04d}"
    with_inputs_specs = [
        ("Reference", sequence["reference_frames"], all_frame_indices, "GT"),
        ("Baseline", sequence["baseline_frames"], all_frame_indices, "baseline"),
        ("Candidate", sequence["candidate_frames"], all_frame_indices, "candidate"),
    ]
    pred_only_specs = [
        ("Ground truth", sequence["reference_frames"], predicted_frame_indices, "GT"),
        ("Baseline", sequence["baseline_frames"], predicted_frame_indices, "baseline"),
        ("Candidate", sequence["candidate_frames"], predicted_frame_indices, "candidate"),
    ]
    render_temporal_profile(
        rows,
        source_frame,
        with_inputs_specs,
        sequence["interpolated_frames"],
        int(args.line_thickness),
        int(args.time_scale),
        int(args.profile_width),
        float(args.source_scale),
        f"{title_base} | reference + predictions",
        output_dir / "temporal_profile_with_inputs.png",
        cv2,
        np,
    )
    render_temporal_profile(
        rows,
        source_frame,
        pred_only_specs,
        sequence["interpolated_frames"],
        int(args.line_thickness),
        int(args.time_scale),
        int(args.profile_width),
        float(args.source_scale),
        f"{title_base} | predictions only",
        output_dir / "temporal_profile_predictions_only.png",
        cv2,
        np,
    )

    metrics_records = build_metrics_records(sequence, rows, int(args.line_thickness))
    write_metrics_csv(metrics_records, output_dir / "temporal_profile_metrics.csv")
    summary = {
        "compare_root": str(compare_root),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "output_dir": str(output_dir),
        "rows": [
            {
                "name": row["name"],
                "y": row["y"],
                "x_range": [row["x_start"], row["x_end"]],
            }
            for row in rows
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "temporal_profile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
