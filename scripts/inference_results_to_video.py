from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inference import BASELINE_MODEL_NAME
from scripts.inference import run_inference_batch
from scripts.train import build_merged_dataframe
from scripts.train import resolve_model_class
from scripts.train import set_seed
from src.utils.config import load_yaml_file
from src.utils.logger import build_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and export videos directly without saving images.")
    parser.add_argument("--config", required=True, type=str, help="Path to one inference video config file.")
    return parser.parse_args(argv)


def make_even_size(height: int, width: int) -> tuple[int, int]:
    even_height = height if height % 2 == 0 else height + 1
    even_width = width if width % 2 == 0 else width + 1
    return even_height, even_width


def open_video_writer(
    base_path: Path,
    fps: int,
    frame_shape: tuple[int, int],
    cv2: Any,
    logger: Any,
) -> tuple[Any, Path, tuple[int, int]]:
    height, width = make_even_size(frame_shape[0], frame_shape[1])
    candidates = [
        (base_path, "mp4v"),
        (base_path, "avc1"),
        (base_path.with_suffix(".avi"), "XVID"),
        (base_path.with_suffix(".avi"), "MJPG"),
    ]

    for output_path, codec in candidates:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            logger.info("video_writer path=%s codec=%s size=%sx%s", output_path, codec, width, height)
            return writer, output_path, (height, width)

    raise RuntimeError(f"Failed to open VideoWriter for base path: {base_path}")


def prepare_video_frame(frame: Any, target_shape: tuple[int, int], is_rgb: bool, cv2: Any, np: Any) -> Any:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if is_rgb else frame
    target_height, target_width = target_shape
    if frame_bgr.shape[0] == target_height and frame_bgr.shape[1] == target_width:
        return frame_bgr

    if frame_bgr.shape[0] > target_height or frame_bgr.shape[1] > target_width:
        return cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas[: frame_bgr.shape[0], : frame_bgr.shape[1]] = frame_bgr
    return canvas


def build_flow_error_vis(bg_img_np: Any, init_flow_np: Any, final_flow_np: Any, cv2: Any, np: Any) -> Any:
    diff_mag_np = np.linalg.norm(final_flow_np - init_flow_np, axis=2)
    scale = max(float(np.percentile(diff_mag_np, 99.0)), 1e-6)
    diff_mag_u8 = np.round(np.clip(diff_mag_np / scale, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(diff_mag_u8, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(bg_img_np.astype(np.uint8), 0.2, heatmap_bgr, 0.8, 0.0)


def build_grid_frame(
    batch_index: int,
    flow_to_image: Any,
    inference_result: dict[str, Any],
    model_name: str,
    pad: int,
    tile_scale: float,
    warp: Any,
    cv2: Any,
    np: Any,
) -> Any:
    img0 = inference_result["img0"]
    img1 = inference_result["img1"]
    imgt = inference_result["imgt"]
    bmv = inference_result["bmv"]
    fmv = inference_result["fmv"]
    imgt_pred = inference_result["imgt_pred"]
    init_bmv = inference_result["init_bmv"]
    init_fmv = inference_result["init_fmv"]
    up_flow0_1 = inference_result["up_flow0_1"]
    up_flow1_1 = inference_result["up_flow1_1"]
    up_mask_1 = inference_result["up_mask_1"]

    img0_warped = warp(img0, up_flow0_1)
    img1_warped = warp(img1, up_flow1_1)

    img0_np = np.round(img0[batch_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img1_np = np.round(img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    imgt_np = np.round(imgt[batch_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img_pred_np = np.round(imgt_pred[batch_index].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    flow_mask_np = np.round(up_mask_1[batch_index, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
    img0_warped_np = np.round(img0_warped[batch_index].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img1_warped_np = np.round(img1_warped[batch_index].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    if model_name == BASELINE_MODEL_NAME:
        flow_t_to_0_np = flow_to_image(up_flow0_1[batch_index].detach().cpu().permute(1, 2, 0).numpy())
        flow_t_to_1_np = flow_to_image(up_flow1_1[batch_index].detach().cpu().permute(1, 2, 0).numpy())
        flow_t_to_0_label = "flow_pred(t->0)"
        flow_t_to_1_label = "flow_pred(t->1)"
        flow_t_to_0_is_rgb = True
        flow_t_to_1_is_rgb = True
    else:
        init_flow_1_to_0_np = init_bmv[batch_index].detach().cpu().permute(1, 2, 0).numpy()
        init_flow_1_to_2_np = init_fmv[batch_index].detach().cpu().permute(1, 2, 0).numpy()
        final_flow_1_to_0_np = up_flow0_1[batch_index].detach().cpu().permute(1, 2, 0).numpy()
        final_flow_1_to_2_np = up_flow1_1[batch_index].detach().cpu().permute(1, 2, 0).numpy()
        flow_t_to_0_np = build_flow_error_vis(img0_np, init_flow_1_to_0_np, final_flow_1_to_0_np, cv2, np)
        flow_t_to_1_np = build_flow_error_vis(img1_np, init_flow_1_to_2_np, final_flow_1_to_2_np, cv2, np)
        flow_t_to_0_label = "flow_error_vis(t->0)"
        flow_t_to_1_label = "flow_error_vis(t->1)"
        flow_t_to_0_is_rgb = False
        flow_t_to_1_is_rgb = False

    entries = [
        ("img0", img0_np, False),
        ("img1", img1_np, False),
        ("img_GT", imgt_np, False),
        (flow_t_to_0_label, flow_t_to_0_np, flow_t_to_0_is_rgb),
        (flow_t_to_1_label, flow_t_to_1_np, flow_t_to_1_is_rgb),
        ("blending_mask", cv2.cvtColor(flow_mask_np, cv2.COLOR_GRAY2BGR), False),
        ("img0_warped", img0_warped_np, False),
        ("img1_warped", img1_warped_np, False),
        ("img_pred", img_pred_np, False),
    ]

    tile_width = max(16, int(img0_np.shape[1] * tile_scale))
    tile_height = max(16, int(img0_np.shape[0] * tile_scale))
    canvas = np.zeros((3 * tile_height + 4 * pad, 3 * tile_width + 4 * pad, 3), dtype=np.uint8)

    for entry_index, (label, frame, is_rgb) in enumerate(entries):
        tile = prepare_video_frame(frame, (tile_height, tile_width), is_rgb, cv2, np)
        row_index = entry_index // 3
        col_index = entry_index % 3
        x0 = pad + col_index * (tile_width + pad)
        y0 = pad + row_index * (tile_height + pad)
        canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
        cv2.rectangle(canvas, (x0, y0), (x0 + tile_width, y0 + 24), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x0 + 8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_yaml_file(config_path)
    mode = str(config["mode"])
    model_name = str(config["model_name"])
    inference_preset = str(config["inference_preset"])
    flow_approx_method = str(config.get("flow_approx_method", "combination"))
    scale_factor = float(config.get("scale_factor", 1.0))
    seed = int(config.get("seed", 1234))
    batch_size = int(config.get("batch_size", 1))
    only_fps = int(config["only_fps"])
    input_fps = int(config["input_fps"])
    fps = int(config.get("fps", 60))
    export_grid = bool(config.get("export_grid", False))
    tile_scale = float(config.get("tile_scale", 0.5))
    pad = int(config.get("pad", 8))
    export_all = bool(config.get("export_all", False))
    export_vfi60 = bool(config.get("export_vfi60", True))
    ignore_valid = bool(config.get("ignore_valid", True))
    record_filter = config.get("record")
    mode_filter = config.get("target_mode")
    single_files = list(config.get("single_files", []))

    root_dir = Path(str(config["root_dir"]))
    if not root_dir.is_absolute():
        root_dir = PROJECT_ROOT / root_dir

    dataset_root_dir = Path(str(config["dataset_root_dir"]))
    if not dataset_root_dir.is_absolute():
        dataset_root_dir = PROJECT_ROOT / dataset_root_dir

    checkpoint_path = Path(str(config["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    out_dir = Path(str(config["out_dir"]))
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    summary = {
        "mode": mode,
        "model_name": model_name,
        "inference_preset": inference_preset,
        "root_dir": str(root_dir),
        "dataset_root_dir": str(dataset_root_dir),
        "checkpoint_path": str(checkpoint_path),
        "out_dir": str(out_dir),
        "fps": fps,
        "ignore_valid": ignore_valid,
        "record": record_filter,
        "target_mode": mode_filter,
        "export_grid": export_grid,
        "export_all": export_all,
        "export_vfi60": export_vfi60,
    }
    if mode == "dry-run":
        print(json.dumps(summary, indent=2))
        return

    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.data.dataset_loader import FlowEstimationTrainDataset
    from src.data.dataset_loader import VFITrainDataset
    from src.data.image_ops import flow_to_image
    from src.models.external.IFRNet.utils import warp

    logger = build_logger("scripts.inference_results_to_video")
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s model=%s", device, model_name)

    dataframe = build_merged_dataframe(root_dir, out_dir, inference_preset, only_fps, logger)
    if not ignore_valid and "valid" in dataframe.columns:
        dataframe = dataframe[dataframe["valid"] == True].reset_index(drop=True)
    if record_filter is not None:
        dataframe = dataframe[dataframe["record"] == str(record_filter)].reset_index(drop=True)
    if mode_filter is not None:
        dataframe = dataframe[dataframe["mode"] == str(mode_filter)].reset_index(drop=True)
    if len(dataframe) == 0:
        raise RuntimeError("No samples matched the current filters.")

    model = resolve_model_class(model_name)().to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    if len(single_files) == 0:
        single_files = [
            "image_0.png",
            "image_pred.png",
            "image_1.png",
            "image_gt.png",
            "flow_t_to_0.png",
            "flow_t_to_1.png",
            "flow_mask.png",
            "image_0_warped.png",
            "image_1_warped.png",
        ]

    video_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for (record, mode_name), group_dataframe in dataframe.groupby(["record", "mode"], sort=False):
            group_dataframe = group_dataframe.reset_index(drop=True)
            if model_name == BASELINE_MODEL_NAME:
                dataset = VFITrainDataset(group_dataframe, str(dataset_root_dir), False, input_fps)
            else:
                dataset = FlowEstimationTrainDataset(group_dataframe, str(dataset_root_dir), input_fps, False)

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            progress = tqdm(loader, desc=f"{record}_{mode_name}", leave=True)
            mode_tag = str(mode_name).replace("/", "_")
            mp4_dir = out_dir / str(record)
            mp4_dir.mkdir(parents=True, exist_ok=True)

            pred_writer = None
            gt_writer = None
            pred_video_path = mp4_dir / f"{mode_tag}_vfi60_pred.mp4"
            gt_video_path = mp4_dir / f"{mode_tag}_vfi60_gt.mp4"
            pred_video_actual_path = pred_video_path
            gt_video_actual_path = gt_video_path
            pred_video_shape = None
            gt_video_shape = None

            grid_writer = None
            grid_video_path = mp4_dir / f"{mode_tag}_grid.mp4"
            grid_video_actual_path = grid_video_path
            grid_video_shape = None

            single_writers: dict[str, Any] = {}
            single_writer_shapes: dict[str, tuple[int, int]] = {}
            sample_count = 0

            for batch in progress:
                inference_result = run_inference_batch(batch, device, flow_approx_method, model, model_name, scale_factor)
                img0 = inference_result["img0"]
                img1 = inference_result["img1"]
                imgt = inference_result["imgt"]
                imgt_pred = inference_result["imgt_pred"]
                init_bmv = inference_result["init_bmv"]
                init_fmv = inference_result["init_fmv"]
                up_flow0_1 = inference_result["up_flow0_1"]
                up_flow1_1 = inference_result["up_flow1_1"]
                up_mask_1 = inference_result["up_mask_1"]

                img0_warped = warp(img0, up_flow0_1)
                img1_warped = warp(img1, up_flow1_1)

                for batch_index in range(int(imgt_pred.shape[0])):
                    img0_np = np.round(img0[batch_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    img1_np = np.round(img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    imgt_np = np.round(imgt[batch_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    img_pred_np = np.round(imgt_pred[batch_index].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    flow_mask_np = np.round(up_mask_1[batch_index, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
                    img0_warped_np = np.round(img0_warped[batch_index].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    img1_warped_np = np.round(img1_warped[batch_index].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

                    if model_name == BASELINE_MODEL_NAME:
                        flow_t_to_0_np = flow_to_image(up_flow0_1[batch_index].detach().cpu().permute(1, 2, 0).numpy())
                        flow_t_to_1_np = flow_to_image(up_flow1_1[batch_index].detach().cpu().permute(1, 2, 0).numpy())
                        flow_t_to_0_is_rgb = True
                        flow_t_to_1_is_rgb = True
                    else:
                        init_flow_1_to_0_np = init_bmv[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                        init_flow_1_to_2_np = init_fmv[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                        final_flow_1_to_0_np = up_flow0_1[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                        final_flow_1_to_2_np = up_flow1_1[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                        flow_t_to_0_np = build_flow_error_vis(img0_np, init_flow_1_to_0_np, final_flow_1_to_0_np, cv2, np)
                        flow_t_to_1_np = build_flow_error_vis(img1_np, init_flow_1_to_2_np, final_flow_1_to_2_np, cv2, np)
                        flow_t_to_0_is_rgb = False
                        flow_t_to_1_is_rgb = False

                    if pred_writer is None and export_vfi60:
                        pred_writer, pred_video_actual_path, pred_video_shape = open_video_writer(pred_video_path, fps, img0_np.shape[:2], cv2, logger)
                        gt_writer, gt_video_actual_path, gt_video_shape = open_video_writer(gt_video_path, fps, img0_np.shape[:2], cv2, logger)

                    if sample_count == 0 and pred_writer is not None:
                        pred_writer.write(prepare_video_frame(img0_np, pred_video_shape, False, cv2, np))
                        gt_writer.write(prepare_video_frame(img0_np, gt_video_shape, False, cv2, np))
                    if pred_writer is not None:
                        pred_writer.write(prepare_video_frame(img_pred_np, pred_video_shape, False, cv2, np))
                        pred_writer.write(prepare_video_frame(img1_np, pred_video_shape, False, cv2, np))
                        gt_writer.write(prepare_video_frame(imgt_np, gt_video_shape, False, cv2, np))
                        gt_writer.write(prepare_video_frame(img1_np, gt_video_shape, False, cv2, np))

                    if export_all:
                        stream_frames = {
                            "image_0.png": (img0_np, False),
                            "image_pred.png": (img_pred_np, False),
                            "image_1.png": (img1_np, False),
                            "image_gt.png": (imgt_np, False),
                            "flow_t_to_0.png": (flow_t_to_0_np, flow_t_to_0_is_rgb),
                            "flow_t_to_1.png": (flow_t_to_1_np, flow_t_to_1_is_rgb),
                            "flow_mask.png": (cv2.cvtColor(flow_mask_np, cv2.COLOR_GRAY2BGR), False),
                            "image_0_warped.png": (img0_warped_np, False),
                            "image_1_warped.png": (img1_warped_np, False),
                        }
                        for filename in single_files:
                            stream_entry = stream_frames.get(filename)
                            if stream_entry is None:
                                continue
                            frame, is_rgb = stream_entry
                            if filename == "image_0.png" and sample_count > 0:
                                continue
                            if filename not in single_writers:
                                writer_path = out_dir / "single" / f"{record}_{str(mode_name).replace('/', '__')}_{filename.replace('.png', '')}.mp4"
                                single_writers[filename], _single_output_path, single_writer_shapes[filename] = open_video_writer(
                                    writer_path,
                                    fps,
                                    frame.shape[:2],
                                    cv2,
                                    logger,
                                )
                            single_writers[filename].write(
                                prepare_video_frame(frame, single_writer_shapes[filename], is_rgb, cv2, np),
                            )

                    if export_grid:
                        grid_frame = build_grid_frame(
                            batch_index,
                            flow_to_image,
                            inference_result,
                            model_name,
                            pad,
                            tile_scale,
                            warp,
                            cv2,
                            np,
                        )
                        if grid_writer is None:
                            grid_writer, grid_video_actual_path, grid_video_shape = open_video_writer(
                                grid_video_path,
                                fps,
                                grid_frame.shape[:2],
                                cv2,
                                logger,
                            )
                        grid_writer.write(prepare_video_frame(grid_frame, grid_video_shape, False, cv2, np))

                    sample_count += 1

            if pred_writer is not None:
                pred_writer.release()
                gt_writer.release()
            if grid_writer is not None:
                grid_writer.release()
            for writer in single_writers.values():
                writer.release()

            video_rows.append(
                {
                    "record": str(record),
                    "mode": str(mode_name),
                    "record_name": f"{record}_{mode_name}",
                    "samples": int(len(group_dataframe)),
                    "pred_video_path": str(pred_video_actual_path) if export_vfi60 else "",
                    "gt_video_path": str(gt_video_actual_path) if export_vfi60 else "",
                    "grid_video_path": str(grid_video_actual_path) if export_grid else "",
                }
            )
            logger.info("record=%s mode=%s samples=%s", record, mode_name, len(group_dataframe))

    pd.DataFrame(video_rows).to_csv(out_dir / "video_metrics.csv", index=False)


if __name__ == "__main__":
    main()
