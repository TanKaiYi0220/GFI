from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_merged_dataframe
from scripts.train import resolve_model_class
from scripts.train import set_seed
from scripts.train_flow_approx import build_flow_init
from scripts.train_flow_approx import FLOW_APPROX_METHODS
from src.utils.config import load_yaml_file
from src.utils.logger import build_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and export videos directly without saving images.")
    parser.add_argument("--config", required=True, type=str, help="Path to one inference video config file.")
    return parser.parse_args(argv)


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
    layout = str(config.get("layout", "basic"))
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

    if model_name == "IFRNet_Residual" and flow_approx_method not in FLOW_APPROX_METHODS:
        raise ValueError(f"Unsupported flow_approx_method: {flow_approx_method}")
    if layout not in {"basic", "debug"}:
        raise ValueError(f"Unsupported layout: {layout}")

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
        "layout": layout,
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
            "image_merge.png",
            "flow_1_to_0.png",
            "flow_1_to_2.png",
            "flow_mask.png",
        ]

    video_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for (record, mode_name), group_dataframe in dataframe.groupby(["record", "mode"], sort=False):
            group_dataframe = group_dataframe.reset_index(drop=True)
            if model_name == "IFRNet":
                dataset = VFITrainDataset(group_dataframe, str(dataset_root_dir), False, input_fps)
            else:
                dataset = FlowEstimationTrainDataset(group_dataframe, str(dataset_root_dir), input_fps, False)

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            progress = tqdm(loader, desc=f"{record}_{mode_name}", leave=True)
            pred_writer = None
            gt_writer = None
            grid_writer = None
            single_writers: dict[str, Any] = {}
            sample_count = 0

            for batch in progress:
                if model_name == "IFRNet":
                    img0, imgt, img1, bmv, fmv, embt, _info = batch
                    img0 = img0.to(device)
                    imgt = imgt.to(device)
                    img1 = img1.to(device)
                    embt = embt.to(device)
                    imgt_pred, up_flow0_1, up_flow1_1, up_mask_1 = model.inference(img0, img1, embt, scale_factor)
                    imgt_merge = None
                else:
                    img0, imgt, img1, _bmv, _fmv, bmv_30, fmv_30, embt, _info = batch
                    img0 = img0.to(device)
                    imgt = imgt.to(device)
                    img1 = img1.to(device)
                    bmv_30 = bmv_30.to(device)
                    fmv_30 = fmv_30.to(device)
                    embt = embt.to(device)
                    approx_bmv, approx_fmv = build_flow_init(fmv_30, bmv_30, embt, flow_approx_method)
                    imgt_pred, up_flow0_1, up_flow1_1, up_mask_1, _up_res_1, imgt_merge = model.inference(
                        img0,
                        img1,
                        embt,
                        scale_factor,
                        init_flow0=approx_bmv,
                        init_flow1=approx_fmv,
                    )

                for batch_index in range(int(imgt_pred.shape[0])):
                    img0_np = np.round(img0[batch_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    img1_np = np.round(img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    imgt_np = np.round(imgt[batch_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    img_pred_np = np.round(imgt_pred[batch_index].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    image_merge_np = None if imgt_merge is None else np.round(imgt_merge[batch_index].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    flow_1_to_0_np = flow_to_image(up_flow0_1[batch_index].detach().cpu().permute(1, 2, 0).numpy())
                    flow_1_to_2_np = flow_to_image(up_flow1_1[batch_index].detach().cpu().permute(1, 2, 0).numpy())
                    flow_mask_np = np.round(up_mask_1[batch_index, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)

                    if pred_writer is None and export_vfi60:
                        height, width = img0_np.shape[:2]
                        pred_path = out_dir / f"{record}_{str(mode_name).replace('/', '__')}_vfi60_pred.mp4"
                        gt_path = out_dir / f"{record}_{str(mode_name).replace('/', '__')}_vfi60_gt.mp4"
                        pred_writer = cv2.VideoWriter(str(pred_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        gt_writer = cv2.VideoWriter(str(gt_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not pred_writer.isOpened() or not gt_writer.isOpened():
                            raise RuntimeError(f"Failed to open main video writers for {record} {mode_name}")

                    if sample_count == 0 and pred_writer is not None:
                        pred_writer.write(cv2.cvtColor(img0_np, cv2.COLOR_RGB2BGR))
                        gt_writer.write(cv2.cvtColor(img0_np, cv2.COLOR_RGB2BGR))
                    if pred_writer is not None:
                        pred_writer.write(cv2.cvtColor(img_pred_np, cv2.COLOR_RGB2BGR))
                        pred_writer.write(cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR))
                        gt_writer.write(cv2.cvtColor(imgt_np, cv2.COLOR_RGB2BGR))
                        gt_writer.write(cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR))

                    if export_all:
                        stream_frames = {
                            "image_0.png": img0_np,
                            "image_pred.png": img_pred_np,
                            "image_1.png": img1_np,
                            "image_gt.png": imgt_np,
                            "image_merge.png": image_merge_np,
                            "flow_1_to_0.png": flow_1_to_0_np,
                            "flow_1_to_2.png": flow_1_to_2_np,
                            "flow_mask.png": cv2.cvtColor(flow_mask_np, cv2.COLOR_GRAY2BGR),
                        }
                        for filename in single_files:
                            frame = stream_frames.get(filename)
                            if frame is None:
                                continue
                            if filename == "image_0.png" and sample_count > 0:
                                continue
                            if filename not in single_writers:
                                stream_dir = out_dir / "single"
                                stream_dir.mkdir(parents=True, exist_ok=True)
                                writer_path = stream_dir / f"{record}_{str(mode_name).replace('/', '__')}_{filename.replace('.png', '')}.mp4"
                                single_writers[filename] = cv2.VideoWriter(
                                    str(writer_path),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    fps,
                                    (frame.shape[1], frame.shape[0]),
                                )
                                if not single_writers[filename].isOpened():
                                    raise RuntimeError(f"Failed to open single stream writer: {writer_path}")
                            if frame.ndim == 2:
                                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            else:
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            single_writers[filename].write(frame)

                    if export_grid:
                        if layout == "debug":
                            entries = [
                                ("img0", img0_np),
                                ("img1", img1_np),
                                ("gt", imgt_np),
                                ("pred", img_pred_np),
                                ("merge", image_merge_np if image_merge_np is not None else np.zeros_like(img0_np)),
                                ("mask", cv2.cvtColor(flow_mask_np, cv2.COLOR_GRAY2BGR)),
                                ("flow 1->0", flow_1_to_0_np),
                                ("flow 1->2", flow_1_to_2_np),
                                ("blank", np.zeros_like(img0_np)),
                            ]
                            rows_count, cols_count = 3, 3
                        else:
                            entries = [
                                ("img0", img0_np),
                                ("img1", img1_np),
                                ("gt", imgt_np),
                                ("pred", img_pred_np),
                                ("merge", image_merge_np if image_merge_np is not None else np.zeros_like(img0_np)),
                                ("mask", cv2.cvtColor(flow_mask_np, cv2.COLOR_GRAY2BGR)),
                            ]
                            rows_count, cols_count = 2, 3

                        tile_width = max(16, int(img0_np.shape[1] * tile_scale))
                        tile_height = max(16, int(img0_np.shape[0] * tile_scale))
                        canvas = np.zeros(
                            (
                                rows_count * tile_height + (rows_count + 1) * pad,
                                cols_count * tile_width + (cols_count + 1) * pad,
                                3,
                            ),
                            dtype=np.uint8,
                        )
                        for entry_index, (label, frame) in enumerate(entries):
                            tile = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            tile = cv2.resize(tile, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                            row_index = entry_index // cols_count
                            col_index = entry_index % cols_count
                            x0 = pad + col_index * (tile_width + pad)
                            y0 = pad + row_index * (tile_height + pad)
                            canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
                            cv2.rectangle(canvas, (x0, y0), (x0 + tile_width, y0 + 24), (0, 0, 0), -1)
                            cv2.putText(canvas, label, (x0 + 8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                        if grid_writer is None:
                            grid_path = out_dir / f"{record}_{str(mode_name).replace('/', '__')}_{layout}_grid.mp4"
                            grid_writer = cv2.VideoWriter(str(grid_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], canvas.shape[0]))
                            if not grid_writer.isOpened():
                                raise RuntimeError(f"Failed to open grid writer: {grid_path}")
                        grid_writer.write(canvas)

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
                    "pred_video_path": str(out_dir / f"{record}_{str(mode_name).replace('/', '__')}_vfi60_pred.mp4") if export_vfi60 else "",
                    "gt_video_path": str(out_dir / f"{record}_{str(mode_name).replace('/', '__')}_vfi60_gt.mp4") if export_vfi60 else "",
                }
            )
            logger.info("record=%s mode=%s samples=%s", record, mode_name, len(group_dataframe))

    pd.DataFrame(video_rows).to_csv(out_dir / "video_metrics.csv", index=False)


if __name__ == "__main__":
    main()
