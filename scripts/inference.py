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
from scripts.train import read_model_init_args
from scripts.train import resolve_model_class
from scripts.train import set_seed
from src.engine.flow_approx import build_flow_init
from src.engine.flow_approx import FLOW_APPROX_METHODS
from src.models.external.IFRNet.utils import warp
from src.utils.config import load_yaml_file
from src.utils.logger import build_logger

BASELINE_MODEL_NAME: str = "IFRNet"
RESIDUAL_MODEL_NAME: str = "IFRNet_Residual"
RESIDUAL_FLOW_APPROX_MODEL_NAME: str = "IFRNet_Residual_FlowApprox"
# Model variants:
# - IFRNet: baseline
# - IFRNet_Residual: residual model initialized by bmv/fmv from the 60fps motion labels
# - IFRNet_Residual_FlowApprox: residual model initialized by approximated 60fps motion from bmv_30/fmv_30

InferenceBatchResult = dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference from one config file.")
    parser.add_argument("--config", required=True, type=str, help="Path to one inference config file.")
    return parser.parse_args(argv)


def save_flow_diff_visuals(
    save_dir: Path,
    name: str,
    bg_img_np: Any,
    init_flow_np: Any,
    final_flow_np: Any,
    flow_to_image: Any,
    save_image: Any,
    cv2: Any,
    np: Any,
    threshold: float,
    percentile: float,
) -> dict[str, float]:
    diff_flow_np = final_flow_np - init_flow_np
    diff_mag_np = np.linalg.norm(diff_flow_np, axis=2)
    scale = max(float(np.percentile(diff_mag_np, percentile)), 1e-6)
    diff_mag_u8 = np.round(np.clip(diff_mag_np / scale, 0.0, 1.0) * 255.0).astype(np.uint8)
    diff_mag_color = cv2.applyColorMap(diff_mag_u8, cv2.COLORMAP_TURBO)

    height, width = diff_mag_color.shape[:2]
    colorbar_values = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    colorbar_u8 = np.round(colorbar_values * 255.0).astype(np.uint8)
    colorbar = cv2.applyColorMap(colorbar_u8, cv2.COLORMAP_TURBO)
    colorbar = cv2.resize(colorbar, (28, height), interpolation=cv2.INTER_NEAREST)

    diff_mag_with_colorbar = np.full((height, width + 126, 3), 255, dtype=np.uint8)
    diff_mag_with_colorbar[:, :width] = diff_mag_color
    diff_mag_with_colorbar[:, width + 8 : width + 36] = colorbar
    overlay = cv2.addWeighted(bg_img_np.astype(np.uint8), 0.2, diff_mag_color, 0.8, 0.0)
    overlay_with_colorbar = np.full((height, width + 126, 3), 255, dtype=np.uint8)
    overlay_with_colorbar[:, :width] = overlay
    overlay_with_colorbar[:, width + 8 : width + 36] = colorbar

    for canvas in (diff_mag_with_colorbar, overlay_with_colorbar):
        cv2.putText(canvas, "|Δflow|", (width + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        for tick_index in range(5):
            tick_value = tick_index / 4.0
            tick_y = int((1.0 - tick_value) * (height - 1))
            actual_value = tick_value * scale
            cv2.line(canvas, (width + 8, tick_y), (width + 35, tick_y), (0, 0, 0), 1)
            cv2.putText(
                canvas,
                f"{actual_value:.2f}",
                (width + 42, min(max(tick_y + 4, 12), height - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    changed_mask = np.where(diff_mag_np > threshold, 255, 0).astype(np.uint8)
    save_image(save_dir / f"init_flow_{name}.png", flow_to_image(init_flow_np))
    save_image(save_dir / f"diff_flow_{name}.png", flow_to_image(diff_flow_np))
    save_image(save_dir / f"diff_mag_{name}.png", diff_mag_color)
    save_image(save_dir / f"diff_mag_cb_{name}.png", diff_mag_with_colorbar)
    save_image(save_dir / f"diff_mag_overlay_{name}.png", overlay)
    save_image(save_dir / f"diff_mag_overlay_cb_{name}.png", overlay_with_colorbar)
    save_image(save_dir / f"diff_changed_thr_{threshold:.2f}_{name}.png", changed_mask)
    return {
        "diff_mag_mean": float(diff_mag_np.mean()),
        "diff_mag_max": float(diff_mag_np.max()),
        "diff_changed_ratio": float((changed_mask > 0).mean()),
        "diff_percentile_value": float(scale),
    }


def run_inference_batch(
    batch: Any,
    device: Any,
    flow_approx_method: str,
    model: Any,
    model_name: str,
    scale_factor: float,
) -> InferenceBatchResult:
    if model_name == BASELINE_MODEL_NAME:
        img0, imgt, img1, bmv, fmv, embt, _info = batch
        img0 = img0.to(device)
        imgt = imgt.to(device)
        img1 = img1.to(device)
        bmv = bmv.to(device)
        fmv = fmv.to(device)
        embt = embt.to(device)
        imgt_pred, up_flow0_1, up_flow1_1, up_mask_1 = model.inference(img0, img1, embt, scale_factor)
        return {
            "bmv": bmv,
            "embt": embt,
            "fmv": fmv,
            "img0": img0,
            "img1": img1,
            "imgt": imgt,
            "imgt_merge": None,
            "imgt_pred": imgt_pred,
            "init_bmv": None,
            "init_fmv": None,
            "up_flow0_1": up_flow0_1,
            "up_flow1_1": up_flow1_1,
            "up_mask_1": up_mask_1,
        }

    if model_name == RESIDUAL_FLOW_APPROX_MODEL_NAME:
        img0, imgt, img1, bmv, fmv, bmv_30, fmv_30, embt, _info = batch
        img0 = img0.to(device)
        imgt = imgt.to(device)
        img1 = img1.to(device)
        bmv = bmv.to(device)
        fmv = fmv.to(device)
        bmv_30 = bmv_30.to(device)
        fmv_30 = fmv_30.to(device)
        embt = embt.to(device)
        init_bmv, init_fmv = build_flow_init(fmv_30, bmv_30, embt, flow_approx_method)
        imgt_pred, up_flow0_1, up_flow1_1, up_mask_1, _up_res_1, imgt_merge = model.inference(
            img0,
            img1,
            embt,
            scale_factor,
            init_flow0=init_bmv,
            init_flow1=init_fmv,
        )
        return {
            "bmv": bmv,
            "embt": embt,
            "fmv": fmv,
            "img0": img0,
            "img1": img1,
            "imgt": imgt,
            "imgt_merge": imgt_merge,
            "imgt_pred": imgt_pred,
            "init_bmv": init_bmv,
            "init_fmv": init_fmv,
            "up_flow0_1": up_flow0_1,
            "up_flow1_1": up_flow1_1,
            "up_mask_1": up_mask_1,
        }

    if model_name == RESIDUAL_MODEL_NAME:
        img0, imgt, img1, bmv, fmv, _, _, embt, _info = batch
        img0 = img0.to(device)
        imgt = imgt.to(device)
        img1 = img1.to(device)
        bmv = bmv.to(device)
        fmv = fmv.to(device)
        embt = embt.to(device)
        imgt_pred, up_flow0_1, up_flow1_1, up_mask_1, _up_res_1, imgt_merge = model.inference(
            img0,
            img1,
            embt,
            scale_factor,
            init_flow0=bmv,
            init_flow1=fmv,
        )
        return {
            "bmv": bmv,
            "embt": embt,
            "fmv": fmv,
            "img0": img0,
            "img1": img1,
            "imgt": imgt,
            "imgt_merge": imgt_merge,
            "imgt_pred": imgt_pred,
            "init_bmv": bmv,
            "init_fmv": fmv,
            "up_flow0_1": up_flow0_1,
            "up_flow1_1": up_flow1_1,
            "up_mask_1": up_mask_1,
        }

    raise ValueError(f"Unsupported model_name: {model_name}")


def save_selected_sample_artifacts(
    cv2: Any,
    flow_diff_percentile: float,
    flow_diff_threshold: float,
    flow_to_image: Any,
    inference_result: InferenceBatchResult,
    np: Any,
    save_dir: Path,
    save_image: Any,
) -> dict[str, str]:
    img0 = inference_result["img0"]
    img1 = inference_result["img1"]
    imgt = inference_result["imgt"]
    bmv = inference_result["bmv"]
    fmv = inference_result["fmv"]
    imgt_pred = inference_result["imgt_pred"]
    imgt_merge = inference_result["imgt_merge"]
    init_bmv = inference_result["init_bmv"]
    init_fmv = inference_result["init_fmv"]
    up_flow0_1 = inference_result["up_flow0_1"]
    up_flow1_1 = inference_result["up_flow1_1"]
    up_mask_1 = inference_result["up_mask_1"]

    img0_warped = warp(img0, up_flow0_1)
    img1_warped = warp(img1, up_flow1_1)
    img0_bmv_warped = warp(img0, bmv)
    img1_fmv_warped = warp(img1, fmv)

    img0_np = np.round(img0[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img1_np = np.round(img1[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    imgt_np = np.round(imgt[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img_pred_np = np.round(imgt_pred[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    imgt_merge_np = None if imgt_merge is None else np.round(imgt_merge[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    bmv_np = flow_to_image(bmv[0].detach().cpu().permute(1, 2, 0).numpy())
    fmv_np = flow_to_image(fmv[0].detach().cpu().permute(1, 2, 0).numpy())
    flow_1_to_0_np = flow_to_image(up_flow0_1[0].detach().cpu().permute(1, 2, 0).numpy())
    flow_1_to_2_np = flow_to_image(up_flow1_1[0].detach().cpu().permute(1, 2, 0).numpy())
    flow_mask_np = np.round(up_mask_1[0, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
    img0_warped_np = np.round(img0_warped[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img1_warped_np = np.round(img1_warped[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img0_bmv_warped_np = np.round(img0_bmv_warped[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img1_fmv_warped_np = np.round(img1_fmv_warped[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    image_paths = {
        "image_0_path": str(save_dir / "image_0.png"),
        "image_1_path": str(save_dir / "image_1.png"),
        "image_gt_path": str(save_dir / "image_gt.png"),
        "image_pred_path": str(save_dir / "image_pred.png"),
        "image_merge_path": str(save_dir / "image_merge.png") if imgt_merge_np is not None else "",
        "bmv_path": str(save_dir / "bmv.png"),
        "fmv_path": str(save_dir / "fmv.png"),
        "flow_1_to_0_path": str(save_dir / "flow_1_to_0.png"),
        "flow_1_to_2_path": str(save_dir / "flow_1_to_2.png"),
        "flow_mask_path": str(save_dir / "flow_mask.png"),
        "image_0_warped_path": str(save_dir / "image_0_warped.png"),
        "image_1_warped_path": str(save_dir / "image_1_warped.png"),
        "image_0_bmv_warped_path": str(save_dir / "image_0_bmv_warped.png"),
        "image_1_fmv_warped_path": str(save_dir / "image_1_fmv_warped.png"),
    }

    save_image(Path(image_paths["image_0_path"]), img0_np)
    save_image(Path(image_paths["image_1_path"]), img1_np)
    save_image(Path(image_paths["image_gt_path"]), imgt_np)
    save_image(Path(image_paths["image_pred_path"]), img_pred_np)
    if imgt_merge_np is not None:
        save_image(Path(image_paths["image_merge_path"]), imgt_merge_np)
    save_image(Path(image_paths["bmv_path"]), bmv_np)
    save_image(Path(image_paths["fmv_path"]), fmv_np)
    save_image(Path(image_paths["flow_1_to_0_path"]), flow_1_to_0_np)
    save_image(Path(image_paths["flow_1_to_2_path"]), flow_1_to_2_np)
    save_image(Path(image_paths["flow_mask_path"]), flow_mask_np)
    save_image(Path(image_paths["image_0_warped_path"]), img0_warped_np)
    save_image(Path(image_paths["image_1_warped_path"]), img1_warped_np)
    save_image(Path(image_paths["image_0_bmv_warped_path"]), img0_bmv_warped_np)
    save_image(Path(image_paths["image_1_fmv_warped_path"]), img1_fmv_warped_np)

    if init_bmv is not None and init_fmv is not None:
        init_flow_1_to_0_np = init_bmv[0].detach().cpu().permute(1, 2, 0).numpy()
        init_flow_1_to_2_np = init_fmv[0].detach().cpu().permute(1, 2, 0).numpy()
        final_flow_1_to_0_np = up_flow0_1[0].detach().cpu().permute(1, 2, 0).numpy()
        final_flow_1_to_2_np = up_flow1_1[0].detach().cpu().permute(1, 2, 0).numpy()
        save_flow_diff_visuals(
            save_dir,
            "1_to_0",
            img0_np,
            init_flow_1_to_0_np,
            final_flow_1_to_0_np,
            flow_to_image,
            save_image,
            cv2,
            np,
            flow_diff_threshold,
            flow_diff_percentile,
        )
        save_flow_diff_visuals(
            save_dir,
            "1_to_2",
            img1_np,
            init_flow_1_to_2_np,
            final_flow_1_to_2_np,
            flow_to_image,
            save_image,
            cv2,
            np,
            flow_diff_threshold,
            flow_diff_percentile,
        )

    return image_paths


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config: dict[str, Any] = load_yaml_file(config_path)
    mode = str(config["mode"])
    model_name = str(config["model_name"])
    model_init_args = read_model_init_args(config)
    inference_preset = str(config["inference_preset"])
    flow_approx_method = str(config["flow_approx_method"])
    scale_factor = float(config["scale_factor"])
    flow_diff_threshold = float(config.get("flow_diff_threshold", 1.0))
    flow_diff_percentile = float(config.get("flow_diff_percentile", 99.0))
    save_topk_worst_psnr = int(config.get("save_topk_worst_psnr", 3))
    save_topk_best_psnr = int(config.get("save_topk_best_psnr", 0))
    save_topk_largest_flow_diff = int(config.get("save_topk_largest_flow_diff", 3))
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    only_fps = int(config["only_fps"])
    input_fps = int(config["input_fps"])

    root_dir = Path(str(config["root_dir"]))
    if not root_dir.is_absolute():
        root_dir = PROJECT_ROOT / root_dir

    dataset_root_dir = Path(str(config["dataset_root_dir"]))
    if not dataset_root_dir.is_absolute():
        dataset_root_dir = PROJECT_ROOT / dataset_root_dir

    checkpoint_path = Path(str(config["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    output_dir = Path(str(config["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if model_name == RESIDUAL_FLOW_APPROX_MODEL_NAME and flow_approx_method not in FLOW_APPROX_METHODS:
        raise ValueError(f"Unsupported flow_approx_method: {flow_approx_method}")

    summary = {
        "mode": mode,
        "model_name": model_name,
        "inference_preset": inference_preset,
        "root_dir": str(root_dir),
        "dataset_root_dir": str(dataset_root_dir),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "batch_size": batch_size,
        "only_fps": only_fps,
        "input_fps": input_fps,
        "scale_factor": scale_factor,
        "flow_approx_method": flow_approx_method,
        "flow_diff_threshold": flow_diff_threshold,
        "flow_diff_percentile": flow_diff_percentile,
        "save_topk_worst_psnr": save_topk_worst_psnr,
        "save_topk_best_psnr": save_topk_best_psnr,
        "save_topk_largest_flow_diff": save_topk_largest_flow_diff,
    }
    if len(model_init_args) > 0:
        summary["model_init_args"] = model_init_args
    if mode == "dry-run":
        print(json.dumps(summary, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import Subset
    from tqdm import tqdm

    from src.data.dataset_loader import FlowEstimationTrainDataset
    from src.data.dataset_loader import VFITrainDataset
    from src.data.image_ops import flow_to_image
    from src.data.image_ops import save_image
    from src.engine.evaluation import AverageMeter
    from src.engine.evaluation import calculate_psnr
    logger = build_logger("scripts.inference")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s model=%s", device, model_name)

    dataframe = build_merged_dataframe(root_dir, output_dir, inference_preset, only_fps, logger)
    if "valid" in dataframe.columns:
        dataframe = dataframe[dataframe["valid"] == True].reset_index(drop=True)
    model_class = resolve_model_class(model_name)
    model = model_class(**model_init_args).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    # print("Load Pretrained Weights from IFRNet_Vimeo90K.pth as Baseline")
    # state_dict = torch.load("src/models/external/IFRNet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    psnr_meter = AverageMeter()
    rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for (record, mode_name), group_dataframe in dataframe.groupby(["record", "mode"], sort=False):
            group_dataframe = group_dataframe.reset_index(drop=True)
            if model_name == BASELINE_MODEL_NAME:
                dataset = VFITrainDataset(group_dataframe, str(dataset_root_dir), False, input_fps)
            else:
                dataset = FlowEstimationTrainDataset(group_dataframe, str(dataset_root_dir), input_fps, False)

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            record_meter = AverageMeter()
            progress = tqdm(loader, desc=f"{record}_{mode_name}", leave=True)
            sample_offset = 0
            group_rows: list[dict[str, object]] = []

            for batch in progress:
                inference_result = run_inference_batch(batch, device, flow_approx_method, model, model_name, scale_factor)
                imgt = inference_result["imgt"]
                imgt_pred = inference_result["imgt_pred"]
                init_bmv = inference_result["init_bmv"]
                init_fmv = inference_result["init_fmv"]
                up_flow0_1 = inference_result["up_flow0_1"]
                up_flow1_1 = inference_result["up_flow1_1"]

                for batch_index in range(int(imgt_pred.shape[0])):
                    row = group_dataframe.iloc[sample_offset + batch_index]
                    frame_range = f"frame_{int(row['img0']):04d}_{int(row['img2']):04d}"
                    psnr_value = float(calculate_psnr(imgt[batch_index], imgt_pred[batch_index]).detach().cpu().item())
                    psnr_meter.update(psnr_value, 1)
                    record_meter.update(psnr_value, 1)

                    diff_1_to_0 = {"diff_mag_mean": -1.0, "diff_mag_max": -1.0, "diff_changed_ratio": -1.0, "diff_percentile_value": -1.0}
                    diff_1_to_2 = {"diff_mag_mean": -1.0, "diff_mag_max": -1.0, "diff_changed_ratio": -1.0, "diff_percentile_value": -1.0}
                    if init_bmv is not None and init_fmv is not None:
                        init_flow_1_to_0_np = init_bmv[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                        init_flow_1_to_2_np = init_fmv[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                        final_flow_1_to_0_np = up_flow0_1[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                        final_flow_1_to_2_np = up_flow1_1[batch_index].detach().cpu().permute(1, 2, 0).numpy()
                        diff_mag_1_to_0_np = np.linalg.norm(final_flow_1_to_0_np - init_flow_1_to_0_np, axis=2)
                        diff_mag_1_to_2_np = np.linalg.norm(final_flow_1_to_2_np - init_flow_1_to_2_np, axis=2)
                        diff_1_to_0 = {
                            "diff_mag_mean": float(diff_mag_1_to_0_np.mean()),
                            "diff_mag_max": float(diff_mag_1_to_0_np.max()),
                            "diff_changed_ratio": float((diff_mag_1_to_0_np > flow_diff_threshold).mean()),
                            "diff_percentile_value": float(max(np.percentile(diff_mag_1_to_0_np, flow_diff_percentile), 1e-6)),
                        }
                        diff_1_to_2 = {
                            "diff_mag_mean": float(diff_mag_1_to_2_np.mean()),
                            "diff_mag_max": float(diff_mag_1_to_2_np.max()),
                            "diff_changed_ratio": float((diff_mag_1_to_2_np > flow_diff_threshold).mean()),
                            "diff_percentile_value": float(max(np.percentile(diff_mag_1_to_2_np, flow_diff_percentile), 1e-6)),
                        }

                    group_rows.append(
                        {
                            "sample_index": int(sample_offset + batch_index),
                            "record": str(record),
                            "mode": str(mode_name),
                            "record_name": f"{record}_{mode_name}",
                            "frame_range": frame_range,
                            "valid": bool(row["valid"]) if "valid" in row.index else True,
                            "distance_index_mean": float(row["D_index Mean"]) if "D_index Mean" in row.index else -1.0,
                            "distance_index_median": float(row["D_index Median"]) if "D_index Median" in row.index else -1.0,
                            "psnr": psnr_value,
                            "flow_diff_1_to_0_mean": diff_1_to_0["diff_mag_mean"],
                            "flow_diff_1_to_0_max": diff_1_to_0["diff_mag_max"],
                            "flow_diff_1_to_0_changed_ratio": diff_1_to_0["diff_changed_ratio"],
                            "flow_diff_1_to_0_percentile_value": diff_1_to_0["diff_percentile_value"],
                            "flow_diff_1_to_2_mean": diff_1_to_2["diff_mag_mean"],
                            "flow_diff_1_to_2_max": diff_1_to_2["diff_mag_max"],
                            "flow_diff_1_to_2_changed_ratio": diff_1_to_2["diff_changed_ratio"],
                            "flow_diff_1_to_2_percentile_value": diff_1_to_2["diff_percentile_value"],
                        }
                    )

                sample_offset += int(imgt_pred.shape[0])
                progress.set_postfix({"mean_psnr": f"{record_meter.avg:.6f}"})

            record_rows.append(
                {
                    "record": str(record),
                    "mode": str(mode_name),
                    "record_name": f"{record}_{mode_name}",
                    "samples": int(len(group_dataframe)),
                    "mean_psnr": float(record_meter.avg),
                }
            )
            group_metrics_df = pd.DataFrame(group_rows)
            selected_sample_reasons: dict[int, list[str]] = {}

            if save_topk_worst_psnr > 0:
                for sample_index in group_metrics_df.nsmallest(save_topk_worst_psnr, "psnr")["sample_index"].tolist():
                    selected_sample_reasons.setdefault(int(sample_index), []).append("worst_psnr")
            if save_topk_best_psnr > 0:
                for sample_index in group_metrics_df.nlargest(save_topk_best_psnr, "psnr")["sample_index"].tolist():
                    selected_sample_reasons.setdefault(int(sample_index), []).append("best_psnr")
            if model_name in (RESIDUAL_MODEL_NAME, RESIDUAL_FLOW_APPROX_MODEL_NAME) and save_topk_largest_flow_diff > 0:
                for sample_index in group_metrics_df.nlargest(save_topk_largest_flow_diff, "flow_diff_1_to_0_changed_ratio")["sample_index"].tolist():
                    selected_sample_reasons.setdefault(int(sample_index), []).append("largest_flow_diff_1_to_0")
                for sample_index in group_metrics_df.nlargest(save_topk_largest_flow_diff, "flow_diff_1_to_2_changed_ratio")["sample_index"].tolist():
                    selected_sample_reasons.setdefault(int(sample_index), []).append("largest_flow_diff_1_to_2")

            group_metrics_df["selected_for_save"] = group_metrics_df["sample_index"].map(lambda sample_index: int(sample_index) in selected_sample_reasons)
            group_metrics_df["save_reason"] = group_metrics_df["sample_index"].map(
                lambda sample_index: ";".join(selected_sample_reasons.get(int(sample_index), [])),
            )
            for column_name in (
                "image_0_path",
                "image_1_path",
                "image_gt_path",
                "image_pred_path",
                "image_merge_path",
                "bmv_path",
                "fmv_path",
                "flow_1_to_0_path",
                "flow_1_to_2_path",
                "flow_mask_path",
                "image_0_warped_path",
                "image_1_warped_path",
                "image_0_bmv_warped_path",
                "image_1_fmv_warped_path",
            ):
                group_metrics_df[column_name] = ""

            selected_indices = sorted(selected_sample_reasons.keys())
            if len(selected_indices) > 0:
                selected_dataset = Subset(dataset, selected_indices)
                selected_loader = DataLoader(selected_dataset, batch_size=1, shuffle=False)
                selected_progress = tqdm(selected_loader, desc=f"save_{record}_{mode_name}", leave=True)

                for selected_batch_index, batch in enumerate(selected_progress):
                    selected_row = group_metrics_df[group_metrics_df["sample_index"] == selected_indices[selected_batch_index]].iloc[0]
                    frame_range = str(selected_row["frame_range"])
                    save_dir = output_dir / str(record) / str(mode_name) / frame_range

                    inference_result = run_inference_batch(batch, device, flow_approx_method, model, model_name, scale_factor)
                    image_paths = save_selected_sample_artifacts(
                        cv2,
                        flow_diff_percentile,
                        flow_diff_threshold,
                        flow_to_image,
                        inference_result,
                        np,
                        save_dir,
                        save_image,
                    )

                    for column_name, path_value in image_paths.items():
                        group_metrics_df.loc[group_metrics_df["sample_index"] == selected_indices[selected_batch_index], column_name] = path_value

            rows.extend(group_metrics_df.to_dict("records"))
            logger.info("record=%s mode=%s samples=%s mean_psnr=%.6f", record, mode_name, len(group_dataframe), record_meter.avg)

    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(record_rows).to_csv(output_dir / "record_metrics.csv", index=False)
    logger.info("samples=%s mean_psnr=%.6f output_dir=%s", len(rows), psnr_meter.avg, output_dir)


if __name__ == "__main__":
    main()
