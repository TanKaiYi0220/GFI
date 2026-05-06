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
    parser = argparse.ArgumentParser(description="Run inference from one config file.")
    parser.add_argument("--config", required=True, type=str, help="Path to one inference config file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config: dict[str, Any] = load_yaml_file(config_path)
    mode = str(config["mode"])
    model_name = str(config["model_name"])
    inference_preset = str(config["inference_preset"])
    flow_approx_method = str(config["flow_approx_method"])
    scale_factor = float(config["scale_factor"])
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

    if model_name == "IFRNet_Residual" and flow_approx_method not in FLOW_APPROX_METHODS:
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
    }
    if mode == "dry-run":
        print(json.dumps(summary, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset_loader import FlowEstimationTrainDataset
    from src.data.dataset_loader import VFITrainDataset
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

    if model_name == "IFRNet":
        dataset = VFITrainDataset(dataframe, str(dataset_root_dir), False, input_fps)
    else:
        dataset = FlowEstimationTrainDataset(dataframe, str(dataset_root_dir), input_fps, False)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = resolve_model_class(model_name)().to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    psnr_meter = AverageMeter()
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            if model_name == "IFRNet":
                img0, imgt, img1, _bmv, _fmv, embt, info = batch
                img0 = img0.to(device)
                imgt = imgt.to(device)
                img1 = img1.to(device)
                embt = embt.to(device)
                imgt_pred, _flow0, _flow1, _mask = model.inference(img0, img1, embt, scale_factor)
            else:
                img0, imgt, img1, _bmv_60, _fmv_60, bmv_30, fmv_30, embt, info = batch
                img0 = img0.to(device)
                imgt = imgt.to(device)
                img1 = img1.to(device)
                bmv_30 = bmv_30.to(device)
                fmv_30 = fmv_30.to(device)
                embt = embt.to(device)
                approx_bmv, approx_fmv = build_flow_init(fmv_30, bmv_30, embt, flow_approx_method)
                imgt_pred, _flow0, _flow1, _mask, _res, _merge = model.inference(
                    img0,
                    img1,
                    embt,
                    scale_factor,
                    init_flow0=approx_bmv,
                    init_flow1=approx_fmv,
                )

            for batch_index in range(int(imgt_pred.shape[0])):
                psnr_value = float(calculate_psnr(imgt[batch_index], imgt_pred[batch_index]).detach().cpu().item())
                psnr_meter.update(psnr_value, 1)

                record_name = str(info["record_name"][batch_index]).replace("/", "__")
                frame_range = str(info["frame_range"][batch_index])
                prediction_path = output_dir / record_name / f"{frame_range}.png"
                prediction_image = imgt_pred[batch_index].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                prediction_image = np.round(prediction_image * 255.0).astype(np.uint8)
                save_image(prediction_path, prediction_image)

                rows.append(
                    {
                        "record_name": str(info["record_name"][batch_index]),
                        "frame_range": frame_range,
                        "psnr": psnr_value,
                        "prediction_path": str(prediction_path),
                    }
                )

    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    logger.info("samples=%s mean_psnr=%.6f output_dir=%s", len(rows), psnr_meter.avg, output_dir)


if __name__ == "__main__":
    main()
