from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import VFITrainDataset
from src.models.registry import get_model_class

CSV_PATH: Path = PROJECT_ROOT / "outputs" / "functional_validation_20260502_212347" / "preprocess_data" / "ARPG_2_preprocessed" / "4_fps_60_raw_sequence_frame_index.csv"
DATASET_ROOT: Path = PROJECT_ROOT / "outputs" / "functional_validation_20260502_212347" / "dataset_subset"
RECORD_NAME: str = "ARPG_2"
MODE_PATH: str = "4_Difficult/4_Difficult_0/fps_60"
MODEL_NAMES: tuple[str, ...] = ("IFRNet", "IFRNet_Residual")
INPUT_FPS: int = 30
AUGMENT: bool = True
BATCH_SIZE: int = 1
NUM_SAMPLES: int = 2
LEARNING_RATE: float = 1e-4
SEED: int = 1234


def build_dataframe(csv_path: Path, record_name: str, mode_path: str, num_samples: int) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path).copy()
    dataframe["record"] = record_name
    dataframe["mode"] = mode_path
    return dataframe.iloc[:num_samples].reset_index(drop=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(dataframe: pd.DataFrame, dataset_root: Path, augment: bool, input_fps: int, batch_size: int) -> DataLoader:
    dataset = VFITrainDataset(
        dataframe=dataframe,
        dataset_root_dir=str(dataset_root),
        augment=augment,
        input_fps=input_fps,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def run_one_step(model_name: str, loader: DataLoader, learning_rate: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = get_model_class(model_name)
    model = model_class().to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    batch = next(iter(loader))
    img0, imgt, img1, bmv, fmv, embt, info = batch

    img0 = img0.to(device)
    imgt = imgt.to(device)
    img1 = img1.to(device)
    bmv = bmv.to(device)
    fmv = fmv.to(device)
    embt = embt.to(device)

    optimizer.zero_grad()
    if model_name == "IFRNet":
        flow = torch.cat([bmv, fmv], dim=1).float()
        imgt_pred, loss_rec, loss_geo, loss_dis, flow0, flow1, mask = model(img0, img1, embt, imgt, flow)
    else:
        imgt_pred, loss_rec, loss_geo, loss_dis, flow0, flow1, mask = model(
            img0,
            img1,
            embt,
            imgt,
            init_flow0=bmv,
            init_flow1=fmv,
        )

    total_loss = loss_rec + loss_geo + loss_dis
    total_loss.backward()
    optimizer.step()

    print(f"=== {model_name} ===")
    print(f"device={device}")
    print(f"dataset_len={len(loader.dataset)}")
    print(f"img0_shape={tuple(img0.shape)}")
    print(f"imgt_pred_shape={tuple(imgt_pred.shape)}")
    print(f"flow0_shape={tuple(flow0.shape)}")
    print(f"flow1_shape={tuple(flow1.shape)}")
    print(f"mask_shape={tuple(mask.shape)}")
    print(f"loss_rec={float(loss_rec.detach().cpu()):.6f}")
    print(f"loss_geo={float(loss_geo.detach().cpu()):.6f}")
    print(f"loss_dis={float(loss_dis.detach().cpu()):.6f}")
    print(f"loss_total={float(total_loss.detach().cpu()):.6f}")
    print(f"info={info}")


def main() -> None:
    set_seed(SEED)
    dataframe = build_dataframe(CSV_PATH, RECORD_NAME, MODE_PATH, NUM_SAMPLES)
    loader = build_loader(dataframe, DATASET_ROOT, AUGMENT, INPUT_FPS, BATCH_SIZE)

    print(f"csv_path={CSV_PATH}")
    print(f"dataset_root={DATASET_ROOT}")
    print(f"records={len(dataframe)}")
    for model_name in MODEL_NAMES:
        run_one_step(model_name, loader, LEARNING_RATE)


if __name__ == "__main__":
    main()
