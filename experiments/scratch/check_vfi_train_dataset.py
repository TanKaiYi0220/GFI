from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import VFITrainDataset, FlowEstimationTrainDataset

CSV_PATH: Path = PROJECT_ROOT / "outputs" / "functional_validation_20260502_212347" / "preprocess_data" / "ARPG_2_preprocessed" / "4_fps_60_raw_sequence_frame_index.csv"
DATASET_ROOT: Path = PROJECT_ROOT / "outputs" / "functional_validation_20260502_212347" / "dataset_subset"
RECORD_NAME: str = "ARPG_2"
MODE_PATH: str = "4_Difficult/4_Difficult_0/fps_60"
INPUT_FPS: int = 30
AUGMENT: bool = True
SAMPLE_INDEX: int = 0


def main() -> None:
    dataframe = pd.read_csv(CSV_PATH)
    dataframe["record"] = RECORD_NAME
    dataframe["mode"] = MODE_PATH

    dataset = VFITrainDataset(
        dataframe=dataframe,
        dataset_root_dir=str(DATASET_ROOT),
        augment=AUGMENT,
        input_fps=INPUT_FPS,
    )
    sample = dataset[SAMPLE_INDEX]

    img0, imgt, img1, bmv, fmv, embt, info = sample

    print("=== VFITrainDataset Sample ===")
    print(f"csv_path={CSV_PATH}")
    print(f"dataset_root={DATASET_ROOT}")
    print(f"dataset_len={len(dataset)}")
    print(f"sample_index={SAMPLE_INDEX}")
    print(f"augment={AUGMENT}")
    print(f"img0_shape={tuple(img0.shape)}")
    print(f"imgt_shape={tuple(imgt.shape)}")
    print(f"img1_shape={tuple(img1.shape)}")
    print(f"bmv_shape={tuple(bmv.shape)}")
    print(f"fmv_shape={tuple(fmv.shape)}")
    print(f"embt_shape={tuple(embt.shape)}")
    print(f"info={info}")

    dataset = FlowEstimationTrainDataset(
        dataframe=dataframe,
        dataset_root_dir=str(DATASET_ROOT),
        augment=AUGMENT,
        input_fps=INPUT_FPS,
    )
    sample = dataset[SAMPLE_INDEX]

    img0, imgt, img1, bmv_60, fmv_60, bmv_30, fmv_30, img_30_0, img_30_1, embt, info = sample

    print("\n=== FlowEstimationTrainDataset Sample ===")
    print(f"csv_path={CSV_PATH}")
    print(f"dataset_root={DATASET_ROOT}")
    print(f"dataset_len={len(dataset)}")
    print(f"sample_index={SAMPLE_INDEX}")
    print(f"augment={AUGMENT}")
    print(f"img0_shape={tuple(img0.shape)}")
    print(f"imgt_shape={tuple(imgt.shape)}")
    print(f"img1_shape={tuple(img1.shape)}")
    print(f"bmv_60_shape={tuple(bmv_60.shape)}")
    print(f"fmv_60_shape={tuple(fmv_60.shape)}")
    print(f"bmv_30_shape={tuple(bmv_30.shape)}")
    print(f"fmv_30_shape={tuple(fmv_30.shape)}")
    print(f"img_30_0_shape={tuple(img_30_0.shape)}")
    print(f"img_30_1_shape={tuple(img_30_1.shape)}")
    print(f"embt_shape={tuple(embt.shape)}")
    print(f"info={info}")


if __name__ == "__main__":
    main()
