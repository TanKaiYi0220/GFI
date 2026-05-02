from __future__ import annotations

from pathlib import Path
import logging
import sys

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_merged_dataframe
from src.data.dataset_config import DATASET_PRESETS
from src.data.dataset_config import build_dataset_preset
from src.data.dataset_config import make_record_config
from src.data.dataset_loader import VFITrainDataset

CSV_ROOT_DIR: Path = PROJECT_ROOT / "outputs" / "functional_validation_20260502_212347" / "preprocess_data"
DATASET_ROOT_DIR: Path = PROJECT_ROOT / "outputs" / "functional_validation_20260502_212347" / "dataset_subset"
PRESET_KEY: str = "tmp_dual_mode_arpg2"
ONLY_FPS: int = 60
SAMPLE_INDICES: tuple[int, int] = (0, 49)


def build_logger() -> logging.Logger:
    logger = logging.getLogger("CheckMergedDataframeDualModes")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    return logger


def register_temp_preset() -> None:
    DATASET_PRESETS[PRESET_KEY] = build_dataset_preset(
        name="ARPG_2 Dual Difficult Modes",
        records={
            "ARPG_2": make_record_config(
                main_indices=("4", "4"),
                difficulties=("Difficult",),
                sub_indices=("0", "1"),
                fps_values=(30, 60),
                max_indices=(400, 800),
            ),
        },
    )


def print_summary(dataframe: pd.DataFrame) -> None:
    print("=== merged dataframe summary ===")
    print(f"rows={len(dataframe)}")
    print(f"records={sorted(dataframe['record'].unique().tolist())}")
    print(f"modes={sorted(dataframe['mode'].unique().tolist())}")
    print("rows_per_mode=")
    print(dataframe.groupby("mode").size())
    print("valid_counts=")
    print(dataframe.groupby("mode")["valid"].sum())


def print_sample_paths(dataframe: pd.DataFrame, dataset_root_dir: Path) -> None:
    dataset = VFITrainDataset(
        dataframe=dataframe,
        dataset_root_dir=str(dataset_root_dir),
        augment=False,
        input_fps=30,
    )

    print("\n=== sample path checks ===")
    for sample_index in SAMPLE_INDICES:
        row = dataframe.iloc[sample_index]
        sample = dataset[sample_index]
        img0, imgt, img1, bmv, fmv, embt, info = sample
        image_path = dataset._build_modality_path(
            str(row["record"]),
            str(row["mode"]),
            int(row["img0"]),
            "colorNoScreenUI",
        )
        print(f"sample_index={sample_index}")
        print(f"mode={row['mode']}")
        print(f"image_path={image_path}")
        print(f"img0_shape={tuple(img0.shape)}")
        print(f"bmv_shape={tuple(bmv.shape)}")
        print(f"info={info}")


def main() -> None:
    register_temp_preset()
    logger = build_logger()
    dataframe = build_merged_dataframe(
        root_dir=CSV_ROOT_DIR,
        dataset_preset_name=PRESET_KEY,
        only_fps=ONLY_FPS,
        logger=logger,
    )
    print_summary(dataframe)
    print_sample_paths(dataframe, DATASET_ROOT_DIR)


if __name__ == "__main__":
    main()
