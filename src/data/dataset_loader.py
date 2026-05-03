from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.augment import random_crop
from src.data.augment import random_horizontal_flip
from src.data.augment import random_reverse_channel
from src.data.augment import random_reverse_time
from src.data.augment import random_resize
from src.data.augment import random_rotate
from src.data.augment import random_vertical_flip
from src.data.image_ops import load_backward_velocity
from src.data.image_ops import load_png

DEFAULT_MODALITY_CONFIG: dict[str, dict[str, str]] = {
    "colorNoScreenUI": {
        "prefix": "colorNoScreenUI_",
        "ext": ".png",
        "loader": "image",
        "subdir": "",
    },
    "colorScreenWithUI": {
        "prefix": "colorScreenWithUI_",
        "ext": ".png",
        "loader": "image",
        "subdir": "",
    },
    "backwardVel_Depth": {
        "prefix": "backwardVel_Depth_",
        "ext": ".exr",
        "loader": "game_motion",
        "subdir": "",
    },
    "forwardVel_Depth": {
        "prefix": "forwardVel_Depth_",
        "ext": ".exr",
        "loader": "game_motion",
        "subdir": "",
    },
}


def build_distance_indexing(row: pd.Series) -> list[float]:
    distance_mean = float(row["D_index Mean"]) if "D_index Mean" in row.index else -1.0
    distance_median = float(row["D_index Median"]) if "D_index Median" in row.index else -1.0
    return [distance_mean, distance_median]


def build_embedding_tensor() -> torch.Tensor:
    return torch.tensor([[[0.5]]], dtype=torch.float32)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    contiguous_image = np.ascontiguousarray(image.transpose(2, 0, 1).astype(np.float32))
    return torch.from_numpy(contiguous_image / 255.0)


def flow_to_tensor(flow: np.ndarray) -> torch.Tensor:
    contiguous_flow = np.ascontiguousarray(flow.transpose(2, 0, 1).astype(np.float32))
    return torch.from_numpy(contiguous_flow)


def image_tensor_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(image_tensor.detach().cpu().permute(1, 2, 0).float().numpy())


def flow_tensor_to_numpy(flow_tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(flow_tensor.detach().cpu().permute(1, 2, 0).float().numpy())


class BaseDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_root_dir: str,
        input_fps: int,
        modality_config: dict[str, dict[str, str]],
        transform: Any | None,
        record: str | None,
        mode: str | None,
    ) -> None:
        self.dataframe = dataframe
        self.dataset_root_dir = dataset_root_dir
        self.df_fps = int(dataframe.iloc[0]["fps"])
        self.input_fps = input_fps
        self.modality_config = modality_config
        self.transform = transform
        self.record = record
        self.mode = mode

    def __len__(self) -> int:
        return int(len(self.dataframe) * (self.input_fps / self.df_fps))

    def _build_base_dir(self, record: str, mode: str) -> str:
        return os.path.join(self.dataset_root_dir, record, mode)

    def _build_modality_path(self, record: str, mode: str, frame_idx: int, modality_name: str) -> str:
        modality_spec = self.modality_config[modality_name]
        base_dir = self._build_base_dir(record, mode)
        subdir = modality_spec.get("subdir", "")
        if subdir != "":
            base_dir = os.path.join(base_dir, subdir)

        filename = f"{modality_spec['prefix']}{frame_idx}{modality_spec['ext']}"
        return os.path.join(base_dir, filename)

    def _load_image(self, path: str) -> np.ndarray:
        image = load_png(Path(path))
        return image[:, :, :3]

    def _load_game_motion(self, path: str) -> np.ndarray:
        game_motion, _depth = load_backward_velocity(Path(path))
        return game_motion

    def __getitem__(self, index: int) -> object:
        raise NotImplementedError("BaseDataset is abstract. Use one concrete dataset class instead.")


class FlowEstimationDataset(BaseDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_root_dir: str,
        input_fps: int,
        record: str,
        mode: str,
        modality_config: dict[str, dict[str, str]] = DEFAULT_MODALITY_CONFIG,
        transform: Any | None = None,
    ) -> None:
        super().__init__(dataframe, dataset_root_dir, input_fps, modality_config, transform, record, mode)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.df_fps == self.input_fps:
            row = self.dataframe.iloc[index]
            frame_0_idx = int(row["img0"])
            frame_1_idx = int(row["img1"])
            mode = self.mode or ""
        else:
            row = self.dataframe.iloc[index * 2]
            frame_0_idx = int(row["img0"]) // 2
            frame_1_idx = int(row["img2"]) // 2
            mode = (self.mode or "").replace("fps_60", "fps_30")

        record = self.record or str(row["record"])
        img_0_path = self._build_modality_path(record, mode, frame_0_idx, "colorNoScreenUI")
        img_1_path = self._build_modality_path(record, mode, frame_1_idx, "colorNoScreenUI")
        backward_path = self._build_modality_path(record, mode, frame_1_idx, "backwardVel_Depth")

        return {
            "frame_range": f"frame_{frame_0_idx:04d}_{frame_1_idx:04d}",
            "input": {"colorNoScreenUI": (img_0_path, img_1_path)},
            "ground_truth": {"backwardVel_Depth": backward_path},
            "valid": bool(row["valid"]) if "valid" in row.index else True,
        }


class VFIDataset(BaseDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_root_dir: str,
        input_fps: int,
        modality_config: dict[str, dict[str, str]] = DEFAULT_MODALITY_CONFIG,
        transform: Any | None = None,
    ) -> None:
        super().__init__(dataframe, dataset_root_dir, input_fps, modality_config, transform, None, None)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        frame_0_idx = int(row["img0"])
        frame_1_idx = int(row["img1"])
        frame_2_idx = int(row["img2"])
        record = str(row["record"])
        mode = str(row["mode"])

        img_0_path = self._build_modality_path(record, mode, frame_0_idx, "colorNoScreenUI")
        img_1_path = self._build_modality_path(record, mode, frame_1_idx, "colorNoScreenUI")
        img_2_path = self._build_modality_path(record, mode, frame_2_idx, "colorNoScreenUI")
        backward_path = self._build_modality_path(record, mode, frame_1_idx, "backwardVel_Depth")
        forward_path = self._build_modality_path(record, mode, frame_1_idx, "forwardVel_Depth")

        return {
            "frame_range": f"frame_{frame_0_idx:04d}_{frame_2_idx:04d}",
            "input": {"colorNoScreenUI": (img_0_path, img_2_path)},
            "ground_truth": {
                "backwardVel_Depth": backward_path,
                "forwardVel_Depth": forward_path,
                "colorNoScreenUI": img_1_path,
            },
            "valid": bool(row["valid"]) if "valid" in row.index else True,
            "distance_indexing": build_distance_indexing(row),
        }


class VFITrainDataset(BaseDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_root_dir: str,
        augment: bool,
        input_fps: int,
        modality_config: dict[str, dict[str, str]] = DEFAULT_MODALITY_CONFIG,
        transform: Any | None = None,
    ) -> None:
        super().__init__(dataframe, dataset_root_dir, input_fps, modality_config, transform, None, None)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row = self.dataframe.iloc[index]
        frame_0_idx = int(row["img0"])
        frame_1_idx = int(row["img1"])
        frame_2_idx = int(row["img2"])
        record = str(row["record"])
        mode = str(row["mode"])

        info = {
            "frame_range": f"frame_{frame_0_idx:04d}_{frame_2_idx:04d}",
            "valid": bool(row["valid"]) if "valid" in row.index else True,
            "distance_indexing": build_distance_indexing(row),
        }

        img_0_path = self._build_modality_path(record, mode, frame_0_idx, "colorNoScreenUI")
        img_1_path = self._build_modality_path(record, mode, frame_1_idx, "colorNoScreenUI")
        img_2_path = self._build_modality_path(record, mode, frame_2_idx, "colorNoScreenUI")
        backward_path = self._build_modality_path(record, mode, frame_1_idx, "backwardVel_Depth")
        forward_path = self._build_modality_path(record, mode, frame_1_idx, "forwardVel_Depth")

        img0 = self._load_image(img_0_path)
        imgt = self._load_image(img_1_path)
        img1 = self._load_image(img_2_path)
        bmv = self._load_game_motion(backward_path)
        fmv = self._load_game_motion(forward_path)

        if self.augment:
            img0, imgt, img1, bmv, fmv = random_crop(img0, imgt, img1, bmv, fmv, (224, 224))
            img0, imgt, img1, bmv, fmv = random_reverse_channel(img0, imgt, img1, bmv, fmv, 0.5)
            img0, imgt, img1, bmv, fmv = random_vertical_flip(img0, imgt, img1, bmv, fmv, 0.3)
            img0, imgt, img1, bmv, fmv = random_horizontal_flip(img0, imgt, img1, bmv, fmv, 0.5)
            img0, imgt, img1, bmv, fmv = random_rotate(img0, imgt, img1, bmv, fmv, 0.05)

        img0_tensor = image_to_tensor(img0)
        imgt_tensor = image_to_tensor(imgt)
        img1_tensor = image_to_tensor(img1)
        bmv_tensor = flow_to_tensor(bmv)
        fmv_tensor = flow_to_tensor(fmv)
        embt_tensor = build_embedding_tensor()

        return img0_tensor, imgt_tensor, img1_tensor, bmv_tensor, fmv_tensor, embt_tensor, info


class CachedVFITrainDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        augment: bool,
    ) -> None:
        self.manifest = pd.read_csv(manifest_path)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row = self.manifest.iloc[index]
        cache_path = Path(str(row["cache_path"]))
        payload = torch.load(str(cache_path), map_location="cpu")
        tensors = payload["tensors"]
        info = dict(payload["info"])

        img0 = image_tensor_to_numpy(tensors["img0"])
        imgt = image_tensor_to_numpy(tensors["imgt"])
        img1 = image_tensor_to_numpy(tensors["img1"])
        bmv = flow_tensor_to_numpy(tensors["bmv"])
        fmv = flow_tensor_to_numpy(tensors["fmv"])

        if self.augment:
            img0, imgt, img1, bmv, fmv = random_crop(img0, imgt, img1, bmv, fmv, (224, 224))
            img0, imgt, img1, bmv, fmv = random_reverse_channel(img0, imgt, img1, bmv, fmv, 0.5)
            img0, imgt, img1, bmv, fmv = random_vertical_flip(img0, imgt, img1, bmv, fmv, 0.3)
            img0, imgt, img1, bmv, fmv = random_horizontal_flip(img0, imgt, img1, bmv, fmv, 0.5)
            img0, imgt, img1, bmv, fmv = random_rotate(img0, imgt, img1, bmv, fmv, 0.05)

        img0_tensor = image_to_tensor(img0)
        imgt_tensor = image_to_tensor(imgt)
        img1_tensor = image_to_tensor(img1)
        bmv_tensor = flow_to_tensor(bmv)
        fmv_tensor = flow_to_tensor(fmv)
        embt_tensor = build_embedding_tensor()

        return img0_tensor, imgt_tensor, img1_tensor, bmv_tensor, fmv_tensor, embt_tensor, info


class FlowEstimationTrainDataset(BaseDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_root_dir: str,
        input_fps: int,
        augment: bool,
        modality_config: dict[str, dict[str, str]] = DEFAULT_MODALITY_CONFIG,
        transform: Any | None = None,
    ) -> None:
        super().__init__(dataframe, dataset_root_dir, input_fps, modality_config, transform, None, None)
        self.augment = augment

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if self.df_fps == self.input_fps:
            raise NotImplementedError("FlowEstimationTrainDataset currently expects 30fps input against 60fps targets.")

        row = self.dataframe.iloc[index * 2]
        frame_30_0_idx = int(row["img0"]) // 2
        frame_30_1_idx = int(row["img2"]) // 2
        frame_60_0_idx = int(row["img0"])
        frame_60_1_idx = int(row["img1"])
        frame_60_2_idx = int(row["img2"])
        record = str(self.dataframe.iloc[index]["record"])
        mode = str(self.dataframe.iloc[index]["mode"])
        mode_30 = mode.replace("fps_60", "fps_30")

        info = {
            "frame_range": f"frame_{frame_60_0_idx:04d}_{frame_60_2_idx:04d}",
            "valid": bool(row["valid"]) if "valid" in row.index else True,
            "distance_indexing": build_distance_indexing(row),
        }

        img_60_0_path = self._build_modality_path(record, mode, frame_60_0_idx, "colorNoScreenUI")
        img_60_1_path = self._build_modality_path(record, mode, frame_60_1_idx, "colorNoScreenUI")
        img_60_2_path = self._build_modality_path(record, mode, frame_60_2_idx, "colorNoScreenUI")
        bmv_60_path = self._build_modality_path(record, mode, frame_60_1_idx, "backwardVel_Depth")
        fmv_60_path = self._build_modality_path(record, mode, frame_60_1_idx, "forwardVel_Depth")
        bmv_30_path = self._build_modality_path(record, mode_30, frame_30_1_idx, "backwardVel_Depth")
        fmv_30_path = self._build_modality_path(record, mode_30, frame_30_0_idx, "forwardVel_Depth")
        img_30_0_path = self._build_modality_path(record, mode_30, frame_30_0_idx, "colorNoScreenUI")
        img_30_1_path = self._build_modality_path(record, mode_30, frame_30_1_idx, "colorNoScreenUI")

        img0 = self._load_image(img_60_0_path)
        imgt = self._load_image(img_60_1_path)
        img1 = self._load_image(img_60_2_path)
        bmv_60 = self._load_game_motion(bmv_60_path)
        fmv_60 = self._load_game_motion(fmv_60_path)
        bmv_30 = self._load_game_motion(bmv_30_path)
        fmv_30 = self._load_game_motion(fmv_30_path)
        img0_30 = self._load_image(img_30_0_path)
        img1_30 = self._load_image(img_30_1_path)

        img0_tensor = image_to_tensor(img0)
        imgt_tensor = image_to_tensor(imgt)
        img1_tensor = image_to_tensor(img1)
        bmv_60_tensor = flow_to_tensor(bmv_60)
        fmv_60_tensor = flow_to_tensor(fmv_60)
        bmv_30_tensor = flow_to_tensor(bmv_30)
        fmv_30_tensor = flow_to_tensor(fmv_30)
        img0_30_tensor = image_to_tensor(img0_30)
        img1_30_tensor = image_to_tensor(img1_30)
        embt_tensor = build_embedding_tensor()

        # Add paths to info for debugging purposes
        # info["img_60_2_path"] = img_60_2_path
        # info["img_30_1_path"] = img_30_1_path

        return (
            img0_tensor,
            imgt_tensor,
            img1_tensor,
            bmv_60_tensor,
            fmv_60_tensor,
            bmv_30_tensor,
            fmv_30_tensor,
            img0_30_tensor,
            img1_30_tensor,
            embt_tensor,
            info,
        )
