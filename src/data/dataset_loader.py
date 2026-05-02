from __future__ import annotations

from typing import Any


class VFITrainDataset:
    """Minimal dataframe-backed dataset shell for project-specific VFI loading."""

    def __init__(
        self,
        dataframe: Any,
        dataset_root_dir: str,
        augment: bool,
        input_fps: int,
    ) -> None:
        self.dataframe = dataframe
        self.dataset_root_dir = dataset_root_dir
        self.augment = augment
        self.input_fps = input_fps

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> object:
        raise NotImplementedError(
            "Implement row-to-tensor loading in src/data/dataset_loader.py for your dataframe schema.",
        )

