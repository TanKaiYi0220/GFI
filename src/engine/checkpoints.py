from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingState:
    start_epoch: int
    global_step: int
    best_psnr: float
    mode: str


def save_checkpoint(
    checkpoint_path: Path,
    model: Any,
    optimizer: Any,
    epoch: int,
    best_psnr: float,
) -> None:
    import torch

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
        },
        str(checkpoint_path),
    )


def is_raw_model_state_dict(checkpoint: Any, torch_module: Any) -> bool:
    if not isinstance(checkpoint, dict) or len(checkpoint) == 0:
        return False

    return all(isinstance(key, str) and torch_module.is_tensor(value) for key, value in checkpoint.items())


def extract_pretrained_state_dict(checkpoint: Any, checkpoint_path: Path, torch_module: Any) -> Any:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]

    if is_raw_model_state_dict(checkpoint=checkpoint, torch_module=torch_module):
        return checkpoint

    available_keys = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    raise KeyError(
        "pretrained_checkpoint_path must point to either a raw model state_dict or a full training checkpoint "
        f"containing a 'model' key: path={checkpoint_path}, keys={available_keys}"
    )


def load_training_state(
    resume_path: str | None,
    pretrained_checkpoint_path: str | None,
    model: Any,
    optimizer: Any,
    device: Any,
    logger: logging.Logger,
    iters_per_epoch: int,
    model_name: str,
) -> TrainingState:
    import torch

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        if "model" not in checkpoint or "optimizer" not in checkpoint or "epoch" not in checkpoint:
            raise KeyError(f"resume_path must point to a full training checkpoint with model, optimizer, and epoch: {resume_path}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = int(checkpoint["epoch"]) + 1
        logger.info("Resumed from %s at epoch %s", resume_path, start_epoch)
        return TrainingState(
            start_epoch=start_epoch,
            global_step=start_epoch * iters_per_epoch,
            best_psnr=float(checkpoint.get("best_psnr", 0.0)),
            mode="resume",
        )

    if pretrained_checkpoint_path is not None:
        pretrained_path = Path(pretrained_checkpoint_path)
        logger.info("Loading pretrained checkpoint from %s", pretrained_path)
        checkpoint = torch.load(str(pretrained_path), map_location=device)
        model.load_state_dict(
            extract_pretrained_state_dict(
                checkpoint=checkpoint,
                checkpoint_path=pretrained_path,
                torch_module=torch,
            )
        )
        return TrainingState(
            start_epoch=0,
            global_step=0,
            best_psnr=0.0,
            mode="pretrained",
        )

    logger.info("Training %s from scratch", model_name)
    return TrainingState(
        start_epoch=0,
        global_step=0,
        best_psnr=0.0,
        mode="scratch",
    )


def load_inference_state_dict(checkpoint_path: Path, device: Any) -> Any:
    import torch

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if is_raw_model_state_dict(checkpoint=checkpoint, torch_module=torch):
        return checkpoint

    available_keys = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    raise KeyError(
        "checkpoint_path must point to either a raw model state_dict or a full training checkpoint "
        f"containing a 'model' key: path={checkpoint_path}, keys={available_keys}"
    )


def load_inference_checkpoint(model: Any, checkpoint_path: Path, device: Any) -> None:
    external_loader = getattr(model, "load_external_checkpoint", None)
    if callable(external_loader):
        external_loader(checkpoint_path=checkpoint_path, device=device)
        return

    state_dict = load_inference_state_dict(checkpoint_path=checkpoint_path, device=device)
    model.load_state_dict(state_dict)
