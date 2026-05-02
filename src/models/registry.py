from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.IFRNet import Model as IFRNetModel
from src.models.IFRNet_Residual import Model as IFRNetResidualModel


MODEL_REGISTRY: dict[str, type[Any]] = {
    "IFRNet": IFRNetModel,
    "IFRNet_Residual": IFRNetResidualModel,
}


MODEL_CONFIGS: dict[str, dict[str, str | None]] = {
    "IFRNet": {
        "default_output_dir": "outputs/ifrnet_train",
        "default_resume_path": None,
        "pretrained_checkpoint": None,
    },
    "IFRNet_Residual": {
        "default_output_dir": "outputs/ifrnet_residual_train",
        "default_resume_path": None,
        "pretrained_checkpoint": None,
    },
}


def get_model_class(model_name: str) -> type[Any]:
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        available_models = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{model_name}'. Available models: {available_models}")

    return model_class


def get_model_config(model_name: str) -> dict[str, str | None]:
    model_config = MODEL_CONFIGS.get(model_name)
    if model_config is None:
        available_models = ", ".join(sorted(MODEL_CONFIGS.keys()))
        raise KeyError(f"Unknown model '{model_name}'. Available models: {available_models}")

    return model_config
