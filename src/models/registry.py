from __future__ import annotations

from typing import Any

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "IFRNet": {
        "default_resume_path": None,
        "default_output_dir": "outputs/train/IFRNet",
        "pretrained_checkpoint": None,
    },
    "IFRNet_Residual": {
        "default_resume_path": None,
        "default_output_dir": "outputs/train/IFRNet_Residual",
        "pretrained_checkpoint": None,
    },
}

MODEL_CLASSES: dict[str, type[Any]] = {}


def register_model_class(model_name: str, model_class: type[Any]) -> None:
    """Register one model class for the training and inference entrypoints."""
    MODEL_CLASSES[model_name] = model_class


def get_model_class(model_name: str) -> type[Any]:
    """Resolve one registered model class by name."""
    model_class: type[Any] | None = MODEL_CLASSES.get(model_name)
    if model_class is None:
        available_names: str = ", ".join(sorted(MODEL_CLASSES.keys()))
        raise KeyError(f"Model '{model_name}' is not registered. Registered models: {available_names}.")

    return model_class


def get_model_config(model_name: str) -> dict[str, Any]:
    """Resolve one model config entry by name."""
    model_config: dict[str, Any] | None = MODEL_CONFIGS.get(model_name)
    if model_config is None:
        available_names: str = ", ".join(sorted(MODEL_CONFIGS.keys()))
        raise KeyError(f"Model config '{model_name}' is not defined. Available configs: {available_names}.")

    return dict(model_config)

