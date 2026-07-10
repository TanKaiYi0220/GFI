from __future__ import annotations

from typing import Any

BASELINE_MODEL_NAME: str = "IFRNet"
RESIDUAL_MODEL_NAME: str = "IFRNet_Residual"
RESIDUAL_FLOW_APPROX_MODEL_NAME: str = "IFRNet_Residual_FlowApprox"
RIFE_MODEL_NAME: str = "RIFE"
UPRNET_MODEL_NAME: str = "UPRNet"
EMAVFI_MODEL_NAME: str = "EMAVFI"
SGMVFI_MODEL_NAME: str = "SGMVFI"

TRAIN_MODEL_NAMES: tuple[str, ...] = (
    BASELINE_MODEL_NAME,
    RESIDUAL_MODEL_NAME,
    RESIDUAL_FLOW_APPROX_MODEL_NAME,
    RIFE_MODEL_NAME,
    UPRNET_MODEL_NAME,
    EMAVFI_MODEL_NAME,
    SGMVFI_MODEL_NAME,
)
INFERENCE_MODEL_NAMES: tuple[str, ...] = TRAIN_MODEL_NAMES
MODEL_NAMES: tuple[str, ...] = INFERENCE_MODEL_NAMES
FLOW_APPROX_MODEL_NAMES: tuple[str, ...] = (RESIDUAL_FLOW_APPROX_MODEL_NAME,)
IMAGE_ONLY_VFI_MODEL_NAMES: tuple[str, ...] = (RIFE_MODEL_NAME, UPRNET_MODEL_NAME, EMAVFI_MODEL_NAME, SGMVFI_MODEL_NAME)


def uses_flow_approx_model(model_name: str) -> bool:
    return model_name in FLOW_APPROX_MODEL_NAMES


def uses_image_only_vfi_model(model_name: str) -> bool:
    return model_name in IMAGE_ONLY_VFI_MODEL_NAMES


def resolve_model_class(model_name: str) -> type[Any]:
    if model_name == BASELINE_MODEL_NAME:
        from src.models.IFRNet import Model as IFRNetModel

        return IFRNetModel
    if model_name == RESIDUAL_MODEL_NAME:
        from src.models.IFRNet_Residual import Model as IFRNetResidualModel

        return IFRNetResidualModel
    if model_name == RESIDUAL_FLOW_APPROX_MODEL_NAME:
        from src.models.IFRNet_Residual import Model as IFRNetResidualModel

        return IFRNetResidualModel
    if model_name == RIFE_MODEL_NAME:
        from src.models.RIFE import Model as RIFEModel

        return RIFEModel
    if model_name == UPRNET_MODEL_NAME:
        from src.models.UPRNet import Model as UPRNetModel

        return UPRNetModel
    if model_name == EMAVFI_MODEL_NAME:
        from src.models.EMAVFI import Model as EMAVFIModel

        return EMAVFIModel
    if model_name == SGMVFI_MODEL_NAME:
        from src.models.SGMVFI import Model as SGMVFIModel

        return SGMVFIModel

    available_models = ", ".join(MODEL_NAMES)
    raise KeyError(f"Unknown model '{model_name}'. Available models: {available_models}")


def set_model_convex_upsampling(model: Any, enabled: bool, context: str) -> bool:
    setter = getattr(model, "set_convex_upsampling", None)
    if not callable(setter):
        raise TypeError(
            f"{context} requested eval_convex_upsampling={enabled}, "
            f"but model type {type(model).__name__} does not support it."
        )
    previous_value = bool(getattr(model, "convex_upsampling"))
    setter(enabled)
    return previous_value
