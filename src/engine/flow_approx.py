from __future__ import annotations

from typing import Any

FLOW_APPROX_METHODS: tuple[str, ...] = ("single", "combination")


def flow_approx(flow: Any, time: Any, forward: bool) -> Any:
    return time * flow if forward else (1 - time) * flow


def flow_approx_combination(fmv: Any, bmv: Any, time: Any, forward: bool) -> Any:
    if forward:
        return (1 - time) * (1 - time) * fmv - time * (1 - time) * bmv

    return -(1 - time) * time * fmv + time * time * bmv


def build_flow_init(
    fmv_30: Any,
    bmv_30: Any,
    embt: Any,
    flow_approx_method: str,
) -> tuple[Any, Any]:
    time = embt.reshape(embt.shape[0], 1, 1, 1)

    if flow_approx_method == "single":
        approx_fmv = flow_approx(fmv_30, time, True)
        approx_bmv = flow_approx(bmv_30, time, False)
        return approx_bmv, approx_fmv

    if flow_approx_method == "combination":
        approx_fmv = flow_approx_combination(fmv_30, bmv_30, time, True)
        approx_bmv = flow_approx_combination(fmv_30, bmv_30, time, False)
        return approx_bmv, approx_fmv

    available_methods = ", ".join(FLOW_APPROX_METHODS)
    raise ValueError(f"Unsupported flow_approx_method '{flow_approx_method}'. Available methods: {available_methods}")
