from __future__ import annotations

from typing import Iterable, Protocol


class ParameterLike(Protocol):
    def numel(self) -> int:
        raise NotImplementedError


class ModelWithParameters(Protocol):
    def parameters(self) -> Iterable[ParameterLike]:
        raise NotImplementedError


def count_model_parameters(model: ModelWithParameters) -> int:
    return int(sum(int(parameter.numel()) for parameter in model.parameters()))


def build_model_parameter_summary(model: ModelWithParameters) -> dict[str, int | float]:
    parameter_count = count_model_parameters(model)
    return {
        "model_param_count": parameter_count,
        "model_param_million": parameter_count / 1_000_000.0,
    }
