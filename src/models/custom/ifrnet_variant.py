from __future__ import annotations

from typing import TypedDict


class CustomModelSpec(TypedDict):
    name: str
    family: str
    variant: str
    losses: list[str]


def build_custom_model_spec(name: str, family: str, variant: str, losses: list[str]) -> CustomModelSpec:
    """Describe the active custom model variant."""
    return CustomModelSpec(name=name, family=family, variant=variant, losses=list(losses))
