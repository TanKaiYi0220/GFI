from __future__ import annotations

from typing import TypedDict


class ExternalModelSpec(TypedDict):
    name: str
    source: str
    entrypoint: str


def build_external_model_spec(name: str, source: str, entrypoint: str) -> ExternalModelSpec:
    """Describe an external model without exposing implementation details."""
    return ExternalModelSpec(name=name, source=source, entrypoint=entrypoint)
