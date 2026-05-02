from __future__ import annotations

from typing import TypedDict


class BlockSpec(TypedDict):
    name: str
    channels: int
    activation: str


def build_block_spec(name: str, channels: int, activation: str) -> BlockSpec:
    """Describe a reusable network block."""
    return BlockSpec(name=name, channels=channels, activation=activation)
