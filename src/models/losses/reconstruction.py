from __future__ import annotations


def compute_l1_loss(predictions: list[float], targets: list[float]) -> float:
    """Compute the mean absolute error for scalar predictions."""
    absolute_errors: list[float] = [abs(prediction - target) for prediction, target in zip(predictions, targets)]
    return sum(absolute_errors) / len(absolute_errors)
