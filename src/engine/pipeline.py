from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import TypedDict

from src.utils.io import ensure_directory


class TrainingContext(TypedDict):
    experiment_name: str
    model_name: str
    output_directories: dict[str, str]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    epochs: int
    batch_size: int


class EvaluationPlan(TypedDict):
    experiment_name: str
    checkpoint_path: str
    metrics: list[str]


class InferencePlan(TypedDict):
    experiment_name: str
    model_name: str
    checkpoint_path: str
    output_path: str


def prepare_training_context(project_root: Path, config: dict[str, Any], output_root: Path) -> TrainingContext:
    """Prepare directories and summarize the training run context."""
    experiment_config: dict[str, Any] = require_section(config=config, section_name="experiment")
    model_config: dict[str, Any] = require_section(config=config, section_name="model")
    train_config: dict[str, Any] = require_section(config=config, section_name="train")

    experiment_name: str = str(experiment_config.get("name"))
    model_name: str = str(model_config.get("name"))
    epochs: int = int(train_config.get("epochs"))
    batch_size: int = int(train_config.get("batch_size"))
    output_directories: dict[str, str] = prepare_output_directories(project_root=project_root, output_root=output_root)

    return TrainingContext(
        experiment_name=experiment_name,
        model_name=model_name,
        output_directories=output_directories,
        optimizer=build_optimizer_config(train_config=train_config),
        scheduler=build_scheduler_config(train_config=train_config),
        epochs=epochs,
        batch_size=batch_size,
    )


def prepare_evaluation_plan(config: dict[str, Any], checkpoint_path: Path) -> EvaluationPlan:
    """Summarize an evaluation run."""
    experiment_config: dict[str, Any] = require_section(config=config, section_name="experiment")
    model_config: dict[str, Any] = require_section(config=config, section_name="model")

    experiment_name: str = str(experiment_config.get("name"))
    loss_names: list[str] = list(model_config.get("losses", []))
    return EvaluationPlan(
        experiment_name=experiment_name,
        checkpoint_path=str(checkpoint_path.resolve()),
        metrics=loss_names,
    )


def prepare_inference_plan(config: dict[str, Any], checkpoint_path: Path, output_path: Path) -> InferencePlan:
    """Summarize a formal inference run."""
    experiment_config: dict[str, Any] = require_section(config=config, section_name="experiment")
    model_config: dict[str, Any] = require_section(config=config, section_name="model")

    experiment_name: str = str(experiment_config.get("name"))
    model_name: str = str(model_config.get("name"))
    return InferencePlan(
        experiment_name=experiment_name,
        model_name=model_name,
        checkpoint_path=str(checkpoint_path.resolve()),
        output_path=str(output_path.resolve()),
    )


def prepare_output_directories(project_root: Path, output_root: Path) -> dict[str, str]:
    """Ensure the standard output directories exist."""
    resolved_output_root: Path = (project_root / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()
    checkpoints_dir: Path = ensure_directory(path=resolved_output_root / "checkpoints")
    logs_dir: Path = ensure_directory(path=resolved_output_root / "logs")
    predictions_dir: Path = ensure_directory(path=resolved_output_root / "predictions")
    figures_dir: Path = ensure_directory(path=resolved_output_root / "figures")

    return {
        "root": str(resolved_output_root),
        "checkpoints": str(checkpoints_dir),
        "logs": str(logs_dir),
        "predictions": str(predictions_dir),
        "figures": str(figures_dir),
    }


def build_optimizer_config(train_config: dict[str, Any]) -> dict[str, Any]:
    """Extract the optimizer configuration from the training section."""
    optimizer_name: str = str(train_config.get("optimizer"))
    learning_rate: float = float(train_config.get("learning_rate"))
    return {
        "name": optimizer_name,
        "learning_rate": learning_rate,
    }


def build_scheduler_config(train_config: dict[str, Any]) -> dict[str, Any]:
    """Extract the scheduler configuration from the training section."""
    scheduler_name: str = str(train_config.get("scheduler"))
    return {
        "name": scheduler_name,
    }


def require_section(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Read a named config section and validate its shape."""
    section: object = config.get(section_name)
    if not isinstance(section, dict):
        raise KeyError(f"Config is missing a '{section_name}' section.")

    return section
