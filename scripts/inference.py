from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any
from typing import Iterable

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.pipeline import prepare_inference_plan
from src.utils.config import load_experiment_config
from src.utils.logger import build_logger


def parse_args() -> argparse.Namespace:
    """Parse the inference CLI arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Inference template for the formal research pipeline.")
    parser.add_argument("--config", required=True, help="Path to an experiment config YAML file.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file.")
    parser.add_argument("--output-path", required=True, help="Path to the output prediction file.")
    parser.add_argument("--mode", required=True, choices=["dry-run", "infer"], help="Use 'dry-run' to validate config or 'infer' after filling the template hooks.")
    return parser.parse_args()


def main() -> int:
    """Load config and run the selected inference mode."""
    args: argparse.Namespace = parse_args()
    project_root: Path = PROJECT_ROOT
    config_path: Path = resolve_cli_path(project_root=project_root, cli_path=args.config)
    checkpoint_path: Path = resolve_cli_path(project_root=project_root, cli_path=args.checkpoint)
    output_path: Path = resolve_cli_path(project_root=project_root, cli_path=args.output_path)
    config: dict[str, Any] = load_experiment_config(project_root=project_root, experiment_config_path=config_path)
    plan: dict[str, object] = prepare_inference_plan(config=config, checkpoint_path=checkpoint_path, output_path=output_path)
    logger: logging.Logger = build_logger(logger_name="scripts.inference")

    if args.mode == "dry-run":
        logger.info("Validated inference config and prepared the runtime plan.")
        print(json.dumps(plan, indent=2))
        return 0

    run_inference_template(config=config, checkpoint_path=checkpoint_path, output_path=output_path, logger=logger)
    return 0


def resolve_cli_path(project_root: Path, cli_path: str) -> Path:
    """Resolve a CLI path relative to the project root."""
    raw_path: Path = Path(cli_path)
    return raw_path.resolve() if raw_path.is_absolute() else (project_root / raw_path).resolve()


def run_inference_template(
    config: dict[str, Any],
    checkpoint_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Run the general inference template after the project-specific hooks are implemented."""
    model: object = build_model(config=config)
    load_checkpoint(checkpoint_path=checkpoint_path, model=model)
    inference_loader: Iterable[object] = build_inference_loader(config=config)

    predictions: list[dict[str, object]] = []
    for batch in inference_loader:
        batch_predictions: list[dict[str, object]] = run_inference_step(model=model, batch=batch)
        predictions.extend(batch_predictions)

    save_predictions(output_path=output_path, predictions=predictions)
    logger.info("Completed inference and wrote predictions to %s.", output_path)


def build_model(config: dict[str, Any]) -> object:
    """Replace this hook with your real model construction code."""
    raise NotImplementedError(
        "Implement build_model() in scripts/inference.py with your IFRNet-based model builder.",
    )


def load_checkpoint(checkpoint_path: Path, model: object) -> None:
    """Replace this hook with your checkpoint loading logic."""
    raise NotImplementedError(
        "Implement load_checkpoint() in scripts/inference.py with your checkpoint restore logic.",
    )


def build_inference_loader(config: dict[str, Any]) -> Iterable[object]:
    """Replace this hook with your inference dataloader construction code."""
    raise NotImplementedError(
        "Implement build_inference_loader() in scripts/inference.py with your dataset and dataloader setup.",
    )


def run_inference_step(model: object, batch: object) -> list[dict[str, object]]:
    """Replace this hook with your actual inference logic."""
    raise NotImplementedError(
        "Implement run_inference_step() in scripts/inference.py with your forward-pass and decoding logic.",
    )


def save_predictions(output_path: Path, predictions: list[dict[str, object]]) -> None:
    """Replace this hook with your prediction serialization logic."""
    raise NotImplementedError(
        "Implement save_predictions() in scripts/inference.py with your output serialization logic.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
