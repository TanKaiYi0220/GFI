from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def load_yaml_file(config_path: Path) -> dict[str, Any]:
    """Load a JSON-formatted YAML-compatible mapping from disk."""
    with config_path.open("r", encoding="utf-8") as handle:
        payload: object = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"Config at '{config_path}' must be a mapping.")

    return dict(payload)


def load_experiment_config(project_root: Path, experiment_config_path: Path) -> dict[str, Any]:
    """Load a full experiment config with referenced sub-configs."""
    experiment_payload: dict[str, Any] = load_yaml_file(config_path=experiment_config_path)
    experiment_section: object = experiment_payload.get("experiment")
    refs_section: object = experiment_payload.get("config_refs")

    if not isinstance(experiment_section, dict):
        raise KeyError(f"Experiment config '{experiment_config_path}' needs an 'experiment' section.")

    if not isinstance(refs_section, dict):
        raise KeyError(f"Experiment config '{experiment_config_path}' needs a 'config_refs' section.")

    resolved_sections: dict[str, Any] = {"experiment": dict(experiment_section)}
    for section_name in ("paths", "data", "model", "train"):
        reference: object = refs_section.get(section_name)
        if not isinstance(reference, str) or reference == "":
            raise KeyError(f"Experiment config '{experiment_config_path}' is missing config ref '{section_name}'.")

        resolved_path: Path = resolve_config_path(project_root=project_root, config_reference=reference)
        resolved_sections[section_name] = load_yaml_file(config_path=resolved_path)

    return resolved_sections


def resolve_config_path(project_root: Path, config_reference: str) -> Path:
    """Resolve a config path relative to the project root."""
    config_path: Path = Path(config_reference)
    resolved_path: Path = config_path.resolve() if config_path.is_absolute() else (project_root / config_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Referenced config file does not exist: '{resolved_path}'.")

    return resolved_path
