from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def load_yaml_file(config_path: Path) -> dict[str, Any]:
    """Load a JSON-formatted YAML-compatible config file."""
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_experiment_config(project_root: Path, experiment_config_path: Path) -> dict[str, Any]:
    """Load an experiment config and its referenced sub-configs."""
    experiment_payload: dict[str, Any] = load_yaml_file(config_path=experiment_config_path)
    resolved_sections: dict[str, Any] = {"experiment": experiment_payload["experiment"]}
    refs_section: dict[str, str] = experiment_payload["config_refs"]

    for section_name in ("paths", "data", "model", "train"):
        config_reference = refs_section[section_name]
        config_path = Path(config_reference)
        resolved_path: Path = project_root / config_path
        resolved_sections[section_name] = load_yaml_file(config_path=resolved_path)

    return resolved_sections
