from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterator

from src.utils.config import load_yaml_file

DEFAULT_PATHS_CONFIG_PATH: Path = Path(__file__).parents[2] / "configs" / "paths" / "default.yaml"
ACTIVE_DATASET_ROOT_KEY: str = "default_dataset_root"


@dataclass(frozen=True)
class DatasetConfig:
    record: str
    main_idx: str
    difficulty: str
    sub_idx: str
    fps: int
    max_index: int

    @property
    def mode_path(self) -> str:
        return f"{self.main_idx}_{self.difficulty}/{self.main_idx}_{self.difficulty}_{self.sub_idx}/fps_{self.fps}"

    @property
    def mode_name(self) -> str:
        return f"{self.main_idx}_{self.difficulty}_{self.sub_idx}_fps_{self.fps}"

    @property
    def mode_index(self) -> str:
        return f"{self.main_idx}_fps_{self.fps}"

    @property
    def record_name(self) -> str:
        return self.record


@dataclass(frozen=True)
class RecordConfig:
    main_indices: tuple[str, ...]
    difficulties: tuple[str, ...]
    sub_indices: tuple[str, ...]
    fps_values: tuple[int, ...]
    max_indices: tuple[int, ...]


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    records: dict[str, RecordConfig]


def make_record_config(
    main_indices: tuple[str, ...],
    difficulties: tuple[str, ...],
    sub_indices: tuple[str, ...],
    fps_values: tuple[int, ...],
    max_indices: tuple[int, ...],
) -> RecordConfig:
    """Build one record config."""
    return RecordConfig(
        main_indices=main_indices,
        difficulties=difficulties,
        sub_indices=sub_indices,
        fps_values=fps_values,
        max_indices=max_indices,
    )


def build_dataset_preset(name: str, records: dict[str, RecordConfig]) -> DatasetPreset:
    """Build a named dataset preset."""
    return DatasetPreset(name=name, records=records)


def iter_dataset_configs(dataset_preset: DatasetPreset) -> Iterator[DatasetConfig]:
    """Yield one concrete dataset config for each main-index, difficulty, and fps combination."""
    for record_name, record_config in dataset_preset.records.items():
        fps_to_max_index: dict[int, int] = dict(zip(record_config.fps_values, record_config.max_indices))

        for main_idx, sub_idx in zip(record_config.main_indices, record_config.sub_indices):
            for difficulty, fps in product(record_config.difficulties, record_config.fps_values):
                yield DatasetConfig(
                    record=record_name,
                    main_idx=main_idx,
                    difficulty=difficulty,
                    sub_idx=sub_idx,
                    fps=fps,
                    max_index=fps_to_max_index[fps],
                )


def build_sequence_directory(dataset_config: DatasetConfig, paths_config_path: Path | None) -> Path:
    """Resolve the directory for one dataset sequence config under the active root directory."""
    active_root_dir: Path = resolve_active_dataset_root(paths_config_path=paths_config_path)
    return active_root_dir / dataset_config.record / dataset_config.mode_path


def get_dataset_preset(preset_name: str) -> DatasetPreset:
    """Return one registered dataset preset by key."""
    dataset_preset: DatasetPreset | None = DATASET_PRESETS.get(preset_name)
    if dataset_preset is None:
        available_names: str = ", ".join(sorted(DATASET_PRESETS.keys()))
        raise KeyError(f"Unknown dataset preset '{preset_name}'. Available presets: {available_names}.")

    return dataset_preset


def list_dataset_presets() -> tuple[str, ...]:
    """Return all registered preset names."""
    return tuple(sorted(DATASET_PRESETS.keys()))


def load_dataset_roots(paths_config_path: Path | None) -> dict[str, Path]:
    """Load dataset root mappings from the shared paths config."""
    config_path: Path = DEFAULT_PATHS_CONFIG_PATH if paths_config_path is None else paths_config_path
    paths_config: dict[str, object] = load_yaml_file(config_path=config_path)
    dataset_roots_raw: dict[str, str] = paths_config["dataset_roots"]
    return {root_key: Path(root_value) for root_key, root_value in dataset_roots_raw.items()}


def resolve_active_dataset_root(paths_config_path: Path | None) -> Path:
    """Resolve the active dataset root from the global selector and shared paths config."""
    dataset_roots: dict[str, Path] = load_dataset_roots(paths_config_path=paths_config_path)
    return dataset_roots[ACTIVE_DATASET_ROOT_KEY]


def set_active_dataset_root_key(root_key: str) -> None:
    """Set the active dataset root key for the current Python process."""
    global ACTIVE_DATASET_ROOT_KEY
    ACTIVE_DATASET_ROOT_KEY = root_key


def parse_args() -> argparse.Namespace:
    """Parse module-level smoke-check arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Dataset preset smoke checks.")
    parser.add_argument("--preset", required=False, default="train_vfx_0416", help="Preset key to inspect.")
    parser.add_argument("--limit", required=False, type=int, default=5, help="Number of example configs to print.")
    parser.add_argument("--paths-config", required=False, help="Optional path to the shared paths config YAML file.")
    return parser.parse_args()


def run_smoke_check(preset_name: str, limit: int, paths_config_path: Path | None) -> None:
    """Run a lightweight smoke check for one dataset preset."""
    dataset_preset: DatasetPreset = get_dataset_preset(preset_name=preset_name)
    dataset_configs: list[DatasetConfig] = list(iter_dataset_configs(dataset_preset=dataset_preset))
    active_root_dir: Path = resolve_active_dataset_root(paths_config_path=paths_config_path)

    print(f"preset_key={preset_name}")
    print(f"preset_name={dataset_preset.name}")
    print(f"active_root_key={ACTIVE_DATASET_ROOT_KEY}")
    print(f"root_dir={active_root_dir}")
    print(f"record_count={len(dataset_preset.records)}")
    print(f"config_count={len(dataset_configs)}")
    print(f"available_presets={list_dataset_presets()}")

    for dataset_config in dataset_configs[:limit]:
        sequence_directory: Path = build_sequence_directory(
            dataset_config=dataset_config,
            paths_config_path=paths_config_path,
        )
        print(
            "config="
            f"{dataset_config.record_name} | "
            f"{dataset_config.mode_name} | "
            f"max_index={dataset_config.max_index} | "
            f"path={sequence_directory}",
        )


MINOR_DATASET_PRESET: DatasetPreset = build_dataset_preset(
    name="AnimeFantasyRPG_3",
    records={
        "AnimeFantasyRPG_3_60": make_record_config(
            main_indices=("0", "1"),
            difficulties=("Easy", "Medium"),
            sub_indices=("1", "1"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

FULL_DATASET_PRESET: DatasetPreset = build_dataset_preset(
    name="AnimeFantasyRPG_3_Full",
    records={
        "AnimeFantasyRPG_3_60": make_record_config(
            main_indices=("0", "1", "2", "3"),
            difficulties=("Easy", "Medium", "Difficult"),
            sub_indices=("1", "1", "1", "1"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

TRAIN_DATASET_PRESET: DatasetPreset = build_dataset_preset(
    name="AnimeFantasyRPG_3_Full_Train",
    records={
        "AnimeFantasyRPG_3_60": make_record_config(
            main_indices=("0", "1", "2"),
            difficulties=("Easy", "Medium", "Difficult"),
            sub_indices=("2", "2", "2"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "AnimeFantasyRPG_2_60": make_record_config(
            main_indices=("4",),
            difficulties=("Easy", "Medium", "Difficult"),
            sub_indices=("2",),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

VFX_DATASET_PRESET: DatasetPreset = build_dataset_preset(
    name="AnimeFantasyRPG_3_VFX",
    records={
        "AnimeFantasyRPG_3_60": make_record_config(
            main_indices=("0", "1", "2", "3"),
            difficulties=("Difficult",),
            sub_indices=("1", "1", "1", "1"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

STAIR_DATASET_PRESET: DatasetPreset = build_dataset_preset(
    name="AnimeFantasyRPG_2_STAIR",
    records={
        "AnimeFantasyRPG_2_60": make_record_config(
            main_indices=("4",),
            difficulties=("Easy", "Medium", "Difficult"),
            sub_indices=("0",),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

TEST_DATASET_PRESET: DatasetPreset = build_dataset_preset(
    name="AnimeFantasyRPG_3_Test",
    records={
        "AnimeFantasyRPG_3_60": make_record_config(
            main_indices=("3",),
            difficulties=("Easy", "Medium", "Difficult"),
            sub_indices=("2",),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

TEST_VFX_DATASET_PRESET: DatasetPreset = build_dataset_preset(
    name="AnimeFantasyRPG_Test_VFX",
    records={
        "AnimeFantasyRPG_3_60": make_record_config(
            main_indices=("3", "3"),
            difficulties=("Difficult",),
            sub_indices=("3", "5"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "AnimeFantasyRPG_5_60": make_record_config(
            main_indices=("0", "3", "3"),
            difficulties=("Medium", "Difficult"),
            sub_indices=("2", "2", "5"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

TRAIN_VFX_0326_DATASET_PRESET: DatasetPreset = build_dataset_preset(
    name="VFI_0326_Train_VFX",
    records={
        "ARPG_3": make_record_config(
            main_indices=("0", "0", "0", "0", "1", "1", "1", "1", "2", "2", "2", "2"),
            difficulties=("Difficult",),
            sub_indices=("0", "1", "3", "4", "0", "1", "3", "4", "0", "1", "3", "4"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "ARPG_2": make_record_config(
            main_indices=("4", "4", "4", "4"),
            difficulties=("Difficult",),
            sub_indices=("0", "1", "3", "4"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

TEST_VFX_0326_DATASET_PRESET: DatasetPreset = build_dataset_preset(
    name="VFI_0326_Test_VFX",
    records={
        "ARPG_3": make_record_config(
            main_indices=("3", "3", "3", "3"),
            difficulties=("Difficult",),
            sub_indices=("0", "1", "3", "4"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

TEST_UNSEEN_VFX_0326_DATASET_PRESET: DatasetPreset = build_dataset_preset(
    name="VFI_0326_Test_Unseen_VFX",
    records={
        "ARPG_3": make_record_config(
            main_indices=("0", "0", "1", "1", "2", "2", "3", "3"),
            difficulties=("Difficult",),
            sub_indices=("2", "5", "2", "5", "2", "5", "2", "5"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "ARPG_2": make_record_config(
            main_indices=("4", "4"),
            difficulties=("Difficult",),
            sub_indices=("2", "5"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

TEST_3D_VFX_DATASET_PRESET: DatasetPreset = build_dataset_preset(
    name="VFI_0326_Test_3D_VFX",
    records={
        "ARPG_5": make_record_config(
            main_indices=("0", "0", "0", "1", "1", "1", "3", "3", "3", "4", "4", "4"),
            difficulties=("Difficult",),
            sub_indices=("0", "1", "2", "0", "1", "2", "0", "1", "2", "0", "1", "2"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "ARPG_6": make_record_config(
            main_indices=("0", "0", "0", "2", "2", "2", "5", "5", "5", "6", "6", "6"),
            difficulties=("Difficult",),
            sub_indices=("0", "1", "2", "0", "1", "2", "0", "1", "2", "0", "1", "2"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "ARPG_7": make_record_config(
            main_indices=("1", "1", "1", "2", "2", "2", "3", "3", "3", "6", "6", "6"),
            difficulties=("Difficult",),
            sub_indices=("0", "1", "2", "0", "1", "2", "0", "1", "2", "0", "1", "2"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

TRAIN_VFX_0416_DATASET_PRESET: DatasetPreset = build_dataset_preset(
    name="ARPG_2nd_Patch_Train_VFX",
    records={
        "ARPG_3": make_record_config(
            main_indices=("0", "0", "0", "0", "1", "1", "1", "1", "2", "2", "2", "2"),
            difficulties=("Difficult",),
            sub_indices=("0", "1", "3", "4", "0", "1", "3", "4", "0", "1", "3", "4"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "ARPG_2": make_record_config(
            main_indices=("4", "4", "4", "4"),
            difficulties=("Difficult",),
            sub_indices=("0", "1", "3", "4"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "ARPG_5": make_record_config(
            main_indices=("0", "0", "1", "1", "4", "4"),
            difficulties=("Difficult",),
            sub_indices=("0", "1", "0", "1", "0", "1"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "ARPG_6": make_record_config(
            main_indices=("0", "0", "2", "2", "5", "5"),
            difficulties=("Difficult",),
            sub_indices=("0", "1", "0", "1", "0", "1"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "ARPG_7": make_record_config(
            main_indices=("1", "1", "2", "2"),
            difficulties=("Difficult",),
            sub_indices=("0", "1", "0", "1"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

TEST_VFX_0416_DATASET_PRESET: DatasetPreset = build_dataset_preset(
    name="ARPG_2nd_Patch_Test_VFX",
    records={
        "ARPG_3": make_record_config(
            main_indices=("0", "0", "1", "1", "2", "2", "3", "3"),
            difficulties=("Difficult",),
            sub_indices=("2", "5", "2", "5", "2", "5", "2", "5"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "ARPG_2": make_record_config(
            main_indices=("4", "4"),
            difficulties=("Difficult",),
            sub_indices=("2", "5"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "ARPG_5": make_record_config(
            main_indices=("0", "1", "4", "3", "3", "3"),
            difficulties=("Difficult",),
            sub_indices=("2", "2", "2", "0", "1", "2"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "ARPG_6": make_record_config(
            main_indices=("0", "2", "5", "6", "6", "6"),
            difficulties=("Difficult",),
            sub_indices=("2", "2", "2", "0", "1", "2"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
        "ARPG_7": make_record_config(
            main_indices=("1", "2", "3", "3", "3", "6", "6", "6"),
            difficulties=("Difficult",),
            sub_indices=("2", "2", "0", "1", "2", "0", "1", "2"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

SMOKE_ARPG2_DUAL_PRESET: DatasetPreset = build_dataset_preset(
    name="ARPG_2 Dual Difficult Smoke",
    records={
        "ARPG_2": make_record_config(
            main_indices=("4", "4"),
            difficulties=("Difficult",),
            sub_indices=("0", "1"),
            fps_values=(30, 60),
            max_indices=(400, 800),
        ),
    },
)

DATASET_PRESETS: dict[str, DatasetPreset] = {
    "minor": MINOR_DATASET_PRESET,
    "full": FULL_DATASET_PRESET,
    "train": TRAIN_DATASET_PRESET,
    "vfx": VFX_DATASET_PRESET,
    "stair": STAIR_DATASET_PRESET,
    "test": TEST_DATASET_PRESET,
    "test_vfx": TEST_VFX_DATASET_PRESET,
    "train_vfx_0326": TRAIN_VFX_0326_DATASET_PRESET,
    "test_vfx_0326": TEST_VFX_0326_DATASET_PRESET,
    "test_unseen_vfx_0326": TEST_UNSEEN_VFX_0326_DATASET_PRESET,
    "test_3d_vfx_0326": TEST_3D_VFX_DATASET_PRESET,
    "train_vfx_0416": TRAIN_VFX_0416_DATASET_PRESET,
    "test_vfx_0416": TEST_VFX_0416_DATASET_PRESET,
    "smoke_arpg2_dual": SMOKE_ARPG2_DUAL_PRESET,
}

TRAIN_VFX_0416_DATASET_CONFIGS: DatasetPreset = TRAIN_VFX_0416_DATASET_PRESET
TEST_VFX_0416_DATASET_CONFIGS: DatasetPreset = TEST_VFX_0416_DATASET_PRESET


if __name__ == "__main__":
    module_args: argparse.Namespace = parse_args()
    run_smoke_check(
        preset_name=module_args.preset,
        limit=module_args.limit,
        paths_config_path=Path(module_args.paths_config) if module_args.paths_config is not None else None,
    )
