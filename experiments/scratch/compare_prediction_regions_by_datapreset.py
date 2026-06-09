from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from typing import TypedDict

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity

from src.data.dataset_config import get_dataset_preset
from src.data.dataset_config import iter_dataset_configs
from src.data.dataset_config import list_dataset_presets
from src.data.dataset_loader import depth_to_tensor
from src.data.dataset_loader import FlowEstimationTrainDataset
from src.data.dataset_loader import flow_to_tensor
from src.data.dataset_loader import VFITrainDataset
from src.data.image_ops import flow_to_image
from src.data.image_ops import load_backward_velocity
from src.data.image_ops import save_image
from src.engine.flow_approx import build_flow_init_result
from src.engine.flow_approx import flatten_target_index
from src.engine.flow_approx import make_source_grid
from src.engine.flow_approx import SPLATTING_FLOW_APPROX_METHODS
from src.utils.config import load_yaml_file
from scripts.inference import BASELINE_MODEL_NAME
from scripts.inference import InferenceBatchResult
from scripts.inference import RESIDUAL_FLOW_APPROX_MODEL_NAME
from scripts.inference import RESIDUAL_MODEL_NAME
from scripts.inference import run_inference_batch
from scripts.inference import save_selected_sample_artifacts
from scripts.train import read_model_init_args
from scripts.train import resolve_model_class


class SamplePreset(TypedDict):
    sample_id: str
    sample_dir: str
    dataset_root_dir: str
    record: str
    mode: str
    model_result_dirs: dict[str, str]
    frame_0: int
    frame_t: int
    frame_1: int
    fps: int


class DataPreset(TypedDict):
    models: list["CompareModelConfig"]
    samples: list[SamplePreset]


class CompareModelConfig(TypedDict):
    key: str
    name: str
    result_root: str
    inference_config: str
    epoch: str


class InferenceModelSpec(TypedDict):
    config_path: str
    model_name: str
    flow_approx_method: str
    scale_factor: float
    input_fps: int
    flow_diff_threshold: float
    flow_diff_percentile: float
    model: Any


RegionMaps = dict[str, np.ndarray]
PredictionMaps = dict[str, np.ndarray]
InferenceResultMaps = dict[str, InferenceBatchResult]

KERNEL_SIZE: int = 9
CONFIDENCE_THRESHOLD: float = 0.08
WINNER_ALPHA_MAX: float = 0.70
WINNER_ALPHA_FULL_RMSE_DELTA: float = 8.0
NEUTRAL_ALPHA: float = 0.30
SAVE_OVERLAY_IMAGES: bool = True
SAVE_SELECTED_CASES_ONLY: bool = False
SELECTED_CASES_PER_BUCKET: int = 12
SEQUENCE_LENGTH: int = 0
OUTPUT_ROOT: Path = Path(tempfile.gettempdir()) / "GFI_prediction_region_compare"
MODEL_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.08, 0.04),
    (0.02, 0.38, 0.95),
    (0.10, 0.62, 0.26),
    (0.78, 0.26, 0.86),
    (0.95, 0.58, 0.08),
    (0.00, 0.62, 0.70),
)
NEUTRAL_COLOR: tuple[float, float, float] = (0.55, 0.55, 0.55)

DATA_PRESETS: dict[str, DataPreset] = {
    "arpg_3_1_difficult_5_0452_0454": {
        "models": [
            {
                "key": "baseline",
                "name": "FineTuning",
                "result_root": "",
                "inference_config": "",
                "epoch": "latest",
            },
            {
                "key": "candidate",
                "name": "Splat Training",
                "result_root": "",
                "inference_config": "",
                "epoch": "latest",
            },
        ],
        "samples": [
            {
                "sample_id": "ARPG_3_1_Difficult_5_frame_0452_0454",
                "sample_dir": (
                    r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260527 - Lab Meeting"
                    r"\warping_testing_case\ARPG_3_1_Difficult_5\Medium_frame_0452_0454"
                ),
                "dataset_root_dir": "",
                "record": "",
                "mode": "",
                "model_result_dirs": {
                    "candidate": (
                        r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260527 - Lab Meeting"
                        r"\warping_testing_case\ARPG_3_1_Difficult_5_Splat"
                    ),
                    "baseline": (
                        r"C:\Users\User\Desktop\CGVLab\GFI\Meeting-2026\20260527 - Lab Meeting"
                        r"\warping_testing_case\ARPG_3_1_Difficult_5_FineTuning"
                    ),
                },
                "frame_0": 452,
                "frame_t": 453,
                "frame_1": 454,
                "fps": 60,
            },
        ],
    },
}


def load_config_defaults(argv: list[str] | None) -> dict[str, Any]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str)
    config_args, _remaining = config_parser.parse_known_args(argv)
    if config_args.config is None:
        return {}

    config_path = resolve_input_path(str(config_args.config))
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a JSON object: path={config_path}")

    config["config_path"] = str(config_path)
    return config


def config_default(config: dict[str, Any], key: str, fallback: Any) -> Any:
    return config[key] if key in config else fallback


def is_non_empty_text(value: str) -> bool:
    return value.strip() != ""


def validate_model_key(model_key: str) -> None:
    invalid_characters = set('<>:"/\\|?*')
    if model_key == "":
        raise ValueError("Model key must be non-empty")
    if any(character in invalid_characters for character in model_key):
        raise ValueError(f"Model key contains a path-unsafe character: key={model_key}")


def require_model_text(model_config: dict[str, Any], key: str, model_index: int) -> str:
    if key not in model_config:
        raise KeyError(f"Missing model config field: index={model_index} field={key}")

    value = str(model_config[key])
    if not is_non_empty_text(value):
        raise ValueError(f"Model config field must be non-empty: index={model_index} field={key}")
    return value


def parse_compare_model_configs(raw_models: list[Any]) -> list[CompareModelConfig]:
    if len(raw_models) < 2:
        raise ValueError("Config field models must contain at least two models")

    model_configs: list[CompareModelConfig] = []
    seen_keys: set[str] = set()
    for model_index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise TypeError(f"Each models entry must be an object: index={model_index} type={type(raw_model).__name__}")

        model_key = require_model_text(raw_model, "key", model_index)
        validate_model_key(model_key)
        if model_key in seen_keys:
            raise ValueError(f"Duplicate model key: key={model_key}")
        seen_keys.add(model_key)

        model_configs.append(
            {
                "key": model_key,
                "name": require_model_text(raw_model, "name", model_index),
                "result_root": str(raw_model.get("result_root", "")),
                "inference_config": str(raw_model.get("inference_config", "")),
                "epoch": str(raw_model.get("epoch", "latest")),
            }
        )

    return model_configs


def build_compare_model_configs(args: argparse.Namespace) -> list[CompareModelConfig]:
    raw_models = args.models
    if isinstance(raw_models, list) and len(raw_models) > 0:
        return parse_compare_model_configs(raw_models)

    raise ValueError("Config field models is required when using --dataset-preset")


def validate_compare_model_config_modes(model_configs: list[CompareModelConfig]) -> None:
    inference_count = sum(1 for model_config in model_configs if is_non_empty_text(model_config["inference_config"]))
    if inference_count == len(model_configs):
        return
    if inference_count == 0:
        return
    raise ValueError("All models must use inference_config or all models must use result_root; mixed model sources are not supported")


def model_configs_use_direct_inference(model_configs: list[CompareModelConfig]) -> bool:
    return all(is_non_empty_text(model_config["inference_config"]) for model_config in model_configs)


def normalize_data_preset(data_preset: DataPreset, model_configs: list[CompareModelConfig]) -> DataPreset:
    normalized_samples: list[SamplePreset] = []
    for sample in data_preset["samples"]:
        model_result_dirs = dict(sample.get("model_result_dirs", {}))
        normalized_sample = dict(sample)
        normalized_sample["model_result_dirs"] = model_result_dirs
        normalized_samples.append(normalized_sample)  # type: ignore[arg-type]

    return {
        "models": model_configs,
        "samples": normalized_samples,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config = load_config_defaults(argv)
    parser = argparse.ArgumentParser(description="Compare prediction quality by splatting regions.")
    parser.set_defaults(models=config_default(config, "models", []))
    parser.add_argument("--config", default=config.get("config_path"), type=str, help="JSON config path. CLI arguments override matching config values.")
    preset_group = parser.add_mutually_exclusive_group(required=False)
    preset_group.add_argument("--data-preset", default=config.get("data_preset"), choices=tuple(sorted(DATA_PRESETS.keys())), help="Scratch-local comparison preset.")
    preset_group.add_argument("--dataset-preset", default=config.get("dataset_preset"), choices=list_dataset_presets(), help="Project dataset preset, e.g. train_minor_0507.")
    parser.add_argument("--root-dir", default=config.get("root_dir"), type=str, help="Directory containing preprocessed CSV indexes.")
    parser.add_argument("--dataset-root-dir", default=config.get("dataset_root_dir"), type=str, help="Root directory containing raw frame and velocity assets.")
    parser.add_argument("--only-fps", default=int(config_default(config, "only_fps", 60)), type=int)
    parser.add_argument("--limit", default=int(config_default(config, "limit", 0)), type=int, help="Optional max sample count after filtering. 0 means no limit.")
    parser.add_argument("--mode-filter", default=config_default(config, "mode_filter", ""), type=str, help="Optional exact mode path filter.")
    parser.add_argument(
        "--skip-missing-results",
        default=bool(config_default(config, "skip_missing_results", False)),
        action="store_true",
        help="Skip rows whose prediction artifacts are not saved.",
    )
    parser.add_argument("--result-layout", default=config_default(config, "result_layout", "auto"), choices=("auto", "inference", "training-samples"))
    parser.add_argument("--sample-split", default=config_default(config, "sample_split", "train"), choices=("train", "test"))
    parser.add_argument("--output-path", default=config.get("output_path"), type=str, help="Final output directory. Defaults to a temp directory grouped by preset name.")
    parser.add_argument("--kernel-size", default=int(config_default(config, "kernel_size", KERNEL_SIZE)), type=int)
    parser.add_argument("--confidence-threshold", default=float(config_default(config, "confidence_threshold", CONFIDENCE_THRESHOLD)), type=float)
    parser.add_argument("--winner-alpha-max", default=float(config_default(config, "winner_alpha_max", WINNER_ALPHA_MAX)), type=float)
    parser.add_argument("--winner-alpha-full-rmse-delta", default=float(config_default(config, "winner_alpha_full_rmse_delta", WINNER_ALPHA_FULL_RMSE_DELTA)), type=float)
    parser.add_argument("--neutral-alpha", default=float(config_default(config, "neutral_alpha", NEUTRAL_ALPHA)), type=float)
    parser.add_argument("--save-overlay-images", default=bool(config_default(config, "save_overlay_images", SAVE_OVERLAY_IMAGES)), action="store_true")
    parser.add_argument("--no-save-overlay-images", dest="save_overlay_images", action="store_false")
    parser.add_argument("--save-selected-cases-only", default=bool(config_default(config, "save_selected_cases_only", SAVE_SELECTED_CASES_ONLY)), action="store_true")
    parser.add_argument("--no-save-selected-cases-only", dest="save_selected_cases_only", action="store_false")
    parser.add_argument("--selected-cases-per-bucket", default=int(config_default(config, "selected_cases_per_bucket", SELECTED_CASES_PER_BUCKET)), type=int)
    parser.add_argument("--sequence-length", default=int(config_default(config, "sequence_length", SEQUENCE_LENGTH)), type=int)
    args = parser.parse_args(argv)
    if args.data_preset is None and args.dataset_preset is None:
        parser.error("one of --data-preset or --dataset-preset is required, either in CLI or config")
    if args.data_preset is not None and args.dataset_preset is not None:
        parser.error("only one of --data-preset or --dataset-preset can be set")
    if args.dataset_preset is not None and not (isinstance(args.models, list) and len(args.models) > 0):
        parser.error("config field models is required when using --dataset-preset")
    if int(args.kernel_size) <= 0:
        parser.error("--kernel-size must be positive")
    if int(args.kernel_size) % 2 == 0:
        parser.error("--kernel-size must be odd")
    if float(args.confidence_threshold) < 0.0:
        parser.error("--confidence-threshold must be non-negative")
    if not 0.0 <= float(args.winner_alpha_max) <= 1.0:
        parser.error("--winner-alpha-max must be between 0 and 1")
    if float(args.winner_alpha_full_rmse_delta) <= 0.0:
        parser.error("--winner-alpha-full-rmse-delta must be positive")
    if not 0.0 <= float(args.neutral_alpha) <= 1.0:
        parser.error("--neutral-alpha must be between 0 and 1")
    if int(args.selected_cases_per_bucket) < 0:
        parser.error("--selected-cases-per-bucket must be non-negative")
    if int(args.sequence_length) < 0:
        parser.error("--sequence-length must be non-negative")
    return args


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: path={path}")

    return path


def read_rgb_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(require_file(path)), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: path={path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def read_model_prediction_images(sample: SamplePreset, model_configs: list[CompareModelConfig]) -> tuple[np.ndarray, PredictionMaps]:
    target: np.ndarray | None = None
    predictions: PredictionMaps = {}
    model_result_dirs = sample["model_result_dirs"]
    for model_config in model_configs:
        model_key = model_config["key"]
        if model_key not in model_result_dirs:
            raise KeyError(f"Missing model result dir: sample_id={sample['sample_id']} model_key={model_key}")

        model_dir = Path(model_result_dirs[model_key])
        model_target = read_rgb_image(model_dir / "image_gt.png")
        model_prediction = read_rgb_image(model_dir / "image_pred.png")
        if target is None:
            target = model_target
        else:
            validate_same_shape(f"{model_key}_target", target, model_target)
            if not np.allclose(target, model_target, atol=1.0 / 255.0):
                raise ValueError(f"Ground-truth mismatch: sample_id={sample['sample_id']} model_key={model_key}")

        validate_same_shape(f"{model_key}_prediction", model_target, model_prediction)
        predictions[model_key] = model_prediction

    if target is None:
        raise ValueError(f"No model predictions were read: sample_id={sample['sample_id']}")
    return target, predictions


def save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    image_u8 = np.clip(np.round(image_rgb * 255.0), 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def load_flow_and_depth_tensors(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    flow, depth = load_backward_velocity(require_file(path))
    flow_tensor = flow_to_tensor(flow).unsqueeze(0).to(device)
    depth_tensor = depth_to_tensor(depth).unsqueeze(0).to(device)
    return flow_tensor, depth_tensor


def resolve_model_checkpoint_path(config: dict[str, Any], config_path: Path) -> Path:
    if "checkpoint_path" in config:
        return resolve_input_path(str(config["checkpoint_path"]))

    if "output_dir" not in config:
        raise KeyError(f"Config must contain checkpoint_path or output_dir: path={config_path}")

    checkpoints_dir = resolve_input_path(str(config["output_dir"])) / "checkpoints"
    best_path = checkpoints_dir / "best.pth"
    latest_path = checkpoints_dir / "latest.pth"
    if best_path.is_file():
        return best_path
    if latest_path.is_file():
        return latest_path

    raise FileNotFoundError(
        "Could not resolve model checkpoint from training config: "
        f"path={config_path} best_path={best_path} latest_path={latest_path}"
    )


def load_inference_model(config_path_text: str, device: torch.device) -> InferenceModelSpec:
    config_path = resolve_input_path(config_path_text)
    config = load_yaml_file(config_path)
    model_name = str(config["model_name"])
    model_init_args = read_model_init_args(config)
    checkpoint_path = resolve_model_checkpoint_path(config=config, config_path=config_path)
    model_class = resolve_model_class(model_name)
    model = model_class(**model_init_args).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "config_path": str(config_path),
        "model_name": model_name,
        "flow_approx_method": str(config["flow_approx_method"]),
        "scale_factor": float(config.get("scale_factor", 1.0)),
        "input_fps": int(config["input_fps"]),
        "flow_diff_threshold": float(config.get("flow_diff_threshold", 1.0)),
        "flow_diff_percentile": float(config.get("flow_diff_percentile", 99.0)),
        "model": model,
    }


def load_inference_models(model_configs: list[CompareModelConfig], device: torch.device) -> dict[str, InferenceModelSpec]:
    loaded_models: dict[str, InferenceModelSpec] = {}
    for model_config in model_configs:
        loaded_models[model_config["key"]] = load_inference_model(model_config["inference_config"], device)
    return loaded_models


def resolve_input_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_dataset_args(args: argparse.Namespace, model_configs: list[CompareModelConfig]) -> tuple[Path, Path]:
    needs_result_roots = not model_configs_use_direct_inference(model_configs)
    missing_names = [
        name
        for name in ("root_dir", "dataset_root_dir")
        if getattr(args, name) is None
    ]
    if needs_result_roots:
        missing_model_keys = [
            model_config["key"]
            for model_config in model_configs
            if not is_non_empty_text(model_config["result_root"])
        ]
        if len(missing_model_keys) > 0:
            raise ValueError(f"Missing result_root for models: keys={missing_model_keys}")
    if len(missing_names) > 0:
        raise ValueError(f"--dataset-preset requires these arguments: {', '.join('--' + name.replace('_', '-') for name in missing_names)}")

    return (
        resolve_input_path(str(args.root_dir)),
        resolve_input_path(str(args.dataset_root_dir)),
    )


def load_dataset_preset_dataframe(root_dir: Path, dataset_preset_name: str, only_fps: int) -> pd.DataFrame:
    dataset_preset = get_dataset_preset(dataset_preset_name)
    dataframe_list: list[pd.DataFrame] = []
    for dataset_config in iter_dataset_configs(dataset_preset):
        if dataset_config.fps != only_fps:
            continue

        csv_path = root_dir / f"{dataset_config.record_name}_preprocessed" / f"{dataset_config.mode_index}_raw_sequence_frame_index.csv"
        if not csv_path.is_file():
            continue

        dataframe = pd.read_csv(csv_path)
        dataframe["record"] = dataset_config.record
        dataframe["mode"] = dataset_config.mode_path
        dataframe["fps"] = dataset_config.fps
        dataframe_list.append(dataframe)

    if len(dataframe_list) == 0:
        raise RuntimeError(f"No dataset CSV found under root_dir={root_dir} for preset={dataset_preset_name} only_fps={only_fps}")

    return pd.concat(dataframe_list, ignore_index=True)


def build_frame_range(frame_0: int, frame_1: int) -> str:
    return f"frame_{frame_0:04d}_{frame_1:04d}"


def build_training_sample_key(record: str, mode: str, frame_0: int) -> str:
    mode_parts = mode.split("/")
    if len(mode_parts) < 2:
        raise ValueError(f"Mode path must contain difficulty and sequence parts: mode={mode}")

    sequence_parts = mode_parts[1].split("_")
    if len(sequence_parts) < 3:
        raise ValueError(f"Mode sequence part must look like 1_Medium_5: mode={mode}")

    record_suffix = record.replace("AnimeFantasyRPG_", "ARPG_")
    return f"{record_suffix}_{sequence_parts[0]}_{sequence_parts[1]}_{sequence_parts[2]}_{frame_0:04d}"


def remove_fps_parts(name_parts: list[str]) -> list[str]:
    filtered_parts: list[str] = []
    skip_next = False
    for index, name_part in enumerate(name_parts):
        if skip_next:
            skip_next = False
            continue

        next_part = name_parts[index + 1] if index + 1 < len(name_parts) else ""
        if name_part == "fps" and next_part.isdigit():
            skip_next = True
            continue

        filtered_parts.append(name_part)

    return filtered_parts


def remove_fps_tokens(name_part: str) -> str:
    return "_".join(remove_fps_parts(name_part.split("_")))


def build_case_record_name(sample: SamplePreset) -> str:
    record_name = sample["record"].replace("AnimeFantasyRPG_", "ARPG_")
    mode_parts: list[str] = []
    for mode_part in sample["mode"].split("/"):
        if mode_part == "":
            continue

        normalized_mode_part = remove_fps_tokens(mode_part)
        if normalized_mode_part != "":
            mode_parts.append(normalized_mode_part)

    if record_name != "" and len(mode_parts) > 0:
        return "_".join([record_name, *mode_parts])

    frame_range = build_frame_range(int(sample["frame_0"]), int(sample["frame_1"]))
    sample_id = sample["sample_id"]
    sample_id_without_frame = sample_id[: -len(f"_{frame_range}")] if sample_id.endswith(f"_{frame_range}") else sample_id
    return "_".join(remove_fps_parts(sample_id_without_frame.split("_")))


def resolve_epoch_dir(frame_dir: Path, epoch_name: str) -> Path:
    if epoch_name != "latest":
        return frame_dir / epoch_name

    if not frame_dir.is_dir():
        return frame_dir / "latest"

    epoch_dirs = sorted(
        (path for path in frame_dir.iterdir() if path.is_dir() and path.name.startswith("epoch_")),
        key=lambda path: path.name,
    )
    if len(epoch_dirs) == 0:
        return frame_dir / "latest"

    return epoch_dirs[-1]


def build_inference_result_dir(result_root: Path, record: str, mode: str, frame_range: str) -> Path:
    return result_root / record / mode / frame_range


def build_training_sample_result_dir(result_root: Path, split_name: str, frame_key: str, epoch_name: str) -> Path:
    return resolve_epoch_dir(result_root / "samples" / split_name / frame_key, epoch_name)


def choose_result_dir(
    result_root: Path,
    record: str,
    mode: str,
    frame_range: str,
    frame_key: str,
    split_name: str,
    epoch_name: str,
    result_layout: str,
) -> Path:
    inference_dir = build_inference_result_dir(result_root, record, mode, frame_range)
    training_dir = build_training_sample_result_dir(result_root, split_name, frame_key, epoch_name)
    if result_layout == "inference":
        return inference_dir
    if result_layout == "training-samples":
        return training_dir
    if result_files_exist_in_dir(training_dir):
        return training_dir
    return inference_dir


def build_dataset_sample(
    row: pd.Series,
    dataset_root_dir: Path,
    model_configs: list[CompareModelConfig],
    args: argparse.Namespace,
) -> SamplePreset:
    record = str(row["record"])
    mode = str(row["mode"])
    frame_0 = int(row["img0"])
    frame_t = int(row["img1"])
    frame_1 = int(row["img2"])
    fps = int(row["fps"])
    frame_range = build_frame_range(frame_0, frame_1)
    frame_key = build_training_sample_key(record, mode, frame_0)
    sample_id = f"{record}_{mode.replace('/', '_')}_{frame_range}"
    model_result_dirs: dict[str, str] = {}
    for model_config in model_configs:
        if not is_non_empty_text(model_config["result_root"]):
            continue

        model_result_dirs[model_config["key"]] = str(
            choose_result_dir(
                result_root=resolve_input_path(model_config["result_root"]),
                record=record,
                mode=mode,
                frame_range=frame_range,
                frame_key=frame_key,
                split_name=str(args.sample_split),
                epoch_name=model_config["epoch"],
                result_layout=str(args.result_layout),
            )
        )

    return {
        "sample_id": sample_id,
        "sample_dir": "",
        "dataset_root_dir": str(dataset_root_dir),
        "record": record,
        "mode": mode,
        "model_result_dirs": model_result_dirs,
        "frame_0": frame_0,
        "frame_t": frame_t,
        "frame_1": frame_1,
        "fps": fps,
    }


def result_files_exist_in_dir(result_dir: Path) -> bool:
    return (result_dir / "image_gt.png").is_file() and (result_dir / "image_pred.png").is_file()


def result_files_exist(sample: SamplePreset, model_configs: list[CompareModelConfig]) -> bool:
    model_result_dirs = sample["model_result_dirs"]
    return all(result_files_exist_in_dir(Path(model_result_dirs[model_config["key"]])) for model_config in model_configs)


def build_missing_result_preview(sample: SamplePreset, model_configs: list[CompareModelConfig]) -> dict[str, object]:
    model_result_dirs = sample["model_result_dirs"]
    model_previews = {
        model_config["key"]: {
            "result_dir": model_result_dirs.get(model_config["key"], ""),
            "files_exist": result_files_exist_in_dir(Path(model_result_dirs.get(model_config["key"], ""))),
        }
        for model_config in model_configs
    }
    return {
        "sample_id": sample["sample_id"],
        "models": model_previews,
    }


def build_samples_from_dataset_preset(args: argparse.Namespace, model_configs: list[CompareModelConfig]) -> DataPreset:
    root_dir, dataset_root_dir = require_dataset_args(args, model_configs)
    dataframe = load_dataset_preset_dataframe(root_dir, str(args.dataset_preset), int(args.only_fps))
    if "valid" in dataframe.columns:
        dataframe = dataframe[dataframe["valid"] == True].reset_index(drop=True)
    if str(args.mode_filter) != "":
        dataframe = dataframe[dataframe["mode"] == str(args.mode_filter)].reset_index(drop=True)

    samples: list[SamplePreset] = []
    missing_previews: list[dict[str, object]] = []
    for _index, row in dataframe.iterrows():
        sample = build_dataset_sample(row, dataset_root_dir, model_configs, args)
        if bool(args.skip_missing_results) and not model_configs_use_direct_inference(model_configs) and not result_files_exist(sample, model_configs):
            if len(missing_previews) < 5:
                missing_previews.append(build_missing_result_preview(sample, model_configs))
            continue
        samples.append(sample)
        if int(args.limit) > 0 and len(samples) >= int(args.limit):
            break

    if len(samples) == 0:
        missing_preview_text = json.dumps(missing_previews, indent=2)
        raise RuntimeError(
            "No comparable samples found. Check result roots, mode_filter, and whether inference artifacts were saved. "
            f"models={model_configs} "
            f"missing_preview={missing_preview_text}"
        )

    return {
        "models": model_configs,
        "samples": samples,
    }


def build_flow_path(sample: SamplePreset, mode: str, prefix: str, frame_index: int) -> Path:
    if sample["dataset_root_dir"] != "":
        return Path(sample["dataset_root_dir"]) / sample["record"] / mode / f"{prefix}_{frame_index}.exr"

    return Path(sample["sample_dir"]) / f"{prefix}_{frame_index}_fps30.exr"


def build_nearest_splat_hit_count(source_motion: torch.Tensor) -> torch.Tensor:
    batch_size = int(source_motion.shape[0])
    height = int(source_motion.shape[2])
    width = int(source_motion.shape[3])
    pixel_count = height * width
    source_grid = make_source_grid(batch_size, height, width, source_motion.device, source_motion.dtype)
    target_position = source_grid + source_motion
    target_x = target_position[:, 0].round().long()
    target_y = target_position[:, 1].round().long()
    valid = (target_x >= 0) & (target_x < width) & (target_y >= 0) & (target_y < height)
    flat_target_index = flatten_target_index(
        target_x=target_x.clamp(0, width - 1),
        target_y=target_y.clamp(0, height - 1),
        width=width,
    ).reshape(batch_size, -1)
    flat_valid = valid.reshape(batch_size, -1).to(dtype=source_motion.dtype)
    hit_count = torch.zeros((batch_size, pixel_count), device=source_motion.device, dtype=source_motion.dtype)
    hit_count.scatter_add_(dim=1, index=flat_target_index, src=flat_valid)
    return hit_count.reshape(batch_size, 1, height, width)


def tensor_mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    return mask[0, 0].detach().cpu().numpy().astype(bool)


def tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    return image[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy().astype(np.float32)


def build_sample_dataframe(sample: SamplePreset) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "img0": int(sample["frame_0"]),
                "img1": int(sample["frame_t"]),
                "img2": int(sample["frame_1"]),
                "record": sample["record"],
                "mode": sample["mode"],
                "fps": int(sample["fps"]),
                "valid": True,
            }
        ]
    )


def build_model_dataset(sample: SamplePreset, model_spec: InferenceModelSpec) -> VFITrainDataset | FlowEstimationTrainDataset:
    dataframe = build_sample_dataframe(sample)
    dataset_root_dir = sample["dataset_root_dir"]
    model_name = model_spec["model_name"]
    input_fps = int(model_spec["input_fps"])
    if model_name in (BASELINE_MODEL_NAME, RESIDUAL_MODEL_NAME):
        return VFITrainDataset(dataframe, dataset_root_dir, False, input_fps)

    if model_name == RESIDUAL_FLOW_APPROX_MODEL_NAME:
        include_source_depths = model_spec["flow_approx_method"] in SPLATTING_FLOW_APPROX_METHODS
        return FlowEstimationTrainDataset(dataframe, dataset_root_dir, input_fps, False, include_source_depths)

    raise ValueError(f"Unsupported model_name for direct comparison inference: {model_name}")


def run_model_inference_result(sample: SamplePreset, model_spec: InferenceModelSpec, device: torch.device) -> InferenceBatchResult:
    dataset = build_model_dataset(sample=sample, model_spec=model_spec)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    return run_inference_batch(
        batch=batch,
        device=device,
        flow_approx_method=model_spec["flow_approx_method"],
        model=model_spec["model"],
        model_name=model_spec["model_name"],
        scale_factor=float(model_spec["scale_factor"]),
    )


def run_model_prediction(sample: SamplePreset, model_spec: InferenceModelSpec, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    inference_result = run_model_inference_result(sample=sample, model_spec=model_spec, device=device)
    return tensor_image_to_numpy(inference_result["imgt"]), tensor_image_to_numpy(inference_result["imgt_pred"])


def run_model_prediction_maps(
    sample: SamplePreset,
    loaded_models: dict[str, InferenceModelSpec],
    model_configs: list[CompareModelConfig],
    device: torch.device,
) -> tuple[np.ndarray, PredictionMaps, InferenceResultMaps]:
    target: np.ndarray | None = None
    predictions: PredictionMaps = {}
    inference_results: InferenceResultMaps = {}
    for model_config in model_configs:
        model_key = model_config["key"]
        inference_result = run_model_inference_result(sample=sample, model_spec=loaded_models[model_key], device=device)
        model_target = tensor_image_to_numpy(inference_result["imgt"])
        model_prediction = tensor_image_to_numpy(inference_result["imgt_pred"])
        if target is None:
            target = model_target
        else:
            validate_same_shape(f"{model_key}_target", target, model_target)
            if not np.allclose(target, model_target, atol=1.0 / 255.0):
                raise ValueError(f"Ground-truth mismatch after model inference: sample_id={sample['sample_id']} model_key={model_key}")

        validate_same_shape(f"{model_key}_prediction", model_target, model_prediction)
        predictions[model_key] = model_prediction
        inference_results[model_key] = inference_result

    if target is None:
        raise ValueError(f"No model inference results were produced: sample_id={sample['sample_id']}")
    return target, predictions, inference_results


def build_region_maps(sample: SamplePreset, device: torch.device) -> RegionMaps:
    frame_0 = int(sample["frame_0"])
    frame_1 = int(sample["frame_1"])
    mode_30 = sample["mode"].replace("fps_60", "fps_30")
    bmv_30, source_depth1 = load_flow_and_depth_tensors(build_flow_path(sample, mode_30, "backwardVel_Depth", frame_1 // 2), device)
    fmv_30, source_depth0 = load_flow_and_depth_tensors(build_flow_path(sample, mode_30, "forwardVel_Depth", frame_0 // 2), device)
    embt = torch.tensor([[[0.5]]], dtype=torch.float32, device=device)
    flow_init = build_flow_init_result(fmv_30, bmv_30, embt, "splatting", source_depth0, source_depth1)
    if flow_init.masks is None:
        raise ValueError("Splatting flow init did not return coverage masks.")

    time_tensor = embt.reshape(embt.shape[0], 1, 1, 1)
    bmv_hit_count = build_nearest_splat_hit_count(time_tensor * fmv_30)
    fmv_hit_count = build_nearest_splat_hit_count((1 - time_tensor) * bmv_30)
    bmv_hit = flow_init.masks[:, 0:1] > 0
    fmv_hit = flow_init.masks[:, 1:2] > 0
    bmv_many_to_one = bmv_hit_count > 1
    fmv_many_to_one = fmv_hit_count > 1
    bmv_hole = ~bmv_hit
    fmv_hole = ~fmv_hit
    return {
        "all": np.ones_like(tensor_mask_to_numpy(bmv_hit), dtype=bool),
        "hit_both": tensor_mask_to_numpy(bmv_hit & fmv_hit),
        "clean_both_no_hole_no_multi": tensor_mask_to_numpy(~(bmv_hole | fmv_hole | bmv_many_to_one | fmv_many_to_one)),
        "hole_any": tensor_mask_to_numpy(bmv_hole | fmv_hole),
        "many_to_one_any": tensor_mask_to_numpy(bmv_many_to_one | fmv_many_to_one),
        "bmv_hole": tensor_mask_to_numpy(bmv_hole),
        "fmv_hole": tensor_mask_to_numpy(fmv_hole),
        "bmv_many_to_one": tensor_mask_to_numpy(bmv_many_to_one),
        "fmv_many_to_one": tensor_mask_to_numpy(fmv_many_to_one),
    }


def validate_same_shape(name: str, expected: np.ndarray, current: np.ndarray) -> None:
    if expected.shape != current.shape:
        raise ValueError(f"Shape mismatch for {name}: expected={expected.shape} actual={current.shape}")


def masked_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if mask.dtype != bool:
        raise TypeError(f"Mask must be bool: dtype={mask.dtype}")

    if int(mask.sum()) == 0:
        return float("nan")

    diff = prediction[mask] - target[mask]
    return float(np.mean(diff * diff))


def masked_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if mask.dtype != bool:
        raise TypeError(f"Mask must be bool: dtype={mask.dtype}")

    if int(mask.sum()) == 0:
        return float("nan")

    return float(np.mean(np.abs(prediction[mask] - target[mask])))


def mse_to_psnr(mse: float) -> float:
    if np.isnan(mse):
        return float("nan")

    if mse <= 0.0:
        return float("inf")

    return float(-10.0 * np.log10(mse))


def masked_ssim(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if mask.dtype != bool:
        raise TypeError(f"Mask must be bool: dtype={mask.dtype}")

    if int(mask.sum()) == 0:
        return float("nan")

    ssim_map = build_ssim_map(prediction=prediction, target=target)
    return masked_mean(ssim_map, mask)


def build_ssim_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    _score, ssim_map = structural_similarity(target, prediction, channel_axis=2, data_range=1.0, full=True)
    if ssim_map.ndim == 3:
        ssim_map = np.mean(ssim_map, axis=2)
    return ssim_map.astype(np.float32)


def masked_mean(value_map: np.ndarray, mask: np.ndarray) -> float:
    if mask.dtype != bool:
        raise TypeError(f"Mask must be bool: dtype={mask.dtype}")

    if int(mask.sum()) == 0:
        return float("nan")

    return float(np.mean(value_map[mask]))


def build_local_error_map(
    target: np.ndarray,
    prediction: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    error_map: np.ndarray = np.mean((prediction - target) ** 2, axis=2)
    return cv2.blur(error_map, (kernel_size, kernel_size))


def build_model_local_error_maps(
    target: np.ndarray,
    predictions: PredictionMaps,
    model_configs: list[CompareModelConfig],
    kernel_size: int,
) -> dict[str, np.ndarray]:
    return {
        model_config["key"]: build_local_error_map(
            target=target,
            prediction=predictions[model_config["key"]],
            kernel_size=kernel_size,
        )
        for model_config in model_configs
    }


def build_local_winner_maps_from_errors(
    local_error_maps: dict[str, np.ndarray],
    model_configs: list[CompareModelConfig],
    confidence_threshold: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    model_keys = [model_config["key"] for model_config in model_configs]
    error_stack = np.stack([local_error_maps[model_key] for model_key in model_keys], axis=0)
    sorted_error_stack = np.sort(error_stack, axis=0)
    best_error = sorted_error_stack[0]
    second_error = sorted_error_stack[1]
    confidence: np.ndarray = (second_error - best_error) / (second_error + best_error + 1e-12)
    confident = confidence >= confidence_threshold
    winner_index = np.argmin(error_stack, axis=0)
    neutral = ~confident
    winner_masks: dict[str, np.ndarray] = {}
    for model_index, model_key in enumerate(model_keys):
        winner_masks[model_key] = (winner_index == model_index) & confident

    local_rmse_delta: np.ndarray = np.sqrt(np.maximum(second_error, 0.0)) - np.sqrt(np.maximum(best_error, 0.0))
    return winner_masks, neutral, local_rmse_delta


def build_winner_alpha_map(
    local_rmse_delta: np.ndarray,
    winner_alpha_max: float,
    winner_alpha_full_rmse_delta: float,
) -> np.ndarray:
    full_alpha_delta = winner_alpha_full_rmse_delta / 255.0
    normalized_delta: np.ndarray = np.clip(local_rmse_delta / full_alpha_delta, 0.0, 1.0)
    return (normalized_delta * winner_alpha_max).astype(np.float32)


def blend_color_by_alpha(base_image: np.ndarray, color: np.ndarray, mask: np.ndarray, alpha_map: np.ndarray) -> np.ndarray:
    blended = base_image.copy()
    alpha_values = alpha_map[mask, None]
    blended[mask] = (1.0 - alpha_values) * blended[mask] + alpha_values * color
    return blended


def choose_sample_winner_from_predictions(
    target: np.ndarray,
    predictions: PredictionMaps,
    model_configs: list[CompareModelConfig],
) -> str:
    full_mask = np.ones(target.shape[:2], dtype=bool)
    model_scores = [
        (
            model_config["key"],
            mse_to_psnr(masked_mse(predictions[model_config["key"]], target, full_mask)),
        )
        for model_config in model_configs
    ]
    sorted_scores = sorted(model_scores, key=lambda item: item[1], reverse=True)
    if len(sorted_scores) > 1 and np.isclose(sorted_scores[0][1], sorted_scores[1][1], atol=1e-6):
        return "neutral"
    return f"{sorted_scores[0][0]}_win"


def build_model_region_rows(
    data_preset_name: str,
    sample: SamplePreset,
    model_configs: list[CompareModelConfig],
    region_name: str,
    region_mask: np.ndarray,
    target: np.ndarray,
    predictions: PredictionMaps,
    ssim_maps: dict[str, np.ndarray],
    winner_masks: dict[str, np.ndarray],
    neutral: np.ndarray,
    kernel_size: int,
    confidence_threshold: float,
    sample_winner: str,
    overlay_path: str,
) -> list[dict[str, object]]:
    total_pixels = int(target.shape[0] * target.shape[1])
    region_pixels = int(region_mask.sum())
    neutral_ratio = float("nan") if region_pixels == 0 else float(np.count_nonzero(neutral & region_mask) / region_pixels)
    rows: list[dict[str, object]] = []
    for model_config in model_configs:
        model_key = model_config["key"]
        model_prediction = predictions[model_key]
        model_mse = masked_mse(model_prediction, target, region_mask)
        local_winner_ratio = (
            float("nan")
            if region_pixels == 0
            else float(np.count_nonzero(winner_masks[model_key] & region_mask) / region_pixels)
        )
        rows.append(
            {
                "data_preset": data_preset_name,
                "sample_id": sample["sample_id"],
                "record": sample["record"] if sample["record"] != "" else "scratch",
                "case_record": build_case_record_name(sample),
                "frame_range": build_frame_range(int(sample["frame_0"]), int(sample["frame_1"])),
                "mode": sample["mode"],
                "sample_winner": sample_winner,
                "is_sample_winner": sample_winner == f"{model_key}_win",
                "overlay_path": overlay_path,
                "region": region_name,
                "kernel_size": kernel_size,
                "confidence_threshold": confidence_threshold,
                "model_key": model_key,
                "model_name": model_config["name"],
                "pixels": region_pixels,
                "region_ratio": float(region_pixels / total_pixels),
                "local_winner_ratio": local_winner_ratio,
                "neutral_ratio": neutral_ratio,
                "mse": model_mse,
                "psnr": mse_to_psnr(model_mse),
                "mae": masked_mae(model_prediction, target, region_mask),
                "ssim": masked_mean(ssim_maps[model_key], region_mask),
            }
        )

    return rows


def build_model_color_map(model_configs: list[CompareModelConfig]) -> dict[str, np.ndarray]:
    color_map: dict[str, np.ndarray] = {}
    for model_index, model_config in enumerate(model_configs):
        color = MODEL_COLORS[model_index % len(MODEL_COLORS)]
        color_map[model_config["key"]] = np.array(color, dtype=np.float32)
    return color_map


def draw_overlay_legend(
    overlay: np.ndarray,
    model_configs: list[CompareModelConfig],
    winner_ratios: dict[str, float],
    neutral_ratio: float,
    kernel_size: int,
    confidence_threshold: float,
    winner_alpha_max: float,
    winner_alpha_full_rmse_delta: float,
    neutral_alpha: float,
) -> np.ndarray:
    height, width, _channels = overlay.shape
    panel_width = 420
    canvas = np.ones((height, width + panel_width, 3), dtype=np.float32)
    canvas[:, :width] = overlay
    x0 = width + 28
    cv2.putText(canvas, "Local winner", (x0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0.05, 0.05, 0.05), 2, cv2.LINE_AA)
    white = np.ones(3, dtype=np.float32)
    model_color_map = build_model_color_map(model_configs)
    legend_items: list[tuple[np.ndarray, str]] = []
    for model_config in model_configs:
        model_key = model_config["key"]
        color = model_color_map[model_key]
        swatch_color = (1.0 - winner_alpha_max) * white + winner_alpha_max * color
        ratio = winner_ratios[model_key] * 100.0
        legend_items.append((swatch_color, f"{model_key}: {ratio:.1f}%"))
    neutral_swatch = (1.0 - neutral_alpha) * white + neutral_alpha * np.array(NEUTRAL_COLOR, dtype=np.float32)
    legend_items.append((neutral_swatch, f"neutral: {neutral_ratio * 100.0:.1f}%"))

    for index, (color, text) in enumerate(legend_items):
        y = 92 + index * 45
        color_tuple = tuple(float(channel) for channel in color)
        cv2.rectangle(canvas, (x0, y - 18), (x0 + 30, y + 12), color_tuple, -1)
        cv2.rectangle(canvas, (x0, y - 18), (x0 + 30, y + 12), (0.0, 0.0, 0.0), 1)
        cv2.putText(canvas, text, (x0 + 42, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0.05, 0.05, 0.05), 1, cv2.LINE_AA)

    text_start_y = 115 + len(legend_items) * 45
    text_lines = (
        f"kernel {kernel_size}x{kernel_size}",
        f"relative threshold {confidence_threshold:.2f}",
        f"full alpha gap {winner_alpha_full_rmse_delta:.1f}/255",
        f"winner alpha max {winner_alpha_max:.2f}",
        f"neutral alpha {neutral_alpha:.2f}",
    )
    for index, text in enumerate(text_lines):
        cv2.putText(canvas, text, (x0, text_start_y + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0.05, 0.05, 0.05), 1, cv2.LINE_AA)

    return canvas


def build_local_winner_overlay(
    target: np.ndarray,
    winner_masks: dict[str, np.ndarray],
    neutral: np.ndarray,
    local_rmse_delta: np.ndarray,
    model_configs: list[CompareModelConfig],
    kernel_size: int,
    confidence_threshold: float,
    winner_alpha_max: float,
    winner_alpha_full_rmse_delta: float,
    neutral_alpha: float,
) -> np.ndarray:
    overlay = target
    winner_alpha_map = build_winner_alpha_map(
        local_rmse_delta=local_rmse_delta,
        winner_alpha_max=winner_alpha_max,
        winner_alpha_full_rmse_delta=winner_alpha_full_rmse_delta,
    )
    model_color_map = build_model_color_map(model_configs)
    for model_config in model_configs:
        model_key = model_config["key"]
        overlay = blend_color_by_alpha(
            base_image=overlay,
            color=model_color_map[model_key],
            mask=winner_masks[model_key],
            alpha_map=winner_alpha_map,
        )
    neutral_alpha_map = np.full(neutral.shape, neutral_alpha, dtype=np.float32)
    overlay = blend_color_by_alpha(
        base_image=overlay,
        color=np.array(NEUTRAL_COLOR, dtype=np.float32),
        mask=neutral,
        alpha_map=neutral_alpha_map,
    )
    winner_ratios = {model_config["key"]: float(winner_masks[model_config["key"]].mean()) for model_config in model_configs}
    return draw_overlay_legend(
        overlay=overlay,
        model_configs=model_configs,
        winner_ratios=winner_ratios,
        neutral_ratio=float(neutral.mean()),
        kernel_size=kernel_size,
        confidence_threshold=confidence_threshold,
        winner_alpha_max=winner_alpha_max,
        winner_alpha_full_rmse_delta=winner_alpha_full_rmse_delta,
        neutral_alpha=neutral_alpha,
    )


def build_sample_output_dir(output_dir: Path, sample: SamplePreset, sample_winner: str) -> Path:
    record_name = sample["record"] if sample["record"] != "" else "scratch"
    return output_dir / record_name / sample_winner


def save_sample_overlay(
    sample: SamplePreset,
    target: np.ndarray,
    predictions: PredictionMaps,
    output_dir: Path,
    model_configs: list[CompareModelConfig],
    kernel_size: int,
    confidence_threshold: float,
    winner_alpha_max: float,
    winner_alpha_full_rmse_delta: float,
    neutral_alpha: float,
) -> str:
    overlay_path = output_dir / f"{sample['sample_id']}_win_overlay.png"
    return save_sample_overlay_to_path(
        target=target,
        predictions=predictions,
        overlay_path=overlay_path,
        model_configs=model_configs,
        kernel_size=kernel_size,
        confidence_threshold=confidence_threshold,
        winner_alpha_max=winner_alpha_max,
        winner_alpha_full_rmse_delta=winner_alpha_full_rmse_delta,
        neutral_alpha=neutral_alpha,
    )


def save_sample_overlay_to_path(
    target: np.ndarray,
    predictions: PredictionMaps,
    overlay_path: Path,
    model_configs: list[CompareModelConfig],
    kernel_size: int,
    confidence_threshold: float,
    winner_alpha_max: float,
    winner_alpha_full_rmse_delta: float,
    neutral_alpha: float,
) -> str:
    local_error_maps = build_model_local_error_maps(
        target=target,
        predictions=predictions,
        model_configs=model_configs,
        kernel_size=kernel_size,
    )
    winner_masks, neutral, local_rmse_delta = build_local_winner_maps_from_errors(
        local_error_maps=local_error_maps,
        model_configs=model_configs,
        confidence_threshold=confidence_threshold,
    )
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = build_local_winner_overlay(
        target=target,
        winner_masks=winner_masks,
        neutral=neutral,
        local_rmse_delta=local_rmse_delta,
        model_configs=model_configs,
        kernel_size=kernel_size,
        confidence_threshold=confidence_threshold,
        winner_alpha_max=winner_alpha_max,
        winner_alpha_full_rmse_delta=winner_alpha_full_rmse_delta,
        neutral_alpha=neutral_alpha,
    )
    save_rgb_image(overlay_path, overlay)
    return str(overlay_path)


def build_selected_case_output_dir(output_dir: Path, sample: SamplePreset, selected_case: dict[str, object]) -> Path:
    case_record = build_case_record_name(sample)
    frame_range = build_frame_range(int(sample["frame_0"]), int(sample["frame_1"]))
    return output_dir / case_record / str(selected_case["psnr_bucket"]) / str(selected_case["sample_winner"]) / frame_range


def build_selected_case_frame_output_dir(case_output_dir: Path, sample: SamplePreset) -> Path:
    frame_range = build_frame_range(int(sample["frame_0"]), int(sample["frame_1"]))
    return case_output_dir / frame_range


def build_selected_case_model_output_dir(frame_output_dir: Path, model_key: str) -> Path:
    validate_model_key(model_key)
    return frame_output_dir / model_key


def validate_copy_destination(source_dir: Path, destination_dir: Path) -> None:
    source_resolved = source_dir.resolve()
    destination_resolved = destination_dir.resolve()
    if destination_resolved == source_resolved or source_resolved in destination_resolved.parents:
        raise ValueError(f"Refusing to copy result artifacts into their own source tree: source={source_dir} destination={destination_dir}")


def copy_directory_contents(source_dir: Path, destination_dir: Path) -> None:
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Missing result artifact directory: source_dir={source_dir} destination_dir={destination_dir}")

    validate_copy_destination(source_dir=source_dir, destination_dir=destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source_dir.iterdir(), key=lambda path: path.name):
        destination_path = destination_dir / source_path.name
        if source_path.is_file():
            shutil.copy2(source_path, destination_path)
        elif source_path.is_dir():
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            raise ValueError(f"Unsupported result artifact path type: source_path={source_path}")


def copy_precomputed_case_model_outputs(
    sample: SamplePreset,
    frame_output_dir: Path,
    model_configs: list[CompareModelConfig],
) -> dict[str, str]:
    copied_dirs: dict[str, str] = {}
    model_result_dirs = sample["model_result_dirs"]
    for model_config in model_configs:
        model_key = model_config["key"]
        destination_dir = build_selected_case_model_output_dir(frame_output_dir=frame_output_dir, model_key=model_key)
        copy_directory_contents(source_dir=Path(model_result_dirs[model_key]), destination_dir=destination_dir)
        copied_dirs[model_key] = str(destination_dir)
    return copied_dirs


def save_inference_case_model_outputs(
    frame_output_dir: Path,
    inference_results: InferenceResultMaps,
    loaded_models: dict[str, InferenceModelSpec],
    model_configs: list[CompareModelConfig],
) -> dict[str, str]:
    saved_dirs: dict[str, str] = {}
    for model_config in model_configs:
        model_key = model_config["key"]
        model_output_dir = build_selected_case_model_output_dir(frame_output_dir=frame_output_dir, model_key=model_key)
        model_spec = loaded_models[model_key]
        save_selected_sample_artifacts(
            cv2=cv2,
            flow_diff_percentile=float(model_spec["flow_diff_percentile"]),
            flow_diff_threshold=float(model_spec["flow_diff_threshold"]),
            flow_to_image=flow_to_image,
            inference_result=inference_results[model_key],
            np=np,
            save_dir=model_output_dir,
            save_image=save_image,
        )
        saved_dirs[model_key] = str(model_output_dir)
    return saved_dirs


def analyze_prediction_maps(
    data_preset_name: str,
    sample: SamplePreset,
    model_configs: list[CompareModelConfig],
    output_dir: Path,
    device: torch.device,
    target: np.ndarray,
    predictions: PredictionMaps,
    kernel_size: int,
    confidence_threshold: float,
    winner_alpha_max: float,
    winner_alpha_full_rmse_delta: float,
    neutral_alpha: float,
    save_overlay_images: bool,
) -> list[dict[str, object]]:
    region_maps = build_region_maps(sample=sample, device=device)
    local_error_maps = build_model_local_error_maps(
        target=target,
        kernel_size=kernel_size,
        predictions=predictions,
        model_configs=model_configs,
    )
    winner_masks, neutral, local_rmse_delta = build_local_winner_maps_from_errors(
        local_error_maps=local_error_maps,
        model_configs=model_configs,
        confidence_threshold=confidence_threshold,
    )
    ssim_maps = {model_config["key"]: build_ssim_map(prediction=predictions[model_config["key"]], target=target) for model_config in model_configs}

    sample_winner = choose_sample_winner_from_predictions(target=target, predictions=predictions, model_configs=model_configs)
    sample_output_dir = build_sample_output_dir(output_dir=output_dir, sample=sample, sample_winner=sample_winner)
    overlay_path = ""
    if save_overlay_images:
        overlay_path = save_sample_overlay(
            sample=sample,
            target=target,
            predictions=predictions,
            output_dir=sample_output_dir,
            model_configs=model_configs,
            kernel_size=kernel_size,
            confidence_threshold=confidence_threshold,
            winner_alpha_max=winner_alpha_max,
            winner_alpha_full_rmse_delta=winner_alpha_full_rmse_delta,
            neutral_alpha=neutral_alpha,
        )

    rows: list[dict[str, object]] = []
    for region_name, region_mask in region_maps.items():
        rows.extend(
            build_model_region_rows(
                data_preset_name=data_preset_name,
                sample=sample,
                model_configs=model_configs,
                region_name=region_name,
                region_mask=region_mask,
                target=target,
                predictions=predictions,
                ssim_maps=ssim_maps,
                winner_masks=winner_masks,
                neutral=neutral,
                kernel_size=kernel_size,
                confidence_threshold=confidence_threshold,
                sample_winner=sample_winner,
                overlay_path=overlay_path,
            )
        )

    return rows


def analyze_sample(
    data_preset_name: str,
    sample: SamplePreset,
    model_configs: list[CompareModelConfig],
    output_dir: Path,
    device: torch.device,
    kernel_size: int,
    confidence_threshold: float,
    winner_alpha_max: float,
    winner_alpha_full_rmse_delta: float,
    neutral_alpha: float,
    save_overlay_images: bool,
) -> list[dict[str, object]]:
    target, predictions = read_model_prediction_images(sample=sample, model_configs=model_configs)
    return analyze_prediction_maps(
        data_preset_name=data_preset_name,
        sample=sample,
        model_configs=model_configs,
        output_dir=output_dir,
        device=device,
        target=target,
        predictions=predictions,
        kernel_size=kernel_size,
        confidence_threshold=confidence_threshold,
        winner_alpha_max=winner_alpha_max,
        winner_alpha_full_rmse_delta=winner_alpha_full_rmse_delta,
        neutral_alpha=neutral_alpha,
        save_overlay_images=save_overlay_images,
    )


def analyze_sample_with_models(
    data_preset_name: str,
    sample: SamplePreset,
    model_configs: list[CompareModelConfig],
    output_dir: Path,
    loaded_models: dict[str, InferenceModelSpec],
    device: torch.device,
    kernel_size: int,
    confidence_threshold: float,
    winner_alpha_max: float,
    winner_alpha_full_rmse_delta: float,
    neutral_alpha: float,
    save_overlay_images: bool,
) -> list[dict[str, object]]:
    target, predictions, _inference_results = run_model_prediction_maps(
        sample=sample,
        loaded_models=loaded_models,
        model_configs=model_configs,
        device=device,
    )
    return analyze_prediction_maps(
        data_preset_name=data_preset_name,
        sample=sample,
        model_configs=model_configs,
        output_dir=output_dir,
        device=device,
        target=target,
        predictions=predictions,
        kernel_size=kernel_size,
        confidence_threshold=confidence_threshold,
        winner_alpha_max=winner_alpha_max,
        winner_alpha_full_rmse_delta=winner_alpha_full_rmse_delta,
        neutral_alpha=neutral_alpha,
        save_overlay_images=save_overlay_images,
    )


def build_effective_config(
    args: argparse.Namespace,
    data_preset_name: str,
    output_dir: Path,
    model_configs: list[CompareModelConfig],
) -> dict[str, object]:
    return {
        "config": str(args.config) if args.config is not None else "",
        "data_preset": str(args.data_preset) if args.data_preset is not None else "",
        "dataset_preset": str(args.dataset_preset) if args.dataset_preset is not None else "",
        "root_dir": str(args.root_dir) if args.root_dir is not None else "",
        "dataset_root_dir": str(args.dataset_root_dir) if args.dataset_root_dir is not None else "",
        "models": model_configs,
        "only_fps": int(args.only_fps),
        "limit": int(args.limit),
        "mode_filter": str(args.mode_filter),
        "skip_missing_results": bool(args.skip_missing_results),
        "result_layout": str(args.result_layout),
        "sample_split": str(args.sample_split),
        "output_path": str(output_dir),
        "kernel_size": int(args.kernel_size),
        "confidence_threshold": float(args.confidence_threshold),
        "winner_alpha_max": float(args.winner_alpha_max),
        "winner_alpha_full_rmse_delta": float(args.winner_alpha_full_rmse_delta),
        "neutral_alpha": float(args.neutral_alpha),
        "save_overlay_images": bool(args.save_overlay_images),
        "save_selected_cases_only": bool(args.save_selected_cases_only),
        "selected_cases_per_bucket": int(args.selected_cases_per_bucket),
        "sequence_length": int(args.sequence_length),
        "resolved_data_preset_name": data_preset_name,
    }


def write_run_config(data_preset_name: str, data_preset: DataPreset, output_dir: Path, args: argparse.Namespace) -> None:
    effective_config = build_effective_config(
        args=args,
        data_preset_name=data_preset_name,
        output_dir=output_dir,
        model_configs=data_preset["models"],
    )
    run_config = {
        "config": effective_config,
        "data_preset": data_preset_name,
        "models": data_preset["models"],
        "kernel_size": int(args.kernel_size),
        "confidence_threshold": float(args.confidence_threshold),
        "winner_alpha_max": float(args.winner_alpha_max),
        "winner_alpha_full_rmse_delta": float(args.winner_alpha_full_rmse_delta),
        "neutral_alpha": float(args.neutral_alpha),
        "save_overlay_images": bool(args.save_overlay_images),
        "save_selected_cases_only": bool(args.save_selected_cases_only),
        "selected_cases_per_bucket": int(args.selected_cases_per_bucket),
        "sequence_length": int(args.sequence_length),
        "output_dir": str(output_dir),
        "samples": data_preset["samples"],
    }
    (output_dir / "analysis_config.json").write_text(json.dumps(effective_config, indent=2), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


def build_model_config_snapshot(
    inference_config_path_text: str | None,
    result_root_text: str | None,
) -> dict[str, object]:
    if inference_config_path_text is None:
        return {
            "inference_config_path": "",
            "result_root": "" if result_root_text is None else str(resolve_input_path(result_root_text)),
            "config": None,
        }

    inference_config_path = require_file(resolve_input_path(inference_config_path_text))
    inference_config = load_yaml_file(inference_config_path)
    if not isinstance(inference_config, dict):
        raise TypeError(
            f"Inference config must be a mapping: path={inference_config_path}, "
            f"type={type(inference_config).__name__}"
        )

    checkpoint_path = resolve_model_checkpoint_path(
        config=inference_config,
        config_path=inference_config_path,
    )
    return {
        "inference_config_path": str(inference_config_path),
        "resolved_checkpoint_path": str(checkpoint_path),
        "config": inference_config,
    }


def write_model_config_snapshots(output_dir: Path, model_configs: list[CompareModelConfig]) -> None:
    snapshots: dict[str, object] = {}
    for model_config in model_configs:
        model_key = model_config["key"]
        snapshot = build_model_config_snapshot(
            inference_config_path_text=model_config["inference_config"] if is_non_empty_text(model_config["inference_config"]) else None,
            result_root_text=model_config["result_root"] if is_non_empty_text(model_config["result_root"]) else None,
        )
        snapshot["name"] = model_config["name"]
        snapshot["epoch"] = model_config["epoch"]
        snapshots[model_key] = snapshot
        (output_dir / f"{model_key}_config.json").write_text(
            json.dumps(snapshot, indent=2),
            encoding="utf-8",
        )

    (output_dir / "model_configs.json").write_text(
        json.dumps(snapshots, indent=2),
        encoding="utf-8",
    )


def resolve_output_dir(args: argparse.Namespace, data_preset_name: str) -> Path:
    if args.output_path is None or str(args.output_path) == "":
        return OUTPUT_ROOT / data_preset_name

    return resolve_input_path(str(args.output_path))


def write_record_metrics(output_dir: Path, metrics: pd.DataFrame) -> None:
    for record_name, record_metrics in metrics.groupby("record", sort=False):
        record_dir = output_dir / str(record_name)
        record_dir.mkdir(parents=True, exist_ok=True)
        record_metrics.to_csv(record_dir / "region_winner_ratios.csv", index=False)
        record_summary = record_metrics[record_metrics["region"] == "all"].copy()
        record_summary.to_csv(record_dir / "summary.csv", index=False)


def build_case_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    all_region_metrics = metrics[metrics["region"] == "all"].copy()
    summary_rows: list[dict[str, object]] = []
    group_columns = ["data_preset", "sample_id", "record", "case_record", "frame_range", "mode", "sample_winner", "overlay_path"]
    for group_values, group in all_region_metrics.groupby(group_columns, sort=False):
        group_dict = dict(zip(group_columns, group_values))
        sorted_group = group.sort_values("psnr", ascending=False).reset_index(drop=True)
        best_row = sorted_group.iloc[0]
        second_psnr = float(sorted_group.iloc[1]["psnr"]) if len(sorted_group) > 1 else float("nan")
        best_psnr = float(best_row["psnr"])
        psnr_delta_abs = best_psnr - second_psnr if not np.isnan(second_psnr) else float("nan")
        summary_row = {
            **group_dict,
            "winner_model_key": str(best_row["model_key"]),
            "winner_model_name": str(best_row["model_name"]),
            "winner_psnr": best_psnr,
            "case_quality_psnr": float(group["psnr"].mean()),
            "psnr_delta_abs": psnr_delta_abs,
            "model_count": int(len(group)),
        }
        summary_rows.append(summary_row)

    return pd.DataFrame(summary_rows)


def select_cases_by_quality(case_summary: pd.DataFrame, selected_cases_per_bucket: int) -> pd.DataFrame:
    if selected_cases_per_bucket == 0:
        return case_summary.iloc[0:0].copy()

    selected_parts: list[pd.DataFrame] = []
    for (_case_record, sample_winner), winner_cases in case_summary.groupby(["case_record", "sample_winner"], sort=False):
        if sample_winner == "neutral":
            continue

        high_cases = winner_cases.nlargest(selected_cases_per_bucket, "case_quality_psnr").copy()
        high_cases["psnr_bucket"] = "high_psnr"
        low_cases = winner_cases.nsmallest(selected_cases_per_bucket, "case_quality_psnr").copy()
        low_cases["psnr_bucket"] = "low_psnr"
        selected_parts.extend((high_cases, low_cases))

    if len(selected_parts) == 0:
        return case_summary.iloc[0:0].copy()

    selected_cases = pd.concat(selected_parts, ignore_index=True)
    return selected_cases.drop_duplicates(subset=["sample_id", "sample_winner", "psnr_bucket"]).reset_index(drop=True)


def write_selected_case_metrics(output_dir: Path, selected_cases: pd.DataFrame) -> None:
    selected_cases.to_csv(output_dir / "selected_cases.csv", index=False)
    for record_name, record_cases in selected_cases.groupby("case_record", sort=False):
        record_dir = output_dir / str(record_name)
        record_dir.mkdir(parents=True, exist_ok=True)
        record_cases.to_csv(record_dir / "selected_cases.csv", index=False)


def build_sequence_lookup_key(sample: SamplePreset) -> tuple[str, str, int]:
    return sample["record"], sample["mode"], int(sample["frame_0"])


def build_sequence_sample_lookup(samples: list[SamplePreset]) -> dict[tuple[str, str, int], SamplePreset]:
    return {build_sequence_lookup_key(sample): sample for sample in samples}


def collect_sequence_samples(
    sample: SamplePreset,
    sample_lookup: dict[tuple[str, str, int], SamplePreset],
    sequence_length: int,
) -> list[SamplePreset]:
    frame_step = int(sample["frame_1"]) - int(sample["frame_0"])
    if frame_step <= 0:
        raise ValueError(f"Frame step must be positive: sample_id={sample['sample_id']} frame_step={frame_step}")

    sequence_samples: list[SamplePreset] = []
    for offset in range(-sequence_length, sequence_length + 1):
        frame_0 = int(sample["frame_0"]) + offset * frame_step
        lookup_key = (sample["record"], sample["mode"], frame_0)
        if lookup_key not in sample_lookup:
            missing_range = build_frame_range(frame_0, frame_0 + frame_step)
            raise KeyError(
                "Missing sequence sample for selected case: "
                f"sample_id={sample['sample_id']} record={sample['record']} mode={sample['mode']} frame_range={missing_range}"
            )
        sequence_samples.append(sample_lookup[lookup_key])

    return sequence_samples


def write_selected_case_region_metrics(case_output_dir: Path, selected_case: dict[str, object], metrics: pd.DataFrame) -> str:
    sample_id = str(selected_case["sample_id"])
    sample_metrics = metrics[metrics["sample_id"] == sample_id].copy()
    if len(sample_metrics) == 0:
        raise ValueError(f"No region metrics found for selected case: sample_id={sample_id}")

    output_path = case_output_dir / "region_metrics.csv"
    sample_metrics.to_csv(output_path, index=False)
    return str(output_path)


def copy_precomputed_sequence_outputs(
    sequence_samples: list[SamplePreset],
    case_output_dir: Path,
    model_configs: list[CompareModelConfig],
) -> list[dict[str, object]]:
    saved_frames: list[dict[str, object]] = []
    for sequence_sample in sequence_samples:
        frame_output_dir = build_selected_case_frame_output_dir(case_output_dir=case_output_dir, sample=sequence_sample)
        model_output_dirs = copy_precomputed_case_model_outputs(
            sample=sequence_sample,
            frame_output_dir=frame_output_dir,
            model_configs=model_configs,
        )
        saved_frames.append(
            {
                "frame_range": build_frame_range(int(sequence_sample["frame_0"]), int(sequence_sample["frame_1"])),
                "model_output_dirs": model_output_dirs,
            }
        )

    return saved_frames


def save_inference_sequence_outputs(
    sequence_samples: list[SamplePreset],
    case_output_dir: Path,
    loaded_models: dict[str, InferenceModelSpec],
    model_configs: list[CompareModelConfig],
    device: torch.device,
) -> list[dict[str, object]]:
    saved_frames: list[dict[str, object]] = []
    for sequence_sample in sequence_samples:
        _target, _predictions, inference_results = run_model_prediction_maps(
            sample=sequence_sample,
            loaded_models=loaded_models,
            model_configs=model_configs,
            device=device,
        )
        frame_output_dir = build_selected_case_frame_output_dir(case_output_dir=case_output_dir, sample=sequence_sample)
        model_output_dirs = save_inference_case_model_outputs(
            frame_output_dir=frame_output_dir,
            inference_results=inference_results,
            loaded_models=loaded_models,
            model_configs=model_configs,
        )
        saved_frames.append(
            {
                "frame_range": build_frame_range(int(sequence_sample["frame_0"]), int(sequence_sample["frame_1"])),
                "model_output_dirs": model_output_dirs,
            }
        )

    return saved_frames


def save_selected_case_overlays(
    data_preset: DataPreset,
    selected_cases: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    model_configs: list[CompareModelConfig],
    kernel_size: int,
    confidence_threshold: float,
    winner_alpha_max: float,
    winner_alpha_full_rmse_delta: float,
    neutral_alpha: float,
    sequence_length: int,
) -> pd.DataFrame:
    sample_by_id = {sample["sample_id"]: sample for sample in data_preset["samples"]}
    sample_lookup = build_sequence_sample_lookup(data_preset["samples"])
    saved_cases = selected_cases.copy()
    overlay_paths: list[str] = []
    selected_case_dirs: list[str] = []
    region_metric_paths: list[str] = []
    sequence_outputs: list[str] = []
    selected_records = saved_cases.to_dict("records")
    for selected_case in tqdm(selected_records, desc="save_selected_overlays", leave=True):
        sample_id = str(selected_case["sample_id"])
        sample = sample_by_id[sample_id]
        target, predictions = read_model_prediction_images(sample=sample, model_configs=model_configs)
        case_output_dir = build_selected_case_output_dir(output_dir=output_dir, sample=sample, selected_case=selected_case)
        sequence_samples = collect_sequence_samples(sample=sample, sample_lookup=sample_lookup, sequence_length=sequence_length)
        overlay_path = save_sample_overlay_to_path(
            target=target,
            predictions=predictions,
            overlay_path=case_output_dir / "win_overlay.png",
            model_configs=model_configs,
            kernel_size=kernel_size,
            confidence_threshold=confidence_threshold,
            winner_alpha_max=winner_alpha_max,
            winner_alpha_full_rmse_delta=winner_alpha_full_rmse_delta,
            neutral_alpha=neutral_alpha,
        )
        region_metric_path = write_selected_case_region_metrics(case_output_dir=case_output_dir, selected_case=selected_case, metrics=metrics)
        saved_sequence_outputs = copy_precomputed_sequence_outputs(
            sequence_samples=sequence_samples,
            case_output_dir=case_output_dir,
            model_configs=model_configs,
        )
        overlay_paths.append(overlay_path)
        selected_case_dirs.append(str(case_output_dir))
        region_metric_paths.append(region_metric_path)
        sequence_outputs.append(json.dumps(saved_sequence_outputs))

    saved_cases["selected_overlay_path"] = overlay_paths
    saved_cases["selected_case_dir"] = selected_case_dirs
    saved_cases["selected_region_metrics_path"] = region_metric_paths
    saved_cases["selected_sequence_outputs"] = sequence_outputs
    return saved_cases


def save_selected_case_model_overlays(
    data_preset: DataPreset,
    selected_cases: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    model_configs: list[CompareModelConfig],
    loaded_models: dict[str, InferenceModelSpec],
    device: torch.device,
    kernel_size: int,
    confidence_threshold: float,
    winner_alpha_max: float,
    winner_alpha_full_rmse_delta: float,
    neutral_alpha: float,
    sequence_length: int,
) -> pd.DataFrame:
    sample_by_id = {sample["sample_id"]: sample for sample in data_preset["samples"]}
    sample_lookup = build_sequence_sample_lookup(data_preset["samples"])
    saved_cases = selected_cases.copy()
    overlay_paths: list[str] = []
    selected_case_dirs: list[str] = []
    region_metric_paths: list[str] = []
    sequence_outputs: list[str] = []
    selected_records = saved_cases.to_dict("records")
    for selected_case in tqdm(selected_records, desc="save_selected_model_overlays", leave=True):
        sample_id = str(selected_case["sample_id"])
        sample = sample_by_id[sample_id]
        target, predictions, _inference_results = run_model_prediction_maps(
            sample=sample,
            loaded_models=loaded_models,
            model_configs=model_configs,
            device=device,
        )
        case_output_dir = build_selected_case_output_dir(output_dir=output_dir, sample=sample, selected_case=selected_case)
        sequence_samples = collect_sequence_samples(sample=sample, sample_lookup=sample_lookup, sequence_length=sequence_length)
        overlay_path = save_sample_overlay_to_path(
            target=target,
            predictions=predictions,
            overlay_path=case_output_dir / "win_overlay.png",
            model_configs=model_configs,
            kernel_size=kernel_size,
            confidence_threshold=confidence_threshold,
            winner_alpha_max=winner_alpha_max,
            winner_alpha_full_rmse_delta=winner_alpha_full_rmse_delta,
            neutral_alpha=neutral_alpha,
        )
        region_metric_path = write_selected_case_region_metrics(case_output_dir=case_output_dir, selected_case=selected_case, metrics=metrics)
        saved_sequence_outputs = save_inference_sequence_outputs(
            sequence_samples=sequence_samples,
            case_output_dir=case_output_dir,
            loaded_models=loaded_models,
            model_configs=model_configs,
            device=device,
        )
        overlay_paths.append(overlay_path)
        selected_case_dirs.append(str(case_output_dir))
        region_metric_paths.append(region_metric_path)
        sequence_outputs.append(json.dumps(saved_sequence_outputs))

    saved_cases["selected_overlay_path"] = overlay_paths
    saved_cases["selected_case_dir"] = selected_case_dirs
    saved_cases["selected_region_metrics_path"] = region_metric_paths
    saved_cases["selected_sequence_outputs"] = sequence_outputs
    return saved_cases


def run_analysis(data_preset_name: str, data_preset: DataPreset, args: argparse.Namespace) -> Path:
    model_configs = data_preset["models"]
    validate_compare_model_config_modes(model_configs)
    output_dir = resolve_output_dir(args=args, data_preset_name=data_preset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(data_preset_name=data_preset_name, data_preset=data_preset, output_dir=output_dir, args=args)
    write_model_config_snapshots(output_dir=output_dir, model_configs=model_configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    save_all_overlays = bool(args.save_overlay_images) and not bool(args.save_selected_cases_only)
    loaded_models: dict[str, InferenceModelSpec] = {}
    use_direct_inference = model_configs_use_direct_inference(model_configs)
    if use_direct_inference:
        loaded_models = load_inference_models(model_configs=model_configs, device=device)

    with torch.no_grad():
        for sample in tqdm(data_preset["samples"], desc=f"compare_{data_preset_name}", leave=True):
            if use_direct_inference:
                rows.extend(
                    analyze_sample_with_models(
                        data_preset_name=data_preset_name,
                        sample=sample,
                        model_configs=model_configs,
                        output_dir=output_dir,
                        loaded_models=loaded_models,
                        device=device,
                        kernel_size=int(args.kernel_size),
                        confidence_threshold=float(args.confidence_threshold),
                        winner_alpha_max=float(args.winner_alpha_max),
                        winner_alpha_full_rmse_delta=float(args.winner_alpha_full_rmse_delta),
                        neutral_alpha=float(args.neutral_alpha),
                        save_overlay_images=save_all_overlays,
                    )
                )
            else:
                rows.extend(
                    analyze_sample(
                        data_preset_name=data_preset_name,
                        sample=sample,
                        model_configs=model_configs,
                        output_dir=output_dir,
                        device=device,
                        kernel_size=int(args.kernel_size),
                        confidence_threshold=float(args.confidence_threshold),
                        winner_alpha_max=float(args.winner_alpha_max),
                        winner_alpha_full_rmse_delta=float(args.winner_alpha_full_rmse_delta),
                        neutral_alpha=float(args.neutral_alpha),
                        save_overlay_images=save_all_overlays,
                    )
                )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "region_winner_ratios.csv", index=False)
    summary = metrics[metrics["region"] == "all"].copy()
    summary.to_csv(output_dir / "summary.csv", index=False)
    write_record_metrics(output_dir=output_dir, metrics=metrics)
    case_summary = build_case_summary(metrics)
    selected_cases = select_cases_by_quality(case_summary=case_summary, selected_cases_per_bucket=int(args.selected_cases_per_bucket))
    if bool(args.save_overlay_images) and bool(args.save_selected_cases_only) and len(selected_cases) > 0:
        with torch.no_grad():
            if use_direct_inference:
                selected_cases = save_selected_case_model_overlays(
                    data_preset=data_preset,
                    selected_cases=selected_cases,
                    metrics=metrics,
                    output_dir=output_dir,
                    model_configs=model_configs,
                    loaded_models=loaded_models,
                    device=device,
                    kernel_size=int(args.kernel_size),
                    confidence_threshold=float(args.confidence_threshold),
                    winner_alpha_max=float(args.winner_alpha_max),
                    winner_alpha_full_rmse_delta=float(args.winner_alpha_full_rmse_delta),
                    neutral_alpha=float(args.neutral_alpha),
                    sequence_length=int(args.sequence_length),
                )
            else:
                selected_cases = save_selected_case_overlays(
                    data_preset=data_preset,
                    selected_cases=selected_cases,
                    metrics=metrics,
                    output_dir=output_dir,
                    model_configs=model_configs,
                    kernel_size=int(args.kernel_size),
                    confidence_threshold=float(args.confidence_threshold),
                    winner_alpha_max=float(args.winner_alpha_max),
                    winner_alpha_full_rmse_delta=float(args.winner_alpha_full_rmse_delta),
                    neutral_alpha=float(args.neutral_alpha),
                    sequence_length=int(args.sequence_length),
                )
    write_selected_case_metrics(output_dir=output_dir, selected_cases=selected_cases)
    return output_dir


def main() -> None:
    args = parse_args()
    if args.data_preset is not None:
        data_preset_name = str(args.data_preset)
        raw_data_preset = DATA_PRESETS[data_preset_name]
        model_configs = (
            parse_compare_model_configs(args.models)
            if isinstance(args.models, list) and len(args.models) > 0
            else raw_data_preset["models"]
        )
        validate_compare_model_config_modes(model_configs)
        data_preset = normalize_data_preset(DATA_PRESETS[data_preset_name], model_configs)
    else:
        model_configs = build_compare_model_configs(args)
        validate_compare_model_config_modes(model_configs)
        data_preset_name = str(args.dataset_preset)
        data_preset = build_samples_from_dataset_preset(args, model_configs)

    output_dir = run_analysis(data_preset_name=data_preset_name, data_preset=data_preset, args=args)
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
