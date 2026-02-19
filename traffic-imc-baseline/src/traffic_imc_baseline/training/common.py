from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml


@dataclass(frozen=True)
class RunMetadata:
    name_key: str
    code: int
    seed: Optional[int]


@dataclass(frozen=True)
class EarlyStoppingMetadata:
    enabled: bool
    patience: int


@dataclass(frozen=True)
class TrainerMetadata:
    max_epochs: int
    gradient_clip_val: float


@dataclass(frozen=True)
class ModelAdapter:
    key: str
    wandb_prefix: str
    output_subdir: str
    checkpoint_filename: str
    load_config: Callable[[Path], Any]
    build_datamodule: Callable[[Any, Any], Any]
    build_model: Callable[[Any, Any, Any], Any]
    get_run_metadata: Callable[[Any], RunMetadata]
    get_early_stopping_metadata: Callable[[Any], EarlyStoppingMetadata]
    get_trainer_metadata: Callable[[Any], TrainerMetadata]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format: {path}")
    return config


def read_section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' section must be a mapping")
    return value


def read_path_config_file(raw: dict[str, Any]) -> str:
    if "path_config_file" not in raw:
        raise ValueError("Missing required key: 'path_config_file'")

    path_config_file = raw["path_config_file"]
    if not isinstance(path_config_file, str):
        raise ValueError("'path_config_file' must be a string")

    return path_config_file


def resolve_path(base_dir: Path, target: str) -> Path:
    target_path = Path(target)
    if target_path.is_absolute():
        return target_path
    return (base_dir / target_path).resolve()


def resolve_seed(cli_seed: Optional[int], config_seed: Optional[int]) -> Optional[int]:
    if cli_seed is not None:
        return cli_seed
    return config_seed


def build_output_dir(output_subdir: str, name_key: str, code: int) -> str:
    return f"./output/{output_subdir}/{name_key}_{code:02d}"
