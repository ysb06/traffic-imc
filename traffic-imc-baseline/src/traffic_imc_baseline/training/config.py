from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

SupportedModel = Literal[
    "agcrn",
    "bigst",
    "dcrnn",
    "gwnet",
    "lstm",
    "mlcaformer",
    "mtgnn",
    "stgcn",
    "stid",
]
LoggerType = Literal["wandb", "csv", "tensorboard", "none"]
MonitorMode = Literal["min", "max"]
AcceleratorType = Literal["auto", "cpu", "gpu", "mps"]
Float32MatmulPrecision = Literal["highest", "high", "medium"]
DeviceSpec = int | str | list[int]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictBaseModel):
    name: str = "default"
    code: int = 0
    seed: Optional[int] = None


class PathsConfig(StrictBaseModel):
    path_config_file: str

    def resolve_path_config_file(self, config_file: Path) -> Path:
        path = Path(self.path_config_file)
        if path.is_absolute():
            return path
        return (config_file.parent / path).resolve()


class TrainerConfig(StrictBaseModel):
    max_epochs: int = 20
    gradient_clip_val: float = 1.0
    resume_ckpt_path: Optional[str] = None


class RuntimeConfig(StrictBaseModel):
    accelerator: Optional[AcceleratorType] = None
    devices: Optional[DeviceSpec] = None
    cpu_num_threads: Optional[int] = Field(default=None, ge=1)
    cpu_num_interop_threads: Optional[int] = Field(default=None, ge=1)
    float32_matmul_precision: Optional[Float32MatmulPrecision] = None


class EarlyStoppingConfig(StrictBaseModel):
    enabled: bool = True
    monitor: str = "val_loss"
    mode: MonitorMode = "min"
    patience: int = 10
    verbose: bool = True


class CheckpointConfig(StrictBaseModel):
    enabled: bool = True
    monitor: str = "val_loss"
    mode: MonitorMode = "min"
    save_top_k: int = 1
    save_last: bool = True
    filename: Optional[str] = None


class LearningRateMonitorConfig(StrictBaseModel):
    enabled: bool = True
    logging_interval: Literal["step", "epoch"] = "step"


class CallbacksConfig(StrictBaseModel):
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    lr_monitor: LearningRateMonitorConfig = Field(
        default_factory=LearningRateMonitorConfig
    )


class LoggingConfig(StrictBaseModel):
    type: LoggerType = "wandb"
    project: str = "Traffic-IMC"
    save_dir: str = "./logs"
    log_model: bool | Literal["all"] = "all"


class ModelConfig(StrictBaseModel):
    name: SupportedModel
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value


class DataConfig(StrictBaseModel):
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(StrictBaseModel):
    paths: PathsConfig
    model: ModelConfig
    data: DataConfig = Field(default_factory=DataConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def resolved_path_config_file(self, config_file: Path) -> Path:
        return self.paths.resolve_path_config_file(config_file)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config format: {path}")

    return raw


def load_training_config(path: Path) -> TrainingConfig:
    raw = read_yaml(path)
    return TrainingConfig.model_validate(raw)
