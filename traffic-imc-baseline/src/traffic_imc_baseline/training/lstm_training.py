from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from traffic_imc_dataset.utils import PathConfig

from ..models.lstm.datamodule import LSTMDataModule
from ..models.lstm.module import LSTMLightningModule
from .common import (
    EarlyStoppingMetadata,
    ModelAdapter,
    RunMetadata,
    TrainerMetadata,
    read_path_config_file,
    read_section,
    read_yaml,
)


@dataclass
class RunConfig:
    name_key: str = "default"
    code: int = 0
    seed: Optional[int] = None


@dataclass
class DataModuleConfig:
    train_val_split: float = 0.8
    seq_length: int = 24
    batch_size: int = 512
    num_workers: int = 0
    shuffle_training: bool = True


@dataclass
class ModelConfig:
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    output_size: int = 1
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 10


@dataclass
class TrainerConfig:
    max_epochs: int = 20
    gradient_clip_val: float = 1.0


@dataclass
class LSTMTrainingConfig:
    path_config_file: str
    run: RunConfig = field(default_factory=RunConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


def _load_training_config(path: Path) -> LSTMTrainingConfig:
    raw = read_yaml(path)

    return LSTMTrainingConfig(
        path_config_file=read_path_config_file(raw),
        run=RunConfig(**read_section(raw, "run")),
        datamodule=DataModuleConfig(**read_section(raw, "datamodule")),
        model=ModelConfig(**read_section(raw, "model")),
        early_stopping=EarlyStoppingConfig(**read_section(raw, "early_stopping")),
        trainer=TrainerConfig(**read_section(raw, "trainer")),
    )


def _build_datamodule(cfg: LSTMTrainingConfig, path_config: PathConfig) -> LSTMDataModule:
    return LSTMDataModule(
        training_dataset_path=path_config.metr_imc_training_path,
        test_dataset_path=path_config.metr_imc_test_path,
        test_missing_path=path_config.metr_imc_test_missing_path,
        train_val_split=cfg.datamodule.train_val_split,
        seq_length=cfg.datamodule.seq_length,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
        shuffle_training=cfg.datamodule.shuffle_training,
    )


def _build_model(
    cfg: LSTMTrainingConfig,
    data: LSTMDataModule,
    path_config: PathConfig,
) -> LSTMLightningModule:
    del path_config

    if data.scaler is None:
        raise ValueError("Scaler is not fitted in the data module.")

    return LSTMLightningModule(
        scaler=data.scaler,
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        output_size=cfg.model.output_size,
        learning_rate=cfg.model.learning_rate,
        dropout_rate=cfg.model.dropout_rate,
        scheduler_factor=cfg.model.scheduler_factor,
        scheduler_patience=cfg.model.scheduler_patience,
    )


def _get_run_metadata(cfg: LSTMTrainingConfig) -> RunMetadata:
    return RunMetadata(
        name_key=cfg.run.name_key,
        code=cfg.run.code,
        seed=cfg.run.seed,
    )


def _get_early_stopping_metadata(cfg: LSTMTrainingConfig) -> EarlyStoppingMetadata:
    return EarlyStoppingMetadata(
        enabled=cfg.early_stopping.enabled,
        patience=cfg.early_stopping.patience,
    )


def _get_trainer_metadata(cfg: LSTMTrainingConfig) -> TrainerMetadata:
    return TrainerMetadata(
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )


def get_adapter() -> ModelAdapter:
    return ModelAdapter(
        key="lstm",
        wandb_prefix="LSTM",
        output_subdir="lstm",
        checkpoint_filename="best-{epoch:02d}-{val_loss:.4f}",
        load_config=_load_training_config,
        build_datamodule=_build_datamodule,
        build_model=_build_model,
        get_run_metadata=_get_run_metadata,
        get_early_stopping_metadata=_get_early_stopping_metadata,
        get_trainer_metadata=_get_trainer_metadata,
    )


def main(config_path: str, seed: Optional[int] = None) -> None:
    from .runner import run_training

    run_training(adapter=get_adapter(), config_path=config_path, seed=seed)
