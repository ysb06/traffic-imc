from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from traffic_imc_dataset.utils import PathConfig

from ..models.mlcaformer.datamodule import MLCAFormerDataModule
from ..models.mlcaformer.module import MLCAFormerLightningModule
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
    in_steps: int = 24
    out_steps: int = 1
    steps_per_day: int = 24
    batch_size: int = 4
    num_workers: int = 0
    shuffle_training: bool = True
    scale_method: str = "normal"


@dataclass
class ModelConfig:
    input_dim: int = 3
    output_dim: int = 1
    input_embedding_dim: int = 24
    tod_embedding_dim: int = 24
    dow_embedding_dim: int = 24
    nid_embedding_dim: int = 24
    col_embedding_dim: int = 80
    feed_forward_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 5


@dataclass
class TrainerConfig:
    max_epochs: int = 100
    gradient_clip_val: float = 1.0


@dataclass
class MLCAFormerTrainingConfig:
    path_config_file: str
    run: RunConfig = field(default_factory=RunConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


def _load_training_config(path: Path) -> MLCAFormerTrainingConfig:
    raw = read_yaml(path)

    return MLCAFormerTrainingConfig(
        path_config_file=read_path_config_file(raw),
        run=RunConfig(**read_section(raw, "run")),
        datamodule=DataModuleConfig(**read_section(raw, "datamodule")),
        model=ModelConfig(**read_section(raw, "model")),
        early_stopping=EarlyStoppingConfig(**read_section(raw, "early_stopping")),
        trainer=TrainerConfig(**read_section(raw, "trainer")),
    )


def _build_datamodule(
    cfg: MLCAFormerTrainingConfig,
    path_config: PathConfig,
) -> MLCAFormerDataModule:
    return MLCAFormerDataModule(
        training_dataset_path=path_config.metr_imc_training_path,
        test_dataset_path=path_config.metr_imc_test_path,
        test_missing_path=path_config.metr_imc_test_missing_path,
        train_val_split=cfg.datamodule.train_val_split,
        in_steps=cfg.datamodule.in_steps,
        out_steps=cfg.datamodule.out_steps,
        steps_per_day=cfg.datamodule.steps_per_day,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
        shuffle_training=cfg.datamodule.shuffle_training,
        scale_method=cfg.datamodule.scale_method,
    )


def _build_model(
    cfg: MLCAFormerTrainingConfig,
    data: MLCAFormerDataModule,
    path_config: PathConfig,
) -> MLCAFormerLightningModule:
    del path_config

    if data.num_nodes is None:
        raise ValueError("num_nodes is not initialized in the data module.")

    return MLCAFormerLightningModule(
        num_nodes=data.num_nodes,
        in_steps=cfg.datamodule.in_steps,
        out_steps=cfg.datamodule.out_steps,
        steps_per_day=cfg.datamodule.steps_per_day,
        input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim,
        input_embedding_dim=cfg.model.input_embedding_dim,
        tod_embedding_dim=cfg.model.tod_embedding_dim,
        dow_embedding_dim=cfg.model.dow_embedding_dim,
        nid_embedding_dim=cfg.model.nid_embedding_dim,
        col_embedding_dim=cfg.model.col_embedding_dim,
        feed_forward_dim=cfg.model.feed_forward_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        learning_rate=cfg.model.learning_rate,
        scheduler_factor=cfg.model.scheduler_factor,
        scheduler_patience=cfg.model.scheduler_patience,
        scaler=data.scaler,
    )


def _get_run_metadata(cfg: MLCAFormerTrainingConfig) -> RunMetadata:
    return RunMetadata(
        name_key=cfg.run.name_key,
        code=cfg.run.code,
        seed=cfg.run.seed,
    )


def _get_early_stopping_metadata(
    cfg: MLCAFormerTrainingConfig,
) -> EarlyStoppingMetadata:
    return EarlyStoppingMetadata(
        enabled=cfg.early_stopping.enabled,
        patience=cfg.early_stopping.patience,
    )


def _get_trainer_metadata(cfg: MLCAFormerTrainingConfig) -> TrainerMetadata:
    return TrainerMetadata(
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )


def get_adapter() -> ModelAdapter:
    return ModelAdapter(
        key="mlcaformer",
        wandb_prefix="MLCAFormer",
        output_subdir="mlcaformer",
        checkpoint_filename="mlcaformer-best-{epoch:02d}-{val_loss:.4f}",
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
