from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from traffic_imc_dataset.utils import PathConfig

from ..models.agcrn.datamodule import AGCRNDataModule
from ..models.agcrn.module import AGCRNLightningModule
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
    batch_size: int = 64
    num_workers: int = 0
    shuffle_training: bool = True
    scale_method: str = "normal"
    target_sensors: Optional[list[str]] = None


@dataclass
class ModelConfig:
    input_dim: int = 1
    output_dim: int = 1
    rnn_units: int = 64
    num_layers: int = 2
    embed_dim: int = 10
    cheb_k: int = 2
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.7
    loss_func: str = "mse"


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 15


@dataclass
class TrainerConfig:
    max_epochs: int = 100
    gradient_clip_val: float = 5.0


@dataclass
class AGCRNTrainingConfig:
    path_config_file: str
    run: RunConfig = field(default_factory=RunConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


def _load_training_config(path: Path) -> AGCRNTrainingConfig:
    raw = read_yaml(path)

    return AGCRNTrainingConfig(
        path_config_file=read_path_config_file(raw),
        run=RunConfig(**read_section(raw, "run")),
        datamodule=DataModuleConfig(**read_section(raw, "datamodule")),
        model=ModelConfig(**read_section(raw, "model")),
        early_stopping=EarlyStoppingConfig(**read_section(raw, "early_stopping")),
        trainer=TrainerConfig(**read_section(raw, "trainer")),
    )


def _build_datamodule(cfg: AGCRNTrainingConfig, path_config: PathConfig) -> AGCRNDataModule:
    return AGCRNDataModule(
        training_dataset_path=path_config.metr_imc_training_path,
        test_dataset_path=path_config.metr_imc_test_path,
        test_missing_path=path_config.metr_imc_test_missing_path,
        train_val_split=cfg.datamodule.train_val_split,
        in_steps=cfg.datamodule.in_steps,
        out_steps=cfg.datamodule.out_steps,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
        shuffle_training=cfg.datamodule.shuffle_training,
        target_sensors=cfg.datamodule.target_sensors,
        scale_method=cfg.datamodule.scale_method,
    )


def _build_model(
    cfg: AGCRNTrainingConfig,
    data: AGCRNDataModule,
    path_config: PathConfig,
) -> AGCRNLightningModule:
    del path_config

    if data.num_nodes is None:
        raise ValueError("num_nodes is not initialized in the data module.")

    return AGCRNLightningModule(
        num_nodes=data.num_nodes,
        in_steps=cfg.datamodule.in_steps,
        out_steps=cfg.datamodule.out_steps,
        input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim,
        rnn_units=cfg.model.rnn_units,
        num_layers=cfg.model.num_layers,
        embed_dim=cfg.model.embed_dim,
        cheb_k=cfg.model.cheb_k,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        scheduler_step_size=cfg.model.scheduler_step_size,
        scheduler_gamma=cfg.model.scheduler_gamma,
        loss_func=cfg.model.loss_func,
        scaler=data.scaler,
    )


def _get_run_metadata(cfg: AGCRNTrainingConfig) -> RunMetadata:
    return RunMetadata(
        name_key=cfg.run.name_key,
        code=cfg.run.code,
        seed=cfg.run.seed,
    )


def _get_early_stopping_metadata(
    cfg: AGCRNTrainingConfig,
) -> EarlyStoppingMetadata:
    return EarlyStoppingMetadata(
        enabled=cfg.early_stopping.enabled,
        patience=cfg.early_stopping.patience,
    )


def _get_trainer_metadata(cfg: AGCRNTrainingConfig) -> TrainerMetadata:
    return TrainerMetadata(
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )


def get_adapter() -> ModelAdapter:
    return ModelAdapter(
        key="agcrn",
        wandb_prefix="AGCRN",
        output_subdir="agcrn",
        checkpoint_filename="agcrn-best-{epoch:02d}-{val_loss:.4f}",
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
