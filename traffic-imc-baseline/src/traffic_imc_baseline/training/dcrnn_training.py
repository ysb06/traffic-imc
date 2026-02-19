from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from traffic_imc_dataset.components.adj_mx import AdjacencyMatrix
from traffic_imc_dataset.utils import PathConfig

from ..models.dcrnn.datamodule import DCRNNSplitDataModule
from ..models.dcrnn.module import DCRNNLightningModule
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
    seq_len: int = 24
    horizon: int = 1
    batch_size: int = 64
    num_workers: int = 0
    shuffle_training: bool = True
    add_time_in_day: bool = True
    add_day_in_week: bool = False


@dataclass
class ModelConfig:
    rnn_units: int = 64
    num_rnn_layers: int = 2
    max_diffusion_step: int = 2
    filter_type: str = "dual_random_walk"
    use_curriculum_learning: bool = True
    cl_decay_steps: int = 2000
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    scheduler_milestones: list[int] = field(default_factory=lambda: [20, 30, 40, 50])
    scheduler_gamma: float = 0.1


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 10


@dataclass
class TrainerConfig:
    max_epochs: int = 20
    gradient_clip_val: float = 1.0


@dataclass
class DCRNNTrainingConfig:
    path_config_file: str
    run: RunConfig = field(default_factory=RunConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


def _load_training_config(path: Path) -> DCRNNTrainingConfig:
    raw = read_yaml(path)

    return DCRNNTrainingConfig(
        path_config_file=read_path_config_file(raw),
        run=RunConfig(**read_section(raw, "run")),
        datamodule=DataModuleConfig(**read_section(raw, "datamodule")),
        model=ModelConfig(**read_section(raw, "model")),
        early_stopping=EarlyStoppingConfig(**read_section(raw, "early_stopping")),
        trainer=TrainerConfig(**read_section(raw, "trainer")),
    )


def _build_datamodule(
    cfg: DCRNNTrainingConfig,
    path_config: PathConfig,
) -> DCRNNSplitDataModule:
    return DCRNNSplitDataModule(
        training_data_path=path_config.metr_imc_training_path,
        test_data_path=path_config.metr_imc_test_path,
        test_missing_path=path_config.metr_imc_test_missing_path,
        adj_mx_path=path_config.adj_mx_path,
        seq_len=cfg.datamodule.seq_len,
        horizon=cfg.datamodule.horizon,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
        shuffle_training=cfg.datamodule.shuffle_training,
        train_val_split=cfg.datamodule.train_val_split,
        add_time_in_day=cfg.datamodule.add_time_in_day,
        add_day_in_week=cfg.datamodule.add_day_in_week,
    )


def _build_model(
    cfg: DCRNNTrainingConfig,
    data: DCRNNSplitDataModule,
    path_config: PathConfig,
) -> DCRNNLightningModule:
    if data.scaler is None:
        raise ValueError("Scaler is not fitted in the data module.")

    adj_mx = AdjacencyMatrix.import_from_pickle(path_config.adj_mx_path)

    return DCRNNLightningModule(
        adj_mx=adj_mx.adj_mx,
        num_nodes=data.num_nodes,
        input_dim=data.input_dim,
        output_dim=data.output_dim,
        seq_len=cfg.datamodule.seq_len,
        horizon=cfg.datamodule.horizon,
        rnn_units=cfg.model.rnn_units,
        num_rnn_layers=cfg.model.num_rnn_layers,
        max_diffusion_step=cfg.model.max_diffusion_step,
        filter_type=cfg.model.filter_type,
        use_curriculum_learning=cfg.model.use_curriculum_learning,
        cl_decay_steps=cfg.model.cl_decay_steps,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        scheduler_milestones=cfg.model.scheduler_milestones,
        scheduler_gamma=cfg.model.scheduler_gamma,
        scaler=data.scaler,
    )


def _get_run_metadata(cfg: DCRNNTrainingConfig) -> RunMetadata:
    return RunMetadata(
        name_key=cfg.run.name_key,
        code=cfg.run.code,
        seed=cfg.run.seed,
    )


def _get_early_stopping_metadata(
    cfg: DCRNNTrainingConfig,
) -> EarlyStoppingMetadata:
    return EarlyStoppingMetadata(
        enabled=cfg.early_stopping.enabled,
        patience=cfg.early_stopping.patience,
    )


def _get_trainer_metadata(cfg: DCRNNTrainingConfig) -> TrainerMetadata:
    return TrainerMetadata(
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )


def get_adapter() -> ModelAdapter:
    return ModelAdapter(
        key="dcrnn",
        wandb_prefix="DCRNN",
        output_subdir="dcrnn",
        checkpoint_filename="dcrnn-best-{epoch:02d}-{val_loss:.4f}",
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
