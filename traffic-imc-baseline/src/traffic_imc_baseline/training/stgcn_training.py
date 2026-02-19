from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import torch
from traffic_imc_dataset.components.adj_mx import AdjacencyMatrix
from traffic_imc_dataset.utils import PathConfig

from ..models.stgcn.datamodule import STGCNSplitDataModule
from ..models.stgcn.module import STGCNLightningModule
from ..models.stgcn.utils import GSO_TYPE, prepare_gso_for_model
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
    n_his: int = 24
    n_pred: int = 1
    batch_size: int = 64
    num_workers: int = 0
    shuffle_training: bool = True


@dataclass
class ModelConfig:
    learning_rate: float = 0.001
    scheduler_factor: float = 0.95
    scheduler_patience: int = 10
    dropout_rate: float = 0.5
    Kt: int = 3
    stblock_num: int = 2
    Ks: int = 3
    act_func: Literal["glu", "gtu", "relu", "silu"] = "glu"
    graph_conv_type: Literal["cheb_graph_conv", "graph_conv"] = "graph_conv"
    gso_type: GSO_TYPE = "sym_norm_lap"
    force_symmetric: bool = True
    enable_bias: bool = True


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 20


@dataclass
class TrainerConfig:
    max_epochs: int = 100
    gradient_clip_val: float = 1.0


@dataclass
class STGCNTrainingConfig:
    path_config_file: str
    run: RunConfig = field(default_factory=RunConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


def _load_training_config(path: Path) -> STGCNTrainingConfig:
    raw = read_yaml(path)

    return STGCNTrainingConfig(
        path_config_file=read_path_config_file(raw),
        run=RunConfig(**read_section(raw, "run")),
        datamodule=DataModuleConfig(**read_section(raw, "datamodule")),
        model=ModelConfig(**read_section(raw, "model")),
        early_stopping=EarlyStoppingConfig(**read_section(raw, "early_stopping")),
        trainer=TrainerConfig(**read_section(raw, "trainer")),
    )


def _get_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_datamodule(
    cfg: STGCNTrainingConfig,
    path_config: PathConfig,
) -> STGCNSplitDataModule:
    return STGCNSplitDataModule(
        training_data_path=path_config.metr_imc_training_path,
        test_data_path=path_config.metr_imc_test_path,
        test_missing_path=path_config.metr_imc_test_missing_path,
        adj_mx_path=path_config.adj_mx_path,
        n_his=cfg.datamodule.n_his,
        n_pred=cfg.datamodule.n_pred,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
        shuffle_training=cfg.datamodule.shuffle_training,
        train_val_split=cfg.datamodule.train_val_split,
    )


def _build_model(
    cfg: STGCNTrainingConfig,
    data: STGCNSplitDataModule,
    path_config: PathConfig,
) -> STGCNLightningModule:
    if data.scaler is None:
        raise ValueError("Scaler is not fitted in the data module.")

    adj_mx_obj = AdjacencyMatrix.import_from_pickle(path_config.adj_mx_path)
    gso_tensor = prepare_gso_for_model(
        adj_mx=adj_mx_obj.adj_mx,
        gso_type=cfg.model.gso_type,
        graph_conv_type=cfg.model.graph_conv_type,
        device=_get_training_device(),
        force_symmetric=cfg.model.force_symmetric,
    )

    return STGCNLightningModule(
        gso=gso_tensor,
        learning_rate=cfg.model.learning_rate,
        scheduler_factor=cfg.model.scheduler_factor,
        scheduler_patience=cfg.model.scheduler_patience,
        dropout_rate=cfg.model.dropout_rate,
        n_his=cfg.datamodule.n_his,
        Kt=cfg.model.Kt,
        stblock_num=cfg.model.stblock_num,
        Ks=cfg.model.Ks,
        act_func=cfg.model.act_func,
        graph_conv_type=cfg.model.graph_conv_type,
        enable_bias=cfg.model.enable_bias,
        scaler=data.scaler,
    )


def _get_run_metadata(cfg: STGCNTrainingConfig) -> RunMetadata:
    return RunMetadata(
        name_key=cfg.run.name_key,
        code=cfg.run.code,
        seed=cfg.run.seed,
    )


def _get_early_stopping_metadata(
    cfg: STGCNTrainingConfig,
) -> EarlyStoppingMetadata:
    return EarlyStoppingMetadata(
        enabled=cfg.early_stopping.enabled,
        patience=cfg.early_stopping.patience,
    )


def _get_trainer_metadata(cfg: STGCNTrainingConfig) -> TrainerMetadata:
    return TrainerMetadata(
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )


def get_adapter() -> ModelAdapter:
    return ModelAdapter(
        key="stgcn",
        wandb_prefix="STGCN",
        output_subdir="stgcn",
        checkpoint_filename="stgcn-best-{epoch:02d}-{val_loss:.4f}",
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
