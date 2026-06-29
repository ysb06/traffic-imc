from typing import Any, Literal

from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.loggers.logger import Logger

from .config import LoggingConfig, RunConfig
from .spec import TrainingSpec


TrainerLogger = Logger | Literal[False]


def build_logger(
    cfg: LoggingConfig,
    run: RunConfig,
    spec: TrainingSpec,
    wandb_config: dict[str, Any] | None = None,
) -> TrainerLogger:
    run_name = f"{spec.display_name}-{run.name}-{run.code:02d}"

    if cfg.type == "none":
        return False

    if cfg.type == "csv":
        return CSVLogger(
            save_dir=cfg.save_dir,
            name=spec.output_subdir,
            version=f"{run.name}_{run.code:02d}",
        )

    if cfg.type == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger

        return TensorBoardLogger(
            save_dir=cfg.save_dir,
            name=spec.output_subdir,
            version=f"{run.name}_{run.code:02d}",
        )

    if cfg.type == "wandb":
        kwargs: dict[str, Any] = {
            "name": run_name,
            "project": cfg.project,
            "save_dir": cfg.save_dir,
            "log_model": cfg.log_model,
        }
        if wandb_config is not None:
            kwargs["config"] = wandb_config
        return WandbLogger(**kwargs)

    raise ValueError(f"Unsupported logger type: {cfg.type}")
