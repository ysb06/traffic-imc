from pathlib import Path
from typing import Optional

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from traffic_imc_dataset.utils import PathConfig

from .common import ModelAdapter, build_output_dir, resolve_path, resolve_seed


def run_training(
    adapter: ModelAdapter,
    config_path: str,
    seed: Optional[int] = None,
) -> None:
    config_file = Path(config_path).resolve()
    cfg = adapter.load_config(config_file)

    run_metadata = adapter.get_run_metadata(cfg)
    early_stopping_metadata = adapter.get_early_stopping_metadata(cfg)
    trainer_metadata = adapter.get_trainer_metadata(cfg)

    final_seed = resolve_seed(seed, run_metadata.seed)
    if final_seed is not None:
        seed_everything(final_seed, workers=True)

    path_config_file = resolve_path(config_file.parent, cfg.path_config_file)
    path_config = PathConfig.from_yaml(path_config_file)

    data = adapter.build_datamodule(cfg, path_config)
    data.setup()
    model = adapter.build_model(cfg, data, path_config)

    output_dir = build_output_dir(
        adapter.output_subdir,
        run_metadata.name_key,
        run_metadata.code,
    )

    wandb_logger = WandbLogger(
        name=f"{adapter.wandb_prefix}-{run_metadata.name_key}-{run_metadata.code:02d}",
        project="Traffic-IMC",
        log_model="all",
    )

    callbacks = []
    if early_stopping_metadata.enabled:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=early_stopping_metadata.patience,
                verbose=True,
            )
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=adapter.checkpoint_filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = Trainer(
        max_epochs=trainer_metadata.max_epochs,
        default_root_dir=output_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=trainer_metadata.gradient_clip_val,
    )

    trainer.fit(model, data)

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        raise RuntimeError(
            "Best checkpoint path is empty. Cannot run test without a best checkpoint."
        )
    if not Path(best_model_path).is_file():
        raise RuntimeError(
            f"Best checkpoint file does not exist: {best_model_path}"
        )

    trainer.test(model, data, ckpt_path="best")
