from dataclasses import asdict
import gc
from pathlib import Path
from typing import Any, Optional, Union

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from traffic_imc_dataset.utils import PathConfig

from .config import RuntimeConfig, TrainingConfig, load_training_config, read_yaml
from .loggers import build_logger
from .registry import get_spec
from .runtime import configure_pin_memory


def _resolve_seed(cli_seed: Optional[int], config_seed: Optional[int]) -> Optional[int]:
    if cli_seed is not None:
        return cli_seed
    return config_seed


def _build_output_dir(output_subdir: str, run_name: str, code: int) -> str:
    return f"./output/{output_subdir}/{run_name}_{code:02d}"


def _configure_torch_runtime(cfg: RuntimeConfig) -> None:
    cuda_available = torch.cuda.is_available()
    configure_pin_memory(cfg.accelerator, cuda_available)
    if cfg.cpu_num_threads is not None:
        torch.set_num_threads(cfg.cpu_num_threads)
    if cfg.cpu_num_interop_threads is not None:
        torch.set_num_interop_threads(cfg.cpu_num_interop_threads)
    if cuda_available and cfg.accelerator not in {"cpu", "mps"}:
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision or "high")


def _trainer_runtime_kwargs(cfg: RuntimeConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if cfg.accelerator is not None:
        kwargs["accelerator"] = cfg.accelerator
    if cfg.devices is not None:
        kwargs["devices"] = cfg.devices
    return kwargs


def _resolve_resume_ckpt_path(
    ckpt_path: Optional[str],
    config_file: Path,
) -> Optional[str]:
    if ckpt_path is None:
        return None

    path = Path(ckpt_path)
    if path.is_absolute():
        return str(path)
    return str((config_file.parent / path).resolve())


def _require_existing_test_ckpt_path(
    ckpt_path: Optional[str],
    config_file: Path,
) -> str:
    resolved = _resolve_resume_ckpt_path(ckpt_path, config_file)
    if resolved is None:
        raise RuntimeError(
            "--test_only requires trainer.resume_ckpt_path to point to a checkpoint file."
        )
    if not Path(resolved).is_file():
        raise RuntimeError(f"Test checkpoint file does not exist: {resolved}")
    return resolved


def _build_effective_wandb_config(
    *,
    config_file: Path,
    cfg: TrainingConfig,
    params: Any,
    path_config_file: Path,
    path_config: PathConfig,
    path_config_raw: dict[str, Any],
    final_seed: Optional[int],
) -> dict[str, Any]:
    baseline_config = cfg.model_dump(mode="json")
    baseline_config["run"]["seed"] = final_seed
    baseline_config["data"]["params"] = params.data.model_dump(mode="json")
    baseline_config["model"]["params"] = params.model.model_dump(mode="json")

    resolved_path_config = asdict(path_config)
    resolved_path_config.pop("raw", None)

    return {
        "config_files": {
            "baseline": str(config_file),
            "path_config": str(path_config_file),
        },
        "baseline_config": baseline_config,
        "path_config": resolved_path_config,
        "path_config_raw": path_config_raw,
    }


def run_training(config_path: str, seed: Optional[int] = None) -> None:
    config_file = Path(config_path).resolve()
    cfg = load_training_config(config_file)

    final_seed = _resolve_seed(seed, cfg.run.seed)
    spec = get_spec(cfg.model.name)
    params = spec.validate_params(cfg.data.params, cfg.model.params)

    path_config_file = cfg.resolved_path_config_file(config_file)
    path_config_raw = read_yaml(path_config_file)
    path_config = PathConfig.from_yaml(
        path_config_file,
        base_dir=path_config_file.parent,
    )

    wandb_config = _build_effective_wandb_config(
        config_file=config_file,
        cfg=cfg,
        params=params,
        path_config_file=path_config_file,
        path_config=path_config,
        path_config_raw=path_config_raw,
        final_seed=final_seed,
    )

    logger = build_logger(cfg.logging, cfg.run, spec, wandb_config=wandb_config)

    _configure_torch_runtime(cfg.runtime)

    if final_seed is not None:
        seed_everything(final_seed, workers=True)

    resume_ckpt_path = _resolve_resume_ckpt_path(
        cfg.trainer.resume_ckpt_path,
        config_file,
    )

    output_dir = _build_output_dir(
        spec.output_subdir,
        cfg.run.name,
        cfg.run.code,
    )

    data = spec.build_datamodule(params.data, path_config)
    data.setup("fit")
    model = spec.build_model(params.model, params.data, data, path_config)

    callbacks = []
    if cfg.callbacks.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.callbacks.early_stopping.monitor,
                mode=cfg.callbacks.early_stopping.mode,
                patience=cfg.callbacks.early_stopping.patience,
                verbose=cfg.callbacks.early_stopping.verbose,
            )
        )

    checkpoint_callback: Optional[ModelCheckpoint] = None
    if cfg.callbacks.checkpoint.enabled:
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename=(
                cfg.callbacks.checkpoint.filename or spec.default_checkpoint_filename
            ),
            save_top_k=cfg.callbacks.checkpoint.save_top_k,
            monitor=cfg.callbacks.checkpoint.monitor,
            mode=cfg.callbacks.checkpoint.mode,
            save_last=cfg.callbacks.checkpoint.save_last,
        )
        callbacks.append(checkpoint_callback)

    if cfg.callbacks.lr_monitor.enabled and logger is not False:
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.callbacks.lr_monitor.logging_interval,
            )
        )

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        default_root_dir=output_dir,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        **_trainer_runtime_kwargs(cfg.runtime),
    )

    try:
        trainer.fit(model, data, ckpt_path=resume_ckpt_path)

        if checkpoint_callback is None or cfg.callbacks.checkpoint.save_top_k == 0:
            trainer.test(model, data)
            return

        best_model_path = checkpoint_callback.best_model_path
        if not best_model_path:
            raise RuntimeError(
                "Best checkpoint path is empty. Cannot run test without a best checkpoint."
            )
        if not Path(best_model_path).is_file():
            raise RuntimeError(f"Best checkpoint file does not exist: {best_model_path}")

        trainer.test(model, data, ckpt_path="best")
    finally:
        if cfg.logging.type == "wandb":
            import wandb
            if wandb.run is not None:
                wandb.finish()

        gc.collect()
        torch.cuda.empty_cache()


def run_only_test(
    config_path: str,
    seed: Optional[int] = None,
    extra_callbacks: Optional[list[Any]] = None,
) -> list[dict[str, Any]]:
    config_file = Path(config_path).resolve()
    cfg = load_training_config(config_file)

    final_seed = _resolve_seed(seed, cfg.run.seed)
    spec = get_spec(cfg.model.name)
    params = spec.validate_params(cfg.data.params, cfg.model.params)

    path_config_file = cfg.resolved_path_config_file(config_file)
    path_config_raw = read_yaml(path_config_file)
    path_config = PathConfig.from_yaml(
        path_config_file,
        base_dir=path_config_file.parent,
    )

    wandb_config = _build_effective_wandb_config(
        config_file=config_file,
        cfg=cfg,
        params=params,
        path_config_file=path_config_file,
        path_config=path_config,
        path_config_raw=path_config_raw,
        final_seed=final_seed,
    )

    logger = build_logger(cfg.logging, cfg.run, spec, wandb_config=wandb_config)

    _configure_torch_runtime(cfg.runtime)

    if final_seed is not None:
        seed_everything(final_seed, workers=True)

    test_ckpt_path = _require_existing_test_ckpt_path(
        cfg.trainer.resume_ckpt_path,
        config_file,
    )
    output_dir = _build_output_dir(
        spec.output_subdir,
        cfg.run.name,
        cfg.run.code,
    )

    data = spec.build_datamodule(params.data, path_config)
    data.setup("test")
    model = spec.build_model(params.model, params.data, data, path_config)

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        default_root_dir=output_dir,
        logger=logger,
        callbacks=extra_callbacks or [],
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        **_trainer_runtime_kwargs(cfg.runtime),
    )

    try:
        return trainer.test(model, data, ckpt_path=test_ckpt_path) or []
    finally:
        if cfg.logging.type == "wandb":
            import wandb
            if wandb.run is not None:
                wandb.finish()

        gc.collect()
        torch.cuda.empty_cache()
