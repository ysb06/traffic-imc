from traffic_imc_dataset.utils import PathConfig

from ...models.mlcaformer.datamodule import MLCAFormerDataModule
from ...models.mlcaformer.module import MLCAFormerLightningModule
from ..spec import StrictParams, TrainingSpec


class MLCAFormerDataParams(StrictParams):
    train_val_split: float = 0.8
    in_steps: int = 24
    out_steps: int = 24
    steps_per_day: int = 24
    batch_size: int = 16
    num_workers: int = 0
    shuffle_training: bool = False


class MLCAFormerModelParams(StrictParams):
    input_dim: int = 3
    output_dim: int = 1
    input_embedding_dim: int = 24
    tod_embedding_dim: int = 24
    dow_embedding_dim: int = 24
    nid_embedding_dim: int = 24
    col_embedding_dim: int = 80
    feed_forward_dim: int = 256
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.0003
    adam_epsilon: float = 1.0e-8
    milestones: tuple[int, ...] = (15, 30, 50)
    lr_decay_rate: float = 0.1


def build_datamodule(
    params: MLCAFormerDataParams,
    path_config: PathConfig,
) -> MLCAFormerDataModule:
    return MLCAFormerDataModule(
        training_dataset_path=path_config.traffic_imc_training_path,
        test_dataset_path=path_config.traffic_imc_test_path,
        test_missing_path=path_config.traffic_imc_test_missing_path,
        **params.model_dump(),
    )


def build_model(
    params: MLCAFormerModelParams,
    data_params: MLCAFormerDataParams,
    data: MLCAFormerDataModule,
    path_config: PathConfig,
) -> MLCAFormerLightningModule:
    del path_config

    if data.num_nodes is None:
        raise ValueError("num_nodes is not initialized in the data module.")

    return MLCAFormerLightningModule(
        num_nodes=data.num_nodes,
        in_steps=data_params.in_steps,
        out_steps=data_params.out_steps,
        steps_per_day=data_params.steps_per_day,
        scaler=data.scaler,
        **params.model_dump(),
    )


def get_spec() -> TrainingSpec[MLCAFormerDataParams, MLCAFormerModelParams]:
    return TrainingSpec(
        key="mlcaformer",
        display_name="MLCAFormer",
        output_subdir="mlcaformer",
        default_checkpoint_filename="mlcaformer-best-{epoch:02d}-{val_loss:.4f}",
        data_params_type=MLCAFormerDataParams,
        model_params_type=MLCAFormerModelParams,
        build_datamodule=build_datamodule,
        build_model=build_model,
    )
