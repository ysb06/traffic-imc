from typing import Optional

from traffic_imc_dataset.utils import PathConfig

from ...models.agcrn.datamodule import AGCRNDataModule
from ...models.agcrn.module import AGCRNLightningModule
from ..spec import StrictParams, TrainingSpec


class AGCRNDataParams(StrictParams):
    train_val_split: float = 0.8
    in_steps: int = 24
    out_steps: int = 24
    batch_size: int = 64
    num_workers: int = 0
    shuffle_training: bool = False
    target_sensors: Optional[list[str]] = None


class AGCRNModelParams(StrictParams):
    input_dim: int = 1
    output_dim: int = 1
    rnn_units: int = 64
    num_layers: int = 2
    embed_dim: int = 10
    cheb_k: int = 2
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    adam_epsilon: float = 1.0e-8


def build_datamodule(
    params: AGCRNDataParams,
    path_config: PathConfig,
) -> AGCRNDataModule:
    return AGCRNDataModule(
        training_dataset_path=path_config.traffic_imc_training_path,
        test_dataset_path=path_config.traffic_imc_test_path,
        test_missing_path=path_config.traffic_imc_test_missing_path,
        **params.model_dump(),
    )


def build_model(
    params: AGCRNModelParams,
    data_params: AGCRNDataParams,
    data: AGCRNDataModule,
    path_config: PathConfig,
) -> AGCRNLightningModule:
    del path_config

    if data.num_nodes is None:
        raise ValueError("num_nodes is not initialized in the data module.")

    return AGCRNLightningModule(
        num_nodes=data.num_nodes,
        in_steps=data_params.in_steps,
        out_steps=data_params.out_steps,
        scaler=data.scaler,
        **params.model_dump(),
    )


def get_spec() -> TrainingSpec[AGCRNDataParams, AGCRNModelParams]:
    return TrainingSpec(
        key="agcrn",
        display_name="AGCRN",
        output_subdir="agcrn",
        default_checkpoint_filename="agcrn-best-{epoch:02d}-{val_loss:.4f}",
        data_params_type=AGCRNDataParams,
        model_params_type=AGCRNModelParams,
        build_datamodule=build_datamodule,
        build_model=build_model,
    )
