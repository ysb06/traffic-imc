from typing import Optional

from traffic_imc_dataset.utils import PathConfig

from ...models.stid.datamodule import STIDDataModule
from ...models.stid.module import STIDLightningModule
from ..spec import StrictParams, TrainingSpec


class STIDDataParams(StrictParams):
    train_val_split: float = 0.8
    in_steps: int = 24
    out_steps: int = 24
    batch_size: int = 64
    num_workers: int = 0
    shuffle_training: bool = False
    target_sensors: Optional[list[str]] = None


class STIDModelParams(StrictParams):
    input_dim: int = 3
    output_dim: int = 1
    embed_dim: int = 32
    num_layer: int = 3
    if_node: bool = True
    node_dim: int = 32
    if_time_in_day: bool = True
    if_day_in_week: bool = True
    temp_dim_tid: int = 32
    temp_dim_diw: int = 32
    time_of_day_size: int = 24
    day_of_week_size: int = 7
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    milestones: tuple[int, ...] = (1, 50, 80)
    gamma: float = 0.5


def build_datamodule(
    params: STIDDataParams,
    path_config: PathConfig,
) -> STIDDataModule:
    return STIDDataModule(
        training_dataset_path=path_config.traffic_imc_training_path,
        test_dataset_path=path_config.traffic_imc_test_path,
        test_missing_path=path_config.traffic_imc_test_missing_path,
        **params.model_dump(),
    )


def build_model(
    params: STIDModelParams,
    data_params: STIDDataParams,
    data: STIDDataModule,
    path_config: PathConfig,
) -> STIDLightningModule:
    del path_config

    if data.num_nodes is None:
        raise ValueError("num_nodes is not initialized in the data module.")

    return STIDLightningModule(
        num_nodes=data.num_nodes,
        in_steps=data_params.in_steps,
        out_steps=data_params.out_steps,
        scaler=data.scaler,
        **params.model_dump(),
    )


def get_spec() -> TrainingSpec[STIDDataParams, STIDModelParams]:
    return TrainingSpec(
        key="stid",
        display_name="STID",
        output_subdir="stid",
        default_checkpoint_filename="stid-best-{epoch:02d}-{val_loss:.4f}",
        data_params_type=STIDDataParams,
        model_params_type=STIDModelParams,
        build_datamodule=build_datamodule,
        build_model=build_model,
    )
