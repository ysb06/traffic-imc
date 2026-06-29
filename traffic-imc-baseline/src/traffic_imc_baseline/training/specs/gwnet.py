from traffic_imc_dataset.utils import PathConfig

from ...models.gwnet.datamodule import GWNetDataModule
from ...models.gwnet.module import GWNetLightningModule
from ..spec import StrictParams, TrainingSpec


class GWNetDataParams(StrictParams):
    train_val_split: float = 0.8
    seq_len: int = 24
    horizon: int = 24
    batch_size: int = 64
    num_workers: int = 0
    shuffle_training: bool = False
    add_time_in_day: bool = True
    add_day_in_week: bool = False


class GWNetModelParams(StrictParams):
    input_dim: int = 2
    output_dim: int = 1
    dropout: float = 0.3
    gcn_bool: bool = True
    addaptadj: bool = True
    randomadj: bool = True
    apt_only: bool = False
    adjtype: str = "doubletransition"
    residual_channels: int = 32
    dilation_channels: int = 32
    skip_channels: int = 256
    end_channels: int = 512
    kernel_size: int = 2
    blocks: int = 4
    layers: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 0.0001


def build_datamodule(
    params: GWNetDataParams,
    path_config: PathConfig,
) -> GWNetDataModule:
    return GWNetDataModule(
        training_data_path=path_config.traffic_imc_training_path,
        test_data_path=path_config.traffic_imc_test_path,
        test_missing_path=path_config.traffic_imc_test_missing_path,
        adj_mx_path=path_config.adj_mx_path,
        **params.model_dump(),
    )


def build_model(
    params: GWNetModelParams,
    data_params: GWNetDataParams,
    data: GWNetDataModule,
    path_config: PathConfig,
) -> GWNetLightningModule:
    del path_config

    if data.scaler is None:
        raise ValueError("Scaler is not fitted in the data module.")
    if params.input_dim != data.input_dim:
        raise ValueError(
            "GWNet input_dim must match DataModule feature count: "
            f"{params.input_dim} != {data.input_dim}"
        )

    return GWNetLightningModule(
        adj_mx=data.adj_mx,
        num_nodes=data.num_nodes,
        seq_len=data_params.seq_len,
        horizon=data_params.horizon,
        scaler=data.scaler,
        **params.model_dump(),
    )


def get_spec() -> TrainingSpec[GWNetDataParams, GWNetModelParams]:
    return TrainingSpec(
        key="gwnet",
        display_name="Graph WaveNet",
        output_subdir="gwnet",
        default_checkpoint_filename="gwnet-best-{epoch:02d}-{val_loss:.4f}",
        data_params_type=GWNetDataParams,
        model_params_type=GWNetModelParams,
        build_datamodule=build_datamodule,
        build_model=build_model,
    )
