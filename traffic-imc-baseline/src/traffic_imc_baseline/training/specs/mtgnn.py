from traffic_imc_dataset.utils import PathConfig

from ...models.mtgnn.datamodule import MTGNNDataModule
from ...models.mtgnn.module import MTGNNLightningModule
from ..spec import StrictParams, TrainingSpec


class MTGNNDataParams(StrictParams):
    train_val_split: float = 0.8
    seq_len: int = 24
    horizon: int = 24
    batch_size: int = 64
    num_workers: int = 0
    shuffle_training: bool = False
    add_time_in_day: bool = True
    add_day_in_week: bool = False


class MTGNNModelParams(StrictParams):
    input_dim: int = 2
    output_dim: int = 1
    gcn_true: bool = True
    buildA_true: bool = True
    gcn_depth: int = 2
    dropout: float = 0.3
    subgraph_size: int = 20
    node_dim: int = 40
    dilation_exponential: int = 1
    conv_channels: int = 32
    residual_channels: int = 32
    skip_channels: int = 64
    end_channels: int = 128
    layers: int = 3
    propalpha: float = 0.05
    tanhalpha: float = 3
    layer_norm_affine: bool = True
    learning_rate: float = 0.001
    weight_decay: float = 0.0001


def build_datamodule(
    params: MTGNNDataParams,
    path_config: PathConfig,
) -> MTGNNDataModule:
    return MTGNNDataModule(
        training_data_path=path_config.traffic_imc_training_path,
        test_data_path=path_config.traffic_imc_test_path,
        test_missing_path=path_config.traffic_imc_test_missing_path,
        adj_mx_path=path_config.adj_mx_path,
        **params.model_dump(),
    )


def build_model(
    params: MTGNNModelParams,
    data_params: MTGNNDataParams,
    data: MTGNNDataModule,
    path_config: PathConfig,
) -> MTGNNLightningModule:
    del path_config

    if data.scaler is None:
        raise ValueError("Scaler is not fitted in the data module.")
    if params.input_dim != data.input_dim:
        raise ValueError(
            "MTGNN input_dim must match DataModule feature count: "
            f"{params.input_dim} != {data.input_dim}"
        )

    return MTGNNLightningModule(
        adj_mx=data.adj_mx,
        num_nodes=data.num_nodes,
        seq_len=data_params.seq_len,
        horizon=data_params.horizon,
        scaler=data.scaler,
        **params.model_dump(),
    )


def get_spec() -> TrainingSpec[MTGNNDataParams, MTGNNModelParams]:
    return TrainingSpec(
        key="mtgnn",
        display_name="MTGNN",
        output_subdir="mtgnn",
        default_checkpoint_filename="mtgnn-best-{epoch:02d}-{val_loss:.4f}",
        data_params_type=MTGNNDataParams,
        model_params_type=MTGNNModelParams,
        build_datamodule=build_datamodule,
        build_model=build_model,
    )
