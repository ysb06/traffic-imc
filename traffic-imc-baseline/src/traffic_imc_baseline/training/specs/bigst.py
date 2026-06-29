from traffic_imc_dataset.utils import PathConfig

from ...models.bigst.datamodule import BigSTDataModule
from ...models.bigst.module import BigSTLightningModule
from ..spec import StrictParams, TrainingSpec


class BigSTDataParams(StrictParams):
    train_val_split: float = 0.8
    input_length: int = 24
    output_length: int = 24
    batch_size: int = 64
    num_workers: int = 0
    shuffle_training: bool = False


class BigSTModelParams(StrictParams):
    input_dim: int = 3
    output_dim: int = 1
    hid_dim: int = 32
    num_layers: int = 3
    tau: float = 0.25
    random_feature_dim: int = 64
    node_dim: int = 32
    time_dim: int = 32
    time_num: int = 24
    week_num: int = 7
    dropout: float = 0.3
    use_residual: bool = True
    use_bn: bool = True
    use_spatial: bool = True
    spatial_loss_weight: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    adam_epsilon: float = 1.0e-8
    milestones: tuple[int, ...] = (80, 100)
    gamma: float = 0.1


def build_datamodule(
    params: BigSTDataParams,
    path_config: PathConfig,
) -> BigSTDataModule:
    return BigSTDataModule(
        training_data_path=path_config.traffic_imc_training_path,
        test_data_path=path_config.traffic_imc_test_path,
        test_missing_path=path_config.traffic_imc_test_missing_path,
        adj_mx_path=path_config.adj_mx_path,
        **params.model_dump(),
    )


def build_model(
    params: BigSTModelParams,
    data_params: BigSTDataParams,
    data: BigSTDataModule,
    path_config: PathConfig,
) -> BigSTLightningModule:
    del path_config

    if data.scaler is None:
        raise ValueError("Scaler is not fitted in the data module.")
    if params.input_dim != data.input_dim:
        raise ValueError(
            "BigST input_dim must match DataModule feature count: "
            f"{params.input_dim} != {data.input_dim}"
        )

    return BigSTLightningModule(
        adj_mx=data.adj_mx,
        num_nodes=data.num_nodes,
        input_length=data_params.input_length,
        output_length=data_params.output_length,
        scaler=data.scaler,
        **params.model_dump(),
    )


def get_spec() -> TrainingSpec[BigSTDataParams, BigSTModelParams]:
    return TrainingSpec(
        key="bigst",
        display_name="BigST",
        output_subdir="bigst",
        default_checkpoint_filename="bigst-best-{epoch:02d}-{val_loss:.4f}",
        data_params_type=BigSTDataParams,
        model_params_type=BigSTModelParams,
        build_datamodule=build_datamodule,
        build_model=build_model,
    )
