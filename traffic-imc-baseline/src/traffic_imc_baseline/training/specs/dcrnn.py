from traffic_imc_dataset.utils import PathConfig

from ...models.dcrnn.datamodule import DCRNNSplitDataModule
from ...models.dcrnn.module import DCRNNLightningModule
from ..spec import StrictParams, TrainingSpec


class DCRNNDataParams(StrictParams):
    train_val_split: float = 0.8
    seq_len: int = 24
    horizon: int = 24
    batch_size: int = 64
    num_workers: int = 0
    shuffle_training: bool = False
    add_time_in_day: bool = True
    add_day_in_week: bool = False


class DCRNNModelParams(StrictParams):
    rnn_units: int = 64
    num_rnn_layers: int = 2
    max_diffusion_step: int = 2
    filter_type: str = "dual_random_walk"
    use_curriculum_learning: bool = True
    cl_decay_steps: int = 2000
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    adam_epsilon: float = 1.0e-3
    lr_decay_ratio: float = 0.1
    lr_decay_steps: tuple[int, ...] = (20, 30, 40, 50)
    min_learning_rate: float = 2.0e-6


def build_datamodule(
    params: DCRNNDataParams,
    path_config: PathConfig,
) -> DCRNNSplitDataModule:
    return DCRNNSplitDataModule(
        training_data_path=path_config.traffic_imc_training_path,
        test_data_path=path_config.traffic_imc_test_path,
        test_missing_path=path_config.traffic_imc_test_missing_path,
        adj_mx_path=path_config.adj_mx_path,
        **params.model_dump(),
    )


def build_model(
    params: DCRNNModelParams,
    data_params: DCRNNDataParams,
    data: DCRNNSplitDataModule,
    path_config: PathConfig,
) -> DCRNNLightningModule:
    del path_config

    if data.scaler is None:
        raise ValueError("Scaler is not fitted in the data module.")

    return DCRNNLightningModule(
        adj_mx=data.adj_mx,
        num_nodes=data.num_nodes,
        input_dim=data.input_dim,
        output_dim=data.output_dim,
        seq_len=data_params.seq_len,
        horizon=data_params.horizon,
        scaler=data.scaler,
        **params.model_dump(),
    )


def get_spec() -> TrainingSpec[DCRNNDataParams, DCRNNModelParams]:
    return TrainingSpec(
        key="dcrnn",
        display_name="DCRNN",
        output_subdir="dcrnn",
        default_checkpoint_filename="dcrnn-best-{epoch:02d}-{val_loss:.4f}",
        data_params_type=DCRNNDataParams,
        model_params_type=DCRNNModelParams,
        build_datamodule=build_datamodule,
        build_model=build_model,
    )
