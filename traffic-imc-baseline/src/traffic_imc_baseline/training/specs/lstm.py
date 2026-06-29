from traffic_imc_dataset.utils import PathConfig

from ...models.lstm.datamodule import LSTMDataModule
from ...models.lstm.module import LSTMLightningModule
from ..spec import StrictParams, TrainingSpec


class LSTMDataParams(StrictParams):
    train_val_split: float = 0.8
    seq_length: int = 24
    pred_length: int = 24
    batch_size: int = 512
    num_workers: int = 0
    shuffle_training: bool = False
    allow_nan: bool = False


class LSTMModelParams(StrictParams):
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    output_size: int = 24
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    dropout_rate: float = 0.2
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10


def build_datamodule(
    params: LSTMDataParams,
    path_config: PathConfig,
) -> LSTMDataModule:
    return LSTMDataModule(
        training_dataset_path=path_config.traffic_imc_training_path,
        test_dataset_path=path_config.traffic_imc_test_path,
        test_missing_path=path_config.traffic_imc_test_missing_path,
        **params.model_dump(),
    )


def build_model(
    params: LSTMModelParams,
    data_params: LSTMDataParams,
    data: LSTMDataModule,
    path_config: PathConfig,
) -> LSTMLightningModule:
    del path_config

    if data.scaler is None:
        raise ValueError("Scaler is not fitted in the data module.")

    if params.output_size != data_params.pred_length:
        raise ValueError(
            "LSTM model.output_size must match data.params.pred_length: "
            f"{params.output_size} != {data_params.pred_length}"
        )

    return LSTMLightningModule(
        scaler=data.scaler,
        **params.model_dump(),
    )


def get_spec() -> TrainingSpec[LSTMDataParams, LSTMModelParams]:
    return TrainingSpec(
        key="lstm",
        display_name="LSTM",
        output_subdir="lstm",
        default_checkpoint_filename="lstm-best-{epoch:02d}-{val_loss:.4f}",
        data_params_type=LSTMDataParams,
        model_params_type=LSTMModelParams,
        build_datamodule=build_datamodule,
        build_model=build_model,
    )
