from typing import Literal

import torch
from traffic_imc_dataset.components.adj_mx import AdjacencyMatrix
from traffic_imc_dataset.utils import PathConfig

from ...models.stgcn.datamodule import STGCNSplitDataModule
from ...models.stgcn.module import STGCNLightningModule
from ...models.stgcn.utils import GSO_TYPE, prepare_gso_for_model
from ..spec import StrictParams, TrainingSpec


class STGCNDataParams(StrictParams):
    train_val_split: float = 0.8
    n_his: int = 24
    n_pred: int = 24
    batch_size: int = 64
    num_workers: int = 0
    shuffle_training: bool = False


class STGCNModelParams(StrictParams):
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    rmsprop_alpha: float = 0.9
    rmsprop_epsilon: float = 1.0e-10
    lr_decay_step: int = 5
    lr_decay_rate: float = 0.7
    dropout_rate: float = 0.5
    Kt: int = 3
    stblock_num: int = 2
    Ks: int = 3
    act_func: Literal["glu", "gtu", "relu", "silu"] = "glu"
    graph_conv_type: Literal["cheb_graph_conv", "graph_conv"] = "graph_conv"
    gso_type: GSO_TYPE = "rw_norm_lap"
    force_symmetric: bool = False
    enable_bias: bool = True


def _get_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_datamodule(
    params: STGCNDataParams,
    path_config: PathConfig,
) -> STGCNSplitDataModule:
    return STGCNSplitDataModule(
        training_data_path=path_config.traffic_imc_training_path,
        test_data_path=path_config.traffic_imc_test_path,
        test_missing_path=path_config.traffic_imc_test_missing_path,
        adj_mx_path=path_config.adj_mx_path,
        **params.model_dump(),
    )


def build_model(
    params: STGCNModelParams,
    data_params: STGCNDataParams,
    data: STGCNSplitDataModule,
    path_config: PathConfig,
) -> STGCNLightningModule:
    if data.scaler is None:
        raise ValueError("Scaler is not fitted in the data module.")

    adj_mx_obj = AdjacencyMatrix.import_from_pickle(path_config.adj_mx_path)
    gso_tensor = prepare_gso_for_model(
        adj_mx=adj_mx_obj.adj_mx,
        gso_type=params.gso_type,
        graph_conv_type=params.graph_conv_type,
        device=torch.device("cpu"),
        force_symmetric=params.force_symmetric,
    )

    return STGCNLightningModule(
        gso=gso_tensor,
        n_his=data_params.n_his,
        n_pred=data_params.n_pred,
        scaler=data.scaler,
        **params.model_dump(exclude={"gso_type", "force_symmetric"}),
    )


def get_spec() -> TrainingSpec[STGCNDataParams, STGCNModelParams]:
    return TrainingSpec(
        key="stgcn",
        display_name="STGCN",
        output_subdir="stgcn",
        default_checkpoint_filename="stgcn-best-{epoch:02d}-{val_loss:.4f}",
        data_params_type=STGCNDataParams,
        model_params_type=STGCNModelParams,
        build_datamodule=build_datamodule,
        build_model=build_model,
    )
