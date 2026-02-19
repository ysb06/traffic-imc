from pathlib import Path
from typing import Literal, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from traffic_imc_dataset.components import MissingMasks
from traffic_imc_dataset.components.adj_mx import AdjacencyMatrix
from traffic_imc_dataset.components.metr_imc.traffic_data import TrafficData

from .dataset import DCRNNDataset

DCRNNDatasetSample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
DCRNNTrainBatch = Tuple[torch.Tensor, torch.Tensor]
DCRNNTestBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _to_time_first(
    x_stacked: torch.Tensor,
    y_stacked: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert batch-first tensors to DCRNN time-first flattened format."""
    batch_size = x_stacked.size(0)
    seq_len = x_stacked.size(1)
    num_nodes = x_stacked.size(2)
    input_dim = x_stacked.size(3)

    horizon = y_stacked.size(1)
    output_dim = y_stacked.size(3)

    x_batch = x_stacked.permute(1, 0, 2, 3).reshape(
        seq_len,
        batch_size,
        num_nodes * input_dim,
    )
    y_batch = y_stacked.permute(1, 0, 2, 3).reshape(
        horizon,
        batch_size,
        num_nodes * output_dim,
    )
    return x_batch, y_batch


def collate_dcrnn_train(batch: list[DCRNNDatasetSample]) -> DCRNNTrainBatch:
    x_list, y_list, _ = zip(*batch)
    x_stacked = torch.stack(x_list, dim=0)
    y_stacked = torch.stack(y_list, dim=0)
    return _to_time_first(x_stacked, y_stacked)


def collate_dcrnn_test(batch: list[DCRNNDatasetSample]) -> DCRNNTestBatch:
    x_list, y_list, y_missing_list = zip(*batch)
    x_stacked = torch.stack(x_list, dim=0)
    y_stacked = torch.stack(y_list, dim=0)

    x_batch, y_batch = _to_time_first(x_stacked, y_stacked)
    y_missing_batch = torch.stack(y_missing_list, dim=0).permute(1, 0, 2)
    return x_batch, y_batch, y_missing_batch


class DCRNNSplitDataModule(L.LightningDataModule):
    """DCRNN DataModule with separate training and test dataset files."""

    def __init__(
        self,
        training_data_path: str,
        test_data_path: str,
        test_missing_path: str,
        adj_mx_path: str,
        seq_len: int = 24,
        horizon: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle_training: bool = True,
        train_val_split: float = 0.8,
        add_time_in_day: bool = True,
        add_day_in_week: bool = False,
    ):
        super().__init__()

        self.training_data_path = Path(training_data_path)
        self.test_data_path = Path(test_data_path)
        self.test_missing_path = Path(test_missing_path)
        self.adj_mx_path = Path(adj_mx_path)

        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training
        self.train_val_split = train_val_split
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.adj_mx_raw: Optional[AdjacencyMatrix] = None
        self._scaler: Optional[MinMaxScaler] = None

        self.training_dataset: Optional[DCRNNDataset] = None
        self.validation_dataset: Optional[DCRNNDataset] = None
        self.test_dataset: Optional[DCRNNDataset] = None

    @property
    def scaler(self) -> Optional[MinMaxScaler]:
        return self._scaler

    @property
    def adj_mx(self) -> np.ndarray:
        if self.adj_mx_raw is None:
            raise ValueError("DataModule not setup. Call setup() first.")
        return self.adj_mx_raw.adj_mx

    @property
    def num_nodes(self) -> int:
        if self.adj_mx_raw is None:
            raise ValueError("DataModule not setup. Call setup() first.")
        return len(self.adj_mx_raw.sensor_ids)

    @property
    def input_dim(self) -> int:
        dim = 1
        if self.add_time_in_day:
            dim += 1
        if self.add_day_in_week:
            dim += 7
        return dim

    @property
    def output_dim(self) -> int:
        return 1

    def _prepare_scaler(self, train_data: np.ndarray) -> None:
        ref_data = train_data.reshape(-1, 1)
        ref_data = ref_data[~np.isnan(ref_data).any(axis=1)]

        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler.fit(ref_data)

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None:
            raise ValueError("Scaler must be fitted before applying scaling")

        scaled_values = self._scaler.transform(df.values.reshape(-1, 1))
        scaled_values = scaled_values.reshape(df.shape)
        return pd.DataFrame(scaled_values, index=df.index, columns=df.columns)

    def _load_training_data(
        self,
        ordered_sensor_ids: list[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raw = TrafficData.import_from_hdf(str(self.training_data_path))
        raw_df = raw.data[ordered_sensor_ids]

        split_idx = int(len(raw_df) * self.train_val_split)
        train_df = raw_df.iloc[:split_idx]
        val_df = raw_df.iloc[split_idx:]
        return train_df, val_df

    def _load_test_data(
        self,
        ordered_sensor_ids: list[str],
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        raw = TrafficData.import_from_hdf(str(self.test_data_path))
        raw_df = raw.data[ordered_sensor_ids]

        missing_masks = MissingMasks.import_from_hdf(str(self.test_missing_path))
        missing_mask_df = missing_masks.data[ordered_sensor_ids]

        missing_mask_aligned = missing_mask_df.reindex(
            index=raw_df.index,
            columns=raw_df.columns,
            fill_value=False,
        )
        return raw_df, missing_mask_aligned.values

    def setup(
        self,
        stage: Optional[Literal["fit", "validate", "test", "predict"]] = None,
    ):
        self.adj_mx_raw = AdjacencyMatrix.import_from_pickle(str(self.adj_mx_path))
        ordered_sensor_ids = self.adj_mx_raw.sensor_ids

        if stage in ["fit", "validate", None]:
            train_df, val_df = self._load_training_data(ordered_sensor_ids)

            self._prepare_scaler(train_df.values)

            train_df_scaled = self._apply_scaling(train_df)
            val_df_scaled = self._apply_scaling(val_df)

            self.training_dataset = DCRNNDataset(
                train_df_scaled,
                seq_len=self.seq_len,
                horizon=self.horizon,
                add_time_in_day=self.add_time_in_day,
                add_day_in_week=self.add_day_in_week,
                missing_mask=None,
            )
            self.validation_dataset = DCRNNDataset(
                val_df_scaled,
                seq_len=self.seq_len,
                horizon=self.horizon,
                add_time_in_day=self.add_time_in_day,
                add_day_in_week=self.add_day_in_week,
                missing_mask=None,
            )

        if stage in ["test", None]:
            test_df, test_missing_mask = self._load_test_data(ordered_sensor_ids)

            if self._scaler is None:
                train_df, _ = self._load_training_data(ordered_sensor_ids)
                self._prepare_scaler(train_df.values)

            test_df_scaled = self._apply_scaling(test_df)

            self.test_dataset = DCRNNDataset(
                test_df_scaled,
                seq_len=self.seq_len,
                horizon=self.horizon,
                add_time_in_day=self.add_time_in_day,
                add_day_in_week=self.add_day_in_week,
                missing_mask=test_missing_mask,
            )

    def train_dataloader(self) -> DataLoader:
        if self.training_dataset is None:
            raise ValueError("Training dataset is not initialized. Call setup() first.")

        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_dcrnn_train,
        )

    def val_dataloader(self) -> DataLoader:
        if self.validation_dataset is None:
            raise ValueError(
                "Validation dataset is not initialized. Call setup() first."
            )

        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_dcrnn_train,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("Test dataset is not initialized. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_dcrnn_test,
        )
