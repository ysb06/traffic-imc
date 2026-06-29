"""MTGNN DataModule."""

from pathlib import Path
from typing import Literal, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from traffic_imc_baseline.training.runtime import should_pin_memory

from traffic_imc_dataset.components import MissingMasks
from traffic_imc_dataset.components.adj_mx import AdjacencyMatrix
from traffic_imc_dataset.components.traffic_imc.traffic_data import TrafficData

from .dataset import MTGNNDataset

MTGNNSample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
MTGNNTrainBatch = Tuple[torch.Tensor, torch.Tensor]
MTGNNTestBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _to_channel_first(x_stacked: torch.Tensor) -> torch.Tensor:
    return x_stacked.permute(0, 3, 2, 1).contiguous()


def collate_mtgnn_train(batch: list[MTGNNSample]) -> MTGNNTrainBatch:
    x_list, y_list, _ = zip(*batch)
    x_batch = _to_channel_first(torch.stack(x_list, dim=0))
    y_batch = torch.stack(y_list, dim=0)
    return x_batch, y_batch


def collate_mtgnn_test(batch: list[MTGNNSample]) -> MTGNNTestBatch:
    x_list, y_list, y_missing_list = zip(*batch)
    x_batch = _to_channel_first(torch.stack(x_list, dim=0))
    y_batch = torch.stack(y_list, dim=0)
    y_missing_batch = torch.stack(y_missing_list, dim=0)
    return x_batch, y_batch, y_missing_batch


class MTGNNDataModule(L.LightningDataModule):
    """MTGNN DataModule with separate train/test HDF files."""

    def __init__(
        self,
        training_data_path: str,
        test_data_path: str,
        test_missing_path: str,
        adj_mx_path: str,
        seq_len: int = 24,
        horizon: int = 24,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle_training: bool = False,
        train_val_split: float = 0.8,
        add_time_in_day: bool = True,
        add_day_in_week: bool = False,
    ) -> None:
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
        self._scaler: Optional[StandardScaler] = None

        self.training_dataset: Optional[MTGNNDataset] = None
        self.validation_dataset: Optional[MTGNNDataset] = None
        self.test_dataset: Optional[MTGNNDataset] = None

    @property
    def scaler(self) -> Optional[StandardScaler]:
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
        if len(ref_data) == 0:
            raise ValueError("No valid data available to fit scaler.")

        self._scaler = StandardScaler()
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
    ) -> None:
        self.adj_mx_raw = AdjacencyMatrix.import_from_pickle(str(self.adj_mx_path))
        ordered_sensor_ids = self.adj_mx_raw.sensor_ids

        if stage in ["fit", "validate", None]:
            train_df, val_df = self._load_training_data(ordered_sensor_ids)

            self._prepare_scaler(train_df.values)
            train_df_scaled = self._apply_scaling(train_df)
            val_df_scaled = self._apply_scaling(val_df)

            self.training_dataset = MTGNNDataset(
                train_df_scaled,
                seq_len=self.seq_len,
                horizon=self.horizon,
                add_time_in_day=self.add_time_in_day,
                add_day_in_week=self.add_day_in_week,
                missing_mask=None,
            )
            self.validation_dataset = MTGNNDataset(
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
            self.test_dataset = MTGNNDataset(
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
            pin_memory=should_pin_memory(),
            collate_fn=collate_mtgnn_train,
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
            pin_memory=should_pin_memory(),
            collate_fn=collate_mtgnn_train,
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
            pin_memory=should_pin_memory(),
            collate_fn=collate_mtgnn_test,
        )
