"""BigST DataModule."""

from pathlib import Path
from typing import Literal, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from traffic_imc_baseline.training.runtime import should_pin_memory

from traffic_imc_dataset.components import MissingMasks
from traffic_imc_dataset.components.adj_mx import AdjacencyMatrix
from traffic_imc_dataset.components.traffic_imc.traffic_data import TrafficData

from .dataset import BigSTDataset

BigSTSample = Tuple[torch.Tensor, ...]
BigSTTrainBatch = Tuple[torch.Tensor, torch.Tensor]
BigSTTestBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def collate_bigst_train(batch: list[BigSTSample]) -> BigSTTrainBatch:
    x_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]
    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    return x_batch, y_batch


def collate_bigst_test(batch: list[BigSTSample]) -> BigSTTestBatch:
    x_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]
    y_missing_list = [item[2] for item in batch]
    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    y_missing_batch = torch.stack(y_missing_list, dim=0)
    return x_batch, y_batch, y_missing_batch


def asym_adj(adj: np.ndarray) -> np.ndarray:
    adj_sp = sp.coo_matrix(adj)
    rowsum = np.array(adj_sp.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    normalized = d_mat.dot(adj_sp)
    return np.asarray(normalized.astype(np.float32).todense())


def build_bigst_supports(adj_mx: np.ndarray) -> list[np.ndarray]:
    return [asym_adj(adj_mx)]


class BigSTDataModule(L.LightningDataModule):
    """BigST DataModule with Traffic-IMC HDF inputs."""

    def __init__(
        self,
        training_data_path: str,
        test_data_path: str,
        test_missing_path: str,
        adj_mx_path: str,
        input_length: int = 24,
        output_length: int = 24,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle_training: bool = False,
        train_val_split: float = 0.8,
    ) -> None:
        super().__init__()
        self.training_data_path = Path(training_data_path)
        self.test_data_path = Path(test_data_path)
        self.test_missing_path = Path(test_missing_path)
        self.adj_mx_path = Path(adj_mx_path)

        self.input_length = input_length
        self.output_length = output_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training
        self.train_val_split = train_val_split

        self.adj_mx_raw: Optional[AdjacencyMatrix] = None
        self._scaler: Optional[StandardScaler] = None

        self.training_dataset: Optional[BigSTDataset] = None
        self.validation_dataset: Optional[BigSTDataset] = None
        self.test_dataset: Optional[BigSTDataset] = None

    @property
    def scaler(self) -> Optional[StandardScaler]:
        return self._scaler

    @property
    def adj_mx(self) -> np.ndarray:
        if self.adj_mx_raw is None:
            raise ValueError("DataModule not setup. Call setup() first.")
        return self.adj_mx_raw.adj_mx

    @property
    def sensor_ids(self) -> list[str]:
        if self.adj_mx_raw is None:
            raise ValueError("DataModule not setup. Call setup() first.")
        return self.adj_mx_raw.sensor_ids

    @property
    def num_nodes(self) -> int:
        return len(self.sensor_ids)

    @property
    def input_dim(self) -> int:
        return 3

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

    def _apply_scaling(self, *datasets: BigSTDataset) -> None:
        if self._scaler is None:
            raise ValueError("Scaler must be fitted before applying scaling.")

        for dataset in datasets:
            dataset.apply_scaler(self._scaler)

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
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raw = TrafficData.import_from_hdf(str(self.test_data_path))
        raw_df = raw.data[ordered_sensor_ids]

        missing_masks = MissingMasks.import_from_hdf(str(self.test_missing_path))
        missing_mask_df = missing_masks.data[ordered_sensor_ids]
        missing_mask_aligned = missing_mask_df.reindex(
            index=raw_df.index,
            columns=raw_df.columns,
            fill_value=False,
        )
        missing_mask_aligned = missing_mask_aligned.fillna(False).astype(bool)
        return raw_df, missing_mask_aligned

    def setup(
        self,
        stage: Optional[Literal["fit", "validate", "test", "predict"]] = None,
    ) -> None:
        self.adj_mx_raw = AdjacencyMatrix.import_from_pickle(str(self.adj_mx_path))
        ordered_sensor_ids = self.adj_mx_raw.sensor_ids

        if stage in ["fit", "validate", None]:
            train_df, val_df = self._load_training_data(ordered_sensor_ids)
            self._prepare_scaler(train_df.values)

            self.training_dataset = BigSTDataset(
                train_df,
                input_length=self.input_length,
                output_length=self.output_length,
                missing_mask=None,
            )
            self.validation_dataset = BigSTDataset(
                val_df,
                input_length=self.input_length,
                output_length=self.output_length,
                missing_mask=None,
            )
            self._apply_scaling(self.training_dataset, self.validation_dataset)

        if stage in ["test", None]:
            test_df, test_missing_mask = self._load_test_data(ordered_sensor_ids)

            if self._scaler is None:
                train_df, _ = self._load_training_data(ordered_sensor_ids)
                self._prepare_scaler(train_df.values)

            self.test_dataset = BigSTDataset(
                test_df,
                input_length=self.input_length,
                output_length=self.output_length,
                missing_mask=test_missing_mask,
            )
            self._apply_scaling(self.test_dataset)

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
            collate_fn=collate_bigst_train,
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
            collate_fn=collate_bigst_train,
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
            collate_fn=collate_bigst_test,
        )
