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

from .dataset import STGCNDatasetWithMissing

STGCNBatchWithMissing = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
STGCNBatchSimple = Tuple[torch.Tensor, torch.Tensor]


def collate_stgcn_train(batch: list[STGCNBatchWithMissing]) -> STGCNBatchSimple:
    x_list, y_list, _ = zip(*batch)
    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    return x_batch, y_batch


def collate_stgcn_test(batch: list[STGCNBatchWithMissing]) -> STGCNBatchWithMissing:
    x_list, y_list, y_missing_list = zip(*batch)
    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    y_missing_batch = torch.stack(y_missing_list, dim=0)
    return x_batch, y_batch, y_missing_batch


class STGCNSplitDataModule(L.LightningDataModule):
    """STGCN DataModule with separate training and test dataset files."""

    def __init__(
        self,
        training_data_path: str,
        test_data_path: str,
        test_missing_path: str,
        adj_mx_path: str,
        n_his: int = 24,
        n_pred: int = 24,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle_training: bool = False,
        train_val_split: float = 0.8,
    ):
        super().__init__()
        self.training_data_path = Path(training_data_path)
        self.test_data_path = Path(test_data_path)
        self.test_missing_path = Path(test_missing_path)
        self.adj_mx_path = Path(adj_mx_path)

        self.n_his = n_his
        self.n_pred = n_pred
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training
        self.train_val_split = train_val_split

        self.adj_mx_raw: Optional[AdjacencyMatrix] = None
        self.training_dataset: Optional[STGCNDatasetWithMissing] = None
        self.validation_dataset: Optional[STGCNDatasetWithMissing] = None
        self.test_dataset: Optional[STGCNDatasetWithMissing] = None

        self._scaler: Optional[StandardScaler] = None

    @property
    def scaler(self) -> Optional[StandardScaler]:
        return self._scaler

    def _ordered_sensor_ids(self) -> list:
        if self.adj_mx_raw is None:
            self.adj_mx_raw = AdjacencyMatrix.import_from_pickle(self.adj_mx_path)
        return self.adj_mx_raw.sensor_ids

    def _prepare_scaler(self, train_data: np.ndarray) -> None:
        ref_data = train_data.reshape(-1, 1)
        ref_data = ref_data[~np.isnan(ref_data).any(axis=1)]
        if len(ref_data) == 0:
            raise ValueError("No valid data available to fit scaler.")

        self._scaler = StandardScaler()
        self._scaler.fit(ref_data)

    def _apply_scaling(self, *datasets: STGCNDatasetWithMissing) -> None:
        if self._scaler is None:
            raise ValueError("Scaler is not fitted. Call _prepare_scaler first.")

        for dataset in datasets:
            dataset.apply_scaler(self._scaler)

    def _load_training_data(
        self, ordered_sensor_ids: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raw = TrafficData.import_from_hdf(self.training_data_path)
        raw_df = raw.data[ordered_sensor_ids]

        total_rows = len(raw_df)
        split_idx = int(total_rows * self.train_val_split)

        train_df = raw_df.iloc[:split_idx]
        val_df = raw_df.iloc[split_idx:]

        return train_df, val_df

    def _load_test_data(
        self, ordered_sensor_ids: list
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        raw = TrafficData.import_from_hdf(self.test_data_path)
        raw_df = raw.data[ordered_sensor_ids]

        missing_masks = MissingMasks.import_from_hdf(self.test_missing_path)
        missing_mask_df = missing_masks.data[ordered_sensor_ids]

        missing_mask_aligned = missing_mask_df.reindex(
            index=raw_df.index, columns=raw_df.columns, fill_value=False
        )
        missing_mask_aligned = missing_mask_aligned.fillna(False).astype(bool)
        return raw_df, missing_mask_aligned.values

    def _setup_fit(self, ordered_sensor_ids: list) -> None:
        if (
            self.training_dataset is not None
            and self.validation_dataset is not None
            and self._scaler is not None
        ):
            return

        train_df, val_df = self._load_training_data(ordered_sensor_ids)
        training_data_array = train_df.values

        if self._scaler is None:
            self._prepare_scaler(training_data_array)

        datasets_to_scale = []
        if self.training_dataset is None:
            self.training_dataset = STGCNDatasetWithMissing(
                training_data_array,
                self.n_his,
                self.n_pred,
                missing_mask=None,
            )
            datasets_to_scale.append(self.training_dataset)

        if self.validation_dataset is None:
            self.validation_dataset = STGCNDatasetWithMissing(
                val_df.values,
                self.n_his,
                self.n_pred,
                missing_mask=None,
            )
            datasets_to_scale.append(self.validation_dataset)

        if datasets_to_scale:
            self._apply_scaling(*datasets_to_scale)

    def _ensure_scaler(self, ordered_sensor_ids: list) -> None:
        if self._scaler is not None:
            return

        train_df, _ = self._load_training_data(ordered_sensor_ids)
        self._prepare_scaler(train_df.values)

    def _setup_test(self, ordered_sensor_ids: list) -> None:
        if self.test_dataset is not None:
            return

        self._ensure_scaler(ordered_sensor_ids)
        test_df, test_missing_mask = self._load_test_data(ordered_sensor_ids)
        self.test_dataset = STGCNDatasetWithMissing(
            test_df.values,
            self.n_his,
            self.n_pred,
            missing_mask=test_missing_mask,
        )
        self._apply_scaling(self.test_dataset)

    def setup(
        self,
        stage: Optional[Literal["fit", "validate", "test", "predict"]] = None,
    ):
        ordered_sensor_ids = self._ordered_sensor_ids()

        if stage in ["fit", "validate", None]:
            self._setup_fit(ordered_sensor_ids)

        if stage in ["test", None]:
            self._setup_test(ordered_sensor_ids)

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
            collate_fn=collate_stgcn_train,
        )

    def val_dataloader(self) -> DataLoader:
        if self.validation_dataset is None:
            raise ValueError("Validation dataset is not initialized. Call setup() first.")
        
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=should_pin_memory(),
            collate_fn=collate_stgcn_train,
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
            collate_fn=collate_stgcn_test,
        )
