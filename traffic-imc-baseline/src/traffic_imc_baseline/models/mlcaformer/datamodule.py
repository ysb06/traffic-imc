"""MLCAFormer DataModule."""

from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from traffic_imc_dataset.components import MissingMasks
from traffic_imc_dataset.components.metr_imc.traffic_data import TrafficData

from .dataset import MLCAFormerDataset

MLCAFormerSample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
MLCAFormerTrainBatch = Tuple[torch.Tensor, torch.Tensor]
MLCAFormerTestBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def collate_mlcaformer_train(batch: list[MLCAFormerSample]) -> MLCAFormerTrainBatch:
    x_list, y_list, _ = zip(*batch)
    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    return x_batch, y_batch


def collate_mlcaformer_test(batch: list[MLCAFormerSample]) -> MLCAFormerTestBatch:
    x_list, y_list, y_missing_list = zip(*batch)
    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    y_missing_batch = torch.stack(y_missing_list, dim=0)
    return x_batch, y_batch, y_missing_batch


class MLCAFormerDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for MLCAFormer."""

    def __init__(
        self,
        training_dataset_path: str,
        test_dataset_path: str,
        test_missing_path: str,
        train_val_split: float = 0.8,
        in_steps: int = 24,
        out_steps: int = 1,
        steps_per_day: int = 24,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle_training: bool = True,
        scale_method: Optional[Literal["normal", "strict", "none"]] = "normal",
        collate_fn: Callable[[list[MLCAFormerSample]], MLCAFormerTrainBatch] = collate_mlcaformer_train,
        test_collate_fn: Callable[[list[MLCAFormerSample]], MLCAFormerTestBatch] = collate_mlcaformer_test,
    ):
        super().__init__()
        self.training_dataset_path = Path(training_dataset_path)
        self.test_dataset_path = Path(test_dataset_path)
        self.test_missing_path = Path(test_missing_path)

        self.train_val_split = train_val_split
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training
        self.scale_method = scale_method

        self.collate_fn = collate_fn
        self.test_collate_fn = test_collate_fn

        self._scaler: Optional[MinMaxScaler] = None

        self.train_dataset: Optional[MLCAFormerDataset] = None
        self.val_dataset: Optional[MLCAFormerDataset] = None
        self.test_dataset: Optional[MLCAFormerDataset] = None

        self.test_missing_mask: Optional[pd.DataFrame] = None
        self.num_nodes: Optional[int] = None
        self.sensor_ids: Optional[list[str]] = None

    @property
    def scaler(self) -> Optional[MinMaxScaler]:
        if self.scale_method in (None, "none"):
            return None

        if self._scaler is None:
            train_df, _ = self._load_training_data()
            self._prepare_scaler(train_df)
        return self._scaler

    def _load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raw = TrafficData.import_from_hdf(str(self.training_dataset_path))
        raw_df = raw.data

        total_rows = len(raw_df)
        split_idx = int(total_rows * self.train_val_split)

        train_df = raw_df.iloc[:split_idx]
        val_df = raw_df.iloc[split_idx:]
        return train_df, val_df

    def _load_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raw = TrafficData.import_from_hdf(str(self.test_dataset_path))
        raw_df = raw.data

        missing_masks = MissingMasks.import_from_hdf(str(self.test_missing_path))
        missing_mask_df = missing_masks.data

        missing_mask_aligned = missing_mask_df.reindex(
            index=raw_df.index,
            columns=raw_df.columns,
            fill_value=False,
        )
        return raw_df, missing_mask_aligned

    def _get_strict_scaler_data(self, dataset: MLCAFormerDataset) -> np.ndarray:
        values: list[np.ndarray] = []
        for i in tqdm(range(len(dataset)), desc="Extracting strict scaler data"):
            x, y, _ = dataset[i]
            values.append(x[:, :, 0].numpy().reshape(-1))
            values.append(y[:, :, 0].numpy().reshape(-1))
        return np.concatenate(values).reshape(-1, 1)

    def _prepare_scaler(self, train_df: pd.DataFrame) -> None:
        if self.scale_method in (None, "none"):
            self._scaler = None
            return

        if self.scale_method == "strict":
            temp_dataset = MLCAFormerDataset(
                train_df,
                in_steps=self.in_steps,
                out_steps=self.out_steps,
                steps_per_day=self.steps_per_day,
                missing_mask=None,
            )
            ref_data = self._get_strict_scaler_data(temp_dataset)
        else:
            ref_data = train_df.values.reshape(-1, 1)
            ref_data = ref_data[~np.isnan(ref_data).any(axis=1)]

        if len(ref_data) == 0:
            raise ValueError("No valid data available to fit scaler.")

        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler.fit(ref_data)

    def _apply_scaling(self, *datasets: MLCAFormerDataset) -> None:
        if self.scale_method in (None, "none"):
            return

        if self._scaler is None:
            raise ValueError("Scaler is not fitted. Call _prepare_scaler first.")

        for dataset in datasets:
            dataset.apply_scaler(self._scaler)

    def setup(
        self,
        stage: Optional[Literal["fit", "validate", "test", "predict"]] = None,
    ) -> None:
        train_df, val_df = self._load_training_data()
        test_df, test_missing_mask = self._load_test_data()
        self.test_missing_mask = test_missing_mask

        self.num_nodes = train_df.shape[1]
        self.sensor_ids = list(train_df.columns)

        self.train_dataset = MLCAFormerDataset(
            train_df,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
            missing_mask=None,
        )
        self.val_dataset = MLCAFormerDataset(
            val_df,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
            missing_mask=None,
        )
        self.test_dataset = MLCAFormerDataset(
            test_df,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
            missing_mask=test_missing_mask,
        )

        self._prepare_scaler(train_df)
        self._apply_scaling(self.train_dataset, self.val_dataset, self.test_dataset)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Call setup() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.test_collate_fn,
        )
