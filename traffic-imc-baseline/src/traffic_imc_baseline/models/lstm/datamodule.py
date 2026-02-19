import logging
from typing import Callable, List, Literal, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from traffic_imc_dataset.components import MissingMasks
from traffic_imc_dataset.components.metr_imc.traffic_data import get_raw

from .dataset import TrafficMultiSensorDataType, TrafficMultiSensorDataset

logger = logging.getLogger(__name__)

TrainCollateInput = List[TrafficMultiSensorDataType]
TrainBatchType = Tuple[Tensor, Tensor]
TestBatchType = Tuple[Tensor, Tensor, List[bool]]


def collate_lstm_train(batch: TrainCollateInput) -> TrainBatchType:
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]

    xs_t = torch.stack([torch.from_numpy(x).float() for x in xs], dim=0)
    ys_t = torch.stack([torch.from_numpy(y).float() for y in ys], dim=0)
    return xs_t, ys_t


def collate_lstm_test(batch: TrainCollateInput) -> TestBatchType:
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    y_is_missing_list = [item[2] for item in batch]

    xs_t = torch.stack([torch.from_numpy(x).float() for x in xs], dim=0)
    ys_t = torch.stack([torch.from_numpy(y).float() for y in ys], dim=0)
    return xs_t, ys_t, y_is_missing_list


class LSTMDataModule(L.LightningDataModule):
    """LSTM 학습용 다중 센서 DataModule.

    - Train/Validation: (x, y) 배치 반환
    - Test: (x, y, y_is_missing_list) 배치 반환
    """

    def __init__(
        self,
        training_dataset_path: str,
        test_dataset_path: str,
        test_missing_path: str,
        train_val_split: float = 0.8,
        seq_length: int = 24,
        batch_size: int = 512,
        num_workers: int = 0,
        shuffle_training: bool = True,
        scale_method: Optional[Literal["normal", "strict", "none"]] = "normal",
        allow_nan: bool = False,
        collate_fn: Callable[[TrainCollateInput], TrainBatchType] = collate_lstm_train,
        test_collate_fn: Callable[[TrainCollateInput], TestBatchType] = collate_lstm_test,
    ):
        super().__init__()
        self.training_dataset_path = training_dataset_path
        self.test_dataset_path = test_dataset_path
        self.test_missing_path = test_missing_path
        self.train_val_split = train_val_split

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training
        self.scale_method = scale_method
        self.allow_nan = allow_nan

        self.collate_fn = collate_fn
        self.test_collate_fn = test_collate_fn

        self._scaler: Optional[MinMaxScaler] = None
        self.train_dataset: Optional[TrafficMultiSensorDataset] = None
        self.val_dataset: Optional[TrafficMultiSensorDataset] = None
        self.test_dataset: Optional[TrafficMultiSensorDataset] = None

    @property
    def scaler(self) -> Optional[MinMaxScaler]:
        if self.scale_method in (None, "none"):
            return None

        if self._scaler is None:
            logger.info("Scaler not found. Creating scaler from training data...")
            train_df, _ = self._load_training_data()
            self._prepare_scaler(train_df)
        return self._scaler

    def _load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Loading training data from {self.training_dataset_path}")
        raw = get_raw(self.training_dataset_path)
        raw_df = raw.data

        split_idx = int(len(raw_df) * self.train_val_split)
        train_df = raw_df.iloc[:split_idx]
        val_df = raw_df.iloc[split_idx:]

        logger.info(
            f"Training split complete - Train: {len(train_df)} rows, "
            f"Val: {len(val_df)} rows"
        )
        return train_df, val_df

    def _load_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Loading test data from {self.test_dataset_path}")
        raw = get_raw(self.test_dataset_path)
        test_df = raw.data

        logger.info(f"Loading test missing mask from {self.test_missing_path}")
        missing_masks = MissingMasks.import_from_hdf(self.test_missing_path)
        missing_mask = missing_masks.data

        missing_mask = missing_mask.reindex(index=test_df.index, columns=test_df.columns)
        missing_mask = missing_mask.fillna(False)
        return test_df, missing_mask

    def _prepare_scaler(self, train_df: pd.DataFrame) -> None:
        if self.scale_method in (None, "none"):
            self._scaler = None
            return

        if self.scale_method == "strict":
            temp_dataset = TrafficMultiSensorDataset(
                train_df,
                seq_length=self.seq_length,
                allow_nan=self.allow_nan,
            )
            ref_data = self._get_strict_scaler_data(temp_dataset)
        else:
            ref_data = train_df.to_numpy().reshape(-1, 1)
            ref_data = ref_data[~np.isnan(ref_data).any(axis=1)]

        if len(ref_data) == 0:
            raise ValueError("No valid data available to fit scaler.")

        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler.fit(ref_data)

    def _get_strict_scaler_data(self, dataset: TrafficMultiSensorDataset) -> np.ndarray:
        data_list: list[np.ndarray] = []
        for i in tqdm(range(len(dataset)), desc="Extracting strict scaler data"):
            x, y, _ = dataset[i]
            data_list.append(x)
            data_list.append(y.reshape(1, 1))
        return np.concatenate(data_list, axis=0).reshape(-1, 1)

    def _apply_scaling(self, *datasets: TrafficMultiSensorDataset) -> None:
        if self.scale_method in (None, "none"):
            return

        if self._scaler is None:
            raise ValueError("Scaler is not initialized.")

        for dataset in datasets:
            dataset.apply_scaler(self._scaler)

    def setup(
        self, stage: Optional[Literal["fit", "validate", "test", "predict"]] = None
    ) -> None:
        logger.info(f"Setting up data for stage: {stage}")

        train_df, val_df = self._load_training_data()
        test_df, test_missing_mask = self._load_test_data()

        self.train_dataset = TrafficMultiSensorDataset(
            train_df,
            seq_length=self.seq_length,
            allow_nan=self.allow_nan,
        )
        self.val_dataset = TrafficMultiSensorDataset(
            val_df,
            seq_length=self.seq_length,
            allow_nan=self.allow_nan,
        )
        self.test_dataset = TrafficMultiSensorDataset(
            test_df,
            seq_length=self.seq_length,
            allow_nan=self.allow_nan,
            missing_mask=test_missing_mask,
        )

        self._prepare_scaler(train_df)
        self._apply_scaling(self.train_dataset, self.val_dataset, self.test_dataset)

        logger.info(
            f"Setup complete - Train: {len(self.train_dataset)}, "
            f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}"
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized. Call setup() first.")

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
            raise ValueError("Validation dataset is not initialized. Call setup() first.")

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
            raise ValueError("Test dataset is not initialized. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.test_collate_fn,
        )
