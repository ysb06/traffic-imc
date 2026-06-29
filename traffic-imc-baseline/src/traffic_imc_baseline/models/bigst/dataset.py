"""BigST dataset."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class BigSTDataset(Dataset):
    """PyTorch Dataset for BigST.

    Each sample returns:
    - x: (input_length, num_nodes, 3) where features are [traffic, tod, dow]
    - y: (output_length, num_nodes, 1)
    - y_is_missing: (output_length, num_nodes)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        input_length: int = 24,
        output_length: int = 24,
        missing_mask: Optional[pd.DataFrame | np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.num_nodes = data.shape[1]
        self.sensor_ids = list(data.columns)

        self.data_values = data.values.astype(np.float32)
        self.scaled_data = self.data_values.copy()

        if missing_mask is None:
            self.missing_mask = None
        else:
            if isinstance(missing_mask, pd.DataFrame):
                mask_values = missing_mask.values.astype(bool)
            else:
                mask_values = missing_mask.astype(bool)
            if mask_values.shape != self.data_values.shape:
                raise ValueError(
                    "missing_mask shape must match data shape: "
                    f"{mask_values.shape} != {self.data_values.shape}"
                )
            self.missing_mask = mask_values

        self.tod = self._extract_tod(data.index)
        self.dow = self._extract_dow(data.index)
        self.valid_indices = self._compute_valid_indices()

        if len(self.valid_indices) == 0:
            raise ValueError(
                "No valid BigST samples generated. "
                "Check data length and NaN distribution."
            )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        t = int(self.valid_indices[idx])

        x = np.zeros((self.input_length, self.num_nodes, 3), dtype=np.float32)
        x[:, :, 0] = self.scaled_data[t : t + self.input_length, :]
        x[:, :, 1] = self.tod[t : t + self.input_length, np.newaxis]
        x[:, :, 2] = self.dow[t : t + self.input_length, np.newaxis]

        y_start = t + self.input_length
        y_end = y_start + self.output_length
        y = np.zeros((self.output_length, self.num_nodes, 1), dtype=np.float32)
        y[:, :, 0] = self.scaled_data[y_start:y_end, :]

        if self.missing_mask is None:
            y_missing = np.zeros((self.output_length, self.num_nodes), dtype=bool)
        else:
            y_missing = self.missing_mask[y_start:y_end, :]

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(y_missing).to(torch.bool),
        )

    def apply_scaler(self, scaler: StandardScaler) -> None:
        flat_data = self.data_values.reshape(-1, 1)
        scaled_flat = scaler.transform(flat_data)
        self.scaled_data = scaled_flat.reshape(self.data_values.shape)

    def _compute_valid_indices(self) -> np.ndarray:
        total_window = self.input_length + self.output_length
        num_possible = len(self.data_values) - total_window + 1
        if num_possible <= 0:
            return np.array([], dtype=np.int64)

        valid_indices: list[int] = []
        for i in range(num_possible):
            window = self.data_values[i : i + total_window, :]
            if not np.any(np.isnan(window)):
                valid_indices.append(i)
        return np.array(valid_indices, dtype=np.int64)

    def _extract_tod(self, index: pd.DatetimeIndex) -> np.ndarray:
        seconds = index.hour * 3600 + index.minute * 60 + index.second
        return np.asarray(seconds / (24 * 60 * 60), dtype=np.float32)

    def _extract_dow(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek.values.astype(np.float32)
