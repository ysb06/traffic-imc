"""AGCRN dataset."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class AGCRNDataset(Dataset):
    """PyTorch Dataset for AGCRN.

    Each sample returns exactly:
    - x: (in_steps, n_vertex, 1)
    - y: (out_steps, n_vertex, 1)
    - y_is_missing: (out_steps, n_vertex)

    If ``missing_mask`` is not provided, ``y_is_missing`` is all False.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        in_steps: int = 24,
        out_steps: int = 1,
        missing_mask: Optional[pd.DataFrame | np.ndarray] = None,
    ):
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.n_vertex = data.shape[1]
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

        self.valid_indices = self._compute_valid_indices()
        if len(self.valid_indices) == 0:
            raise ValueError(
                "No valid AGCRN samples generated. "
                "Check data length and NaN distribution."
            )

    def _compute_valid_indices(self) -> np.ndarray:
        total_window = self.in_steps + self.out_steps
        num_possible = len(self.data_values) - total_window + 1
        if num_possible <= 0:
            return np.array([], dtype=np.int64)

        valid_indices: list[int] = []
        for i in range(num_possible):
            window = self.data_values[i : i + total_window, :]
            if not np.any(np.isnan(window)):
                valid_indices.append(i)

        return np.array(valid_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.valid_indices[idx])

        x = np.zeros((self.in_steps, self.n_vertex, 1), dtype=np.float32)
        x[:, :, 0] = self.scaled_data[t : t + self.in_steps, :]

        y = np.zeros((self.out_steps, self.n_vertex, 1), dtype=np.float32)
        y[:, :, 0] = self.scaled_data[
            t + self.in_steps : t + self.in_steps + self.out_steps,
            :,
        ]

        if self.missing_mask is None:
            y_missing = np.zeros((self.out_steps, self.n_vertex), dtype=bool)
        else:
            y_missing = self.missing_mask[
                t + self.in_steps : t + self.in_steps + self.out_steps,
                :,
            ]

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(y_missing).to(torch.bool),
        )

    def apply_scaler(self, scaler: MinMaxScaler) -> None:
        flat_data = self.data_values.reshape(-1, 1)
        scaled_flat = scaler.transform(flat_data)
        self.scaled_data = scaled_flat.reshape(self.data_values.shape)
