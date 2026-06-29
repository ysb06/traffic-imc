from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class STGCNDatasetWithMissing(Dataset):
    """Lazy STGCN dataset with NaN-window filtering and missing-mask support.

    Each sample returns:
    - x: (1, n_his, n_vertex)
    - y: (n_pred, n_vertex)
    - y_is_missing: (n_pred, n_vertex)
    """

    def __init__(
        self,
        data: np.ndarray,
        n_his: int,
        n_pred: int,
        missing_mask: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        if n_his <= 0:
            raise ValueError("n_his must be a positive integer.")
        if n_pred <= 0:
            raise ValueError("n_pred must be a positive integer.")

        self.n_his = n_his
        self.n_pred = n_pred
        self.data_values = np.ascontiguousarray(data, dtype=np.float32)
        if self.data_values.ndim != 2:
            raise ValueError(
                "STGCN data must be a 2D array of shape "
                "(time_steps, n_vertex)."
            )
        self.scaled_data = self.data_values.copy()
        self.n_vertex = self.data_values.shape[1]

        if missing_mask is None:
            self.missing_mask = None
        else:
            mask_values = np.array(missing_mask, dtype=bool, copy=True)
            if mask_values.shape != self.data_values.shape:
                raise ValueError(
                    "missing_mask shape must match data shape: "
                    f"{mask_values.shape} != {self.data_values.shape}"
                )
            self.missing_mask = np.ascontiguousarray(mask_values)

        self.valid_indices = self._compute_valid_indices()
        if len(self.valid_indices) == 0:
            raise ValueError(
                "All samples contained NaNs and were filtered out. "
                "Check your data."
            )

    def _compute_valid_indices(self) -> np.ndarray:
        total_window = self.n_his + self.n_pred
        num_possible = len(self.data_values) - total_window + 1
        if num_possible <= 0:
            return np.array([], dtype=np.int64)

        row_has_nan = np.isnan(self.data_values).any(axis=1).astype(np.int64)
        row_nan_cumsum = np.concatenate(
            [np.array([0], dtype=np.int64), np.cumsum(row_has_nan)]
        )

        starts = np.arange(num_possible, dtype=np.int64)
        ends = starts + total_window
        window_nan_counts = row_nan_cumsum[ends] - row_nan_cumsum[starts]
        return starts[window_nan_counts == 0]

    def apply_scaler(self, scaler: StandardScaler) -> None:
        flat_data = self.data_values.reshape(-1, 1)
        scaled_flat = scaler.transform(flat_data)
        self.scaled_data = np.ascontiguousarray(
            scaled_flat.reshape(self.data_values.shape),
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.valid_indices[idx])
        y_start = t + self.n_his
        y_end = y_start + self.n_pred

        x = torch.from_numpy(self.scaled_data[t:y_start, :]).unsqueeze(0)
        y = torch.from_numpy(self.scaled_data[y_start:y_end, :])

        if self.missing_mask is None:
            y_missing = torch.zeros((self.n_pred, self.n_vertex), dtype=torch.bool)
        else:
            y_missing = torch.from_numpy(self.missing_mask[y_start:y_end, :])

        return x, y, y_missing
