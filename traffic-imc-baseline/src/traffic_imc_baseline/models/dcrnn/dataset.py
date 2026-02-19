from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DCRNNDataset(Dataset):
    """PyTorch Dataset for DCRNN model.

    Each sample returns exactly:
    - x: (seq_len, num_nodes, input_dim)
    - y: (horizon, num_nodes, output_dim)
    - y_is_missing: (horizon, num_nodes)

    When ``missing_mask`` is not provided, ``y_is_missing`` is all False.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int = 24,
        horizon: int = 1,
        add_time_in_day: bool = True,
        add_day_in_week: bool = False,
        missing_mask: Optional[np.ndarray] = None,
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        if missing_mask is not None and missing_mask.shape != data.shape:
            raise ValueError(
                "missing_mask shape must match data shape: "
                f"{missing_mask.shape} != {data.shape}"
            )
        self.missing_mask = missing_mask

        self.input_dim = 1
        if self.add_time_in_day:
            self.input_dim += 1
        if self.add_day_in_week:
            self.input_dim += 7

        self.output_dim = 1

        self.x, self.y, self.y_missing = self._data_transform(data)

        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.y_missing[idx]

    def _data_transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples, num_nodes = df.shape

        traffic_data = np.expand_dims(df.values, axis=-1)
        data_list = [traffic_data]

        if self.add_time_in_day:
            time_ind = (
                df.index.values - df.index.values.astype("datetime64[D]")
            ) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind.reshape(-1, 1, 1), [1, num_nodes, 1])
            data_list.append(time_in_day)

        if self.add_day_in_week:
            day_in_week = np.zeros((num_samples, num_nodes, 7), dtype=np.float32)
            day_indices = pd.to_datetime(df.index).dayofweek.values
            for t in range(num_samples):
                day_in_week[t, :, day_indices[t]] = 1.0
            data_list.append(day_in_week)

        data = np.concatenate(data_list, axis=-1).astype(np.float32)

        x_offsets = np.arange(-self.seq_len + 1, 1)
        y_offsets = np.arange(1, self.horizon + 1)

        min_t = abs(min(x_offsets))
        max_t = num_samples - max(y_offsets)

        x_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        missing_list: list[np.ndarray] = []

        for t in range(min_t, max_t):
            x_list.append(data[t + x_offsets, :, :])
            y_list.append(data[t + y_offsets, :, :1])

            if self.missing_mask is None:
                missing_t = np.zeros((self.horizon, num_nodes), dtype=bool)
            else:
                missing_t = self.missing_mask[t + y_offsets, :]
            missing_list.append(missing_t)

        if not x_list:
            raise ValueError(
                "No samples generated. "
                "Check whether data length is greater than seq_len + horizon."
            )

        x = np.stack(x_list, axis=0)
        y = np.stack(y_list, axis=0)
        y_missing = np.stack(missing_list, axis=0)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(y_missing).to(torch.bool),
        )

    @property
    def num_nodes(self) -> int:
        return self.x.shape[2]
