from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

TrafficMultiSensorDataType = Tuple[np.ndarray, np.ndarray, np.ndarray]
TrafficMultiSensorTestDataType = Tuple[np.ndarray, np.ndarray, np.ndarray, int]


class TrafficMultiSensorDataset(Dataset):
    """Multi-sensor dataset for LSTM.

    Each sample returns exactly: x, y, y_is_missing.
    - x: (seq_length, 1)
    - y: (pred_length,)
    - y_is_missing: (pred_length,)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        seq_length: int = 24,
        pred_length: int = 24,
        allow_nan: bool = False,
        missing_mask: Optional[pd.DataFrame] = None,
        include_sensor_idx: bool = False,
    ):
        super().__init__()
        if len(data) < seq_length + pred_length:
            raise ValueError(
                "Data length should be at least seq_length + pred_length"
            )

        self.data_df = data
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.allow_nan = allow_nan
        self.sensor_names = list(data.columns)
        self.missing_mask = missing_mask
        self.include_sensor_idx = include_sensor_idx

        self._raw_data: dict[str, np.ndarray] = {
            sensor_name: data[sensor_name].to_numpy().reshape(-1, 1)
            for sensor_name in self.sensor_names
        }
        self._scaled_data: dict[str, np.ndarray] = {
            sensor_name: values.copy() for sensor_name, values in self._raw_data.items()
        }

        self.index_mapping = self._build_index_mapping()

    def _build_index_mapping(self) -> list[tuple[str, int]]:
        index_mapping: list[tuple[str, int]] = []
        for sensor_name in self.sensor_names:
            values = self._raw_data[sensor_name]
            for cursor in self._valid_cursors(values):
                index_mapping.append((sensor_name, cursor))
        return index_mapping

    def _valid_cursors(self, values: np.ndarray) -> list[int]:
        time_len = len(values)
        num_possible = time_len - self.seq_length - self.pred_length + 1
        if num_possible <= 0:
            return []

        all_i = np.arange(num_possible)
        if self.allow_nan:
            return all_i.tolist()

        isnan_arr = np.isnan(values).reshape(-1)
        cumsum = np.cumsum(isnan_arr, dtype=np.int32)
        cumsum = np.insert(cumsum, 0, 0)

        x_nan_count = cumsum[all_i + self.seq_length] - cumsum[all_i]
        target_start = all_i + self.seq_length
        target_end = target_start + self.pred_length
        y_nan_count = cumsum[target_end] - cumsum[target_start]
        valid_mask = (x_nan_count == 0) & (y_nan_count == 0)

        return all_i[valid_mask].tolist()

    def __len__(self) -> int:
        return len(self.index_mapping)

    def __getitem__(
        self,
        index: int,
    ) -> TrafficMultiSensorDataType | TrafficMultiSensorTestDataType:
        sensor_name, cursor = self.index_mapping[index]
        sensor_data = self._scaled_data[sensor_name]

        x = sensor_data[cursor : cursor + self.seq_length]
        y = sensor_data[
            cursor + self.seq_length : cursor + self.seq_length + self.pred_length
        ].reshape(-1)

        y_is_missing = np.zeros(self.pred_length, dtype=bool)
        if self.missing_mask is not None:
            target_index = self.data_df.index[
                cursor + self.seq_length : cursor + self.seq_length + self.pred_length
            ]
            y_is_missing = self.missing_mask.loc[
                target_index,
                sensor_name,
            ].to_numpy(dtype=bool)

        if self.include_sensor_idx:
            return x, y, y_is_missing, self.sensor_names.index(sensor_name)

        return x, y, y_is_missing

    def apply_scaler(self, scaler: StandardScaler):
        for sensor_name, raw_values in self._raw_data.items():
            self._scaled_data[sensor_name] = scaler.transform(raw_values)
