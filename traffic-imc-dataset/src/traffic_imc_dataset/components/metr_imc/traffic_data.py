from datetime import datetime, timedelta
import logging
import os
from typing import List, Optional, Set, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DateOffset
from tqdm import tqdm

from .interpolation import Interpolator
from .outlier import OutlierProcessor

logger = logging.getLogger(__name__)


class TrafficData:
    @staticmethod
    def import_from_pickle(
        filepath: str, dtype: Optional[Union[str, type]] = float
    ) -> "TrafficData":
        raw: pd.DataFrame = pd.read_pickle(filepath)

        temp = {road_id: {} for road_id in raw["linkID"].unique()}

        # Group by date
        traffic_date_group = raw.groupby("statDate")
        # Load date by date
        logger.info(f"Loading data from pickle file ({filepath})...")
        with tqdm(total=len(traffic_date_group)) as pbar:
            for date, group in traffic_date_group:
                # Load hour by hour
                for n in range(24):
                    pbar.set_description(f"{date} {n}h", refresh=True)

                    row_col_key = "hour{:02d}".format(
                        n
                    )  # Column name to read (e.g., hour00, hour01, ...)
                    row_index = datetime.strptime(date, "%Y-%m-%d") + timedelta(
                        hours=n
                    )  # Datetime index to use
                    for _, row in group.iterrows():
                        temp[row["linkID"]][row_index] = row[row_col_key]

                pbar.update(1)

        data = pd.DataFrame(temp)
        data.sort_index(inplace=True)
        notna_road_ids = data.columns[data.notna().any()].tolist()
        data = data.loc[:, notna_road_ids]

        return TrafficData(data, dtype=dtype)

    @staticmethod
    def import_from_hdf(
        filepath: Union[str, Path],
        key: Optional[str] = None,
        dtype: Optional[Union[str, type]] = float,
    ) -> "TrafficData":
        logger.info(f"Loading data from {filepath}...")
        if key is not None:
            data = pd.read_hdf(filepath, key=key)
        else:
            data = pd.read_hdf(filepath)

        return TrafficData(data, dtype=dtype, path=filepath)

    def __init__(
        self,
        raw: pd.DataFrame,
        dtype: Optional[Union[str, type]] = None,
        freq: Optional[str] = "h",
        path: Optional[str] = None,
    ) -> None:
        raw.sort_index(inplace=True)
        if dtype is not None:
            raw = self._as_type(raw, dtype)
        if freq is None:
            freq = pd.infer_freq(raw.index)
            logger.info(f"Inferred frequency: {freq}")
        raw = raw.asfreq(freq)
        self._verify_data(raw)
        self.data = raw

        self.path = path

    def _as_type(self, data: pd.DataFrame, dtype: Union[str, type]) -> pd.DataFrame:
        if np.issubdtype(np.dtype(dtype), np.integer):
            if not np.issubdtype(data.dtypes.unique()[0], np.integer):
                logger.warning("Rounding data to integer")
            return data.round().astype(dtype)
        else:
            return data.astype(dtype)

    def _verify_data(self, raw: pd.DataFrame) -> None:
        if not raw.index.is_monotonic_increasing:
            raise ValueError("Data not sorted by time")

    def select_sensors(self, sensor_ids: List[str]) -> None:
        """Select sensors by their IDs.

        Args:
            sensor_ids: List of sensor IDs to select.
        """
        available_sensor_ids = set(self.data.columns)
        selected_sensor_ids = [sid for sid in sensor_ids if sid in available_sensor_ids]
        missing_sensor_ids = set(sensor_ids) - set(selected_sensor_ids)

        if missing_sensor_ids:
            logger.warning(
                f"The {len(missing_sensor_ids)} sensor IDs are not found in the data"
            )
            if len(missing_sensor_ids) <= 10:
                logger.warning(f"Missing sensor IDs: {missing_sensor_ids}")

        self.data = self.data[selected_sensor_ids]

    def split(self, split_date: pd.Timestamp) -> Tuple["TrafficData", "TrafficData"]:
        """Split the data into two TrafficData instances at the specified date.

        Args:
            split_date: The date to split the data on.

        Returns:
            A tuple containing two TrafficData instances: (data_before_split, data_after_split)
        """
        data_before = self.data[self.data.index < split_date]
        data_after = self.data[self.data.index >= split_date]

        return TrafficData(data_before), TrafficData(data_after)

    def to_hdf(self, filepath: str, key: str = "data") -> None:
        logger.info(f"Saving data to {filepath}...")
        self.data.to_hdf(filepath, key=key)
        logger.info(f"Saving Complete...{self.data.shape}")

    def to_excel(self, filepath: str) -> None:
        logger.info(f"Saving data to {filepath}...")
        self.data.to_excel(filepath)
        logger.info(f"Saving Complete...{self.data.shape}")


def get_raw(path: str) -> TrafficData:
    """
    Load raw traffic data from the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    return TrafficData.import_from_hdf(path)


class MissingMasks:
    @staticmethod
    def import_from_traffic_data(traffic_data: TrafficData) -> "MissingMasks":
        missing_mask = traffic_data.data.isna()
        return MissingMasks(missing_mask)
    
    @staticmethod
    def import_from_traffic_data_frame(traffic_data_df: pd.DataFrame) -> "MissingMasks":
        missing_mask = traffic_data_df.isna()
        return MissingMasks(missing_mask)

    @staticmethod
    def import_from_hdf(filepath: Union[str, Path], key: str = "data") -> "MissingMasks":
        logger.info(f"Loading missing mask from {filepath}...")
        missing_mask = pd.read_hdf(filepath, key=key)
        if type(missing_mask) is not pd.DataFrame:
            raise ValueError("Missing mask data is not a DataFrame")
        return MissingMasks(missing_mask)

    def __init__(self, missing_mask: pd.DataFrame) -> None:
        self.data = missing_mask

    def to_hdf(self, filepath: str, key: str = "data") -> None:
        logger.info(f"Saving missing mask to {filepath}...")
        self.data.to_hdf(filepath, key=key)
        logger.info(f"Saving Complete...{self.data.shape}")
