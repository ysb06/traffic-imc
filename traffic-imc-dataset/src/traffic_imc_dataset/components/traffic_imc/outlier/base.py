import logging
from typing import Dict

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

logger = logging.getLogger(__name__)


class OutlierProcessor:
    def __init__(self):
        self.name = self.__class__.__name__.lower().removesuffix("outlierprocessor")
        self.successed_list = []
        self.failed_list = []

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Processing {self.name}...")
        df_clean = self._process(df.copy())
        return df_clean


class RemovingWeirdZeroOutlierProcessor(OutlierProcessor):
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        def extend_nans_around_zeros(series: pd.Series) -> pd.Series:
            nan_indices = series[series.isna()].index

            for idx in nan_indices:
                idx_pos = series.index.get_loc(idx)

                i = idx_pos - 1
                while i >= 0 and series.iat[i] == 0:
                    series.iat[i] = np.nan
                    i -= 1

                i = idx_pos + 1
                while i < len(series) and series.iat[i] == 0:
                    series.iat[i] = np.nan
                    i += 1

            self.successed_list.append(series.name)
            return series

        return df.apply(extend_nans_around_zeros)


class SimpleAbsoluteOutlierProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 8000) -> None:
        super().__init__()
        self.threshold = threshold

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.mask(np.abs(df) > self.threshold)

        self.successed_list.extend(df_clean.columns)
        return df_clean


class TrafficCapacityAbsoluteOutlierProcessor(OutlierProcessor):
    def __init__(
        self,
        road_speed_limits: Dict[str, int],
        lane_counts: Dict[str, int],
        adjustment_rate: float = 1.0,  # Deprecated
    ) -> None:
        super().__init__()
        self.road_speed_limits = road_speed_limits
        self.lane_counts = lane_counts

    def _get_road_capacity(self, road_name: str) -> float:
        speed_limit = self.road_speed_limits[road_name]
        lane_count = self.lane_counts[road_name]
        # alpha = 10 * (100 - speed_limit)
        # if speed_limit > 100:
        #     alpha /= 2
        if speed_limit <= 80:
            max_capacity = 3000
        elif speed_limit <= 100:
            max_capacity = 3300
        else:
            max_capacity = 3450
        # Capacity values are restricted to handbook-level representative values.
        # Detailed level-of-service effects are not modeled in this approximation.
        # Signalized intersections are also assumed to be idealized.
        # return (2200 - alpha) * lane_count * self.adjustment_rate
        return max_capacity * lane_count

    def _process_road_data(self, series: pd.Series) -> pd.Series:
        road_name = series.name
        capacity = self._get_road_capacity(road_name)
        series_clean = series.mask(series > capacity)
        self.successed_list.append(road_name)  # Record as successful
        return series_clean

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self._process_road_data)


class HourlyInSensorZscoreOutlierProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 3.0) -> None:
        super().__init__()
        self.threshold = threshold

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        def zscore_within_group(x: pd.Series) -> pd.Series:
            return (x - x.mean()) / x.std()

        grouped: DataFrameGroupBy = df.groupby(df.index.hour)
        zscored: pd.DataFrame = grouped.transform(zscore_within_group)

        if zscored.isna().all().all():
            self.successed_list.extend(df.columns)
        else:
            self.failed_list.extend(df.columns)

        df_clean = df.mask(zscored.abs() > self.threshold)
        return df_clean


class MonthlyHourlyInSensorZscoreOutlierProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 3.0) -> None:
        super().__init__()
        self.threshold = threshold

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        def zscore_within_group(x: pd.Series) -> pd.Series:
            return (x - x.mean()) / x.std()

        # Group by both month and hour
        grouped: DataFrameGroupBy = df.groupby([df.index.month, df.index.hour])
        zscored: pd.DataFrame = grouped.transform(zscore_within_group)

        if zscored.isna().all().all():
            self.successed_list.extend(df.columns)
        else:
            self.failed_list.extend(df.columns)

        df_clean = df.mask(zscored.abs() > self.threshold)
        return df_clean


class HourlyZscoreOutlierProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 3.0) -> None:
        super().__init__()
        self.threshold = threshold

    def _detect_outliers_zscore_df(self, df: pd.DataFrame, threshold=5) -> pd.DataFrame:
        outliers = pd.DataFrame(index=df.index, columns=df.columns, dtype=bool)
        info = []
        for hour in range(24):
            hourly_data: pd.DataFrame = df[df.index.hour == hour]

            mean = hourly_data.stack().mean()
            std = hourly_data.stack().std()

            if pd.isna(mean) or pd.isna(std) or std == 0:
                raise ValueError(f"Hourly Z-Score stats invalid in hour {hour}")

            z_scores = (hourly_data - mean) / std
            outliers.loc[hourly_data.index] = np.abs(z_scores) > threshold
            info.append(
                {
                    "hour": hour,
                    "mean": mean,
                    "std": std,
                    "threshold": threshold,
                    "outlier_threshold": mean + threshold * std,
                }
            )
        return outliers, info

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        outliers, _ = self._detect_outliers_zscore_df(df, self.threshold)
        if outliers.isna().all().all():
            raise ValueError("Outliers are all NaN or invalid")

        df_clean = df.mask(outliers)
        self.successed_list.extend(df.columns)
        return df_clean


class MADOutlierProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 3.5, adjustment: float = 1e-9) -> None:
        super().__init__()
        self.threshold = threshold
        self.adjustment = adjustment

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self._apply_threshold_to_series)

    def _apply_threshold_to_series(self, series: pd.Series) -> pd.Series:
        median = series.median()
        deviation = (series - median).abs()
        mad = deviation.median()

        if pd.isna(mad):
            self.failed_list.append(series.name)
        else:
            self.successed_list.append(series.name)

        modified_z_score = 0.6745 * deviation / (mad + self.adjustment)
        series_clean = series.where(modified_z_score <= self.threshold)

        return series_clean


class TrimmedMeanOutlierProcessor(OutlierProcessor):
    def __init__(
        self,
        rate: float = 0.05,
        threshold: float = 3,
        adjustment: float = 1e-9,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.adjustment = adjustment
        self.rate = rate

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self._apply_threshold_to_series)

    def _apply_threshold_to_series(self, series: pd.Series) -> pd.Series:
        cleaned_series = series.dropna().sort_values()
        n = len(cleaned_series)
        cut = int(n * self.rate)

        # If the trim window is too large, treat as failed
        if cut * 2 >= n:
            self.failed_list.append(series.name)
            return series

        trimmed_data = cleaned_series.iloc[cut:-cut]
        trimmed_mean = trimmed_data.mean()
        trimmed_std = trimmed_data.std()

        if pd.isna(trimmed_mean) or pd.isna(trimmed_std) or trimmed_std == 0:
            self.failed_list.append(series.name)
        else:
            self.successed_list.append(series.name)

        zscore = (series - trimmed_mean).abs() / (trimmed_std + self.adjustment)
        series_clean = series.where(zscore <= self.threshold)

        return series_clean
