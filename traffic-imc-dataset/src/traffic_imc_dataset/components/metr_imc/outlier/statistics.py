import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats.mstats import winsorize
from .base import OutlierProcessor


class SimpleZscoreOutlierProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 3.0) -> None:
        super().__init__()
        self.threshold = threshold

    def _detect_outliers_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        z_scores = stats.zscore(df, axis=None, nan_policy="omit")
        if np.isnan(z_scores).all() or np.isinf(z_scores).all():
            self.successed_list.extend(df.columns)
        else:
            self.failed_list.extend(df.columns)

        outliers = np.abs(z_scores) > self.threshold
        return outliers

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        outliers = self._detect_outliers_zscore(df)
        df_clean = df.mask(outliers)
        return df_clean


class InSensorZscoreOutlierProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 3.0) -> None:
        super().__init__()
        self.threshold = threshold

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self._apply_threshold_to_series)

    def _apply_threshold_to_series(self, series: pd.Series) -> pd.Series:
        z_scores = stats.zscore(series, nan_policy="omit")
        if pd.isna(z_scores).all() or np.isinf(z_scores).all():
            self.successed_list.append(series.name)
        else:
            self.failed_list.append(series.name)

        series[np.abs(z_scores) > self.threshold] = np.nan
        return series


class WinsorizedOutlierProcessor(OutlierProcessor):
    def __init__(self, rate: float = 0.05, zscore_threshold: float = 3) -> None:
        super().__init__()
        self.rate = rate
        self.zscore_threshold = zscore_threshold

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self._apply_threshold_to_series)

    def _apply_threshold_to_series(self, series: pd.Series) -> pd.Series:
        w_data = winsorize(series, limits=[self.rate, self.rate])
        w_series = pd.Series(
            w_data, index=series.index, dtype=series.dtype, name=series.name
        )

        w_mean = w_series.mean()
        w_std = w_series.std()

        # Fail when mean/std is NaN or std is zero
        if pd.isna(w_mean) or pd.isna(w_std) or w_std == 0:
            self.failed_list.append(series.name)
        else:
            self.successed_list.append(series.name)

        zscore = (w_series - w_mean).abs() / w_std
        series_clean = w_series.mask(zscore > self.zscore_threshold)

        self.successed_list.append(series.name)
        return series_clean
