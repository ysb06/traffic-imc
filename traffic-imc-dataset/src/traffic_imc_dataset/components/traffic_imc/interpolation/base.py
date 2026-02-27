import pandas as pd

import logging

logger = logging.getLogger(__name__)


class Interpolator:
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._interpolate(df.copy())


class LinearInterpolator(Interpolator):
    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.interpolate(method="linear", axis=0)


class SplineLinearInterpolator(Interpolator):
    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.interpolate(method="slinear", axis=0)


class TimeMeanFillInterpolator(Interpolator):
    def _fill_na_with_same_time(self, s: pd.Series) -> pd.Series:
        s_filled = s.copy()
        for hour in range(24):
            hourly_data = s[s.index.hour == hour]
            mean_value = hourly_data.mean(skipna=True)
            s_filled.loc[s_filled.index.hour == hour] = s_filled.loc[
                s_filled.index.hour == hour
            ].fillna(mean_value)

        return s_filled
        # Historical note: a previous variant produced questionable values.

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.apply(self._fill_na_with_same_time, axis=0)
        result = result.round()
        return result


class ShiftFillInterpolator(Interpolator):
    def __init__(self, periods: int = 168) -> None:
        self.periods = periods

    def _fill_na_with_shifted(self, s: pd.Series) -> pd.Series:
        """Optimized helper that fills missing values from historical periods."""
        # Fast path when there is no missing value
        if not s.isna().any():
            return s

        s_filled = s.copy()
        na_indices = s[s.isna()].index
        values_dict = s.to_dict()
        min_idx = s.index.min()

        # Handle each missing timestamp
        for na_idx in na_indices:
            shifted_time_idx = na_idx - pd.Timedelta(hours=self.periods)
            while shifted_time_idx >= min_idx:
                # Use value when it exists and is not NaN
                if shifted_time_idx in values_dict and not pd.isna(
                    values_dict[shifted_time_idx]
                ):
                    s_filled.loc[na_idx] = values_dict[shifted_time_idx]
                    break
                # Otherwise move one more period back
                shifted_time_idx -= pd.Timedelta(hours=self.periods)

        return s_filled

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.apply(self._fill_na_with_shifted, axis=0)
        return result


class MonthlyMeanFillInterpolator(Interpolator):
    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for col in result.columns:
            result_index: pd.DatetimeIndex = result.index
            groups = result[col].groupby([result_index.year, result_index.month])

            # Compute per-group means
            group_means: pd.Series = groups.mean()

            for key, mean_value in group_means.items():
                yr, mon = key
                mask = (
                    (result_index.year == yr)
                    & (result_index.month == mon)
                    & (result[col].isna())
                )

                result.loc[mask, col] = mean_value

        return result
