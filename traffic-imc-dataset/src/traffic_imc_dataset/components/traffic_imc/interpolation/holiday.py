from .base import Interpolator
import pandas as pd
import holidays

class HMHMeanFillInterpolator(Interpolator):
    def __init__(self) -> None:
        super().__init__()

    def _is_holiday(self, date: pd.Timestamp) -> bool:
        korean_holidays = holidays.KR(years=date.year)
        return date.weekday() >= 5 or date in korean_holidays

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.copy()

        is_holiday = pd.Series(
            df_filled.index.map(self._is_holiday), index=df_filled.index
        )

        grouping_keys = [df_filled.index.month, df_filled.index.hour, is_holiday]
        group_means = df_filled.groupby(grouping_keys).transform("mean")

        df_filled.fillna(group_means, inplace=True)

        if df_filled.isnull().values.any():
            hourly_means = df_filled.groupby(df_filled.index.hour).transform("mean")
            df_filled.fillna(hourly_means, inplace=True)

        return df_filled