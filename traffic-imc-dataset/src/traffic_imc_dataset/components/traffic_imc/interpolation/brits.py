import logging

import torch
import numpy as np
import pandas as pd
from pypots.imputation import BRITS
from sklearn.preprocessing import StandardScaler

from .base import Interpolator

logger = logging.getLogger(__name__)


class BRITSInterpolator(Interpolator):
    def __init__(
        self,
        n_steps: int = 24,
        rnn_hidden_size: int = 64,
        batch_size: int = 32,
        epochs: int = 50,
        device: str | None = None,
        random_seed: int | None = None,
        fallback_method: str = "linear",
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.n_steps = n_steps
        self.rnn_hidden_size = rnn_hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if device is not None else self._detect_device()
        self.random_seed = random_seed
        self.fallback_method = fallback_method
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.model = None

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _reshape(self, df: pd.DataFrame) -> tuple[np.ndarray, dict]:
        n_features = df.shape[1]
        total_len = len(df)

        scaled_values = self.scaler.fit_transform(df.values)

        n_samples = total_len // self.n_steps

        if n_samples == 0:
            raise ValueError(
                f"Data length ({total_len}) is shorter than n_steps ({self.n_steps})."
            )

        valid_len = n_samples * self.n_steps

        truncated_values = scaled_values[:valid_len]
        x_3d = truncated_values.reshape(n_samples, self.n_steps, n_features)

        meta = {
            "columns": df.columns,
            "index": df.index,
            "original_length": total_len,
            "valid_length": valid_len,
            "n_samples": n_samples,
            "n_features": n_features,
            "scaled_remainder": (
                scaled_values[valid_len:] if valid_len < total_len else None
            ),
        }

        return x_3d, meta

    def _inverse_reshape(self, imputed_3d: np.ndarray, meta: dict) -> pd.DataFrame:
        n_features = meta["n_features"]

        imputed_2d = imputed_3d.reshape(-1, n_features)

        if meta["scaled_remainder"] is not None:
            imputed_2d = np.vstack([imputed_2d, meta["scaled_remainder"]])

        final_values = self.scaler.inverse_transform(imputed_2d)

        df_result = pd.DataFrame(
            final_values, index=meta["index"], columns=meta["columns"]
        )

        return df_result

    def _apply_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        remaining_nans = df.isna().sum().sum()
        if remaining_nans == 0:
            return df

        if self.verbose > 0:
            logger.warning(
                f"{remaining_nans} NaN values remain after BRITS. "
                f"Applying fallback method: {self.fallback_method}"
            )

        fallback_strategies = {
            "linear": lambda x: x.interpolate(
                method="linear", limit_direction="both", axis=0
            ),
            "ffill": lambda x: x.ffill().bfill(),
            "bfill": lambda x: x.bfill().ffill(),
            "median": lambda x: x.fillna(x.median()),
        }

        df = fallback_strategies.get(
            self.fallback_method, fallback_strategies["linear"]
        )(df)

        final_nans = df.isna().sum().sum()
        if final_nans > 0:
            if self.verbose > 0:
                logger.warning(
                    f"{final_nans} NaN values still remain. "
                    f"Filling with 0 as last resort."
                )
            df = df.fillna(0)

        return df

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        x_3d, meta = self._reshape(df)

        if self.verbose > 0:
            logger.info(
                f"Tensor shape: {x_3d.shape} "
                f"(samples={meta['n_samples']}, steps={self.n_steps}, features={meta['n_features']})"
            )
            nan_count = np.isnan(x_3d).sum()
            total_count = x_3d.size
            logger.info(
                f"Missing values: {nan_count:,} / {total_count:,} ({nan_count/total_count*100:.2f}%)"
            )
            if meta["valid_length"] < meta["original_length"]:
                remainder = meta["original_length"] - meta["valid_length"]
                logger.info(
                    f"Note: Last {remainder} time steps will be processed by fallback"
                )
            logger.info(
                f"Running BRITS (rnn_hidden_size={self.rnn_hidden_size}, "
                f"epochs={self.epochs}, device={self.device})..."
            )

        self.model = BRITS(
            n_steps=self.n_steps,
            n_features=meta["n_features"],
            rnn_hidden_size=self.rnn_hidden_size,
            batch_size=self.batch_size,
            epochs=self.epochs,
            device=self.device,
            model_saving_strategy=None,
        )

        dataset = {"X": x_3d}
        self.model.fit(dataset)
        predictions = self.model.predict(dataset)

        imputed_3d = predictions["imputation"]

        df_imputed = self._inverse_reshape(imputed_3d, meta)

        df_result = df.copy()
        nan_mask = df.isna()
        df_result[nan_mask] = df_imputed[nan_mask]

        df_result = self._apply_fallback(df_result)

        if self.verbose > 0:
            logger.info("BRITS interpolation completed.")

        return df_result
