import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import warnings

from .base import Interpolator
from ...adj_mx import AdjacencyMatrix


class SpatialMICEInterpolator(Interpolator):
    def __init__(
        self,
        adj_matrix: "AdjacencyMatrix",
        n_spatial_features: int = 10,
        n_estimators: int = 10,
        max_iter: int = 16,
        random_state: int = 42,
        verbose: int = 0,
        fallback_method: str = "linear",
        n_jobs: int = -2,
        suppress_warnings: bool = True,
    ) -> None:
        super().__init__()
        self.adj_matrix = adj_matrix
        self.n_spatial_features = n_spatial_features
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.fallback_method = fallback_method
        self.n_jobs = n_jobs
        self.suppress_warnings = suppress_warnings

    def _get_top_n_neighbors(self, sensor_id: str) -> list:
        if sensor_id not in self.adj_matrix.sensor_id_to_idx:
            return []

        idx = self.adj_matrix.sensor_id_to_idx[sensor_id]
        adj_vector = self.adj_matrix.adj_mx[idx]

        # Sort by adjacency strength (descending)
        sorted_indices = np.argsort(adj_vector)[::-1]

        # Keep only sensors with adjacency > 0 (excluding self and sparse zeros)
        top_n_indices = [
            i for i in sorted_indices if adj_vector[i] > 0 and i != idx
        ][: self.n_spatial_features]

        return [self.adj_matrix.sensor_ids[i] for i in top_n_indices]

    def _compute_global_features(self, df: pd.DataFrame) -> dict:
        network_avg = df.mean(axis=1, skipna=True)
        # Fill boundary NaNs so g(t) is always available.
        network_avg = network_avg.interpolate(method="linear").fillna(network_avg.mean())
        return {"network_avg": network_avg}

    def _prepare_sensor_subset(
        self,
        target_sensor: str,
        df: pd.DataFrame,
        global_features: dict,
    ) -> pd.DataFrame:
        neighbor_ids = self._get_top_n_neighbors(target_sensor)
        valid_neighbors = [nid for nid in neighbor_ids if nid in df.columns]

        if not valid_neighbors and self.verbose > 0:
            print(
                f"No valid neighbors for sensor {target_sensor}. "
                "Using only global features."
            )

        # Target sensor + neighbor sensors + global feature g(t)
        subset_data = {
            target_sensor: df[target_sensor],
            **{nid: df[nid] for nid in valid_neighbors},
            "_network_avg": global_features["network_avg"],
        }

        return pd.DataFrame(subset_data)

    def _impute_single_sensor(
        self, target_sensor: str, subset_df: pd.DataFrame
    ) -> tuple:
        if subset_df[target_sensor].isna().all():
            if self.verbose > 0:
                print(
                    f"Sensor {target_sensor} has no observed values. "
                    "Using network average fallback."
                )
            return target_sensor, subset_df["_network_avg"].to_numpy()

        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            ),
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0,
        )

        with warnings.catch_warnings():
            if self.suppress_warnings:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

            imputed_array = imputer.fit_transform(subset_df)

        return target_sensor, imputed_array[:, 0]

    def _apply_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        remaining_nans = df.isna().sum().sum()
        if remaining_nans == 0:
            return df

        if self.verbose > 0:
            print(
                f"Warning: {remaining_nans} NaN values remain after MICE. "
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

        # Final safety net
        final_nans = df.isna().sum().sum()
        if final_nans > 0:
            if self.verbose > 0:
                print(
                    f"Warning: {final_nans} NaN values still remain. "
                    f"Filling with 0 as last resort."
                )
            df = df.fillna(0)

        return df

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        imputed_df = df.copy()

        # Process only columns (sensors) with missing values
        columns_with_nan = df.columns[df.isnull().any()].tolist()

        if not columns_with_nan:
            if self.verbose > 0:
                print("No missing values found. Skipping interpolation.")
            return imputed_df

        # Compute global features
        global_features = self._compute_global_features(df)

        if self.verbose > 0:
            print(
                f"Processing {len(columns_with_nan)} sensors with n_jobs={self.n_jobs}..."
            )

        # Prebuild per-sensor slices for memory efficiency
        sensor_subsets = {
            sensor: self._prepare_sensor_subset(sensor, df, global_features)
            for sensor in columns_with_nan
        }

        # Parallel execution with joblib (slice-only inputs to reduce memory usage)
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._impute_single_sensor)(sensor, sensor_subsets[sensor])
            for sensor in columns_with_nan
        )

        # Apply results to output DataFrame
        for sensor_id, imputed_values in results:
            imputed_df[sensor_id] = imputed_values

        if self.verbose > 0:
            print("Interpolation completed.")

        return self._apply_fallback(imputed_df)
