import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import TYPE_CHECKING
import warnings

from .base import Interpolator

if TYPE_CHECKING:
    from ...adj_mx import AdjacencyMatrix


class LegacyMICEInterpolator(Interpolator):
    """
    MICE (Multivariate Imputation by Chained Equations) Interpolator

    Imputes missing values by processing each sensor independently.
    This baseline design is intended to reduce computation and noise coupling
    when the number of sensors is large (~2000).

    Features:
    - Temporal patterns (hour-of-day, day-of-week) with cyclic encoding
    - Time-lag features (lag 1, 2, 3)
    - Spatial proxy: Network-wide average traffic at each timestamp
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_iter: int = 16,
        random_state: int = 42,
        verbose: int = 0,
        fallback_method: str = "linear",
        n_jobs: int = -3,
        suppress_warnings: bool = True,
        track_warnings: bool = True,
    ) -> None:
        """
        Args:
            n_estimators: Number of trees in ExtraTreesRegressor
            max_iter: Maximum MICE iterations
            random_state: Random seed for reproducibility
            verbose: Logging level (0=silent, 1=progress, 2=detailed)
            fallback_method: Fallback interpolation for remaining NaNs
                           ('linear', 'ffill', 'bfill', 'median')
            suppress_warnings: Suppress ConvergenceWarning display (default: True)
            track_warnings: Track and report warning counts (default: False)
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.fallback_method = fallback_method
        self.n_jobs = n_jobs  # Number of CPU cores to use in parallel
        self.suppress_warnings = suppress_warnings
        self.track_warnings = track_warnings
        self.warning_counts = {"convergence": 0}  # Warning counter

    def _compute_global_features(self, df: pd.DataFrame) -> dict:
        """Compute global features shared across all sensors."""
        network_avg = df.mean(axis=1, skipna=True)

        df_index = pd.DatetimeIndex(df.index)

        return {
            # Spatial proxy: network average
            "network_avg": network_avg,
            "network_avg_lag1": network_avg.shift(1).fillna(network_avg),
            "network_avg_lag2": network_avg.shift(2).fillna(network_avg),
            # Temporal features (cyclic encoding)
            "hour_sin": np.sin(2 * np.pi * df_index.hour / 24),
            "hour_cos": np.cos(2 * np.pi * df_index.hour / 24),
            "dayofweek_sin": np.sin(2 * np.pi * df_index.dayofweek / 7),
            "dayofweek_cos": np.cos(2 * np.pi * df_index.dayofweek / 7),
        }

    def _create_feature_matrix(
        self, sensor_data: pd.Series, global_features: dict
    ) -> pd.DataFrame:
        """Build a feature matrix for a single sensor."""
        features = pd.DataFrame({sensor_data.name: sensor_data})

        # Add global features
        for key, value in global_features.items():
            features[key] = value

        # Add lag features
        for lag in [1, 2, 3]:
            lag_col = f"{sensor_data.name}_lag{lag}"
            features[lag_col] = sensor_data.shift(lag).fillna(sensor_data)

        return features

    def _impute_sensor(
        self, sensor_data: pd.Series, global_features: dict
    ) -> np.ndarray:
        """Run MICE imputation for a single sensor."""
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            ),
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0,  # Keep imputer's own verbose output disabled
        )

        feature_matrix = self._create_feature_matrix(sensor_data, global_features)

        # Track warnings with warnings.catch_warnings
        with warnings.catch_warnings(record=True) as w:
            if self.suppress_warnings:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            else:
                warnings.filterwarnings("always", category=ConvergenceWarning)

            imputed_array = imputer.fit_transform(feature_matrix)

            # Count warnings
            if self.track_warnings:
                for warning in w:
                    if issubclass(warning.category, ConvergenceWarning):
                        self.warning_counts["convergence"] += 1

        return imputed_array[:, 0]  # Return first column (original sensor series)

    def _apply_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fallback handling for NaNs remaining after MICE."""
        remaining_nans = df.isna().sum().sum()
        if remaining_nans == 0:
            return df

        if self.verbose > 0:
            print(
                f"Warning: {remaining_nans} NaN values remain after MICE. "
                f"Applying fallback method: {self.fallback_method}"
            )

        # Apply fallback strategy
        fallback_strategies = {
            "linear": lambda x: x.interpolate(
                method="linear", limit_direction="both", axis=0
            ),
            "ffill": lambda x: x.fillna(method="ffill").fillna(method="bfill"),
            "bfill": lambda x: x.fillna(method="bfill").fillna(method="ffill"),
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
        """
        Run independent MICE imputation for each sensor.

        Args:
            df: Wide-format DataFrame (index=DatetimeIndex, columns=sensor_ids)

        Returns:
            Interpolated DataFrame with same shape as input
        """
        # Reset warning counter
        self.warning_counts = {"convergence": 0}

        global_features = self._compute_global_features(df)

        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._impute_sensor)(df[col], global_features) for col in df.columns
        )

        imputed_data = dict(zip(df.columns, results))
        result_df = pd.DataFrame(imputed_data, index=df.index)

        # Print warning statistics
        if self.track_warnings and self.verbose > 0:
            total_sensors = len(df.columns)
            conv_count = self.warning_counts["convergence"]
            print(f"\n[Warning Statistics]")
            print(
                f"  ConvergenceWarning: {conv_count}/{total_sensors} sensors "
                f"({conv_count/total_sensors*100:.1f}%)"
            )

        return self._apply_fallback(result_df)


class SpatialMICEInterpolator(Interpolator):
    """
    Spatial MICE Interpolator with Adjacency Matrix

    Uses an adjacency matrix to select spatially close sensors for MICE imputation,
    following a similar idea to the KNN interpolator design.

    Features:
    - Spatial neighbors from adjacency matrix
    - Temporal patterns (hour-of-day, day-of-week) with cyclic encoding
    - Network-wide average traffic (global feature)
    - Memory-efficient slice-based parallel execution
    """

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
        """
        Args:
            adj_matrix: AdjacencyMatrix object (contains sensor distance/connectivity)
            n_spatial_features (N): Number of top adjacent sensors used for imputation
            n_estimators: Number of trees in ExtraTreesRegressor
            max_iter: Maximum MICE iterations
            random_state: Random seed for reproducibility
            verbose: Logging level (0=silent, 1=progress, 2=detailed)
            fallback_method: Fallback interpolation for remaining NaNs
                           ('linear', 'ffill', 'bfill', 'median')
            n_jobs: Parallel worker count (-1=all CPU cores, 1=serial)
            suppress_warnings: Suppress ConvergenceWarning display (default: True)
        """
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
        """Return top-N nearest neighbor sensor IDs from adjacency matrix."""
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
        """Compute global feature g(t) shared across all sensors."""
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
        """
        Prepare a minimal data slice for a single sensor.
        Include only required columns for memory efficiency.
        """
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
        """
        Impute missing values for a single sensor with MICE (parallel helper).

        Args:
            target_sensor: Target sensor ID
            subset_df: Slice DataFrame containing target/neighbor/global features
        """
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

        # Return only target sensor result (first column)
        return target_sensor, imputed_array[:, 0]

    def _apply_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fallback handling for NaNs remaining after MICE."""
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
        """
        Run parallel MICE imputation per sensor (column) using top-N adjacent sensors as features.
        """
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
