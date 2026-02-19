import pandas as pd
import numpy as np
import logging
from sklearn.impute import KNNImputer
from joblib import Parallel, delayed
from typing import Tuple
from ...adj_mx import AdjacencyMatrix  # Assumes the adjacency class defined above
from .base import Interpolator

logger = logging.getLogger(__name__)


class SpatialKNNInterpolator(Interpolator):
    def __init__(
        self,
        adj_matrix: AdjacencyMatrix,
        n_spatial_features: int = 10,
        k_time_neighbors: int = 5,
        n_jobs: int = -2,
    ):
        """
        Args:
            adj_matrix: AdjacencyMatrix object (includes sensor distance/connectivity info)
            n_spatial_features (N): Number of top neighboring sensors used for imputation
            k_time_neighbors (k): Number of temporal neighbors (rows) for KNN imputation
            n_jobs (int): Number of parallel workers (-1=all CPU cores, 1=serial)
        """
        super().__init__()
        self.adj_matrix = adj_matrix
        self.n_spatial_features = n_spatial_features
        self.k_time_neighbors = k_time_neighbors
        self.n_jobs = n_jobs

    def _get_top_n_neighbors(self, sensor_id: str) -> list:
        """Return top-N nearest neighbor sensor IDs from the adjacency matrix."""
        if sensor_id not in self.adj_matrix.sensor_id_to_idx:
            return []

        idx = self.adj_matrix.sensor_id_to_idx[sensor_id]
        # Get adjacency vector of target sensor
        adj_vector = self.adj_matrix.adj_mx[idx]

        # Sort by adjacency strength descending
        sorted_indices = np.argsort(adj_vector)[::-1]

        # Keep only sensors with positive adjacency (exclude self and sparse zeros)
        # Then take top N
        top_n_indices = [i for i in sorted_indices if adj_vector[i] > 0 and i != idx][
            : self.n_spatial_features
        ]

        # Convert indices back to sensor IDs
        return [self.adj_matrix.sensor_ids[i] for i in top_n_indices]

    def _impute_single_sensor(
        self, target_sensor: str, subset_df: pd.DataFrame
    ) -> Tuple[str, np.ndarray]:
        """
        Impute missing values for a single sensor (parallel worker helper).

        Args:
            target_sensor: Target sensor ID
            subset_df: Slice DataFrame containing target/neighbor sensors and network_avg
        """
        # KNN Imputation
        knni = KNNImputer(n_neighbors=self.k_time_neighbors)
        imputed_subset = knni.fit_transform(subset_df)

        # Return only the target sensor output (first column)
        return target_sensor, imputed_subset[:, 0]

    def _prepare_sensor_subset(
        self, target_sensor: str, df: pd.DataFrame, network_avg: pd.Series
    ) -> pd.DataFrame:
        """
        Prepare a minimal slice for one sensor.
        Includes only required columns for memory efficiency.
        """
        neighbor_ids = self._get_top_n_neighbors(target_sensor)
        valid_neighbors = [nid for nid in neighbor_ids if nid in df.columns]

        if not valid_neighbors:
            logger.info(
                f"No valid neighbors for sensor {target_sensor}. "
                "Using only network average."
            )

        # Include target sensor + neighbors + network_avg only
        subset_df = pd.DataFrame(
            {
                target_sensor: df[target_sensor],
                **{nid: df[nid] for nid in valid_neighbors},
                "_network_avg": network_avg,
            }
        )

        return subset_df

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run KNN imputation in parallel.
        For each sensor(column), use top-N neighboring sensor values as features.
        """
        # Output DataFrame for imputed values
        imputed_df = df.copy()

        # Process only columns (sensors) with missing values
        columns_with_nan = df.columns[df.isnull().any()].tolist()

        if not columns_with_nan:
            logger.info("No missing values found. Skipping interpolation.")
            return imputed_df

        # Global feature: network average per timestamp
        # If all sensors are missing at a timestamp, interpolate linearly over time
        network_avg = df.mean(axis=1, skipna=True).interpolate(method="linear")
        # If still missing at boundaries, fill with overall mean
        network_avg = network_avg.fillna(network_avg.mean())

        logger.info(
            f"Processing {len(columns_with_nan)} sensors with n_jobs={self.n_jobs}..."
        )

        # Prebuild per-sensor slices for memory efficiency
        # (pass only required columns to workers, not the full DataFrame)
        sensor_subsets = {
            sensor: self._prepare_sensor_subset(sensor, df, network_avg)
            for sensor in columns_with_nan
        }

        # Parallel execution with joblib (slice-based to reduce memory usage)
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._impute_single_sensor)(sensor, sensor_subsets[sensor])
            for sensor in columns_with_nan
        )

        # Write results back to the output DataFrame
        for sensor_id, imputed_values in results:
            imputed_df[sensor_id] = imputed_values

        logger.info("Interpolation completed.")
        return imputed_df
