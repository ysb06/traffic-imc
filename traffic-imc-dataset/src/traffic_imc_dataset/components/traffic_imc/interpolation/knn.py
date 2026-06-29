import pandas as pd
import numpy as np
import logging
from sklearn.impute import KNNImputer
from joblib import Parallel, delayed
from typing import Tuple
from ...adj_mx import AdjacencyMatrix
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
        super().__init__()
        self.adj_matrix = adj_matrix
        self.n_spatial_features = n_spatial_features
        self.k_time_neighbors = k_time_neighbors
        self.n_jobs = n_jobs

    def _get_top_n_neighbors(self, sensor_id: str) -> list:
        if sensor_id not in self.adj_matrix.sensor_id_to_idx:
            return []

        idx = self.adj_matrix.sensor_id_to_idx[sensor_id]
        adj_vector = self.adj_matrix.adj_mx[idx]
        sorted_indices = np.argsort(adj_vector)[::-1]
        top_n_indices = [i for i in sorted_indices if adj_vector[i] > 0 and i != idx][
            : self.n_spatial_features
        ]

        return [self.adj_matrix.sensor_ids[i] for i in top_n_indices]

    def _impute_single_sensor(
        self, target_sensor: str, subset_df: pd.DataFrame
    ) -> Tuple[str, np.ndarray]:
        # KNN Imputation
        knni = KNNImputer(n_neighbors=self.k_time_neighbors)
        imputed_subset = knni.fit_transform(subset_df)

        return target_sensor, imputed_subset[:, 0]

    def _prepare_sensor_subset(
        self, target_sensor: str, df: pd.DataFrame, network_avg: pd.Series
    ) -> pd.DataFrame:
        neighbor_ids = self._get_top_n_neighbors(target_sensor)
        valid_neighbors = [nid for nid in neighbor_ids if nid in df.columns]

        if not valid_neighbors:
            logger.info(
                f"No valid neighbors for sensor {target_sensor}. "
                "Using only network average."
            )

        subset_df = pd.DataFrame(
            {
                target_sensor: df[target_sensor],
                **{nid: df[nid] for nid in valid_neighbors},
                "_network_avg": network_avg,
            }
        )

        return subset_df

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        imputed_df = df.copy()
        columns_with_nan = df.columns[df.isnull().any()].tolist()

        if not columns_with_nan:
            logger.info("No missing values found. Skipping interpolation.")
            return imputed_df

        network_avg = df.mean(axis=1, skipna=True).interpolate(method="linear")
        network_avg = network_avg.fillna(network_avg.mean())

        logger.info(
            f"Processing {len(columns_with_nan)} sensors with n_jobs={self.n_jobs}..."
        )

        sensor_subsets = {
            sensor: self._prepare_sensor_subset(sensor, df, network_avg)
            for sensor in columns_with_nan
        }

        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._impute_single_sensor)(sensor, sensor_subsets[sensor])
            for sensor in columns_with_nan
        )

        for sensor_id, imputed_values in results:
            imputed_df[sensor_id] = imputed_values

        logger.info("Interpolation completed.")
        return imputed_df
