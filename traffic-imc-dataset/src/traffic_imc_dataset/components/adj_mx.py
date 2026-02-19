import os
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging
import pickle
from tqdm import tqdm
from .metr_ids import IdList
from .distance_imc import DistancesImc
from pathlib import Path

logger = logging.getLogger(__name__)


class AdjacencyMatrix:
    @staticmethod
    def import_from_pickle(filepath: Union[str, Path]) -> "AdjacencyMatrix":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return AdjacencyMatrix(*data)

    @staticmethod
    def import_from_components(
        id_list: IdList, distances_imc: DistancesImc, normalized_k=0.1
    ) -> "AdjacencyMatrix":
        def get_adjacency_matrix(
            distance_df: pd.DataFrame,
            sensor_ids: List[str],
            normalized_k: float,
        ):
            num_sensors = len(sensor_ids)
            dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
            dist_mx[:] = np.inf

            sensor_id_to_index: Dict[str, int] = {}  # Builds sensor id to index map.
            for i, sensor_id in enumerate(sensor_ids):
                sensor_id_to_index[sensor_id] = i

            # Fills cells in the matrix with distances.
            for row in tqdm(
                distance_df.values, total=len(distance_df), desc="Filling Matrix"
            ):
                if row[0] not in sensor_id_to_index or row[1] not in sensor_id_to_index:
                    continue
                dist_mx[sensor_id_to_index[row[0]], sensor_id_to_index[row[1]]] = row[2]

            # Calculates the standard deviation as theta.
            distances = dist_mx[~np.isinf(dist_mx)].flatten()
            std = distances.std()
            adj_mx: np.ndarray = np.exp(-np.square(dist_mx / std))
            # Make the adjacent matrix symmetric by taking the max.
            # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

            # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
            adj_mx[adj_mx < normalized_k] = 0

            return adj_mx, sensor_id_to_index

        sensor_ids = id_list.data
        adj_mx, sendsor_id_to_idx = get_adjacency_matrix(
            distances_imc.data, sensor_ids, normalized_k
        )

        return AdjacencyMatrix(sensor_ids, sendsor_id_to_idx, adj_mx)

    def __init__(
        self, raw_ids: List[str], raw_id_map: Dict[str, int], raw_adj_mx: np.ndarray
    ) -> None:
        self._raw = (raw_ids, raw_id_map, raw_adj_mx)

    @property
    def sensor_ids(self) -> List[str]:
        return self._raw[0]

    @property
    def sensor_id_to_idx(self) -> Dict[str, int]:
        return self._raw[1]

    @property
    def adj_mx(self) -> np.ndarray:
        return self._raw[2]

    @property
    def data_exists(self) -> bool:
        return self._raw is not None

    def to_pickle(self, filepath: str) -> None:
        logger.info(f"Saving data to {filepath}...")
        with open(filepath, "wb") as f:
            pickle.dump(self._raw, f)
        logger.info("Saving Complete")
