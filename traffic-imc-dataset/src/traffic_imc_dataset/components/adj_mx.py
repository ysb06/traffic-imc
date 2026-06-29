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
        id_list: IdList,
        distances_imc: DistancesImc,
        normalized_k: float = 0.1,
        top_k: Optional[int] = 10,
    ) -> "AdjacencyMatrix":
        def apply_rowwise_top_k(
            adj_mx: np.ndarray,
            top_k: Optional[int],
        ) -> np.ndarray:
            if top_k is None:
                return adj_mx

            pruned = np.zeros_like(adj_mx)
            diag_indices = np.diag_indices_from(adj_mx)
            pruned[diag_indices] = adj_mx[diag_indices]
            if top_k == 0:
                return pruned

            for row_idx, row in enumerate(adj_mx):
                candidate_indices = np.flatnonzero(row > 0)
                candidate_indices = candidate_indices[candidate_indices != row_idx]
                if len(candidate_indices) == 0:
                    continue

                sorted_indices = candidate_indices[
                    np.argsort(row[candidate_indices], kind="stable")[::-1]
                ]
                keep_indices = sorted_indices[:top_k]
                pruned[row_idx, keep_indices] = row[keep_indices]

            return pruned

        def get_adjacency_matrix(
            distance_df: pd.DataFrame,
            sensor_ids: List[str],
            normalized_k: float,
            top_k: Optional[int],
        ):
            num_sensors = len(sensor_ids)
            dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
            dist_mx[:] = np.inf

            sensor_id_to_index: Dict[str, int] = {}
            for i, sensor_id in enumerate(sensor_ids):
                sensor_id_to_index[sensor_id] = i

            for row in tqdm(
                distance_df.values, total=len(distance_df), desc="Filling Matrix"
            ):
                if row[0] not in sensor_id_to_index or row[1] not in sensor_id_to_index:
                    continue
                dist_mx[sensor_id_to_index[row[0]], sensor_id_to_index[row[1]]] = row[2]

            distances = dist_mx[~np.isinf(dist_mx)].flatten()
            std = distances.std()
            adj_mx: np.ndarray = np.exp(-np.square(dist_mx / std))
            adj_mx[adj_mx < normalized_k] = 0
            adj_mx = apply_rowwise_top_k(adj_mx, top_k)

            return adj_mx, sensor_id_to_index

        sensor_ids = id_list.data
        adj_mx, sendsor_id_to_idx = get_adjacency_matrix(
            distances_imc.data, sensor_ids, normalized_k, top_k
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
