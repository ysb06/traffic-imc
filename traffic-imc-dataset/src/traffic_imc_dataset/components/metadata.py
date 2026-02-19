import logging
import os
from typing import List, Optional, Union

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm

from .metr_ids import IdList

logger = logging.getLogger(__name__)


class Metadata:
    @staticmethod
    def import_from_nodelink(nodelink_path: str) -> "Metadata":
        road_data = gpd.read_file(nodelink_path)
        road_data = road_data[
            [
                "LINK_ID",  # Link ID
                "ROAD_NAME",  # Road name
                "LANES",  # Number of lanes
                "ROAD_RANK",  # Road rank
                "ROAD_TYPE",  # Road type
                "MAX_SPD",  # Maximum speed limit
                "REST_VEH",  # Restricted vehicle type
            ]
        ]

        return Metadata(road_data)

    @staticmethod
    def import_from_hdf(filepath: str, key: str = "data") -> "Metadata":
        logger.info(f"Loading data from {filepath}...")
        data = pd.read_hdf(filepath, key=key)
        return Metadata(data)

    def __init__(self, raw: gpd.GeoDataFrame) -> None:
        self._raw = raw
        self._sensor_filter = raw["LINK_ID"].unique()
        self.data = raw.copy()

    @property
    def sensor_filter(self) -> List[str]:
        return self._sensor_filter

    @sensor_filter.setter
    def sensor_filter(self, sensor_ids: Union[List[str], IdList]) -> None:
        if isinstance(sensor_ids, IdList):
            sensor_ids = sensor_ids.data

        requested_ids = set(sensor_ids)
        original_ids = set(self._raw["LINK_ID"])
        missing_sensors = requested_ids - original_ids
        if missing_sensors:
            logger.warning(
                f"The following sensors do not exist in the data: {', '.join(missing_sensors)}"
            )
        target_ids = requested_ids & original_ids
        self._sensor_filter = list(target_ids)

        self.data = self._raw[self._raw["LINK_ID"].isin(target_ids)].copy()

    def to_hdf(self, filepath: str, key: str = "data") -> None:
        logger.info(f"Saving data to {filepath}...")
        self.data.to_hdf(filepath, key=key)
        logger.info("Saving Complete")

    def to_excel(self, filepath: str) -> None:
        logger.info(f"Saving data to {filepath}...")
        self.data.to_excel(filepath)
        logger.info("Saving Complete")
