import os
import numpy as np
import pandas as pd
import geopandas as gpd

import logging

logger = logging.getLogger(__name__)


class SensorLocations:
    @staticmethod
    def import_from_csv(
        filepath: str,
        id_col: str = "sensor_id",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> "SensorLocations":
        with open(filepath, "r") as file:
            first_line = file.readline().strip().split(",")

        def is_first_line_float(line):
            try:
                for item in line:
                    float(item)
                return True
            except ValueError:
                return False

        if is_first_line_float(first_line):
            raw = pd.read_csv(filepath, header=None)
            raw.columns = [id_col, lat_col, lon_col]  # Set default column names
        else:
            raw = pd.read_csv(filepath)

        raw = raw[[id_col, lat_col, lon_col]]
        raw[id_col] = raw[id_col].astype(str)
        raw.columns = ["sensor_id", "latitude", "longitude"]

        return SensorLocations(raw)

    @staticmethod
    def import_from_nodelink(nodelink_road_path) -> "SensorLocations":
        def calculate_mean_coords(geometry):
            x_coords = [point[0] for point in geometry.coords]
            y_coords = [point[1] for point in geometry.coords]
            mean_x = np.mean(x_coords)
            mean_y = np.mean(y_coords)

            return pd.Series([mean_x, mean_y], index=["longitude", "latitude"])

        road_data: gpd.GeoDataFrame = gpd.read_file(nodelink_road_path)
        road_data = road_data.to_crs(epsg=4326)
        result = pd.DataFrame(columns=["sensor_id", "latitude", "longitude"])
        result["sensor_id"] = road_data["LINK_ID"]
        result[["longitude", "latitude"]] = road_data["geometry"].apply(
            calculate_mean_coords
        )

        return SensorLocations(result)

    def __init__(self, raw: pd.DataFrame) -> None:
        self._raw = raw
        self._sensor_filter = self._raw["sensor_id"].to_list()
        self.data = self._raw.copy()

    @property
    def sensor_filter(self) -> list[str]:
        return self._sensor_filter

    @sensor_filter.setter
    def sensor_filter(self, sensor_ids: list[str]) -> None:
        missing_sensors = set(sensor_ids) - set(self._raw["sensor_id"])
        if missing_sensors:
            logger.warning(
                f"The following sensors do not exist in the data: {', '.join(missing_sensors)}"
            )
        new_sensor_ids = [
            sensor_id
            for sensor_id in sensor_ids
            if sensor_id in set(self._raw["sensor_id"])
        ]
        self._sensor_filter = new_sensor_ids
        self.data = self._raw[self._raw["sensor_id"].isin(new_sensor_ids)].copy()

    def to_shapefile(self, filepath: str) -> None:
        logger.info(f"Saving data to {filepath}...")
        data = pd.DataFrame(columns=["LINK_ID"])
        data["LINK_ID"] = self.data["sensor_id"]
        geometry = gpd.points_from_xy(self.data["longitude"], self.data["latitude"])
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

        gdf.to_file(filepath, index=False)
        logger.info("Saving Complete")

    def to_csv(self, filepath: str) -> None:
        logger.info(f"Saving data to {filepath}...")
        self.data.to_csv(filepath)
        logger.info("Saving Complete")
