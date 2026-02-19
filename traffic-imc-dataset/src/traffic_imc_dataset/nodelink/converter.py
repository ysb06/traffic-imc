import logging
import os
from functools import reduce
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)


class NodeLinkData:
    def __init__(
        self,
        node_data: gpd.GeoDataFrame,
        link_data: gpd.GeoDataFrame,
        turn_data: pd.DataFrame,
    ) -> None:
        self.node_data = node_data
        self.link_data = link_data
        self.turn_data = turn_data

    def filter_by_gu_codes(self, gu_code_list: List[str]) -> "NodeLinkData":
        node_cols = ["NODE_ID"]
        link_cols = ["LINK_ID", "F_NODE", "T_NODE"]
        turn_cols = ["NODE_ID", "ST_LINK", "ED_LINK"]

        filtered_nodes = self._filter_data(self.node_data, node_cols, gu_code_list)
        filtered_links = self._filter_data(self.link_data, link_cols, gu_code_list)
        filtered_turns = self._filter_data(self.turn_data, turn_cols, gu_code_list)

        return NodeLinkData(filtered_nodes, filtered_links, filtered_turns)

    def export(
        self,
        node_output_path: str,
        link_output_path: str,
        turn_output_path: str,
    ) -> None:
        turn_data = gpd.GeoDataFrame(self.turn_data)

        logger.info(f"Exporting datasets...")
        logger.info(f"Node Data: {node_output_path}")
        self.node_data.to_file(node_output_path, encoding="utf-8")
        logger.info(f"Link Data: {link_output_path}")
        self.link_data.to_file(link_output_path, encoding="utf-8")
        logger.info(f"Turn Data: {turn_output_path}")
        turn_data.to_file(turn_output_path, encoding="utf-8")
        logger.info("Exporting completed.")

    def _filter_data(
        self,
        data: Union[gpd.GeoDataFrame, pd.DataFrame],
        columns: List[str],
        code_list: List[str],
    ) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        condition_list = [data[column].str[:3].isin(code_list) for column in columns]
        return data[reduce(lambda x, y: x | y, condition_list)]


class NodeLink(NodeLinkData):
    def __init__(self, root_path, encoding="cp949") -> None:
        self.node_path = os.path.join(root_path, "MOCT_NODE.shp")
        self.link_path = os.path.join(root_path, "MOCT_LINK.shp")
        self.turn_path = os.path.join(root_path, "TURNINFO.dbf")

        logger.info("Loading data...")
        node_data: gpd.GeoDataFrame = gpd.read_file(self.node_path, encoding=encoding)
        link_data: gpd.GeoDataFrame = gpd.read_file(self.link_path, encoding=encoding)
        turn_data: pd.DataFrame = gpd.read_file(self.turn_path, encoding=encoding)

        super().__init__(node_data, link_data, turn_data)


def get_sensor_node_list(
    traffic_data: pd.DataFrame,
    sensor_attr: List[str] = [
        "road_bhf_fclts_id",
        "road_bhf_fclts_nm",
        "instl_lc_nm",
        "road_bhf_area_nm",
    ],
    sensor_coord_attr: List[str] = ["lo_ycrd", "la_xcrd"],
) -> gpd.GeoDataFrame:
    unique_sensors = traffic_data[sensor_attr + sensor_coord_attr].drop_duplicates()

    xs = unique_sensors[sensor_coord_attr[0]]
    ys = unique_sensors[sensor_coord_attr[1]]

    seonsor_geometry = [Point(xy) for xy in zip(xs, ys)]
    sensor_data = unique_sensors[sensor_attr]

    sensor_gdf = gpd.GeoDataFrame(sensor_data, geometry=seonsor_geometry)
    sensor_gdf.crs = "EPSG:4326"

    return sensor_gdf
