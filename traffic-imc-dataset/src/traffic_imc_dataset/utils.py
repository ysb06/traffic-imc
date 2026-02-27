import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Union
import yaml


@dataclass
class PathConfig:
    """Path configuration using dataclass"""

    # Root directory
    root_dir_path: str

    # Core dataset file paths
    traffic_imc_path: str
    traffic_imc_missing_path: str
    traffic_imc_training_path: str
    traffic_imc_training_missing_path: str
    traffic_imc_test_path: str
    traffic_imc_test_missing_path: str
    sensor_ids_path: str
    metadata_path: str
    sensor_locations_path: str
    distances_path: str
    adj_mx_path: str

    # Nodelink paths
    nodelink_dir_path: str
    nodelink_node_path: str
    nodelink_link_path: str
    nodelink_turn_path: str

    # IMCRTS paths
    imcrts_dir_path: str
    imcrts_path: str

    # Miscellaneous paths
    misc_dir_path: str
    imcrts_excel_path: str
    traffic_excel_path: str
    traffic_shapefile_path: str
    distances_shapefile_path: str

    # Path Raws
    raw: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "PathConfig":
        """Create PathConfig from YAML file"""
        with open(config_path, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        return cls._build_from_config(config)

    @classmethod
    def _build_from_config(cls, config: Dict[str, Any]) -> "PathConfig":
        """Build PathConfig from configuration dictionary"""
        root_dir = Path(config["root_dir"])

        # Core dataset file paths
        dataset_filenames = config["dataset"]["filenames"]
        traffic_imc_path = root_dir / dataset_filenames["traffic_imc"]
        traffic_imc_missing_path = root_dir / dataset_filenames["traffic_imc_missing"]
        traffic_imc_training_path = root_dir / dataset_filenames["traffic_imc_training"]
        traffic_imc_training_missing_path = root_dir / dataset_filenames["traffic_imc_training_missing"]
        traffic_imc_test_path = root_dir / dataset_filenames["traffic_imc_test"]
        traffic_imc_test_missing_path = root_dir / dataset_filenames["traffic_imc_test_missing"]
        sensor_ids_path = root_dir / dataset_filenames["sensor_ids"]
        metadata_path = root_dir / dataset_filenames["metadata"]
        sensor_locations_path = root_dir / dataset_filenames["sensor_locations"]
        distances_path = root_dir / dataset_filenames["distances"]
        adj_mx_path = root_dir / dataset_filenames["adjacency_matrix"]
        # Nodelink paths
        nodelink_dir = root_dir / config["nodelink"]["dir"]
        nodelink_filenames = config["nodelink"]["filenames"]
        nodelink_node_path = nodelink_dir / nodelink_filenames["node"]
        nodelink_link_path = nodelink_dir / nodelink_filenames["link"]
        nodelink_turn_path = nodelink_dir / nodelink_filenames["turn"]

        # IMCRTS paths
        imcrts_dir = root_dir / config["imcrts"]["dir"]
        imcrts_filenames = config["imcrts"]["filenames"]
        imcrts_path = imcrts_dir / imcrts_filenames["data"]

        # Miscellaneous paths
        misc_dir = root_dir / config["misc"]["dir"]
        misc_filenames = config["misc"]["filenames"]
        imcrts_excel_path = misc_dir / misc_filenames["imcrts_excel"]
        traffic_excel_path = misc_dir / misc_filenames["traffic_excel"]
        traffic_shapefile_path = misc_dir / misc_filenames["traffic_shape"]
        distances_shapefile_path = misc_dir / misc_filenames["distances_shape"]

        return cls(
            root_dir_path=str(root_dir),
            traffic_imc_path=str(traffic_imc_path),
            traffic_imc_missing_path=str(traffic_imc_missing_path),
            traffic_imc_training_path=str(traffic_imc_training_path),
            traffic_imc_training_missing_path=str(traffic_imc_training_missing_path),
            traffic_imc_test_path=str(traffic_imc_test_path),
            traffic_imc_test_missing_path=str(traffic_imc_test_missing_path),
            sensor_ids_path=str(sensor_ids_path),
            metadata_path=str(metadata_path),
            sensor_locations_path=str(sensor_locations_path),
            distances_path=str(distances_path),
            adj_mx_path=str(adj_mx_path),
            nodelink_dir_path=str(nodelink_dir),
            nodelink_node_path=str(nodelink_node_path),
            nodelink_link_path=str(nodelink_link_path),
            nodelink_turn_path=str(nodelink_turn_path),
            imcrts_dir_path=str(imcrts_dir),
            imcrts_path=str(imcrts_path),
            misc_dir_path=str(misc_dir),
            imcrts_excel_path=str(imcrts_excel_path),
            traffic_excel_path=str(traffic_excel_path),
            traffic_shapefile_path=str(traffic_shapefile_path),
            distances_shapefile_path=str(distances_shapefile_path),
            raw=config,
        )

    def create_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.misc_dir_path,
            self.nodelink_dir_path,
            self.imcrts_dir_path,
        ]

        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
