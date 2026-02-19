import logging
import os
from typing import List, Optional

import geopandas as gpd
import networkx as nx
import pandas as pd

from .components import (
    AdjacencyMatrix,
    DistancesImc,
    IdList,
    Metadata,
    SensorLocations,
    TrafficData,
    MissingMasks,
)
from .components.metr_imc.outlier import OutlierProcessor
from .components.metr_imc.interpolation import Interpolator
from .components.metr_imc.outlier.base import (
    SimpleAbsoluteOutlierProcessor,
    RemovingWeirdZeroOutlierProcessor,
)
from .imcrts.collector import IMCRTSCollector
from .nodelink.converter import NodeLink
from .nodelink.downloader import download_nodelink
from .utils import PathConfig

logger = logging.getLogger(__name__)
PATH_CONF = PathConfig.from_yaml("./configs/config.yaml")
PATH_CONF.create_directories()


# Other Settings
NODELINK_RAW_URL = (
    "https://www.its.go.kr/opendata/nodelinkFileSDownload/DF_180/0"  # 2022-12-28
)
TARGET_REGION_CODES = [
    "161",
    "162",
    "163",
    "164",
    "165",
    "166",
    "167",
    "168",
    "169",
]  # All Incheon Regions
IMCRTS_START_DATE = "20230126"
IMCRTS_END_DATE = "20260115"
TRAINING_END_DATE = "2025-11-30 23:59:59"


# def generate_subset_dataset(
#     target_nodelink: list[str],
#     save_dir_path: str,
# ):
#     metr_imc_filename = PATH_CONF.raw["dataset"]["filenames"]["metr_imc"]
#     sensor_ids_filename = PATH_CONF.raw["dataset"]["filenames"]["sensor_ids"]
#     metadata_filename = PATH_CONF.raw["dataset"]["filenames"]["metadata"]
#     sensor_locations_filename = PATH_CONF.raw["dataset"]["filenames"][
#         "sensor_locations"
#     ]
#     distances_filename = PATH_CONF.raw["dataset"]["filenames"]["distances"]
#     adjacency_matrix_filename = PATH_CONF.raw["dataset"]["filenames"][
#         "adjacency_matrix"
#     ]

#     metr_imc_save_path = os.path.join(save_dir_path, metr_imc_filename)
#     sensor_ids_save_path = os.path.join(save_dir_path, sensor_ids_filename)
#     metadata_save_path = os.path.join(save_dir_path, metadata_filename)
#     sensor_locations_save_path = os.path.join(save_dir_path, sensor_locations_filename)
#     distances_save_path = os.path.join(save_dir_path, distances_filename)
#     adj_mx_save_path = os.path.join(save_dir_path, adjacency_matrix_filename)

#     traffic_data = TrafficData.import_from_hdf(PATH_CONF.metr_imc_path)
#     wz_outlier_processor = RemovingWeirdZeroOutlierProcessor()
#     traffic_data.data = wz_outlier_processor.process(traffic_data.data)
#     traffic_data.select_sensors(target_nodelink)
#     traffic_data.to_hdf(metr_imc_save_path)

#     generate_dataset(
#         traffic_data_path=metr_imc_save_path,
#         ids_output_path=sensor_ids_save_path,
#         metadata_output_path=metadata_save_path,
#         sensor_locations_output_path=sensor_locations_save_path,
#         distances_output_path=distances_save_path,
#         adj_mx_output_path=adj_mx_save_path,
#     )


def generate_raw_dataset(api_key: Optional[str] = None):
    # Generating Core Files
    generate_nodelink_raw()
    generate_imcrts_raw(api_key=api_key)
    generate_metr_imc_raw()
    generate_dataset()

    # Generating Misc
    generate_metr_imc_shapefile()
    generate_distances_shapefile()

    # Generating excel files
    generate_metr_imc_excel()


def generate_subset(
    subset_path_conf: PathConfig,
    target_nodelinks_path: Optional[str] = None,
    target_data_start: Optional[str] = None,
    target_data_end: Optional[str] = None,
    cluster_count: Optional[int] = 1,
    missing_rate_threshold: float = 0.9,
    outlier_processors: Optional[List[OutlierProcessor]] = None,
    interpolation_processors: Optional[List[Interpolator]] = None,
):
    """
    Generate a subset dataset by spatial/temporal filtering from the raw dataset.

    Args:
        subset_path_conf: PathConfig defining subset dataset output paths.
        target_nodelinks_path: Road shapefile path for spatial filtering (None = all roads).
        target_data_start: Start date for temporal filtering (None = full range).
        target_data_end: End date for temporal filtering (None = full range).
    """
    # 1. Create directories
    subset_path_conf.create_directories()
    logger.info(f"Generating subset dataset at: {subset_path_conf.root_dir_path}")

    # 2. Load full raw dataset
    logger.info("Loading raw METR-IMC data...")
    traffic_data = TrafficData.import_from_hdf(PATH_CONF.metr_imc_path)
    df = traffic_data.data

    adj_mx_raw = AdjacencyMatrix.import_from_pickle(PATH_CONF.adj_mx_path)
    adj_mx = adj_mx_raw.adj_mx
    G = nx.from_numpy_array(adj_mx)
    g_idx_to_sensor = {value: key for key, value in adj_mx_raw.sensor_id_to_idx.items()}
    logger.info(f"Original data: {len(df)} rows, {len(df.columns)} sensors")

    # 4. Spatial filtering
    # 4.1 Extract LINK_ID from shapefile, then select columns directly
    if target_nodelinks_path:
        logger.info(f"Filtering sensors from shapefile: {target_nodelinks_path}")
        target_roads = gpd.read_file(target_nodelinks_path)
        target_link_ids = target_roads["LINK_ID"].tolist()
        # Keep only intersecting columns (ignore nonexistent LINK_IDs)
        valid_link_ids = [lid for lid in target_link_ids if lid in df.columns]
        df = df[valid_link_ids]
        logger.info(f"After spatial filtering: {len(df.columns)} sensors")

    # 4.2 Keep only nodes in the largest connected components in adjacency graph
    if cluster_count is not None and cluster_count > 0:
        # Get indices of sensors currently remaining in df
        sensor_to_g_idx = adj_mx_raw.sensor_id_to_idx
        current_sensor_ids = set(df.columns)
        current_node_indices = [
            sensor_to_g_idx[sid] for sid in current_sensor_ids if sid in sensor_to_g_idx
        ]

        # Build subgraph from current sensors
        subgraph = G.subgraph(current_node_indices).copy()
        connected_components = list(nx.connected_components(subgraph))

        # Sort by size (largest first)
        connected_components.sort(key=len, reverse=True)
        logger.info(
            f"Found {len(connected_components)} connected components, "
            f"sizes: {[len(c) for c in connected_components[:5]]}..."
        )

        # Select top-N components by size
        selected_components = connected_components[:cluster_count]
        logger.info(
            f"Selected top {cluster_count} component(s) with sizes: "
            f"{[len(c) for c in selected_components]}"
        )

        # Convert selected node indices back to sensor_id
        selected_node_indices = set()
        for component in selected_components:
            selected_node_indices.update(component)

        selected_sensor_ids = [
            g_idx_to_sensor[idx]
            for idx in selected_node_indices
            if idx in g_idx_to_sensor
        ]

        # Filter DataFrame to selected sensors only
        valid_sensor_ids = [sid for sid in selected_sensor_ids if sid in df.columns]
        df = df[valid_sensor_ids]
        logger.info(
            f"After cluster filtering: {len(df.columns)} sensors "
            f"from {len(selected_components)} component(s)"
        )

    # 5. Temporal filtering (based on DatetimeIndex)
    if target_data_start:
        df = df.loc[target_data_start:]

    if target_data_end:
        df = df.loc[:target_data_end]

    if target_data_start or target_data_end:
        logger.info(f"Filtering time range: {target_data_start} ~ {target_data_end}")
        logger.info(f"After temporal filtering: {len(df)} rows")

    # 6. Base data correction
    # 6.1 Missing-rate filtering (computed on current DataFrame)
    if missing_rate_threshold < 1.0:
        logger.info(
            f"Filtering sensors by missing rate threshold: {missing_rate_threshold * 100:.1f}%"
        )
        missing_mask = df.isna()
        sensor_missing_counts = missing_mask.sum()
        sensor_missing_rates = sensor_missing_counts / len(df)

        # Keep sensors whose missing rate is below threshold
        filtered_sensors = sensor_missing_rates[
            sensor_missing_rates < missing_rate_threshold
        ].index.tolist()
        df = df[filtered_sensors]
        logger.info(
            f"After missing rate filtering: {len(df.columns)} sensors (removed {len(sensor_missing_rates) - len(filtered_sensors)} sensors)"
        )

    # 6.2 Default outlier processing and interpolation
    logger.info("Processing default outliers...")
    default_outlier_processors: List[OutlierProcessor] = [
        SimpleAbsoluteOutlierProcessor(threshold=3450),
        RemovingWeirdZeroOutlierProcessor(),
    ]
    for processor in default_outlier_processors:
        df = processor.process(df)

    # 6.3 Create base missing mask (before interpolation)
    logger.info("Creating missing masks before interpolation...")
    missing_mask = MissingMasks.import_from_traffic_data_frame(df)

    # 7. Update filtered data
    traffic_data.data = df

    # 8. Split train/test by time (before interpolation)
    logger.info("Splitting train/test data (before interpolation)...")
    training_df = df.loc[:TRAINING_END_DATE].copy()
    test_df_raw = df.loc[TRAINING_END_DATE:].copy()
    logger.info(
        f"Training data: {len(training_df)} rows, Test data: {len(test_df_raw)} rows"
    )

    # 9. Split train/test missing masks (before interpolation)
    training_missing = missing_mask.data.loc[training_df.index, training_df.columns]
    test_missing = missing_mask.data.loc[test_df_raw.index, test_df_raw.columns]

    # 10. Apply requested outlier/interpolation processing on training data
    logger.info("Processing outliers and interpolation on training data only...")
    training_traffic_data = TrafficData(training_df)
    _apply_outlier_and_interpolation_inplace(
        traffic_data=training_traffic_data,
        outlier_processors=outlier_processors,
        interpolation_processors=interpolation_processors,
    )

    # 11. Apply requested outlier/interpolation processing on full data
    logger.info("Processing outliers and interpolation on full data...")
    full_traffic_data = TrafficData(df.copy())
    _apply_outlier_and_interpolation_inplace(
        traffic_data=full_traffic_data,
        outlier_processors=outlier_processors,
        interpolation_processors=interpolation_processors,
    )

    # 12. Extract test split from interpolated full data
    logger.info("Extracting test data from interpolated full data...")
    test_df_interpolated = full_traffic_data.data.loc[TRAINING_END_DATE:].copy()
    test_traffic_data = TrafficData(test_df_interpolated)
    # TODO: Refactor so all outputs are saved only after all processing steps finish
    # 13. Save all processed datasets
    logger.info("Saving all processed datasets...")
    # 13.1 Full dataset (new_raw_data)
    logger.info(f"Saving interpolated full data to {subset_path_conf.metr_imc_path}")
    full_traffic_data.to_hdf(subset_path_conf.metr_imc_path)
    missing_mask.to_hdf(subset_path_conf.metr_imc_missing_path)

    # 13.2 Training dataset
    logger.info(f"Saving training data to {subset_path_conf.metr_imc_training_path}")
    training_traffic_data.to_hdf(subset_path_conf.metr_imc_training_path)
    MissingMasks(training_missing).to_hdf(
        subset_path_conf.metr_imc_training_missing_path
    )

    # 13.3 Test dataset
    logger.info(f"Saving test data to {subset_path_conf.metr_imc_test_path}")
    test_traffic_data.to_hdf(subset_path_conf.metr_imc_test_path)
    MissingMasks(test_missing).to_hdf(subset_path_conf.metr_imc_test_missing_path)

    # 14. Call generate_dataset() using subset PathConfig paths
    logger.info("Generating dataset components...")
    generate_dataset(
        traffic_data_path=subset_path_conf.metr_imc_path,
        nodelink_link_path=PATH_CONF.nodelink_link_path,  # Use raw path
        nodelink_turn_path=PATH_CONF.nodelink_turn_path,  # Use raw path
        ids_output_path=subset_path_conf.sensor_ids_path,
        metadata_output_path=subset_path_conf.metadata_path,
        sensor_locations_output_path=subset_path_conf.sensor_locations_path,
        distances_output_path=subset_path_conf.distances_path,
        adj_mx_output_path=subset_path_conf.adj_mx_path,
    )

    # 15. Generate shapefiles
    logger.info("Generating shapefiles...")
    generate_metr_imc_shapefile(
        metr_imc_path=subset_path_conf.metr_imc_path,
        node_link_path=PATH_CONF.nodelink_link_path,
        output_path=subset_path_conf.metr_shapefile_path,
    )

    generate_distances_shapefile(
        distances_path=subset_path_conf.distances_path,
        sensor_locations_path=subset_path_conf.sensor_locations_path,
        output_path=subset_path_conf.distances_shapefile_path,
    )

    logger.info(
        f"Subset dataset generation completed: {subset_path_conf.root_dir_path}"
    )


# ------------------------------------------------------------------------------ #


def _apply_outlier_and_interpolation_inplace(
    traffic_data: TrafficData,
    outlier_processors: Optional[List[OutlierProcessor]],
    interpolation_processors: Optional[List[Interpolator]],
):
    """
    Apply outlier processing and interpolation in-place to a TrafficData object.
    """
    df = traffic_data.data

    if outlier_processors:
        logger.info("Processing outliers...")
        for processor in outlier_processors:
            df = processor.process(df)

    if interpolation_processors:
        logger.info("Processing interpolation...")
        for processor in interpolation_processors:
            df = processor.interpolate(df)

    traffic_data.data = df


def generate_metr_imc_excel(
    metr_imc_path: str = PATH_CONF.metr_imc_path,
    output_dir: str = PATH_CONF.misc_dir_path,
    max_rows_per_file: int = 1000000,
):
    """
    Save `metr_imc.h5` data to Excel.
    If row count exceeds Excel limit (1,048,576), split into multiple files.

    Args:
        metr_imc_path: HDF5 file path.
        output_dir: Output directory (if None, use HDF5 file directory).
        max_rows_per_file: Maximum rows per file (default: 1,000,000).
    """
    logger.info("Loading METR-IMC data from HDF5...")
    traffic_data = TrafficData.import_from_hdf(metr_imc_path)
    df = traffic_data.data

    # Configure output directory
    if output_dir is None:
        output_dir = os.path.dirname(metr_imc_path)

    total_rows = len(df)
    logger.info(f"Total rows: {total_rows}, Total sensors: {len(df.columns)}")

    # Save as a single file when under Excel row limit
    if total_rows <= max_rows_per_file:
        output_path = os.path.join(output_dir, "metr-imc.xlsx")
        logger.info(f"Saving to {output_path}...")
        df.to_excel(output_path, engine="openpyxl")
        logger.info("Excel file saved successfully")
    else:
        # Split into multiple files
        num_files = (total_rows + max_rows_per_file - 1) // max_rows_per_file
        logger.info(f"Data exceeds Excel limit. Splitting into {num_files} files...")

        for i in range(num_files):
            start_idx = i * max_rows_per_file
            end_idx = min((i + 1) * max_rows_per_file, total_rows)
            df_chunk = df.iloc[start_idx:end_idx]

            output_path = os.path.join(output_dir, f"metr-imc_part{i+1:02d}.xlsx")
            logger.info(
                f"Saving part {i+1}/{num_files} ({end_idx - start_idx} rows) to {output_path}..."
            )
            df_chunk.to_excel(output_path, engine="openpyxl")

        logger.info(f"All {num_files} Excel files saved successfully")


def generate_distances_shapefile(
    distances_path: str = PATH_CONF.distances_path,
    sensor_locations_path: str = PATH_CONF.sensor_locations_path,
    output_path: str = PATH_CONF.distances_shapefile_path,
):
    distances = DistancesImc.import_from_csv(distances_path)
    sensor_locations = SensorLocations.import_from_csv(sensor_locations_path)
    distances.to_shapefile(sensor_locations.data, filepath=output_path)


def generate_dataset(
    # Inputs
    traffic_data_path: str = PATH_CONF.metr_imc_path,
    nodelink_link_path: str = PATH_CONF.nodelink_link_path,
    nodelink_turn_path: str = PATH_CONF.nodelink_turn_path,
    # Outputs
    ids_output_path: str = PATH_CONF.sensor_ids_path,
    metadata_output_path: str = PATH_CONF.metadata_path,
    sensor_locations_output_path: str = PATH_CONF.sensor_locations_path,
    distances_output_path: str = PATH_CONF.distances_path,
    adj_mx_output_path: str = PATH_CONF.adj_mx_path,
):
    traffic_data = TrafficData.import_from_hdf(traffic_data_path)

    # Sensor IDs
    metr_ids = IdList(traffic_data.data.columns.to_list())
    metr_ids.to_txt(ids_output_path)

    # Metadata
    metadata = Metadata.import_from_nodelink(nodelink_link_path)
    metadata.sensor_filter = metr_ids.data  # TODO: revisit this behavior
    metadata.to_hdf(metadata_output_path)

    # Sensor Locations
    sensor_locations = SensorLocations.import_from_nodelink(nodelink_link_path)
    sensor_locations.sensor_filter = metr_ids.data
    sensor_locations.to_csv(sensor_locations_output_path)

    # Distances
    distances = DistancesImc.import_from_nodelink(
        nodelink_link_path,
        nodelink_turn_path,
        target_ids=metr_ids.data,
        distance_limit=9000,
    )
    distances.to_csv(distances_output_path)

    # Adjacency Matrix
    adj_mx: AdjacencyMatrix = AdjacencyMatrix.import_from_components(
        metr_ids, distances
    )
    adj_mx.to_pickle(adj_mx_output_path)


def generate_nodelink_raw(
    nodelink_url: str = NODELINK_RAW_URL,
    region_codes: list[str] = TARGET_REGION_CODES,
    download_target_dir: str = PATH_CONF.nodelink_dir_path,
    node_output_path: str = PATH_CONF.nodelink_node_path,
    link_output_path: str = PATH_CONF.nodelink_link_path,
    turn_output_path: str = PATH_CONF.nodelink_turn_path,
):
    logger.info("Downloading Node-Link Data...")
    nodelink_raw_path = download_nodelink(download_target_dir, nodelink_url)
    nodelink_data = NodeLink(nodelink_raw_path).filter_by_gu_codes(region_codes)
    nodelink_data.export(
        node_output_path=node_output_path,
        link_output_path=link_output_path,
        turn_output_path=turn_output_path,
    )
    logger.info("Downloading Done")


def generate_imcrts_raw(
    api_key: Optional[str] = None,
    start_date: str = IMCRTS_START_DATE,
    end_date: str = IMCRTS_END_DATE,
    imcrts_output_path: str = PATH_CONF.imcrts_path,
):
    logger.info("Collecting IMCRTS Data...")
    resolved_api_key = api_key or os.environ.get("DATA_API_KEY")
    if resolved_api_key is None:
        raise ValueError(
            "API key is missing. Pass `--api-key` or set the `DATA_API_KEY` environment variable."
        )

    collector = IMCRTSCollector(
        api_key=resolved_api_key,
        start_date=start_date,
        end_date=end_date,
    )
    collector.collect(ignore_empty=True)
    collector.to_pickle(imcrts_output_path)
    logger.info("Collecting Done")


def generate_metr_imc_raw(
    # Inputs
    road_data_path: str = PATH_CONF.nodelink_link_path,
    traffic_data_path: str = PATH_CONF.imcrts_path,
    # Outputs
    metr_imc_path: str = PATH_CONF.metr_imc_path,
    metr_imc_missing_path: str = PATH_CONF.metr_imc_missing_path,
):
    road_data: gpd.GeoDataFrame = gpd.read_file(road_data_path)
    traffic_data = TrafficData.import_from_pickle(traffic_data_path)

    logger.info("Matching Link IDs...")
    traffic_data.select_sensors(road_data["LINK_ID"].tolist())
    logger.info(f"Saving Traffic Data to {metr_imc_path}...")
    traffic_data.to_hdf(metr_imc_path)
    missing_masks = MissingMasks.import_from_traffic_data(traffic_data)
    logger.info(f"Saving Missing Masks to {metr_imc_missing_path}...")
    missing_masks.to_hdf(metr_imc_missing_path)
    logger.info("Matching Done")


def generate_metr_imc_shapefile(
    metr_imc_path: str = PATH_CONF.metr_imc_path,
    node_link_path: str = PATH_CONF.nodelink_link_path,
    output_path: str = PATH_CONF.metr_shapefile_path,
):
    traffic_data = TrafficData.import_from_hdf(metr_imc_path)
    road_data: gpd.GeoDataFrame = gpd.read_file(node_link_path)
    traffic_link_ids = set(traffic_data.data.columns)
    filtered_roads = road_data[road_data["LINK_ID"].isin(traffic_link_ids)].copy()
    filtered_roads.to_file(output_path)


def split_train_test_data(
    raw_dataset_path: str = PATH_CONF.metr_imc_path,
    raw_missing_path: str = PATH_CONF.metr_imc_missing_path,
    training_dataset_path: str = PATH_CONF.metr_imc_training_path,
    training_missing_path: str = PATH_CONF.metr_imc_training_missing_path,
    test_dataset_path: str = PATH_CONF.metr_imc_test_path,
    test_missing_path: str = PATH_CONF.metr_imc_test_missing_path,
    training_end_date: str = TRAINING_END_DATE,
):
    traffic_data = TrafficData.import_from_hdf(raw_dataset_path)
    missing_masks = MissingMasks.import_from_hdf(raw_missing_path)
    df = traffic_data.data

    training_data = df.loc[:training_end_date]
    test_data = df.loc[training_end_date:]

    training_missing = missing_masks.data.loc[
        training_data.index[0] : training_data.index[-1], training_data.columns
    ]
    test_missing = missing_masks.data.loc[
        test_data.index[0] : test_data.index[-1], test_data.columns
    ]

    training_traffic_data = TrafficData(training_data)
    test_traffic_data = TrafficData(test_data)

    training_missing_masks = MissingMasks(training_missing)
    test_missing_masks = MissingMasks(test_missing)

    logger.info(f"Saving training data to {training_dataset_path}...")
    training_traffic_data.to_hdf(training_dataset_path)
    logger.info(f"Saving training missing masks to {training_missing_path}...")
    training_missing_masks.to_hdf(training_missing_path)
    logger.info(f"Saving test data to {test_dataset_path}...")
    test_traffic_data.to_hdf(test_dataset_path)
    logger.info(f"Saving test missing masks to {test_missing_path}...")
    test_missing_masks.to_hdf(test_missing_path)
