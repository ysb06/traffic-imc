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
from .components.traffic_imc.outlier import OutlierProcessor
from .components.traffic_imc.interpolation import Interpolator
from .components.traffic_imc.outlier.base import (
    RemovingWeirdZeroOutlierProcessor,
    TrafficCapacityAbsoluteOutlierProcessor,
)
from .imcrts.collector import IMCRTSCollector
from .nodelink.converter import NodeLink
from .nodelink.downloader import download_nodelink
from .utils import PathConfig

logger = logging.getLogger(__name__)


# Other Settings
NODELINK_RAW_URL = "https://www.its.go.kr/opendata/nodelinkFileSDownload/DF_180/0"
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
IMCRTS_END_DATE = "20260425"
TRAINING_END_DATE = "2026-01-25 23:59:59"


def generate_raw_dataset(raw_path_conf: PathConfig, api_key: Optional[str] = None):
    # Generating Core Files
    generate_nodelink_raw(
        download_target_dir=raw_path_conf.nodelink_dir_path,
        node_output_path=raw_path_conf.nodelink_node_path,
        link_output_path=raw_path_conf.nodelink_link_path,
        turn_output_path=raw_path_conf.nodelink_turn_path,
    )
    generate_imcrts_raw(api_key=api_key, imcrts_output_path=raw_path_conf.imcrts_path)
    generate_traffic_imc_raw(
        road_data_path=raw_path_conf.nodelink_link_path,
        traffic_data_path=raw_path_conf.imcrts_path,
        traffic_imc_path=raw_path_conf.traffic_imc_path,
        traffic_imc_missing_path=raw_path_conf.traffic_imc_missing_path,
    )
    generate_dataset(
        traffic_data_path=raw_path_conf.traffic_imc_path,
        nodelink_link_path=raw_path_conf.nodelink_link_path,
        nodelink_turn_path=raw_path_conf.nodelink_turn_path,
        ids_output_path=raw_path_conf.sensor_ids_path,
        metadata_output_path=raw_path_conf.metadata_path,
        sensor_locations_output_path=raw_path_conf.sensor_locations_path,
        distances_output_path=raw_path_conf.distances_path,
        adj_mx_output_path=raw_path_conf.adj_mx_path,
    )

    # Generating Misc
    generate_traffic_imc_shapefile(
        traffic_imc_path=raw_path_conf.traffic_imc_path,
        node_link_path=raw_path_conf.nodelink_link_path,
        output_path=raw_path_conf.traffic_shapefile_path,
    )
    generate_distances_shapefile(
        distances_path=raw_path_conf.distances_path,
        sensor_locations_path=raw_path_conf.sensor_locations_path,
        output_path=raw_path_conf.distances_shapefile_path,
    )

    # Generating excel files
    generate_traffic_imc_excel(
        traffic_imc_path=raw_path_conf.traffic_imc_path,
        output_dir=raw_path_conf.misc_dir_path,
    )


def generate_subset(
    subset_path_conf: PathConfig,
    raw_path_conf: PathConfig,
    target_nodelinks_path: Optional[str] = None,
    target_data_start: Optional[str] = None,
    target_data_end: Optional[str] = None,
    cluster_count: Optional[int] = 1,
    missing_rate_threshold: float = 0.9,
    outlier_processors: Optional[List[OutlierProcessor]] = None,
    interpolation_processors: Optional[List[Interpolator]] = None,
):
    # 1. Create directories
    subset_path_conf.create_directories()
    logger.info(f"Generating subset dataset at: {subset_path_conf.root_dir_path}")

    # 2. Load full raw dataset
    logger.info("Loading raw METR-IMC data...")
    traffic_data = TrafficData.import_from_hdf(raw_path_conf.traffic_imc_path)
    df = traffic_data.data

    adj_mx_raw = AdjacencyMatrix.import_from_pickle(raw_path_conf.adj_mx_path)
    adj_mx = adj_mx_raw.adj_mx
    G = nx.from_numpy_array(adj_mx)
    g_idx_to_sensor = {value: key for key, value in adj_mx_raw.sensor_id_to_idx.items()}
    logger.info(f"Original data: {len(df)} rows, {len(df.columns)} sensors")

    # 3. Spatial filtering (Extract LINK_ID from shapefile, then select columns directly)
    if target_nodelinks_path:
        logger.info(f"Filtering sensors from shapefile: {target_nodelinks_path}")
        target_roads = gpd.read_file(target_nodelinks_path)
        target_link_ids = target_roads["LINK_ID"].tolist()
        valid_link_ids = [lid for lid in target_link_ids if lid in df.columns]
        df = df[valid_link_ids]
        logger.info(f"After spatial filtering: {len(df.columns)} sensors")

    # 4. Spatial filtering (Keep only nodes in the largest connected components in adjacency graph)
    if cluster_count is not None and cluster_count > 0:
        sensor_to_g_idx = adj_mx_raw.sensor_id_to_idx
        current_sensor_ids = set(df.columns)
        current_node_indices = [
            sensor_to_g_idx[sid] for sid in current_sensor_ids if sid in sensor_to_g_idx
        ]

        subgraph = G.subgraph(current_node_indices).copy()
        connected_components = list(nx.connected_components(subgraph))

        connected_components.sort(key=len, reverse=True)
        logger.info(
            f"Found {len(connected_components)} connected components, "
            f"sizes: {[len(c) for c in connected_components[:5]]}..."
        )

        selected_components = connected_components[:cluster_count]
        logger.info(
            f"Selected top {cluster_count} component(s) with sizes: "
            f"{[len(c) for c in selected_components]}"
        )

        selected_node_indices = set()
        for component in selected_components:
            selected_node_indices.update(component)

        selected_sensor_ids = [
            g_idx_to_sensor[idx]
            for idx in selected_node_indices
            if idx in g_idx_to_sensor
        ]

        valid_sensor_ids = [sid for sid in selected_sensor_ids if sid in df.columns]
        df = df[valid_sensor_ids]
        logger.info(
            f"After cluster filtering: {len(df.columns)} sensors "
            f"from {len(selected_components)} component(s)"
        )

    # 5. Temporal filtering
    if target_data_start:
        df = df.loc[target_data_start:]

    if target_data_end:
        df = df.loc[:target_data_end]

    if target_data_start or target_data_end:
        logger.info(f"Filtering time range: {target_data_start} ~ {target_data_end}")
        logger.info(f"After temporal filtering: {len(df)} rows")

    # 6. Base data correction
    # 6.1 Missing-rate filtering
    if missing_rate_threshold < 1.0:
        logger.info(
            f"Filtering sensors by missing rate threshold: {missing_rate_threshold * 100:.1f}%"
        )
        missing_mask = df.isna()
        sensor_missing_counts = missing_mask.sum()
        sensor_missing_rates = sensor_missing_counts / len(df)

        filtered_sensors = sensor_missing_rates[
            sensor_missing_rates < missing_rate_threshold
        ].index.tolist()
        df = df[filtered_sensors]
        logger.info(
            f"After missing rate filtering: {len(df.columns)} sensors (removed {len(sensor_missing_rates) - len(filtered_sensors)} sensors)"
        )

    # 6.2 Prepare outlier processors.
    logger.info("Preparing outlier processors...")
    road_metadata = Metadata.import_from_nodelink(raw_path_conf.nodelink_link_path)
    lane_counts = (
        road_metadata.data[road_metadata.data["LINK_ID"].isin(df.columns)]
        .set_index("LINK_ID")["LANES"]
        .to_dict()
    )
    effective_outlier_processors: List[OutlierProcessor] = [
        TrafficCapacityAbsoluteOutlierProcessor(lane_counts=lane_counts),
        RemovingWeirdZeroOutlierProcessor(),
    ]
    if outlier_processors:
        effective_outlier_processors.extend(outlier_processors)

    # 7. Split train/test by time before split-local outlier processing and interpolation.
    logger.info("Splitting train/test data...")
    split_ts = pd.Timestamp(TRAINING_END_DATE)
    training_df = df.loc[df.index <= split_ts].copy()
    test_df_raw = df.loc[df.index > split_ts].copy()
    logger.info(
        f"Training data: {len(training_df)} rows, Test data: {len(test_df_raw)} rows"
    )

    # 8. Apply outlier processing
    training_traffic_data, training_missing = _process_split(
        split_name="training",
        df=training_df,
        outlier_processors=effective_outlier_processors,
        interpolation_processors=interpolation_processors,
    )

    test_traffic_data, test_missing = _process_split(
        split_name="test",
        df=test_df_raw,
        outlier_processors=effective_outlier_processors,
        interpolation_processors=interpolation_processors,
    )

    # 9. Reconstruct the full dataset and full invalid-value mask from independently processed splits.
    logger.info("Combining independently processed train/test splits...")
    full_df_interpolated = pd.concat(
        [training_traffic_data.data, test_traffic_data.data],
        axis=0,
    ).sort_index()
    full_traffic_data = TrafficData(full_df_interpolated)
    full_missing = pd.concat(
        [training_missing.data, test_missing.data],
        axis=0,
    ).sort_index()

    assert training_missing.data.shape == training_traffic_data.data.shape
    assert test_missing.data.shape == test_traffic_data.data.shape
    assert full_missing.shape == full_traffic_data.data.shape
    assert full_missing.index.equals(full_traffic_data.data.index)
    assert list(full_missing.columns) == list(full_traffic_data.data.columns)

    # 10. Save all processed datasets
    logger.info("Saving all processed datasets...")
    # 10.1 Full dataset
    logger.info(f"Saving interpolated full data to {subset_path_conf.traffic_imc_path}")
    full_traffic_data.to_hdf(subset_path_conf.traffic_imc_path)
    MissingMasks(full_missing).to_hdf(subset_path_conf.traffic_imc_missing_path)

    # 10.2 Training dataset
    logger.info(f"Saving training data to {subset_path_conf.traffic_imc_training_path}")
    training_traffic_data.to_hdf(subset_path_conf.traffic_imc_training_path)
    training_missing.to_hdf(subset_path_conf.traffic_imc_training_missing_path)

    # 10.3 Test dataset
    logger.info(f"Saving test data to {subset_path_conf.traffic_imc_test_path}")
    test_traffic_data.to_hdf(subset_path_conf.traffic_imc_test_path)
    test_missing.to_hdf(subset_path_conf.traffic_imc_test_missing_path)

    # 11. Call generate_dataset() using subset PathConfig paths
    logger.info("Generating dataset components...")
    generate_dataset(
        traffic_data_path=subset_path_conf.traffic_imc_path,
        nodelink_link_path=raw_path_conf.nodelink_link_path,  # Use raw path
        nodelink_turn_path=raw_path_conf.nodelink_turn_path,  # Use raw path
        ids_output_path=subset_path_conf.sensor_ids_path,
        metadata_output_path=subset_path_conf.metadata_path,
        sensor_locations_output_path=subset_path_conf.sensor_locations_path,
        distances_output_path=subset_path_conf.distances_path,
        adj_mx_output_path=subset_path_conf.adj_mx_path,
    )

    # 12. Generate shapefiles
    logger.info("Generating shapefiles...")
    generate_traffic_imc_shapefile(
        traffic_imc_path=subset_path_conf.traffic_imc_path,
        node_link_path=raw_path_conf.nodelink_link_path,
        output_path=subset_path_conf.traffic_shapefile_path,
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


def _process_split(
    split_name: str,
    df: pd.DataFrame,
    outlier_processors: Optional[List[OutlierProcessor]],
    interpolation_processors: Optional[List[Interpolator]],
) -> tuple[TrafficData, MissingMasks]:
    traffic_data = TrafficData(df.copy())
    original_missing = traffic_data.data.isna()
    original_missing_count = int(original_missing.sum().sum())

    _apply_outliers_inplace(
        traffic_data=traffic_data,
        outlier_processors=outlier_processors,
    )

    invalid_after_outlier = traffic_data.data.isna()
    invalid_after_outlier_count = int(invalid_after_outlier.sum().sum())
    new_invalid_from_outlier_count = int(
        (invalid_after_outlier & ~original_missing).sum().sum()
    )

    missing_mask = MissingMasks.import_from_traffic_data_frame(traffic_data.data)

    _apply_interpolation_inplace(
        traffic_data=traffic_data,
        interpolation_processors=interpolation_processors,
    )

    remaining_nan_after_interpolation_count = int(traffic_data.data.isna().sum().sum())
    logger.info(
        "%s split missing mask stats: "
        "original_missing=%d, invalid_after_outlier=%d, "
        "new_invalid_from_outlier=%d, remaining_nan_after_interpolation=%d",
        split_name,
        original_missing_count,
        invalid_after_outlier_count,
        new_invalid_from_outlier_count,
        remaining_nan_after_interpolation_count,
    )

    return traffic_data, missing_mask


def _apply_outliers_inplace(
    traffic_data: TrafficData,
    outlier_processors: Optional[List[OutlierProcessor]],
) -> None:
    df = traffic_data.data

    if outlier_processors:
        logger.info("Processing outliers...")
        for processor in outlier_processors:
            df = processor.process(df)

    traffic_data.data = df


def _apply_interpolation_inplace(
    traffic_data: TrafficData,
    interpolation_processors: Optional[List[Interpolator]],
) -> None:
    df = traffic_data.data

    if interpolation_processors:
        logger.info("Processing interpolation...")
        for processor in interpolation_processors:
            df = processor.interpolate(df)
        df = _clip_negative_values(df)

    traffic_data.data = df


def _clip_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    negative_count = int((df < 0).sum().sum())
    if negative_count:
        logger.info("Clipping %d negative traffic values to 0.", negative_count)
        return df.mask(df < 0, 0)

    return df


def generate_traffic_imc_excel(
    traffic_imc_path: str,
    output_dir: str,
    max_rows_per_file: int = 1000000,
):
    logger.info("Loading METR-IMC data from HDF5...")
    traffic_data = TrafficData.import_from_hdf(traffic_imc_path)
    df = traffic_data.data

    total_rows = len(df)
    logger.info(f"Total rows: {total_rows}, Total sensors: {len(df.columns)}")

    if total_rows <= max_rows_per_file:
        output_path = os.path.join(output_dir, "traffic-imc.xlsx")
        logger.info(f"Saving to {output_path}...")
        df.to_excel(output_path, engine="openpyxl")
        logger.info("Excel file saved successfully")
    else:
        num_files = (total_rows + max_rows_per_file - 1) // max_rows_per_file
        logger.info(f"Data exceeds Excel limit. Splitting into {num_files} files...")

        for i in range(num_files):
            start_idx = i * max_rows_per_file
            end_idx = min((i + 1) * max_rows_per_file, total_rows)
            df_chunk = df.iloc[start_idx:end_idx]

            output_path = os.path.join(output_dir, f"traffic-imc_part{i+1:02d}.xlsx")
            logger.info(
                f"Saving part {i+1}/{num_files} ({end_idx - start_idx} rows) to {output_path}..."
            )
            df_chunk.to_excel(output_path, engine="openpyxl")

        logger.info(f"All {num_files} Excel files saved successfully")


def generate_distances_shapefile(
    distances_path: str,
    sensor_locations_path: str,
    output_path: str,
):
    distances = DistancesImc.import_from_csv(distances_path)
    sensor_locations = SensorLocations.import_from_csv(sensor_locations_path)
    distances.to_shapefile(sensor_locations.data, filepath=output_path)


def generate_dataset(
    # Inputs
    traffic_data_path: str,
    nodelink_link_path: str,
    nodelink_turn_path: str,
    # Outputs
    ids_output_path: str,
    metadata_output_path: str,
    sensor_locations_output_path: str,
    distances_output_path: str,
    adj_mx_output_path: str,
):
    traffic_data = TrafficData.import_from_hdf(traffic_data_path)

    # Sensor IDs
    metr_ids = IdList(traffic_data.data.columns.to_list())
    metr_ids.to_txt(ids_output_path)

    # Metadata
    metadata = Metadata.import_from_nodelink(nodelink_link_path)
    metadata.sensor_filter = metr_ids.data
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
    download_target_dir: str,
    node_output_path: str,
    link_output_path: str,
    turn_output_path: str,
    nodelink_url: str = NODELINK_RAW_URL,
    region_codes: list[str] = TARGET_REGION_CODES,
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
    imcrts_output_path: str,
    api_key: Optional[str] = None,
    start_date: str = IMCRTS_START_DATE,
    end_date: str = IMCRTS_END_DATE,
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


def generate_traffic_imc_raw(
    # Inputs
    road_data_path: str,
    traffic_data_path: str,
    # Outputs
    traffic_imc_path: str,
    traffic_imc_missing_path: str,
):
    road_data: gpd.GeoDataFrame = gpd.read_file(road_data_path)
    traffic_data = TrafficData.import_from_pickle(traffic_data_path)

    logger.info("Matching Link IDs...")
    traffic_data.select_sensors(road_data["LINK_ID"].tolist())
    logger.info(f"Saving Traffic Data to {traffic_imc_path}...")
    traffic_data.to_hdf(traffic_imc_path)
    missing_masks = MissingMasks.import_from_traffic_data(traffic_data)
    logger.info(f"Saving Missing Masks to {traffic_imc_missing_path}...")
    missing_masks.to_hdf(traffic_imc_missing_path)
    logger.info("Matching Done")


def generate_traffic_imc_shapefile(
    traffic_imc_path: str,
    node_link_path: str,
    output_path: str,
):
    traffic_data = TrafficData.import_from_hdf(traffic_imc_path)
    road_data: gpd.GeoDataFrame = gpd.read_file(node_link_path)
    traffic_link_ids = set(traffic_data.data.columns)
    filtered_roads = road_data[road_data["LINK_ID"].isin(traffic_link_ids)].copy()
    filtered_roads.to_file(output_path)


def split_train_test_data(
    raw_dataset_path: str,
    raw_missing_path: str,
    training_dataset_path: str,
    training_missing_path: str,
    test_dataset_path: str,
    test_missing_path: str,
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
