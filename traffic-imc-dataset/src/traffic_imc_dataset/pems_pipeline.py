import numpy as np
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from .components import TrafficData, AdjacencyMatrix

logger = logging.getLogger(__name__)


def create_pems_dataset(
    traffic_data: TrafficData,
    adj_mx: AdjacencyMatrix,
    output_path: str,
    include_tod: bool = True,
    include_dow: bool = True,
) -> None:
    """
    Create dataset.npz file for PEMS from TrafficData and AdjacencyMatrix.
    
    The output dataset will have sensors ordered according to AdjacencyMatrix's sensor_ids.
    This ensures consistency between the graph structure and time series data.
    
    Args:
        traffic_data: TrafficData object containing time series data
        adj_mx: AdjacencyMatrix object containing graph structure and sensor ordering
        output_path: Path to save dataset.npz file
        include_tod: Whether to include time-of-day feature (0-1 normalized hour)
        include_dow: Whether to include day-of-week feature (0-6 for Mon-Sun)
    
    Output format:
        dataset.npz with key 'data':
            shape: (Total_Timesteps, N_sensors, C_features)
            where C_features = 1 (traffic) + 1 (tod if enabled) + 1 (dow if enabled)
    """
    logger.info("Creating PEMS dataset from TrafficData and AdjacencyMatrix...")
    
    # Get sensor order from AdjacencyMatrix
    sensor_ids = adj_mx.sensor_ids
    n_sensors = len(sensor_ids)
    logger.info(f"Number of sensors: {n_sensors}")
    
    # Verify all sensors exist in TrafficData
    traffic_sensors = set(traffic_data.data.columns)
    missing_sensors = set(sensor_ids) - traffic_sensors
    if missing_sensors:
        raise ValueError(
            f"The following sensors in AdjacencyMatrix are not found in TrafficData: {missing_sensors}"
        )
    
    # Reorder traffic data according to AdjacencyMatrix sensor order
    ordered_traffic = traffic_data.data[sensor_ids]
    logger.info(f"Traffic data shape: {ordered_traffic.shape}")
    logger.info(f"Time range: {ordered_traffic.index[0]} to {ordered_traffic.index[-1]}")
    
    # Get time series as numpy array: (T, N)
    traffic_values = ordered_traffic.values  # shape: (T, N)
    n_timesteps = traffic_values.shape[0]
    
    # Initialize feature list
    features = [traffic_values[:, :, np.newaxis]]  # shape: (T, N, 1)
    feature_names = ['traffic']
    
    # Add time-of-day feature (0-1 normalized)
    if include_tod:
        time_index: pd.DatetimeIndex = ordered_traffic.index
        hours = time_index.hour + time_index.minute / 60.0
        tod = hours / 24.0  # Normalize to [0, 1)
        tod_feature = np.tile(tod[:, np.newaxis, np.newaxis], (1, n_sensors, 1))  # (T, N, 1)
        features.append(tod_feature)
        feature_names.append('tod')
        logger.info("Added time-of-day feature")
    
    # Add day-of-week feature (0-6)
    if include_dow:
        time_index = ordered_traffic.index
        dow = time_index.dayofweek.values  # Monday=0, Sunday=6
        dow_feature = np.tile(dow[:, np.newaxis, np.newaxis], (1, n_sensors, 1))  # (T, N, 1)
        features.append(dow_feature)
        feature_names.append('dow')
        logger.info("Added day-of-week feature")
    
    # Concatenate all features: (T, N, C)
    data = np.concatenate(features, axis=-1).astype(np.float32)
    
    logger.info(f"Final dataset shape: {data.shape}")
    logger.info(f"Features: {feature_names}")
    logger.info(f"Data range - min: {np.nanmin(data[:, :, 0]):.2f}, max: {np.nanmax(data[:, :, 0]):.2f}")
    
    # Save to npz file
    output_path: Path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(output_path, data=data)
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Saved shape: {data.shape} = (timesteps, nodes, features)")
    
    return data


def verify_dataset_compatibility(
    dataset_path: str,
    adj_mx: AdjacencyMatrix,
) -> bool:
    """
    Verify that dataset.npz is compatible with AdjacencyMatrix.
    
    Args:
        dataset_path: Path to dataset.npz file
        adj_mx: AdjacencyMatrix object to check compatibility with
        
    Returns:
        True if compatible, False otherwise
    """
    data = np.load(dataset_path)['data']
    n_sensors_in_data = data.shape[1]
    n_sensors_in_adj = len(adj_mx.sensor_ids)
    
    if n_sensors_in_data != n_sensors_in_adj:
        logger.error(
            f"Incompatible sensor counts: dataset has {n_sensors_in_data}, "
            f"adjacency matrix has {n_sensors_in_adj}"
        )
        return False
    
    logger.info(f"Dataset and AdjacencyMatrix are compatible ({n_sensors_in_data} sensors)")
    return True
