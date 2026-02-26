import argparse
import logging
from pathlib import Path
from typing import List, Tuple

from .components.adj_mx import AdjacencyMatrix
from .components.metr_imc.interpolation import Interpolator
from .components.metr_imc.interpolation.bgcp import BGCPInterpolator
from .components.metr_imc.interpolation.brits import BRITSInterpolator
from .components.metr_imc.interpolation.knn import SpatialKNNInterpolator
from .components.metr_imc.interpolation.mice import SpatialMICEInterpolator
from .components.metr_imc.interpolation.trmf import TRMFInterpolator
from .pipeline import generate_raw_dataset, generate_subset
from .utils import PathConfig

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate METR-IMC raw and interpolated subset datasets."
    )
    parser.add_argument(
        "--api-key",
        help="data.go.kr API key. If omitted, DATA_API_KEY environment variable is used.",
    )
    parser.add_argument(
        "--config-dir",
        default="./configs",
        help="Directory containing config.yaml and config_*.yaml files (default: ./configs).",
    )
    return parser.parse_args()


def _config_path(config_dir: str, filename: str) -> str:
    return str(Path(config_dir) / filename)


def generate_interpolated_subset(
    key: str, interpolator: Interpolator, config_dir: str, raw_path_conf: PathConfig
) -> None:
    subset_path_conf = PathConfig.from_yaml(
        _config_path(config_dir, f"config_{key}.yaml")
    )
    interpolation_processors: List[Interpolator] = [
        interpolator,
    ]

    generate_subset(
        subset_path_conf=subset_path_conf,
        raw_path_conf=raw_path_conf,
        cluster_count=1,
        missing_rate_threshold=0.9,
        interpolation_processors=interpolation_processors,
    )


def main() -> None:
    args = parse_args()
    raw_path_conf = PathConfig.from_yaml(_config_path(args.config_dir, "config.yaml"))
    raw_path_conf.create_directories()
    base_subset_path_conf = PathConfig.from_yaml(
        _config_path(args.config_dir, "config_base.yaml")
    )

    # Generate Raw and Base Datasets
    generate_raw_dataset(raw_path_conf, api_key=args.api_key)
    generate_subset(
        subset_path_conf=base_subset_path_conf,
        raw_path_conf=raw_path_conf,
        cluster_count=1,
        missing_rate_threshold=0.9,
    )

    # Generate Data Interpolation Subset
    base_adj_mx = AdjacencyMatrix.import_from_pickle(base_subset_path_conf.adj_mx_path)

    interpolation_processors: List[Tuple[str, Interpolator]] = [
        ("mice", SpatialMICEInterpolator(base_adj_mx)),
        ("knn", SpatialKNNInterpolator(base_adj_mx)),
        ("bgcp", BGCPInterpolator()),
        ("trmf", TRMFInterpolator()),
        ("brits", BRITSInterpolator()),
    ]
    for key, interpolator in interpolation_processors:
        logger.info(f'Generating interpolated subset with "{key}" interpolator.')
        generate_interpolated_subset(key, interpolator, args.config_dir, raw_path_conf)


if __name__ == "__main__":
    main()
