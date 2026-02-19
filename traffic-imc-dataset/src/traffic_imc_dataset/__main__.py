import argparse
import logging
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
    return parser.parse_args()


def generate_interpolated_subset(key: str, interpolator: Interpolator):
    subset_path_conf = PathConfig.from_yaml(f"./configs/config_{key}.yaml")
    interpolation_processors: List[Interpolator] = [
        interpolator,
    ]

    generate_subset(
        subset_path_conf=subset_path_conf,
        cluster_count=1,
        missing_rate_threshold=0.9,
        interpolation_processors=interpolation_processors,
    )


def main() -> None:
    args = parse_args()

    # Generate Raw Datasets
    generate_raw_dataset(api_key=args.api_key)

    # Generate base subset datasets (for testing)
    base_subset_path_conf = PathConfig.from_yaml("./configs/config_base.yaml")
    generate_subset(
        subset_path_conf=base_subset_path_conf,
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
        generate_interpolated_subset(key, interpolator)


if __name__ == "__main__":
    main()
