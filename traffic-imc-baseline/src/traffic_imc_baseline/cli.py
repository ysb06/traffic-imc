import argparse
from typing import Optional, Sequence

from .training.registry import get_adapter, list_models


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Traffic-IMC baseline training",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list_models(),
        help="Model name to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config yaml file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from .training.runner import run_training

    adapter = get_adapter(args.model)
    run_training(adapter=adapter, config_path=args.config, seed=args.seed)
