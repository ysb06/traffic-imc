import argparse
from typing import Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Traffic-IMC baseline training",
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
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Run only the test loop using trainer.resume_ckpt_path from the config",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from .training.runner import run_only_test, run_training

    if args.test_only:
        run_only_test(config_path=args.config, seed=args.seed)
        return

    run_training(config_path=args.config, seed=args.seed)


if __name__ == "__main__":
    main()
