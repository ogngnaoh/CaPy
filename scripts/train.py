"""CaPy training script.

Loads config, initializes data pipeline, model, and training loop.
Full implementation in Session 7.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml model.embedding_dim=512
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training.

    Returns:
        Parsed arguments with config path and optional overrides.
    """
    parser = argparse.ArgumentParser(
        description="Train the CaPy tri-modal contrastive model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional dotpath config overrides, e.g. model.embedding_dim=512.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    args = parse_args()
    print(f"Config: {args.config}")
    print(f"Overrides: {args.overrides}")
    raise NotImplementedError(
        "Training loop not yet implemented. See Session 7 in capy-guide.md."
    )


if __name__ == "__main__":
    main()
