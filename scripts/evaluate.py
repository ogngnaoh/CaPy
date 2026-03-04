"""CaPy evaluation script.

Loads a trained checkpoint and runs retrieval + clustering metrics.
Full implementation in Session 12.

Usage:
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint checkpoints/best.pt
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation.

    Returns:
        Parsed arguments with config path and checkpoint path.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CaPy model on retrieval and clustering."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for evaluation."""
    args = parse_args()
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    raise NotImplementedError(
        "Evaluation not yet implemented. See Session 12 in capy-guide.md."
    )


if __name__ == "__main__":
    main()
