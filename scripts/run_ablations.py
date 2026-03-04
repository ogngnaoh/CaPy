"""CaPy ablation study runner.

Runs bi-modal ablation experiments (mol↔morph, mol↔expr, morph↔expr)
and embedding dimension sweeps. Full implementation in Session 10.

Usage:
    python scripts/run_ablations.py --config configs/default.yaml
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ablation studies.

    Returns:
        Parsed arguments with config path.
    """
    parser = argparse.ArgumentParser(
        description="Run CaPy ablation experiments (bi-modal + embedding dim sweep)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for ablation studies."""
    args = parse_args()
    print(f"Config: {args.config}")
    raise NotImplementedError(
        "Ablation runner not yet implemented. See Session 10 in capy-guide.md."
    )


if __name__ == "__main__":
    main()
