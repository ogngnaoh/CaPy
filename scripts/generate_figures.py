"""CaPy figure generation script.

Produces publication-quality figures from evaluation results.
Full implementation in Session 13.

Usage:
    python scripts/generate_figures.py --results-dir results/
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for figure generation.

    Returns:
        Parsed arguments with results directory path.
    """
    parser = argparse.ArgumentParser(
        description="Generate publication figures from CaPy evaluation results."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing evaluation result JSON/CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for figure generation."""
    args = parse_args()
    print(f"Results dir: {args.results_dir}")
    raise NotImplementedError(
        "Figure generation not yet implemented. See Session 13 in capy-guide.md."
    )


if __name__ == "__main__":
    main()
