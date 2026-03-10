"""Inspect raw Rosetta CDRP-bio profiles to validate column names and shapes.

Run after download to confirm data format before preprocessing:
    python scripts/inspect_data.py --data-dir data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def inspect_morph(data_dir: Path) -> None:
    """Print morphology CSV column info."""
    morph_path = data_dir / "morphology" / "replicate_level_cp_normalized.csv.gz"
    if not morph_path.exists():
        print(f"[SKIP] Morphology file not found: {morph_path}")
        return

    print(f"\n{'='*60}")
    print(f"MORPHOLOGY: {morph_path}")
    print(f"{'='*60}")
    df = pd.read_csv(morph_path, nrows=5)
    full_shape = pd.read_csv(morph_path, usecols=[0]).shape[0]
    print(f"Shape: ({full_shape}, {len(df.columns)})")

    # Key metadata columns
    key_cols = [
        "Metadata_Sample_Dose",
        "Metadata_broad_sample",
        "Metadata_Plate",
        "Metadata_pert_type",
    ]
    for col in key_cols:
        present = col in df.columns
        sample = df[col].iloc[0] if present else "N/A"
        print(f"  {col}: {'FOUND' if present else 'MISSING'} (sample: {sample})")

    # Feature prefix counts
    prefixes = ("Cells_", "Nuclei_", "Cytoplasm_")
    for p in prefixes:
        count = sum(1 for c in df.columns if c.startswith(p))
        print(f"  {p}* columns: {count}")

    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    print(f"  Metadata_* columns: {len(meta_cols)}")

    # Show first few non-metadata columns
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    print(f"  Non-metadata columns: {len(feature_cols)}")
    print(f"  First 5 features: {feature_cols[:5]}")


def inspect_expr(data_dir: Path) -> None:
    """Print expression CSV column info."""
    expr_path = data_dir / "expression" / "replicate_level_l1k.csv.gz"
    if not expr_path.exists():
        print(f"[SKIP] Expression file not found: {expr_path}")
        return

    print(f"\n{'='*60}")
    print(f"EXPRESSION: {expr_path}")
    print(f"{'='*60}")
    df = pd.read_csv(expr_path, nrows=5, low_memory=False)
    full_shape = pd.read_csv(expr_path, usecols=[0]).shape[0]
    print(f"Shape: ({full_shape}, {len(df.columns)})")

    # Key metadata columns
    key_cols = [
        "pert_sample_dose",
        "pert_id",
        "det_plate",
        "pert_type",
    ]
    for col in key_cols:
        present = col in df.columns
        sample = df[col].iloc[0] if present else "N/A"
        print(f"  {col}: {'FOUND' if present else 'MISSING'} (sample: {sample})")

    # L1000 probe columns (end with _at)
    at_cols = [c for c in df.columns if c.endswith("_at")]
    print(f"  *_at columns (L1000 probes): {len(at_cols)}")
    if at_cols:
        print(f"  First 5 probes: {at_cols[:5]}")

    # Metadata vs feature breakdown
    meta_prefixes = ("pert_", "det_", "distil_", "cell_", "rna_")
    meta_cols = [c for c in df.columns if any(c.startswith(p) for p in meta_prefixes)]
    print(f"  Metadata-prefix columns: {len(meta_cols)}")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Inspect raw Rosetta profiles.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to raw data directory (default: data/raw)",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    inspect_morph(data_dir)
    inspect_expr(data_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
