"""QC, normalization, and scaffold-based train/val/test splitting.

WHY THIS WORKS
--------------
Scaffold splitting groups molecules by their Bemis-Murcko core scaffold
(the ring/linker skeleton without side chains).  This is *critical* in
drug-discovery ML: a random split lets near-identical analogues leak
between train and test, inflating performance on seen chemistry rather
than measuring true generalization.

We assign whole scaffold groups to one split — greedy largest-first,
which tends to give the best balance while guaranteeing zero scaffold
overlap.  This mirrors the methodology used in MoleculeNet, OGB, and
most serious cheminformatics benchmarks.

Normalization uses ``RobustScaler`` (median / IQR) for morphology
features because CellProfiler features have heavy-tailed distributions
with occasional outliers from segmentation artefacts.  Expression data
from L1000 is already z-scored, so we only verify and optionally
re-normalize.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — pandas / numpy / sklearn / rdkit
# ---------------------------------------------------------------------------

_pd = None
_np = None


def _ensure_imports() -> None:
    global _pd, _np  # noqa: PLW0603
    if _pd is None:
        import pandas as pd

        _pd = pd
    if _np is None:
        import numpy as np

        _np = np


# ---------------------------------------------------------------------------
# Scaffold utility (self-contained — no torch dependency)
# ---------------------------------------------------------------------------


def _get_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko scaffold SMILES for a molecule.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES of the generic scaffold, or the original SMILES
        if scaffold extraction fails.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import (
        GetScaffoldForMol,
        MakeScaffoldGeneric,
    )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaffold = GetScaffoldForMol(mol)
        scaffold = MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(scaffold)
    except Exception:  # noqa: BLE001
        return smiles


# ---------------------------------------------------------------------------
# QC helpers
# ---------------------------------------------------------------------------


def remove_controls(
    df: pd.DataFrame,  # noqa: F821
    smiles_col: str = "smiles",
    compound_col: str = "compound_id",
) -> pd.DataFrame:  # noqa: F821
    """Remove DMSO and other negative controls.

    Args:
        df: Compound-level data frame.
        smiles_col: Column containing SMILES strings.
        compound_col: Column containing compound identifiers.

    Returns:
        Data frame with control rows removed.
    """
    _ensure_imports()
    n_before = len(df)

    # Remove rows with missing SMILES or known control identifiers
    mask = df[smiles_col].notna() & (df[smiles_col] != "")
    if compound_col in df.columns:
        control_patterns = ["DMSO", "dmso", "EMPTY", "UNTREATED"]
        for pat in control_patterns:
            mask = mask & ~df[compound_col].str.contains(pat, case=False, na=False)

    df = df.loc[mask].copy()
    logger.info(
        "remove_controls: %d -> %d rows (%d removed).",
        n_before,
        len(df),
        n_before - len(df),
    )
    return df


def feature_qc(
    df: pd.DataFrame,  # noqa: F821
    morph_cols: list[str],
    expr_cols: list[str],
    nan_threshold: float = 0.5,
    train_mask: pd.Series | None = None,  # noqa: F821
) -> tuple[pd.DataFrame, list[str], list[str]]:  # noqa: F821
    """Remove features with too many NaNs or zero variance.

    Statistics (NaN fraction, variance) are computed on the training
    set only when ``train_mask`` is provided, preventing data leakage
    from val/test into feature selection.

    Args:
        df: Compound-level data frame.
        morph_cols: Morphology feature column names.
        expr_cols: Expression feature column names.
        nan_threshold: Drop features with NaN fraction above this value.
        train_mask: Boolean Series selecting training rows.  If ``None``,
            statistics are computed on the full dataframe (legacy behavior).

    Returns:
        Tuple of (cleaned df, surviving morph_cols, surviving expr_cols).
    """
    _ensure_imports()
    stats_df = df.loc[train_mask] if train_mask is not None else df

    def _filter_cols(cols: list[str]) -> list[str]:
        kept = []
        for c in cols:
            if c not in df.columns:
                continue
            if not _pd.api.types.is_numeric_dtype(df[c]):
                continue
            nan_frac = stats_df[c].isna().mean()
            if nan_frac > nan_threshold:
                continue
            if stats_df[c].std() == 0:
                continue
            kept.append(c)
        return kept

    morph_kept = _filter_cols(morph_cols)
    expr_kept = _filter_cols(expr_cols)
    logger.info(
        "feature_qc: morph %d -> %d, expr %d -> %d.",
        len(morph_cols),
        len(morph_kept),
        len(expr_cols),
        len(expr_kept),
    )
    # Drop rows with remaining NaNs in kept features
    all_kept = morph_kept + expr_kept
    df = df.dropna(subset=all_kept).copy()
    return df, morph_kept, expr_kept


def normalize_features(
    df: pd.DataFrame,  # noqa: F821
    morph_cols: list[str],
    expr_cols: list[str],
    clip_range: float = 5.0,
    train_mask: pd.Series | None = None,  # noqa: F821
) -> tuple[pd.DataFrame, object]:  # noqa: F821
    """Apply RobustScaler to morphology and verify expression z-scores.

    When ``train_mask`` is provided, the scaler is fitted on training
    rows only and applied to all rows — preventing data leakage from
    val/test into normalization statistics.

    Args:
        df: Data frame with QC-passed features.
        morph_cols: Morphology feature column names.
        expr_cols: Expression feature column names.
        clip_range: Clip scaled morphology values to ``[-clip_range, clip_range]``.
        train_mask: Boolean Series selecting training rows.  If ``None``,
            the scaler is fitted on the full dataframe (legacy behavior).

    Returns:
        Tuple of (normalized df, fitted RobustScaler for morphology).
    """
    _ensure_imports()
    from sklearn.preprocessing import RobustScaler

    df = df.copy()

    # Morphology: RobustScaler fitted on train only, applied to all
    scaler = RobustScaler()
    if train_mask is not None:
        if not train_mask.index.equals(df.index):
            raise ValueError(
                "train_mask index does not align with df index. "
                "Recompute train_mask after any row-dropping operations."
            )
        scaler.fit(df.loc[train_mask, morph_cols])
    else:
        scaler.fit(df[morph_cols])
    df[morph_cols] = scaler.transform(df[morph_cols])
    df[morph_cols] = df[morph_cols].clip(-clip_range, clip_range)

    # Expression: verify z-scores using train stats; re-normalize if drift is large
    if train_mask is not None:
        stats_df = df.loc[train_mask, expr_cols]
    else:
        stats_df = df[expr_cols]
    expr_mean = stats_df.mean().mean()
    expr_std = stats_df.std().mean()
    logger.info(
        "Expression stats (train): mean=%.4f, std=%.4f (expected ~0, ~1).",
        expr_mean,
        expr_std,
    )
    if abs(expr_mean) > 0.5 or abs(expr_std - 1.0) > 0.5:
        logger.warning("Expression data deviates from z-score; re-normalizing.")
        train_mean = stats_df.mean()
        train_std = stats_df.std()
        train_std = train_std.replace(0, 1.0)  # guard against division by zero
        df[expr_cols] = (df[expr_cols] - train_mean) / train_std

    return df, scaler


# ---------------------------------------------------------------------------
# Scaffold split
# ---------------------------------------------------------------------------


def scaffold_split(
    df: pd.DataFrame,  # noqa: F821
    smiles_col: str = "smiles",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:  # noqa: F821
    """Assign each row to train/val/test by Bemis-Murcko scaffold.

    Greedy largest-first assignment: scaffold groups are sorted by size
    (descending) and assigned to the split furthest below its target.

    Args:
        df: Data frame with a SMILES column.
        smiles_col: Column containing SMILES strings.
        train_ratio: Target fraction for training set.
        val_ratio: Target fraction for validation set.
        test_ratio: Target fraction for test set.
        seed: Random seed for deterministic tie-breaking.

    Returns:
        Data frame with an added ``"split"`` column (``"train"``/``"val"``/``"test"``).
    """
    _ensure_imports()
    import random

    df = df.copy()
    n = len(df)

    # Group indices by scaffold
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, smi in zip(df.index, df[smiles_col]):
        scaffold = _get_scaffold(smi)
        scaffold_to_indices[scaffold].append(idx)

    # Sort scaffold groups largest-first; break ties deterministically
    rng = random.Random(seed)
    groups = list(scaffold_to_indices.values())
    rng.shuffle(groups)  # randomize within same-size groups
    groups.sort(key=len, reverse=True)

    # Greedy assignment
    train_idx, val_idx, test_idx = [], [], []
    targets = {
        "train": (train_idx, train_ratio * n),
        "val": (val_idx, val_ratio * n),
        "test": (test_idx, test_ratio * n),
    }

    for group in groups:
        # Pick the split furthest below its target
        deficits = {
            name: target - len(bucket) for name, (bucket, target) in targets.items()
        }
        best = max(deficits, key=deficits.get)
        targets[best][0].extend(group)

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    for split_name in ["train", "val", "test"]:
        cnt = (df["split"] == split_name).sum()
        logger.info("scaffold_split: %s = %d (%.1f%%)", split_name, cnt, 100 * cnt / n)

    return df


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_raw_profiles(
    raw_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:  # noqa: F821
    """Load raw morphology and expression profile CSVs.

    Args:
        raw_dir: Directory containing ``morphology/`` and ``expression/``
            subdirectories.

    Returns:
        Tuple of (morphology DataFrame, expression DataFrame).
    """
    _ensure_imports()
    raw_dir = Path(raw_dir)
    morph_path = (
        raw_dir
        / "morphology"
        / "replicate_level_cp_normalized_variable_selected.csv.gz"
    )
    expr_path = raw_dir / "expression" / "replicate_level_l1k.csv.gz"

    logger.info("Loading morphology profiles from %s", morph_path)
    morph_df = _pd.read_csv(morph_path)

    logger.info("Loading expression profiles from %s", expr_path)
    expr_df = _pd.read_csv(expr_path, low_memory=False)

    return morph_df, expr_df


def aggregate_to_compound_level(
    df: pd.DataFrame,  # noqa: F821
    compound_col: str,
) -> pd.DataFrame:  # noqa: F821
    """Aggregate replicate-level profiles to compound-level medians.

    The Rosetta dataset provides replicate-level measurements.  We take
    the median across replicates for each compound to get a single
    robust profile per compound, consistent with standard practice in
    Cell Painting and L1000 analyses.

    Args:
        df: Replicate-level DataFrame with a compound identifier column.
        compound_col: Column name containing compound identifiers to
            group by.

    Returns:
        Compound-level DataFrame with median feature values and first
        occurrence of metadata columns.
    """
    _ensure_imports()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [
        c for c in df.columns if c not in numeric_cols and c != compound_col
    ]

    n_before = len(df)
    n_compounds = df[compound_col].nunique()

    # Median for numeric features, first value for metadata
    agg_dict: dict[str, str] = {}
    for c in numeric_cols:
        agg_dict[c] = "median"
    for c in non_numeric_cols:
        agg_dict[c] = "first"

    result = df.groupby(compound_col, as_index=False).agg(agg_dict)

    logger.info(
        "aggregate_to_compound_level: %d replicates -> %d compounds (groupby %s).",
        n_before,
        n_compounds,
        compound_col,
    )
    return result


def match_compounds(
    morph_df: pd.DataFrame,  # noqa: F821
    expr_df: pd.DataFrame,  # noqa: F821
    metadata_df: pd.DataFrame | None = None,  # noqa: F821
) -> pd.DataFrame:  # noqa: F821
    """Merge morphology and expression data by compound identifier.

    Cell Painting uses ``Metadata_broad_sample`` and L1000 uses
    ``pert_id`` — both are Broad compound IDs of the form ``BRD-…``.

    Args:
        morph_df: Morphology profiles (CellProfiler features).
        expr_df: Expression profiles (L1000 genes).
        metadata_df: Optional compound metadata with SMILES and MOA.

    Returns:
        Merged data frame with morphology features, expression features,
        and compound metadata.
    """
    _ensure_imports()

    # Identify compound ID columns
    morph_id_col = "Metadata_broad_sample"
    expr_id_col = "pert_id"

    # Standardize to a common column name
    morph_df = morph_df.rename(columns={morph_id_col: "compound_id"})
    expr_df = expr_df.rename(columns={expr_id_col: "compound_id"})

    # Inner join on compound_id
    merged = morph_df.merge(
        expr_df, on="compound_id", how="inner", suffixes=("_morph", "_expr")
    )
    logger.info(
        "match_compounds: morph=%d, expr=%d, matched=%d.",
        len(morph_df),
        len(expr_df),
        len(merged),
    )

    if metadata_df is not None:
        # Find the compound ID column in metadata
        meta_id_col = None
        for candidate in ("broad_id", "compound_id"):
            if candidate in metadata_df.columns:
                meta_id_col = candidate
                break

        if meta_id_col is None:
            logger.warning(
                "Metadata has no 'broad_id' or 'compound_id' column "
                "(columns: %s). Skipping metadata merge.",
                list(metadata_df.columns[:5]),
            )
        else:
            # Use .map() instead of .merge() to avoid pandas merge internals
            # that fail with KeyError on certain DataFrame states (pandas 2.x).
            meta_dedup = metadata_df.drop_duplicates(subset=meta_id_col)
            meta_indexed = meta_dedup.set_index(meta_id_col)
            new_cols = [c for c in meta_indexed.columns if c not in merged.columns]
            for col in new_cols:
                merged[col] = merged["compound_id"].map(meta_indexed[col])
            n_matched = merged[new_cols[0]].notna().sum() if new_cols else 0
            logger.info(
                "Metadata mapped: %d/%d compounds matched via %s.",
                n_matched,
                len(merged),
                meta_id_col,
            )

    return merged


def save_processed_data(
    df: pd.DataFrame,  # noqa: F821
    morph_cols: list[str],
    expr_cols: list[str],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save split data to parquet files and feature column names to JSON.

    Args:
        df: Full data frame with ``"split"`` column.
        morph_cols: Retained morphology feature column names.
        expr_cols: Retained expression feature column names.
        output_dir: Output directory (e.g. ``data/processed``).

    Returns:
        Dict mapping split names and ``"feature_cols"`` to their file paths.
    """
    _ensure_imports()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}
    for split_name in ["train", "val", "test"]:
        split_df = df.loc[df["split"] == split_name]
        path = output_dir / f"{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        result[split_name] = path
        logger.info("Saved %s: %d rows -> %s", split_name, len(split_df), path)

    # Save retained feature column names
    cols_path = output_dir / "feature_columns.json"
    with open(cols_path, "w") as f:
        json.dump({"morph_cols": morph_cols, "expr_cols": expr_cols}, f, indent=2)
    result["feature_cols"] = cols_path

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def preprocess_pipeline(cfg: DictConfig) -> dict[str, Path]:  # noqa: F821
    """Run the full preprocessing pipeline from raw profiles to split parquets.

    Args:
        cfg: OmegaConf config (must have ``data.raw_dir``, ``data.processed_dir``,
            ``data.nan_threshold``, ``data.clip_range``, ``data.train_ratio``,
            ``data.val_ratio``, ``data.test_ratio``, ``seed``).

    Returns:
        Dict mapping split names to file paths.
    """
    _ensure_imports()

    raw_dir = Path(cfg.data.raw_dir)
    processed_dir = Path(cfg.data.processed_dir)

    # 1. Load raw (replicate-level) profiles
    morph_df, expr_df = load_raw_profiles(raw_dir)

    # 1.5 Aggregate replicates to compound-level medians
    morph_df = aggregate_to_compound_level(
        morph_df, compound_col="Metadata_broad_sample"
    )
    expr_df = aggregate_to_compound_level(expr_df, compound_col="pert_id")

    # 2. Load compound metadata (SMILES, MOA) if available
    metadata_df = None
    metadata_path = raw_dir / "metadata" / "repurposing_samples.txt"
    if metadata_path.exists():
        # Skip !-prefixed comment/header lines (the CLUE repurposing file
        # has 9 such lines).  We avoid pandas comment="!" because that
        # parameter also strips text after ! inside data cells, corrupting rows.
        import io

        with open(metadata_path) as fh:
            clean_lines = [line for line in fh if not line.startswith("!")]
        metadata_df = _pd.read_csv(io.StringIO("".join(clean_lines)), sep="\t")
        logger.info(
            "Loaded compound metadata: %d rows from %s", len(metadata_df), metadata_path
        )
        logger.info(
            "Metadata columns: %s (shape=%s)",
            list(metadata_df.columns[:6]),
            metadata_df.shape,
        )
    else:
        logger.warning(
            "Compound metadata not found at %s — SMILES may be missing.", metadata_path
        )

    # 3. Match compounds across modalities
    df = match_compounds(morph_df, expr_df, metadata_df=metadata_df)

    # 4. Identify feature columns (numeric only — exclude string metadata
    # that gets suffixed during the morph/expr merge, e.g. pert_type_morph).
    numeric_cols = set(df.select_dtypes(include="number").columns)
    morph_cols = [
        c
        for c in df.columns
        if not c.startswith("Metadata_")
        and c != "compound_id"
        and c.endswith("_morph")
        and c in numeric_cols
    ]
    expr_cols = [
        c
        for c in df.columns
        if c != "compound_id" and c.endswith("_expr") and c in numeric_cols
    ]
    # Fallback: if suffix-based detection yields nothing, use heuristics
    if not morph_cols:
        morph_cols = [
            c for c in df.columns if c.startswith(("Cells_", "Nuclei_", "Cytoplasm_"))
        ]
    if not expr_cols:
        non_meta = [
            c
            for c in df.columns
            if not c.startswith("Metadata_")
            and c != "compound_id"
            and c not in morph_cols
            and c != "smiles"
            and c != "split"
            and c in numeric_cols
        ]
        expr_cols = non_meta
    logger.info(
        "Feature columns: %d morph, %d expr (from %d total columns).",
        len(morph_cols),
        len(expr_cols),
        len(df.columns),
    )

    # 5. Remove controls
    df = remove_controls(df)

    # 6. Scaffold split FIRST — so QC and normalization use train stats only
    df = scaffold_split(
        df,
        smiles_col="smiles",
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.seed,
    )
    train_mask = df["split"] == "train"

    # 7. Feature QC (stats computed on train only)
    df, morph_cols, expr_cols = feature_qc(
        df,
        morph_cols,
        expr_cols,
        nan_threshold=cfg.data.nan_threshold,
        train_mask=train_mask,
    )
    # Recompute train_mask after QC may have dropped rows
    train_mask = df["split"] == "train"

    # 8. Normalize (scaler fitted on train only, applied to all)
    df, _ = normalize_features(
        df,
        morph_cols,
        expr_cols,
        clip_range=cfg.data.clip_range,
        train_mask=train_mask,
    )

    # 9. Save
    return save_processed_data(df, morph_cols, expr_cols, processed_dir)
