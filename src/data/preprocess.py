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

The pipeline operates at the *treatment level* (compound + dose), not
compound level.  This follows Haghighi et al. (Nature Methods 2022):
each (compound, dose) pair is a distinct biological perturbation with
its own phenotypic profile.  Normalization uses per-plate z-scoring
against DMSO controls followed by global ``RobustScaler``, and a
replicate-correlation quality filter removes noisy treatments.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from src.utils.logging import get_logger

# Prefixes that indicate metadata columns — must be excluded from feature lists.
_METADATA_PREFIXES = ("pert_", "det_", "distil_", "cell_", "Metadata_", "rna_")

# CellProfiler feature prefixes (morphology).
_CELLPROFILER_PREFIXES = ("Cells_", "Nuclei_", "Cytoplasm_")

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
# Per-plate normalization (Step 2)
# ---------------------------------------------------------------------------


def normalize_per_plate(
    df: pd.DataFrame,  # noqa: F821
    feature_cols: list[str],
    plate_col: str = "Metadata_Plate",
    control_col: str = "Metadata_pert_type",
    control_value: str = "control",
) -> pd.DataFrame:  # noqa: F821
    """Z-score features per plate using DMSO/control wells as reference.

    For each plate, compute the mean and std of control wells and
    z-score all wells on that plate.  This removes plate-level batch
    effects, following Haghighi et al. (Nature Methods 2022).

    Args:
        df: Replicate-level DataFrame with plate and control columns.
        feature_cols: Numeric feature columns to normalize.
        plate_col: Column identifying the plate.
        control_col: Column identifying control vs. treatment wells.
        control_value: Value in ``control_col`` that marks controls.

    Returns:
        DataFrame with per-plate z-scored features.
    """
    _ensure_imports()
    df = df.copy()
    plates = df[plate_col].unique()
    n_plates = len(plates)
    n_no_controls = 0

    for plate in plates:
        plate_mask = df[plate_col] == plate
        ctrl_mask = plate_mask & (df[control_col] == control_value)

        if ctrl_mask.sum() == 0:
            n_no_controls += 1
            # Fall back to plate-wide stats if no controls on this plate
            mu = df.loc[plate_mask, feature_cols].mean()
            sigma = df.loc[plate_mask, feature_cols].std()
        else:
            mu = df.loc[ctrl_mask, feature_cols].mean()
            sigma = df.loc[ctrl_mask, feature_cols].std()

        # Guard: avoid division by zero
        sigma = sigma.replace(0, 1.0)
        df.loc[plate_mask, feature_cols] = (
            df.loc[plate_mask, feature_cols] - mu
        ) / sigma

    if n_no_controls > 0:
        logger.warning(
            "normalize_per_plate: %d/%d plates had no '%s' controls "
            "(used plate-wide stats as fallback).",
            n_no_controls,
            n_plates,
            control_value,
        )
    logger.info(
        "normalize_per_plate: z-scored %d features across %d plates.",
        len(feature_cols),
        n_plates,
    )
    return df


# ---------------------------------------------------------------------------
# Replicate correlation filter (Step 3)
# ---------------------------------------------------------------------------


def filter_by_replicate_correlation(
    df: pd.DataFrame,  # noqa: F821
    feature_cols: list[str],
    treatment_col: str,
    control_col: str = "Metadata_pert_type",
    control_value: str = "control",
    percentile: int = 90,
) -> pd.DataFrame:  # noqa: F821
    """Keep only treatments with replicate correlation above a null threshold.

    For each treatment, compute mean pairwise Pearson correlation across
    replicates.  Build a null distribution from control wells.  Keep
    treatments whose replicate correlation exceeds the given percentile
    of the null distribution.

    Args:
        df: Replicate-level DataFrame.
        feature_cols: Numeric feature columns.
        treatment_col: Column identifying treatments.
        control_col: Column identifying control wells.
        control_value: Value in ``control_col`` marking controls.
        percentile: Percentile of null distribution to use as threshold.

    Returns:
        DataFrame with low-correlation treatments removed.
    """
    _ensure_imports()

    def _mean_pairwise_corr(group_df: pd.DataFrame) -> float:  # noqa: F821
        """Compute mean pairwise Pearson correlation for a group."""
        vals = group_df[feature_cols].values
        if len(vals) < 2:
            return _np.nan
        corr_matrix = _np.corrcoef(vals)
        # Extract upper triangle (excluding diagonal)
        n = corr_matrix.shape[0]
        upper_tri = corr_matrix[_np.triu_indices(n, k=1)]
        if len(upper_tri) == 0:
            return _np.nan
        return float(_np.nanmean(upper_tri))

    n_before = df[treatment_col].nunique()

    # Build null distribution from control replicates
    ctrl_df = df[df[control_col] == control_value]
    null_corrs = []
    if len(ctrl_df) > 0:
        # Group controls by plate (if available) to get per-plate null
        plate_col = "Metadata_Plate" if "Metadata_Plate" in ctrl_df.columns else None
        if plate_col:
            for _, group in ctrl_df.groupby(plate_col):
                if len(group) >= 2:
                    c = _mean_pairwise_corr(group)
                    if not _np.isnan(c):
                        null_corrs.append(c)
        else:
            c = _mean_pairwise_corr(ctrl_df)
            if not _np.isnan(c):
                null_corrs.append(c)

    if len(null_corrs) == 0:
        logger.warning(
            "filter_by_replicate_correlation: no control replicates found, "
            "skipping filter."
        )
        return df

    threshold = float(_np.percentile(null_corrs, percentile))

    # Compute replicate correlation for each treatment
    treatment_groups = df[df[control_col] != control_value].groupby(treatment_col)
    keep_treatments = set()
    for trt, group in treatment_groups:
        corr = _mean_pairwise_corr(group)
        if _np.isnan(corr) or corr > threshold:
            # Keep treatments with good correlation or singletons (NaN)
            keep_treatments.add(trt)

    # Keep controls + passing treatments
    keep_mask = (df[control_col] == control_value) | (
        df[treatment_col].isin(keep_treatments)
    )
    result = df[keep_mask].copy()

    n_after = result[treatment_col].nunique()
    logger.info(
        "filter_by_replicate_correlation: kept %d/%d treatments (threshold=%.4f).",
        n_after,
        n_before,
        threshold,
    )
    return result


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
        df: Treatment-level data frame.
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
        df: Treatment-level data frame.
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
    n_before = len(df)
    df = df.dropna(subset=all_kept).copy()
    if len(df) < n_before:
        logger.info(
            "feature_qc: dropped %d rows with NaN in kept features.",
            n_before - len(df),
        )
    if len(df) == 0:
        logger.warning("feature_qc: ALL rows dropped — 0 treatments remain.")
    return df, morph_kept, expr_kept


def normalize_features(
    df: pd.DataFrame,  # noqa: F821
    morph_cols: list[str],
    expr_cols: list[str],
    clip_range: float = 5.0,
    train_mask: pd.Series | None = None,  # noqa: F821
) -> tuple[pd.DataFrame, object]:  # noqa: F821
    """Apply RobustScaler to morphology; verify and clip expression.

    Morphology features (post-per-plate-z-score) need global RobustScaler
    for cross-plate standardization.  Expression features (already z-scored
    by the L1000 pipeline) only need clipping — applying RobustScaler on
    pre-normalized data would compress variance and contribute to collapse.

    Args:
        df: Data frame with QC-passed features.
        morph_cols: Morphology feature column names.
        expr_cols: Expression feature column names.
        clip_range: Clip scaled values to ``[-clip_range, clip_range]``.
        train_mask: Boolean Series selecting training rows.  If ``None``,
            the scaler is fitted on the full dataframe (legacy behavior).

    Returns:
        Tuple of (normalized df, fitted RobustScaler for morphology).
    """
    _ensure_imports()
    from sklearn.preprocessing import RobustScaler

    df = df.copy()

    if train_mask is not None:
        if not train_mask.index.equals(df.index):
            raise ValueError(
                "train_mask index does not align with df index. "
                "Recompute train_mask after any row-dropping operations."
            )

    # Morphology: RobustScaler fitted on train only, applied to all
    morph_scaler = RobustScaler()
    if train_mask is not None:
        morph_scaler.fit(df.loc[train_mask, morph_cols])
    else:
        morph_scaler.fit(df[morph_cols])
    df[morph_cols] = morph_scaler.transform(df[morph_cols])
    df[morph_cols] = df[morph_cols].clip(-clip_range, clip_range)

    # Expression: already z-scored by L1000 pipeline.  Verify stats and
    # only re-normalize if substantially drifted.  Always clip outliers.
    if train_mask is not None:
        expr_stats = df.loc[train_mask, expr_cols]
    else:
        expr_stats = df[expr_cols]
    expr_mean = expr_stats.mean().mean()
    expr_std = expr_stats.std().mean()
    logger.info(
        "Expression pre-norm stats (train): mean=%.4f, std=%.4f.",
        expr_mean,
        expr_std,
    )

    if abs(expr_mean) > 0.5 or abs(expr_std - 1.0) > 0.5:
        logger.warning(
            "Expression deviates from z-score (mean=%.2f, std=%.2f); "
            "applying train-fitted z-normalization.",
            expr_mean,
            expr_std,
        )
        train_mean = expr_stats.mean()
        train_std = expr_stats.std()
        train_std = train_std.replace(0, 1.0)
        df[expr_cols] = (df[expr_cols] - train_mean) / train_std

    df[expr_cols] = df[expr_cols].clip(-clip_range, clip_range)

    # Log diagnostics for both modalities
    if train_mask is not None:
        stats_df = df.loc[train_mask]
    else:
        stats_df = df
    for name, cols in [("morph", morph_cols), ("expr", expr_cols)]:
        feat_stds = stats_df[cols].std()
        n_near_zero = int((feat_stds < 0.01).sum())
        logger.info(
            "%s feature stats: mean_of_stds=%.4f, min_std=%.4f, "
            "max_std=%.4f, near_zero_var=%d/%d.",
            name,
            feat_stds.mean(),
            feat_stds.min(),
            feat_stds.max(),
            n_near_zero,
            len(cols),
        )
        if n_near_zero > 0:
            logger.warning(
                "%s feature QC: %d/%d features have near-zero variance "
                "(std < 0.01) — these contribute noise, not signal.",
                name,
                n_near_zero,
                len(cols),
            )

    return df, morph_scaler


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
    Multiple treatments (different doses) of the same compound share
    the same scaffold and therefore land in the same split.

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
    morph_path = raw_dir / "morphology" / "replicate_level_cp_normalized.csv.gz"
    expr_path = raw_dir / "expression" / "replicate_level_l1k.csv.gz"

    logger.info("Loading morphology profiles from %s", morph_path)
    morph_df = _pd.read_csv(morph_path)

    logger.info("Loading expression profiles from %s", expr_path)
    expr_df = _pd.read_csv(expr_path, low_memory=False)

    return morph_df, expr_df


def aggregate_to_treatment_level(
    df: pd.DataFrame,  # noqa: F821
    treatment_col: str,
) -> pd.DataFrame:  # noqa: F821
    """Aggregate replicate-level profiles to treatment-level means.

    Following Haghighi et al. (Nature Methods 2022), we aggregate by
    treatment (compound + dose) using mean, not median.  Each treatment
    is a distinct biological perturbation.

    Args:
        df: Replicate-level DataFrame with a treatment identifier column.
        treatment_col: Column name containing treatment identifiers to
            group by (e.g. ``Metadata_Sample_Dose`` or ``pert_sample_dose``).

    Returns:
        Treatment-level DataFrame with mean feature values and first
        occurrence of metadata columns.
    """
    _ensure_imports()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [
        c for c in df.columns if c not in numeric_cols and c != treatment_col
    ]

    n_before = len(df)
    n_treatments = df[treatment_col].nunique()

    # Mean for numeric features, first value for metadata
    agg_dict: dict[str, str] = {}
    for c in numeric_cols:
        agg_dict[c] = "mean"
    for c in non_numeric_cols:
        agg_dict[c] = "first"

    result = df.groupby(treatment_col, as_index=False).agg(agg_dict)

    logger.info(
        "aggregate_to_treatment_level: %d replicates -> %d treatments (groupby %s).",
        n_before,
        n_treatments,
        treatment_col,
    )
    return result


# Keep backward-compatible alias
def aggregate_to_compound_level(
    df: pd.DataFrame,  # noqa: F821
    compound_col: str,
) -> pd.DataFrame:  # noqa: F821
    """Aggregate replicate-level profiles to compound-level medians.

    .. deprecated::
        Use :func:`aggregate_to_treatment_level` instead.

    Args:
        df: Replicate-level DataFrame with a compound identifier column.
        compound_col: Column name containing compound identifiers.

    Returns:
        Compound-level DataFrame with median feature values.
    """
    _ensure_imports()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [
        c for c in df.columns if c not in numeric_cols and c != compound_col
    ]

    n_before = len(df)
    n_compounds = df[compound_col].nunique()

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


def match_treatments(
    morph_df: pd.DataFrame,  # noqa: F821
    expr_df: pd.DataFrame,  # noqa: F821
    metadata_df: pd.DataFrame | None = None,  # noqa: F821
    morph_id_col: str = "Metadata_Sample_Dose",
    expr_id_col: str = "pert_sample_dose",
) -> pd.DataFrame:  # noqa: F821
    """Merge morphology and expression data by treatment identifier.

    Cell Painting uses ``Metadata_Sample_Dose`` and L1000 uses
    ``pert_sample_dose`` — both encode compound+dose as the treatment key.
    After merging, a ``compound_id`` column is extracted for SMILES lookup
    and scaffold splitting.

    Args:
        morph_df: Morphology profiles (CellProfiler features).
        expr_df: Expression profiles (L1000 genes).
        metadata_df: Optional compound metadata with SMILES and MOA.
        morph_id_col: Treatment ID column in morphology data.
        expr_id_col: Treatment ID column in expression data.

    Returns:
        Merged data frame with treatment_id, compound_id, and all features.
    """
    _ensure_imports()

    # Standardize to a common column name
    morph_df = morph_df.rename(columns={morph_id_col: "treatment_id"})
    expr_df = expr_df.rename(columns={expr_id_col: "treatment_id"})

    # Inner join on treatment_id
    merged = morph_df.merge(
        expr_df, on="treatment_id", how="inner", suffixes=("_morph", "_expr")
    )
    logger.info(
        "match_treatments: morph=%d, expr=%d, matched=%d.",
        len(morph_df),
        len(expr_df),
        len(merged),
    )

    # Extract compound_id from treatment_id by stripping dose portion
    # Expected format: "BRD-K12345678_10.0" or "BRD-K12345678-001-01-1_10.0"
    # We extract the BRD prefix (13-char short form).
    _brd_pattern = re.compile(r"^(BRD-[A-Za-z]\d{8})")

    def _extract_compound_id(treatment_id: str) -> str | None:
        m = _brd_pattern.search(str(treatment_id))
        return m.group(1) if m else None

    merged["compound_id"] = merged["treatment_id"].apply(_extract_compound_id)
    n_no_brd = merged["compound_id"].isna().sum()
    if n_no_brd > 0:
        logger.warning(
            "match_treatments: %d/%d treatments have no BRD compound ID "
            "(will be dropped during control removal).",
            n_no_brd,
            len(merged),
        )

    # Merge metadata (SMILES, MOA) using compound_id
    if metadata_df is not None:
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

            def _normalize_brd(s: _pd.Series) -> _pd.Series:
                return s.astype(str).str.extract(r"^(BRD-[A-Za-z]\d{8})", expand=False)

            meta_dedup = metadata_df.drop_duplicates(subset=meta_id_col).copy()
            meta_dedup["_norm_id"] = _normalize_brd(meta_dedup[meta_id_col])
            meta_dedup = meta_dedup.dropna(subset=["_norm_id"]).drop_duplicates(
                subset="_norm_id"
            )
            meta_indexed = meta_dedup.set_index("_norm_id")

            new_cols = [
                c
                for c in meta_indexed.columns
                if c not in merged.columns and c != meta_id_col
            ]
            for col in new_cols:
                merged[col] = merged["compound_id"].map(meta_indexed[col])

            n_matched = merged[new_cols[0]].notna().sum() if new_cols else 0
            logger.info(
                "Metadata mapped: %d/%d treatments matched via %s.",
                n_matched,
                len(merged),
                meta_id_col,
            )

    return merged


# Keep backward-compatible alias
def match_compounds(
    morph_df: pd.DataFrame,  # noqa: F821
    expr_df: pd.DataFrame,  # noqa: F821
    metadata_df: pd.DataFrame | None = None,  # noqa: F821
) -> pd.DataFrame:  # noqa: F821
    """Merge morphology and expression data by compound identifier.

    .. deprecated::
        Use :func:`match_treatments` instead.

    Args:
        morph_df: Morphology profiles (CellProfiler features).
        expr_df: Expression profiles (L1000 genes).
        metadata_df: Optional compound metadata with SMILES and MOA.

    Returns:
        Merged data frame with compound_id and all features.
    """
    _ensure_imports()

    morph_id_col = "Metadata_broad_sample"
    expr_id_col = "pert_id"

    morph_df = morph_df.rename(columns={morph_id_col: "compound_id"})
    expr_df = expr_df.rename(columns={expr_id_col: "compound_id"})

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

            def _normalize_brd(s: _pd.Series) -> _pd.Series:
                return s.astype(str).str.extract(r"^(BRD-[A-Za-z]\d{8})", expand=False)

            meta_dedup = metadata_df.drop_duplicates(subset=meta_id_col).copy()
            meta_dedup["_norm_id"] = _normalize_brd(meta_dedup[meta_id_col])
            meta_dedup = meta_dedup.dropna(subset=["_norm_id"]).drop_duplicates(
                subset="_norm_id"
            )
            meta_indexed = meta_dedup.set_index("_norm_id")

            merged["_norm_id"] = _normalize_brd(merged["compound_id"])
            new_cols = [
                c
                for c in meta_indexed.columns
                if c not in merged.columns and c != meta_id_col
            ]
            for col in new_cols:
                merged[col] = merged["_norm_id"].map(meta_indexed[col])
            merged = merged.drop(columns=["_norm_id"])

            n_matched = merged[new_cols[0]].notna().sum() if new_cols else 0
            logger.info(
                "Metadata mapped: %d/%d compounds matched via %s.",
                n_matched,
                len(merged),
                meta_id_col,
            )

    return merged


# ---------------------------------------------------------------------------
# Feature detection (Step 5)
# ---------------------------------------------------------------------------


def detect_feature_columns(
    df: pd.DataFrame,  # noqa: F821
) -> tuple[list[str], list[str]]:
    """Identify morphology and expression feature columns.

    Morphology features are CellProfiler columns (``Cells_``, ``Nuclei_``,
    ``Cytoplasm_`` prefixes).  Expression features are L1000 probe IDs
    (``*_at`` suffix).  Falls back to negative selection if ``_at`` pattern
    finds nothing.

    Args:
        df: Merged DataFrame.

    Returns:
        Tuple of (morph_cols, expr_cols).
    """
    _ensure_imports()
    numeric_cols = set(df.select_dtypes(include="number").columns)

    # Morphology: CellProfiler features
    morph_cols = [
        c
        for c in df.columns
        if c in numeric_cols and c.startswith(_CELLPROFILER_PREFIXES)
    ]

    # Expression: L1000 probe IDs (end with _at)
    expr_cols = [c for c in df.columns if c in numeric_cols and c.endswith("_at")]

    # Fallback if _at pattern finds nothing (data format different than expected)
    if len(expr_cols) == 0:
        logger.warning(
            "No '_at' columns found. Using negative selection for expr features."
        )
        _excluded_prefixes = _METADATA_PREFIXES + _CELLPROFILER_PREFIXES
        _excluded_exact = {"compound_id", "treatment_id", "smiles", "split"}
        morph_set = set(morph_cols)
        expr_cols = [
            c
            for c in df.columns
            if c in numeric_cols
            and c not in _excluded_exact
            and not any(c.startswith(p) for p in _excluded_prefixes)
            and c not in morph_set
        ]

    logger.info(
        "detect_feature_columns: %d morph, %d expr (from %d total columns).",
        len(morph_cols),
        len(expr_cols),
        len(df.columns),
    )
    if len(morph_cols) < 100:
        logger.warning(
            "Suspiciously few morph features: %d "
            "(expected ~1000 CellProfiler features).",
            len(morph_cols),
        )
    if len(expr_cols) < 900:
        logger.warning(
            "Suspiciously few expr features: %d (expected 978 L1000 landmark genes).",
            len(expr_cols),
        )
    return morph_cols, expr_cols


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
        if len(split_df) == 0:
            logger.warning(
                "Split '%s' has 0 rows — downstream loading will produce "
                "an empty dataset.",
                split_name,
            )
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
# Main pipeline (Step 8)
# ---------------------------------------------------------------------------


def preprocess_pipeline(cfg: DictConfig) -> dict[str, Path]:  # noqa: F821
    """Run the full preprocessing pipeline from raw profiles to split parquets.

    Pipeline order follows Haghighi et al. (Nature Methods 2022):
    per-plate normalization -> replicate correlation filter ->
    treatment-level aggregation -> cross-modal matching ->
    feature detection -> control removal -> scaffold split ->
    feature QC -> global normalization -> save.

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

    # Config fields with defaults for backward compatibility
    morph_id_col = getattr(cfg.data, "morph_id_col", "Metadata_Sample_Dose")
    expr_id_col = getattr(cfg.data, "expr_id_col", "pert_sample_dose")
    per_plate_normalize = getattr(cfg.data, "per_plate_normalize", True)
    replicate_corr_percentile = getattr(cfg.data, "replicate_corr_percentile", 90)

    # 1. Load raw (replicate-level) profiles
    morph_df, expr_df = load_raw_profiles(raw_dir)
    logger.info(
        "Loaded raw profiles: morph=%s, expr=%s.",
        morph_df.shape,
        expr_df.shape,
    )

    # Detect morph feature columns early (before any transformation)
    morph_feature_cols = [
        c for c in morph_df.columns if c.startswith(_CELLPROFILER_PREFIXES)
    ]
    expr_feature_cols = [c for c in expr_df.columns if c.endswith("_at")]
    if len(expr_feature_cols) == 0:
        # Fallback: all numeric non-metadata columns
        expr_numeric = set(expr_df.select_dtypes(include="number").columns)
        expr_feature_cols = [
            c
            for c in expr_df.columns
            if c in expr_numeric
            and not any(c.startswith(p) for p in _METADATA_PREFIXES)
        ]
    logger.info(
        "Raw feature counts: morph=%d, expr=%d.",
        len(morph_feature_cols),
        len(expr_feature_cols),
    )

    # 2. Per-plate normalization
    # When using replicate_level_cp_normalized.csv.gz, morphology is already
    # z-scored against DMSO controls by the upstream cytominer pipeline.
    # L1000 data is also pre-z-scored.  We only apply per-plate norm when
    # the user explicitly enables it AND provides raw (augmented) profiles.
    if per_plate_normalize:
        # Check if morphology data looks raw (not already normalized).
        # The _normalized file has near-zero mean features; _augmented does not.
        morph_sample_mean = morph_df[morph_feature_cols[:20]].mean().abs().mean()
        if morph_sample_mean > 5.0:
            # Likely raw/augmented data — apply per-plate normalization
            logger.info(
                "Morphology appears raw (mean=%.1f); applying per-plate "
                "normalization.",
                morph_sample_mean,
            )
            if "Metadata_Plate" in morph_df.columns:
                morph_ctrl_col = "Metadata_pert_type"
                morph_ctrl_val = "control"
                if morph_ctrl_col in morph_df.columns:
                    ctrl_vals = morph_df[morph_ctrl_col].unique()
                    for candidate in ["control", "negcon", "DMSO"]:
                        if candidate in ctrl_vals:
                            morph_ctrl_val = candidate
                            break
                morph_df = normalize_per_plate(
                    morph_df,
                    morph_feature_cols,
                    plate_col="Metadata_Plate",
                    control_col=morph_ctrl_col,
                    control_value=morph_ctrl_val,
                )
            else:
                logger.warning(
                    "No Metadata_Plate column in morphology data; "
                    "skipping per-plate normalization for morph."
                )
        else:
            logger.info(
                "Morphology appears pre-normalized (mean=%.2f); "
                "skipping per-plate normalization.",
                morph_sample_mean,
            )

        # Expression: always skip — L1000 data is pre-z-scored.
        logger.info(
            "Skipping per-plate normalization for expression "
            "(L1000 data is already z-scored)."
        )

    # 3. Replicate correlation filter (quality gate)
    if replicate_corr_percentile > 0:
        if (
            "Metadata_pert_type" in morph_df.columns
            and morph_id_col in morph_df.columns
        ):
            morph_ctrl_col = "Metadata_pert_type"
            morph_ctrl_val = "control"
            if morph_ctrl_col in morph_df.columns:
                ctrl_vals = morph_df[morph_ctrl_col].unique()
                for candidate in ["control", "negcon", "DMSO"]:
                    if candidate in ctrl_vals:
                        morph_ctrl_val = candidate
                        break
            morph_df = filter_by_replicate_correlation(
                morph_df,
                morph_feature_cols,
                treatment_col=morph_id_col,
                control_col=morph_ctrl_col,
                control_value=morph_ctrl_val,
                percentile=replicate_corr_percentile,
            )

        if "pert_type" in expr_df.columns and expr_id_col in expr_df.columns:
            expr_ctrl_val = "ctl_vehicle"
            if "pert_type" in expr_df.columns:
                ctrl_vals = expr_df["pert_type"].unique()
                for candidate in ["ctl_vehicle", "control", "DMSO"]:
                    if candidate in ctrl_vals:
                        expr_ctrl_val = candidate
                        break
            expr_df = filter_by_replicate_correlation(
                expr_df,
                expr_feature_cols,
                treatment_col=expr_id_col,
                control_col="pert_type",
                control_value=expr_ctrl_val,
                percentile=replicate_corr_percentile,
            )

    # 4. Aggregate replicates to treatment level (mean)
    morph_df = aggregate_to_treatment_level(morph_df, treatment_col=morph_id_col)
    expr_df = aggregate_to_treatment_level(expr_df, treatment_col=expr_id_col)

    # 5. Load compound metadata (SMILES, MOA) if available
    metadata_df = None
    metadata_path = raw_dir / "metadata" / "repurposing_samples.txt"
    if metadata_path.exists():
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

    # 6. Match treatments across modalities
    df = match_treatments(
        morph_df,
        expr_df,
        metadata_df=metadata_df,
        morph_id_col=morph_id_col,
        expr_id_col=expr_id_col,
    )

    # 7. Detect feature columns (fixed detection)
    morph_cols, expr_cols = detect_feature_columns(df)

    # 8. Remove controls
    df = remove_controls(df)

    # 9. Scaffold split (grouped by compound — all doses same split)
    df = scaffold_split(
        df,
        smiles_col="smiles",
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.seed,
    )
    train_mask = df["split"] == "train"

    # 10. Feature QC (stats computed on train only)
    df, morph_cols, expr_cols = feature_qc(
        df,
        morph_cols,
        expr_cols,
        nan_threshold=cfg.data.nan_threshold,
        train_mask=train_mask,
    )
    # Recompute train_mask after QC may have dropped rows
    train_mask = df["split"] == "train"

    # 11. Global normalization (RobustScaler on train, apply all, clip)
    df, _ = normalize_features(
        df,
        morph_cols,
        expr_cols,
        clip_range=cfg.data.clip_range,
        train_mask=train_mask,
    )

    # 12. Save
    return save_processed_data(df, morph_cols, expr_cols, processed_dir)
