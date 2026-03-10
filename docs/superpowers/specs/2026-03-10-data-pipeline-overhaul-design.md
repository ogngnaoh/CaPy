# CaPy Data Pipeline Overhaul — Design Spec

**Date:** 2026-03-10
**Status:** APPROVED (revised 2026-03-10 after audit)

## Problem

Training exhibits embedding collapse, loss plateau, and poor retrieval metrics. Root cause: 3 critical bugs and 2 high-severity issues in the data pipeline.

## Bugs

| # | Severity | Issue |
|---|----------|-------|
| 1 | CRITICAL | Wrong alignment columns: `Metadata_broad_sample`/`pert_id` (compound-only) instead of `Metadata_Sample_Dose`/`pert_sample_dose` (treatment-level) |
| 2 | CRITICAL | L1000 features detected by `_expr` merge suffix, but probe IDs (`*_at`) don't overlap with CellProfiler names so suffix never added |
| 3 | CRITICAL | Downloads variable-selected file (63-727 features) instead of full normalized file (~1,677 features) |
| 4 | HIGH | Double normalization: per-plate z-score on already-normalized data destroys variance |
| 5 | HIGH | No replicate correlation filter (Rosetta uses 90th percentile) |

## Design Decisions

- Training samples = treatment-level (compound+dose), not compound-level
- Morphology file = `replicate_level_cp_normalized.csv.gz` (pre-z-scored by cytominer, all features retained)
- Expression = `replicate_level_l1k.csv.gz` (already z-scored by L1000 pipeline)
- Normalization = morph: RobustScaler (global, train-only); expr: verify z-score + clip
- **No per-plate z-score** when using `_normalized` file (already done upstream by cytominer)
- Splitting = scaffold-grouped (all doses of same compound in same split)
- Quality filter = 90th percentile replicate correlation
- NaN threshold = 5% (matches Rosetta reference, was 50%)

## Pipeline Order

1. Load pre-normalized profiles
2. Replicate correlation filter (90th percentile)
3. Aggregate replicates to treatment-level (mean)
4. Match treatments + merge SMILES metadata
5. Identify feature columns (CellProfiler prefixes + `_at` suffix)
6. Remove controls
7. Scaffold split (grouped by compound)
8. Feature QC (train-only stats, 5% NaN threshold)
9. Global normalization (RobustScaler morph / verify+clip expr)
10. Save

## Rosetta Reference

```python
ds_info_dict = {
    "CDRP-bio": ["CDRPBIO-BBBC036-Bray", ["Metadata_Sample_Dose", "pert_sample_dose"]],
}
# Feature patterns: CP = "Cells_|Cytoplasm_|Nuclei_", L1K = "_at"
# Aggregation: .groupby(labelCol)[features].mean()
# Quality filter: replicate correlation > 90th percentile of null
# Reference uses replicate_level_cp_normalized.csv.gz (pre-z-scored by cytominer)
```

## S3 Files Available

| File | Size | Description |
|------|------|-------------|
| `replicate_level_cp_augmented.csv.gz` | 290 MB | Raw CellProfiler features (needs per-plate norm) |
| `replicate_level_cp_normalized.csv.gz` | 298 MB | Z-scored by cytominer (what reference uses) |
| `replicate_level_cp_normalized_variable_selected.csv.gz` | 112 MB | Normalized + feature-selected (~1,000 features) |
| `replicate_level_l1k.csv.gz` | 25 MB | L1000 (pre-z-scored) |
