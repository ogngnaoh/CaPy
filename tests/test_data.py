"""Tests for CaPy data pipeline (src/data/).

Covers download, preprocess, featurize, and dataset modules.
Tests skip gracefully when heavy dependencies are missing.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import (
    SIMPLE_SMILES,
    requires_ml,
    requires_numpy,
    requires_pandas,
    requires_rdkit,
    requires_torch,
)

# ===================================================================
# TestDownload — pure-Python, no heavy deps
# ===================================================================


class TestDownload:
    """Tests for src.data.download (no heavy dependencies needed)."""

    def test_file_exists_and_valid_missing(self, tmp_path: Path) -> None:
        """Non-existent file returns False."""
        from src.data.download import _file_exists_and_valid

        assert _file_exists_and_valid(tmp_path / "nope.csv") is False

    def test_file_exists_and_valid_too_small(self, tmp_path: Path) -> None:
        """File smaller than min_size_bytes returns False."""
        from src.data.download import _file_exists_and_valid

        small = tmp_path / "small.csv"
        small.write_text("hi")
        assert _file_exists_and_valid(small, min_size_bytes=1000) is False

    def test_file_exists_and_valid_ok(self, tmp_path: Path) -> None:
        """File meeting size threshold returns True."""
        from src.data.download import _file_exists_and_valid

        big = tmp_path / "big.csv"
        big.write_text("x" * 2000)
        assert _file_exists_and_valid(big, min_size_bytes=1000) is True

    def test_dry_run_does_not_download(self, tmp_path: Path) -> None:
        """Dry-run mode should not create any files."""
        from src.data.download import download_rosetta_profiles

        result = download_rosetta_profiles(tmp_path / "raw", dry_run=True)
        # Result should have keys but no actual files on disk
        assert "morphology" in result
        assert "expression" in result
        assert not result["morphology"].exists()

    def test_idempotent_skip(self, tmp_path: Path) -> None:
        """If files already exist and are valid, download is skipped."""
        from src.data.download import download_rosetta_profiles

        # Create fake pre-existing files
        morph = (
            tmp_path
            / "morphology"
            / "replicate_level_cp_normalized_variable_selected.csv.gz"
        )
        expr = tmp_path / "expression" / "replicate_level_l1k.csv.gz"
        morph.parent.mkdir(parents=True)
        expr.parent.mkdir(parents=True)
        morph.write_text("x" * 2000)
        expr.write_text("x" * 2000)

        result = download_rosetta_profiles(tmp_path, dry_run=False)
        assert result["morphology"] == morph
        assert result["expression"] == expr


# ===================================================================
# TestFeaturize — requires rdkit + torch
# ===================================================================


class TestFeaturize:
    """Tests for src.data.featurize (requires rdkit + torch)."""

    @requires_rdkit
    @requires_torch
    def test_benzene_graph_shape(self) -> None:
        """Benzene (c1ccccc1) should have 6 atoms and 12 edges (6 bonds x 2)."""
        from src.data.featurize import NUM_ATOM_FEATURES, smiles_to_graph

        g = smiles_to_graph("c1ccccc1")
        assert g is not None
        assert g.x.shape == (6, NUM_ATOM_FEATURES)
        assert g.edge_index.shape == (2, 12)

    @requires_rdkit
    @requires_torch
    def test_benzene_aromaticity(self) -> None:
        """All atoms in benzene should be aromatic (feature index 7 = 1)."""
        from src.data.featurize import smiles_to_graph

        g = smiles_to_graph("c1ccccc1")
        assert g is not None
        # Feature 7 is "is aromatic"
        assert (g.x[:, 7] == 1).all()

    @requires_rdkit
    @requires_torch
    def test_both_edge_directions(self) -> None:
        """Each bond should produce edges in both directions."""
        from src.data.featurize import smiles_to_graph

        g = smiles_to_graph("CCO")  # ethanol: C-C-O, 2 bonds
        assert g is not None
        assert g.edge_index.shape[1] == 4  # 2 bonds x 2 directions

        # Verify both directions exist for each bond
        edges = set()
        for col in range(g.edge_index.shape[1]):
            src, dst = g.edge_index[0, col].item(), g.edge_index[1, col].item()
            edges.add((src, dst))

        for src, dst in list(edges):
            assert (dst, src) in edges, f"Missing reverse edge ({dst}, {src})"

    @requires_rdkit
    @requires_torch
    def test_invalid_smiles_returns_none(self) -> None:
        """Invalid SMILES should return None, not crash."""
        from src.data.featurize import smiles_to_graph

        assert smiles_to_graph("NOT_A_SMILES") is None
        assert smiles_to_graph("") is None

    @requires_rdkit
    @requires_torch
    def test_all_simple_smiles_parse(self) -> None:
        """All 10 test molecules should successfully featurize."""
        from src.data.featurize import smiles_to_graph

        for smi in SIMPLE_SMILES:
            g = smiles_to_graph(smi)
            assert g is not None, f"Failed to parse: {smi}"
            assert g.x.shape[0] > 0

    @requires_rdkit
    @requires_torch
    def test_single_atom_molecule(self) -> None:
        """Methane (C) has 1 atom and 0 bonds."""
        from src.data.featurize import NUM_ATOM_FEATURES, smiles_to_graph

        g = smiles_to_graph("C")
        assert g is not None
        assert g.x.shape == (1, NUM_ATOM_FEATURES)
        assert g.edge_index.shape == (2, 0)  # no bonds
        assert g.edge_attr.shape[0] == 0

    @requires_rdkit
    @requires_torch
    def test_featurize_dataset(self) -> None:
        """featurize_dataset should return a dict mapping ids to Data objects."""
        from src.data.featurize import featurize_dataset

        smiles = ["CCO", "c1ccccc1", "NOT_VALID"]
        ids = ["cpd1", "cpd2", "cpd3"]
        result = featurize_dataset(smiles, ids)
        assert len(result) == 2  # one invalid
        assert "cpd1" in result
        assert "cpd2" in result
        assert "cpd3" not in result

    @requires_rdkit
    @requires_torch
    def test_feature_dims_constants(self) -> None:
        """Public constants should be consistent."""
        from src.data.featurize import (
            ATOM_FEATURE_DIMS,
            BOND_FEATURE_DIMS,
            NUM_ATOM_FEATURES,
            NUM_BOND_FEATURES,
        )

        assert NUM_ATOM_FEATURES == 9
        assert NUM_BOND_FEATURES == 4
        assert len(ATOM_FEATURE_DIMS) == NUM_ATOM_FEATURES
        assert len(BOND_FEATURE_DIMS) == NUM_BOND_FEATURES
        # All dims should be positive integers
        for d in ATOM_FEATURE_DIMS + BOND_FEATURE_DIMS:
            assert isinstance(d, int)
            assert d > 0

    @requires_torch
    @requires_rdkit
    def test_zero_atom_guard(self) -> None:
        """smiles_to_graph returns None for molecules that parse to 0 atoms."""
        from src.data.featurize import smiles_to_graph

        for smiles in ["CCO", "c1ccccc1", "", "invalid_smiles"]:
            result = smiles_to_graph(smiles)
            if result is not None:
                assert result.x.shape[0] > 0, f"Got 0-atom graph for: {smiles}"

    @requires_rdkit
    @requires_torch
    def test_atom_features_in_range(self) -> None:
        """Atom feature values should be within their vocabulary range."""
        from src.data.featurize import ATOM_FEATURE_DIMS, smiles_to_graph

        for smi in SIMPLE_SMILES:
            g = smiles_to_graph(smi)
            if g is None:
                continue
            for feat_idx, dim in enumerate(ATOM_FEATURE_DIMS):
                col = g.x[:, feat_idx]
                assert col.min() >= 0, f"{smi} feat {feat_idx}: min={col.min()}"
                assert (
                    col.max() < dim
                ), f"{smi} feat {feat_idx}: max={col.max()} >= {dim}"


# ===================================================================
# TestPreprocess — requires pandas + numpy
# ===================================================================


class TestPreprocess:
    """Tests for src.data.preprocess (requires pandas + numpy)."""

    @requires_pandas
    def test_remove_controls(self) -> None:
        """DMSO and empty SMILES should be removed."""
        import pandas as pd

        from src.data.preprocess import remove_controls

        df = pd.DataFrame(
            {
                "compound_id": ["cpd1", "DMSO_ctrl", "cpd3", "cpd4"],
                "smiles": ["CCO", "O", "", "c1ccccc1"],
            }
        )
        result = remove_controls(df)
        assert len(result) == 2
        assert "DMSO_ctrl" not in result["compound_id"].values
        assert "" not in result["smiles"].values

    @requires_pandas
    @requires_numpy
    def test_zero_variance_removal(self) -> None:
        """Features with zero variance should be dropped."""
        import pandas as pd

        from src.data.preprocess import feature_qc

        df = pd.DataFrame(
            {
                "feat_a": [1.0, 2.0, 3.0, 4.0],
                "feat_b": [5.0, 5.0, 5.0, 5.0],  # zero variance
                "expr_a": [0.1, 0.2, 0.3, 0.4],
            }
        )
        _, morph_kept, expr_kept = feature_qc(
            df,
            morph_cols=["feat_a", "feat_b"],
            expr_cols=["expr_a"],
            nan_threshold=0.5,
        )
        assert "feat_a" in morph_kept
        assert "feat_b" not in morph_kept
        assert "expr_a" in expr_kept

    @requires_pandas
    @requires_numpy
    def test_high_nan_removal(self) -> None:
        """Features with >50% NaN should be dropped."""
        import numpy as np
        import pandas as pd

        from src.data.preprocess import feature_qc

        df = pd.DataFrame(
            {
                "feat_ok": [1.0, 2.0, 3.0, 4.0],
                "feat_nan": [np.nan, np.nan, np.nan, 1.0],  # 75% NaN
                "expr_ok": [0.1, 0.2, 0.3, 0.4],
            }
        )
        _, morph_kept, _ = feature_qc(
            df,
            morph_cols=["feat_ok", "feat_nan"],
            expr_cols=["expr_ok"],
            nan_threshold=0.5,
        )
        assert "feat_ok" in morph_kept
        assert "feat_nan" not in morph_kept

    @requires_pandas
    @requires_numpy
    def test_normalize_clip_range(self) -> None:
        """Morphology values should be clipped to [-clip_range, clip_range]."""
        import pandas as pd

        from src.data.preprocess import normalize_features

        df = pd.DataFrame(
            {
                "morph_a": [1.0, 2.0, 3.0, 100.0],  # outlier
                "expr_a": [0.0, 0.0, 0.0, 0.0],
            }
        )
        result, _ = normalize_features(
            df,
            morph_cols=["morph_a"],
            expr_cols=["expr_a"],
            clip_range=5.0,
        )
        assert result["morph_a"].max() <= 5.0
        assert result["morph_a"].min() >= -5.0

    @requires_pandas
    @requires_numpy
    def test_normalize_train_mask_no_leakage(self) -> None:
        """Scaler should be fitted on train rows only when train_mask given."""
        import pandas as pd

        from src.data.preprocess import normalize_features

        df = pd.DataFrame(
            {
                "morph_a": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0],
                "expr_a": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        train_mask = pd.Series([True, True, True, True, False, False])

        result_with_mask, _ = normalize_features(
            df,
            morph_cols=["morph_a"],
            expr_cols=["expr_a"],
            clip_range=5.0,
            train_mask=train_mask,
        )
        result_without_mask, _ = normalize_features(
            df,
            morph_cols=["morph_a"],
            expr_cols=["expr_a"],
            clip_range=5.0,
        )
        # With train_mask, scaler is fitted on [1,2,3,4] only,
        # so test rows (100,200) should scale differently
        train_vals_with = result_with_mask.loc[train_mask, "morph_a"]
        train_vals_without = result_without_mask.loc[train_mask, "morph_a"]
        assert not train_vals_with.equals(train_vals_without)

    @requires_pandas
    @requires_rdkit
    def test_scaffold_split_proportions(self) -> None:
        """Split ratios should be approximately correct."""
        import pandas as pd

        from src.data.preprocess import scaffold_split

        # Create a dataset with diverse scaffolds
        smiles = SIMPLE_SMILES * 10  # 100 molecules
        df = pd.DataFrame(
            {
                "smiles": smiles,
                "compound_id": [f"cpd_{i}" for i in range(len(smiles))],
            }
        )
        result = scaffold_split(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        n = len(result)
        train_frac = (result["split"] == "train").sum() / n
        val_frac = (result["split"] == "val").sum() / n
        test_frac = (result["split"] == "test").sum() / n

        # Allow generous tolerance since scaffold groups are indivisible
        assert 0.5 < train_frac < 0.9, f"train={train_frac:.2f}"
        assert val_frac > 0.0, f"val={val_frac:.2f}"
        assert test_frac > 0.0, f"test={test_frac:.2f}"
        # All rows should be assigned
        assert result["split"].notna().all()

    @requires_pandas
    @requires_rdkit
    def test_scaffold_split_determinism(self) -> None:
        """Same seed should produce identical splits."""
        import pandas as pd

        from src.data.preprocess import scaffold_split

        df = pd.DataFrame(
            {
                "smiles": SIMPLE_SMILES,
                "compound_id": [f"cpd_{i}" for i in range(len(SIMPLE_SMILES))],
            }
        )
        r1 = scaffold_split(df, seed=42)
        r2 = scaffold_split(df, seed=42)
        assert (r1["split"].values == r2["split"].values).all()

    @requires_pandas
    @requires_numpy
    def test_aggregate_to_compound_level(self) -> None:
        """Aggregation should produce median of numeric cols and preserve metadata."""
        import numpy as np
        import pandas as pd

        from src.data.preprocess import aggregate_to_compound_level

        df = pd.DataFrame(
            {
                "compound_id": ["cpd1", "cpd1", "cpd1", "cpd2", "cpd2"],
                "feat_a": [1.0, 3.0, 5.0, 10.0, 20.0],
                "feat_b": [2.0, 4.0, 6.0, 100.0, 200.0],
                "plate": ["P1", "P2", "P3", "P1", "P2"],
            }
        )
        result = aggregate_to_compound_level(df, compound_col="compound_id")
        assert len(result) == 2

        cpd1 = result.loc[result["compound_id"] == "cpd1"].iloc[0]
        assert np.isclose(cpd1["feat_a"], 3.0)  # median of [1, 3, 5]
        assert np.isclose(cpd1["feat_b"], 4.0)  # median of [2, 4, 6]
        assert cpd1["plate"] == "P1"  # first value preserved

        cpd2 = result.loc[result["compound_id"] == "cpd2"].iloc[0]
        assert np.isclose(cpd2["feat_a"], 15.0)  # median of [10, 20]
        assert np.isclose(cpd2["feat_b"], 150.0)  # median of [100, 200]

    @requires_pandas
    def test_match_compounds_with_metadata(self) -> None:
        """match_compounds should map metadata columns via compound ID."""
        import pandas as pd

        from src.data.preprocess import match_compounds

        morph_df = pd.DataFrame(
            {
                "Metadata_broad_sample": [
                    "BRD-K12345678",
                    "BRD-K23456789",
                    "BRD-K34567890",
                ],
                "feat_a": [1.0, 2.0, 3.0],
            }
        )
        expr_df = pd.DataFrame(
            {
                "pert_id": [
                    "BRD-K12345678",
                    "BRD-K23456789",
                    "BRD-K34567890",
                ],
                "gene_x": [0.5, 0.6, 0.7],
            }
        )
        metadata_df = pd.DataFrame(
            {
                "broad_id": [
                    "BRD-K12345678",
                    "BRD-K23456789",
                    "BRD-K34567890",
                ],
                "smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
                "pert_iname": ["ethanol", "benzene", "acetic_acid"],
            }
        )

        result = match_compounds(morph_df, expr_df, metadata_df=metadata_df)
        assert len(result) == 3
        assert "smiles" in result.columns
        assert "compound_id" in result.columns
        assert result["smiles"].notna().all()

    @requires_pandas
    def test_match_compounds_normalizes_brd_ids(self) -> None:
        """Metadata with long BRD IDs should match short compound IDs."""
        import pandas as pd

        from src.data.preprocess import match_compounds

        # Morph/expr use short BRD IDs
        morph_df = pd.DataFrame(
            {
                "Metadata_broad_sample": ["BRD-K12345678", "BRD-K23456789"],
                "feat_a": [1.0, 2.0],
            }
        )
        expr_df = pd.DataFrame(
            {
                "pert_id": ["BRD-K12345678", "BRD-K23456789"],
                "gene_x": [0.5, 0.6],
            }
        )
        # Metadata uses long BRD IDs (with batch/plate suffix)
        metadata_df = pd.DataFrame(
            {
                "broad_id": [
                    "BRD-K12345678-001-01-1",
                    "BRD-K23456789-300-15-9",
                ],
                "smiles": ["CCO", "c1ccccc1"],
                "pert_iname": ["ethanol", "benzene"],
            }
        )

        result = match_compounds(morph_df, expr_df, metadata_df=metadata_df)
        assert len(result) == 2
        assert result["smiles"].notna().all()
        assert result["smiles"].tolist() == ["CCO", "c1ccccc1"]

    @requires_pandas
    def test_expr_cols_excludes_string_metadata(self) -> None:
        """Fallback expr_cols detection should skip non-numeric columns."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "compound_id": ["BRD-001", "BRD-002"],
                "Cells_area": [1.0, 2.0],
                "gene_A": [0.5, 0.6],
                "pert_type": ["trt_cp", "trt_cp"],
            }
        )
        numeric_cols = set(df.select_dtypes(include="number").columns)
        morph_cols = [
            c for c in df.columns if c.startswith(("Cells_", "Nuclei_", "Cytoplasm_"))
        ]
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
        assert "gene_A" in non_meta
        assert "pert_type" not in non_meta

    @requires_pandas
    def test_primary_detection_excludes_string_suffixed_cols(self) -> None:
        """Primary _morph/_expr suffix detection must skip non-numeric columns.

        After merge with suffixes=("_morph", "_expr"), string columns like
        pert_type become pert_type_morph / pert_type_expr.  These must be
        excluded from feature column lists to prevent TypeError in feature_qc.
        """
        import pandas as pd

        # Simulate a merged DataFrame with string cols that got suffixed
        df = pd.DataFrame(
            {
                "compound_id": ["BRD-001", "BRD-002", "BRD-003"],
                "feat_a_morph": [1.0, 2.0, 3.0],
                "pert_type_morph": ["trt", "trt", "trt"],  # string!
                "gene_x_expr": [0.5, 0.6, 0.7],
                "pert_type_expr": ["trt_cp", "trt_cp", "trt_cp"],  # string!
            }
        )
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
        assert "feat_a_morph" in morph_cols
        assert "pert_type_morph" not in morph_cols
        assert "gene_x_expr" in expr_cols
        assert "pert_type_expr" not in expr_cols

    @requires_pandas
    @requires_numpy
    def test_feature_qc_skips_non_numeric_cols(self) -> None:
        """feature_qc should not crash when given non-numeric columns."""
        import pandas as pd

        from src.data.preprocess import feature_qc

        df = pd.DataFrame(
            {
                "feat_a": [1.0, 2.0, 3.0, 4.0],
                "str_col": ["a", "b", "c", "d"],  # non-numeric
                "expr_a": [0.1, 0.2, 0.3, 0.4],
            }
        )
        # Pass the string column as if it were a morph feature
        _, morph_kept, expr_kept = feature_qc(
            df,
            morph_cols=["feat_a", "str_col"],
            expr_cols=["expr_a"],
            nan_threshold=0.5,
        )
        assert "feat_a" in morph_kept
        assert "str_col" not in morph_kept
        assert "expr_a" in expr_kept

    @requires_pandas
    @requires_numpy
    def test_normalize_clips_expression_features(self) -> None:
        """Expression features should be clipped to [-clip_range, clip_range].

        Uses data that looks z-scored (mean≈0, std≈1) so re-normalization is
        skipped, but contains extreme outliers from median aggregation.
        """
        import numpy as np
        import pandas as pd

        from src.data.preprocess import normalize_features

        rng = np.random.RandomState(42)
        n = 100
        # z-scored bulk with a few extreme outliers (mimics L1000 after
        # median aggregation of replicates)
        expr_a = rng.standard_normal(n)
        expr_a[0] = 15.0  # outlier
        expr_b = rng.standard_normal(n)
        expr_b[1] = -20.0  # outlier

        df = pd.DataFrame(
            {
                "morph_a": rng.standard_normal(n),
                "expr_a": expr_a,
                "expr_b": expr_b,
            }
        )
        result, _ = normalize_features(
            df,
            morph_cols=["morph_a"],
            expr_cols=["expr_a", "expr_b"],
            clip_range=5.0,
        )
        assert result["expr_a"].max() <= 5.0
        assert result["expr_a"].min() >= -5.0
        assert result["expr_b"].max() <= 5.0
        assert result["expr_b"].min() >= -5.0

    @requires_pandas
    @requires_numpy
    def test_normalize_logs_expression_feature_stats(self, caplog) -> None:
        """normalize_features should log expression feature statistics."""
        import logging

        import pandas as pd

        from src.data.preprocess import normalize_features

        df = pd.DataFrame(
            {
                "morph_a": [1.0, 2.0, 3.0, 4.0],
                "expr_a": [0.0, 1.0, -1.0, 0.5],
                "expr_b": [0.1, 0.1, 0.1, 0.1],  # near-zero variance
            }
        )
        with caplog.at_level(logging.INFO, logger="src.data.preprocess"):
            normalize_features(
                df,
                morph_cols=["morph_a"],
                expr_cols=["expr_a", "expr_b"],
                clip_range=5.0,
            )

        log_text = caplog.text.lower()
        assert "expr feature" in log_text or "near-zero variance" in log_text

    @requires_pandas
    @requires_rdkit
    def test_scaffold_split_no_leakage(self) -> None:
        """No scaffold should appear in more than one split."""
        import pandas as pd

        from src.data.preprocess import _get_scaffold, scaffold_split

        df = pd.DataFrame(
            {
                "smiles": SIMPLE_SMILES * 5,
                "compound_id": [f"cpd_{i}" for i in range(len(SIMPLE_SMILES) * 5)],
            }
        )
        result = scaffold_split(df, seed=42)

        for split_name in ["train", "val", "test"]:
            split_smiles = result.loc[result["split"] == split_name, "smiles"]
            split_scaffolds = {_get_scaffold(s) for s in split_smiles}
            other_smiles = result.loc[result["split"] != split_name, "smiles"]
            other_scaffolds = {_get_scaffold(s) for s in other_smiles}
            overlap = split_scaffolds & other_scaffolds
            assert len(overlap) == 0, f"Scaffold leakage in {split_name}: {overlap}"


# ===================================================================
# TestDataset — requires all ML deps
# ===================================================================


class TestDataset:
    """Tests for src.data.dataset (requires torch + rdkit + pyg)."""

    @requires_ml
    def test_dataset_length(self) -> None:
        """Dataset length should match input size."""
        import torch

        from src.data.dataset import CaPyDataset
        from src.data.featurize import smiles_to_graph

        graphs = [smiles_to_graph(s) for s in SIMPLE_SMILES[:5]]
        graphs = [g for g in graphs if g is not None]
        n = len(graphs)
        morph = torch.randn(n, 100)
        expr = torch.randn(n, 978)
        ids = [f"cpd_{i}" for i in range(n)]

        ds = CaPyDataset(graphs, morph, expr, ids)
        assert len(ds) == n

    @requires_ml
    def test_getitem_types_and_shapes(self) -> None:
        """__getitem__ should return (Data, 1-D tensor, 1-D tensor)."""
        import torch
        from torch_geometric.data import Data

        from src.data.dataset import CaPyDataset
        from src.data.featurize import smiles_to_graph

        graphs = [smiles_to_graph(s) for s in SIMPLE_SMILES[:3]]
        graphs = [g for g in graphs if g is not None]
        n = len(graphs)
        morph = torch.randn(n, 50)
        expr = torch.randn(n, 30)
        ids = [f"cpd_{i}" for i in range(n)]

        ds = CaPyDataset(graphs, morph, expr, ids)
        g, m, e = ds[0]
        assert isinstance(g, Data)
        assert m.shape == (50,)
        assert e.shape == (30,)

    @requires_ml
    def test_collate_fn_shapes(self) -> None:
        """capy_collate_fn should batch graphs and stack vectors."""
        import torch
        from torch_geometric.data import Batch

        from src.data.dataset import CaPyDataset, capy_collate_fn
        from src.data.featurize import smiles_to_graph

        graphs = [smiles_to_graph(s) for s in SIMPLE_SMILES[:4]]
        graphs = [g for g in graphs if g is not None]
        n = len(graphs)
        morph = torch.randn(n, 50)
        expr = torch.randn(n, 30)
        ids = [f"cpd_{i}" for i in range(n)]

        ds = CaPyDataset(graphs, morph, expr, ids)
        batch = [ds[i] for i in range(n)]
        bg, bm, be = capy_collate_fn(batch)

        assert isinstance(bg, Batch)
        assert bm.shape == (n, 50)
        assert be.shape == (n, 30)

    @requires_ml
    def test_dataloader_integration(self) -> None:
        """CaPyDataset should work with torch DataLoader."""
        import torch
        from torch.utils.data import DataLoader

        from src.data.dataset import CaPyDataset, capy_collate_fn
        from src.data.featurize import smiles_to_graph

        graphs = [smiles_to_graph(s) for s in SIMPLE_SMILES[:6]]
        graphs = [g for g in graphs if g is not None]
        n = len(graphs)
        morph = torch.randn(n, 50)
        expr = torch.randn(n, 30)
        ids = [f"cpd_{i}" for i in range(n)]

        ds = CaPyDataset(graphs, morph, expr, ids)
        loader = DataLoader(ds, batch_size=3, collate_fn=capy_collate_fn)

        batches = list(loader)
        assert len(batches) >= 1
        bg, bm, be = batches[0]
        assert bm.shape[0] <= 3

    @requires_ml
    def test_no_nans_in_features(self) -> None:
        """Morph/expr tensors should never contain NaN."""
        import torch

        from src.data.dataset import CaPyDataset
        from src.data.featurize import smiles_to_graph

        graphs = [smiles_to_graph(s) for s in SIMPLE_SMILES[:5]]
        graphs = [g for g in graphs if g is not None]
        n = len(graphs)
        morph = torch.randn(n, 50)
        expr = torch.randn(n, 30)
        ids = [f"cpd_{i}" for i in range(n)]

        ds = CaPyDataset(graphs, morph, expr, ids)
        for i in range(len(ds)):
            _, m, e = ds[i]
            assert not torch.isnan(m).any()
            assert not torch.isnan(e).any()


# ===================================================================
# TestFeatureDetection — validates morph/expr column detection logic
# ===================================================================


class TestFeatureDetection:
    """Tests for morph/expr column detection logic."""

    @requires_pandas
    def test_expr_cols_excludes_metadata(self) -> None:
        """pert_dose_expr and pert_time_expr must NOT be in expr_cols."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "compound_id": ["BRD-K001", "BRD-K002"],
                "gene_A_expr": [1.0, 2.0],
                "gene_B_expr": [3.0, 4.0],
                "pert_dose_expr": [10.0, 5.0],
                "pert_time_expr": [24.0, 48.0],
            }
        )
        numeric_cols = set(df.select_dtypes(include="number").columns)
        _METADATA_PREFIXES = ("pert_", "det_", "distil_", "cell_", "Metadata_", "rna_")
        expr_cols = [
            c
            for c in df.columns
            if c != "compound_id"
            and c.endswith("_expr")
            and c in numeric_cols
            and not any(c.startswith(p) for p in _METADATA_PREFIXES)
        ]
        assert "pert_dose_expr" not in expr_cols
        assert "pert_time_expr" not in expr_cols
        assert "gene_A_expr" in expr_cols
        assert "gene_B_expr" in expr_cols
        assert len(expr_cols) == 2

    @requires_pandas
    def test_morph_cols_includes_cellprofiler_features(self) -> None:
        """CellProfiler columns (Cells_, Nuclei_, Cytoplasm_) must be in morph_cols."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "compound_id": ["BRD-K001", "BRD-K002"],
                "Cells_AreaShape_Area": [100.0, 200.0],
                "Nuclei_Texture_Entropy": [0.5, 0.6],
                "Cytoplasm_Intensity_Mean": [300.0, 400.0],
                "shared_col_morph": [1.0, 2.0],
                "pert_type_morph": ["trt_cp", "trt_cp"],
            }
        )
        numeric_cols = set(df.select_dtypes(include="number").columns)
        _CELLPROFILER_PREFIXES = ("Cells_", "Nuclei_", "Cytoplasm_")
        _METADATA_PREFIXES = ("pert_", "det_", "distil_", "cell_", "Metadata_", "rna_")
        morph_cols = [
            c
            for c in df.columns
            if c in numeric_cols
            and c != "compound_id"
            and (
                c.startswith(_CELLPROFILER_PREFIXES)
                or (
                    c.endswith("_morph")
                    and not any(c.startswith(p) for p in _METADATA_PREFIXES)
                )
            )
        ]
        assert "Cells_AreaShape_Area" in morph_cols
        assert "Nuclei_Texture_Entropy" in morph_cols
        assert "Cytoplasm_Intensity_Mean" in morph_cols
        assert "shared_col_morph" in morph_cols
        assert "pert_type_morph" not in morph_cols
        assert len(morph_cols) == 4
