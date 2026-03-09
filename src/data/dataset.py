"""CaPyDataset — aligned tri-modal data loader for contrastive learning.

WHY THIS WORKS
--------------
Each ``__getitem__`` call returns a ``(graph, morph_vector, expr_vector)``
triplet for the *same* compound.  This alignment is the foundation of
contrastive learning: within a batch, index *i* across all three
modalities refers to the same drug, forming positive pairs.  All other
index combinations are treated as negatives by InfoNCE.

The custom ``capy_collate_fn`` handles the asymmetry between modalities:
PyG graphs have variable sizes (different atoms/bonds) and must be
batched into a single ``Batch`` object (which tracks graph membership
via ``batch`` vector), while the fixed-length morphology and expression
vectors are simply stacked with ``torch.stack``.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

_torch = None
_Data = None
_Batch = None


def _ensure_imports() -> None:
    global _torch, _Data, _Batch  # noqa: PLW0603
    if _torch is None:
        import torch

        _torch = torch
    if _Data is None:
        from torch_geometric.data import Data

        _Data = Data
    if _Batch is None:
        from torch_geometric.data import Batch

        _Batch = Batch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CaPyDataset:
    """Tri-modal dataset returning aligned (graph, morph, expr) triplets.

    This class implements the ``torch.utils.data.Dataset`` interface
    (``__len__`` and ``__getitem__``) without inheriting from it at
    import time, so the module can be imported even when torch is
    unavailable.  The actual ``Dataset`` base class is mixed in at
    instantiation.

    Args:
        mol_graphs: List of PyG ``Data`` objects (one per compound).
        morph_features: Float tensor of shape ``[N, morph_dim]``.
        expr_features: Float tensor of shape ``[N, expr_dim]``.
        compound_ids: List of compound identifier strings, aligned
            with the other three arguments.
    """

    def __init__(
        self,
        mol_graphs: list,
        morph_features: torch.Tensor,  # noqa: F821
        expr_features: torch.Tensor,  # noqa: F821
        compound_ids: list[str],
    ) -> None:
        _ensure_imports()
        assert len(mol_graphs) == morph_features.shape[0] == expr_features.shape[0]
        assert len(compound_ids) == len(mol_graphs)
        if len(mol_graphs) == 0:
            logger.warning("CaPyDataset created with 0 compounds.")

        self.mol_graphs = mol_graphs
        self.morph_features = morph_features
        self.expr_features = expr_features
        self.compound_ids = compound_ids

    def __len__(self) -> int:
        """Return the number of compounds in this dataset split."""
        return len(self.mol_graphs)

    def __getitem__(self, idx: int) -> tuple:
        """Return the (graph, morph_vector, expr_vector) triplet at *idx*.

        Args:
            idx: Integer index into the dataset.

        Returns:
            Tuple of (PyG Data, morph tensor [morph_dim], expr tensor [expr_dim]).
        """
        return (
            self.mol_graphs[idx],
            self.morph_features[idx],
            self.expr_features[idx],
        )


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def capy_collate_fn(
    batch: list[tuple],
) -> tuple:
    """Collate a list of (graph, morph, expr) triplets into a batch.

    Graphs are batched via PyG ``Batch.from_data_list`` (which
    concatenates adjacency matrices and adds a ``batch`` vector).
    Morphology and expression vectors are stacked into 2-D tensors.

    Args:
        batch: List of triplets from ``CaPyDataset.__getitem__``.

    Returns:
        Tuple of (Batch, morph_tensor [B, morph_dim], expr_tensor [B, expr_dim]).
    """
    _ensure_imports()
    graphs, morphs, exprs = zip(*batch)
    batched_graph = _Batch.from_data_list(list(graphs))
    morph_tensor = _torch.stack(morphs)
    expr_tensor = _torch.stack(exprs)
    return batched_graph, morph_tensor, expr_tensor


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_split_dataset(
    processed_dir: str | Path,
    split: str,
    mol_graphs: dict[str, Data],  # noqa: F821
) -> CaPyDataset:
    """Load a specific split from processed parquet files.

    Args:
        processed_dir: Directory containing ``{split}.parquet`` and
            ``feature_columns.json``.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        mol_graphs: Mapping from compound_id to PyG ``Data`` objects
            (produced by ``featurize_dataset``).

    Returns:
        A ``CaPyDataset`` instance for the requested split.
    """
    _ensure_imports()
    import pandas as pd

    processed_dir = Path(processed_dir)

    # Load feature column names
    with open(processed_dir / "feature_columns.json") as f:
        col_info = json.load(f)
    morph_cols = col_info["morph_cols"]
    expr_cols = col_info["expr_cols"]

    # Load split data
    df = pd.read_parquet(processed_dir / f"{split}.parquet")
    n_before = len(df)

    # Filter to compounds that have graphs
    df = df.loc[df["compound_id"].isin(mol_graphs)].reset_index(drop=True)
    if len(df) < n_before:
        logger.warning(
            "%s split: %d/%d compounds dropped (missing molecular graphs).",
            split,
            n_before - len(df),
            n_before,
        )

    graphs = [mol_graphs[cid] for cid in df["compound_id"]]
    morph = _torch.tensor(df[morph_cols].values, dtype=_torch.float32)
    expr = _torch.tensor(df[expr_cols].values, dtype=_torch.float32)
    compound_ids = df["compound_id"].tolist()

    logger.info(
        "Loaded %s split: %d compounds, morph_dim=%d, expr_dim=%d.",
        split,
        len(graphs),
        morph.shape[1],
        expr.shape[1],
    )
    return CaPyDataset(graphs, morph, expr, compound_ids)
