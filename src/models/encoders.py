# WHY THIS WORKS ──────────────────────────────────────────────────────
# Three encoders map heterogeneous inputs into a *shared* 256-dim unit
# hypersphere.  L2 normalization at the output is critical: InfoNCE
# uses cosine similarity (dot product of unit vectors), so all
# embeddings must live on the same sphere for the loss to be meaningful.
#
# **MolecularEncoder (GIN):**
# Graph Isomorphism Network is the most expressive message-passing GNN
# under the Weisfeiler-Leman hierarchy.  Each GINConv layer updates
# node h via h' = MLP((1+ε)·h + Σ_neighbors h_j), where ε is learned
# (train_eps=True).  We use 5 layers so information propagates up to 5
# hops — enough to cover most drug-like molecules (diameter ≤ 10).
# Global mean pooling aggregates node embeddings into a single graph
# vector; mean (not sum) gives size-invariant representations.
#
# **AtomEncoder:**
# OGB convention — each of the 9 atom features (atomic number, degree,
# etc.) is stored as an integer index.  We look up each in its own
# Embedding table and *sum* the resulting vectors.  Summing (not
# concatenating) keeps the hidden dimension constant regardless of
# feature count, matching the GIN layer width.
#
# **TabularEncoder (MLP):**
# For morphology (~1000 floats) and expression (978 floats), a simple
# MLP with residual connections works well.  We use LayerNorm (not
# BatchNorm) because LN is independent of batch statistics — stable
# with small batches and during evaluation.  Residual connections ease
# gradient flow through the 2 hidden blocks and let the network learn
# an identity mapping if the extra capacity isn't needed.
# ─────────────────────────────────────────────────────────────────────

"""Encoders for CaPy: GIN for molecules, MLP for tabular modalities.

Note: Unlike pure-function modules (losses.py, featurize.py), we import
torch eagerly here because ``nn.Module`` must be available at class
definition time.  The lazy ``__getattr__`` in ``src/models/__init__.py``
ensures this module is only loaded when one of its classes is accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as f  # noqa: N812
from torch_geometric.nn import GINConv, global_mean_pool

from src.data.featurize import ATOM_FEATURE_DIMS
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# AtomEncoder
# ---------------------------------------------------------------------------


class AtomEncoder(nn.Module):
    """Embed integer atom features into a dense vector via learned tables.

    Each of the 9 atom feature slots has its own ``nn.Embedding``.
    Embeddings are **summed** (OGB convention) to produce a single
    ``[num_atoms, hidden_dim]`` matrix.

    Args:
        hidden_dim: Embedding dimension (must match GIN hidden_dim).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, hidden_dim) for vocab_size in ATOM_FEATURE_DIMS]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed atom features.

        Args:
            x: Integer feature matrix, shape ``[num_atoms, 9]``.

        Returns:
            Dense embeddings, shape ``[num_atoms, hidden_dim]``.
        """
        out = torch.zeros(x.size(0), self.embeddings[0].embedding_dim, device=x.device)
        for i, emb in enumerate(self.embeddings):
            out = out + emb(x[:, i])
        return out


# ---------------------------------------------------------------------------
# MolecularEncoder (GIN)
# ---------------------------------------------------------------------------


class MolecularEncoder(nn.Module):
    """5-layer GIN with global mean pooling and projection head.

    Args:
        cfg: Config with ``model.gin.num_layers``, ``model.gin.hidden_dim``,
            and ``model.embedding_dim``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        num_layers: int = cfg.model.gin.num_layers
        hidden_dim: int = cfg.model.gin.hidden_dim
        embedding_dim: int = cfg.model.embedding_dim

        self.atom_encoder = AtomEncoder(hidden_dim)

        # GIN layers: MLP inside GINConv has no BN; BN is applied externally
        # after each conv (standard OGB pattern).
        # NOTE: edge_attr from featurize.py is intentionally unused — GINConv
        # operates on node features only. Bond features are stored for potential
        # future use with GINEConv but do not affect the current architecture.
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Projection head: hidden_dim → mlp_hidden → embedding_dim
        proj_hidden: int = cfg.model.mlp.hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, embedding_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of molecular graphs.

        Args:
            x: Atom feature matrix ``[total_atoms, 9]``.
            edge_index: COO edge indices ``[2, total_edges]``.
            batch: Graph membership vector ``[total_atoms]``.

        Returns:
            L2-normalized embeddings ``[B, embedding_dim]``.
        """
        h = self.atom_encoder(x)

        for conv, bn in zip(self.convs, self.bns):
            h = f.relu(bn(conv(h, edge_index)))

        # Pool to graph-level
        h = global_mean_pool(h, batch)

        # Project and normalize
        h = self.projection(h)
        return f.normalize(h, p=2, dim=-1)


# ---------------------------------------------------------------------------
# TabularEncoder (MLP with residual blocks)
# ---------------------------------------------------------------------------


class TabularEncoder(nn.Module):
    """MLP encoder for tabular modalities (morphology or expression).

    Architecture: input projection → N residual blocks → projection → L2 norm.
    Uses LayerNorm for stability with variable batch sizes.

    Args:
        input_dim: Number of input features.
        cfg: Config with ``model.mlp.hidden_dim``, ``model.mlp.num_residual_blocks``,
            ``model.mlp.dropout``, and ``model.embedding_dim``.
    """

    def __init__(self, input_dim: int, cfg: DictConfig) -> None:
        super().__init__()

        hidden_dim: int = cfg.model.mlp.hidden_dim
        num_blocks: int = cfg.model.mlp.num_residual_blocks
        dropout: float = cfg.model.mlp.dropout
        embedding_dim: int = cfg.model.embedding_dim

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        # Projection head
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode tabular features.

        Args:
            x: Input features ``[B, input_dim]``.

        Returns:
            L2-normalized embeddings ``[B, embedding_dim]``.
        """
        h = self.input_layer(x)

        for block in self.res_blocks:
            h = block(h) + h  # residual connection

        h = self.projection(h)
        return f.normalize(h, p=2, dim=-1)
