# WHY THIS WORKS ──────────────────────────────────────────────────────
# CaPyModel is the "orchestrator" that owns the three encoders and the
# learnable temperature, then combines their outputs with InfoNCE.
#
# **Learnable temperature:**
# We store log(τ_init) as an unconstrained nn.Parameter and recover
# τ = exp(log_temperature).  Optimizing in log-space is standard
# (CLIP, SupCon) because (a) it keeps τ positive without constraints,
# and (b) multiplicative updates in τ-space become additive in log-τ
# space, matching how SGD works.  We clamp τ ∈ [0.01, 10] to prevent
# numerical blowup; .clamp() passes gradients when within range but
# zeroes them at the boundaries (hard clamp, not straight-through).
#
# **Loss aggregation:**
# L_total = λ₁·L(mol↔morph) + λ₂·L(mol↔expr) + λ₃·L(morph↔expr)
# Default λ = 1.0 for all three (equal weighting).  The scientific
# hypothesis is that tri-modal beats any bi-modal pair — ablation
# studies will set individual λ to 0 to test this.
#
# **Forward returns a dict:**
# Returning {z_mol, z_morph, z_expr, temperature} lets the trainer
# log temperature over time and use the embeddings for evaluation
# without re-running the encoders.
# ─────────────────────────────────────────────────────────────────────

"""CaPy tri-modal contrastive model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.models.encoders import MolecularEncoder, TabularEncoder
from src.models.losses import info_nce
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import Batch

logger = get_logger(__name__)


class CaPyModel(nn.Module):
    """Tri-modal contrastive model for drug discovery.

    Encodes molecular graphs, morphology features, and gene expression
    features into a shared 256-dim L2-normalized embedding space.

    Args:
        cfg: Full experiment config (reads ``model.*`` and ``training.*``).
        morph_dim: Number of morphology input features.
        expr_dim: Number of expression input features.
    """

    def __init__(self, cfg: DictConfig, morph_dim: int, expr_dim: int) -> None:
        super().__init__()

        self.mol_encoder = MolecularEncoder(cfg)
        self.morph_encoder = TabularEncoder(morph_dim, cfg)
        self.expr_encoder = TabularEncoder(expr_dim, cfg)

        # Learnable temperature: store log(τ) so exp(log_temp) = τ_init = 0.07
        temp_init: float = cfg.model.temperature_init
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temp_init)))

        # Loss weights
        self._lambda_mol_morph: float = cfg.training.lambda_mol_morph
        self._lambda_mol_expr: float = cfg.training.lambda_mol_expr
        self._lambda_morph_expr: float = cfg.training.lambda_morph_expr

        logger.info(
            "CaPyModel initialized: morph_dim=%d, expr_dim=%d, temp_init=%.4f",
            morph_dim,
            expr_dim,
            temp_init,
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature, clamped to [0.01, 10.0]."""
        return self.log_temperature.exp().clamp(0.01, 10.0)

    def forward(
        self,
        batch_graphs: Batch,
        morph: torch.Tensor,
        expr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode all three modalities.

        Args:
            batch_graphs: A PyG ``Batch`` of molecular graphs.
            morph: Morphology features ``[B, morph_dim]``.
            expr: Expression features ``[B, expr_dim]``.

        Returns:
            Dict with keys ``z_mol``, ``z_morph``, ``z_expr``, ``temperature``.
        """
        z_mol = self.mol_encoder(
            batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch
        )
        z_morph = self.morph_encoder(morph)
        z_expr = self.expr_encoder(expr)

        return {
            "z_mol": z_mol,
            "z_morph": z_morph,
            "z_expr": z_expr,
            "temperature": self.temperature,
        }

    def compute_loss(
        self,
        z_mol: torch.Tensor,
        z_morph: torch.Tensor,
        z_expr: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted tri-modal InfoNCE loss.

        Args:
            z_mol: Molecular embeddings ``[B, 256]``.
            z_morph: Morphology embeddings ``[B, 256]``.
            z_expr: Expression embeddings ``[B, 256]``.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict has per-pair
            losses and the current temperature for logging.
        """
        temp = self.temperature

        loss_mol_morph = info_nce(z_mol, z_morph, temp)
        loss_mol_expr = info_nce(z_mol, z_expr, temp)
        loss_morph_expr = info_nce(z_morph, z_expr, temp)

        total = (
            self._lambda_mol_morph * loss_mol_morph
            + self._lambda_mol_expr * loss_mol_expr
            + self._lambda_morph_expr * loss_morph_expr
        )

        loss_dict = {
            "loss_total": total.item(),
            "loss_mol_morph": loss_mol_morph.item(),
            "loss_mol_expr": loss_mol_expr.item(),
            "loss_morph_expr": loss_morph_expr.item(),
            "temperature": temp.item(),
        }

        return total, loss_dict
