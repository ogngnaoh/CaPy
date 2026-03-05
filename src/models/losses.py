# WHY THIS WORKS ──────────────────────────────────────────────────────
# Symmetric InfoNCE (NT-Xent) treats each sample in a batch as having
# exactly one positive partner (the same compound's other modality) and
# N-1 negatives (all other compounds in the batch).
#
# The logits matrix z_a @ z_b.T / τ has shape [N, N].  The diagonal
# entries (i, i) are the positive-pair similarities.  F.cross_entropy
# with labels=arange(N) maximizes the diagonal while pushing off-
# diagonal entries down — this is equivalent to the log-softmax over
# each row, selecting the diagonal element.
#
# We symmetrize by averaging loss(z_a→z_b) and loss(z_b→z_a), because
# retrieving morphology from a molecule should be as good as the
# reverse.  PyTorch's F.cross_entropy handles the LogSumExp numerically
# (subtracting the row max internally), so we don't need manual tricks.
#
# The temperature τ is a *learnable scalar* (stored as log τ for
# unconstrained optimization).  Lower τ sharpens the softmax, making
# the model more confident — but too low causes gradient saturation.
# Clamping τ ∈ [0.01, 10] prevents numerical instability at extremes.
# ─────────────────────────────────────────────────────────────────────

"""Symmetric InfoNCE (NT-Xent) loss for contrastive learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy heavy imports
# ---------------------------------------------------------------------------

_torch = None
_F = None


def _ensure_imports() -> None:
    """Import torch and torch.nn.functional on first use."""
    global _torch, _F  # noqa: PLW0603
    if _torch is None:
        import torch

        _torch = torch
    if _F is None:
        import torch.nn.functional as f  # noqa: N812

        _F = f


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def info_nce(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: torch.Tensor | float,
) -> torch.Tensor:
    """Compute symmetric InfoNCE (NT-Xent) loss.

    Assumes that ``z_a[i]`` and ``z_b[i]`` form a positive pair (same
    compound, different modalities) and all other pairs are negatives.

    Args:
        z_a: L2-normalized embeddings from modality A, shape ``[N, D]``.
        z_b: L2-normalized embeddings from modality B, shape ``[N, D]``.
        temperature: Scalar temperature (float or 0-dim Tensor).  When
            passed as a ``nn.Parameter`` (via ``log_temp.exp()``),
            gradients flow back to the learnable temperature.

    Returns:
        Scalar loss (mean of both cross-entropy directions).
    """
    _ensure_imports()

    # Similarity matrix: [N, N]
    logits = z_a @ z_b.T / temperature

    # Positive pairs lie on the diagonal
    labels = _torch.arange(logits.size(0), device=logits.device)

    # Symmetrize: A→B and B→A
    loss = (_F.cross_entropy(logits, labels) + _F.cross_entropy(logits.T, labels)) / 2

    return loss
