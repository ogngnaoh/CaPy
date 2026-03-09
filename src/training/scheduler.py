# WHY THIS WORKS ──────────────────────────────────────────────────────
# Cosine annealing with linear warmup is the standard LR schedule for
# contrastive learning (SimCLR, CLIP, MoCo v3 all use it).
#
# **Warmup (first 10 epochs):** LR ramps linearly from 0 to base_lr.
# This prevents early gradient explosions: randomly initialized
# encoders produce near-random embeddings, so InfoNCE gradients are
# large and noisy.  Starting with a tiny LR lets BatchNorm stats
# stabilize and the temperature parameter find a reasonable range.
#
# **Cosine decay:** After warmup, LR follows 0.5*(1+cos(π*t)) where
# t ∈ [0, 1] maps the remaining epochs.  This decays smoothly to ~0,
# spending more time at moderate LRs than linear decay (which rushes
# through the sweet spot).  The final near-zero LR lets the model
# settle into a good basin.
#
# **Per-epoch stepping:** We step once per epoch, not per batch.
# With 200 epochs this gives a smooth curve.  Per-batch stepping would
# work but adds complexity for no benefit at this scale.
#
# Implementation via LambdaLR: we pass a single lambda that returns
# the multiplier for each epoch.  PyTorch applies it to each param
# group's base_lr independently, so GIN (1e-4) and MLPs (1e-3) each
# get correctly scaled learning rates.
# ─────────────────────────────────────────────────────────────────────

"""Cosine annealing with linear warmup scheduler for CaPy training."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CosineAnnealingWithWarmup(LambdaLR):
    """Cosine annealing LR schedule with linear warmup.

    During the warmup phase (``epoch < warmup_epochs``), the learning rate
    increases linearly from 0 to ``base_lr``.  After warmup, the LR
    follows a cosine decay to approximately 0.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of linear warmup epochs.
        total_epochs: Total training epochs (warmup + cosine decay).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda)

        logger.info(
            "CosineAnnealingWithWarmup: warmup=%d, total=%d epochs",
            warmup_epochs,
            total_epochs,
        )
