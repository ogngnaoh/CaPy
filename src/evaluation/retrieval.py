# WHY THIS WORKS ──────────────────────────────────────────────────────
# Cross-modal retrieval is the primary evaluation for contrastive models.
# Given N aligned pairs, we compute an N×N similarity matrix and check
# where the true match (the diagonal) ranks among all candidates.
#
# **Rank computation:**  `ranks = (sim > sim.diag().unsqueeze(1)).sum(1)`
# counts how many items score *strictly higher* than the true match.
# This gives 0-indexed ranks (rank 0 = top-1 retrieval success).
# Using strict inequality handles ties correctly: if the true match
# ties with k other items, it gets rank k (conservative), not rank 0.
#
# **Recall@k** = fraction of queries whose true match is in the top-k.
# **MRR** (Mean Reciprocal Rank) = average of 1/(rank+1), rewarding
# higher placements more than lower ones.
#
# We evaluate all 6 directions (mol→morph, morph→mol, etc.) because
# retrieval is asymmetric: z_a @ z_b.T ≠ z_b @ z_a.T in terms of
# per-query rankings.  The mean across directions gives one number
# for early stopping.
# ─────────────────────────────────────────────────────────────────────

"""Cross-modal retrieval evaluation for CaPy embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

_torch = None


def _ensure_imports() -> None:
    global _torch  # noqa: PLW0603
    if _torch is None:
        import torch

        _torch = torch


# ---------------------------------------------------------------------------
# Core retrieval metrics
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    ks: list[int] | None = None,
) -> dict[str, float]:
    """Compute retrieval metrics for a single query direction.

    Assumes ``z_a[i]`` and ``z_b[i]`` are aligned positive pairs
    (same compound, different modalities).  Embeddings should be
    L2-normalized so that cosine similarity = dot product.

    Args:
        z_a: Query embeddings ``[N, D]``.
        z_b: Gallery embeddings ``[N, D]``.
        ks: List of k values for Recall@k (default ``[1, 5, 10]``).

    Returns:
        Dict with keys ``"R@1"``, ``"R@5"``, ``"R@10"``, ``"MRR"``.
    """
    _ensure_imports()
    if ks is None:
        ks = [1, 5, 10]

    if z_a.shape[0] == 0:
        return {f"R@{k}": 0.0 for k in ks} | {"MRR": 0.0}

    # N×N cosine similarity (dot product for L2-normalized vectors)
    sim = z_a @ z_b.T  # [N, N]

    # Rank of the true match (diagonal) — 0-indexed
    diag_scores = sim.diag().unsqueeze(1)  # [N, 1]
    ranks = (sim > diag_scores).sum(dim=1)  # [N]

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"R@{k}"] = (ranks < k).float().mean().item()

    metrics["MRR"] = (1.0 / (ranks.float() + 1.0)).mean().item()

    return metrics


# ---------------------------------------------------------------------------
# Alignment & uniformity (Wang & Isola 2020)
# ---------------------------------------------------------------------------


def compute_alignment(
    z_a: torch.Tensor, z_b: torch.Tensor, alpha: float = 2.0
) -> float:
    """Mean pairwise distance between aligned positive pairs.

    Lower = better (positive pairs are close together).

    Args:
        z_a: L2-normalized embeddings ``[N, D]``.
        z_b: L2-normalized embeddings ``[N, D]`` (aligned with z_a).
        alpha: Exponent for the distance (default 2 = squared L2).

    Returns:
        Alignment score (float).
    """
    _ensure_imports()
    return (z_a - z_b).norm(dim=1).pow(alpha).mean().item()


def compute_uniformity(z: torch.Tensor, t: float = 2.0) -> float:
    """Log-mean-exp of pairwise Gaussian kernel on embeddings.

    Lower = more uniform (spread on hypersphere). Near 0 = collapsed.

    Args:
        z: L2-normalized embeddings ``[N, D]``.
        t: Temperature for the Gaussian kernel (default 2).

    Returns:
        Uniformity score (float).
    """
    _ensure_imports()
    sq_pdist = _torch.pdist(z, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log().item()


# ---------------------------------------------------------------------------
# All-direction evaluation
# ---------------------------------------------------------------------------

_DIRECTIONS = [
    ("mol->morph", "z_mol", "z_morph"),
    ("morph->mol", "z_morph", "z_mol"),
    ("mol->expr", "z_mol", "z_expr"),
    ("expr->mol", "z_expr", "z_mol"),
    ("morph->expr", "z_morph", "z_expr"),
    ("expr->morph", "z_expr", "z_morph"),
]


def evaluate_all_retrieval(
    z_mol: torch.Tensor,
    z_morph: torch.Tensor,
    z_expr: torch.Tensor,
    ks: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate retrieval across all 6 cross-modal directions.

    Args:
        z_mol: Molecular embeddings ``[N, 256]``.
        z_morph: Morphology embeddings ``[N, 256]``.
        z_expr: Expression embeddings ``[N, 256]``.
        ks: List of k values for Recall@k (default ``[1, 5, 10]``).

    Returns:
        Dict with per-direction metrics (e.g. ``"mol->morph/R@1"``)
        and mean metrics (``"mean_R@1"``, ``"mean_R@5"``,
        ``"mean_R@10"``, ``"mean_MRR"``).
    """
    _ensure_imports()
    if ks is None:
        ks = [1, 5, 10]

    embeddings = {"z_mol": z_mol, "z_morph": z_morph, "z_expr": z_expr}
    all_metrics: dict[str, float] = {}

    with _torch.no_grad():
        for direction_name, key_a, key_b in _DIRECTIONS:
            dir_metrics = compute_retrieval_metrics(
                embeddings[key_a], embeddings[key_b], ks=ks
            )
            for metric_name, value in dir_metrics.items():
                all_metrics[f"{direction_name}/{metric_name}"] = value

    # Compute means across 6 directions
    metric_names = [f"R@{k}" for k in ks] + ["MRR"]
    for metric_name in metric_names:
        values = [all_metrics[f"{d[0]}/{metric_name}"] for d in _DIRECTIONS]
        all_metrics[f"mean_{metric_name}"] = sum(values) / len(values)

    # Alignment & uniformity diagnostics
    _pairs = [
        ("mol_morph", z_mol, z_morph),
        ("mol_expr", z_mol, z_expr),
        ("morph_expr", z_morph, z_expr),
    ]
    for pair_name, za, zb in _pairs:
        all_metrics[f"align_{pair_name}"] = compute_alignment(za, zb)
    for mod_name, z in [("mol", z_mol), ("morph", z_morph), ("expr", z_expr)]:
        all_metrics[f"uniform_{mod_name}"] = compute_uniformity(z)
    # Aggregate alignment/uniformity
    all_metrics["mean_alignment"] = sum(
        all_metrics[f"align_{p}"] for p, _, _ in _pairs
    ) / len(_pairs)
    all_metrics["mean_uniformity"] = (
        sum(all_metrics[f"uniform_{m}"] for m in ["mol", "morph", "expr"]) / 3.0
    )

    logger.info(
        "Retrieval: mean_R@1=%.4f, mean_R@5=%.4f, mean_R@10=%.4f, mean_MRR=%.4f",
        all_metrics["mean_R@1"],
        all_metrics["mean_R@5"],
        all_metrics["mean_R@10"],
        all_metrics["mean_MRR"],
    )
    logger.info(
        "Diagnostics: mean_alignment=%.4f, mean_uniformity=%.4f",
        all_metrics["mean_alignment"],
        all_metrics["mean_uniformity"],
    )

    return all_metrics
