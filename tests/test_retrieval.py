"""Tests for CaPy retrieval evaluation (src/evaluation/retrieval.py).

Covers cross-modal retrieval metrics: Recall@k and MRR across all
6 query directions.
"""

import pytest

from tests.conftest import requires_torch

# ---------------------------------------------------------------------------
# TestComputeRetrievalMetrics
# ---------------------------------------------------------------------------


@requires_torch
class TestComputeRetrievalMetrics:
    """Tests for the single-direction compute_retrieval_metrics function."""

    def test_perfect_matching(self) -> None:
        """Identity embeddings → R@1=1.0, MRR=1.0."""
        import torch

        from src.evaluation.retrieval import compute_retrieval_metrics

        z = torch.eye(10)
        metrics = compute_retrieval_metrics(z, z)
        assert metrics["R@1"] == pytest.approx(1.0)
        assert metrics["MRR"] == pytest.approx(1.0)

    def test_random_lower_recall(self) -> None:
        """Random embeddings should have R@1 well below 1.0."""
        import torch

        from src.evaluation.retrieval import compute_retrieval_metrics

        torch.manual_seed(42)
        n = 100
        z_a = torch.randn(n, 64)
        z_b = torch.randn(n, 64)
        z_a = z_a / z_a.norm(dim=1, keepdim=True)
        z_b = z_b / z_b.norm(dim=1, keepdim=True)

        metrics = compute_retrieval_metrics(z_a, z_b)
        assert metrics["R@1"] < 0.5

    def test_recall_monotonic(self) -> None:
        """R@1 ≤ R@5 ≤ R@10 must hold for any embeddings."""
        import torch

        from src.evaluation.retrieval import compute_retrieval_metrics

        torch.manual_seed(123)
        n = 50
        z_a = torch.randn(n, 32)
        z_b = torch.randn(n, 32)
        z_a = z_a / z_a.norm(dim=1, keepdim=True)
        z_b = z_b / z_b.norm(dim=1, keepdim=True)

        metrics = compute_retrieval_metrics(z_a, z_b)
        assert metrics["R@1"] <= metrics["R@5"] <= metrics["R@10"]

    def test_mrr_computation(self) -> None:
        """Perfect match → MRR=1.0, verified via identity matrix."""
        import torch

        from src.evaluation.retrieval import compute_retrieval_metrics

        z = torch.eye(5)
        metrics = compute_retrieval_metrics(z, z)
        assert metrics["MRR"] == pytest.approx(1.0)

    def test_small_batch(self) -> None:
        """N=2 edge case: works and R@5=R@10=1.0 (only 2 candidates)."""
        import torch

        from src.evaluation.retrieval import compute_retrieval_metrics

        z = torch.eye(2)
        metrics = compute_retrieval_metrics(z, z, ks=[1, 5, 10])
        # With identity, R@1 should be 1.0 (perfect match)
        assert metrics["R@1"] == pytest.approx(1.0)
        # With only 2 items, R@5 and R@10 must be 1.0
        assert metrics["R@5"] == pytest.approx(1.0)
        assert metrics["R@10"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestEvaluateAllRetrieval
# ---------------------------------------------------------------------------


@requires_torch
class TestEvaluateAllRetrieval:
    """Tests for the all-directions evaluate_all_retrieval function."""

    def test_all_directions_present(self) -> None:
        """All 6 directions × 4 metrics + 4 means + 8 align/uniform = 36 keys."""
        import torch

        from src.evaluation.retrieval import evaluate_all_retrieval

        n = 10
        z = torch.eye(n)
        metrics = evaluate_all_retrieval(z, z, z)
        # 6 directions × 4 metrics + 4 means + 3 align + 3 uniform + 2 agg = 36
        assert len(metrics) == 36

        # Check all direction keys exist
        directions = [
            "mol->morph",
            "morph->mol",
            "mol->expr",
            "expr->mol",
            "morph->expr",
            "expr->morph",
        ]
        for d in directions:
            for k in ["R@1", "R@5", "R@10", "MRR"]:
                assert f"{d}/{k}" in metrics

        # Check mean keys
        for k in ["mean_R@1", "mean_R@5", "mean_R@10", "mean_MRR"]:
            assert k in metrics

    def test_perfect_alignment(self) -> None:
        """Identical embeddings across modalities → all means = 1.0."""
        import torch

        from src.evaluation.retrieval import evaluate_all_retrieval

        z = torch.eye(10)
        metrics = evaluate_all_retrieval(z, z, z)
        assert metrics["mean_R@1"] == pytest.approx(1.0)
        assert metrics["mean_R@5"] == pytest.approx(1.0)
        assert metrics["mean_R@10"] == pytest.approx(1.0)
        assert metrics["mean_MRR"] == pytest.approx(1.0)

    def test_mean_is_average(self) -> None:
        """mean_R@k equals the arithmetic mean of the 6 directions."""
        import torch

        from src.evaluation.retrieval import evaluate_all_retrieval

        torch.manual_seed(99)
        n = 20
        z_mol = torch.randn(n, 16)
        z_morph = torch.randn(n, 16)
        z_expr = torch.randn(n, 16)
        z_mol = z_mol / z_mol.norm(dim=1, keepdim=True)
        z_morph = z_morph / z_morph.norm(dim=1, keepdim=True)
        z_expr = z_expr / z_expr.norm(dim=1, keepdim=True)

        metrics = evaluate_all_retrieval(z_mol, z_morph, z_expr)

        directions = [
            "mol->morph",
            "morph->mol",
            "mol->expr",
            "expr->mol",
            "morph->expr",
            "expr->morph",
        ]
        for k in [1, 5, 10]:
            expected_mean = sum(metrics[f"{d}/R@{k}"] for d in directions) / 6
            assert metrics[f"mean_R@{k}"] == pytest.approx(expected_mean)

        expected_mrr = sum(metrics[f"{d}/MRR"] for d in directions) / 6
        assert metrics["mean_MRR"] == pytest.approx(expected_mrr)


# ---------------------------------------------------------------------------
# TestAlignmentUniformity
# ---------------------------------------------------------------------------


@requires_torch
class TestAlignmentUniformity:
    """Tests for alignment and uniformity diagnostics."""

    def test_perfect_alignment_is_zero(self) -> None:
        """Identical embeddings → alignment = 0."""
        import torch

        from src.evaluation.retrieval import compute_alignment

        z = torch.randn(20, 32)
        z = z / z.norm(dim=1, keepdim=True)
        assert compute_alignment(z, z) == pytest.approx(0.0, abs=1e-6)

    def test_uniformity_collapsed_near_zero(self) -> None:
        """All-same embeddings (collapsed) → uniformity ≈ 0."""
        import torch

        from src.evaluation.retrieval import compute_uniformity

        z = torch.ones(20, 32)
        z = z / z.norm(dim=1, keepdim=True)
        # All identical → pairwise distances = 0 → exp(0) = 1 → log(1) = 0
        assert compute_uniformity(z) == pytest.approx(0.0, abs=1e-6)

    def test_uniformity_spread_is_negative(self) -> None:
        """Well-spread embeddings → uniformity < 0."""
        import torch

        from src.evaluation.retrieval import compute_uniformity

        torch.manual_seed(42)
        z = torch.randn(50, 64)
        z = z / z.norm(dim=1, keepdim=True)
        assert compute_uniformity(z) < 0.0

    def test_keys_in_evaluate_all_retrieval(self) -> None:
        """Alignment/uniformity keys appear in full evaluation output."""
        import torch

        from src.evaluation.retrieval import evaluate_all_retrieval

        torch.manual_seed(7)
        n = 15
        z_mol = torch.randn(n, 32)
        z_morph = torch.randn(n, 32)
        z_expr = torch.randn(n, 32)
        for z in [z_mol, z_morph, z_expr]:
            z.div_(z.norm(dim=1, keepdim=True))

        metrics = evaluate_all_retrieval(z_mol, z_morph, z_expr)

        # 3 alignment keys
        for pair in ["mol_morph", "mol_expr", "morph_expr"]:
            assert f"align_{pair}" in metrics
        # 3 uniformity keys
        for mod in ["mol", "morph", "expr"]:
            assert f"uniform_{mod}" in metrics
        # 2 aggregate keys
        assert "mean_alignment" in metrics
        assert "mean_uniformity" in metrics
