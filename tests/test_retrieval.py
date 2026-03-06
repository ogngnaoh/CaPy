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
        """All 6 directions × 4 metrics + 4 means = 28 keys."""
        import torch

        from src.evaluation.retrieval import evaluate_all_retrieval

        n = 10
        z = torch.eye(n)
        metrics = evaluate_all_retrieval(z, z, z)
        # 6 directions × 4 metrics + 4 means = 28
        assert len(metrics) == 28

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
