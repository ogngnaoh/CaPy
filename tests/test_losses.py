"""Tests for CaPy loss functions (src/models/losses.py).

Covers symmetric InfoNCE (NT-Xent) with learnable temperature.
"""

from __future__ import annotations

import math

from tests.conftest import requires_torch


@requires_torch
class TestInfoNCE:
    """Test suite for the symmetric InfoNCE loss."""

    def test_perfect_matching_loss_value(self) -> None:
        """Identity embeddings (perfect alignment) give expected loss."""
        import torch

        from src.models.losses import info_nce

        n = 8
        # When z_a == z_b (identity), the diagonal of z_a @ z_b.T / τ has
        # value 1/τ and off-diagonal values are cosine similarities of
        # random orthogonal-ish vectors — but for *identity matrix* inputs
        # we know exactly: diag = 1/τ, off-diag = 0.
        z = torch.eye(n)
        temperature = 1.0

        loss = info_nce(z, z, temperature)

        # Expected: -log(e^1 / (e^1 + (n-1)*e^0)) = -log(e / (e + n - 1))
        expected = -math.log(math.e / (math.e + n - 1))
        assert abs(loss.item() - expected) < 1e-5

    def test_random_higher_than_perfect(self) -> None:
        """Random (unaligned) pairs should have higher loss than aligned."""
        import torch

        from src.models.losses import info_nce

        n_samples, dim = 32, 256
        torch.manual_seed(42)
        z_a = torch.nn.functional.normalize(torch.randn(n_samples, dim), dim=-1)
        z_b = torch.nn.functional.normalize(torch.randn(n_samples, dim), dim=-1)
        z_aligned = z_a.clone()

        loss_random = info_nce(z_a, z_b, 0.07)
        loss_aligned = info_nce(z_a, z_aligned, 0.07)

        assert loss_random.item() > loss_aligned.item()

    def test_symmetry(self) -> None:
        """info_nce(a, b, τ) == info_nce(b, a, τ)."""
        import torch

        from src.models.losses import info_nce

        torch.manual_seed(0)
        z_a = torch.nn.functional.normalize(torch.randn(16, 64), dim=-1)
        z_b = torch.nn.functional.normalize(torch.randn(16, 64), dim=-1)

        loss_ab = info_nce(z_a, z_b, 0.1)
        loss_ba = info_nce(z_b, z_a, 0.1)

        assert abs(loss_ab.item() - loss_ba.item()) < 1e-6

    def test_batch_size_one(self) -> None:
        """Batch size 1 should not crash and should return a valid scalar."""
        import torch

        from src.models.losses import info_nce

        z_a = torch.nn.functional.normalize(torch.randn(1, 32), dim=-1)
        z_b = torch.nn.functional.normalize(torch.randn(1, 32), dim=-1)

        loss = info_nce(z_a, z_b, 0.07)

        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_gradient_flows_through_temperature(self) -> None:
        """Gradients should flow back to the learnable log_temperature."""
        import torch

        from src.models.losses import info_nce

        torch.manual_seed(1)
        z_a = torch.nn.functional.normalize(torch.randn(8, 64), dim=-1)
        z_b = torch.nn.functional.normalize(torch.randn(8, 64), dim=-1)

        log_temp = torch.tensor(math.log(1.0 / 0.07), requires_grad=True)
        temperature = log_temp.exp()

        loss = info_nce(z_a, z_b, temperature)
        loss.backward()

        assert log_temp.grad is not None
        assert log_temp.grad.item() != 0.0

    def test_lower_temperature_sharper(self) -> None:
        """Lower temperature should give higher loss for random pairs.

        Lower τ sharpens the softmax, making the model more "confident"
        in its predictions. For random (non-matching) pairs this means
        more confident *wrong* answers → higher cross-entropy loss.
        """
        import torch

        from src.models.losses import info_nce

        torch.manual_seed(7)
        n_samples, dim = 32, 128
        z_a = torch.nn.functional.normalize(torch.randn(n_samples, dim), dim=-1)
        z_b = torch.nn.functional.normalize(torch.randn(n_samples, dim), dim=-1)

        loss_low_t = info_nce(z_a, z_b, 0.01)
        loss_high_t = info_nce(z_a, z_b, 1.0)

        assert loss_low_t.item() > loss_high_t.item()
