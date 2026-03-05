"""End-to-end integration test: synthetic data → model → loss → gradients.

Verifies the full pipeline from fake tri-modal inputs through the CaPy
model forward pass, loss computation, and backward pass.
"""

from __future__ import annotations

from tests.conftest import requires_pyg


@requires_pyg
class TestEndToEnd:
    """Integration tests for the complete forward-backward pipeline."""

    def test_synthetic_forward_backward(self) -> None:
        """Full pipeline: 32 synthetic samples → forward → loss → backward.

        Creates fake molecular graphs (5-10 atoms each), morphology vectors,
        and expression vectors. Runs CaPyModel forward, computes tri-modal
        InfoNCE loss, calls backward, and verifies every parameter received
        a gradient.
        """
        import torch
        from omegaconf import OmegaConf
        from torch_geometric.data import Batch, Data

        from src.data.featurize import ATOM_FEATURE_DIMS
        from src.models.capy import CaPyModel

        # --- Config ---
        morph_dim = 100
        expr_dim = 50
        n_samples = 32

        cfg = OmegaConf.create(
            {
                "model": {
                    "embedding_dim": 256,
                    "gin": {"num_layers": 3, "hidden_dim": 64},
                    "mlp": {
                        "hidden_dim": 128,
                        "num_residual_blocks": 2,
                        "dropout": 0.1,
                    },
                    "temperature_init": 0.07,
                },
                "training": {
                    "lambda_mol_morph": 1.0,
                    "lambda_mol_expr": 1.0,
                    "lambda_morph_expr": 1.0,
                },
            }
        )

        # --- Synthetic data ---
        torch.manual_seed(42)

        graphs = []
        for _ in range(n_samples):
            num_atoms = torch.randint(5, 11, (1,)).item()
            # Random integer atom features within valid vocab ranges
            x = torch.stack(
                [torch.randint(0, vs, (num_atoms,)) for vs in ATOM_FEATURE_DIMS],
                dim=1,
            )
            # Random edges (at least one edge per graph)
            num_edges = num_atoms * 2
            edge_index = torch.stack(
                [
                    torch.randint(0, num_atoms, (num_edges,)),
                    torch.randint(0, num_atoms, (num_edges,)),
                ]
            )
            graphs.append(Data(x=x, edge_index=edge_index))

        batch_graphs = Batch.from_data_list(graphs)
        morph = torch.randn(n_samples, morph_dim)
        expr = torch.randn(n_samples, expr_dim)

        # --- Forward ---
        model = CaPyModel(cfg, morph_dim=morph_dim, expr_dim=expr_dim)
        out = model(batch_graphs, morph, expr)

        assert out["z_mol"].shape == (n_samples, 256)
        assert out["z_morph"].shape == (n_samples, 256)
        assert out["z_expr"].shape == (n_samples, 256)

        # --- Loss ---
        total_loss, loss_dict = model.compute_loss(
            out["z_mol"], out["z_morph"], out["z_expr"]
        )

        assert total_loss.shape == ()
        assert torch.isfinite(total_loss), f"Loss is not finite: {total_loss.item()}"
        assert total_loss.requires_grad

        for key, val in loss_dict.items():
            assert val == val, f"{key} is NaN"  # NaN != NaN

        # --- Backward ---
        total_loss.backward()

        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert (
            params_without_grad == []
        ), f"Parameters missing gradients: {params_without_grad}"
