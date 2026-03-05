"""Tests for CaPy model encoders (src/models/encoders.py, src/models/capy.py).

Covers MolecularEncoder, TabularEncoder, and the combined CaPyModel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from omegaconf import OmegaConf

from tests.conftest import requires_ml, requires_pyg, requires_torch

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import Batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_gin_cfg() -> DictConfig:
    """Create a minimal config with small GIN for fast tests."""
    return OmegaConf.create(
        {
            "model": {
                "embedding_dim": 256,
                "gin": {"num_layers": 3, "hidden_dim": 64},
                "mlp": {"hidden_dim": 512, "num_residual_blocks": 2, "dropout": 0.1},
                "temperature_init": 0.07,
            },
            "training": {
                "lambda_mol_morph": 1.0,
                "lambda_mol_expr": 1.0,
                "lambda_morph_expr": 1.0,
            },
        }
    )


def _make_batch_graphs(smiles_list: list[str]) -> Batch:
    """Convert a list of SMILES into a PyG Batch."""
    from torch_geometric.data import Batch

    from src.data.featurize import smiles_to_graph

    graphs = [smiles_to_graph(s) for s in smiles_list]
    graphs = [g for g in graphs if g is not None]
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# MolecularEncoder tests
# ---------------------------------------------------------------------------


@requires_pyg
class TestMolecularEncoder:
    """Tests for the GIN-based molecular encoder."""

    def test_output_shape(self) -> None:
        """Output should be [B, 256]."""
        from src.models.encoders import MolecularEncoder

        cfg = _small_gin_cfg()
        encoder = MolecularEncoder(cfg)
        batch = _make_batch_graphs(["CCO", "c1ccccc1", "CC(=O)O"])

        out = encoder(batch.x, batch.edge_index, batch.batch)

        assert out.shape == (3, 256)

    def test_l2_normalized(self) -> None:
        """Output embeddings should have unit L2 norm."""
        import torch

        from src.models.encoders import MolecularEncoder

        cfg = _small_gin_cfg()
        encoder = MolecularEncoder(cfg)
        batch = _make_batch_graphs(["CCO", "c1ccccc1"])

        out = encoder(batch.x, batch.edge_index, batch.batch)
        norms = torch.norm(out, p=2, dim=-1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_gradient_flow(self) -> None:
        """Gradients should flow from output back to atom encoder embeddings."""
        from src.models.encoders import MolecularEncoder

        cfg = _small_gin_cfg()
        encoder = MolecularEncoder(cfg)
        batch = _make_batch_graphs(["CCO"])

        out = encoder(batch.x, batch.edge_index, batch.batch)
        out.sum().backward()

        # Check that atom encoder embedding weights got gradients
        for emb in encoder.atom_encoder.embeddings:
            assert emb.weight.grad is not None

    def test_single_atom_molecule(self) -> None:
        """Methane (single atom, no bonds) should not crash."""
        from src.models.encoders import MolecularEncoder

        cfg = _small_gin_cfg()
        encoder = MolecularEncoder(cfg)
        encoder.eval()  # BN requires >1 sample in train mode
        batch = _make_batch_graphs(["C"])  # methane

        out = encoder(batch.x, batch.edge_index, batch.batch)

        assert out.shape == (1, 256)

    def test_batched_vs_individual_consistency(self) -> None:
        """Batched output should match individual forward passes (eval mode)."""
        import torch

        from src.models.encoders import MolecularEncoder

        cfg = _small_gin_cfg()
        encoder = MolecularEncoder(cfg)
        encoder.eval()  # disable BN running stats updates

        smiles = ["CCO", "c1ccccc1"]
        batch = _make_batch_graphs(smiles)
        batched_out = encoder(batch.x, batch.edge_index, batch.batch)

        individual_outs = []
        for smi in smiles:
            b = _make_batch_graphs([smi])
            individual_outs.append(encoder(b.x, b.edge_index, b.batch))
        individual_out = torch.cat(individual_outs, dim=0)

        assert torch.allclose(batched_out, individual_out, atol=1e-5)


# ---------------------------------------------------------------------------
# TabularEncoder tests
# ---------------------------------------------------------------------------


@requires_torch
class TestTabularEncoder:
    """Tests for the MLP-based tabular encoder."""

    def test_output_shape(self) -> None:
        """Output should be [B, 256] for any input_dim."""
        import torch

        from src.models.encoders import TabularEncoder

        cfg = _small_gin_cfg()
        encoder = TabularEncoder(input_dim=978, cfg=cfg)

        x = torch.randn(4, 978)
        out = encoder(x)

        assert out.shape == (4, 256)

    def test_l2_normalized(self) -> None:
        """Output embeddings should have unit L2 norm."""
        import torch

        from src.models.encoders import TabularEncoder

        cfg = _small_gin_cfg()
        encoder = TabularEncoder(input_dim=500, cfg=cfg)

        x = torch.randn(8, 500)
        out = encoder(x)
        norms = torch.norm(out, p=2, dim=-1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_gradient_flow(self) -> None:
        """Gradients should reach the input layer weights."""
        import torch

        from src.models.encoders import TabularEncoder

        cfg = _small_gin_cfg()
        encoder = TabularEncoder(input_dim=100, cfg=cfg)

        x = torch.randn(4, 100)
        out = encoder(x)
        out.sum().backward()

        # First linear layer should have gradients
        assert encoder.input_layer[0].weight.grad is not None

    def test_multiple_input_dims(self) -> None:
        """Encoder should work with different input dimensions."""
        import torch

        from src.models.encoders import TabularEncoder

        cfg = _small_gin_cfg()

        for dim in [100, 500, 978, 1200]:
            encoder = TabularEncoder(input_dim=dim, cfg=cfg)
            x = torch.randn(2, dim)
            out = encoder(x)
            assert out.shape == (2, 256), f"Failed for input_dim={dim}"


# ---------------------------------------------------------------------------
# CaPyModel tests
# ---------------------------------------------------------------------------


@requires_ml
class TestCaPyModel:
    """Tests for the combined tri-modal model."""

    def test_forward_shapes(self) -> None:
        """Forward should return correct embedding shapes."""
        import torch

        from src.models.capy import CaPyModel

        cfg = _small_gin_cfg()
        model = CaPyModel(cfg, morph_dim=500, expr_dim=978)

        batch = _make_batch_graphs(["CCO", "c1ccccc1", "CC(=O)O", "C"])
        morph = torch.randn(4, 500)
        expr = torch.randn(4, 978)

        out = model(batch, morph, expr)

        assert out["z_mol"].shape == (4, 256)
        assert out["z_morph"].shape == (4, 256)
        assert out["z_expr"].shape == (4, 256)
        assert out["temperature"].shape == ()

    def test_compute_loss_returns_valid_dict(self) -> None:
        """compute_loss should return total loss and a dict with all keys."""
        import torch

        from src.models.capy import CaPyModel

        cfg = _small_gin_cfg()
        model = CaPyModel(cfg, morph_dim=500, expr_dim=978)

        batch = _make_batch_graphs(["CCO", "c1ccccc1"])
        morph = torch.randn(2, 500)
        expr = torch.randn(2, 978)

        out = model(batch, morph, expr)
        total_loss, loss_dict = model.compute_loss(
            out["z_mol"], out["z_morph"], out["z_expr"]
        )

        # Total loss is a scalar tensor with gradients
        assert total_loss.shape == ()
        assert total_loss.requires_grad

        # Loss dict has all expected keys
        expected_keys = {
            "loss_total",
            "loss_mol_morph",
            "loss_mol_expr",
            "loss_morph_expr",
            "temperature",
        }
        assert set(loss_dict.keys()) == expected_keys

        # All values are finite
        for k, v in loss_dict.items():
            assert not (v != v), f"{k} is NaN"  # NaN != NaN

    def test_temperature_clamping(self) -> None:
        """Temperature should be clamped to [0.01, 10.0] at extremes."""
        import torch

        from src.models.capy import CaPyModel

        cfg = _small_gin_cfg()
        model = CaPyModel(cfg, morph_dim=100, expr_dim=100)

        # Force temperature very high
        with torch.no_grad():
            model.log_temperature.fill_(100.0)
        assert model.temperature.item() == pytest.approx(10.0)

        # Force temperature very low
        with torch.no_grad():
            model.log_temperature.fill_(-100.0)
        assert model.temperature.item() == pytest.approx(0.01)
