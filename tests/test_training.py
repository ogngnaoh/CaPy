"""Tests for CaPy training infrastructure.

Covers the CosineAnnealingWithWarmup scheduler and the Trainer class
using synthetic data on CPU.
"""

import pytest

from tests.conftest import requires_ml, requires_torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_setup(n_train: int = 32, n_val: int = 16):
    """Create a minimal training setup with synthetic data.

    Returns:
        Tuple of (cfg, model, train_loader, val_loader, optimizer, scheduler, device).
    """
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from torch_geometric.data import Data

    from src.data.dataset import CaPyDataset, capy_collate_fn
    from src.models.capy import CaPyModel
    from src.training.scheduler import CosineAnnealingWithWarmup

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "model": {
                "embedding_dim": 32,
                "gin": {"num_layers": 2, "hidden_dim": 32},
                "mlp": {"hidden_dim": 32, "num_residual_blocks": 1, "dropout": 0.0},
                "temperature_init": 0.07,
            },
            "training": {
                "epochs": 3,
                "batch_size": 16,
                "optimizer": "adamw",
                "lr_mlp": 1e-3,
                "lr_gin": 1e-4,
                "weight_decay": 1e-4,
                "grad_clip": 1.0,
                "lambda_mol_morph": 1.0,
                "lambda_mol_expr": 1.0,
                "lambda_morph_expr": 1.0,
                "scheduler": "cosine",
                "warmup_epochs": 1,
                "early_stopping_patience": 5,
                "early_stopping_metric": "mean_R@10",
            },
            "evaluation": {"retrieval_ks": [1, 5, 10]},
            "logging": {
                "use_wandb": False,
                "log_every_n_steps": 5,
                "checkpoint_dir": "checkpoints_test",
            },
        }
    )

    morph_dim = 8
    expr_dim = 6
    device = "cpu"

    # Atom feature vocab sizes: [10, 4, 6, 5, 5, 3, 7, 2, 2]
    atom_dims = [10, 4, 6, 5, 5, 3, 7, 2, 2]
    # Bond feature vocab sizes: [4, 2, 2, 6]
    bond_dims = [4, 2, 2, 6]

    def _make_graph():
        n_atoms = torch.randint(3, 8, (1,)).item()
        # Integer atom features within vocab ranges
        x = torch.stack(
            [torch.randint(0, d, (n_atoms,)) for d in atom_dims], dim=1
        ).long()
        n_edges = n_atoms * 2
        edge_index = torch.randint(0, n_atoms, (2, n_edges))
        edge_attr = torch.stack(
            [torch.randint(0, d, (n_edges,)) for d in bond_dims], dim=1
        ).long()
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _make_dataset(n: int) -> CaPyDataset:
        graphs = [_make_graph() for _ in range(n)]
        morph = torch.randn(n, morph_dim)
        expr = torch.randn(n, expr_dim)
        ids = [f"cpd_{i}" for i in range(n)]
        return CaPyDataset(graphs, morph, expr, ids)

    train_ds = _make_dataset(n_train)
    val_ds = _make_dataset(n_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        collate_fn=capy_collate_fn,
        drop_last=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        collate_fn=capy_collate_fn,
        drop_last=False,
    )

    model = CaPyModel(cfg, morph_dim, expr_dim)

    # Two param groups: GIN vs everything else
    gin_params = list(model.mol_encoder.parameters())
    gin_param_ids = {id(p) for p in gin_params}
    other_params = [p for p in model.parameters() if id(p) not in gin_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": gin_params, "lr": cfg.training.lr_gin},
            {"params": other_params, "lr": cfg.training.lr_mlp},
        ],
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = CosineAnnealingWithWarmup(
        optimizer, cfg.training.warmup_epochs, cfg.training.epochs
    )

    return cfg, model, train_loader, val_loader, optimizer, scheduler, device


# ---------------------------------------------------------------------------
# TestCosineAnnealingWithWarmup
# ---------------------------------------------------------------------------


@requires_torch
class TestCosineAnnealingWithWarmup:
    """Tests for the CosineAnnealingWithWarmup scheduler."""

    def _make_optimizer(self, base_lr: float = 0.1):
        import torch

        param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.SGD([param], lr=base_lr)

    def test_warmup_linear(self) -> None:
        """LR at epoch i during warmup = base_lr × (i+1)/warmup."""
        from src.training.scheduler import CosineAnnealingWithWarmup

        base_lr = 0.1
        warmup_epochs = 5
        optimizer = self._make_optimizer(base_lr)
        scheduler = CosineAnnealingWithWarmup(optimizer, warmup_epochs, 20)

        # Epoch 0: LR = base_lr * 1/5 (non-zero from the start)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(
            base_lr / warmup_epochs, rel=1e-5
        )

        for epoch in range(1, warmup_epochs):
            scheduler.step()
            expected = base_lr * (epoch + 1) / warmup_epochs
            assert optimizer.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-5)

    def test_cosine_decay(self) -> None:
        """After warmup, LR starts at base_lr and decays to ~0."""
        from src.training.scheduler import CosineAnnealingWithWarmup

        base_lr = 0.1
        warmup = 5
        total = 25
        optimizer = self._make_optimizer(base_lr)
        scheduler = CosineAnnealingWithWarmup(optimizer, warmup, total)

        # Step through warmup
        for _ in range(warmup):
            scheduler.step()

        # At epoch=warmup, LR should be base_lr (warmup just completed)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-4)

        # Step through rest and check final LR ≈ 0
        for _ in range(warmup, total - 1):
            scheduler.step()

        assert optimizer.param_groups[0]["lr"] < 0.01 * base_lr

    def test_midpoint(self) -> None:
        """Midpoint of cosine phase → LR ≈ 0.5 × base_lr."""
        import math

        from src.training.scheduler import CosineAnnealingWithWarmup

        base_lr = 0.1
        warmup = 0
        total = 20
        optimizer = self._make_optimizer(base_lr)
        scheduler = CosineAnnealingWithWarmup(optimizer, warmup, total)

        # Step to midpoint
        midpoint = total // 2
        for _ in range(midpoint):
            scheduler.step()

        expected = 0.5 * base_lr * (1 + math.cos(math.pi * midpoint / total))
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-4)

    def test_multiple_param_groups(self) -> None:
        """Each param group is scaled independently."""
        import torch

        from src.training.scheduler import CosineAnnealingWithWarmup

        p1 = torch.nn.Parameter(torch.zeros(1))
        p2 = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD(
            [
                {"params": [p1], "lr": 0.1},
                {"params": [p2], "lr": 0.01},
            ]
        )
        scheduler = CosineAnnealingWithWarmup(
            optimizer,
            warmup_epochs=2,
            total_epochs=10,
        )

        # After 1 warmup step (epoch=1): LR = base_lr * (1+1)/2 = base_lr
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 2 / 2, rel=1e-5)
        assert optimizer.param_groups[1]["lr"] == pytest.approx(0.01 * 2 / 2, rel=1e-5)


# ---------------------------------------------------------------------------
# TestTrainer
# ---------------------------------------------------------------------------


@requires_ml
class TestTrainer:
    """Tests for the Trainer class using synthetic data."""

    def test_single_epoch_finite_loss(self) -> None:
        """_train_one_epoch returns a dict with finite train_loss and grad_norm."""
        import math

        from src.training.trainer import Trainer

        cfg, model, train_loader, val_loader, optimizer, scheduler, device = (
            _make_synthetic_setup()
        )
        trainer = Trainer(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
        )
        stats = trainer._train_one_epoch(0)
        assert isinstance(stats, dict)
        assert "train_loss" in stats
        assert "grad_norm" in stats
        assert "temperature" in stats
        assert math.isfinite(stats["train_loss"])
        assert math.isfinite(stats["grad_norm"])

    def test_fit_completes(self, tmp_path) -> None:
        """fit() returns a dict and completes without errors."""
        from omegaconf import OmegaConf

        from src.training.trainer import Trainer

        cfg, model, train_loader, val_loader, optimizer, scheduler, device = (
            _make_synthetic_setup()
        )
        # Use tmp_path for checkpoint dir
        cfg = OmegaConf.merge(
            cfg, {"logging": {"checkpoint_dir": str(tmp_path / "ckpts")}}
        )
        trainer = Trainer(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
        )
        result = trainer.fit()
        assert isinstance(result, dict)

    def test_checkpoint_save_load(self, tmp_path) -> None:
        """Save + load checkpoint, verify state_dict keys survive round-trip."""
        from omegaconf import OmegaConf

        from src.training.trainer import Trainer

        cfg, model, train_loader, val_loader, optimizer, scheduler, device = (
            _make_synthetic_setup()
        )
        ckpt_dir = tmp_path / "ckpts"
        cfg = OmegaConf.merge(cfg, {"logging": {"checkpoint_dir": str(ckpt_dir)}})

        trainer = Trainer(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
        )
        # Run one epoch and manually trigger checkpoint
        trainer._train_one_epoch(0)
        trainer._save_checkpoint(0, {"mean_R@10": 0.5})

        ckpt_path = ckpt_dir / "best_model.pt"
        assert ckpt_path.exists()

        # Load into a fresh trainer
        cfg2, model2, _, _, optimizer2, scheduler2, device2 = _make_synthetic_setup()
        cfg2 = OmegaConf.merge(cfg2, {"logging": {"checkpoint_dir": str(ckpt_dir)}})
        trainer2 = Trainer(
            cfg2, model2, train_loader, val_loader, optimizer2, scheduler2, device2
        )
        epoch = trainer2.load_checkpoint(ckpt_path)
        assert epoch == 0

    def test_no_val_skips_early_stopping(self) -> None:
        """val_loader=None → runs all epochs without error."""
        from src.training.trainer import Trainer

        cfg, model, train_loader, _, optimizer, scheduler, device = (
            _make_synthetic_setup()
        )
        trainer = Trainer(cfg, model, train_loader, None, optimizer, scheduler, device)
        result = trainer.fit()
        assert isinstance(result, dict)
        # With no val, best_metrics should be empty
        assert len(result) == 0
