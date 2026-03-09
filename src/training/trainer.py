# WHY THIS WORKS ──────────────────────────────────────────────────────
# The Trainer orchestrates the training loop with three key design
# decisions for contrastive learning:
#
# 1. **model.eval() in validation** — GIN uses BatchNorm, which
#    computes batch statistics in train mode but running statistics in
#    eval mode.  Small val batches (or single-atom molecules like
#    methane) would produce unstable batch stats and crash.
#
# 2. **Collect all val embeddings, then compute retrieval** —
#    Retrieval requires ranking each query against ALL gallery items.
#    Per-batch retrieval would only rank within that batch (e.g., 128
#    candidates instead of the full val set), inflating metrics.
#
# 3. **Early stopping on mean_R@10** — We stop when the model stops
#    improving at retrieval, not when loss plateaus.  Loss can keep
#    decreasing while the model overfits to training negatives,
#    causing retrieval performance to degrade.
#
# 4. **Scheduler.step() per epoch, not per batch** — Cosine annealing
#    over 200 epochs gives a smooth LR curve.  The scheduler is
#    stepped after validation so that early stopping sees the model
#    trained at the current LR, not the next one.
# ─────────────────────────────────────────────────────────────────────

"""Training loop for CaPy tri-modal contrastive model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.evaluation.retrieval import evaluate_all_retrieval
from src.utils.logging import get_logger, log_metrics

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

    from src.models.capy import CaPyModel

logger = get_logger(__name__)


class Trainer:
    """Training loop with validation, early stopping, and checkpointing.

    Args:
        cfg: Full experiment configuration.
        model: CaPyModel instance.
        train_loader: DataLoader for training split.
        val_loader: DataLoader for validation split (None disables
            early stopping and validation metrics).
        optimizer: Configured optimizer with param groups.
        scheduler: LR scheduler (stepped once per epoch).
        device: Device to train on (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: CaPyModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: str | torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)

        # Read config
        self.epochs: int = cfg.training.epochs
        self.grad_clip: float = cfg.training.grad_clip
        self.log_every_n_steps: int = cfg.logging.log_every_n_steps
        self.checkpoint_dir = Path(cfg.logging.checkpoint_dir)
        self.early_stopping_patience: int = cfg.training.early_stopping_patience
        self.early_stopping_metric: str = cfg.training.early_stopping_metric
        self.retrieval_ks: list[int] = list(cfg.evaluation.retrieval_ks)

        # Mixed precision (bf16) — safe on H100 without GradScaler
        self.use_amp: bool = getattr(cfg.training, "use_amp", False)
        self.amp_dtype = torch.bfloat16 if self.use_amp else None

        # Tracking state
        self._best_metric: float = -math.inf
        self._patience_counter: int = 0
        self._global_step: int = 0
        self._best_metrics: dict[str, float] = {}

        if val_loader is None:
            logger.warning(
                "No val_loader provided — early stopping and validation disabled."
            )

    def fit(self, start_epoch: int = 0) -> dict[str, float]:
        """Run the full training loop.

        Args:
            start_epoch: Epoch to resume from (default 0). Set via
                the return value of ``load_checkpoint``.

        Returns:
            Dict of best validation metrics (empty if no val_loader).
        """
        logger.info(
            "Starting training: %d epochs, device=%s, amp=%s",
            self.epochs,
            self.device,
            self.amp_dtype if self.use_amp else "off",
        )

        for epoch in range(start_epoch, self.epochs):
            train_stats = self._train_one_epoch(epoch)

            val_metrics: dict[str, float] = {}
            if self.val_loader is not None:
                val_metrics = self._validate(epoch)

                if not val_metrics:
                    # Empty val loader — skip early stopping for this epoch
                    self.scheduler.step()
                    continue

                # Early stopping check
                current = val_metrics.get(self.early_stopping_metric, -math.inf)
                if current > self._best_metric:
                    self._best_metric = current
                    self._patience_counter = 0
                    self._best_metrics = val_metrics.copy()
                    self._save_checkpoint(epoch, val_metrics)
                    logger.info(
                        "Epoch %d: new best %s=%.4f — checkpoint saved.",
                        epoch,
                        self.early_stopping_metric,
                        current,
                    )
                else:
                    self._patience_counter += 1
                    if self._patience_counter >= self.early_stopping_patience:
                        logger.info(
                            "Early stopping at epoch %d (patience=%d exhausted).",
                            epoch,
                            self.early_stopping_patience,
                        )
                        break

            self.scheduler.step()

            # Enriched epoch summary
            epoch_summary: dict[str, float] = {"epoch": epoch}
            epoch_summary.update(train_stats)

            # Learning rates from optimizer param groups
            lrs = [pg["lr"] for pg in self.optimizer.param_groups]
            if len(lrs) >= 2:
                epoch_summary["lr_gin"] = lrs[0]
                epoch_summary["lr_mlp"] = lrs[1]

            if val_metrics:
                epoch_summary[self.early_stopping_metric] = val_metrics.get(
                    self.early_stopping_metric, 0.0
                )
                # Per-direction R@10
                for key, value in val_metrics.items():
                    if key.endswith("/R@10"):
                        epoch_summary[key] = value

            log_metrics(epoch_summary, step=epoch, prefix="epoch/")

        logger.info(
            "Training complete. Best %s=%.4f",
            self.early_stopping_metric,
            self._best_metric,
        )
        return self._best_metrics

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Dict with ``train_loss``, ``grad_norm``, per-pair losses,
            and ``temperature``.
        """
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0
        last_loss_dict: dict[str, float] = {}

        # Accumulate per-modality embedding statistics for collapse detection
        embed_norms: dict[str, list[float]] = {
            f"embed_{mod}_norm_mean": [] for mod in ["mol", "morph", "expr"]
        }
        embed_cos: dict[str, list[float]] = {
            f"embed_{mod}_cos_std": [] for mod in ["mol", "morph", "expr"]
        }

        for batch_graphs, morph, expr in self.train_loader:
            batch_graphs = batch_graphs.to(self.device)
            morph = morph.to(self.device)
            expr = expr.to(self.device)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(batch_graphs, morph, expr)
                loss, loss_dict = self.model.compute_loss(
                    outputs["z_mol"], outputs["z_morph"], outputs["z_expr"]
                )

            # Per-modality embedding diagnostics (detached, no grad impact)
            with torch.no_grad():
                for mod_name, z in [
                    ("mol", outputs["z_mol"]),
                    ("morph", outputs["z_morph"]),
                    ("expr", outputs["z_expr"]),
                ]:
                    embed_norms[f"embed_{mod_name}_norm_mean"].append(
                        z.norm(dim=-1).mean().item()
                    )
                    if z.size(0) > 1:
                        sim = z @ z.T
                        mask = torch.triu(
                            torch.ones_like(sim, dtype=torch.bool), diagonal=1
                        )
                        embed_cos[f"embed_{mod_name}_cos_std"].append(
                            sim[mask].std().item()
                        )
                    else:
                        embed_cos[f"embed_{mod_name}_cos_std"].append(0.0)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm.item()
            n_batches += 1
            self._global_step += 1
            last_loss_dict = loss_dict

            if self._global_step % self.log_every_n_steps == 0:
                step_metrics = {**loss_dict, "grad_norm": grad_norm.item()}
                log_metrics(step_metrics, step=self._global_step, prefix="train/")

        if n_batches == 0:
            logger.warning(
                "Epoch %d: train_loader yielded 0 batches — is the dataset empty?",
                epoch,
            )

        mean_loss = total_loss / max(1, n_batches)
        mean_grad_norm = total_grad_norm / max(1, n_batches)

        # Read temperature from the model's loss module
        temperature = float(
            getattr(self.model, "temperature", torch.tensor(0.07)).detach().cpu()
        )

        epoch_stats: dict[str, float] = {
            "train_loss": mean_loss,
            "grad_norm": mean_grad_norm,
            "temperature": temperature,
        }
        # Include per-pair losses from last batch
        for k, v in last_loss_dict.items():
            if k != "total_loss":
                epoch_stats[k] = v

        # Average embedding stats across batches
        for key, values in {**embed_norms, **embed_cos}.items():
            if values:
                epoch_stats[key] = sum(values) / len(values)

        return epoch_stats

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        """Run validation: compute loss and retrieval metrics.

        Collects all embeddings first, then computes retrieval over the
        full validation set (not per-batch).

        Args:
            epoch: Current epoch number.

        Returns:
            Dict of validation metrics including retrieval and loss.
        """
        self.model.eval()

        all_z_mol = []
        all_z_morph = []
        all_z_expr = []

        for batch_graphs, morph, expr in self.val_loader:
            batch_graphs = batch_graphs.to(self.device)
            morph = morph.to(self.device)
            expr = expr.to(self.device)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(batch_graphs, morph, expr)
            all_z_mol.append(outputs["z_mol"])
            all_z_morph.append(outputs["z_morph"])
            all_z_expr.append(outputs["z_expr"])

        if not all_z_mol:
            logger.warning("Validation loader yielded 0 batches — skipping metrics.")
            return {}

        z_mol = torch.cat(all_z_mol, dim=0).float()
        z_morph = torch.cat(all_z_morph, dim=0).float()
        z_expr = torch.cat(all_z_expr, dim=0).float()

        # Retrieval metrics
        retrieval_metrics = evaluate_all_retrieval(
            z_mol, z_morph, z_expr, ks=self.retrieval_ks
        )

        # Validation loss in fp32 for reliable early-stopping signal
        _, loss_dict = self.model.compute_loss(z_mol, z_morph, z_expr)

        val_loss = {f"val_{k}": v for k, v in loss_dict.items()}
        val_metrics = {**retrieval_metrics, **val_loss}
        log_metrics(val_metrics, step=epoch, prefix="val/")

        return val_metrics

    def _save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            metrics: Current validation metrics.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "best_model.pt"

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "best_metric": self._best_metric,
                "patience_counter": self._patience_counter,
                "global_step": self._global_step,
                "config": OmegaConf.to_container(self.cfg, resolve=True),
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint and restore training state.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The epoch number the checkpoint was saved at.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self._best_metric = checkpoint.get("best_metric", -math.inf)
        self._best_metrics = checkpoint.get("metrics", {})
        self._patience_counter = checkpoint.get("patience_counter", 0)
        self._global_step = checkpoint.get("global_step", 0)

        epoch = checkpoint["epoch"]
        logger.info(
            "Loaded checkpoint from epoch %d (best_metric=%.4f)",
            epoch,
            self._best_metric,
        )
        return epoch
