# WHY THIS WORKS ──────────────────────────────────────────────────────
# We wrap Python's stdlib logging + wandb behind a thin API so that:
# 1. Every module uses `get_logger(__name__)` — standard Python pattern
# 2. `log_metrics()` writes to both console AND wandb in one call
# 3. `setup_wandb()` is gated on `cfg.logging.use_wandb` — training
#    works identically with or without wandb (offline-first design)
#
# The duplicate-handler guard (checking `logger.handlers`) prevents the
# common bug where importing a module twice in notebooks or tests adds
# a second StreamHandler, causing every message to print twice.
# ─────────────────────────────────────────────────────────────────────

"""Logging utilities for CaPy — console + optional wandb integration."""

import logging
import sys

from omegaconf import DictConfig


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a named logger with console output.

    Prevents duplicate handlers when called multiple times with the
    same name (e.g. in notebooks or test re-imports).

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_wandb(cfg: DictConfig) -> object | None:
    """Initialize a wandb run if enabled in config.

    Args:
        cfg: Full experiment config; reads ``cfg.logging.use_wandb``,
            ``cfg.logging.wandb_project``, and ``cfg.logging.wandb_entity``.

    Returns:
        The :class:`wandb.sdk.wandb_run.Run` object if wandb is enabled,
        otherwise ``None``.
    """
    if not cfg.logging.use_wandb:
        return None

    import wandb

    run = wandb.init(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.get("wandb_entity"),
        config=dict(cfg),
    )
    return run


def log_metrics(metrics: dict, step: int | None = None, prefix: str = "") -> None:
    """Log metrics to both the console logger and wandb (if active).

    Args:
        metrics: Dictionary of metric names to values.
        step: Optional global step for wandb x-axis.
        prefix: Optional prefix prepended to each metric key
            (e.g. ``"val/"``).
    """
    import wandb

    logger = get_logger("capy.metrics")

    prefixed = {f"{prefix}{k}": v for k, v in metrics.items()} if prefix else metrics

    # Console
    parts = [
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in prefixed.items()
    ]
    logger.info(" | ".join(parts))

    # wandb (no-op if no active run)
    if wandb.run is not None:
        wandb.log(prefixed, step=step)
