# WHY THIS WORKS ──────────────────────────────────────────────────────
# Reproducibility in ML requires controlling ALL sources of randomness.
# PyTorch maintains separate RNG states for CPU and each CUDA device.
# NumPy and Python's `random` module each have their own RNG.
# Even Python dict/set iteration order depends on PYTHONHASHSEED.
#
# OmegaConf gives us two key capabilities for experiment management:
# 1. Config merging — layer sweep.yaml on top of default.yaml
# 2. CLI overrides — `model.embedding_dim=512` without editing YAML
# This means one base config + small delta files for each experiment.
# ─────────────────────────────────────────────────────────────────────

"""Configuration loading and reproducibility utilities for CaPy."""

import os
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> DictConfig:
    """Load a YAML config file and apply optional CLI-style overrides.

    Args:
        path: Path to the YAML configuration file.
        overrides: Optional list of dotpath overrides,
            e.g. ``["model.embedding_dim=512", "training.lr_mlp=5e-4"]``.

    Returns:
        Merged configuration as an OmegaConf DictConfig.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg: DictConfig = OmegaConf.load(path)  # type: ignore[assignment]

    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, override_cfg)  # type: ignore[assignment]

    return cfg


def seed_everything(seed: int) -> None:
    """Seed all random number generators for reproducibility.

    Seeds Python's ``random``, NumPy, PyTorch (CPU + CUDA), and sets
    ``PYTHONHASHSEED``.  Also configures cuDNN for deterministic behavior.

    Note:
        PyTorch Geometric reuses ``torch.manual_seed``, so no separate
        PyG seeding is required.

    Args:
        seed: The random seed to use across all RNGs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
