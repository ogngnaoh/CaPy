"""CaPy training script.

Loads config, initializes data pipeline, model, and training loop.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml model.embedding_dim=512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable when running as a script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training.

    Returns:
        Parsed arguments with config path and optional overrides.
    """
    parser = argparse.ArgumentParser(
        description="Train the CaPy tri-modal contrastive model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional dotpath config overrides, e.g. model.embedding_dim=512.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import capy_collate_fn, load_split_dataset
    from src.data.featurize import featurize_dataset
    from src.models.capy import CaPyModel
    from src.training.scheduler import CosineAnnealingWithWarmup
    from src.training.trainer import Trainer
    from src.utils.config import load_config, seed_everything
    from src.utils.logging import get_logger, setup_wandb

    logger = get_logger(__name__)

    args = parse_args()

    # 1. Load config and seed
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    logger.info(
        "Config loaded from %s with %d overrides.",
        args.config,
        len(args.overrides),
    )

    # 2. Setup wandb
    setup_wandb(cfg)

    # 3. Device and GPU optimizations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for matmul and cuDNN.")
    logger.info("Using device: %s", device)

    # 4. Load processed data — read feature columns for dims
    processed_dir = Path(cfg.data.processed_dir)
    with open(processed_dir / "feature_columns.json") as f:
        col_info = json.load(f)
    morph_dim = len(col_info["morph_cols"])
    expr_dim = len(col_info["expr_cols"])
    logger.info("Feature dims: morph=%d, expr=%d", morph_dim, expr_dim)

    # Collect all unique SMILES across splits for featurization
    all_smiles = []
    all_ids = []
    for split in ["train", "val", "test"]:
        split_path = processed_dir / f"{split}.parquet"
        if split_path.exists():
            df = pd.read_parquet(split_path)
            all_smiles.extend(df["smiles"].tolist())
            all_ids.extend(df["compound_id"].tolist())

    # Deduplicate (same compound may appear across splits — shouldn't, but be safe)
    seen = set()
    unique_smiles, unique_ids = [], []
    for smi, cid in zip(all_smiles, all_ids):
        if cid not in seen:
            seen.add(cid)
            unique_smiles.append(smi)
            unique_ids.append(cid)

    # 5. Featurize all molecules once
    mol_graphs = featurize_dataset(unique_smiles, unique_ids)

    # 6. Build datasets
    train_ds = load_split_dataset(processed_dir, "train", mol_graphs)
    val_ds = load_split_dataset(processed_dir, "val", mol_graphs)

    if len(train_ds) == 0:
        raise ValueError(
            "Training dataset is empty — cannot train. "
            "Check preprocessing logs for dropped compounds."
        )
    if len(val_ds) == 0:
        logger.warning("Validation dataset is empty — early stopping will be disabled.")

    # 7. Build DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=capy_collate_fn,
        drop_last=True,  # contrastive learning needs consistent batch sizes
        num_workers=2,
        persistent_workers=True,
    )
    val_loader = None
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=capy_collate_fn,
            drop_last=False,
            num_workers=2,
            persistent_workers=True,
        )

    logger.info(
        "DataLoaders: train=%d batches, val=%s batches",
        len(train_loader),
        len(val_loader) if val_loader else 0,
    )

    # 8. Create model
    model = CaPyModel(cfg, morph_dim, expr_dim)

    # 9. Create optimizer with separate LR for GIN
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

    # 10. Create scheduler
    scheduler = CosineAnnealingWithWarmup(
        optimizer, cfg.training.warmup_epochs, cfg.training.epochs
    )

    # 11. Train
    trainer = Trainer(
        cfg, model, train_loader, val_loader, optimizer, scheduler, device
    )
    best_metrics = trainer.fit()

    # 12. Log final results
    logger.info("Training finished. Best metrics:")
    for k, v in sorted(best_metrics.items()):
        logger.info("  %s: %.4f", k, v)


if __name__ == "__main__":
    main()
