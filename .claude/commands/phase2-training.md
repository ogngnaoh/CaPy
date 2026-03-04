Implement training infrastructure. Create these files:

1. src/training/trainer.py
   Full training loop. AdamW optimizer with separate param groups: GIN params at lr=1e-4, all other params at lr=1e-3. Weight decay 1e-4. Cosine annealing with 10-epoch linear warmup. Gradient clipping max_norm=1.0. wandb logging of: total loss, per-pair losses, temperature, learning rate, epoch time. Early stopping on validation mean Recall@10 with patience=20. Checkpoint best model to checkpoints/best_model.pt.

2. src/training/scheduler.py
   CosineAnnealingWithWarmup class wrapping torch.optim.lr_scheduler. Linear warmup from 0 to target LR over warmup_epochs, then cosine decay.

3. src/evaluation/retrieval.py
   Evaluate all 6 retrieval directions (mol->morph, morph->mol, mol->expr, expr->mol, morph->expr, expr->morph). For each: compute cosine similarity matrix, rank, compute Recall@1, Recall@5, Recall@10, MRR. Return dict of all metrics plus mean across directions.

4. scripts/train.py
   Main entrypoint. Argparse with --config flag. Load YAML via omegaconf. Seed all random sources. Init wandb. Build dataset, dataloaders, model, optimizer, scheduler. Call trainer. Save final results.

5. configs/default.yaml
   Update with any new parameters. All values must have inline comments.

After implementation, use the architect-explainer agent to explain: why different learning rates for GNN vs MLP, why cosine annealing with warmup, why batch size matters MORE in contrastive learning than supervised, what early stopping on retrieval means practically.
