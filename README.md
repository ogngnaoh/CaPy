# CaPy: Contrastive Alignment of Phenotypic Yields

Tri-modal contrastive learning framework for drug discovery.

CaPy learns a shared 256-dimensional embedding space across three biological modalities:
1. **Molecular structure** (SMILES → GIN)
2. **Cell morphology** (CellProfiler features → MLP)
3. **Gene expression** (L1000 landmark genes → MLP)

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Train
python scripts/train.py --config configs/default.yaml

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --config configs/default.yaml
```

## License

MIT
