# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CaPy** (Contrastive Alignment of Phenotypic Yields) is a tri-modal contrastive learning framework for drug discovery. It learns a shared 256-dim embedding space across three biological modalities:
1. **Molecular structure** — SMILES → GNN (Graph Isomorphism Network)
2. **Cell morphology** — ~1,000 CellProfiler features → MLP
3. **Gene expression** — 978 L1000 landmark genes → MLP

Training objective: three pairwise symmetric InfoNCE losses (mol↔morph, mol↔expr, morph↔expr) with learnable temperature. Dataset: Rosetta CDRP-bio (~2,084 matched compounds, U2OS cells).

The full specification is in `capy-prd-fsd.md`.

## Build & Development Commands

```bash
# Install (editable)
pip install -e ".[dev]"

# Format & lint
black --line-length 88 .
ruff check .

# Tests
pytest tests/                        # all tests
pytest tests/test_models.py          # single test file
pytest tests/test_models.py -k "test_forward_pass"  # single test
pytest --cov=src --cov-report=term   # with coverage (target: ≥80%)

# Training
python scripts/train.py --config configs/default.yaml

# Evaluation
python scripts/evaluate.py --checkpoint <path> --config configs/default.yaml

# Docker
docker build -t capy .
docker run --gpus all capy
```

## Architecture

### Data Pipeline (`src/data/`)
- `download.py` — fetches Rosetta CDRP-bio profiles from S3
- `preprocess.py` — QC (remove NaNs, zero-variance, DMSO controls), RobustScaler normalization, Bemis-Murcko scaffold-based train/val/test split
- `featurize.py` — SMILES → PyG `Data` objects (atom/bond features → molecular graph)
- `dataset.py` — `CaPyDataset` returning aligned (graph, morph_vector, expr_vector) triplets

### Model (`src/models/`)
- `encoders.py` — three encoders, each outputting L2-normalized 256-dim vectors:
  - `MolecularEncoder`: 5-layer GIN (hidden_dim=300) with global mean pooling → projection head
  - `MorphologyEncoder`: MLP with LayerNorm, residual connections, dropout
  - `ExpressionEncoder`: same architecture as MorphologyEncoder (different input_dim)
- `capy.py` — combines encoders, computes `L_total = λ₁·L(mol↔morph) + λ₂·L(mol↔expr) + λ₃·L(morph↔expr)`
- `losses.py` — symmetric InfoNCE (NT-Xent) with learnable temperature (init 0.07)

### Training (`src/training/`)
- `trainer.py` — training loop with wandb logging, early stopping (patience=20), checkpointing
- `scheduler.py` — cosine annealing with 10-epoch warmup
- Key hyperparameters: AdamW, LR 1e-3 (MLPs) / 1e-4 (GIN), batch 128, 200 epochs, gradient clip 1.0

### Evaluation (`src/evaluation/`)
- `retrieval.py` — cross-modal retrieval across all 6 directions (Recall@1/5/10, MRR)
- `clustering.py` — zero-shot MOA classification (AMI, ARI, silhouette, k-NN accuracy)
- `interpretability.py` — feature importance, gene-morphology mapping, GSEA

### Config
YAML configs in `configs/` loaded via `omegaconf` or `pydantic`. All experiments tracked with wandb.

## Code Standards

- Python 3.10+
- Formatter: `black` (line length 88)
- Linter: `ruff`
- Type hints required on all public functions
- Google-style docstrings on all classes and public functions
- All random sources must be seeded for reproducibility

## Key Dependencies

- `torch`, `torch-geometric` — deep learning and GNNs
- `rdkit` — SMILES parsing and molecular graph construction
- `scanpy`, `gseapy` — biology data utilities and gene set enrichment
- `wandb` — experiment tracking
- `omegaconf` — config management

## Important Domain Context

- Data split must be by **Bemis-Murcko scaffold** (not random) to prevent molecular structure leakage
- All embeddings are **L2-normalized** before computing InfoNCE loss
- Expression data is already z-scored; morphology data needs RobustScaler normalization
- The core scientific claim to validate: tri-modal > any bi-modal pair on retrieval metrics

## Critical Conventions

- ALL public functions: Google-style docstrings with type hints
- NEVER use raw print() — use the logger from src/utils/logging.py
- Every module must have a corresponding test file
- Embedding dimension is 256 everywhere — constant, not magic number
- Use cfg (omegaconf DictConfig) for all hyperparameters, never hardcode
- InfoNCE loss is SYMMETRIC (average both directions) — verify every time
- Molecular graphs: always validate node features are [num_atoms, feat_dim]
- All random seeds (torch, numpy, random, PyG) must be set via utils/config.py

## What NOT To Do

- Do NOT use raw images — we use pre-extracted CellProfiler features only
- Do NOT commit data/ or checkpoints/ (gitignored)
- Do NOT use Transformers/attention for tabular encoders (MLPs are correct)
- Do NOT hardcode paths — use configs or Path objects

## Teaching Mode

When implementing new components, ALWAYS:
1. Write the code
2. Write the test
3. Explain the key design decision in a WHY THIS WORKS comment block
   at the top of the file, written for someone learning ML/PyTorch

## Current Phase

Phase 1 — Foundation (Data Pipeline + Model Architecture)
