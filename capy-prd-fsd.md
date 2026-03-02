# CaPy: Contrastive Alignment of Phenotypic Yields

### Tri-Modal Contrastive Learning for Drug Discovery

## Product Requirements Document & Functional Specification

**Version:** 1.1
**Author:** Hoang Ngo 
**Date:** February 2026
**Status:** Implementation-Ready

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Project Goals & Success Criteria](#3-project-goals--success-criteria)
4. [Architecture Overview](#4-architecture-overview)
5. [Data Specification](#5-data-specification)
6. [Model Specification](#6-model-specification)
7. [Training Specification](#7-training-specification)
8. [Evaluation Specification](#8-evaluation-specification)
9. [Interpretability & Biology Analysis](#9-interpretability--biology-analysis)
10. [Repository Structure & Engineering Standards](#10-repository-structure--engineering-standards)
11. [Implementation Timeline](#11-implementation-timeline)
12. [Glossary for Non-Biologists](#12-glossary-for-non-biologists)
13. [Key References](#13-key-references)

---

## 1. Executive Summary

CaPy is a contrastive learning framework that learns a shared embedding space across three biological data modalities:

1. **Molecular structure** (what the drug looks like as a molecule)
2. **Cell morphology** (what the cell looks like after treatment — Cell Painting)
3. **Gene expression** (what genes the cell turns on/off after treatment — L1000)

**The core hypothesis:** A joint embedding space capturing all three modalities will outperform any bi-modal pair for drug similarity search and mechanism-of-action (MOA) classification, because the modalities contain complementary — not redundant — information about how drugs affect cells.

**Why this matters to insitro:** insitro's entire platform is built on the thesis that multi-modal data integration yields stronger biological signal. CaPy directly tests this thesis on public data, using the same types of data insitro generates internally (Cell Painting, transcriptomics, molecular libraries).

**Why this stands out for an intern application:** No published work aligns molecular graphs + Cell Painting + gene expression in a single contrastive framework. CellCLIP (NeurIPS 2025) aligns two modalities; CaPy extends to three. The Rosetta dataset providing paired measurements is relatively obscure. The interpretability analysis demonstrates biological curiosity from a CS/math background.

---

## 2. Problem Statement & Motivation

### 2.1 The Drug Discovery Bottleneck

When pharmaceutical companies test thousands of molecules against cells, they generate mountains of data in different formats. A chemist looks at molecular structures. A microscopist looks at cell images. A genomicist looks at gene expression profiles. Each person sees a partial picture. The fundamental challenge is: **how do you combine these views to understand what a drug actually does?**

### 2.2 Current Approaches and Their Limitations

| Approach | What it does | Limitation |
|----------|-------------|------------|
| CellProfiler features + PCA | Extracts handcrafted morphological features, reduces dimensions | No learning — misses complex nonlinear relationships |
| CellCLIP (2025) | Contrastive alignment of Cell Painting images ↔ text labels | Only 2 modalities; uses text, not molecular structure |
| CLOOME (2023) | Contrastive alignment of molecular fingerprints ↔ Cell Painting | Fixed fingerprints (not learned); no gene expression |
| Rosetta MLP baselines (2022) | MLP regression: morphology → expression | Single direction; no shared space; no molecular input |
| scGPT / Geneformer | Foundation models for single-cell transcriptomics | Shown to underperform simple baselines on perturbation prediction |

### 2.3 The Gap CaPy Fills

No existing method:
- Uses **learned** molecular representations (GNNs) rather than fixed fingerprints
- Aligns **all three** modalities in a single shared space
- Enables **any-to-any** cross-modal retrieval (6 directions)
- Quantifies the **marginal value** of each modality via systematic ablation

---

## 3. Project Goals & Success Criteria

### 3.1 Primary Goals

| # | Goal | Measurable Outcome |
|---|------|-------------------|
| G1 | Build a working tri-modal contrastive learning system | Model trains, converges, produces embeddings for all 3 modalities |
| G2 | Demonstrate tri-modal > bi-modal | At least 2 of 3 bi-modal pairs are outperformed by tri-modal on retrieval metrics |
| G3 | Interpretability analysis | Identify top 20 morphological features most aligned with specific gene sets |
| G4 | Production-quality repository | Passes linting, has tests, Docker support, reproducible results |

### 3.2 Stretch Goals

| # | Goal | Measurable Outcome |
|---|------|-------------------|
| S1 | Zero-shot MOA classification | Tri-modal embeddings cluster by MOA better than any single modality (measured by AMI/ARI) |
| S2 | Cross-perturbation-type retrieval | Given a CRISPR knockout, retrieve the chemically similar compound (uses LINCS-ORF + CDRP) |
| S3 | Interactive visualization | Streamlit app showing the embedding space with compound metadata |

### 3.3 Non-Goals (Explicit Scope Limits)

- We do NOT work with raw microscopy images (we use pre-extracted CellProfiler features)
- We do NOT train single-cell foundation models
- We do NOT generate new molecules (that's a stretch project beyond scope)
- We do NOT claim clinical relevance — this is a methods paper, not a drug candidate

---

## 4. Architecture Overview

### 4.1 High-Level Architecture Diagram

```
                        ┌─────────────────────┐
     SMILES string      │   Molecular Encoder  │
    "CC(=O)Oc1cc..."  ──│   (GIN / AttentiveFP)│──── z_mol  (256-dim)
                        │   via PyG            │         │
                        └─────────────────────┘         │
                                                         │
                        ┌─────────────────────┐         │    ┌──────────────┐
     CellPainting       │  Morphology Encoder  │         ├───►│              │
     1000-dim vector  ──│  (MLP + LayerNorm)   │──── z_morph  │  Shared      │
                        │                      │  (256-dim)├──►│  Embedding   │
                        └─────────────────────┘         │    │  Space       │
                                                         │    │  (256-dim)   │
                        ┌─────────────────────┐         │    │              │
     L1000 expression   │  Expression Encoder  │         │    └──────────────┘
     978-dim vector   ──│  (MLP + LayerNorm)   │──── z_expr   
                        │                      │  (256-dim)
                        └─────────────────────┘

    Training Objective: 3 pairwise InfoNCE losses
    ─────────────────────────────────────────────
    L_total = λ₁·L(mol↔morph) + λ₂·L(mol↔expr) + λ₃·L(morph↔expr)
```

### 4.2 Design Decisions & Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Molecular encoder | GIN (Graph Isomorphism Network) | Simplest provably powerful GNN; great PyTorch Geometric support; AttentiveFP as stretch |
| Tabular encoders | MLP with LayerNorm + residual connections | Morphology and expression are already feature vectors — transformers are overkill for ~1000 dims |
| Embedding dimension | 256 | Standard for contrastive learning; large enough for expressiveness, small enough for 2K samples |
| Contrastive loss | InfoNCE (NT-Xent) | Standard, well-understood, works well with small batches |
| Temperature | Learnable (initialized at 0.07) | Following CLIP — learned temperature consistently outperforms fixed |
| Projection head | 2-layer MLP (hidden 512 → output 256) | Following SimCLR finding that projection heads improve representations |

---

## 5. Data Specification

### 5.1 Primary Dataset: Rosetta CDRP-bio

**Source:** Broad Institute Cell Painting Gallery
**Publication:** Haghighi et al., Nature Methods 2022
**URL:** `s3://cellpainting-gallery/cpg0003-rosetta/`
**Total size:** ~8.5 GB (but we only need the pre-computed profiles, ~200 MB)

| Property | Value |
|----------|-------|
| Cell line | U2OS (human bone osteosarcoma) |
| Perturbation type | Chemical (bioactive compounds) |
| Number of compounds | ~2,084 |
| Morphology features | ~1,000 CellProfiler features (after QC) |
| Expression features | 978 L1000 landmark genes |
| Pairing | Matched — same compounds measured in both assays |

### 5.2 What Each Column Means (Non-Biologist Guide)

**Morphology features (Cell Painting):**
Each feature name follows the pattern `{Compartment}_{MeasurementType}_{Channel}_{Detail}`. For example:
- `Cells_AreaShape_Area` → physical size of the cell body
- `Nuclei_Texture_Contrast_DNA_3_01` → how "rough" the nuclear DNA texture looks at scale 3
- `Cytoplasm_Intensity_MeanIntensity_AGP` → average brightness of the actin/golgi/plasma membrane stain

You don't need to understand every feature — the model learns which ones matter. But for interpretability, knowing the naming convention helps you say things like "the model found that cytoplasm texture features are most predictive of cytoskeleton-related genes."

**Expression features (L1000):**
Each column is a gene name (e.g., `TP53`, `BRCA1`, `EGFR`). The value is a z-score indicating how much that gene's activity changed compared to untreated cells. Positive = upregulated (turned on more), negative = downregulated. The 978 "landmark" genes were chosen because they can computationally predict the remaining ~11,000 genes.

**Molecular structure (SMILES):**
SMILES is a line notation for molecular structure. For example:
- `O=C(O)c1ccccc1O` = salicylic acid (aspirin precursor)
- `CC(=O)Oc1ccccc1C(=O)O` = aspirin

We convert SMILES → molecular graph using RDKit:
- Nodes = atoms (features: element type, degree, charge, aromaticity, etc.)
- Edges = bonds (features: bond type — single, double, triple, aromatic)

### 5.3 Data Pipeline

```
Step 1: Download Rosetta profiles
        ↓
Step 2: Match compounds across morphology & expression tables (by compound ID)
        ↓
Step 3: Retrieve SMILES for each compound (from JUMP metadata or PubChem)
        ↓
Step 4: Quality control
        - Remove compounds with missing data in any modality
        - Remove morphology features with >50% NaN or zero variance
        - Remove compounds that are DMSO controls (negative controls)
        ↓
Step 5: Normalization
        - Morphology: RobustScaler (median/IQR) per feature, then clip to [-5, 5]
        - Expression: Already z-scored in L1000; verify and re-normalize if needed
        - Molecules: No normalization needed (graph structure is discrete)
        ↓
Step 6: Train/Val/Test split
        - CRITICAL: Split by Bemis-Murcko scaffold (molecular backbone)
        - This prevents data leakage from structurally similar molecules
        - 70% train / 15% val / 15% test
        - Stratify by MOA annotation where available
        ↓
Step 7: Create PyTorch datasets and dataloaders
```

### 5.4 Data Files We Need to Download

```bash
# Morphology profiles (CellProfiler features, aggregated per compound)
aws s3 cp s3://cellpainting-gallery/cpg0003-rosetta/broad/workspace/profiles/CDRP-bio/ ./data/raw/morphology/ --recursive --no-sign-request

# Expression profiles (L1000 gene expression, aggregated per compound)  
aws s3 cp s3://cellpainting-gallery/cpg0003-rosetta/broad/workspace/profiles/CDRP-bio/ ./data/raw/expression/ --recursive --no-sign-request

# Metadata (compound IDs, SMILES, MOA annotations)
# Source: JUMP consortium metadata or Broad's compound annotations
```

**Fallback if S3 access is complex:** The Rosetta data is also available via the Cell Painting Gallery GitHub repo with direct download links. We will script both approaches.

### 5.5 Supplementary Data Sources

| Source | Purpose | How We Use It |
|--------|---------|---------------|
| PubChem | SMILES for compounds missing from metadata | API lookup by compound name/InChI |
| ChEMBL | MOA annotations for evaluation | Join by compound ID for zero-shot MOA eval |
| Broad Repurposing Hub | Additional MOA labels | Backup annotation source |

---

## 6. Model Specification

### 6.1 Molecular Encoder (GIN)

```python
# Pseudocode — actual implementation in src/models/encoders.py

class MolecularEncoder(nn.Module):
    """
    Graph Isomorphism Network for molecular SMILES.
    
    Input: PyG Data object (graph with node/edge features)
    Output: 256-dim embedding vector
    
    Architecture:
        5 GIN convolution layers (hidden_dim=300)
        → Global mean pooling over nodes
        → Projection MLP (300 → 512 → 256)
    """
    
    # Node features (per atom): 9-dimensional
    #   - Atomic number (one-hot, 28 elements)
    #   - Degree (0-5)
    #   - Formal charge (-2 to +2)
    #   - Num H (0-4)
    #   - Is aromatic (binary)
    #   - Is in ring (binary)
    # Total after encoding: ~78 dim (one-hot expanded)
    
    # Edge features (per bond): 4-dimensional
    #   - Bond type (single/double/triple/aromatic)
    #   - Is conjugated (binary)
    #   - Is in ring (binary)
    #   - Stereo configuration (6 types)
    
    def forward(self, data):
        # data.x = node features [num_atoms, node_feat_dim]
        # data.edge_index = adjacency [2, num_edges]
        # data.edge_attr = edge features [num_edges, edge_feat_dim]
        # data.batch = batch assignment [num_atoms]
        
        x = self.atom_encoder(data.x)          # Linear → hidden_dim
        for conv in self.gin_layers:
            x = conv(x, data.edge_index)        # GIN convolution
            x = F.relu(x)
            x = self.batch_norms[i](x)
        
        x = global_mean_pool(x, data.batch)     # [batch_size, hidden_dim]
        x = self.projection(x)                  # MLP → 256-dim
        x = F.normalize(x, dim=-1)              # L2 normalize
        return x
```

**Why GIN specifically:** The Graph Isomorphism Network is provably as powerful as the Weisfeiler-Lehman graph isomorphism test — meaning it can distinguish any two molecules that differ in structure. It's also simple (just MLPs applied to neighborhoods) and has great PyG support. More complex alternatives (SchNet for 3D, DimeNet for angles) require 3D conformer generation, which adds complexity without guaranteed benefit for this dataset size.

### 6.2 Morphology Encoder (MLP)

```python
class MorphologyEncoder(nn.Module):
    """
    MLP encoder for CellProfiler morphological feature vectors.
    
    Input: [batch_size, ~1000] float tensor (CellProfiler features)
    Output: [batch_size, 256] L2-normalized embedding
    
    Architecture:
        Linear(1000 → 512) → LayerNorm → ReLU → Dropout(0.1)
        Linear(512 → 512) → LayerNorm → ReLU → Dropout(0.1)  [+ residual]
        Linear(512 → 512) → LayerNorm → ReLU → Dropout(0.1)  [+ residual]
        Projection: Linear(512 → 256) → L2 Normalize
    """
```

**Why MLP and not a transformer:** The input is a fixed-length feature vector with no sequential or positional structure. Self-attention over 1,000 features would be computationally expensive and semantically meaningless (there's no reason feature #437 should "attend to" feature #812). MLPs with residual connections are the right tool here.

### 6.3 Expression Encoder (MLP)

```python
class ExpressionEncoder(nn.Module):
    """
    Identical architecture to MorphologyEncoder but for L1000 data.
    
    Input: [batch_size, 978] float tensor (L1000 landmark gene z-scores)
    Output: [batch_size, 256] L2-normalized embedding
    
    Same architecture as MorphologyEncoder (input dim differs: 978 vs ~1000).
    Shared architecture class with configurable input_dim.
    """
```

### 6.4 Full CaPy Model

```python
class CaPy(nn.Module):
    """
    Tri-modal contrastive learning model.
    
    Components:
        - mol_encoder: MolecularEncoder (GIN)
        - morph_encoder: MorphologyEncoder (MLP)
        - expr_encoder: ExpressionEncoder (MLP)
        - log_temperature: learnable scalar (initialized to log(1/0.07))
    
    Forward pass:
        Input: batch of (molecular_graph, morph_vector, expr_vector)
        Output: (z_mol, z_morph, z_expr, temperature)
        All z vectors are L2-normalized 256-dim embeddings.
    """
    
    def compute_loss(self, z_mol, z_morph, z_expr):
        temp = self.log_temperature.exp()
        
        # 3 pairwise InfoNCE losses
        loss_mol_morph = info_nce(z_mol, z_morph, temp)
        loss_mol_expr  = info_nce(z_mol, z_expr, temp)
        loss_morph_expr = info_nce(z_morph, z_expr, temp)
        
        # Weighted combination (learnable or fixed)
        total = (self.lambda1 * loss_mol_morph + 
                 self.lambda2 * loss_mol_expr + 
                 self.lambda3 * loss_morph_expr)
        
        return total, {
            'loss_mol_morph': loss_mol_morph.item(),
            'loss_mol_expr': loss_mol_expr.item(),
            'loss_morph_expr': loss_morph_expr.item(),
        }
```

---

## 7. Training Specification

### 7.1 InfoNCE Loss (Detailed)

This is the core learning signal. Here's exactly how it works:

```python
def info_nce(z_a, z_b, temperature):
    """
    Symmetric InfoNCE (NT-Xent) loss.
    
    For a batch of N paired samples:
    - z_a[i] and z_b[i] are the POSITIVE pair (same compound, different modalities)
    - z_a[i] and z_b[j] where j≠i are NEGATIVE pairs (different compounds)
    
    The loss pushes positive pairs together and negative pairs apart
    in the embedding space.
    
    Args:
        z_a: [N, D] L2-normalized embeddings from modality A
        z_b: [N, D] L2-normalized embeddings from modality B
        temperature: scalar controlling sharpness of similarity distribution
    
    Returns:
        Scalar loss value
    """
    # Similarity matrix: [N, N] where sim[i,j] = cosine_sim(z_a[i], z_b[j])
    # Since z_a and z_b are L2-normalized, this is just matrix multiplication
    logits = (z_a @ z_b.T) / temperature   # [N, N]
    
    # Labels: the diagonal entries are the correct matches
    labels = torch.arange(N, device=z_a.device)  # [0, 1, 2, ..., N-1]
    
    # Cross-entropy loss in both directions
    loss_a_to_b = F.cross_entropy(logits, labels)      # "given z_a[i], find z_b[i]"
    loss_b_to_a = F.cross_entropy(logits.T, labels)    # "given z_b[i], find z_a[i]"
    
    return (loss_a_to_b + loss_b_to_a) / 2
```

**Intuition:** Think of it as a matching game. You have N compounds in a batch. For each compound's molecular embedding, the model must pick the correct morphological embedding out of N choices (and vice versa). Temperature controls how "picky" the matching is — lower temperature means the model must match more precisely.

### 7.2 Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | Weight decay helps regularize with small datasets |
| Learning rate | 1e-3 (encoders), 1e-4 (GIN) | Lower LR for GNN which is more sensitive |
| Weight decay | 1e-4 | |
| Batch size | 128 | ~2K compounds means ~16 batches/epoch — larger batch = more negatives = better contrastive learning |
| Epochs | 200 | With early stopping (patience=20 on val retrieval) |
| LR schedule | Cosine annealing with 10 epoch warmup | |
| Temperature init | 0.07 (log scale: ~2.66) | Following CLIP |
| Lambda weights | [1.0, 1.0, 1.0] initially | Ablate later; potentially learn them |
| Embedding dim | 256 | |
| Dropout | 0.1 | In MLP encoders only |
| Gradient clipping | max_norm=1.0 | Stability for GNN training |

### 7.3 Training Loop Pseudocode

```python
for epoch in range(max_epochs):
    model.train()
    for batch in train_loader:
        mol_graphs, morph_vecs, expr_vecs = batch
        
        # Forward pass — get embeddings
        z_mol = model.mol_encoder(mol_graphs)
        z_morph = model.morph_encoder(morph_vecs)
        z_expr = model.expr_encoder(expr_vecs)
        
        # Compute tri-modal loss
        loss, loss_dict = model.compute_loss(z_mol, z_morph, z_expr)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Log individual losses for monitoring
        wandb.log(loss_dict)
    
    # Validation: compute retrieval metrics on val set
    val_metrics = evaluate_retrieval(model, val_loader)
    
    # Early stopping on mean retrieval Recall@10
    if val_metrics['mean_R@10'] > best_R10:
        best_R10 = val_metrics['mean_R@10']
        save_checkpoint(model, 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 20:
            break
```

### 7.4 Computational Requirements

| Resource | Estimate |
|----------|----------|
| GPU memory | < 4 GB (small dataset, no images) |
| Training time | ~10-30 minutes per full training run on a T4/V100 |
| Total GPU hours for all experiments | ~10-20 hours |
| Storage | ~2 GB (data + checkpoints + logs) |
| **Estimated cloud cost** | **~$5-15 on Colab Pro or Lambda** |

This is an extremely lightweight project computationally. The bottleneck is your iteration speed, not GPU time.

---

## 8. Evaluation Specification

### 8.1 Cross-Modal Retrieval (Primary Metric)

For each of 6 retrieval directions, compute:

```
Given: Compound A's embedding in modality X
Task: Retrieve Compound A's embedding in modality Y from all N test compounds

Metrics:
  - Recall@1:  Is the correct match the #1 result?
  - Recall@5:  Is the correct match in the top 5?
  - Recall@10: Is the correct match in the top 10?
  - MRR:       Mean Reciprocal Rank (1/rank of correct match)
```

The 6 directions: mol→morph, morph→mol, mol→expr, expr→mol, morph→expr, expr→morph.

**Baselines to compare against:**
1. **Random:** Recall@1 = 1/N ≈ 0.003 for N=300 test compounds
2. **Bi-modal (mol↔morph only):** Train CaPy with only the mol↔morph loss
3. **Bi-modal (mol↔expr only):** Train CaPy with only the mol↔expr loss
4. **Bi-modal (morph↔expr only):** Train CaPy with only the morph↔expr loss
5. **Rosetta MLP baseline:** Reproduce the Rosetta paper's MLP regression (R² metric)

### 8.2 Zero-Shot MOA Classification (Secondary Metric)

```
Step 1: Embed all test compounds using the tri-modal model
Step 2: For each modality, compute pairwise cosine similarity
Step 3: Apply k-means or agglomerative clustering
Step 4: Compare clusters to ground-truth MOA labels using:
        - Adjusted Mutual Information (AMI)
        - Adjusted Rand Index (ARI)
        - Silhouette score within MOA groups

Also compute: k-NN MOA accuracy
  - For each compound, find its k nearest neighbors in embedding space
  - Classify by majority vote of neighbors' MOA labels
  - Report accuracy for k=5,10,20
```

**MOA labels come from:** Broad Repurposing Hub annotations and/or ChEMBL target annotations. Not all compounds have MOA labels — report the number of annotated compounds used.

### 8.3 Ablation Study (Core Scientific Contribution)

This is what separates "I built a CLIP variant" from "I answered a scientific question."

| Experiment | What it tests |
|------------|---------------|
| Tri-modal (all 3 losses) | Full model |
| Bi-modal: mol↔morph | Does expression add value beyond morphology? |
| Bi-modal: mol↔expr | Does morphology add value beyond expression? |
| Bi-modal: morph↔expr | Does molecular structure add value? |
| Single encoder + supervised MOA | How far does supervised learning get? |
| Random embeddings | Sanity check |
| Vary λ weights | Which modality pair contributes most? |
| Vary embedding dim (64, 128, 256, 512) | How much capacity is needed? |

**Present results as a table:**

```
| Model Variant       | mol→morph R@10 | morph→expr R@10 | ... | MOA AMI |
|---------------------|---------------|-----------------|-----|---------|
| CaPy (full)      |     0.XX      |      0.XX       | ... |  0.XX   |
| Bi-modal mol↔morph  |     0.XX      |       —         | ... |  0.XX   |
| Bi-modal morph↔expr |      —        |      0.XX       | ... |  0.XX   |
| ...                 |               |                 |     |         |
```

### 8.4 Visualization

1. **t-SNE / UMAP** of the joint embedding space, colored by MOA class
2. **Retrieval heatmap:** 6×1 grid showing the similarity matrix for each retrieval direction
3. **Training curves:** Loss per modality pair over epochs (shows which pairs converge first)
4. **Lambda sensitivity:** Bar chart of R@10 vs. lambda weight ratios

---

## 9. Interpretability & Biology Analysis

This section is what transforms the project from a ML exercise into something that demonstrates biological curiosity. You don't need deep bio knowledge — the analysis itself teaches you.

### 9.1 Feature Importance via Embedding Alignment

```python
# After training, for each morphology feature:
# 1. Compute correlation between that feature and the morph→expr similarity
# 2. Features with high correlation are "expression-predictive morphology features"

# Concretely:
for feature_idx in range(n_morph_features):
    feature_values = morph_data[:, feature_idx]  # [N]
    expr_retrieval_scores = cosine_sim(z_morph, z_expr).diagonal()  # [N]
    correlation = pearsonr(feature_values, expr_retrieval_scores)
    
# Report: Top 20 morphology features most correlated with expression alignment
# Interpret: Are they from nucleus? cytoplasm? specific stains?
```

### 9.2 Gene-Morphology Mapping

```python
# Which genes are best predicted from morphology?
# For each gene:
# 1. Get the gene's expression values across all compounds
# 2. Get the morph→expr retrieval rank for each compound
# 3. Genes where high expression correlates with high retrieval = "morphology-visible genes"

# Then: Run Gene Set Enrichment Analysis (GSEA) on the top 100 genes
# using MSigDB Hallmark gene sets (available via Python's gseapy package)
# This answers: "The morphology captures information about [cell cycle / DNA damage / metabolism / ...]"
```

### 9.3 Molecular Substructure Analysis

```python
# Which parts of molecules drive morphological similarity?
# Use GNN attention/gradient-based attribution:

# 1. For a given molecule, compute grad of z_mol w.r.t. node features
# 2. Atoms with highest gradient magnitude are most "important" for the embedding
# 3. Map these to common substructures using RDKit's GetSubstructMatches

# Alternatively (simpler):
# 1. Group compounds by Bemis-Murcko scaffold
# 2. Compute mean embedding per scaffold
# 3. Find scaffolds whose embeddings are closest to specific MOA cluster centers
# This answers: "Compounds with [benzimidazole / pyridine / ...] scaffolds cluster with [tubulin / kinase / ...] inhibitors"
```

### 9.4 Modality Disagreement Analysis (Most Novel)

```python
# Find compounds where morphology and expression DISAGREE:
# 1. For each compound, rank all other compounds by morph similarity
# 2. Separately rank by expression similarity
# 3. Find compound pairs that are similar in morphology but dissimilar in expression (or vice versa)

# These are the most biologically interesting cases:
# - Morph-similar but expr-different: drugs that LOOK the same under microscope but have different molecular mechanisms
# - Expr-similar but morph-different: drugs that change the same genes but affect different cellular structures

# Present as a table of top 10 "disagreement" compound pairs with their known targets
```

---

## 10. Repository Structure & Engineering Standards

### 10.1 Directory Structure

```
capy/
├── README.md                    # Project overview, results summary, how to reproduce
├── LICENSE                      # MIT
├── pyproject.toml               # Dependencies and project metadata
├── Dockerfile                   # Reproducible environment
├── .github/
│   └── workflows/
│       └── ci.yml               # Linting + tests on push
├── configs/
│   ├── default.yaml             # Default hyperparameters
│   ├── ablation_bimodal.yaml    # Config for bi-modal experiments
│   └── sweep.yaml               # Hyperparameter sweep config
├── data/
│   ├── raw/                     # Downloaded data (gitignored)
│   ├── processed/               # Cleaned, normalized data (gitignored)
│   └── README.md                # Data provenance and download instructions
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA of Rosetta dataset
│   ├── 02_results_analysis.ipynb    # Main results and figures
│   └── 03_interpretability.ipynb    # Biology analysis
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py          # Script to download Rosetta data
│   │   ├── preprocess.py        # QC, normalization, splitting
│   │   ├── dataset.py           # PyTorch Dataset classes
│   │   └── featurize.py         # SMILES → molecular graph conversion
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py          # MolecularEncoder, MorphologyEncoder, ExpressionEncoder
│   │   ├── capy.py           # CaPy model (combines encoders + loss)
│   │   └── losses.py            # InfoNCE implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loop with logging
│   │   └── scheduler.py         # LR scheduling utilities
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── retrieval.py         # Cross-modal retrieval metrics
│   │   ├── clustering.py        # MOA clustering evaluation
│   │   └── interpretability.py  # Feature importance, gene mapping
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # YAML config loading
│       └── logging.py           # Wandb/console logging
├── scripts/
│   ├── train.py                 # Main training entrypoint
│   ├── evaluate.py              # Run evaluation on saved model
│   ├── run_ablations.py         # Run all ablation experiments
│   └── generate_figures.py      # Create paper-quality figures
└── tests/
    ├── test_data.py             # Test data loading and preprocessing
    ├── test_models.py           # Test forward passes, output shapes
    ├── test_losses.py           # Test InfoNCE implementation
    └── test_retrieval.py        # Test retrieval metric computation
```

### 10.2 Engineering Standards

| Standard | Specification |
|----------|--------------|
| Python version | 3.10+ |
| Formatter | `black` (line length 88) |
| Linter | `ruff` |
| Type hints | Required on all public functions |
| Docstrings | Google style, required on all classes and public functions |
| Testing | `pytest`, minimum 80% coverage on `src/` |
| Config management | YAML files loaded via `omegaconf` or `pydantic` |
| Experiment tracking | Weights & Biases (`wandb`) |
| Reproducibility | Seed all random sources; log every config; Docker for environment |
| Git | Conventional commits; `.gitignore` for data/checkpoints; no large files |

### 10.3 Key Dependencies

```toml
[project]
name = "capy"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    # Core ML
    "torch>=2.1",
    "torch-geometric>=2.4",
    
    # Chemistry
    "rdkit>=2023.09",       # SMILES parsing, molecular graphs
    
    # Biology
    "scanpy>=1.9",          # Single-cell / expression data utilities
    "gseapy>=1.0",          # Gene set enrichment analysis
    
    # Data
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "scipy>=1.11",
    
    # Visualization
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "umap-learn>=0.5",
    
    # Infrastructure
    "wandb>=0.16",
    "omegaconf>=2.3",
    "tqdm>=4.66",
    
    # Code quality
    "black>=24.0",
    "ruff>=0.2",
    "pytest>=7.4",
    "pytest-cov>=4.1",
]
```

### 10.4 Docker Configuration

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY pyproject.toml .
RUN pip install -e ".[dev]"

COPY . .

# Default: run training
CMD ["python", "scripts/train.py", "--config", "configs/default.yaml"]
```

---

## 11. Implementation Timeline

### Phase 1: Foundation (Weeks 1–2)

**Week 1: Data Pipeline**
- [ ] Set up repo structure, pyproject.toml, Docker
- [ ] Write `data/download.py` — download Rosetta CDRP-bio profiles
- [ ] Write `data/preprocess.py` — QC, normalization, scaffold splitting
- [ ] Write `data/featurize.py` — SMILES → PyG molecular graph
- [ ] Write `data/dataset.py` — CaPyDataset class (tri-modal)
- [ ] Write `tests/test_data.py` — verify shapes, no NaNs, split integrity
- [ ] Notebook `01_data_exploration.ipynb` — visualize distributions, check alignment

**Week 2: Model Architecture**
- [ ] Write `models/encoders.py` — all three encoders
- [ ] Write `models/losses.py` — InfoNCE loss
- [ ] Write `models/capy.py` — full model
- [ ] Write `tests/test_models.py` — forward pass shapes, gradient flow
- [ ] Write `tests/test_losses.py` — loss value sanity checks
- [ ] Verify end-to-end: data → model → loss → backward pass works

**Milestone:** Can run `python scripts/train.py` and see loss decrease.

### Phase 2: Training & Baselines (Weeks 3–4)

**Week 3: Training Infrastructure**
- [ ] Write `training/trainer.py` — full training loop with wandb logging
- [ ] Write `evaluation/retrieval.py` — all 6 retrieval directions + metrics
- [ ] Implement early stopping, checkpointing, LR scheduling
- [ ] First full training run — verify convergence
- [ ] Debug and tune (batch size, LR, temperature)

**Week 4: Baselines & Bi-Modal Ablations**
- [ ] Train bi-modal variant: mol ↔ morph only
- [ ] Train bi-modal variant: mol ↔ expr only
- [ ] Train bi-modal variant: morph ↔ expr only
- [ ] Implement MLP regression baseline (reproduce Rosetta paper)
- [ ] Compare all variants — generate comparison table

**Milestone:** Have the core result: tri-modal vs. bi-modal performance comparison.

### Phase 3: Analysis & Interpretation (Weeks 5–6)

**Week 5: Evaluation & MOA Analysis**
- [ ] Write `evaluation/clustering.py` — MOA clustering metrics
- [ ] Run zero-shot MOA classification
- [ ] UMAP visualizations of embedding spaces
- [ ] Embedding dimension ablation (64, 128, 256, 512)
- [ ] Lambda weight sensitivity analysis

**Week 6: Interpretability**
- [ ] Write `evaluation/interpretability.py`
- [ ] Feature importance: which morph features predict expression?
- [ ] Gene-morphology mapping + GSEA
- [ ] Modality disagreement analysis
- [ ] Notebook `03_interpretability.ipynb` — all biology figures

**Milestone:** Can tell a biological story about what the model learned.

### Phase 4: Polish & Ship (Weeks 7–8)

**Week 7: Code Quality & Documentation**
- [ ] Hit 80% test coverage
- [ ] Run black + ruff, fix all issues
- [ ] Write comprehensive README with results
- [ ] Ensure Docker builds and reproduces key results
- [ ] Add CI workflow (GitHub Actions)

**Week 8: Presentation & Stretch Goals**
- [ ] Notebook `02_results_analysis.ipynb` — paper-quality figures
- [ ] Write 2-page technical summary (PDF)
- [ ] [Stretch] Streamlit demo app
- [ ] [Stretch] Try AttentiveFP as alternative molecular encoder
- [ ] Final review and polish

**Milestone:** GitHub repo is portfolio-ready. README shows results. Code is clean.

---

## 12. Glossary for Non-Biologists

| Term | What it means | Why it matters for this project |
|------|--------------|-------------------------------|
| **SMILES** | A text encoding of molecular structure (e.g., `CCO` = ethanol) | We convert these to graphs for the GNN encoder |
| **Cell Painting** | A lab protocol that stains 6 cellular components with fluorescent dyes and photographs them | Source of our morphology features — how the cell *looks* |
| **CellProfiler** | Software that extracts ~1,000 numerical measurements from Cell Painting images | We use pre-computed CellProfiler features, not raw images |
| **L1000** | An assay that measures expression of 978 landmark genes | Source of our expression features — what genes the cell *activated* |
| **MOA (Mechanism of Action)** | How a drug works — e.g., "inhibits tubulin" or "blocks HDAC enzyme" | Our evaluation metric: do drugs with the same MOA cluster together? |
| **Perturbation** | Anything done to a cell to change it — adding a drug, knocking out a gene | Each data point in Rosetta is one perturbation (one compound) |
| **Bemis-Murcko scaffold** | The "backbone" of a molecule — removing side chains to get core structure | We split train/test by scaffold so similar molecules don't leak |
| **U2OS** | A human bone cancer cell line used in the Rosetta experiments | Just the cell type — all data comes from this one type |
| **Gene Set Enrichment Analysis (GSEA)** | Statistical test: is a specific biological pathway (e.g., "inflammation") over-represented in our gene list? | How we translate "these 50 genes are most predictable from morphology" into biological meaning |
| **z-score** | How many standard deviations a value is from the mean | L1000 expression values are z-scored: +2 = strongly upregulated |
| **GIN (Graph Isomorphism Network)** | A type of GNN that's provably as powerful as the WL graph isomorphism test | Our molecular encoder architecture |
| **InfoNCE** | A contrastive loss function: "among N options, pick the correct match" | The loss function that makes CaPy learn |
| **Contrastive learning** | Learning by comparing: "these two things are similar, those are different" | The entire training paradigm — no explicit labels needed |

---

## 13. Key References

### Must-Read (in this order)

1. **Rosetta dataset paper** — Haghighi et al., "High-dimensional gene expression and morphology profiles of cells across 28,000 genetic and chemical perturbations," *Nature Methods*, 2022. **This is your primary data source. Read the abstract and methods.**

2. **CLIP** — Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," *ICML*, 2021. **The architecture pattern you're implementing. Read sections 2.1-2.3.**

3. **CellCLIP** — Weinberger et al., "Learning Perturbation Effects in Cell Painting via Text-Guided Contrastive Learning," *NeurIPS/ICLR Workshop*, 2025. **The closest prior work. Understand how yours differs (3 modalities, learned molecular representations).**

4. **GIN** — Xu et al., "How Powerful are Graph Neural Networks?", *ICLR*, 2019. **Your molecular encoder architecture.**

### Good to Skim

5. **insitro POSH paper** — "A pooled Cell Painting CRISPR screening platform enables de novo inference of gene function," *Nature Communications*, 2025. **Shows how insitro thinks about Cell Painting data.**

6. **Cell Painting review** — Bray et al., "Cell Painting, a high-content image-based assay for morphological profiling," *Nature Protocols*, 2016.

7. **PyTorch Geometric tutorial** — Official PyG docs for GIN implementation.
