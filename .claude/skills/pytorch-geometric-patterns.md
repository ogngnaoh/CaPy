---
name: pytorch-geometric-patterns
description: PyTorch Geometric patterns for molecular graphs. Load when working on molecular encoder or data featurization.
---

# PyG Patterns for CaPy

## SMILES to Graph
Use RDKit to parse SMILES, extract atom and bond features, create torch_geometric.data.Data objects. Node features shape: [num_atoms, feat_dim]. Edge index shape: [2, num_edges] in COO format. Must include BOTH directions (i->j AND j->i).

## Batching Variable-Size Graphs
PyG DataLoader handles this via Batch.from_data_list(). Creates single disconnected graph with batch vector mapping nodes to original graphs. global_mean_pool(x, batch) pools per-graph.

## GIN Layer
GINConv wraps an MLP applied to aggregated neighborhoods. Pattern:
mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
conv = GINConv(mlp)

## Common Pitfalls
- edge_index must be Long tensor, not Float
- Edge features need both directions
- batch vector is auto-created by DataLoader
- Forgetting to L2-normalize output before contrastive loss
