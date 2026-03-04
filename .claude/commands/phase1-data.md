Implement the complete data pipeline for CaPy. Create these files:

1. src/data/download.py
   Download Rosetta CDRP-bio profiles. Try S3 first (aws s3 cp with --no-sign-request), fall back to direct HTTP links from Cell Painting Gallery GitHub. Include a --dry-run flag. Use tqdm for progress. Save to data/raw/.

2. src/data/preprocess.py
   Match compounds across morphology and expression by compound ID. Remove compounds missing any modality. Remove zero-variance and >50% NaN morphology features. RobustScaler for morphology (clip to [-5,5]). Verify z-scores for expression. Bemis-Murcko scaffold split 70/15/15, stratified by MOA where available. Save to data/processed/.

3. src/data/featurize.py
   SMILES to PyG molecular graphs. 9-dim node features (atomic number one-hot, degree, formal charge, num H, is aromatic, is in ring). 4-dim edge features (bond type, conjugated, in ring, stereo). Both directions for edges. Use RDKit.

4. src/data/dataset.py
   CaPyDataset(torch.utils.data.Dataset) returning (mol_graph, morph_vector, expr_vector) triples. Custom collate_fn that uses PyG Batch for molecular graphs and torch.stack for vectors.

5. tests/test_data.py
   Test with synthetic data (no real download needed): correct shapes, no NaNs, split proportions, graph validity (edge_index is Long, both directions present), collate function produces correct batch.

After implementation, use the architect-explainer agent to explain: scaffold splitting and data leakage prevention, PyG batching of variable-size graphs, why RobustScaler over StandardScaler for morphology features.
