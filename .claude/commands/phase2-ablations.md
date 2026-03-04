Implement ablation study infrastructure. Create:

1. configs/ablation_mol_morph.yaml (lambda1=1.0, lambda2=0.0, lambda3=0.0)
2. configs/ablation_mol_expr.yaml (lambda1=0.0, lambda2=1.0, lambda3=0.0)
3. configs/ablation_morph_expr.yaml (lambda1=0.0, lambda2=0.0, lambda3=1.0)

All ablation configs inherit from default.yaml and only override lambda weights. Use the same random seed as default.

4. scripts/run_ablations.py
   Run tri-modal + 3 bi-modal variants sequentially. Same seeds across all variants. Save each variant's metrics to results/. Generate comparison table (markdown) at the end.

5. src/evaluation/clustering.py
   MOA evaluation: k-means clustering (k = number of unique MOAs), agglomerative clustering. Metrics: AMI, ARI, silhouette score. k-NN MOA classification accuracy for k=5, 10, 20. Handle compounds without MOA labels (exclude from evaluation, report count).

After implementation, use the architect-explainer agent to explain: what the ablation results will tell us biologically, how to interpret AMI and ARI for a clustering beginner, what it means if one bi-modal pair dominates.
