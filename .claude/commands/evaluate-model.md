Create scripts/evaluate.py that performs full model evaluation:

1. Load the best tri-modal checkpoint from checkpoints/best_model.pt
2. Embed all test-split compounds in all 3 modalities (molecular, morphology, expression)
3. Run MOA clustering evaluation:
   - k-means clustering (k = number of unique MOAs in test set)
   - Agglomerative clustering (same k)
   - Metrics: AMI, ARI, silhouette score
4. Run k-NN MOA classification accuracy for k=5, 10, 20
5. Run all 6-direction retrieval metrics (Recall@1, Recall@5, Recall@10, MRR)
6. Save everything to results/evaluation_results.json
7. Report the number of annotated compounds (those with MOA labels) vs total

Handle compounds without MOA labels by excluding them from clustering/classification metrics but including them in retrieval metrics.

Use the config from configs/default.yaml for model architecture parameters. Log results to console using the project logger.

After implementation, use the architect-explainer agent to explain: what hub points are in retrieval and why retrieval can be asymmetric, what silhouette score measures geometrically.
