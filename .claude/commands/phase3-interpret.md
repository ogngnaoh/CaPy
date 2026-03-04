Implement interpretability analysis. Create:

1. src/evaluation/interpretability.py
   Functions for:
   - feature_importance: for each morphology feature, compute Pearson correlation between that feature's values and the morph->expr embedding alignment score. Return top 20 features.
   - gene_morphology_mapping: for each gene, correlate expression values with morph->expr retrieval rank. Top 100 genes are "morphology-visible".
   - scaffold_analysis: group compounds by Bemis-Murcko scaffold, compute mean embedding per scaffold, find scaffolds closest to each MOA cluster center.
   - modality_disagreement: for each compound pair, compute rank in morph space and rank in expr space. Find pairs with largest rank discrepancy. Return top 10 in each direction (morph-similar/expr-different and expr-similar/morph-different).

2. notebooks/03_interpretability.ipynb
   Section 1: Top 20 morphology features bar chart, color-coded by compartment
   Section 2: Gene-morphology mapping results
   Section 3: GSEA on top 100 genes using gseapy with MSigDB Hallmark gene sets
   Section 4: Top 10 disagreement compound pairs table with known targets
   Section 5: Scaffold-to-MOA mapping visualization
   All figures paper-quality (300 DPI, clean fonts, colorblind-friendly).

After implementation, use the architect-explainer agent to explain: how to read GSEA results, what morphology-visible genes are biologically, why modality disagreement is the most novel analysis in the project.
