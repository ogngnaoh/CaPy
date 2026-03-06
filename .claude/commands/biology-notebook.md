Create notebooks/03_interpretability.ipynb with these sections:

1. **Feature Importance** — Bar chart of top 20 morphology features by correlation with morph→expr alignment. Color bars by CellProfiler compartment (Cells=blue, Nuclei=green, Cytoplasm=orange). Add explanatory markdown: what high correlation means biologically.

2. **Gene-Morphology Mapping** — Ranked list/chart of genes most correlated with morphology-based retrieval. Markdown explaining what "morphology-visible" genes are.

3. **GSEA** — Run Gene Set Enrichment Analysis on top 100 morphology-visible genes using gseapy with MSigDB Hallmark gene sets (human, h.all). Display enrichment table and top enrichment plots. Markdown explaining how to read GSEA results.

4. **Modality Disagreement** — Table of top 10 compound pairs with largest rank discrepancy between morph and expr spaces (5 morph-similar/expr-different, 5 expr-similar/morph-different). Include known drug targets if available. Markdown explaining why disagreement is biologically interesting.

5. **Molecular Substructure Analysis** — Group compounds by Bemis-Murcko scaffold, map scaffolds to nearest MOA cluster centers. Visualize scaffold→MOA relationships.

All figures paper-quality (300 DPI, clean fonts, colorblind-friendly). Include explanatory markdown cells between every section — write for a CS/math student learning biology.

After implementation, use the architect-explainer agent to explain: how to read GSEA enrichment plots, what Hallmark gene sets represent, why modality disagreement is the most novel analysis.
