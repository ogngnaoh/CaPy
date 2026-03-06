Create notebooks/02_results_analysis.ipynb with these sections (each in its own cell group):

1. **Load & Embed** — Load best checkpoint, embed all test compounds in 3 modalities
2. **UMAP** — Joint embedding space colored by MOA class (10x8 figure, clean palette, legend outside plot, colorblind-friendly)
3. **Training Curves** — Total loss + per-pair losses over epochs (from wandb export or saved logs)
4. **Retrieval Heatmap** — 6-direction similarity grid (annotated heatmap, diverging colormap)
5. **Ablation Comparison** — Grouped bar chart of R@10 for tri-modal vs each bi-modal variant
6. **Lambda Sensitivity** — Bar chart of R@10 across different lambda weight ratios
7. **Save Figures** — Save all to results/figures/ as PNG (300 DPI) + SVG

All figures must be paper-quality: 300 DPI, clean fonts (no default matplotlib), no chartjunk, colorblind-friendly palette (use seaborn or a curated list). Add explanatory markdown cells between each section.

Include `%matplotlib inline` and set the matplotlib style at the top of the notebook.
