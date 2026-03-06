Create/update three documentation files and verify the Dockerfile:

1. **README.md** (project root) with these sections:
   - Title + one-sentence description
   - Key results table (ablation comparison, winning values bolded)
   - Embedded UMAP figure: `![Embedding Space](results/figures/embedding_umap.png)`
   - Why this matters (one paragraph for hiring manager, one for researcher)
   - Architecture diagram (ASCII art matching PRD section 4.1)
   - Quick start (clone, install, download data, train, evaluate)
   - Project structure (annotated directory tree)
   - Key design decisions table ("we chose X because Y")
   - Full results (retrieval metrics + MOA clustering tables)
   - Interpretability highlights (top findings from biology analysis)
   - References (5 key papers: CLIP, SimCLR, GIN, Cell Painting, L1000)
   - MIT License badge

2. **data/README.md** with:
   - Data provenance (Rosetta CDRP-bio, Broad Institute)
   - Download instructions (S3 + direct HTTP)
   - Raw file format description
   - Preprocessing steps summary
   - Processed file contents and shapes

3. **docs/technical_summary.md** — 2-page technical summary:
   - Problem (2 paragraphs)
   - Method (3 paragraphs + architecture diagram)
   - Results (1 paragraph + main results table)
   - Discussion (1 paragraph)
   - References (5 key papers)

4. **Dockerfile** — Verify/update to match current dependencies in pyproject.toml

Make the README impressive but honest. Do not overclaim results.
