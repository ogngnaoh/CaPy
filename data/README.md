# Data Directory

## Download Instructions

### Automated Download
```bash
python -m src.data.download --output-dir data/raw
```

### Manual Download (Rosetta CDRP-bio)

**Source:** Broad Institute Cell Painting Gallery
**Paper:** Haghighi et al., Nature Methods 2022

```bash
# Morphology profiles (CellProfiler features)
aws s3 cp s3://cellpainting-gallery/cpg0003-rosetta/broad/workspace/profiles/CDRP-bio/ \
    ./data/raw/morphology/ --recursive --no-sign-request

# Expression profiles (L1000 genes)
aws s3 cp s3://cellpainting-gallery/cpg0003-rosetta/broad/workspace/profiles/CDRP-bio/ \
    ./data/raw/expression/ --recursive --no-sign-request
```

## Directory Structure

```
data/
├── raw/              # Downloaded data (gitignored)
│   ├── morphology/   # CellProfiler features
│   ├── expression/   # L1000 gene expression
│   └── metadata/     # Compound IDs, SMILES, MOA
├── processed/        # Cleaned, normalized, split (gitignored)
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
└── README.md         # This file
```

## Dataset Summary

| Property           | Value                     |
|--------------------|---------------------------|
| Cell line          | U2OS                      |
| Compounds          | ~2,084                    |
| Morphology features| ~1,000 (after QC)         |
| Expression features| 978 L1000 landmark genes  |
| Split              | 70/15/15 by Bemis-Murcko scaffold |
