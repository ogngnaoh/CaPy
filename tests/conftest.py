"""Shared fixtures and skip markers for CaPy test suite.

WHY THIS WORKS
--------------
Heavy ML dependencies (torch, rdkit, pandas, numpy, torch_geometric) are
optional at test time.  We try-import each one and expose ``HAS_*`` flags
plus ``pytest.mark`` skip decorators so individual tests declare their
requirements declaratively.  ``pytest`` collects *all* tests but skips
those whose dependencies are missing — no import errors, no manual
exclusion lists.
"""

import pytest

# ---------------------------------------------------------------------------
# Dependency detection
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from rdkit import Chem  # noqa: F401

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    import pandas  # noqa: F401

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch_geometric  # noqa: F401

    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_rdkit = pytest.mark.skipif(not HAS_RDKIT, reason="rdkit not installed")
requires_pandas = pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
requires_numpy = pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
requires_pyg = pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")

# Composite: all ML deps needed for end-to-end dataset tests
requires_ml = pytest.mark.skipif(
    not all([HAS_TORCH, HAS_RDKIT, HAS_PANDAS, HAS_NUMPY, HAS_PYG]),
    reason="one or more ML dependencies not installed (torch/rdkit/pandas/numpy/pyg)",
)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SIMPLE_SMILES = [
    "CCO",  # ethanol
    "c1ccccc1",  # benzene
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "C(=O)O",  # formic acid
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",  # pyrene
    "CC(=O)NC1=CC=C(O)C=C1",  # acetaminophen
    "ClC(Cl)Cl",  # chloroform
    "C",  # methane (single atom)
]
