"""Data pipeline: download, preprocess, featurize, and dataset classes.

Imports are lazy so the package can be imported even when heavy
dependencies (torch, rdkit, pandas, etc.) are not installed.
"""

from __future__ import annotations


def __getattr__(name: str):  # noqa: ANN204
    """Lazy-load public API objects on first access."""
    _exports = {
        "CaPyDataset": ("src.data.dataset", "CaPyDataset"),
        "capy_collate_fn": ("src.data.dataset", "capy_collate_fn"),
        "download_rosetta_profiles": ("src.data.download", "download_rosetta_profiles"),
        "smiles_to_graph": ("src.data.featurize", "smiles_to_graph"),
        "preprocess_pipeline": ("src.data.preprocess", "preprocess_pipeline"),
    }
    if name in _exports:
        module_path, attr = _exports[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CaPyDataset",
    "capy_collate_fn",
    "download_rosetta_profiles",
    "preprocess_pipeline",
    "smiles_to_graph",
]
