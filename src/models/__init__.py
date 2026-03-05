"""Model definitions: encoders, losses, and the CaPy model.

Imports are lazy so the package can be imported even when heavy
dependencies (torch, torch_geometric, etc.) are not installed.
"""

from __future__ import annotations


def __getattr__(name: str):  # noqa: ANN204
    """Lazy-load public API objects on first access."""
    _exports = {
        "AtomEncoder": ("src.models.encoders", "AtomEncoder"),
        "MolecularEncoder": ("src.models.encoders", "MolecularEncoder"),
        "TabularEncoder": ("src.models.encoders", "TabularEncoder"),
        "CaPyModel": ("src.models.capy", "CaPyModel"),
        "info_nce": ("src.models.losses", "info_nce"),
    }
    if name in _exports:
        module_path, attr = _exports[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AtomEncoder",
    "CaPyModel",
    "MolecularEncoder",
    "TabularEncoder",
    "info_nce",
]
