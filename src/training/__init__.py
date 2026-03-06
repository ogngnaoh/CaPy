"""Training loop, scheduling, and optimization utilities."""

from __future__ import annotations


def __getattr__(name: str):  # noqa: ANN204
    """Lazy-load public API objects on first access."""
    _exports = {
        "Trainer": ("src.training.trainer", "Trainer"),
        "CosineAnnealingWithWarmup": (
            "src.training.scheduler",
            "CosineAnnealingWithWarmup",
        ),
    }
    if name in _exports:
        module_path, attr = _exports[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Trainer", "CosineAnnealingWithWarmup"]
