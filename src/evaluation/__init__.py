"""Evaluation: retrieval metrics, clustering, and interpretability."""

from __future__ import annotations


def __getattr__(name: str):  # noqa: ANN204
    """Lazy-load public API objects on first access."""
    _exports = {
        "compute_retrieval_metrics": (
            "src.evaluation.retrieval",
            "compute_retrieval_metrics",
        ),
        "evaluate_all_retrieval": (
            "src.evaluation.retrieval",
            "evaluate_all_retrieval",
        ),
    }
    if name in _exports:
        module_path, attr = _exports[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["compute_retrieval_metrics", "evaluate_all_retrieval"]
