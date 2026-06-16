"""Evaluation utilities for ClassifAI.

This package provides helpers for evaluating `classifai.indexers.VectorStore` instances
against ground-truth labelled datasets, plus a small set of classification metrics.

Most users should start with:
    - `classifai.evaluation.main.evaluate`
    - `classifai.evaluation.main.parse_metrics`

Metric implementations are defined in `classifai.evaluation.metrics`.

Key Features:
- Evaluate and compare the performance of multiple `VectorStore` objects on a given dataset.
- Support for various classification metrics, including accuracy, precision, recall, and F1 score.
- Consistent interfaces for datasets and evaluation metrics.
- Optional functionality to save evaluation results for future analysis and comparison.
- Support for providing custom `VectorStore` loading functions to optimize memory usage during evaluation.

Warning:
    This module is currently in development and its API is subject to change in future releases.
    Use with caution in production environments.
"""

import warnings

from .main import Evaluation as Evaluation

# Issue a module-level warning when imported
warnings.warn(
    "The evaluation module is currently in development and its API is subject to change in future releases.",
    FutureWarning,
    stacklevel=2,
)
