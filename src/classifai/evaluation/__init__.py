"""Evaluation utilities for ClassifAI.

This package provides helpers for evaluating `classifai.indexers.VectorStore` instances
against ground-truth labelled datasets, measured via a set of classification metrics.

The core functionality is provided by the `Evaluation` class in `classifai.evaluation`, which is associated with a labelled
testing dataset, and a set of metrics on which to evaluate the performance of a `VectorStore`.

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
