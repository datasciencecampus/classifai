"""This module provides tools for evaluating the performance of `VectorStore` objects.

It includes a suite of evaluation metrics and methods to assess the effectiveness of `VectorStore` implementations,
particularly in classification tasks. The module is designed to facilitate consistent and reproducible evaluation
of `VectorStore` objects.

Key Features:
- Evaluate and compare the performance of multiple `VectorStore` objects on a given dataset.
- Support for various classification metrics, including accuracy, precision, recall, and F1 score.
- Consistent interfaces for datasets and evaluation metrics.
- Optional functionality to save evaluation results for future analysis and comparison.
- Support for providing custom `VectorStore` loading functions to optimize memory usage during evaluation.
"""

from .main import evaluate as evaluate
