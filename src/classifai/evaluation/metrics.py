"""Metrics.py provides a set of evaluation metrics for multiclass, single-label classification tasks.
Each metric is designed to assess the performance of a classification model in different ways,
providing insights into its accuracy, precision, recall, and overall balance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class MetricResult:
    """Represents the result of a metric evaluation."""

    name: str
    value: float

    def __repr__(self) -> str:
        return f"{self.name}: {self.value:.4f}"


class Metric(ABC):
    """Base class for all classification metrics."""

    @abstractmethod
    def evaluate(self, eval_data: pd.DataFrame) -> MetricResult:
        """Evaluate the metric on the provided evaluation data.

        Args:
            eval_data: DataFrame with 'doc_label' and 'ground_truth_label' columns.

        Returns:
            MetricResult containing the metric name and computed value.
        """
        pass


def _get_unique_labels(eval_data: pd.DataFrame) -> set[str]:
    """Return all labels present in either predictions or ground truth."""
    labels = set(eval_data["doc_label"].dropna().unique())
    labels.update(eval_data["ground_truth_label"].dropna().unique())
    return labels


class ClassificationAccuracy(Metric):
    """Calculate the accuracy of results."""

    def evaluate(self, eval_data: pd.DataFrame) -> MetricResult:
        correct_predictions = (eval_data["doc_label"] == eval_data["ground_truth_label"]).sum()
        total_predictions = len(eval_data)
        value = correct_predictions / total_predictions
        return MetricResult(name="accuracy", value=value)


class ClassificationMacroRecall(Metric):
    """Calculate macro recall by averaging per-label recall."""

    def evaluate(self, eval_data: pd.DataFrame) -> MetricResult:
        labels = _get_unique_labels(eval_data)
        if not labels:
            return MetricResult(name="macro_recall", value=0.0)

        recalls = []
        for label in labels:
            true_positives = ((eval_data["doc_label"] == label) & (eval_data["ground_truth_label"] == label)).sum()
            false_negatives = ((eval_data["doc_label"] != label) & (eval_data["ground_truth_label"] == label)).sum()
            denominator = true_positives + false_negatives
            recalls.append(true_positives / denominator if denominator > 0 else 0)

        value = sum(recalls) / len(recalls)
        return MetricResult(name="macro_recall", value=value)


class ClassificationMacroPrecision(Metric):
    """Calculate macro precision by averaging per-label precision."""

    def evaluate(self, eval_data: pd.DataFrame) -> MetricResult:
        labels = _get_unique_labels(eval_data)
        if not labels:
            return MetricResult(name="macro_precision", value=0.0)

        precisions = []
        for label in labels:
            true_positives = ((eval_data["doc_label"] == label) & (eval_data["ground_truth_label"] == label)).sum()
            false_positives = ((eval_data["doc_label"] == label) & (eval_data["ground_truth_label"] != label)).sum()
            denominator = true_positives + false_positives
            precisions.append(true_positives / denominator if denominator > 0 else 0)

        value = sum(precisions) / len(precisions)
        return MetricResult(name="macro_precision", value=value)


class ClassificationMacroF1(Metric):
    """Calculate macro F1 by averaging per-label F1."""

    def evaluate(self, eval_data: pd.DataFrame) -> MetricResult:
        labels = _get_unique_labels(eval_data)
        if not labels:
            return MetricResult(name="macro_f1", value=0.0)

        f1_scores = []
        for label in labels:
            true_positives = ((eval_data["doc_label"] == label) & (eval_data["ground_truth_label"] == label)).sum()
            false_positives = ((eval_data["doc_label"] == label) & (eval_data["ground_truth_label"] != label)).sum()
            false_negatives = ((eval_data["doc_label"] != label) & (eval_data["ground_truth_label"] == label)).sum()

            precision_denominator = true_positives + false_positives
            recall_denominator = true_positives + false_negatives

            precision = true_positives / precision_denominator if precision_denominator > 0 else 0
            recall = true_positives / recall_denominator if recall_denominator > 0 else 0
            f1_scores.append(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)

        value = sum(f1_scores) / len(f1_scores)
        return MetricResult(name="macro_f1", value=value)
