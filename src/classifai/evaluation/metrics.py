"""Metrics.py provides a set of evaluation metrics for multiclass, single-label classification tasks.
Each metric is designed to assess the performance of a classification model in different ways,
providing insights into its accuracy, precision, recall, and overall balance.

Metrics Implemented:
---------------------

1. **compute_classification_accuracy(eval_data)**:
    - **Description**: Calculates the overall accuracy of the classification model.
        Accuracy is defined as the proportion of correct predictions out of the total number of predictions.
    - **Intuition**: This metric gives a general sense of how often the model is correct,
        but it may not be sufficient for imbalanced datasets.

2. **compute_classification_macro_recall(eval_data)**:
    - **Description**: Computes the macro-averaged recall, which is the average recall across all classes,
        treating each class equally regardless of its frequency.
    - **Intuition**: Recall measures the ability of the model to correctly identify all instances of a class.
        Macro recall ensures that performance on smaller classes is not overshadowed by larger classes.

3. **compute_classification_macro_precision(eval_data)**:
    - **Description**: Computes the macro-averaged precision, which is the average precision across all classes,
        treating each class equally regardless of its frequency.
    - **Intuition**: Precision measures the ability of the model to avoid false positives for a class.
        Macro precision ensures that smaller classes are given equal importance as larger ones.

4. **compute_classification_macro_f1(eval_data)**:
    - **Description**: Computes the macro-averaged F1 score, which is the harmonic mean of precision and recall,
        averaged across all classes.
    - **Intuition**: The F1 score balances precision and recall, making it a good metric for imbalanced datasets.
        Macro F1 ensures that all classes contribute equally to the final score.

Helper Function:
-----------------

- **_get_unique_labels(eval_data)**:
    - **Description**: Identifies all unique labels present in the predictions and ground truth.
        This function is used internally by the other metrics to ensure all classes are considered.

Future Work:
------------
- A top-K accuracy metric is planned for implementation, which will evaluate the accuracy of the model
    when considering the top K predictions for each instance.
"""


def _get_unique_labels(eval_data):
    """Return all labels present in either predictions or ground truth."""
    labels = set(eval_data["doc_label"].dropna().unique())
    labels.update(eval_data["ground_truth_label"].dropna().unique())
    return labels


def compute_classification_accuracy(eval_data):
    """Calculate the accuracy of results. Accuracy is defined as the number of correct predictions divided by the total number of predictions."""
    correct_predictions = (eval_data["doc_label"] == eval_data["ground_truth_label"]).sum()
    total_predictions = len(eval_data)
    return correct_predictions / total_predictions


def compute_classification_macro_recall(eval_data):
    """Calculate macro recall by averaging per-label recall in a one-vs-rest fashion."""
    labels = _get_unique_labels(eval_data)
    if not labels:
        return 0

    recalls = []
    for label in labels:
        true_positives = ((eval_data["doc_label"] == label) & (eval_data["ground_truth_label"] == label)).sum()
        false_negatives = ((eval_data["doc_label"] != label) & (eval_data["ground_truth_label"] == label)).sum()
        denominator = true_positives + false_negatives
        recalls.append(true_positives / denominator if denominator > 0 else 0)

    return sum(recalls) / len(recalls)


def compute_classification_macro_precision(eval_data):
    """Calculate macro precision by averaging per-label precision in a one-vs-rest fashion."""
    labels = _get_unique_labels(eval_data)
    if not labels:
        return 0

    precisions = []
    for label in labels:
        true_positives = ((eval_data["doc_label"] == label) & (eval_data["ground_truth_label"] == label)).sum()
        false_positives = ((eval_data["doc_label"] == label) & (eval_data["ground_truth_label"] != label)).sum()
        denominator = true_positives + false_positives
        precisions.append(true_positives / denominator if denominator > 0 else 0)

    return sum(precisions) / len(precisions)


def compute_classification_macro_f1(eval_data):
    """Calculate macro F1 by averaging per-label F1 in a one-vs-rest fashion."""
    labels = _get_unique_labels(eval_data)
    if not labels:
        return 0

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

    return sum(f1_scores) / len(f1_scores)


# TODO: add a top-(k) accuracy metric where we assess the top K answers for accurracy
