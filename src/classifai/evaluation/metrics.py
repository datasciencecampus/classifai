# from sklearn.metrics import roc_auc_score
# TODO: consider adding sci-kit learn metrics instead of manual implenentations, might make adding new metrics easier in the future (see auroc for example at end.)


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
