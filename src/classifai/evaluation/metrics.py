# from sklearn.metrics import roc_auc_score
# TODO: consider adding sci-kit learn metrics instead of manual implenentations, might make adding new metrics easier in the future (see auroc for example at end.)


def classification_accuracy(eval_data):
    """Calculate the accuracy of results. Accuracy is defined as the number of correct predictions divided by the total number of predictions."""
    correct_predictions = (eval_data["doc_label"] == eval_data["ground_truth_label"]).sum()
    total_predictions = len(eval_data)
    return correct_predictions / total_predictions


def classification_recall(eval_data):
    """Calculate the recall of results. Recall is defined as the number of true positives divided by the number of true positives plus the number of false negatives."""
    true_positives = ((eval_data["doc_label"] == 1) & (eval_data["ground_truth_"] == 1)).sum()
    false_negatives = ((eval_data["doc_label"] == 0) & (eval_data["ground_truth_"] == 1)).sum()
    return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0


def classification_precision(eval_data):
    """Calculate the precision of results. Precision is defined as the number of true positives divided by the number of true positives plus the number of false positives."""
    true_positives = ((eval_data["doc_label"] == 1) & (eval_data["ground_truth_"] == 1)).sum()
    false_positives = ((eval_data["doc_label"] == 1) & (eval_data["ground_truth_"] == 0)).sum()
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0


def classification_f1(eval_data):
    """Calculate the F1 score of results. The F1 score is defined as the harmonic mean of precision and recall, and is calculated as 2 * (precision * recall) / (precision + recall)."""
    true_positives = ((eval_data["doc_label"] == 1) & (eval_data["ground_truth_"] == 1)).sum()
    false_positives = ((eval_data["doc_label"] == 1) & (eval_data["ground_truth_"] == 0)).sum()
    false_negatives = ((eval_data["doc_label"] == 0) & (eval_data["ground_truth_"] == 1)).sum()

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


# def classification_auroc(eval_data):
#     """Calculate the Area Under the Receiver Operating Characteristic Curve (AUROC) of results. AUROC is a measure of how well a model can distinguish between classes, and is calculated by plotting the true positive rate against the false positive rate at various threshold settings."""
#     predictions = eval_data['doc_string']
#     ground_truth = eval_data['label']
#     return roc_auc_score(ground_truth, predictions)
