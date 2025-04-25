"""Options for scoring labels"""
import numpy as np

def naive_scorer(scores: list, rankings: list) -> (float, int):
    """Scoring function that returns the top ranked score for a given SOC code label in the original ranking."""

    score = min(scores)
    index = scores.index(score)

    # the lowest (best) ranked score is just the first element of the list
    return score, index


def average_scorer(scores: list, rankings: list) -> (float, int):
    """Scoring function that computes the average score of all scores for a given SOC code label in the original ranking."""

    # calculate the average value from all ranked items
    score = sum(scores) / len(scores)
    index = scores.index(min(scores))

    return score, index


def softmax_scorer(scores: list) -> np.array:
    """Util function to convert distance scores for a ranking to percentages using softmax."""
    scores = np.array(scores).astype(float)
    softmaxed = np.exp(-scores) / sum(np.exp(-scores))

    return np.round(softmaxed, 3)