"""This module evaluates one or more `classifai.indexers.VectorStore` instances on a ground-truth labelled dataset.

1. Validating the ground-truth input with a Pandera schema.
2. Running a batched top-1 `VectorStore.search` over all queries.
3. Merging the ground-truth label into the retrieved results.
4. Validating the merged evaluation frame with a Pandera schema.
5. Computing one or more multiclass, single-label classification metrics.

The evaluation is framed as retrieval-as-classification: for each query, the label of
the top retrieved document (`doc_label`) is treated as the model prediction, and the
provided dataset label is treated as the ground truth (`ground_truth_label`).

Input DataFrames:
    Ground-truth input (`ground_truths`) must include:
        - qid (str): Unique query identifier.
        - text (str): Query text.
        - label (str): Ground-truth label.

    Search evaluation output (`results_df`) is expected to include:
        - query_id (str): Query identifier (from `qid`).
        - query_text (str): Query text.
        - doc_label (str): Predicted label (label of retrieved doc).
        - doc_text (str): Retrieved document text.
        - rank (int): Rank of the retrieved document (>= 0).
        - score (float): Similarity score from the vector store.
        - ground_truth_label (str): Ground-truth label merged in from `ground_truths`.

Metrics:
    Metric functions are defined in `classifai.evaluation.metrics` and are selected via
    `parse_metrics`. Supported metric keys are:
        - "accuracy"
        - "macro_recall"
        - "macro_precision"
        - "macro_f1"
        - "classification_suite" (expands to all metrics above)

Exceptions:
    InvalidMetricError: Raised when requested metric names cannot be parsed.
    EvaluationError: Raised when validation, vectorstore execution, result validation, or
        metric computation fails.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa

from ..exceptions import ClassifaiError
from ..indexers import VectorStore
from ..indexers.dataclasses import VectorStoreSearchInput
from .metrics import (
    compute_classification_accuracy,
    compute_classification_macro_f1,
    compute_classification_macro_precision,
    compute_classification_macro_recall,
)


@dataclass(eq=False)
class InvalidMetricError(ClassifaiError):
    code: str = "invalid_metric_error"


@dataclass(eq=False)
class EvaluationError(ClassifaiError):
    code: str = "evaluation_error"


# pandera model for validating the content of the ground_truths dataframe
_GROUND_TRUTH_SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema(
    {
        "qid": pa.Column(str),
        "text": pa.Column(str),
        "label": pa.Column(str),
    },
    coerce=True,
)

# pandera model for validating the output of the VectorStore search method when passed the groun truth data. Ensures good search results before passing to the metric functions, which may error if the search results are not in the expected format.
_SEARCH_EVAL_OUTPUT_SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema(
    {
        "query_id": pa.Column(str),
        "query_text": pa.Column(str),
        "doc_label": pa.Column(str),
        "doc_text": pa.Column(str),
        "rank": pa.Column(int, pa.Check.ge(0)),
        "score": pa.Column(float),
        "ground_truth_label": pa.Column(str),
    },
    coerce=True,
)


def parse_metrics(metrics: list[str]) -> dict:
    """Parse a list of metric names and return a dictionary mapping metric names to their corresponding functions."""
    # dictionary of metric functions and their key names
    valid_metrics = {
        "accuracy": compute_classification_accuracy,
        "macro_recall": compute_classification_macro_recall,
        "macro_precision": compute_classification_macro_precision,
        "macro_f1": compute_classification_macro_f1,
    }

    # create a dictionary to store identified metrics
    parsed = {}
    for m in metrics:
        if m == "classification_suite":
            return valid_metrics

        if m in valid_metrics:
            parsed[m] = valid_metrics[m]
        else:
            raise ValueError(
                f"Invalid metric: {m}. Valid metrics are: {list(valid_metrics.keys())} or 'classification_suite' for all metrics."
            )
    return parsed


def _run_single_vectorstore_search(vectorstore: VectorStore, ground_truths: pd.DataFrame) -> pd.DataFrame:
    """Run a single `VectorStore` on the evaluation dataset and return the results as a DataFrame.

    Attributes:
    vectorstore: A `VectorStore` object to run on the evaluation dataset.
    ground_truths: A pandas DataFrame containing the ground truth labels for the evaluation dataset. It should have columns 'qid', 'text', and 'label'.

    Returns:
    A pandas DataFrame containing the results of the `VectorStore` search, with columns 'query_id', 'query_text', 'doc_label', 'doc_text', 'rank', 'score', and 'ground_truth_label'.
    """
    # batch the groun_truth rows into batches of N (==8 for now)
    _BATCH_SIZE = 8

    # build a VectorStoreSearchInput object fomr the ground_truths dataframe
    search_input = VectorStoreSearchInput(
        {
            "id": ground_truths["qid"].tolist(),
            "query": ground_truths["text"].tolist(),
        }
    )

    # do the batched search process and get the results utilising the VectorStore's built in batching capabilities
    result_df = vectorstore.search(search_input, n_results=1, batch_size=_BATCH_SIZE)

    # for each query_id in the results dataframe, merge the corresponding ground truth label from the ground_truths dataframe into the results dataframe
    eval_data = result_df.merge(ground_truths[["qid", "label"]], left_on="query_id", right_on="qid", how="left")
    # rename the label column to ground_truth_label to avoid confusion with the doc_label column in the results dataframe
    eval_data = eval_data.rename(columns={"label": "ground_truth_label"})

    return eval_data


def evaluate(  # noqa: C901 PLR0912 PLR0915
    vectorstores: list[VectorStore | Callable[[], VectorStore]],
    vectorstore_names: list[str],
    metrics: list[str],
    ground_truths: pd.DataFrame,
    output_file: str | None = None,
):
    """An evaluator evaluating performance of `VectorStore` objects for a given grount truth-labelled dataset and a set of evaluation metrics.

    Attributes:
    vectorstores: A list of `VectorStore` objects to evaluate, or callable functions that return `VectorStore` objects when executed. Each `VectorStore` will be evaluated on the same dataset and metrics.
    vectorstore_names: A list of names corresponding to the `VectorStore` objects, used for labeling results.
    metrics: A list of evaluation metrics to compute. Supported metrics include 'accuracy@1', 'hit@k', and 'mrr@k'.
    ground_truths: A pandas DataFrame containing the ground truth labels for the evaluation dataset. It should have columns 'qid', 'text', and 'label'.
    output_file: An optional string specifying the path to save the evaluation results as a CSV file. If None, results will not be saved to a file.


    Returns:
    A pandas DataFrame containing the evaluation results, with rows corresponding to `VectorStore` objects and columns corresponding to evaluation metrics. (prints to csv if output_file is provided)
    """
    # validate the ground_truths dataframe
    try:
        _GROUND_TRUTH_SCHEMA.validate(ground_truths)
    except Exception as e:
        raise EvaluationError("Ground truths dataframe failed validation.", context={"cause_message": str(e)}) from e

    # ensure that the vectorstores list is a full list of either vectorstore instances or functions that can be executed to return vectorstore instances, and that the vectorstore_names list is the same length as the vectorstores list.
    if not all(isinstance(vs, VectorStore) or callable(vs) for vs in vectorstores):
        raise ValueError(
            "All items in vectorstores must be instances of VectorStore or callables that will return a VectorStore."
        )

    # ensure that vectorstore_names is a list of strings with the same length as vectorstores
    if not isinstance(vectorstore_names, list) or len(vectorstore_names) != len(vectorstores):
        raise ValueError("vectorstore_names must be a list with the same length as vectorstores.")

    # ensure that all items in vectorstore_names are strings and that they are unique
    if not all(isinstance(name, str) for name in vectorstore_names):
        raise ValueError("All items in vectorstore_names must be strings.")
    if len(set(vectorstore_names)) != len(vectorstore_names):
        raise ValueError("All items in vectorstore_names must be unique.")

    # parse the metrics
    try:
        parsed_metrics = parse_metrics(metrics)
    except Exception as e:
        raise InvalidMetricError(
            "Faild to parse provided metrics.",
            context={"metrics": metrics, "cause_message": str(e)},
        ) from e

    # validate the output_file argument if provided
    if output_file and (not isinstance(output_file, str) or not output_file.strip().endswith(".csv")):
        raise ValueError("The output_file arg must be a valid string ending with '.csv'")

    # create an empty dataframe to store the results of the evaluation - columns will be metrics, rows will be vectorstore names
    overall_results_metrics_df = pd.DataFrame()

    # iterate through the vectostores
    for vs, name in zip(vectorstores, vectorstore_names, strict=False):
        # log the start of processing for the current vectorstore
        print(f"Processing VectorStore: {name}")

        # try to instantiate the vectorstore if it's a provided callable
        try:
            resolved_vs = vs() if callable(vs) else vs
        except Exception as e:
            raise EvaluationError(
                "Failed to instantiate a VectorStore from the provided callable.",
                context={"vectorstore_name": name, "cause_message": str(e)},
            ) from e

        try:
            # initiate the search process, which batches queries from ground truth and combines the ground truth labels into the results.
            results_df = _run_single_vectorstore_search(resolved_vs, ground_truths)
        except Exception as e:
            raise EvaluationError(
                "Something went wrong when a VectorStore tried to perform search on the evaluation dataset.",
                context={"vectorstore_name": name, "cause_message": str(e)},
            ) from e
        finally:
            if callable(vs):
                del resolved_vs

        # validate the results dataframe to ensure it has the expected format to be passed to the evaluation metric functions
        try:
            _SEARCH_EVAL_OUTPUT_SCHEMA.validate(results_df)
        except Exception as e:
            raise EvaluationError(
                "A VectorStore's  collected search results did not match the expected format for evaluation.",
                context={"vectorstore_name": name, "cause_message": str(e)},
            ) from e

        # per vectorstore results
        vs_computed_metrics = {}

        try:
            # for each metric, pass the vectorstore results to the metric function to compute the metric, and store the result in a dictionary
            for metric in parsed_metrics:
                result = parsed_metrics[metric](results_df)
                vs_computed_metrics[metric] = result
        except Exception as e:
            raise EvaluationError(
                "Failed to compute evaluation metrics for a Vectorstore's results.",
                context={"vectorstore_name": name, "metric": metric, "cause_message": str(e)},
            ) from e

        # convert the dictionary of metric results to a df
        vectorstore_metrics_df = pd.DataFrame(vs_computed_metrics, index=[name])

        # add the most recently computed metrics to the overall results dataframe by concatenating the dataframs
        overall_results_metrics_df = pd.concat([overall_results_metrics_df, vectorstore_metrics_df])

    # set the column names of the overall results dataframe to be the metric names
    overall_results_metrics_df.columns = parsed_metrics.keys()

    # finally save the collected results data to file if an output file path was provided, and return the results dataframe
    if output_file is not None:
        try:
            overall_results_metrics_df.to_csv(output_file)
        except Exception as e:
            raise ClassifaiError(
                "Failed to save evaluation results to file.",
                context={"output_file": output_file, "cause_message": str(e)},
            ) from e
    return overall_results_metrics_df
