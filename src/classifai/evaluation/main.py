"""This module evaluates one or more `classifai.indexers.VectorStore` instances on a ground-truth labelled dataset.

1. Validating the ground-truth input with a Pandera schema.
2. Running a batched top-1 `VectorStore.search` over all queries.
3. Merging the ground-truth label into the retrieved results.
4. Validating the merged evaluation frame with a Pandera schema.
5. Validating chosen metrics with a metrics parsing function.
6. Computing one or more multiclass, single-label classification metrics.
7. Utilising an Evaluation class to manage the evaluation process, including saving results and providing access to individual metric results.

Evaluation is (currently with future updates pending) framed as retrieval-as-classification: for each query, the label of
the top retrieved document (`doc_label`) is treated as the model prediction, and the
provided dataset label is treated as the ground truth (`ground_truth_label`).

DataFrames:
    Ground-truth input (`ground_truths`) must include:
        - qid (str): Unique query identifier.
        - text (str): Query text.
        - label (str): Ground-truth label.

    Search evaluation output (`results_df`) is expected to include:
        - query_id (str): Query identifier (automatically generated for, and extracted from VectorStoreSearchInput dataclass).
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

Exceptions:
    InvalidMetricError: Raised when requested metric names cannot be parsed.
    EvaluationError: Raised when validation, vectorstore execution, result validation, or
        metric computation fails.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame, Series

from ..exceptions import ClassifaiError
from ..indexers import VectorStore
from ..indexers.dataclasses import VectorStoreSearchInput
from .metrics import (
    ClassificationAccuracy,
    ClassificationMacroF1,
    ClassificationMacroPrecision,
    ClassificationMacroRecall,
    Metric,
)

# - Error classes for evaluation-specific exceptions, inheriting from ClassifaiError


@dataclass(eq=False)
class InvalidMetricError(ClassifaiError):
    code: str = "invalid_metric_error"


@dataclass(eq=False)
class EvaluationError(ClassifaiError):
    code: str = "evaluation_error"


# - Pandera DataFrameModel declarations for validating input and output DataFrames


class GroundTruthSchema(pa.DataFrameModel):
    text: Series[str]
    label: Series[str]

    class Config:
        coerce = True


class SearchOutputSchema(pa.DataFrameModel):
    query_id: Series[str]
    query_text: Series[str]
    doc_label: Series[str]
    doc_text: Series[str]
    rank: Series[int] = pa.Field(ge=0)
    score: Series[float]
    ground_truth_label: Series[str]

    class Config:
        coerce = True


class MetricType(Enum):
    """Available classification metrics."""

    ACCURACY = ClassificationAccuracy
    MACRO_RECALL = ClassificationMacroRecall
    MACRO_PRECISION = ClassificationMacroPrecision
    MACRO_F1 = ClassificationMacroF1


def parse_metrics(metrics: list[str]) -> dict[str, Metric]:
    """Parse a list of metric names and return a dictionary mapping metric names to their corresponding functions.

    Args:
        metrics (list[str]): A list of metric names as strings.

    Returns:
        dict[str, Metric]: A dictionary where the keys are the metric names (as strings) and the values are
        the corresponding Metric instances.

    Raises:
        ValueError: If a metric name in the input list is invalid, an error is raised with a list of valid metrics.
    """
    parsed = {}
    for m in metrics:
        try:
            metric_type = MetricType[m.upper()]
            parsed[m] = metric_type.value()
        except KeyError as e:
            valid_metrics = [e.name.lower() for e in MetricType]
            raise ValueError(f"Invalid metric: {m}. Valid metrics are: {valid_metrics}") from e
    return parsed


class Evaluation:
    """Evaluation class for assessing the performance of vectorstores against ground truth data.

    This class provides methods to evaluate vectorstores using specified metrics, validate inputs,
    and save results. It supports batch processing and allows for detailed inspection of individual
    metric results.

    Attributes:
        ground_truths (pd.DataFrame): DataFrame containing 'qid', 'text', and 'label' columns.
        batch_size (int): Batch size for vectorstore search operations.
        save_output (bool): Whether to save evaluation results to a file.
        parsed_metrics (dict): Dictionary of parsed metrics to compute.
        metric_results (dict): Dictionary of individual metric results for detailed inspection.
    """

    @pa.check_types
    def __init__(
        self,
        ground_truths: DataFrame[GroundTruthSchema],
        metrics: list[str],
        batch_size: int = 8,
        save_output: bool = False,
    ):
        """Initialize evaluation configuration.

        This constructor is responsible for ensuring that the provided ground_truths
        DataFrame is compatible with the kinds of metrics that are to be calculated.

        Args:
            ground_truths: DataFrame with 'qid', 'text', 'label' columns.
            metrics: List of metric names to compute (e.g., ["accuracy", "macro_f1"]).
            batch_size: Batch size for vectorstore search operations.
            save_output: Whether to save results to file by default.

        Raises:
            InvalidMetricError: If the provided metrics cannot be parsed.
        """
        # assume grount truths has been validated in pa parameter checks, add a qid column to the ground truths and set object attributes
        self.ground_truths = ground_truths.copy()
        self.ground_truths["qid"] = self.ground_truths.index.astype(str)
        self.batch_size = batch_size
        self.save_output = save_output
        self.metric_results = {}

        # parse the provided metrics and store them in the instance
        try:
            self.parsed_metrics = parse_metrics(metrics)
        except Exception as e:
            raise InvalidMetricError(
                "Failed to parse provided metrics.",
                context={"metrics": metrics, "cause_message": str(e)},
            ) from e

    def evaluate(  # noqa: C901, PLR0912
        self,
        vectorstores: list[VectorStore | Callable[[], VectorStore]],
        vectorstore_names: list[str],
        output_file: str | None = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Evaluate multiple VectorStore instances on ground truth data and compute metrics.

        This method validates the input, evaluates each VectorStore instance or callable,
        computes metrics, and optionally saves the results to a CSV file.

        Args:
            vectorstores (list[VectorStore | Callable[[], VectorStore]]):
                A list of VectorStore instances or callables that return VectorStore instances.
            vectorstore_names (list[str]):
                A list of unique names corresponding to the VectorStore instances.
                The length must match the `vectorstores` list.
            output_file (str | None, optional):
                The file path to save the evaluation results as a CSV file.
                Must end with ".csv". If None, results are not saved unless `self.save_output` is True.
            overwrite (bool, optional):
                Whether to overwrite the output file if it already exists. Default is False.

        Returns:
            pd.DataFrame:
                A DataFrame containing the evaluation results for all VectorStore instances.

        Raises:
            ValueError:
                If input validation fails (e.g., mismatched lengths, invalid types, or duplicate names).
            EvaluationError:
                If any step of the evaluation process fails, such as instantiating a VectorStore,
                running a search, validating search results, or computing metrics.
            ClassifaiError:
                If saving the results to a file fails.

        Notes:
            - Each VectorStore instance or callable is processed sequentially.
            - Metrics are computed using `self.parsed_metrics`, and results are stored in `self.metric_results`.
        """
        # Validations

        # Check if output_file is provided and save_output is True, and handle overwrite check logic
        if self.save_output:
            output_file = output_file or "evaluation_results.csv"
            if os.path.exists(output_file) and not overwrite:
                raise ValueError(
                    f"Will attempt to save results to {output_file} but this file already exists!!! Set overwrite=True to overwrite."
                )

            # Ensure the folder exists
            folder_path = os.path.dirname(output_file)
            if folder_path and not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

        # are all in vectorstores either VectorStore instances or callables?
        invalid_items = [
            (i, vs) for i, vs in enumerate(vectorstores) if not isinstance(vs, VectorStore) and not callable(vs)
        ]
        if invalid_items:
            raise ValueError(
                f"All items in vectorstores must be instances of VectorStore or callables. "
                f"Invalid items: {invalid_items}"
            )

        # are vectorstore_names a list of length equal to vectorstores?
        if not isinstance(vectorstore_names, list) or len(vectorstore_names) != len(vectorstores):
            raise ValueError(
                "vectorstore_names must be a list matching vectorstores length. Length vectorStores = ",
                len(vectorstores),
                " Length vectorstore_names = ",
                len(vectorstore_names),
            )

        # are all vectorstore_names strings?
        invalid_names = [(i, name) for i, name in enumerate(vectorstore_names) if not isinstance(name, str)]
        if invalid_names:
            raise ValueError(f"All vectorstore_names must be strings. Invalid entries: {invalid_names}")

        # are all vectorstore_names unique?
        if len(set(vectorstore_names)) != len(vectorstore_names):
            raise ValueError("All vectorstore_names must be unique.")

        # is output_file a string ending with .csv if provided?
        if output_file and (not isinstance(output_file, str) or not output_file.strip().endswith(".csv")):
            raise ValueError("output_file must be a string and end with '.csv'.")

        # Run evaluation
        overall_results_df = pd.DataFrame()

        # Process each VectorStore instance or callable and corresponding name
        for vs, name in zip(vectorstores, vectorstore_names, strict=False):
            print(f"Processing VectorStore: {name}")

            # instantiate vectorstore from callable if appropriate
            try:
                resolved_vs = vs() if callable(vs) else vs
            except Exception as e:
                raise EvaluationError(
                    "Failed to instantiate VectorStore from callable.",
                    context={"vectorstore_name": name, "cause_message": str(e)},
                ) from e

            # run the search function for the current vectorstore across the ground truth queries
            try:
                # returned value should be automtically validated by Pandera DataFrameModel checks.
                results_df = self._run_search(resolved_vs)
            except Exception as e:
                raise EvaluationError(
                    "VectorStore search failed.",
                    context={"vectorstore_name": name, "cause_message": str(e)},
                ) from e
            finally:
                if callable(vs):
                    del resolved_vs

            # Compute metrics for the current VectorStore and store results
            try:
                for _metric_name, metric in self.parsed_metrics.items():
                    result = metric.evaluate(results_df)
                    self.metric_results[result.name] = result.value
            except Exception as e:
                raise EvaluationError(
                    "Metric computation failed.",
                    context={"vectorstore_name": name, "cause_message": str(e)},
                ) from e

            # Append the current VectorStore's metrics to the overall results DataFrame
            vectorstore_df = pd.DataFrame([self.metric_results], index=[name])
            overall_results_df = pd.concat([overall_results_df, vectorstore_df])

        # Save results to CSV if requested
        if self.save_output:
            try:
                # generate the output DataFrame with indexed vectorstore names as a column and save to CSV
                save_results_df = overall_results_df.reset_index()
                save_results_df.rename(columns={"index": "vectorstore_name"}, inplace=True)
                save_results_df.to_csv(output_file, index=False)
            except Exception as e:
                raise ClassifaiError(
                    "Failed to save results.",
                    context={"output_file": output_file, "cause_message": str(e)},
                ) from e

        return overall_results_df

    @pa.check_types
    def _run_search(self, vectorstore: VectorStore) -> DataFrame[SearchOutputSchema]:
        """Executes a search on the provided vector store using the ground truth data and returns a DataFrame containing the evaluation results.

        Args:
            vectorstore (VectorStore): The vector store instance to perform the search on.

        Returns:
            pd.DataFrame: A DataFrame containing the search results merged with the ground truth data.
                          The resulting DataFrame includes the following columns:
                          - query_id: The ID of the query.
                          - query: The text of the query.
                          - ground_truth_label: The corresponding ground truth label for the query.
        """
        # build the VectorStoreSearchInput from the ground_truths dataframe
        search_input = VectorStoreSearchInput(
            {
                "id": self.ground_truths["qid"].tolist(),
                "query": self.ground_truths["text"].tolist(),
            }
        )

        # run the search with the vectorstore
        result_df = vectorstore.search(search_input, n_results=1, batch_size=self.batch_size)

        # merge the ground truth labels into the result dataframe
        eval_data = result_df.merge(
            self.ground_truths[["qid", "label"]],
            left_on="query_id",
            right_on="qid",
            how="left",
        )
        eval_data = eval_data.rename(columns={"label": "ground_truth_label"})
        return eval_data
