"""Module for evaluating classifAI results."""

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class EvaluationResult:
    """Structured evaluation results.

    Attributes
    ----------
        mean_rank: Average position of correct answers in results
        total_samples: Total number of test cases evaluated
        found_ratio: Proportion of correct answers found in results
        rank_distribution: Distribution of ranks for found answers
        not_found_count: Number of correct answers not found
        mean_distance: Average distance across all results
        mean_distance_correct: Average distance for correct matches
    """

    mean_rank: float
    total_samples: int
    found_ratio: float
    rank_distribution: Dict[int, int]
    not_found_count: int
    mean_distance: Optional[float] = None
    mean_distance_correct: Optional[float] = None


class SICAssignmentEvaluator:
    """Evaluate results from API against the validated dataset."""

    def __init__(self):
        """Initialize evaluator with API results."""
        self.api_results = None
        self.validation_set = None

    def load_classifai_results(self, api_results: Dict[str, Any]):
        """Get api results.

        Args:
            api_results: Dictionary containing SIC candidates

        Raises
        ------
            ValueError: If api_results is None or empty
        """

        if not api_results:
            raise ValueError("API results cannot be None or empty")

        self.api_results = api_results
        self._results_lookup = self._create_results_lookup()

    def _create_results_lookup(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create lookup dictionary for faster access to results.

        Returns
        -------
            Dictionary mapping input IDs to their corresponding predictions

        Note:
            The output format is expected to be:
            {
                'data': [
                    {
                        'input_id': 'id1',
                        'response': [prediction1, prediction2, ...]
                    },
                    ...
                ]
            }
        """
        try:
            results_dict = {}
            # Iterate through each item in the data list
            for item in self.api_results.get("data", []):
                # For each input_id in the item, add the corresponding response list
                results_dict[item["input_id"]] = item["response"]
            return results_dict
        except Exception as e:
            print(f"Failed to create results lookup: {str(e)}")
            raise ValueError("Failed to process API results format") from e

    def _process_single_case(
        self,
        correct_code: str,
        responses: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], List[float], Optional[float]]:
        """Process a single test case.

        Args:
            input_id: ID of the test case (as string)
            correct_code: Expected correct code
            responses: List of API responses for this case

        Returns
        -------
            Tuple of (rank of correct answer, all distances, distance of correct answer)
        """
        # Each response entry contains information for all input IDs
        distances = [resp["distance"] for resp in responses]

        for rank, resp in enumerate(responses, start=1):
            if resp["label"] == correct_code:
                return rank, distances, resp["distance"]

        return None, distances, None

    def evaluate_rankings(
        self,
        test_data: pd.DataFrame,
        id_column: str,
        correct_code_column: str,
        description_column: str,
    ) -> EvaluationResult:
        """Calculate rank of validated answers in classifAI results.

        Args:
            test_data: DataFrame containing test cases
            id_column: Name of the column containing input IDs
            correct_code_column: Name of the column containing correct codes
            description_column: Name of the column containing descriptions

        Returns
        -------
            EvaluationResult object containing evaluation metrics
        """
        ranks = []
        rank_distribution = defaultdict(int)
        not_found_count = 0
        all_distances = []
        correct_distances = []

        for _, row in test_data.iterrows():
            try:
                input_id = row[id_column]
                correct_code = row[correct_code_column]

                responses = self._results_lookup.get(input_id)
                if not responses:
                    print(f"No predictions found for input_id {input_id}")
                    not_found_count += 1
                    continue

                rank, distances, correct_distance = self._process_single_case(
                    correct_code, responses
                )

                all_distances.extend(distances)

                if rank:
                    ranks.append(rank)
                    rank_distribution[rank] += 1
                    if correct_distance is not None:
                        correct_distances.append(correct_distance)
                else:
                    not_found_count += 1

            except Exception as e:
                print(f"Error processing row {input_id}: {str(e)}")
                continue

        total_samples = len(test_data)

        return EvaluationResult(
            mean_rank=sum(ranks) / len(ranks) if ranks else float("inf"),
            total_samples=total_samples,
            found_ratio=len(ranks) / total_samples,
            rank_distribution=dict(rank_distribution),
            not_found_count=not_found_count,
            mean_distance=sum(all_distances) / len(all_distances)
            if all_distances
            else None,
            mean_distance_correct=sum(correct_distances)
            / len(correct_distances)
            if correct_distances
            else None,
        )

    def generate_evaluation_report(
        self, eval_result: EvaluationResult
    ) -> pd.DataFrame:
        """Generate a detailed evaluation report."""
        metrics = {
            "Metric": [
                "Mean Rank",
                "Total Samples",
                "Found Ratio",
                "Not Found Count",
                "Mean Distance (All Results)",
                "Mean Distance (Correct Matches)",
            ],
            "Value": [
                f"{eval_result.mean_rank:.2f}",
                eval_result.total_samples,
                f"{eval_result.found_ratio:.2%}",
                eval_result.not_found_count,
                f"{eval_result.mean_distance:.4f}"
                if eval_result.mean_distance
                else "N/A",
                f"{eval_result.mean_distance_correct:.4f}"
                if eval_result.mean_distance_correct
                else "N/A",
            ],
        }

        return pd.DataFrame(metrics)

    def find_missed_cases(
        self,
        test_data: pd.DataFrame,
        id_column: str,
        correct_code_column: str,
        description_column: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Find cases where the correct answer was not in the candidate set.

        Args:
            test_data: DataFrame containing test cases
            id_column: Name of the column containing input IDs
            correct_code_column: Name of the column containing correct codes
            description_column: Name of the column containing descriptions

        Returns
        -------
            Tuple of (DataFrame containing missed cases, DataFrame containing skipped cases)
        """
        missed_cases = []
        skipped_cases = []

        for _, row in test_data.iterrows():
            try:
                input_id = row[id_column]
                description = row[description_column]

                # Skip if correct code is not numeric
                if not pd.api.types.is_numeric_dtype(
                    type(row[correct_code_column])
                ):
                    skipped_cases.append(
                        {
                            "input_id": input_id,
                            "description": description,
                            "correct_code": row[correct_code_column],
                            "reason": "Non-numeric code",
                        }
                    )
                    continue

                correct_code = int(row[correct_code_column])

                # Check if we have predictions for this ID
                predictions = self._results_lookup.get(input_id)
                if not predictions:
                    print(f"No predictions found for input_id {input_id}")
                    continue

                predicted_codes = [int(r["label"]) for r in predictions]

                # If correct code is not in predictions
                if correct_code not in predicted_codes:
                    case_data = self._prepare_missed_case_data(
                        input_id, description, correct_code, predictions
                    )
                    missed_cases.append(case_data)

            except Exception as e:
                print(
                    f"Error processing row with input_id {row[id_column]}: {str(e)}"
                )
                continue

        if skipped_cases:
            print(
                f"\nSkipped {len(skipped_cases)} cases due to non-numeric codes"
            )
            # skipped_df = pd.DataFrame(skipped_cases)
            # skipped_df.to_csv('skipped_cases.csv', index=False)
            # print("Saved skipped cases to 'skipped_cases.csv'")

        return (
            pd.DataFrame(missed_cases) if missed_cases else pd.DataFrame(),
            pd.DataFrame(skipped_cases) if skipped_cases else pd.DataFrame(),
        )

    @staticmethod
    def _prepare_missed_case_data(
        input_id: int,
        description: str,
        correct_code: int,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Prepare data for a missed case.

        Args:
            input_id: ID of the test case
            description: Description text
            correct_code: Expected correct code
            predictions: List of predictions for this case

        Returns
        -------
            Dictionary containing formatted case data
        """
        case_data = {
            "input_id": input_id,
            "description": description,
            "correct_code": correct_code,
            "correct_code_2d": int(str(correct_code)[:2]),
        }

        # Add all predictions with their distances
        for rank, pred in enumerate(predictions, 1):
            pred_code = pred["label"]
            case_data.update(
                {
                    f"predicted_code_rank{rank}": pred_code,
                    f"predicted_code_2d_rank{rank}": int(str(pred_code)[:2]),
                    f"distance_rank{rank}": pred["distance"],
                    f"prediction_description_rank{rank}": pred["description"],
                }
            )

        return case_data

    @staticmethod
    def save_eval_result(
        metadata: dict,
        eval_result: EvaluationResult,
        processed_result: Optional[dict] = None,
        skipped_cases: Optional[pd.DataFrame] = None,
        missed_cases: Optional[pd.DataFrame] = None,
    ) -> str:
        """Save evaluation results to files.

        Args:
            metadata: Dictionary of metadata parameters
            eval_result: EvaluationResult object containing evaluation metrics
            processed_result: Dictionary containing full query search results (optional)
            skipped_cases: DataFrame containing skipped cases (optional)
            missed_cases: DataFrame containing missed cases (optional)

        Returns
        -------
            str: The folder path where results were stored
        """
        if not metadata:
            raise ValueError("Metadata dictionary cannot be empty")

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

        # Create folder name
        folder_name = f"outputs/{formatted_datetime}_{metadata.get('evaluation_type', 'unnamed')}"

        # Create directory safely
        os.makedirs(folder_name, exist_ok=True)

        # Save metadata
        metadata_path = os.path.join(folder_name, "metadata.json")
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile, indent=4)

        # Convert EvaluationResult to dictionary and handle float values
        eval_result_dict = {
            "mean_rank": float(eval_result.mean_rank),
            "total_samples": eval_result.total_samples,
            "found_ratio": float(eval_result.found_ratio),
            "rank_distribution": {
                str(k): v for k, v in eval_result.rank_distribution.items()
            },
            "not_found_count": eval_result.not_found_count,
            "mean_distance": float(eval_result.mean_distance)
            if eval_result.mean_distance is not None
            else None,
            "mean_distance_correct": float(eval_result.mean_distance_correct)
            if eval_result.mean_distance_correct is not None
            else None,
        }

        # Save evaluation result
        eval_path = os.path.join(folder_name, "evaluation_result.json")
        with open(eval_path, "w") as outfile:
            json.dump(eval_result_dict, outfile, indent=4)

        if processed_result is not None:
            processed_result_path = os.path.join(
                folder_name, "processed_result.json"
            )
            with open(processed_result_path, "w") as outfile:
                json.dump(processed_result, outfile, indent=4)
        else:
            print("Skipped saving full results")

        # Save skipped cases if any
        if skipped_cases is not None and not skipped_cases.empty:
            skipped_path = os.path.join(folder_name, "skipped_cases.csv")
            skipped_cases.to_csv(skipped_path, index=False)
        else:
            print("No skipped cases to save")

        # Save missed cases if any
        if missed_cases is not None and not missed_cases.empty:
            missed_path = os.path.join(folder_name, "missed_cases.csv")
            missed_cases.to_csv(missed_path, index=False)
        else:
            print("No missed cases to save")

        print(f"Successfully saved all outputs to {folder_name}")
        return folder_name


@dataclass
class ComparisonResult:
    """Store comparison metrics between two methods.

    Attributes
    ----------
        exact_match_top1: Proportion of G-Code agreements for top classifai result
        exact_match_top5: Proportion of G-Code agreements for any in top 5 classifai results
        digit2_match_top1: Proportion of G-Code agreements at 2-d for top classifai result
        digit2_match_top5: Proportion of G-Code agreements at 2-d for any in top 5 classifai results
        total_samples: Number of samples
        confusion_matrix_2digit: Dataframe with confusion matrix
    """

    exact_match_top1: float
    exact_match_top5: float
    digit2_match_top1: float
    digit2_match_top5: float
    total_samples: int
    confusion_matrix_2digit: Optional[pd.DataFrame] = None


class MethodComparison:
    """Compare G-Code and classifAI results."""

    def __init__(self):
        """Initialize the comparison class."""
        self.classifai = None
        self.g_code = None

    def load_classifai_results(self, api_results: Dict[str, Any]):
        """Get api results.

        Args:
            api_results): Dictionary containing SIC candidates

        Raises
        ------
            ValueError: If api_results is None or empty
        """

        if not api_results:
            raise ValueError("API results cannot be None or empty")

        self.classifai = api_results

    def load_g_code_results(
        self, df: pd.DataFrame, code_column: str, id_column: str
    ) -> None:
        """
        Load results from G-Code (DataFrame with single predictions).

        Args:
            df: DataFrame containing G-Code results
            code_column: Name of the column containing the 5-digit codes
            id_column: Name of the column containing input IDs
        """
        self.g_code = df[[id_column, code_column]].copy()
        self.g_code[id_column] = pd.to_numeric(
            self.g_code[id_column], errors="raise"
        ).astype(int)
        self.g_code.columns = ["id", "sic_5d"]

    def get_2digit_code(self, code: int) -> int:
        """Extract 2-digit code from 5-digit code."""
        return int(str(code)[:2])

    def compare_methods(self) -> ComparisonResult:
        """
        Compare methods at both 5-digit and 2-digit levels.

        Returns
        -------
            ComparisonResult object containing various comparison metrics
        """
        if self.classifai is None or self.g_code is None:
            raise ValueError(
                "Both method results must be loaded before comparison"
            )

        total_samples = 0
        exact_match_top1 = 0
        exact_match_top5 = 0
        digit2_match_top1 = 0
        digit2_match_top5 = 0

        # For confusion matrix
        true_2digit = []
        pred_2digit = []

        # Process each sample
        for item in self.classifai["data"]:
            input_id = item["input_id"]

            # Get method B's prediction for this input
            g_code_pred = self.g_code[self.g_code["id"] == input_id][
                "sic_5d"
            ].iloc[0]

            # Get method A's predictions
            classifai_preds = [r["label"] for r in item["response"]]

            # 5-digit comparisons
            if classifai_preds[0] == g_code_pred:
                exact_match_top1 += 1
            if g_code_pred in classifai_preds:
                exact_match_top5 += 1

            # 2-digit comparisons
            g_code_2digit_pred = self.get_2digit_code(g_code_pred)
            classifai_2digit_preds = [
                self.get_2digit_code(code) for code in classifai_preds
            ]

            true_2digit.append(g_code_2digit_pred)
            pred_2digit.append(self.get_2digit_code(classifai_preds[0]))

            if classifai_2digit_preds[0] == g_code_2digit_pred:
                digit2_match_top1 += 1
            if g_code_2digit_pred in classifai_2digit_preds:
                digit2_match_top5 += 1

            total_samples += 1

        # Calculate confusion matrix for 2-digit codes
        unique_2digit_codes = sorted(list(set(true_2digit + pred_2digit)))
        conf_matrix = confusion_matrix(
            true_2digit, pred_2digit, labels=unique_2digit_codes
        )

        # Convert to DataFrame for better visualization
        conf_df = pd.DataFrame(
            conf_matrix, index=unique_2digit_codes, columns=unique_2digit_codes
        )

        return ComparisonResult(
            exact_match_top1=exact_match_top1 / total_samples,
            exact_match_top5=exact_match_top5 / total_samples,
            digit2_match_top1=digit2_match_top1 / total_samples,
            digit2_match_top5=digit2_match_top5 / total_samples,
            total_samples=total_samples,
            confusion_matrix_2digit=conf_df,
        )

    def generate_report(self, result: ComparisonResult) -> pd.DataFrame:
        """
        Generate a summary report of the comparison.

        Args:
            result: ComparisonResult object from compare_methods()

        Returns
        -------
            DataFrame containing formatted metrics
        """
        metrics = {
            "Metric": [
                "5-Digit Exact Match (Top 1)",
                "5-Digit Exact Match (Top 5)",
                "2-Digit Match (Top 1)",
                "2-Digit Match (Top 5)",
                "Total Samples",
            ],
            "Value": [
                f"{result.exact_match_top1:.2%}",
                f"{result.exact_match_top5:.2%}",
                f"{result.digit2_match_top1:.2%}",
                f"{result.digit2_match_top5:.2%}",
                result.total_samples,
            ],
        }

        return pd.DataFrame(metrics)

    def plot_confusion_matrix(
        self,
        result: ComparisonResult,
        figsize: Tuple[int, int] = (12, 10),
        save_matrix=False,
        output_path=None,
    ) -> None:
        """
        Plot confusion matrix for 2-digit code comparison.

        Args:
            result: ComparisonResult object from compare_methods()
            figsize: Tuple of (width, height) for the plot
        """
        plt.figure(figsize=figsize)
        sns.heatmap(
            result.confusion_matrix_2digit,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            square=True,
        )
        plt.title("2-Digit Code Confusion Matrix")
        plt.xlabel("Predicted Code (ClassifAI)")
        plt.ylabel("True Code (G-Code)")
        plt.tight_layout()
        if save_matrix:
            plt.savefig(output_path)
        else:
            plt.show()

    def save_comparison_result(
        self,
        metadata: dict,
        comparison_result: ComparisonResult,
        processed_result: Optional[dict] = None,
        save_confusion_matrix=True,
    ) -> str:
        """Save evaluation results to files.

        Args:
            metadata: Dictionary of metadata parameters
            eval_result: EvaluationResult object containing comparison metrics
            processed_result: Dictionary containing full query search results (optional)
            save_confusion_matrix: Boolean, option to save the matrix

        Returns
        -------
            str
        """
        if not metadata:
            raise ValueError("Metadata dictionary cannot be empty")

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

        # Create folder name
        folder_name = f"outputs/{formatted_datetime}_{metadata.get('evaluation_type', 'unnamed')}"

        # Create directory safely
        os.makedirs(folder_name, exist_ok=True)

        # Save metadata
        metadata_path = os.path.join(folder_name, "metadata.json")
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile, indent=4)

        # Convert ComparisonResult to dictionary and handle float values
        comparison_result_dict = {
            "5-Digit Exact Match (Top 1)": float(
                comparison_result.exact_match_top1
            ),
            "5-Digit Exact Match (Top 5)": float(
                comparison_result.exact_match_top5
            ),
            "2-Digit Match (Top 1)": float(
                comparison_result.digit2_match_top1
            ),
            "2-Digit Match (Top 5)": float(
                comparison_result.digit2_match_top5
            ),
            "Total Samples": comparison_result.total_samples,
        }

        # Save evaluation result
        comparison_path = os.path.join(folder_name, "comparison_result.json")
        with open(comparison_path, "w") as outfile:
            json.dump(comparison_result_dict, outfile, indent=4)

        if processed_result is not None:
            processed_result_path = os.path.join(
                folder_name, "processed_result.json"
            )
            with open(processed_result_path, "w") as outfile:
                json.dump(processed_result, outfile, indent=4)
        else:
            print("Skipped saving full results")

        # Save skipped cases if any
        if save_confusion_matrix:
            self.plot_confusion_matrix(
                comparison_result,
                save_matrix=True,
                output_path=os.path.join(folder_name, "confusion_matrix.png"),
            )

        print(f"Successfully saved all outputs to {folder_name}")
        return folder_name


class LabelAccuracy:
    """Analyse classification accuracy for scenarios where model predictions can match any of multiple ground truth labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        id_col: str = "id",
        desc_col: str = "description",
        model_label_cols: List[str] = ["model_label_1", "model_label_2"],
        model_score_cols: List[str] = ["model_score_1", "model_score_2"],
        clerical_label_cols: List[str] = [
            "clerical_label_1",
            "clerical_label_2",
        ],
    ):
        """
        Initialize with a DataFrame containing model predictions and ground truth labels.

        Args:
            df: DataFrame with prediction and ground truth data
            id_col: Name of ID column
            desc_col: Name of description column
            model_label_cols: List of column names containing model predictions
            model_score_cols: List of column names containing confidence scores
            clerical_label_cols: List of column names containing ground truth labels
        """
        self.id_col = id_col
        self.desc_col = desc_col
        self.model_label_cols = model_label_cols
        self.model_score_cols = model_score_cols
        self.clerical_label_cols = clerical_label_cols

        # Verify all required columns exist
        required_cols = (
            [id_col, desc_col]
            + model_label_cols
            + model_score_cols
            + clerical_label_cols
        )
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Verify matching lengths of label and score columns
        if len(model_label_cols) != len(model_score_cols):
            raise ValueError(
                "Number of model label columns must match number of score columns"
            )

        self.df = df.copy()
        self._add_derived_columns()

    def _add_derived_columns(self):
        """Add computed columns for analysis."""
        # Get highest confidence score among all predictions
        self.df["max_score"] = self.df[self.model_score_cols].max(axis=1)

        # Check if any model prediction matches any clerical label
        def check_matches(row):
            model_labels = [row[col] for col in self.model_label_cols]
            clerical_labels = [row[col] for col in self.clerical_label_cols]
            return any(pred in clerical_labels for pred in model_labels)

        self.df["is_correct"] = self.df.apply(check_matches, axis=1)

    def get_accuracy(self, threshold: float = 0.0) -> float:
        """
        Calculate accuracy for predictions above the given confidence threshold.

        Args:
            threshold: Minimum confidence score threshold (default: 0.0)

        Returns
        -------
            float: Accuracy as a percentage
        """
        filtered_df = self.df[self.df["max_score"] >= threshold]
        if len(filtered_df) == 0:
            return 0.0
        return 100 * filtered_df["is_correct"].mean()

    def get_coverage(self, threshold: float = 0.0) -> float:
        """
        Calculate percentage of predictions above the given confidence threshold.

        Args:
            threshold: Minimum confidence score threshold (default: 0.0)

        Returns
        -------
            float: Coverage as a percentage
        """
        return 100 * (self.df["max_score"] >= threshold).mean()

    def get_threshold_stats(
        self, thresholds: List[float] = None
    ) -> pd.DataFrame:
        """
        Calculate accuracy and coverage across multiple thresholds.

        Args:
            thresholds: List of threshold values to evaluate (default: None)

        Returns
        -------
            DataFrame with columns: threshold, accuracy, coverage
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21)

        stats = []
        for threshold in thresholds:
            stats.append(
                {
                    "threshold": threshold,
                    "accuracy": self.get_accuracy(threshold),
                    "coverage": self.get_coverage(threshold),
                }
            )

        return pd.DataFrame(stats)

    def plot_threshold_curves(
        self,
        thresholds: List[float] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """
        Plot accuracy and coverage curves against confidence threshold.

        Args:
            thresholds: List of threshold values to evaluate (default: None)
            figsize: Figure size as (width, height) tuple (default: (10, 6))
        """
        stats_df = self.get_threshold_stats(thresholds)

        plt.figure(figsize=figsize)
        plt.plot(
            stats_df["threshold"],
            stats_df["coverage"],
            label="Coverage",
            color="blue",
        )
        plt.plot(
            stats_df["threshold"],
            stats_df["accuracy"],
            label="Accuracy",
            color="orange",
        )

        plt.xlabel("Confidence threshold")
        plt.ylabel("Percentage")
        plt.grid(True)
        plt.legend()
        plt.title("Coverage and Accuracy vs Confidence Threshold")
        plt.tight_layout()
        plt.show()

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics for the classification results.

        Returns
        -------
            Dictionary containing various summary statistics
        """
        return {
            "total_samples": len(self.df),
            "overall_accuracy": self.get_accuracy(),
            "accuracy_above_0.50": self.get_accuracy(0.5),
            "accuracy_above_0.60": self.get_accuracy(0.6),
            "accuracy_above_0.70": self.get_accuracy(0.7),
            "accuracy_above_0.80": self.get_accuracy(0.8),
            "coverage_above_0.50": self.get_coverage(0.5),
            "coverage_above_0.60": self.get_coverage(0.6),
            "coverage_above_0.70": self.get_coverage(0.7),
            "coverage_above_0.80": self.get_coverage(0.8),
        }

    @staticmethod
    def save_output(
        metadata: dict,
        eval_result: dict,
    ) -> str:
        """Save evaluation results to files.

        Args:
            metadata: Dictionary of metadata parameters
            eval_result: Dictionary containing evaluation metrics

        Returns
        -------
            str: The folder path where results were stored
        """
        if not metadata:
            raise ValueError("Metadata dictionary cannot be empty")

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

        # Create folder name
        folder_name = f"outputs/{formatted_datetime}_{metadata.get('evaluation_type', 'unnamed')}"

        # Create directory safely
        os.makedirs(folder_name, exist_ok=True)

        # Save metadata
        metadata_path = os.path.join(folder_name, "metadata.json")
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile, indent=4)

        # Save evaluation result
        eval_path = os.path.join(folder_name, "evaluation_result.json")
        with open(eval_path, "w") as outfile:
            json.dump(eval_result, outfile, indent=4)

        print(f"Successfully saved all outputs to {folder_name}")
        return folder_name
