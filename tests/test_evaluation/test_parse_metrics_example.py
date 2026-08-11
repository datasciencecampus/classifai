"""Unit tests for parse_metrics function."""

from __future__ import annotations

import pytest

from classifai.evaluation.main import parse_metrics
from classifai.evaluation.metrics import (
    ClassificationAccuracy,
    ClassificationMacroF1,
    ClassificationMacroPrecision,
    ClassificationMacroRecall,
    Metric,
)


class TestParseMetricsValidInput:
    """Tests for parse_metrics with valid inputs."""

    def test_parse_single_valid_metric(self):
        """Test parsing a single valid metric name."""
        result = parse_metrics(["accuracy"])

        assert isinstance(result, dict)
        assert len(result) == 1
        assert "accuracy" in result
        assert isinstance(result["accuracy"], ClassificationAccuracy)

    def test_parse_multiple_valid_metrics(self):
        """Test parsing multiple valid metric names."""
        result = parse_metrics(["accuracy", "macro_recall", "macro_precision", "macro_f1"])

        NEXT_ANSWER = 4
        assert len(result) == NEXT_ANSWER
        assert "accuracy" in result
        assert "macro_recall" in result
        assert "macro_precision" in result
        assert "macro_f1" in result
        assert isinstance(result["accuracy"], ClassificationAccuracy)
        assert isinstance(result["macro_recall"], ClassificationMacroRecall)
        assert isinstance(result["macro_precision"], ClassificationMacroPrecision)
        assert isinstance(result["macro_f1"], ClassificationMacroF1)

    def test_parse_metrics_case_insensitive(self):
        """Test that metric names are case insensitive."""
        result_lower = parse_metrics(["accuracy"])
        result_upper = parse_metrics(["ACCURACY"])
        result_mixed = parse_metrics(["Accuracy"])

        assert "accuracy" in result_lower
        assert "ACCURACY" in result_upper
        assert "Accuracy" in result_mixed
        assert isinstance(result_lower["accuracy"], ClassificationAccuracy)
        assert isinstance(result_upper["ACCURACY"], ClassificationAccuracy)
        assert isinstance(result_mixed["Accuracy"], ClassificationAccuracy)

    def test_parse_metrics_dict_keys_match_input_names(self):
        """Test that dict keys match the input metric names exactly."""
        result = parse_metrics(["ACCURACY", "Macro_Recall"])

        # Keys should match the case of input
        assert "ACCURACY" in result
        assert "Macro_Recall" in result

    def test_parse_metrics_returns_metric_instances(self):
        """Test that returned values are Metric instances."""
        result = parse_metrics(["accuracy", "macro_f1"])

        for metric in result.values():
            assert isinstance(metric, Metric)
            assert hasattr(metric, "evaluate")
            assert callable(metric.evaluate)

    def test_parse_empty_list(self):
        """Test parsing an empty metric list."""
        result = parse_metrics([])

        assert isinstance(result, dict)
        assert len(result) == 0


class TestParseMetricsInvalidInput:
    """Tests for parse_metrics with invalid inputs."""

    def test_parse_invalid_metric_name_raises_error(self):
        """Test that invalid metric name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_metrics(["invalid_metric"])

        error_msg = str(exc_info.value)
        assert "invalid_metric" in error_msg.lower()
        assert "Invalid metric" in error_msg

    def test_parse_invalid_metric_error_includes_valid_metrics(self):
        """Test that error message includes list of valid metrics."""
        with pytest.raises(ValueError) as exc_info:
            parse_metrics(["wrong_name"])

        error_msg = str(exc_info.value)
        assert "accuracy" in error_msg.lower()
        assert "macro_recall" in error_msg.lower() or "recall" in error_msg.lower()
        assert "macro_precision" in error_msg.lower() or "precision" in error_msg.lower()
        assert "macro_f1" in error_msg.lower() or "f1" in error_msg.lower()

    def test_parse_typo_in_metric_name_raises_error(self):
        """Test that typos in metric names are caught."""
        with pytest.raises(ValueError):
            parse_metrics(["accuraccy"])  # typo

        with pytest.raises(ValueError):
            parse_metrics(["macro_recal"])  # typo

        with pytest.raises(ValueError):
            parse_metrics(["f1_macro"])  # wrong order

    def test_parse_empty_string_metric_raises_error(self):
        """Test that empty string metric name raises error."""
        with pytest.raises(ValueError):
            parse_metrics([""])

    def test_parse_whitespace_only_metric_raises_error(self):
        """Test that whitespace-only metric name raises error."""
        with pytest.raises(ValueError):
            parse_metrics(["   "])

    def test_parse_invalid_metric_in_list_with_valid_ones(self):
        """Test that invalid metric in list with valid ones raises error."""
        with pytest.raises(ValueError) as exc_info:
            parse_metrics(["accuracy", "invalid_metric", "macro_f1"])

        error_msg = str(exc_info.value)
        assert "invalid_metric" in error_msg.lower()

    def test_parse_invalid_metric_preserves_partial_results_in_error(self):
        """Test that error includes context about the failing metric."""
        with pytest.raises(ValueError) as exc_info:
            parse_metrics(["accuracy", "bad_metric"])

        # Error should reference the bad metric, even if accuracy was already parsed
        assert "bad_metric" in str(exc_info.value).lower()


class TestParseMetricsEdgeCases:
    """Tests for edge cases in parse_metrics."""

    def test_parse_duplicate_metric_names(self):
        """Test handling of duplicate metric names."""
        result = parse_metrics(["accuracy", "accuracy"])

        # Should return dict with single entry (dict keys are unique)
        # or allow duplicates? Implementation dependent
        assert "accuracy" in result
        assert isinstance(result["accuracy"], ClassificationAccuracy)

    def test_parse_all_available_metrics(self):
        """Test parsing all available metrics at once."""
        all_metrics = ["accuracy", "macro_recall", "macro_precision", "macro_f1"]
        expected_metric_count = len(all_metrics)
        result = parse_metrics(all_metrics)

        assert len(result) == expected_metric_count
        for metric_name in all_metrics:
            assert metric_name in result

    def test_parse_mixed_case_metrics(self):
        """Test parsing metrics with various case combinations."""
        result = parse_metrics(["ACCURACY", "Macro_Recall", "mACRO_PRECISION", "macro_F1"])

        NEXT_ANSWER = 4
        assert len(result) == NEXT_ANSWER
        assert "ACCURACY" in result
        assert "Macro_Recall" in result
        assert "mACRO_PRECISION" in result
        assert "macro_F1" in result

    def test_parse_metrics_with_underscores(self):
        """Test that metric names with underscores are handled correctly."""
        result = parse_metrics(["macro_recall", "macro_precision", "macro_f1"])

        assert "macro_recall" in result
        assert "macro_precision" in result
        assert "macro_f1" in result

    def test_parse_metrics_with_spaces_raises_error(self):
        """Test that metric names with spaces are invalid."""
        with pytest.raises(ValueError):
            parse_metrics(["macro recall"])  # space in name

        with pytest.raises(ValueError):
            parse_metrics(["accuracy "])  # trailing space
