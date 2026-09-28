"""Unit tests for FastEmbedVectoriser."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from classifai.exceptions import ExternalServiceError
from classifai.vectorisers import FastEmbedVectoriser


class TestFastEmbedVectoriserInitialization:
    """Tests for FastEmbedVectoriser initialization."""

    @patch("classifai.vectorisers.fastembed.check_deps")
    def test_init_missing_dependencies_raises_error(self, mock_check_deps):
        """Missing fastembed should raise the error surfaced by check_deps."""
        mock_check_deps.side_effect = ImportError("fastembed not installed")

        with pytest.raises(ImportError):
            FastEmbedVectoriser(model_name="sentence-transformers/all-MiniLM-L6-v2")

        mock_check_deps.assert_called_once_with(["fastembed"], extra="fastembed")

    @patch("classifai.vectorisers.fastembed.check_deps")
    @patch("fastembed.TextEmbedding")
    def test_init_valid_model_loads_successfully(self, mock_textembedding, mock_check_deps):
        """A valid model name should load tokenizer and model without raising."""
        vectoriser = FastEmbedVectoriser(model_name="sentence-transformers/all-MiniLM-L6-v2")

        assert vectoriser.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert vectoriser.model is mock_textembedding.return_value

    @patch("classifai.vectorisers.fastembed.check_deps")
    @patch("fastembed.TextEmbedding")
    def test_init_invalid_model_name_raises_external_service_error(self, mock_textembedding, mock_check_deps):
        """A failure loading the model should raise ExternalServiceError."""
        mock_textembedding.side_effect = OSError("model not found")

        with pytest.raises(ExternalServiceError):
            FastEmbedVectoriser(model_name="not-a-real-model")

    @patch("classifai.vectorisers.fastembed.check_deps")
    @patch("fastembed.TextEmbedding")
    def test_init_custom_model_kwargs_passed_through(self, mock_textembedding, mock_check_deps):
        """Custom model_kwargs should be forwarded to TextEmbedding."""
        FastEmbedVectoriser(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"cache_dir": "~/fastembed_models/"},
        )

        _, call_kwargs = mock_textembedding.call_args
        assert call_kwargs["cache_dir"] == "~/fastembed_models/"

    @patch("classifai.vectorisers.fastembed.check_deps")
    @patch("fastembed.TextEmbedding")
    def test_init_specific_model_path_passed_through(self, mock_textembedding, mock_check_deps):
        """The specific_model_path should be forwarded to TextEmbedding."""
        FastEmbedVectoriser(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            specific_model_path="~/fastembed_models/all-MiniLM-L6-v2",
        )

        _, call_kwargs = mock_textembedding.call_args
        assert call_kwargs["specific_model_path"] == "~/fastembed_models/all-MiniLM-L6-v2"


class TestFastEmbedVectoriserTransform:
    """Tests for FastEmbedVectoriser transform method."""

    @pytest.fixture
    def mock_vectoriser(self):
        """Return a FastEmbed instance with check_deps mocked out."""
        with (
            patch("classifai.vectorisers.fastembed.check_deps"),
            patch("fastembed.TextEmbedding"),
        ):
            vectoriser = FastEmbedVectoriser(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Replace with controllable mocks for the transform tests.
            vectoriser.model = Mock()

            yield vectoriser

    def _configure_successful_embed(self, vectoriser, n_texts, dim=4):
        """Configure FastEmbed to return one embedding row per input text."""
        vectoriser.model.embed.return_value = iter([[0.1] * dim for _ in range(n_texts)])

    def test_transform_single_string_converts_to_list_and_processes(self, mock_vectoriser):
        """A single string input should be wrapped in a list and produce one embedding row."""
        self._configure_successful_embed(mock_vectoriser, n_texts=1)

        result = mock_vectoriser.transform("hello world")

        assert result.shape[0] == 1
        call_args, _ = mock_vectoriser.model.embed.call_args
        assert call_args[0] == ["hello world"]

    def test_transform_list_of_strings_processes_correctly(self, mock_vectoriser):
        """A list of strings should be passed through unchanged."""
        self._configure_successful_embed(mock_vectoriser, n_texts=3)

        mock_vectoriser.transform(["a", "b", "c"])

        call_args, _ = mock_vectoriser.model.embed.call_args
        assert call_args[0] == ["a", "b", "c"]

    def test_transform_returns_2d_numpy_array(self, mock_vectoriser):
        """Output should be a 2D numpy array."""
        self._configure_successful_embed(mock_vectoriser, n_texts=2)

        result = mock_vectoriser.transform(["a", "b"])

        assert result.ndim == 2

    def test_transform_output_shape_matches_input_count(self, mock_vectoriser):
        """Number of output rows should match number of input texts."""
        self._configure_successful_embed(mock_vectoriser, n_texts=4)

        result = mock_vectoriser.transform(["a", "b", "c", "d"])
        print(result)

        assert result.shape[0] == 4


class TestFastEmbedVectoriserListSupportedModels:
    """Tests for FastEmbedVectoriser.list_supported_models."""

    @patch("classifai.vectorisers.fastembed.check_deps")
    def test_list_supported_models_missing_dependencies_raises_error(self, mock_check_deps):
        """Missing fastembed should raise the error surfaced by check_deps."""
        mock_check_deps.side_effect = ImportError("fastembed not installed")

        with pytest.raises(ImportError, match="fastembed not installed"):
            FastEmbedVectoriser.list_supported_models()

        mock_check_deps.assert_called_once_with(["fastembed"], extra="fastembed")

    @patch("classifai.vectorisers.fastembed.check_deps")
    @patch("fastembed.TextEmbedding")
    def test_list_supported_models_returns_supported_models(self, mock_textembedding, mock_check_deps):
        """Returns the model list provided by FastEmbed."""
        expected_models = [{"model": "example-model"}]
        mock_textembedding.list_supported_models.return_value = expected_models

        result = FastEmbedVectoriser.list_supported_models()

        assert result == expected_models

    @patch("classifai.vectorisers.fastembed.check_deps")
    @patch("fastembed.TextEmbedding")
    def test_list_supported_models_fastembed_failure_is_propagated(self, mock_textembedding, mock_check_deps):
        """A failure from FastEmbed's listing method is propagated."""
        mock_textembedding.list_supported_models.side_effect = RuntimeError("listing failed")

        with pytest.raises(RuntimeError, match="listing failed"):
            FastEmbedVectoriser.list_supported_models()
