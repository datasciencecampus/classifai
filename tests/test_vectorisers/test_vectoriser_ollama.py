"""Unit tests for OllamaVectoriser."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from classifai.exceptions import ExternalServiceError, VectorisationError
from classifai.vectorisers import OllamaVectoriser


class TestOllamaVectoriserInitialization:
    """Tests for OllamaVectoriser initialization."""

    @patch("classifai.vectorisers.ollama.check_deps")
    def test_init_missing_dependencies_raises_error(self, mock_check_deps):
        """Missing ollama package should raise the error surfaced by check_deps."""
        mock_check_deps.side_effect = ImportError("ollama not installed")

        with pytest.raises(ImportError):
            OllamaVectoriser(model_name="nomic-embed-text")

        mock_check_deps.assert_called_once_with(["ollama"], extra="ollama")

    @patch("classifai.vectorisers.ollama.check_deps")
    def test_init_model_name_stored_correctly(self, mock_check_deps):
        """model_name should be stored as an attribute."""
        vectoriser = OllamaVectoriser(model_name="nomic-embed-text")

        assert vectoriser.model_name == "nomic-embed-text"


class TestOllamaVectoriserTransform:
    """Tests for OllamaVectoriser transform method."""

    @pytest.fixture
    def vectoriser(self):
        """Return an OllamaVectoriser instance with check_deps mocked out."""
        with patch("classifai.vectorisers.ollama.check_deps"):
            return OllamaVectoriser(model_name="nomic-embed-text")

    def _configure_successful_embed_response(self, mock_embed, n_texts, dim=4):
        """Wire up ollama.embed to return a fake embeddings response."""
        fake_response = Mock()
        fake_response.embeddings = [[0.1] * dim for _ in range(n_texts)]
        mock_embed.return_value = fake_response

    @patch("ollama.embed")
    def test_transform_single_string_converts_to_list(self, mock_embed, vectoriser):
        """A single string input should be wrapped in a list before the API call."""
        self._configure_successful_embed_response(mock_embed, n_texts=1)

        vectoriser.transform("hello world")

        _, call_kwargs = mock_embed.call_args
        assert call_kwargs["input"] == ["hello world"]

    @patch("ollama.embed")
    def test_transform_list_processes_correctly(self, mock_embed, vectoriser):
        """A list of strings should be passed through unchanged."""
        texts = ["text1", "text2", "text3"]
        self._configure_successful_embed_response(mock_embed, n_texts=len(texts))

        vectoriser.transform(texts)

        _, call_kwargs = mock_embed.call_args
        assert call_kwargs["input"] == texts

    @patch("ollama.embed")
    def test_transform_returns_2d_numpy_array(self, mock_embed, vectoriser):
        """Output should be a 2D numpy array."""
        self._configure_successful_embed_response(mock_embed, n_texts=2)

        result = vectoriser.transform(["a", "b"])

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    @patch("ollama.embed")
    def test_transform_output_shape_matches_input_count(self, mock_embed, vectoriser):
        """Number of output rows should match number of input texts."""
        texts = ["a", "b", "c", "d"]
        self._configure_successful_embed_response(mock_embed, n_texts=len(texts))

        result = vectoriser.transform(texts)

        assert result.shape[0] == len(texts)

    @patch("ollama.embed")
    def test_transform_model_name_passed_to_embed_call(self, mock_embed, vectoriser):
        """The stored model_name should be passed to ollama.embed."""
        self._configure_successful_embed_response(mock_embed, n_texts=1)

        vectoriser.transform(["hello"])

        _, call_kwargs = mock_embed.call_args
        assert call_kwargs["model"] == vectoriser.model_name

    @patch("ollama.embed")
    def test_transform_service_failure_raises_external_service_error(self, mock_embed, vectoriser):
        """If ollama.embed itself raises, it should be wrapped in ExternalServiceError."""
        mock_embed.side_effect = Exception("connection refused")

        with pytest.raises(ExternalServiceError):
            vectoriser.transform(["hello"])

    @patch("ollama.embed")
    def test_transform_response_parsing_failure_raises_vectorisation_error(self, mock_embed, vectoriser):
        """If extracting/converting .embeddings fails, it should be wrapped in VectorisationError."""
        fake_response = Mock(spec=[])  # no .embeddings attribute at all
        mock_embed.return_value = fake_response

        with pytest.raises(VectorisationError):
            vectoriser.transform(["hello"])
