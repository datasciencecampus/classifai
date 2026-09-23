"""Unit tests for GcpVectoriser."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from classifai.exceptions import ConfigurationError, ExternalServiceError, VectorisationError
from classifai.vectorisers import GcpVectoriser


class TestGcpVectoriserInitialization:
    """Tests for GcpVectoriser initialization."""

    @patch("classifai.vectorisers.gcp.check_deps")
    def test_init_missing_dependencies_raises_error(self, mock_check_deps):
        """Missing google-genai should raise the error surfaced by check_deps."""
        mock_check_deps.side_effect = ImportError("google-genai not installed")

        with pytest.raises(ImportError):
            GcpVectoriser(project_id="my-project", location="europe-west2")

        mock_check_deps.assert_called_once_with(["google-genai"], extra="gcp")

    @patch("classifai.vectorisers.gcp.check_deps")
    @patch("google.genai.Client")
    def test_init_project_id_and_location_authentication_works(self, mock_client, mock_check_deps):
        """project_id + location should be forwarded to the client constructor."""
        vectoriser = GcpVectoriser(project_id="my-project", location="europe-west2")

        _, call_kwargs = mock_client.call_args
        assert call_kwargs["project"] == "my-project"
        assert call_kwargs["location"] == "europe-west2"
        assert vectoriser.vectoriser is mock_client.return_value

    @patch("classifai.vectorisers.gcp.check_deps")
    @patch("google.genai.Client")
    def test_init_api_key_authentication_works(self, mock_client, mock_check_deps):
        """api_key alone should be forwarded to the client constructor."""
        vectoriser = GcpVectoriser(api_key="fake-api-key")

        _, call_kwargs = mock_client.call_args
        assert call_kwargs["api_key"] == "fake-api-key"  # pragma: allowlist secret
        assert vectoriser.vectoriser is mock_client.return_value

    @patch("classifai.vectorisers.gcp.check_deps")
    def test_init_missing_both_auth_methods_raises_configuration_error(self, mock_check_deps):
        """Providing neither project_id/location nor api_key should raise ConfigurationError."""
        with pytest.raises(ConfigurationError):
            GcpVectoriser()

    @patch("classifai.vectorisers.gcp.check_deps")
    def test_init_providing_both_auth_methods_raises_configuration_error(self, mock_check_deps):
        """Providing both project_id and api_key should raise ConfigurationError."""
        with pytest.raises(ConfigurationError):
            GcpVectoriser(project_id="my-project", api_key="fake-api-key")

    @patch("classifai.vectorisers.gcp.check_deps")
    @patch("google.genai.Client")
    def test_init_client_initialisation_failure_raises_configuration_error(self, mock_client, mock_check_deps):
        """If the underlying client constructor raises, wrap it in ConfigurationError."""
        mock_client.side_effect = Exception("bad credentials")

        with pytest.raises(ConfigurationError):
            GcpVectoriser(project_id="my-project", location="europe-west2")

    @patch("classifai.vectorisers.gcp.check_deps")
    @patch("google.genai.Client")
    def test_init_model_name_stored_correctly(self, mock_client, mock_check_deps):
        """model_name should be stored as an attribute."""
        vectoriser = GcpVectoriser(api_key="fake-api-key", model_name="text-embedding-005")

        assert vectoriser.model_name == "text-embedding-005"

    @patch("classifai.vectorisers.gcp.check_deps")
    @patch("google.genai.Client")
    def test_init_task_type_passed_to_embed_content_config(self, mock_client, mock_check_deps):
        """task_type should be forwarded to EmbedContentConfig."""
        with patch("google.genai.types.EmbedContentConfig") as mock_config:
            GcpVectoriser(api_key="fake-api-key", task_type="RETRIEVAL_QUERY")  # pragma: allowlist secret

            _, call_kwargs = mock_config.call_args
            assert call_kwargs["task_type"] == "RETRIEVAL_QUERY"


class TestGcpVectoriserTransform:
    """Tests for GcpVectoriser transform method."""

    @pytest.fixture
    def mock_vectoriser(self):
        """Return a GcpVectoriser instance with a mocked client."""
        with (
            patch("classifai.vectorisers.gcp.check_deps"),
            patch("google.genai.Client"),
        ):
            vectoriser = GcpVectoriser(api_key="fake-api-key")

            # Replace with a controllable mock for the transform tests.
            vectoriser.vectoriser = Mock()

            yield vectoriser

    def _configure_successful_embed_response(self, vectoriser, n_texts, dim=4):
        """Wire up the client mock to return a fake embeddings response."""
        fake_response = Mock()
        fake_response.embeddings = [Mock(values=[0.1] * dim) for _ in range(n_texts)]
        vectoriser.vectoriser.models.embed_content.return_value = fake_response

    def test_transform_single_string_converts_to_list(self, mock_vectoriser):
        """A single string input should be wrapped in a list before the API call."""
        self._configure_successful_embed_response(mock_vectoriser, n_texts=1)

        mock_vectoriser.transform("hello world")

        _, call_kwargs = mock_vectoriser.vectoriser.models.embed_content.call_args
        assert call_kwargs["contents"] == ["hello world"]

    def test_transform_list_processes_correctly(self, mock_vectoriser):
        """A list of strings should be passed through unchanged."""
        texts = ["text1", "text2", "text3"]
        self._configure_successful_embed_response(mock_vectoriser, n_texts=len(texts))

        mock_vectoriser.transform(texts)

        _, call_kwargs = mock_vectoriser.vectoriser.models.embed_content.call_args
        assert call_kwargs["contents"] == texts

    def test_transform_returns_2d_numpy_array(self, mock_vectoriser):
        """Output should be a 2D numpy array."""
        self._configure_successful_embed_response(mock_vectoriser, n_texts=2)

        result = mock_vectoriser.transform(["a", "b"])

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_transform_output_shape_matches_input_count(self, mock_vectoriser):
        """Number of output rows should match number of input texts."""
        texts = ["a", "b", "c", "d"]
        self._configure_successful_embed_response(mock_vectoriser, n_texts=len(texts))

        result = mock_vectoriser.transform(texts)

        assert result.shape[0] == len(texts)

    def test_transform_api_request_failure_raises_external_service_error(self, mock_vectoriser):
        """If the API call itself raises, it should be wrapped in ExternalServiceError."""
        mock_vectoriser.vectoriser.models.embed_content.side_effect = Exception("network error")

        with pytest.raises(ExternalServiceError):
            mock_vectoriser.transform(["hello"])

    def test_transform_unexpected_response_format_raises_vectorisation_error(self, mock_vectoriser):
        """If the response doesn't have the expected .embeddings attribute, raise VectorisationError."""
        mock_vectoriser.vectoriser.models.embed_content.return_value = Mock(spec=[])  # no .embeddings attribute

        with pytest.raises(VectorisationError):
            mock_vectoriser.transform(["hello"])

    def test_transform_model_name_passed_to_api_call(self, mock_vectoriser):
        """The stored model_name should be passed to embed_content."""
        self._configure_successful_embed_response(mock_vectoriser, n_texts=1)

        mock_vectoriser.transform(["hello"])

        _, call_kwargs = mock_vectoriser.vectoriser.models.embed_content.call_args
        assert call_kwargs["model"] == mock_vectoriser.model_name
