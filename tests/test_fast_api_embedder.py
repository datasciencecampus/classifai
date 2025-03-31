# ruff: noqa

import sys

sys.path.append(".")

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

# Import the module to test
from fast_api.embedder import ParquetNumpyVectorStore as VectorStore
from fast_api.embedder import embed_as_array


@pytest.fixture
def sample_embeddings():
    """Fixture to provide sample embedding vectors."""
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / e_norms


@pytest.fixture
def sample_knowledgebase(sample_embeddings):
    """Fixture to provide a sample knowledgebase DataFrame."""
    return pl.DataFrame(
        {
            "label": ["doc1", "doc2", "doc3"],
            "description": ["first doc", "second doc", "third doc"],
            "embeddings": list(sample_embeddings),
            "bridge": ["bridge1", "bridge2", "bridge3"],
        }
    )


@pytest.fixture
def vector_store(sample_knowledgebase):
    """Fixture to provide a VectorStore instance."""
    return VectorStore(sample_knowledgebase)


class TestEmbedAsArray:
    @patch("google.genai.Client")
    def test_embed_as_array_shape(self, mock_client_class):
        # Setup mock response
        mock_embedding1 = MagicMock()
        mock_embedding1.values = [0.1, 0.2, 0.3]

        mock_embedding2 = MagicMock()
        mock_embedding2.values = [0.4, 0.5, 0.6]

        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding1, mock_embedding2]

        # Configure the mock client
        mock_client = mock_client_class.return_value
        mock_client.models.embed_content.return_value = mock_result

        # Test with two documents
        documents = ["Document 1", "Document 2"]
        api_key = "fake_api_key"  # pragma: allowlist secret

        result = embed_as_array(documents, api_key)

        # Check the result shape
        assert result.shape == (2, 3)

        # Verify correct API parameters
        mock_client_class.assert_called_once_with(api_key=api_key)
        mock_client.models.embed_content.assert_called_once()

        # Check the model and contents parameters
        call_args = mock_client.models.embed_content.call_args[1]
        assert call_args["model"] == "text-embedding-004"
        assert call_args["contents"] == documents

    @patch("google.genai.Client")
    def test_embed_as_array_empty_list(self, mock_client_class):
        # Setup mock response for empty list
        mock_result = MagicMock()
        mock_result.embeddings = []

        # Configure the mock client
        mock_client = mock_client_class.return_value
        mock_client.models.embed_content.return_value = mock_result

        # Test with empty list
        documents = []
        api_key = "fake_api_key"  # pragma: allowlist secret

        result = embed_as_array(documents, api_key)

        # Check the result is an empty array
        assert result.shape == (0, 0)

    @patch("google.genai.Client")
    def test_embed_as_array_single_document(self, mock_client_class):
        # Setup mock response
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]

        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]

        # Configure the mock client
        mock_client = mock_client_class.return_value
        mock_client.models.embed_content.return_value = mock_result

        # Test with single document
        documents = ["Single document"]
        api_key = "fake_api_key"  # pragma: allowlist secret

        result = embed_as_array(documents, api_key)

        # Check the result shape (1 document × embedding size)
        assert result.shape == (1, 3)


class TestVectorStore:
    def test_init(self, sample_knowledgebase):
        # Test initialization
        vector_store = VectorStore(sample_knowledgebase)

        assert isinstance(vector_store.knowledgebase, pl.DataFrame)
        assert len(vector_store.knowledgebase) == 3

    def test_query_single(self, vector_store):
        # Test query with a single embedding vector
        query_embedding = np.array([0.7, 0.8, 0.9])  # Identical to doc3
        ids = ["query1"]

        result = vector_store.query(query_embedding, ids, k=2)

        # Verify result structure
        assert "idx" in result
        assert "scores" in result
        assert "ids" in result

        # Verify shape of results
        assert result["idx"].shape == (1, 2)
        assert result["scores"].shape == (1, 2)

        # The top match should be doc3 (index 2) as it has identical embedding
        assert result["idx"][0, 0] == 2

        # With scores_as_distance=True, identical vectors should have distance close to 0
        assert pytest.approx(result["scores"][0, 0], abs=1e-5) == 0.0

    def test_query_multiple(self, vector_store):
        # Test query with multiple embedding vectors
        query_embeddings = np.array(
            [
                [0.7, 0.8, 0.9],  # Identical to doc3
                [0.4, 0.5, 0.6],  # Identical to doc2
            ]
        )
        ids = ["query1", "query2"]

        result = vector_store.query(query_embeddings, ids, k=3)

        # Verify result shape for multiple queries
        assert result["idx"].shape == (2, 3)
        assert result["scores"].shape == (2, 3)

        # Check top matches
        assert result["idx"][0, 0] == 2  # First query matches doc3
        assert result["idx"][1, 0] == 1  # Second query matches doc2

        # Test with scores_as_distance=False
        result_similarity = vector_store.query(
            query_embeddings, ids, k=3, scores_as_distance=False
        )

        # With similarity scores, highest should be close to 1.0 for identical vectors
        assert (
            pytest.approx(result_similarity["scores"][0, 0], abs=1e-5) == 1.0
        )

    def test_create_json_array_response(self, vector_store):
        # Prepare mock query result
        query_result = {
            "scores": np.array([[0.1, 0.2], [0.05, 0.15]]),
            "idx": np.array([[2, 1], [0, 2]]),
            "ids": ["query1", "query2"],
        }

        # Test without bridge information
        result = vector_store.create_json_array_response(
            query_result, include_bridge=False
        )

        # Verify structure
        assert len(result) == 2
        assert result[0]["input_id"] == "query1"
        assert len(result[0]["response"]) == 2

        # Check response fields
        first_match = result[0]["response"][0]
        assert "label" in first_match
        assert "description" in first_match
        assert "distance" in first_match
        assert "rank" in first_match
        assert "bridge" in first_match

        # Check values
        assert first_match["label"] == "doc3"  # From idx 2
        assert first_match["distance"] == 0.1  # From scores
        assert first_match["rank"] == 1  # Lowest distance gets rank 1
        assert (
            first_match["bridge"] == ""
        )  # Empty string when include_bridge=False

        # Test with bridge information
        result_with_bridge = vector_store.create_json_array_response(
            query_result, include_bridge=True
        )

        # Check bridge is included
        first_match_with_bridge = result_with_bridge[0]["response"][0]
        assert first_match_with_bridge["bridge"] == "bridge3"  # From idx 2
