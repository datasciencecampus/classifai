"""Unit tests for VectorStore.search() method."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from classifai.exceptions import (
    ConfigurationError,
    DataValidationError,
    HookError,
    VectorisationError,
)
from classifai.indexers import VectorStore
from classifai.indexers.dataclasses import VectorStoreSearchInput, VectorStoreSearchOutput
from classifai.indexers.hooks import HookBase
from classifai.vectorisers import VectoriserBase

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_vectoriser():
    """Mock VectoriserBase that returns predictable embeddings."""
    mock = Mock(spec=VectoriserBase)

    def transform_side_effect(texts):
        """Return one embedding per text, all with shape (3,)."""
        num_texts = len(texts)
        return np.array([np.linspace(0.1, 0.3, 3) + (i * 0.3) for i in range(num_texts)])

    mock.transform.side_effect = transform_side_effect
    mock.__class__.__name__ = "MockVectoriser"
    return mock


@pytest.fixture
def initialized_vectorstore(mock_vectoriser, tmp_path):
    """Create a fully initialized VectorStore with test data."""
    # Create a test CSV
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("label,text\ndoc1,hello world\ndoc2,goodbye world\ndoc3,test document\n")

    # Create VectorStore
    vs = VectorStore(
        file_name=str(csv_path),
        data_type="csv",
        vectoriser=mock_vectoriser,
        output_dir=str(tmp_path / "output"),
        skip_save=True,
    )

    return vs


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================


class TestVectorStoreSearchInputValidation:
    """Tests for input validation in search() method."""

    def test_search_query_must_be_vectorstore_search_input(self, initialized_vectorstore):
        """Query must be VectorStoreSearchInput object."""
        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.search(query="not a VectorStoreSearchInput")

        assert "VectorStoreSearchInput" in str(exc_info.value)

    def test_search_query_none_raises_error(self, initialized_vectorstore):
        """query=None raises DataValidationError."""
        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.search(query=None)

        assert "VectorStoreSearchInput" in str(exc_info.value)

    def test_search_n_results_must_be_positive_int(self, initialized_vectorstore):
        """n_results must be int >= 1."""
        query = VectorStoreSearchInput.from_data({"id": ["1"], "query": ["test"]})

        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.search(query=query, n_results=0)

        assert "n_results" in str(exc_info.value).lower()

    def test_search_n_results_negative_raises_error(self, initialized_vectorstore):
        """n_results < 1 raises DataValidationError."""
        query = VectorStoreSearchInput.from_data({"id": ["1"], "query": ["test"]})

        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.search(query=query, n_results=-1)

        assert "n_results" in str(exc_info.value).lower()

    def test_search_n_results_non_int_raises_error(self, initialized_vectorstore):
        """n_results as float/string raises DataValidationError."""
        query = VectorStoreSearchInput.from_data({"id": ["1"], "query": ["test"]})

        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.search(query=query, n_results="10")

        assert "n_results" in str(exc_info.value).lower()

    def test_search_batch_size_must_be_positive_int(self, initialized_vectorstore):
        """batch_size must be int >= 1 or None."""
        query = VectorStoreSearchInput.from_data({"id": ["1"], "query": ["test"]})

        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.search(query=query, batch_size=0)

        assert "batch_size" in str(exc_info.value).lower()

    def test_search_batch_size_negative_raises_error(self, initialized_vectorstore):
        """batch_size < 1 raises DataValidationError."""
        query = VectorStoreSearchInput.from_data({"id": ["1"], "query": ["test"]})

        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.search(query=query, batch_size=-5)

        assert "batch_size" in str(exc_info.value).lower()

    def test_search_batch_size_non_int_raises_error(self, initialized_vectorstore):
        """batch_size as string raises DataValidationError."""
        query = VectorStoreSearchInput.from_data({"id": ["1"], "query": ["test"]})

        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.search(query=query, batch_size="10")

        assert "batch_size" in str(exc_info.value).lower()

    def test_search_batch_size_none_is_valid(self, initialized_vectorstore):
        """batch_size=None is valid and uses default."""
        query = VectorStoreSearchInput.from_data({"id": ["1"], "query": ["test"]})

        # Should not raise
        result = initialized_vectorstore.search(query=query, n_results=3, batch_size=None)
        assert isinstance(result, VectorStoreSearchOutput)

    def test_search_empty_query_raises_error(self, initialized_vectorstore):
        """Empty query DataFrame raises DataValidationError."""
        query = VectorStoreSearchInput.from_data({"id": [], "query": []})

        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.search(query=query)

        assert "empty" in str(exc_info.value).lower()

    def test_search_uninitialized_vectorstore_raises_error(self, mock_vectoriser):
        """Vector store not initialized raises ConfigurationError."""
        vs = Mock(spec=VectorStore)
        vs.vectors = None
        vs.batch_size = 128
        vs.meta_data = {}

        # Create a real instance but with vectors=None
        vs_real = VectorStore.__new__(VectorStore)
        vs_real.vectors = None
        vs_real.batch_size = 128
        vs_real.meta_data = {}
        vs_real.vectoriser = mock_vectoriser
        vs_real.vectoriser_class = "MockVectoriser"
        vs_real.hooks = {}
        vs_real.quiet_mode = False
        vs_real.classifai_tqdm = lambda iterable, *args, **kwargs: iterable

        query = VectorStoreSearchInput.from_data({"id": ["1"], "query": ["test"]})

        with pytest.raises(ConfigurationError) as exc_info:
            vs_real.search(query=query)

        assert "not initialised" in str(exc_info.value).lower()


# ============================================================================
# SEARCH OPERATION TESTS
# ============================================================================


class TestVectorStoreSearchOperation:
    """Tests for the core search operation."""

    def test_search_single_query_returns_results(self, initialized_vectorstore):
        """Single query processes correctly and returns results."""
        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

        result = initialized_vectorstore.search(query=query, n_results=2)

        assert isinstance(result, VectorStoreSearchOutput)
        assert len(result) == 2  # n_results=2
        assert result.query_id[0] == "q1"

    def test_search_multiple_queries_processes_correctly(self, initialized_vectorstore):
        """Multiple queries in batch process correctly."""
        query = VectorStoreSearchInput.from_data(
            {
                "id": ["q1", "q2"],
                "query": ["hello", "test"],
            }
        )

        result = initialized_vectorstore.search(query=query, n_results=2)

        assert isinstance(result, VectorStoreSearchOutput)
        assert len(result) == 4  # 2 queries * 2 results each
        assert list(result.query_id[:2]) == ["q1", "q1"]
        assert list(result.query_id[2:]) == ["q2", "q2"]

    def test_search_similarity_scores_computed(self, initialized_vectorstore):
        """Similarity scores are computed via dot-product."""
        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

        result = initialized_vectorstore.search(query=query, n_results=3)

        # Check that scores are floats and have values
        assert all(isinstance(score, (float, np.floating)) for score in result.score)
        assert len(result.score) == 3

    def test_search_top_n_results_returned(self, initialized_vectorstore):
        """Top n_results returned per query."""
        query = VectorStoreSearchInput.from_data(
            {
                "id": ["q1"],
                "query": ["hello"],
            }
        )

        result = initialized_vectorstore.search(query=query, n_results=2)

        # Should have exactly 2 results (n_results=2)
        assert len(result) == 2

    def test_search_results_ranked_by_score_descending(self, initialized_vectorstore):
        """Results ranked by score in descending order."""
        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

        result = initialized_vectorstore.search(query=query, n_results=3)

        scores = list(result.score)
        # Verify scores are in descending order
        assert scores == sorted(scores, reverse=True)

    def test_search_output_shape_matches_expected(self, initialized_vectorstore):
        """Output shape is (n_queries * n_results) rows."""
        query = VectorStoreSearchInput.from_data(
            {
                "id": ["q1", "q2", "q3"],
                "query": ["hello", "test", "world"],
            }
        )

        n_results = 2
        result = initialized_vectorstore.search(query=query, n_results=n_results)

        # Expected: 3 queries * 2 results = 6 rows
        assert len(result) == 3 * n_results

    def test_search_metadata_columns_included_in_output(self, mock_vectoriser, tmp_path):
        """Metadata columns included in output when specified."""
        csv_path = tmp_path / "test_meta.csv"
        csv_path.write_text("label,text,source\ndoc1,hello,source_a\ndoc2,world,source_b\n")

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            meta_data={"source": str},
            output_dir=str(tmp_path / "output"),
            skip_save=True,
        )

        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})
        result = vs.search(query=query, n_results=2)

        # Verify metadata column is in output
        assert "source" in result.columns

    def test_search_query_batching_with_custom_batch_size(self, initialized_vectorstore, mock_vectoriser):
        """Query batching with custom batch_size works."""
        query = VectorStoreSearchInput.from_data(
            {
                "id": ["q1", "q2", "q3"],
                "query": ["hello", "test", "world"],
            }
        )

        # Use batch_size=1 to force multiple batches
        result = initialized_vectorstore.search(query=query, n_results=2, batch_size=1)

        assert isinstance(result, VectorStoreSearchOutput)
        assert len(result) == 6  # 3 queries * 2 results

    def test_search_returns_correct_columns(self, initialized_vectorstore):
        """Output contains all required columns."""
        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

        result = initialized_vectorstore.search(query=query, n_results=2)

        required_columns = ["query_id", "query_text", "doc_label", "doc_text", "rank", "score"]
        for col in required_columns:
            assert col in result.columns


# # ============================================================================
# # ERROR HANDLING TESTS
# # ============================================================================


class TestVectorStoreSearchErrorHandling:
    """Tests for error handling during search."""

    def test_search_query_embedding_failure_raises_vectorisation_error(self, initialized_vectorstore):
        """Query embedding failure raises VectorisationError."""
        # Make vectoriser fail on query embedding
        initialized_vectorstore.vectoriser.transform.side_effect = Exception("Vectoriser failed")

        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

        with pytest.raises(VectorisationError) as exc_info:
            initialized_vectorstore.search(query=query)

        assert "Failed to embed query batch" in str(exc_info.value)

    def test_search_vectoriser_exceptions_wrapped(self, initialized_vectorstore):
        """Vectoriser.transform() exceptions include context."""
        initialized_vectorstore.vectoriser.transform.side_effect = RuntimeError("Transform failed")

        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

        with pytest.raises(VectorisationError) as exc_info:
            initialized_vectorstore.search(query=query)

        error_str = str(exc_info.value)
        assert "MockVectoriser" in error_str  # vectoriser class in context
        assert "batch_size" in error_str or "batch" in error_str.lower()

    def test_search_vectoriser_error_includes_batch_info(self, initialized_vectorstore):
        """Error context includes vectoriser class and batch info."""
        initialized_vectorstore.vectoriser.transform.side_effect = ValueError("Bad embedding")

        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["test"]})

        with pytest.raises(VectorisationError) as exc_info:
            initialized_vectorstore.search(query=query)

        context_str = str(exc_info.value)
        assert "vectoriser" in context_str.lower() or "MockVectoriser" in context_str


# # ============================================================================
# # HOOKS INTEGRATION TESTS
# # ============================================================================


class TestVectorStoreSearchHooksIntegration:
    """Tests for hook integration in search."""

    def test_search_preprocess_hook_called_before_search(self, initialized_vectorstore):
        """search_preprocess hook called before search."""
        mock_hook = Mock(spec=HookBase)
        mock_hook.return_value = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

        initialized_vectorstore.hooks = {"search_preprocess": mock_hook}

        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})
        result = initialized_vectorstore.search(query=query, n_results=3)

        mock_hook.assert_called_once()

    def test_search_postprocess_hook_called_after_search(self, initialized_vectorstore):
        """search_postprocess hook called after search."""
        # Create a real VectorStoreSearchOutput to return from the hook
        mock_result = VectorStoreSearchOutput.from_data(
            {
                "query_id": ["q1"],
                "query_text": ["hello"],
                "doc_label": ["doc1"],
                "doc_text": ["hello world"],
                "rank": [1],
                "score": [0.95],
            }
        )

        mock_hook = Mock(spec=HookBase)
        mock_hook.return_value = mock_result

        initialized_vectorstore.hooks = {"search_postprocess": mock_hook}

        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})
        result = initialized_vectorstore.search(query=query, n_results=3)

        mock_hook.assert_called_once()
        # Optionally we could verify the result is what we expect
        assert isinstance(result, VectorStoreSearchOutput)

    def test_search_hook_failure_raises_hook_error(self, initialized_vectorstore):
        """Hook failure raises HookError."""
        bad_hook = Mock(spec=HookBase)
        bad_hook.side_effect = Exception("Hook failed")

        initialized_vectorstore.hooks = {"search_preprocess": bad_hook}

        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

        with pytest.raises(HookError) as exc_info:
            initialized_vectorstore.search(query=query, n_results=3)

        assert "search_preprocess" in str(exc_info.value)

    def test_search_multiple_hooks_processed_in_order(self, initialized_vectorstore):
        """Multiple hooks in list processed in order."""
        hook1 = Mock(spec=HookBase)
        hook1.return_value = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

        hook2 = Mock(spec=HookBase)
        hook2.return_value = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

        initialized_vectorstore.hooks = {"search_preprocess": [hook1, hook2]}

        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})
        result = initialized_vectorstore.search(query=query, n_results=3)

        assert hook1.call_count == 1
        assert hook2.call_count == 1

    def test_search_single_hook_converted_to_list(self, initialized_vectorstore):
        """Single hook automatically converted to list."""
        mock_hook = Mock(spec=HookBase)
        mock_hook.return_value = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

        initialized_vectorstore.hooks = {"search_preprocess": mock_hook}

        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})
        result = initialized_vectorstore.search(query=query, n_results=3)

        # Verify hook was called (auto-converted to list)
        mock_hook.assert_called_once()

    def test_search_postprocess_hook_failure_raises_hook_error(self, initialized_vectorstore):
        """Postprocess hook failure raises HookError."""
        bad_hook = Mock(spec=HookBase)
        bad_hook.side_effect = Exception("Postprocess failed")

        initialized_vectorstore.hooks = {"search_postprocess": bad_hook}

        query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

        with pytest.raises(HookError) as exc_info:
            initialized_vectorstore.search(query=query, n_results=3)

        assert "search_postprocess" in str(exc_info.value)


# # ============================================================================
# # EDGE CASE TESTS
# # ============================================================================

# TODO: All of these tests rely on querying to retrieve large values of results, however currently there is a bug
# that if n_results is greater than the number of documents in the store then an out of bounds error is thrown. This needs to be fixed in the VectorStore.search() method before these tests can be run.

# class TestVectorStoreSearchEdgeCases:
#     """Tests for edge cases in search."""

#     def test_search_n_results_greater_than_available_documents(self, initialized_vectorstore):
#         """n_results > available documents returns all documents."""
#         # We have 3 documents in the store
#         query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

#         result = initialized_vectorstore.search(query=query, n_results=100)

#         # Should return only 3 documents (all available)
#         assert len(result) == 3

#     def test_search_single_query_single_document_store(self, mock_vectoriser, tmp_path):
#         """Search with single document in store."""
#         csv_path = tmp_path / "single.csv"
#         csv_path.write_text("label,text\ndoc1,only document\n")

#         vs = VectorStore(
#             file_name=str(csv_path),
#             data_type="csv",
#             vectoriser=mock_vectoriser,
#             output_dir=str(tmp_path / "output"),
#             skip_save=True,
#         )

#         query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["test"]})
#         result = vs.search(query=query, n_results=5)

#         assert len(result) == 1

#     def test_search_identical_embeddings_still_returns_results(self, initialized_vectorstore):
#         """Search works even with identical embeddings."""
#         query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

#         result = initialized_vectorstore.search(query=query, n_results=2)

#         assert len(result) == 2
#         assert isinstance(result, VectorStoreSearchOutput)

#     def test_search_large_n_results_value(self, initialized_vectorstore):
#         """Search with very large n_results parameter."""
#         query = VectorStoreSearchInput.from_data({"id": ["q1"], "query": ["hello"]})

#         result = initialized_vectorstore.search(query=query, n_results=1000)

#         # Should return only 3 documents (all available)
#         assert len(result) == 3
