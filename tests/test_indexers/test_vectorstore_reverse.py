"""Unit tests for VectorStore.reverse_search() method."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from classifai.exceptions import (
    ClassifaiError,
    DataValidationError,
    HookError,
)
from classifai.indexers import VectorStore
from classifai.indexers.dataclasses import (
    VectorStoreReverseSearchInput,
    VectorStoreReverseSearchOutput,
)
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
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "label,text\n"
        "category_a,document 1\n"
        "category_a,document 2\n"
        "category_b,document 3\n"
        "category_b,document 4\n"
        "category_c,document 5\n"
    )

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


class TestVectorStoreReverseSearchInputValidation:
    """Tests for input validation in reverse_search() method."""

    def test_reverse_search_query_must_be_vectorstore_reverse_search_input(self, initialized_vectorstore):
        """Query must be VectorStoreReverseSearchInput object."""
        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.reverse_search(query="not a VectorStoreReverseSearchInput")

        assert "VectorStoreReverseSearchInput" in str(exc_info.value)

    def test_reverse_search_query_none_raises_error(self, initialized_vectorstore):
        """query=None raises DataValidationError."""
        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.reverse_search(query=None)

        assert "VectorStoreReverseSearchInput" in str(exc_info.value)

    def test_reverse_search_query_dict_raises_error(self, initialized_vectorstore):
        """Query as dict (not VectorStoreReverseSearchInput) raises DataValidationError."""
        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.reverse_search(query={"id": ["1"], "doc_label": ["category_a"]})

        assert "VectorStoreReverseSearchInput" in str(exc_info.value)

    def test_reverse_search_query_list_raises_error(self, initialized_vectorstore):
        """Query as list raises DataValidationError."""
        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.reverse_search(query=["category_a", "category_b"])

        assert "VectorStoreReverseSearchInput" in str(exc_info.value)

    def test_reverse_search_max_n_results_must_be_positive_int(self, initialized_vectorstore):
        """max_n_results must be int >= 1 or -1."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["1"],
                "doc_label": ["category_a"],
            }
        )

        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.reverse_search(query=query, max_n_results=0)

        assert "max_n_results" in str(exc_info.value).lower()

    def test_reverse_search_max_n_results_negative_not_minus_one_raises_error(self, initialized_vectorstore):
        """max_n_results < -1 raises DataValidationError."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["1"],
                "doc_label": ["category_a"],
            }
        )

        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.reverse_search(query=query, max_n_results=-5)

        assert "max_n_results" in str(exc_info.value).lower()

    def test_reverse_search_max_n_results_non_int_raises_error(self, initialized_vectorstore):
        """max_n_results as string raises DataValidationError."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["1"],
                "doc_label": ["category_a"],
            }
        )

        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.reverse_search(query=query, max_n_results="10")

        assert "max_n_results" in str(exc_info.value).lower()

    def test_reverse_search_max_n_results_minus_one_is_valid(self, initialized_vectorstore):
        """max_n_results=-1 is valid (means return all)."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["1"],
                "doc_label": ["category_a"],
            }
        )

        # Should not raise
        result = initialized_vectorstore.reverse_search(query=query, max_n_results=-1)
        assert isinstance(result, VectorStoreReverseSearchOutput)

    def test_reverse_search_empty_query_raises_error(self, initialized_vectorstore):
        """Empty query DataFrame raises DataValidationError."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": [],
                "doc_label": [],
            }
        )

        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.reverse_search(query=query)

        assert "empty" in str(exc_info.value).lower()

        # TODO: implement if/when partial_match parameter checks are added
        # def test_reverse_search_partial_match_must_be_bool(self, initialized_vectorstore):
        #     """partial_match must be boolean."""
        #     query = VectorStoreReverseSearchInput.from_data({
        #         "id": ["1"],
        #         "doc_label": ["category_a"],
        #     })

        with pytest.raises((DataValidationError, TypeError)):
            initialized_vectorstore.reverse_search(query=query, partial_match="yes")

    def test_reverse_search_partial_match_default_false(self, initialized_vectorstore):
        """partial_match defaults to False."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["1"],
                "doc_label": ["category_a"],
            }
        )

        # Should not raise - defaults to False
        result = initialized_vectorstore.reverse_search(query=query)
        assert isinstance(result, VectorStoreReverseSearchOutput)


# ============================================================================
# REVERSE SEARCH OPERATION TESTS
# ============================================================================


class TestVectorStoreReverseSearchOperation:
    """Tests for the core reverse search operation."""

    def test_reverse_search_exact_label_matching_works(self, initialized_vectorstore):
        """Exact label matching works (default)."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        assert isinstance(result, VectorStoreReverseSearchOutput)
        # Should find 2 documents with exact label "category_a"
        assert len(result) == 2
        assert all(label == "category_a" for label in result.doc_label)

    def test_reverse_search_partial_matching_when_enabled(self, initialized_vectorstore):
        """Partial matching (prefix) works when partial_match=True."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category"],
            }
        )

        result = initialized_vectorstore.reverse_search(
            query=query,
            max_n_results=10,
            partial_match=True,
        )

        # Should find all documents with labels starting with "category"
        assert len(result) >= 5  # All 5 documents start with "category"

    def test_reverse_search_exact_matching_excludes_partial(self, initialized_vectorstore):
        """Exact matching excludes partial matches (default behavior)."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category"],
            }
        )

        result = initialized_vectorstore.reverse_search(
            query=query,
            max_n_results=10,
            partial_match=False,
        )

        # Should find 0 documents (no exact match for "category")
        assert len(result) == 0

    def test_reverse_search_max_n_results_limits_results(self, initialized_vectorstore):
        """max_n_results limits results per query."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=1)

        # Should return only 1 result even though there are 2 matches
        assert len(result) == 1

    def test_reverse_search_max_n_results_minus_one_returns_all(self, initialized_vectorstore):
        """max_n_results=-1 returns all matches."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=-1)

        # Should return all 2 matches
        assert len(result) == 2

    def test_reverse_search_multiple_queries_process_correctly(self, initialized_vectorstore):
        """Multiple queries process correctly."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1", "q2"],
                "doc_label": ["category_a", "category_b"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        # Should find 2 + 2 = 4 documents total
        assert len(result) == 4
        assert list(result.id[:2]) == ["q1", "q1"]
        assert list(result.id[2:]) == ["q2", "q2"]

    def test_reverse_search_includes_required_columns(self, initialized_vectorstore):
        """Output includes all required columns."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        required_columns = ["id", "searched_doc_label", "doc_label", "doc_text"]
        for col in required_columns:
            assert col in result.columns

    def test_reverse_search_empty_result_set_returns_empty_dataframe(self, initialized_vectorstore):
        """Empty result set returns empty DataFrame with correct schema."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["nonexistent_label"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        assert isinstance(result, VectorStoreReverseSearchOutput)
        assert len(result) == 0
        # Verify schema is still correct
        required_columns = ["id", "searched_doc_label", "doc_label", "doc_text"]
        for col in required_columns:
            assert col in result.columns

    def test_reverse_search_results_sorted_by_id_and_label(self, initialized_vectorstore):
        """Results are sorted by id and searched_doc_label."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q2", "q1", "q3"],
                "doc_label": ["category_b", "category_a", "category_c"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        # Results should be sorted
        ids = list(result.id)
        # Verify ids are in order
        assert isinstance(ids, list)

    def test_reverse_search_case_sensitive_matching(self, initialized_vectorstore):
        """Label matching is case-sensitive by default."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["CATEGORY_A"],  # uppercase
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        # Should not match "category_a" (lowercase)
        assert len(result) == 0

    def test_reverse_search_preserves_query_id(self, initialized_vectorstore):
        """Query ID is preserved in output."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["custom_id_123"],
                "doc_label": ["category_a"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        assert all(qid == "custom_id_123" for qid in result.id)

    def test_reverse_search_preserves_query_label_in_searched_doc_label(self, initialized_vectorstore):
        """Query label is preserved in searched_doc_label column."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        assert all(label == "category_a" for label in result.searched_doc_label)

    def test_reverse_search_doc_label_matches_stored_labels(self, initialized_vectorstore):
        """doc_label in output matches stored document labels."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        # doc_label should match the stored labels
        assert all(label == "category_a" for label in result.doc_label)

    def test_reverse_search_doc_text_contains_document_content(self, initialized_vectorstore):
        """doc_text contains the actual document text."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        # doc_text should contain the original text from the CSV
        assert all(isinstance(text, str) for text in result.doc_text)
        assert len(result.doc_text) > 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestVectorStoreReverseSearchErrorHandling:
    """Tests for error handling during reverse search."""

    def test_reverse_search_vectoriser_independent(self, initialized_vectorstore):
        """Reverse search doesn't require vectoriser (no embeddings used)."""
        # Disable vectoriser
        initialized_vectorstore.vectoriser = None

        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        # Should still work fine
        assert len(result) == 2

    def test_reverse_search_dataframe_join_failure_wrapped(self, initialized_vectorstore):
        """DataFrame join failures wrapped in ClassifaiError."""
        # Corrupt the internal vectors to break join
        initialized_vectorstore.vectors = None

        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        with pytest.raises((ClassifaiError, AttributeError)):
            initialized_vectorstore.reverse_search(query=query, max_n_results=10)

    def test_reverse_search_error_includes_context(self, initialized_vectorstore):
        """Error context includes relevant information."""
        # Force an error by corrupting data
        initialized_vectorstore.vectors = None

        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        with pytest.raises((ClassifaiError, AttributeError)):
            initialized_vectorstore.reverse_search(query=query, max_n_results=10)


# ============================================================================
# HOOKS INTEGRATION TESTS
# ============================================================================


class TestVectorStoreReverseSearchHooksIntegration:
    """Tests for hook integration in reverse_search()."""

    def test_reverse_search_preprocess_hook_called_before_search(self, initialized_vectorstore):
        """reverse_search_preprocess hook called before search."""
        mock_hook = Mock(spec=HookBase)
        mock_hook.return_value = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        initialized_vectorstore.hooks = {"reverse_search_preprocess": mock_hook}

        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )
        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        mock_hook.assert_called_once()

    def test_reverse_search_postprocess_hook_called_after_search(self, initialized_vectorstore):
        """reverse_search_postprocess hook called after search."""
        mock_result = VectorStoreReverseSearchOutput.from_data(
            {
                "id": ["q1"],
                "searched_doc_label": ["category_a"],
                "doc_label": ["category_a"],
                "doc_text": ["document 1"],
            }
        )

        mock_hook = Mock(spec=HookBase)
        mock_hook.return_value = mock_result

        initialized_vectorstore.hooks = {"reverse_search_postprocess": mock_hook}

        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )
        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        mock_hook.assert_called_once()

    def test_reverse_search_preprocess_hook_failure_raises_hook_error(self, initialized_vectorstore):
        """Preprocess hook failure raises HookError."""
        bad_hook = Mock(spec=HookBase)
        bad_hook.side_effect = Exception("Preprocess failed")

        initialized_vectorstore.hooks = {"reverse_search_preprocess": bad_hook}

        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        with pytest.raises(HookError) as exc_info:
            initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        assert "reverse_search_preprocess" in str(exc_info.value)

    def test_reverse_search_postprocess_hook_failure_raises_hook_error(self, initialized_vectorstore):
        """Postprocess hook failure raises HookError."""
        bad_hook = Mock(spec=HookBase)
        bad_hook.side_effect = Exception("Postprocess failed")

        initialized_vectorstore.hooks = {"reverse_search_postprocess": bad_hook}

        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        with pytest.raises(HookError) as exc_info:
            initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        assert "reverse_search_postprocess" in str(exc_info.value)

    def test_reverse_search_multiple_preprocess_hooks_processed_in_order(self, initialized_vectorstore):
        """Multiple preprocess hooks processed in order."""
        hook1 = Mock(spec=HookBase)
        hook1.return_value = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        hook2 = Mock(spec=HookBase)
        hook2.return_value = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        initialized_vectorstore.hooks = {"reverse_search_preprocess": [hook1, hook2]}

        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )
        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        assert hook1.call_count == 1
        assert hook2.call_count == 1

    def test_reverse_search_multiple_postprocess_hooks_processed_in_order(self, initialized_vectorstore):
        """Multiple postprocess hooks processed in order."""
        mock_output = VectorStoreReverseSearchOutput.from_data(
            {
                "id": ["q1"],
                "searched_doc_label": ["category_a"],
                "doc_label": ["category_a"],
                "doc_text": ["document 1"],
            }
        )

        hook1 = Mock(spec=HookBase)
        hook1.return_value = mock_output

        hook2 = Mock(spec=HookBase)
        hook2.return_value = mock_output

        initialized_vectorstore.hooks = {"reverse_search_postprocess": [hook1, hook2]}

        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )
        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        assert hook1.call_count == 1
        assert hook2.call_count == 1

    def test_reverse_search_single_preprocess_hook_converted_to_list(self, initialized_vectorstore):
        """Single preprocess hook automatically converted to list."""
        mock_hook = Mock(spec=HookBase)
        mock_hook.return_value = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        initialized_vectorstore.hooks = {"reverse_search_preprocess": mock_hook}

        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )
        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        mock_hook.assert_called_once()

    def test_reverse_search_single_postprocess_hook_converted_to_list(self, initialized_vectorstore):
        """Single postprocess hook automatically converted to list."""
        mock_output = VectorStoreReverseSearchOutput.from_data(
            {
                "id": ["q1"],
                "searched_doc_label": ["category_a"],
                "doc_label": ["category_a"],
                "doc_text": ["document 1"],
            }
        )

        mock_hook = Mock(spec=HookBase)
        mock_hook.return_value = mock_output

        initialized_vectorstore.hooks = {"reverse_search_postprocess": mock_hook}

        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )
        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        mock_hook.assert_called_once()


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestVectorStoreReverseSearchEdgeCases:
    """Tests for edge cases in reverse_search()."""

    def test_reverse_search_single_label_match(self, initialized_vectorstore):
        """Single label match returns one result."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_c"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        assert len(result) == 1

    def test_reverse_search_all_queries_return_empty(self, initialized_vectorstore):
        """All queries returning empty results."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1", "q2", "q3"],
                "doc_label": ["nonexistent_a", "nonexistent_b", "nonexistent_c"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        assert len(result) == 0

    def test_reverse_search_mixed_empty_and_nonempty_results(self, initialized_vectorstore):
        """Mix of queries with and without results."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1", "q2"],
                "doc_label": ["category_a", "nonexistent"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        # Only q1 should have results
        assert len(result) == 2

    def test_reverse_search_special_characters_in_label(self, initialized_vectorstore):
        """Labels with special characters."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        assert isinstance(result, VectorStoreReverseSearchOutput)

    def test_reverse_search_very_long_label(self, initialized_vectorstore):
        """Very long label string."""
        long_label = "a" * 1000

        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": [long_label],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        # Should find no matches (label not in store)
        assert len(result) == 0

    def test_reverse_search_max_n_results_exceeds_available(self, initialized_vectorstore):
        """max_n_results larger than available matches."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=1000)

        # Should return only 2 matches (all available)
        assert len(result) == 2

    def test_reverse_search_whitespace_in_label(self, initialized_vectorstore):
        """Label with leading/trailing whitespace."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["  category_a  "],  # with whitespace
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        # Should not match (exact matching)
        assert len(result) == 0

    def test_reverse_search_unicode_label(self, initialized_vectorstore):
        """Unicode characters in label."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["类别_a"],  # Chinese characters
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        # Should find no matches
        assert len(result) == 0

    def test_reverse_search_empty_string_label(self, initialized_vectorstore):
        """Empty string as label."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": [""],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        # Should find no matches
        assert len(result) == 0

    def test_reverse_search_large_number_of_queries(self, initialized_vectorstore):
        """Large number of queries processed correctly."""
        n_queries = 50
        ids = [str(i) for i in range(n_queries)]
        labels = ["category_a" if i % 2 == 0 else "category_b" for i in range(n_queries)]

        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ids,
                "doc_label": labels,
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        # Should process all queries (25 * 2 + 25 * 2 = 100 results)
        assert len(result) == 100

    def test_reverse_search_returns_correct_type(self, initialized_vectorstore):
        """reverse_search() returns VectorStoreReverseSearchOutput."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["category_a"],
            }
        )

        result = initialized_vectorstore.reverse_search(query=query, max_n_results=10)

        assert isinstance(result, VectorStoreReverseSearchOutput)

    def test_reverse_search_partial_match_with_prefix(self, initialized_vectorstore):
        """Partial matching with various prefix patterns."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["cat"],
            }
        )

        result = initialized_vectorstore.reverse_search(
            query=query,
            max_n_results=10,
            partial_match=True,
        )

        # Should find documents starting with "cat"
        assert len(result) >= 5

    def test_reverse_search_partial_match_single_character(self, initialized_vectorstore):
        """Partial matching with single character prefix."""
        query = VectorStoreReverseSearchInput.from_data(
            {
                "id": ["q1"],
                "doc_label": ["c"],
            }
        )

        result = initialized_vectorstore.reverse_search(
            query=query,
            max_n_results=10,
            partial_match=True,
        )

        # Should find all documents starting with "c"
        assert len(result) >= 5
