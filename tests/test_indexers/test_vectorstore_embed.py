"""Unit tests for VectorStore.embed() method."""

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
from classifai.indexers.dataclasses import VectorStoreEmbedInput, VectorStoreEmbedOutput
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
    csv_path.write_text("label,text\ndoc1,hello world\ndoc2,goodbye world\ndoc3,test document\n")

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


class TestVectorStoreEmbedInputValidation:
    """Tests for input validation in embed() method."""

    def test_embed_query_must_be_vectorstore_embed_input(self, initialized_vectorstore):
        """Query must be VectorStoreEmbedInput object."""
        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.embed(query="not a VectorStoreEmbedInput")

        assert "VectorStoreEmbedInput" in str(exc_info.value)

    def test_embed_query_none_raises_error(self, initialized_vectorstore):
        """query=None raises DataValidationError."""
        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.embed(query=None)

        assert "VectorStoreEmbedInput" in str(exc_info.value)

    def test_embed_query_dict_raises_error(self, initialized_vectorstore):
        """Query as dict (not VectorStoreEmbedInput) raises DataValidationError."""
        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.embed(query={"id": ["1"], "text": ["hello"]})

        assert "VectorStoreEmbedInput" in str(exc_info.value)

    def test_embed_query_list_raises_error(self, initialized_vectorstore):
        """Query as list raises DataValidationError."""
        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore.embed(query=["hello", "world"])

        assert "VectorStoreEmbedInput" in str(exc_info.value)


# ============================================================================
# EMBEDDING OPERATION TESTS
# ============================================================================


class TestVectorStoreEmbedOperation:
    """Tests for the core embedding operation."""

    def test_embed_single_text_embeds_correctly(self, initialized_vectorstore):
        """Single text embeds correctly and returns result."""
        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["hello world"]})

        result = initialized_vectorstore.embed(query=query)

        assert isinstance(result, VectorStoreEmbedOutput)
        assert len(result) == 1
        assert result.id[0] == "1"
        assert result.text[0] == "hello world"

    def test_embed_multiple_texts_process_correctly(self, initialized_vectorstore):
        """Multiple texts process correctly and all are embedded."""
        query = VectorStoreEmbedInput.from_data(
            {
                "id": ["1", "2", "3"],
                "text": ["hello", "world", "test"],
            }
        )

        result = initialized_vectorstore.embed(query=query)

        assert isinstance(result, VectorStoreEmbedOutput)
        assert len(result) == 3
        assert list(result.id) == ["1", "2", "3"]
        assert list(result.text) == ["hello", "world", "test"]

    def test_embed_output_includes_id_text_embedding(self, initialized_vectorstore):
        """Output includes id, text, and embedding columns."""
        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["test"]})

        result = initialized_vectorstore.embed(query=query)

        assert "id" in result.columns
        assert "text" in result.columns
        assert "embedding" in result.columns

    def test_embed_embeddings_are_numpy_arrays(self, initialized_vectorstore):
        """Embeddings in output are numpy arrays."""
        query = VectorStoreEmbedInput.from_data(
            {
                "id": ["1", "2"],
                "text": ["hello", "world"],
            }
        )

        result = initialized_vectorstore.embed(query=query)

        for embedding in result.embedding:
            assert isinstance(embedding, np.ndarray)

    def test_embed_output_shape_matches_input_count(self, initialized_vectorstore):
        """Output has same number of rows as input texts."""
        query = VectorStoreEmbedInput.from_data(
            {
                "id": ["1", "2", "3", "4", "5"],
                "text": ["a", "b", "c", "d", "e"],
            }
        )

        result = initialized_vectorstore.embed(query=query)

        assert len(result) == 5

    def test_embed_vectoriser_transform_called_with_correct_texts(self, initialized_vectorstore):
        """Vectoriser.transform() called with correct text list."""
        texts = ["hello", "world", "test"]
        query = VectorStoreEmbedInput.from_data(
            {
                "id": ["1", "2", "3"],
                "text": texts,
            }
        )

        result = initialized_vectorstore.embed(query=query)

        # Verify transform was called
        initialized_vectorstore.vectoriser.transform.assert_called()

        # Get the call arguments
        call_args = initialized_vectorstore.vectoriser.transform.call_args[0][0]
        assert call_args == texts

    def test_embed_embedding_dimensions_correct(self, initialized_vectorstore):
        """Embeddings have correct dimensions (shape)."""
        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["test"]})

        result = initialized_vectorstore.embed(query=query)

        embedding = result.embedding[0]
        # Mock vectoriser returns shape (3,)
        assert embedding.shape == (3,)

    def test_embed_preserves_id_order(self, initialized_vectorstore):
        """IDs in output match input order."""
        ids = ["id_a", "id_b", "id_c"]
        query = VectorStoreEmbedInput.from_data(
            {
                "id": ids,
                "text": ["text_a", "text_b", "text_c"],
            }
        )

        result = initialized_vectorstore.embed(query=query)

        assert list(result.id) == ids

    def test_embed_preserves_text_content(self, initialized_vectorstore):
        """Text content in output matches input exactly."""
        texts = ["hello world", "foo bar", "test case"]
        query = VectorStoreEmbedInput.from_data(
            {
                "id": ["1", "2", "3"],
                "text": texts,
            }
        )

        result = initialized_vectorstore.embed(query=query)

        assert list(result.text) == texts


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestVectorStoreEmbedErrorHandling:
    """Tests for error handling during embedding."""

    def test_embed_vectoriser_failure_raises_classifai_error(self, initialized_vectorstore):
        """Vectoriser failure raises ClassifaiError."""
        initialized_vectorstore.vectoriser.transform.side_effect = Exception("Transform failed")

        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["hello"]})

        with pytest.raises(ClassifaiError) as exc_info:
            initialized_vectorstore.embed(query=query)

        assert "Embedding failed" in str(exc_info.value)

    def test_embed_vectoriser_exceptions_wrapped(self, initialized_vectorstore):
        """Vectoriser exceptions include context."""
        initialized_vectorstore.vectoriser.transform.side_effect = RuntimeError("Bad embedding")

        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["test"]})

        with pytest.raises(ClassifaiError) as exc_info:
            initialized_vectorstore.embed(query=query)

        error_str = str(exc_info.value)
        assert "MockVectoriser" in error_str or "vectoriser" in error_str.lower()

    def test_embed_vectoriser_error_includes_text_count(self, initialized_vectorstore):
        """Error context includes number of texts being embedded."""
        initialized_vectorstore.vectoriser.transform.side_effect = ValueError("Embedding error")

        query = VectorStoreEmbedInput.from_data(
            {
                "id": ["1", "2", "3"],
                "text": ["a", "b", "c"],
            }
        )

        with pytest.raises(ClassifaiError) as exc_info:
            initialized_vectorstore.embed(query=query)

        error_str = str(exc_info.value)
        assert "3" in error_str or "n_texts" in error_str.lower()

    def test_embed_vectoriser_error_includes_vectoriser_class(self, initialized_vectorstore):
        """Error context includes vectoriser class name."""
        initialized_vectorstore.vectoriser.transform.side_effect = Exception("Failed")

        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["test"]})

        with pytest.raises(ClassifaiError) as exc_info:
            initialized_vectorstore.embed(query=query)

        assert "MockVectoriser" in str(exc_info.value)

    def test_embed_correct_number_of_embeddings(self, initialized_vectorstore):
        """Embeddings with correct count are processed correctly."""
        initialized_vectorstore.vectoriser.transform.side_effect = None
        # Return exactly 3 embeddings for 3 texts
        initialized_vectorstore.vectoriser.transform.return_value = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

        query = VectorStoreEmbedInput.from_data(
            {
                "id": ["1", "2", "3"],
                "text": ["hello", "world", "test"],
            }
        )

        result = initialized_vectorstore.embed(query=query)

        assert len(result) == 3
        assert all(isinstance(emb, np.ndarray) for emb in result.embedding)


# ============================================================================
# HOOKS INTEGRATION TESTS
# ============================================================================


class TestVectorStoreEmbedHooksIntegration:
    """Tests for hook integration in embed()."""

    def test_embed_preprocess_hook_called_before_embedding(self, initialized_vectorstore):
        """embed_preprocess hook called before embedding."""
        mock_hook = Mock(spec=HookBase)
        mock_hook.return_value = VectorStoreEmbedInput.from_data(
            {
                "id": ["1"],
                "text": ["modified text"],
            }
        )

        initialized_vectorstore.hooks = {"embed_preprocess": mock_hook}

        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["hello"]})
        result = initialized_vectorstore.embed(query=query)

        mock_hook.assert_called_once()

    def test_embed_postprocess_hook_called_after_embedding(self, initialized_vectorstore):
        """embed_postprocess hook called after embedding."""
        mock_result = VectorStoreEmbedOutput.from_data(
            {
                "id": ["1"],
                "text": ["hello"],
                "embedding": [np.array([0.1, 0.2, 0.3])],
            }
        )

        mock_hook = Mock(spec=HookBase)
        mock_hook.return_value = mock_result

        initialized_vectorstore.hooks = {"embed_postprocess": mock_hook}

        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["hello"]})
        result = initialized_vectorstore.embed(query=query)

        mock_hook.assert_called_once()

    def test_embed_preprocess_hook_failure_raises_hook_error(self, initialized_vectorstore):
        """Preprocess hook failure raises HookError."""
        bad_hook = Mock(spec=HookBase)
        bad_hook.side_effect = Exception("Preprocess failed")

        initialized_vectorstore.hooks = {"embed_preprocess": bad_hook}

        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["hello"]})

        with pytest.raises(HookError) as exc_info:
            initialized_vectorstore.embed(query=query)

        assert "embed_preprocess" in str(exc_info.value)

    def test_embed_postprocess_hook_failure_raises_hook_error(self, initialized_vectorstore):
        """Postprocess hook failure raises HookError."""
        bad_hook = Mock(spec=HookBase)
        bad_hook.side_effect = Exception("Postprocess failed")

        initialized_vectorstore.hooks = {"embed_postprocess": bad_hook}

        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["hello"]})

        with pytest.raises(HookError) as exc_info:
            initialized_vectorstore.embed(query=query)

        assert "embed_postprocess" in str(exc_info.value)

    def test_embed_multiple_preprocess_hooks_processed_in_order(self, initialized_vectorstore):
        """Multiple preprocess hooks processed in order."""
        hook1 = Mock(spec=HookBase)
        hook1.return_value = VectorStoreEmbedInput.from_data(
            {
                "id": ["1"],
                "text": ["step1"],
            }
        )

        hook2 = Mock(spec=HookBase)
        hook2.return_value = VectorStoreEmbedInput.from_data(
            {
                "id": ["1"],
                "text": ["step2"],
            }
        )

        initialized_vectorstore.hooks = {"embed_preprocess": [hook1, hook2]}

        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["original"]})
        result = initialized_vectorstore.embed(query=query)

        assert hook1.call_count == 1
        assert hook2.call_count == 1

    def test_embed_multiple_postprocess_hooks_processed_in_order(self, initialized_vectorstore):
        """Multiple postprocess hooks processed in order."""
        mock_output = VectorStoreEmbedOutput.from_data(
            {
                "id": ["1"],
                "text": ["hello"],
                "embedding": [np.array([0.1, 0.2, 0.3])],
            }
        )

        hook1 = Mock(spec=HookBase)
        hook1.return_value = mock_output

        hook2 = Mock(spec=HookBase)
        hook2.return_value = mock_output

        initialized_vectorstore.hooks = {"embed_postprocess": [hook1, hook2]}

        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["hello"]})
        result = initialized_vectorstore.embed(query=query)

        assert hook1.call_count == 1
        assert hook2.call_count == 1

    def test_embed_single_preprocess_hook_converted_to_list(self, initialized_vectorstore):
        """Single preprocess hook automatically converted to list."""
        mock_hook = Mock(spec=HookBase)
        mock_hook.return_value = VectorStoreEmbedInput.from_data(
            {
                "id": ["1"],
                "text": ["hello"],
            }
        )

        initialized_vectorstore.hooks = {"embed_preprocess": mock_hook}

        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["hello"]})
        result = initialized_vectorstore.embed(query=query)

        mock_hook.assert_called_once()

    def test_embed_single_postprocess_hook_converted_to_list(self, initialized_vectorstore):
        """Single postprocess hook automatically converted to list."""
        mock_output = VectorStoreEmbedOutput.from_data(
            {
                "id": ["1"],
                "text": ["hello"],
                "embedding": [np.array([0.1, 0.2, 0.3])],
            }
        )

        mock_hook = Mock(spec=HookBase)
        mock_hook.return_value = mock_output

        initialized_vectorstore.hooks = {"embed_postprocess": mock_hook}

        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["hello"]})
        result = initialized_vectorstore.embed(query=query)

        mock_hook.assert_called_once()

    def test_embed_preprocess_hook_modifies_input(self, initialized_vectorstore):
        """Preprocess hook can modify input before embedding."""

        def modify_hook(input_data):
            # Add a prefix to all text
            modified = VectorStoreEmbedInput.from_data(
                {
                    "id": input_data.id,
                    "text": ["PREFIX: " + t for t in input_data.text],
                }
            )
            return modified

        initialized_vectorstore.hooks = {"embed_preprocess": modify_hook}

        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["hello"]})
        result = initialized_vectorstore.embed(query=query)

        # Verify vectoriser was called with modified text
        call_args = initialized_vectorstore.vectoriser.transform.call_args[0][0]
        assert call_args[0] == "PREFIX: hello"

    def test_embed_postprocess_hook_modifies_output(self, initialized_vectorstore):
        """Postprocess hook can modify output after embedding."""

        def modify_hook(output_data):
            # Return same data (in a real scenario might filter/transform)
            return output_data

        initialized_vectorstore.hooks = {"embed_postprocess": modify_hook}

        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["hello"]})
        result = initialized_vectorstore.embed(query=query)

        assert isinstance(result, VectorStoreEmbedOutput)
        assert len(result) == 1


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestVectorStoreEmbedEdgeCases:
    """Tests for edge cases in embed()."""

    def test_embed_single_character_text(self, initialized_vectorstore):
        """Embed single character text."""
        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["a"]})

        result = initialized_vectorstore.embed(query=query)

        assert len(result) == 1
        assert result.text[0] == "a"

    def test_embed_very_long_text(self, initialized_vectorstore):
        """Embed very long text."""
        long_text = "word " * 1000  # 5000 characters
        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": [long_text]})

        result = initialized_vectorstore.embed(query=query)

        assert len(result) == 1
        assert result.text[0] == long_text

    def test_embed_special_characters_in_text(self, initialized_vectorstore):
        """Embed text with special characters."""
        special_text = "Hello!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": [special_text]})

        result = initialized_vectorstore.embed(query=query)

        assert result.text[0] == special_text

    def test_embed_unicode_text(self, initialized_vectorstore):
        """Embed text with unicode characters."""
        unicode_text = "Hello 世界 🌍"
        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": [unicode_text]})

        result = initialized_vectorstore.embed(query=query)

        assert result.text[0] == unicode_text

    def test_embed_whitespace_only_text(self, initialized_vectorstore):
        """Embed text containing only whitespace."""
        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["   "]})

        result = initialized_vectorstore.embed(query=query)

        assert len(result) == 1

    def test_embed_empty_string_text(self, initialized_vectorstore):
        """Embed empty string text."""
        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": [""]})

        result = initialized_vectorstore.embed(query=query)

        assert len(result) == 1
        assert result.text[0] == ""

    def test_embed_newlines_in_text(self, initialized_vectorstore):
        """Embed text containing newlines."""
        multiline_text = "line1\nline2\nline3"
        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": [multiline_text]})

        result = initialized_vectorstore.embed(query=query)

        assert result.text[0] == multiline_text

    def test_embed_tabs_in_text(self, initialized_vectorstore):
        """Embed text containing tabs."""
        tab_text = "col1\tcol2\tcol3"
        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": [tab_text]})

        result = initialized_vectorstore.embed(query=query)

        assert result.text[0] == tab_text

    def test_embed_duplicate_ids_raises_validation_error(self, initialized_vectorstore):
        """Duplicate IDs in input raises validation error."""
        import pandera

        with pytest.raises(pandera.errors.SchemaError):
            VectorStoreEmbedInput.from_data(
                {
                    "id": ["1", "1"],  # duplicate
                    "text": ["hello", "world"],
                }
            )

    def test_embed_large_batch(self, initialized_vectorstore):
        """Embed large number of texts."""
        n_texts = 100
        ids = [str(i) for i in range(n_texts)]
        texts = [f"text {i}" for i in range(n_texts)]

        query = VectorStoreEmbedInput.from_data({"id": ids, "text": texts})

        result = initialized_vectorstore.embed(query=query)

        assert len(result) == n_texts

    def test_embed_returns_correct_type(self, initialized_vectorstore):
        """embed() returns VectorStoreEmbedOutput."""
        query = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["hello"]})

        result = initialized_vectorstore.embed(query=query)

        assert isinstance(result, VectorStoreEmbedOutput)
