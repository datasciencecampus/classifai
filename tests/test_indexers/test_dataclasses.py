"""Unit tests for VectorStore dataclasses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from classifai.indexers import (
    VectorStoreEmbedInput,
    VectorStoreEmbedOutput,
    VectorStoreReverseSearchInput,
    VectorStoreReverseSearchOutput,
    VectorStoreSearchInput,
    VectorStoreSearchOutput,
)


class TestVectorStoreSearchInput:
    """Tests for VectorStoreSearchInput dataclass."""

    def test_init_from_dict_valid_data(self):
        """Valid dict with id and query columns should construct successfully."""
        data = {"id": ["1", "2"], "query": ["hello", "world"]}
        result = VectorStoreSearchInput(data)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["id", "query"]
        assert len(result) == 2

    def test_init_from_dataframe_valid_data(self):
        """Valid DataFrame with id and query columns should construct successfully."""
        df = pd.DataFrame({"id": ["1", "2"], "query": ["hello", "world"]})
        result = VectorStoreSearchInput(df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["id", "query"]
        assert len(result) == 2

    def test_init_missing_required_column_raises_schema_error(self):
        """Missing 'query' column should raise SchemaError."""
        data = {"id": ["1", "2"]}

        with pytest.raises(pa.errors.SchemaError):
            VectorStoreSearchInput(data)

    def test_init_coerces_int_id_to_string(self):
        """Integer id should be coerced to string due to coerce=True."""
        data = {"id": [1, 2], "query": ["hello", "world"]}
        result = VectorStoreSearchInput(data)

        # Verify the values were coerced to strings
        assert all(isinstance(x, str) for x in result["id"])
        assert list(result["id"]) == ["1", "2"]

    def test_property_id_returns_correct_series(self):
        """The id property should return the 'id' column as a Series."""
        data = {"id": ["1", "2"], "query": ["hello", "world"]}
        result = VectorStoreSearchInput(data)

        id_series = result.id
        assert isinstance(id_series, pd.Series)
        assert list(id_series) == ["1", "2"]

    def test_property_query_returns_correct_series(self):
        """The query property should return the 'query' column as a Series."""
        data = {"id": ["1", "2"], "query": ["hello", "world"]}
        result = VectorStoreSearchInput(data)

        query_series = result.query
        assert isinstance(query_series, pd.Series)
        assert list(query_series) == ["hello", "world"]

    def test_from_data_classmethod_valid_data(self):
        """from_data classmethod should construct from dict or DataFrame."""
        data = {"id": ["1", "2"], "query": ["hello", "world"]}
        result = VectorStoreSearchInput.from_data(data)

        assert isinstance(result, VectorStoreSearchInput)
        assert len(result) == 2

    def test_validate_classmethod_on_valid_data_returns_instance(self):
        """Validate classmethod should return a VectorStoreSearchInput instance."""
        df = pd.DataFrame({"id": ["1", "2"], "query": ["hello", "world"]})
        result = VectorStoreSearchInput.validate(df)

        assert isinstance(result, VectorStoreSearchInput)
        assert len(result) == 2


class TestVectorStoreSearchOutput:
    """Tests for VectorStoreSearchOutput dataclass."""

    def test_init_from_dict_valid_data(self):
        """Valid dict with all 6 required columns should construct successfully."""
        data = {
            "query_id": ["1", "1"],
            "query_text": ["what is AI?", "what is AI?"],
            "doc_label": ["doc1", "doc2"],
            "doc_text": [456, "Machine learning..."],
            "rank": [0, 1],
            "score": [0.95, 0.87],
        }
        result = VectorStoreSearchOutput(data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["query_id", "query_text", "doc_label", "doc_text", "rank", "score"]

    def test_init_missing_required_column_raises_schema_error(self):
        """Missing 'score' column should raise SchemaError."""
        data = {
            "query_id": ["1"],
            "query_text": ["query"],
            "doc_label": ["doc1"],
            "doc_text": ["text"],
            "rank": [0],
        }

        with pytest.raises(pa.errors.SchemaError):
            VectorStoreSearchOutput(data)

    def test_init_rank_less_than_zero_raises_schema_error(self):
        """Negative rank should raise SchemaError (rank >= 0 required)."""
        data = {
            "query_id": ["1"],
            "query_text": ["query"],
            "doc_label": ["doc1"],
            "doc_text": ["text"],
            "rank": [-1],
            "score": [0.9],
        }

        with pytest.raises(pa.errors.SchemaError):
            VectorStoreSearchOutput(data)

    def test_init_columns_ordered_preserved_for_same_query(self):
        """Multiple results for same query should be grouped consecutively."""
        data = {
            "query_id": ["1", "1", "2", "2"],
            "query_text": ["q1", "q1", "q2", "q2"],
            "doc_label": ["a", "b", "c", "d"],
            "doc_text": ["t1", "t2", "t3", "t4"],
            "rank": [0, 1, 0, 1],
            "score": [0.9, 0.8, 0.85, 0.75],
        }
        result = VectorStoreSearchOutput(data)

        # Verify grouping: all query_id="1" appear before query_id="2"
        query_ids = result.query_id.tolist()
        assert query_ids == ["1", "1", "2", "2"]

    def test_property_query_id_returns_correct_series(self):
        """The query_id property should return the 'query_id' column."""
        data = {
            "query_id": ["1", "2"],
            "query_text": ["q1", "q2"],
            "doc_label": ["a", "b"],
            "doc_text": ["text1", "text2"],
            "rank": [0, 0],
            "score": [0.9, 0.8],
        }
        result = VectorStoreSearchOutput(data)

        query_id_series = result.query_id
        assert isinstance(query_id_series, pd.Series)
        assert list(query_id_series) == ["1", "2"]

    def test_property_score_returns_correct_series(self):
        """The score property should return the 'score' column."""
        data = {
            "query_id": ["1"],
            "query_text": ["q1"],
            "doc_label": ["a"],
            "doc_text": ["text"],
            "rank": [0],
            "score": [0.95],
        }
        result = VectorStoreSearchOutput(data)

        score_series = result.score
        assert isinstance(score_series, pd.Series)
        assert list(score_series) == [0.95]

    def test_from_data_classmethod_valid_data(self):
        """from_data classmethod should construct from dict or DataFrame."""
        data = {
            "query_id": ["1"],
            "query_text": ["q1"],
            "doc_label": ["d1"],
            "doc_text": ["text"],
            "rank": [0],
            "score": [0.9],
        }
        result = VectorStoreSearchOutput.from_data(data)

        assert isinstance(result, VectorStoreSearchOutput)
        assert len(result) == 1

    def test_validate_classmethod_returns_instance(self):
        """Validate classmethod should return a VectorStoreSearchOutput instance."""
        df = pd.DataFrame(
            {
                "query_id": ["1"],
                "query_text": ["q1"],
                "doc_label": ["d1"],
                "doc_text": ["text"],
                "rank": [0],
                "score": [0.9],
            }
        )
        result = VectorStoreSearchOutput.validate(df)

        assert isinstance(result, VectorStoreSearchOutput)


class TestVectorStoreEmbedInput:
    """Tests for VectorStoreEmbedInput dataclass."""

    def test_init_from_dict_valid_data(self):
        """Valid dict with id and text columns should construct successfully."""
        data = {"id": ["1", "2"], "text": ["hello", "world"]}
        result = VectorStoreEmbedInput(data)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["id", "text"]
        assert len(result) == 2

    def test_init_missing_required_column_raises_schema_error(self):
        """Missing 'text' column should raise SchemaError."""
        data = {"id": ["1", "2"]}

        with pytest.raises(pa.errors.SchemaError):
            VectorStoreEmbedInput(data)

    def test_property_id_returns_correct_series(self):
        """The id property should return the 'id' column as a Series."""
        data = {"id": ["1", "2"], "text": ["hello", "world"]}
        result = VectorStoreEmbedInput(data)

        id_series = result.id
        assert isinstance(id_series, pd.Series)
        assert list(id_series) == ["1", "2"]

    def test_property_text_returns_correct_series(self):
        """The text property should return the 'text' column as a Series."""
        data = {"id": ["1", "2"], "text": ["hello", "world"]}
        result = VectorStoreEmbedInput(data)

        text_series = result.text
        assert isinstance(text_series, pd.Series)
        assert list(text_series) == ["hello", "world"]

    def test_from_data_classmethod_valid_data(self):
        """from_data classmethod should construct from dict or DataFrame."""
        data = {"id": ["1", "2"], "text": ["hello", "world"]}
        result = VectorStoreEmbedInput.from_data(data)

        assert isinstance(result, VectorStoreEmbedInput)
        assert len(result) == 2

    def test_validate_classmethod_returns_instance(self):
        """Validate classmethod should return a VectorStoreEmbedInput instance."""
        df = pd.DataFrame({"id": ["1"], "text": ["hello"]})
        result = VectorStoreEmbedInput.validate(df)

        assert isinstance(result, VectorStoreEmbedInput)


class TestVectorStoreEmbedOutput:
    """Tests for VectorStoreEmbedOutput dataclass."""

    def test_init_from_dict_valid_data(self):
        """Valid dict with id, text, and embedding columns should construct successfully."""
        data = {
            "id": ["1", "2"],
            "text": ["hello", "world"],
            "embedding": [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])],
        }
        result = VectorStoreEmbedOutput(data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["id", "text", "embedding"]

    def test_init_missing_required_column_raises_schema_error(self):
        """Missing 'text' column should raise SchemaError."""
        data = {
            "id": ["1"],
            "embedding": [np.array([0.1, 0.2])],
        }

        with pytest.raises(pa.errors.SchemaError):
            VectorStoreEmbedOutput(data)

    def test_init_non_array_embedding_raises_schema_error(self):
        """Non-numpy-array embedding should raise SchemaError."""
        data = {
            "id": ["1"],
            "text": ["hello"],
            "embedding": [[0.1, 0.2]],  # list, not numpy array
        }

        with pytest.raises(pa.errors.SchemaError):
            VectorStoreEmbedOutput(data)

    def test_property_id_returns_correct_series(self):
        """The id property should return the 'id' column as a Series."""
        data = {
            "id": ["1", "2"],
            "text": ["hello", "world"],
            "embedding": [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
        }
        result = VectorStoreEmbedOutput(data)

        id_series = result.id
        assert isinstance(id_series, pd.Series)
        assert list(id_series) == ["1", "2"]

    def test_property_embedding_returns_correct_series(self):
        """The embedding property should return the 'embedding' column."""
        embedding1 = np.array([0.1, 0.2])
        embedding2 = np.array([0.3, 0.4])
        data = {
            "id": ["1", "2"],
            "text": ["hello", "world"],
            "embedding": [embedding1, embedding2],
        }
        result = VectorStoreEmbedOutput(data)

        embedding_series = result.embedding
        assert isinstance(embedding_series, pd.Series)
        assert isinstance(embedding_series.iloc[0], np.ndarray)

    def test_from_data_classmethod_valid_data(self):
        """from_data classmethod should construct from dict or DataFrame."""
        data = {
            "id": ["1"],
            "text": ["hello"],
            "embedding": [np.array([0.1, 0.2])],
        }
        result = VectorStoreEmbedOutput.from_data(data)

        assert isinstance(result, VectorStoreEmbedOutput)
        assert len(result) == 1


class TestVectorStoreReverseSearchInput:
    """Tests for VectorStoreReverseSearchInput dataclass."""

    def test_init_from_dict_valid_data(self):
        """Valid dict with id and doc_label columns should construct successfully."""
        data = {"id": ["1", "2"], "doc_label": ["label1", "label2"]}
        result = VectorStoreReverseSearchInput(data)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["id", "doc_label"]
        assert len(result) == 2

    def test_init_missing_required_column_raises_schema_error(self):
        """Missing 'doc_label' column should raise SchemaError."""
        data = {"id": ["1", "2"]}

        with pytest.raises(pa.errors.SchemaError):
            VectorStoreReverseSearchInput(data)

    def test_property_id_returns_correct_series(self):
        """The id property should return the 'id' column as a Series."""
        data = {"id": ["1", "2"], "doc_label": ["label1", "label2"]}
        result = VectorStoreReverseSearchInput(data)

        id_series = result.id
        assert isinstance(id_series, pd.Series)
        assert list(id_series) == ["1", "2"]

    def test_property_doc_label_returns_correct_series(self):
        """The doc_label property should return the 'doc_label' column as a Series."""
        data = {"id": ["1", "2"], "doc_label": ["label1", "label2"]}
        result = VectorStoreReverseSearchInput(data)

        doc_label_series = result.doc_label
        assert isinstance(doc_label_series, pd.Series)
        assert list(doc_label_series) == ["label1", "label2"]

    def test_from_data_classmethod_valid_data(self):
        """from_data classmethod should construct from dict or DataFrame."""
        data = {"id": ["1", "2"], "doc_label": ["label1", "label2"]}
        result = VectorStoreReverseSearchInput.from_data(data)

        assert isinstance(result, VectorStoreReverseSearchInput)
        assert len(result) == 2

    # TODO: Uncomment after unique=True constraints merged to main - possibly needed for other dataclass tests as well depending on final implementation of ticket-167
    # def test_init_duplicate_ids_raises_schema_error(self):
    #     """Duplicate 'id' values should raise SchemaError."""
    #     data = {"id": ["1", "1"], "doc_label": ["label1", "label2"]}
    #
    #     with pytest.raises(pa.errors.SchemaError):
    #         VectorStoreReverseSearchInput(data)


class TestVectorStoreReverseSearchOutput:
    """Tests for VectorStoreReverseSearchOutput dataclass."""

    def test_init_from_dict_valid_data(self):
        """Valid dict with all 4 required columns should construct successfully."""
        data = {
            "id": ["1", "1"],
            "searched_doc_label": ["label1", "label1"],
            "doc_label": ["label1", "label1"],
            "doc_text": ["text1", "text2"],
        }
        result = VectorStoreReverseSearchOutput(data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["id", "searched_doc_label", "doc_label", "doc_text"]

    def test_init_empty_dict_creates_valid_structure_with_columns(self):
        """Empty dict should create a DataFrame with correct columns and schema."""
        result = VectorStoreReverseSearchOutput({})

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        # Verify all expected columns exist even when empty
        expected_cols = ["id", "searched_doc_label", "doc_label", "doc_text"]
        assert all(col in result.columns for col in expected_cols)

    def test_init_missing_required_column_raises_schema_error(self):
        """Missing 'doc_text' column should raise SchemaError."""
        data = {
            "id": ["1"],
            "searched_doc_label": ["label"],
            "doc_label": ["label"],
        }

        with pytest.raises(pa.errors.SchemaError):
            VectorStoreReverseSearchOutput(data)

    def test_property_id_returns_correct_series(self):
        """The id property should return the 'id' column as a Series."""
        data = {
            "id": ["1", "2"],
            "searched_doc_label": ["label1", "label2"],
            "doc_label": ["label1", "label2"],
            "doc_text": ["text1", "text2"],
        }
        result = VectorStoreReverseSearchOutput(data)

        id_series = result.id
        assert isinstance(id_series, pd.Series)
        assert list(id_series) == ["1", "2"]

    def test_property_searched_doc_label_returns_correct_series(self):
        """The searched_doc_label property should return the 'searched_doc_label' column."""
        data = {
            "id": ["1", "1"],
            "searched_doc_label": ["label1", "label1"],
            "doc_label": ["label1", "label1"],
            "doc_text": ["text1", "text2"],
        }
        result = VectorStoreReverseSearchOutput(data)

        searched_doc_label_series = result.searched_doc_label
        assert isinstance(searched_doc_label_series, pd.Series)
        assert list(searched_doc_label_series) == ["label1", "label1"]

    def test_from_data_classmethod_empty_data(self):
        """from_data should handle empty data correctly and create valid columns."""
        result = VectorStoreReverseSearchOutput.from_data({})

        assert isinstance(result, VectorStoreReverseSearchOutput)
        assert len(result) == 0
        expected_cols = ["id", "searched_doc_label", "doc_label", "doc_text"]
        assert all(col in result.columns for col in expected_cols)

    def test_from_data_classmethod_valid_data(self):
        """from_data classmethod should construct from dict or DataFrame."""
        data = {
            "id": ["1"],
            "searched_doc_label": ["label"],
            "doc_label": ["label"],
            "doc_text": ["text"],
        }
        result = VectorStoreReverseSearchOutput.from_data(data)

        assert isinstance(result, VectorStoreReverseSearchOutput)
        assert len(result) == 1

    def test_validate_classmethod_returns_instance(self):
        """Validate classmethod should return a VectorStoreReverseSearchOutput instance."""
        df = pd.DataFrame(
            {
                "id": ["1"],
                "searched_doc_label": ["label"],
                "doc_label": ["label"],
                "doc_text": ["text"],
            }
        )
        result = VectorStoreReverseSearchOutput.validate(df)

        assert isinstance(result, VectorStoreReverseSearchOutput)
