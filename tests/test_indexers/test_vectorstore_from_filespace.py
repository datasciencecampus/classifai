"""Unit tests for VectorStore.from_filespace() class method."""

from __future__ import annotations

import json
import logging
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest

from classifai.exceptions import (
    ConfigurationError,
    DataValidationError,
    IndexBuildError,
)
from classifai.indexers import VectorStore
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
def saved_vectorstore(mock_vectoriser, tmp_path):
    """Create and save a VectorStore to disk for loading tests."""
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "label,text,source\ncat_a,hello world,src1\ncat_a,goodbye world,src1\ncat_b,test document,src2\n"
    )

    output_dir = tmp_path / "vectorstore_output"

    vs = VectorStore(
        file_name=str(csv_path),
        data_type="csv",
        vectoriser=mock_vectoriser,
        meta_data={"source": str},
        batch_size=64,
        output_dir=str(output_dir),
        skip_save=False,
    )

    return output_dir, mock_vectoriser


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================


class TestVectorStoreFromFilespaceInputValidation:
    """Tests for input validation in from_filespace()."""

    def test_from_filespace_folder_path_must_be_string(self, mock_vectoriser):
        """folder_path must be string."""
        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(folder_path=123, vectoriser=mock_vectoriser)

        assert "folder_path" in str(exc_info.value).lower()

    def test_from_filespace_folder_path_must_be_non_empty(self, mock_vectoriser):
        """folder_path must be non-empty string."""
        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(folder_path="", vectoriser=mock_vectoriser)

        assert "folder_path" in str(exc_info.value).lower()

    def test_from_filespace_folder_path_must_exist(self, mock_vectoriser):
        """folder_path must be existing directory."""
        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(
                folder_path="/nonexistent/path/to/vectorstore",
                vectoriser=mock_vectoriser,
            )

        assert "folder_path" in str(exc_info.value).lower() or "exist" in str(exc_info.value).lower()

    def test_from_filespace_batch_size_must_be_positive_int_or_none(self, saved_vectorstore):
        """batch_size must be int >= 1 or None."""
        output_dir, mock_vectoriser = saved_vectorstore

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
                batch_size=0,
            )

        assert "batch_size" in str(exc_info.value).lower()

    def test_from_filespace_batch_size_negative_raises_error(self, saved_vectorstore):
        """batch_size < 1 raises DataValidationError."""
        output_dir, mock_vectoriser = saved_vectorstore

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
                batch_size=-5,
            )

        assert "batch_size" in str(exc_info.value).lower()

    def test_from_filespace_batch_size_none_is_valid(self, saved_vectorstore):
        """batch_size=None is valid."""
        output_dir, mock_vectoriser = saved_vectorstore

        # Should not raise
        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
            batch_size=None,
        )
        assert vs is not None

    def test_from_filespace_batch_size_non_int_raises_error(self, saved_vectorstore):
        """batch_size as non-int raises DataValidationError."""
        output_dir, mock_vectoriser = saved_vectorstore

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
                batch_size="64",
            )

        assert "batch_size" in str(exc_info.value).lower()

    def test_from_filespace_hooks_must_be_dict_or_none(self, saved_vectorstore):
        """Hooks must be dict or None."""
        output_dir, mock_vectoriser = saved_vectorstore

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
                hooks="not a dict",
            )

        assert "hooks" in str(exc_info.value).lower()

    def test_from_filespace_hooks_list_raises_error(self, saved_vectorstore):
        """Hooks as list raises DataValidationError."""
        output_dir, mock_vectoriser = saved_vectorstore

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
                hooks=[],
            )

        assert "hooks" in str(exc_info.value).lower()

    def test_from_filespace_vectoriser_must_have_transform_method(self, saved_vectorstore):
        """Vectoriser must have callable .transform() method."""
        output_dir, _ = saved_vectorstore

        bad_vectoriser = Mock()
        bad_vectoriser.transform = None  # Not callable
        bad_vectoriser.__class__.__name__ = "BadVectoriser"

        with pytest.raises(ConfigurationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=bad_vectoriser,
            )

        assert "transform" in str(exc_info.value).lower()

    def test_from_filespace_vectoriser_without_transform_attribute(self, saved_vectorstore):
        """Vectoriser without .transform attribute raises ConfigurationError."""
        output_dir, _ = saved_vectorstore

        bad_vectoriser = Mock(spec=[])  # No attributes
        bad_vectoriser.__class__.__name__ = "BadVectoriser"

        with pytest.raises(ConfigurationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=bad_vectoriser,
            )

        assert "transform" in str(exc_info.value).lower()


# ============================================================================
# FILE LOADING TESTS
# ============================================================================


class TestVectorStoreFromFilespaceFileLoading:
    """Tests for file loading and validation."""

    def test_from_filespace_metadata_json_must_exist(self, mock_vectoriser, tmp_path):
        """metadata.json must exist in folder_path."""
        output_dir = tmp_path / "empty_folder"
        output_dir.mkdir()

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
            )

        assert "metadata" in str(exc_info.value).lower()

    def test_from_filespace_vectors_parquet_must_exist(self, mock_vectoriser, tmp_path):
        """vectors.parquet must exist in folder_path."""
        output_dir = tmp_path / "no_vectors"
        output_dir.mkdir()

        # Create only metadata.json
        metadata = {
            "vectoriser_class": "MockVectoriser",
            "vector_shape": 3,
            "num_vectors": 1,
            "batch_size": 128,
            "created_at": 1000.0,
            "meta_data": {},
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
            )

        assert "parquet" in str(exc_info.value).lower() or "vectors" in str(exc_info.value).lower()

    def test_from_filespace_vectors_parquet_must_not_be_empty(self, mock_vectoriser, tmp_path):
        """vectors.parquet must not be empty."""
        output_dir = tmp_path / "empty_parquet"
        output_dir.mkdir()

        # Create metadata.json
        metadata = {
            "vectoriser_class": "MockVectoriser",
            "vector_shape": 3,
            "num_vectors": 0,
            "batch_size": 128,
            "created_at": 1000.0,
            "meta_data": {},
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create empty parquet file
        empty_df = pl.DataFrame(
            schema={
                "label": pl.Utf8,
                "text": pl.Utf8,
                "embeddings": pl.List(pl.Float32),
                "uuid": pl.Utf8,
            }
        )
        empty_df.write_parquet(output_dir / "vectors.parquet")

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
            )

        assert "empty" in str(exc_info.value).lower()

    def test_from_filespace_malformed_metadata_json_raises_error(self, mock_vectoriser, tmp_path):
        """Malformed metadata.json raises IndexBuildError."""
        output_dir = tmp_path / "bad_metadata"
        output_dir.mkdir()

        # Write invalid JSON
        with open(output_dir / "metadata.json", "w") as f:
            f.write("{invalid json")

        with pytest.raises(IndexBuildError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
            )

        assert "metadata" in str(exc_info.value).lower()

    def test_from_filespace_metadata_not_dict_raises_error(self, mock_vectoriser, tmp_path):
        """metadata.json not containing dict raises DataValidationError."""
        output_dir = tmp_path / "metadata_list"
        output_dir.mkdir()

        # Write JSON array instead of object
        with open(output_dir / "metadata.json", "w") as f:
            json.dump([], f)

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
            )

        assert "object" in str(exc_info.value).lower() or "dict" in str(exc_info.value).lower()

    def test_from_filespace_missing_required_metadata_keys(self, mock_vectoriser, tmp_path):
        """Missing required metadata keys raises DataValidationError."""
        output_dir = tmp_path / "incomplete_metadata"
        output_dir.mkdir()

        # Create incomplete metadata (missing vector_shape)
        metadata = {
            "vectoriser_class": "MockVectoriser",
            "num_vectors": 1,
            "created_at": 1000.0,
            "meta_data": {},
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
            )

        assert "missing" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()

    def test_from_filespace_required_columns_in_parquet(self, saved_vectorstore):
        """Required columns must be present in parquet."""
        output_dir, mock_vectoriser = saved_vectorstore

        # Load existing parquet and remove a required column
        df = pl.read_parquet(output_dir / "vectors.parquet")
        df_missing = df.drop("uuid")
        df_missing.write_parquet(output_dir / "vectors.parquet")

        with pytest.raises((DataValidationError, IndexBuildError)) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
            )

        error_message = str(exc_info.value).lower()
        assert "missing" in error_message or "column" in error_message or "unable to find" in error_message


# ============================================================================
# CONFIGURATION VALIDATION TESTS
# ============================================================================


class TestVectorStoreFromFilespaceConfigurationValidation:
    """Tests for configuration validation."""

    def test_from_filespace_vectoriser_class_must_match_metadata(self, mock_vectoriser, tmp_path):
        """Vectoriser class name must match metadata."""
        output_dir = tmp_path / "class_mismatch"
        output_dir.mkdir()

        # Create metadata with different vectoriser class
        metadata = {
            "vectoriser_class": "DifferentVectoriser",
            "vector_shape": 3,
            "num_vectors": 1,
            "batch_size": 128,
            "created_at": 1000.0,
            "meta_data": {},
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create parquet file
        df = pl.DataFrame(
            {
                "label": ["test"],
                "text": ["hello"],
                "embeddings": [np.array([0.1, 0.2, 0.3])],
                "uuid": ["uuid1"],
            }
        )
        df.write_parquet(output_dir / "vectors.parquet")

        with pytest.raises(ConfigurationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
            )

        assert "vectoriser" in str(exc_info.value).lower() or "class" in str(exc_info.value).lower()

    def test_from_filespace_meta_data_must_be_dict(self, mock_vectoriser, tmp_path):
        """metadata.meta_data must be dict."""
        output_dir = tmp_path / "bad_meta_data"
        output_dir.mkdir()

        # Create metadata with non-dict meta_data
        metadata = {
            "vectoriser_class": "MockVectoriser",
            "vector_shape": 3,
            "num_vectors": 1,
            "batch_size": 128,
            "created_at": 1000.0,
            "meta_data": "not a dict",
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
            )

        assert "meta_data" in str(exc_info.value).lower()


# ============================================================================
# METADATA DESERIALIZATION TESTS
# ============================================================================


class TestVectorStoreFromFilespaceMetadataDeserialization:
    """Tests for metadata deserialization."""

    def test_from_filespace_type_deserialization_str(self, mock_vectoriser, tmp_path):
        """Type deserialization works for str."""
        output_dir = tmp_path / "deserialize_str"
        output_dir.mkdir()

        metadata = {
            "vectoriser_class": "MockVectoriser",
            "vector_shape": 3,
            "num_vectors": 1,
            "batch_size": 128,
            "created_at": 1000.0,
            "meta_data": {"source": "str"},
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        df = pl.DataFrame(
            {
                "label": ["test"],
                "text": ["hello"],
                "embeddings": [np.array([0.1, 0.2, 0.3])],
                "uuid": ["uuid1"],
                "source": ["src1"],
            }
        )
        df.write_parquet(output_dir / "vectors.parquet")

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs.meta_data["source"] == "str"

    def test_from_filespace_type_deserialization_int(self, mock_vectoriser, tmp_path):
        """Type deserialization works for int."""
        output_dir = tmp_path / "deserialize_int"
        output_dir.mkdir()

        metadata = {
            "vectoriser_class": "MockVectoriser",
            "vector_shape": 3,
            "num_vectors": 1,
            "batch_size": 128,
            "created_at": 1000.0,
            "meta_data": {"count": "int"},
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        df = pl.DataFrame(
            {
                "label": ["test"],
                "text": ["hello"],
                "embeddings": [np.array([0.1, 0.2, 0.3])],
                "uuid": ["uuid1"],
                "count": [5],
            }
        )
        df.write_parquet(output_dir / "vectors.parquet")

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs.meta_data["count"] == "int"

    def test_from_filespace_type_deserialization_float(self, mock_vectoriser, tmp_path):
        """Type deserialization works for float."""
        output_dir = tmp_path / "deserialize_float"
        output_dir.mkdir()

        metadata = {
            "vectoriser_class": "MockVectoriser",
            "vector_shape": 3,
            "num_vectors": 1,
            "batch_size": 128,
            "created_at": 1000.0,
            "meta_data": {"score": "float"},
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        df = pl.DataFrame(
            {
                "label": ["test"],
                "text": ["hello"],
                "embeddings": [np.array([0.1, 0.2, 0.3])],
                "uuid": ["uuid1"],
                "score": [0.95],
            }
        )
        df.write_parquet(output_dir / "vectors.parquet")

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs.meta_data["score"] == "float"

    def test_from_filespace_empty_meta_data_dict(self, mock_vectoriser, tmp_path):
        """Empty meta_data dict handled correctly."""
        output_dir = tmp_path / "empty_meta_data"
        output_dir.mkdir()

        metadata = {
            "vectoriser_class": "MockVectoriser",
            "vector_shape": 3,
            "num_vectors": 1,
            "batch_size": 128,
            "created_at": 1000.0,
            "meta_data": {},
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        df = pl.DataFrame(
            {
                "label": ["test"],
                "text": ["hello"],
                "embeddings": [np.array([0.1, 0.2, 0.3])],
                "uuid": ["uuid1"],
            }
        )
        df.write_parquet(output_dir / "vectors.parquet")

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs.meta_data == {}


# ============================================================================
# BACKWARDS COMPATIBILITY TESTS
# ============================================================================


class TestVectorStoreFromFilespaceBackwardsCompatibility:
    """Tests for backwards compatibility with v1.0.0."""

    def test_from_filespace_missing_batch_size_uses_provided(self, mock_vectoriser, tmp_path):
        """Missing batch_size uses provided value."""
        output_dir = tmp_path / "missing_batch_size"
        output_dir.mkdir()

        # Create metadata without batch_size (v1.0.0)
        metadata = {
            "vectoriser_class": "MockVectoriser",
            "vector_shape": 3,
            "num_vectors": 1,
            "created_at": 1000.0,
            "meta_data": {},
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        df = pl.DataFrame(
            {
                "label": ["test"],
                "text": ["hello"],
                "embeddings": [np.array([0.1, 0.2, 0.3])],
                "uuid": ["uuid1"],
            }
        )
        df.write_parquet(output_dir / "vectors.parquet")

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
            batch_size=256,
        )

        assert vs.batch_size == 256

    def test_from_filespace_missing_batch_size_uses_default(self, mock_vectoriser, tmp_path):
        """Missing batch_size uses default when not provided."""
        output_dir = tmp_path / "missing_batch_size_default"
        output_dir.mkdir()

        # Create metadata without batch_size (v1.0.0)
        metadata = {
            "vectoriser_class": "MockVectoriser",
            "vector_shape": 3,
            "num_vectors": 1,
            "created_at": 1000.0,
            "meta_data": {},
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        df = pl.DataFrame(
            {
                "label": ["test"],
                "text": ["hello"],
                "embeddings": [np.array([0.1, 0.2, 0.3])],
                "uuid": ["uuid1"],
            }
        )
        df.write_parquet(output_dir / "vectors.parquet")

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        # Default should be 128
        assert vs.batch_size == 128

    def test_from_filespace_warning_logged_for_missing_batch_size(self, mock_vectoriser, tmp_path, caplog):
        """Warning logged when batch_size missing from metadata."""
        output_dir = tmp_path / "warning_batch_size"
        output_dir.mkdir()

        # Create metadata without batch_size (v1.0.0)
        metadata = {
            "vectoriser_class": "MockVectoriser",
            "vector_shape": 3,
            "num_vectors": 1,
            "created_at": 1000.0,
            "meta_data": {},
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        df = pl.DataFrame(
            {
                "label": ["test"],
                "text": ["hello"],
                "embeddings": [np.array([0.1, 0.2, 0.3])],
                "uuid": ["uuid1"],
            }
        )
        df.write_parquet(output_dir / "vectors.parquet")

        with caplog.at_level(logging.WARNING):
            vs = VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
            )

        warning_messages = [record.message.lower() for record in caplog.records if record.levelname == "WARNING"]
        assert any("outdated" in msg or "batch_size" in msg for msg in warning_messages)


# ============================================================================
# INSTANCE CONSTRUCTION TESTS
# ============================================================================


class TestVectorStoreFromFilespaceInstanceConstruction:
    """Tests for instance construction and attribute assignment."""

    def test_from_filespace_instance_created_without_init(self, saved_vectorstore):
        """Instance created via object.__new__() (not calling __init__)."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        # If __init__ was called, it would fail because file_name/data_type are None
        assert vs is not None
        assert isinstance(vs, VectorStore)

    def test_from_filespace_file_name_is_none(self, saved_vectorstore):
        """file_name attribute is None."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs.file_name is None

    def test_from_filespace_data_type_is_none(self, saved_vectorstore):
        """data_type attribute is None."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs.data_type is None

    def test_from_filespace_vectoriser_attached(self, saved_vectorstore):
        """Vectoriser instance attached correctly."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs.vectoriser is mock_vectoriser

    def test_from_filespace_batch_size_override_priority(self, saved_vectorstore):
        """batch_size override has priority over metadata."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
            batch_size=256,
        )

        # Override should take precedence
        assert vs.batch_size == 256

    def test_from_filespace_meta_data_deserialized(self, saved_vectorstore):
        """meta_data deserialized and set correctly."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert isinstance(vs.meta_data, dict)
        assert "source" in vs.meta_data
        assert vs.meta_data["source"] == "str"

    def test_from_filespace_vectors_loaded(self, saved_vectorstore):
        """Vectors DataFrame loaded from parquet."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs.vectors is not None
        assert isinstance(vs.vectors, pl.DataFrame)
        assert len(vs.vectors) == 3

    def test_from_filespace_vector_shape_set(self, saved_vectorstore):
        """vector_shape attribute set from metadata."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs.vector_shape == 3

    def test_from_filespace_num_vectors_set(self, saved_vectorstore):
        """num_vectors attribute set from metadata."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs.num_vectors == 3

    def test_from_filespace_vectoriser_class_set(self, saved_vectorstore):
        """vectoriser_class attribute set from metadata."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs.vectoriser_class == "MockVectoriser"

    def test_from_filespace_hooks_applied(self, saved_vectorstore):
        """Hooks parameter applied correctly."""
        output_dir, mock_vectoriser = saved_vectorstore

        hooks = {"custom_hook": lambda x: x}

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
            hooks=hooks,
        )

        assert vs.hooks == hooks

    def test_from_filespace_hooks_default_empty_dict(self, saved_vectorstore):
        """Hooks defaults to empty dict when None."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
            hooks=None,
        )

        assert vs.hooks == {}

    def test_from_filespace_quiet_mode_applied(self, saved_vectorstore):
        """quiet_mode parameter applied correctly."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
            quiet_mode=True,
        )

        assert vs.quiet_mode is True


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestVectorStoreFromFilespaceEdgeCases:
    """Tests for edge cases in from_filespace()."""

    def test_from_filespace_large_batch_size_override(self, saved_vectorstore):
        """Very large batch_size override handled correctly."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
            batch_size=10000,
        )

        assert vs.batch_size == 10000

    def test_from_filespace_batch_size_one(self, saved_vectorstore):
        """batch_size=1 is valid."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
            batch_size=1,
        )

        assert vs.batch_size == 1

    def test_from_filespace_multiple_metadata_columns(self, mock_vectoriser, tmp_path):
        """Multiple metadata columns deserialized correctly."""
        output_dir = tmp_path / "multi_meta"
        output_dir.mkdir()

        metadata = {
            "vectoriser_class": "MockVectoriser",
            "vector_shape": 3,
            "num_vectors": 1,
            "batch_size": 128,
            "created_at": 1000.0,
            "meta_data": {"source": "str", "count": "int", "score": "float"},
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        df = pl.DataFrame(
            {
                "label": ["test"],
                "text": ["hello"],
                "embeddings": [np.array([0.1, 0.2, 0.3])],
                "uuid": ["uuid1"],
                "source": ["src1"],
                "count": [5],
                "score": [0.95],
            }
        )
        df.write_parquet(output_dir / "vectors.parquet")

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs.meta_data["source"] == "str"
        assert vs.meta_data["count"] == "int"
        assert vs.meta_data["score"] == "float"

    def test_from_filespace_many_documents(self, mock_vectoriser, tmp_path):
        """Large number of documents loaded correctly."""
        output_dir = tmp_path / "many_docs"
        output_dir.mkdir()

        metadata = {
            "vectoriser_class": "MockVectoriser",
            "vector_shape": 3,
            "num_vectors": 100,
            "batch_size": 128,
            "created_at": 1000.0,
            "meta_data": {},
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create 100 documents
        data = {
            "label": [f"cat_{i % 3}" for i in range(100)],
            "text": [f"document {i}" for i in range(100)],
            "embeddings": [np.random.rand(3) for _ in range(100)],
            "uuid": [f"uuid_{i}" for i in range(100)],
        }
        df = pl.DataFrame(data)
        df.write_parquet(output_dir / "vectors.parquet")

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs.num_vectors == 100
        assert len(vs.vectors) == 100

    def test_from_filespace_returns_vectorstore_instance(self, saved_vectorstore):
        """from_filespace() returns VectorStore instance."""
        output_dir, mock_vectoriser = saved_vectorstore

        vs = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert isinstance(vs, VectorStore)
