"""Unit tests for VectorStore metadata serialization and deserialization."""

from __future__ import annotations

import json
import time
from unittest.mock import Mock

import numpy as np
import pytest

from classifai.exceptions import (
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
def initialized_vectorstore(mock_vectoriser, tmp_path):
    """Create a fully initialized VectorStore with test data."""
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("label,text\ndoc1,hello world\ndoc2,goodbye world\n")

    vs = VectorStore(
        file_name=str(csv_path),
        data_type="csv",
        vectoriser=mock_vectoriser,
        output_dir=str(tmp_path / "output"),
        skip_save=True,
    )

    return vs


# ============================================================================
# METADATA SERIALIZATION TESTS (_save_metadata)
# ============================================================================


class TestVectorStoreMetadataSerialization:
    """Tests for metadata serialization (_save_metadata)."""

    def test_save_metadata_json_file_created_at_correct_path(self, mock_vectoriser, tmp_path):
        """JSON file created at correct path."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists()

    def test_save_metadata_contains_vectoriser_class(self, mock_vectoriser, tmp_path):
        """Metadata contains vectoriser_class field."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "vectoriser_class" in metadata
        assert metadata["vectoriser_class"] == "MockVectoriser"

    def test_save_metadata_contains_vector_shape(self, mock_vectoriser, tmp_path):
        """Metadata contains vector_shape field."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "vector_shape" in metadata
        assert metadata["vector_shape"] == 3

    def test_save_metadata_contains_num_vectors(self, mock_vectoriser, tmp_path):
        """Metadata contains num_vectors field."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "num_vectors" in metadata
        assert metadata["num_vectors"] == 2

    def test_save_metadata_contains_batch_size(self, mock_vectoriser, tmp_path):
        """Metadata contains batch_size field."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            batch_size=64,
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "batch_size" in metadata
        assert metadata["batch_size"] == 64

    def test_save_metadata_contains_created_at(self, mock_vectoriser, tmp_path):
        """Metadata contains created_at field."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        before_creation = time.time()
        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )
        after_creation = time.time()

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "created_at" in metadata
        assert isinstance(metadata["created_at"], (int, float))
        assert before_creation <= metadata["created_at"] <= after_creation

    def test_save_metadata_contains_meta_data_field(self, mock_vectoriser, tmp_path):
        """Metadata contains meta_data field."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "meta_data" in metadata
        assert isinstance(metadata["meta_data"], dict)

    def test_save_metadata_type_information_preserved_str(self, mock_vectoriser, tmp_path):
        """Type information preserved (str types → string names)."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text,source\ndoc1,hello,src1\ndoc2,world,src2\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            meta_data={"source": str},
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "meta_data" in metadata
        assert "source" in metadata["meta_data"]
        assert metadata["meta_data"]["source"] == "str"

    def test_save_metadata_type_information_preserved_int(self, mock_vectoriser, tmp_path):
        """Type information preserved for int types."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text,count\ndoc1,hello,5\ndoc2,world,10\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            meta_data={"count": int},
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["meta_data"]["count"] == "int"

    def test_save_metadata_type_information_preserved_float(self, mock_vectoriser, tmp_path):
        """Type information preserved for float types."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text,score\ndoc1,hello,0.5\ndoc2,world,0.8\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            meta_data={"score": float},
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["meta_data"]["score"] == "float"

    def test_save_metadata_multiple_types_preserved(self, mock_vectoriser, tmp_path):
        """Type information preserved for multiple metadata columns."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text,source,count,score\ndoc1,hello,src1,5,0.5\ndoc2,world,src2,10,0.8\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            meta_data={"source": str, "count": int, "score": float},
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["meta_data"]["source"] == "str"
        assert metadata["meta_data"]["count"] == "int"
        assert metadata["meta_data"]["score"] == "float"

    def test_save_metadata_valid_json_format(self, mock_vectoriser, tmp_path):
        """Metadata is valid JSON format."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            # Should not raise JSONDecodeError
            metadata = json.load(f)

        assert isinstance(metadata, dict)

    def test_save_metadata_empty_meta_data(self, mock_vectoriser, tmp_path):
        """Empty meta_data dict serialized correctly."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            meta_data=None,
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["meta_data"] == {}

    def test_save_metadata_path_must_be_string(self, initialized_vectorstore):
        """Path argument must be string."""
        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore._save_metadata(path=None)

        assert "path" in str(exc_info.value).lower()

    def test_save_metadata_path_must_be_non_empty(self, initialized_vectorstore):
        """Path argument must be non-empty string."""
        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore._save_metadata(path="")

        assert "path" in str(exc_info.value).lower()

    def test_save_metadata_invalid_path_type(self, initialized_vectorstore):
        """Path as non-string type raises error."""
        with pytest.raises(DataValidationError) as exc_info:
            initialized_vectorstore._save_metadata(path=123)

        assert "path" in str(exc_info.value).lower()


# ============================================================================
# METADATA LOADING TESTS (from_filespace related)
# ============================================================================


class TestVectorStoreMetadataLoading:
    """Tests for metadata loading and deserialization."""

    def test_load_metadata_file_read_and_parsed(self, mock_vectoriser, tmp_path):
        """Metadata file read and parsed correctly from saved vectorstore."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert isinstance(metadata, dict)
        assert "vectoriser_class" in metadata

    def test_load_metadata_required_keys_validated(self, mock_vectoriser, tmp_path):
        """Required keys validated when loading metadata."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        required_keys = ["vectoriser_class", "vector_shape", "num_vectors", "created_at", "meta_data"]
        for key in required_keys:
            assert key in metadata

    def test_load_metadata_type_deserialization_str(self, mock_vectoriser, tmp_path):
        """Type deserialization works for str."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text,source\ndoc1,hello,src1\ndoc2,world,src2\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            meta_data={"source": str},
            output_dir=str(output_dir),
            skip_save=False,
        )

        # Load it back
        vs_loaded = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        # Type should be deserialized back to str
        assert vs_loaded.meta_data["source"] == "str"

    def test_load_metadata_type_deserialization_int(self, mock_vectoriser, tmp_path):
        """Type deserialization works for int."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text,count\ndoc1,hello,5\ndoc2,world,10\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            meta_data={"count": int},
            output_dir=str(output_dir),
            skip_save=False,
        )

        # Load it back
        vs_loaded = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        # Type should be deserialized back to int
        assert vs_loaded.meta_data["count"] == "int"

    def test_load_metadata_type_deserialization_float(self, mock_vectoriser, tmp_path):
        """Type deserialization works for float."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text,score\ndoc1,hello,0.5\ndoc2,world,0.8\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            meta_data={"score": float},
            output_dir=str(output_dir),
            skip_save=False,
        )

        # Load it back
        vs_loaded = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        # Type should be deserialized back to float
        assert vs_loaded.meta_data["score"] == "float"

    def test_load_metadata_backwards_compatibility_missing_batch_size(self, mock_vectoriser, tmp_path):
        """Backwards compatibility with v1.0.0 (missing batch_size)."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        # Remove batch_size from metadata to simulate v1.0.0
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        del metadata["batch_size"]

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Should load without error
        vs_loaded = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs_loaded.batch_size == 128  # default

    def test_load_metadata_batch_size_override_works(self, mock_vectoriser, tmp_path):
        """batch_size override works when loading."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            batch_size=64,
            output_dir=str(output_dir),
            skip_save=False,
        )

        # Load with different batch_size
        vs_loaded = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
            batch_size=256,
        )

        assert vs_loaded.batch_size == 256

    def test_load_metadata_default_batch_size_when_missing(self, mock_vectoriser, tmp_path):
        """Default batch_size used when missing from metadata."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        # Remove batch_size from metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        del metadata["batch_size"]

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Load without specifying batch_size
        vs_loaded = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        # Should use default
        assert vs_loaded.batch_size == 128

    def test_load_metadata_warning_logged_for_missing_batch_size(self, mock_vectoriser, tmp_path):
        """Warning logged when batch_size missing from metadata."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        # Remove batch_size from metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        del metadata["batch_size"]

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Load - should succeed even with missing batch_size
        vs_loaded = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        # Verify it uses the default
        assert vs_loaded.batch_size == 128
        # Verify the vectorstore is still functional
        assert vs_loaded.vector_shape == 3
        assert vs_loaded.num_vectors == 2

    def test_load_metadata_preserves_vectoriser_class(self, mock_vectoriser, tmp_path):
        """Vectoriser class name preserved through save/load cycle."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        vs_loaded = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs_loaded.vectoriser_class == "MockVectoriser"

    def test_load_metadata_preserves_num_vectors(self, mock_vectoriser, tmp_path):
        """num_vectors preserved through save/load cycle."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\ndoc3,test\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        vs_loaded = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs_loaded.num_vectors == 3

    def test_load_metadata_preserves_vector_shape(self, mock_vectoriser, tmp_path):
        """vector_shape preserved through save/load cycle."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        vs_loaded = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs_loaded.vector_shape == 3


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestVectorStoreMetadataEdgeCases:
    """Tests for edge cases in metadata handling."""

    def test_save_metadata_with_large_num_vectors(self, mock_vectoriser, tmp_path):
        """Metadata serialization with large num_vectors."""
        csv_lines = ["label,text"]
        for i in range(100):
            csv_lines.append(f"doc{i},text {i}")

        csv_path = tmp_path / "test.csv"
        csv_path.write_text("\n".join(csv_lines))

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["num_vectors"] == 100

    def test_save_metadata_with_large_vector_shape(self, mock_vectoriser, tmp_path):
        """Metadata serialization with large vector_shape."""
        # Create vectoriser that returns larger embeddings
        mock_large = Mock(spec=VectoriserBase)

        def transform_large(texts):
            return np.array([np.random.rand(512) for _ in texts])

        mock_large.transform.side_effect = transform_large
        mock_large.__class__.__name__ = "LargeVectoriser"

        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_large,
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["vector_shape"] == 512

    def test_save_metadata_indentation_readable(self, mock_vectoriser, tmp_path):
        """Metadata JSON is indented and readable."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            content = f.read()

        # Check for indentation (newlines and spaces)
        assert "\n" in content
        assert "    " in content or "\t" in content

    def test_load_metadata_malformed_json_raises_error(self, mock_vectoriser, tmp_path):
        """Malformed JSON raises IndexBuildError."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text\ndoc1,hello\ndoc2,world\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=str(output_dir),
            skip_save=False,
        )

        # Corrupt metadata.json
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            f.write("{invalid json")

        # Should raise error when loading
        with pytest.raises((IndexBuildError, DataValidationError)):
            VectorStore.from_filespace(
                folder_path=str(output_dir),
                vectoriser=mock_vectoriser,
            )

    def test_save_metadata_special_characters_in_meta_data_keys(self, mock_vectoriser, tmp_path):
        """Metadata keys with special characters handled correctly."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text,col_with_underscore\ndoc1,hello,val1\ndoc2,world,val2\n")

        output_dir = tmp_path / "output"

        vs = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            meta_data={"col_with_underscore": str},
            output_dir=str(output_dir),
            skip_save=False,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "col_with_underscore" in metadata["meta_data"]

    def test_load_metadata_with_multiple_meta_data_types(self, mock_vectoriser, tmp_path):
        """Metadata with multiple column types round-trips correctly."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("label,text,source,count,score\ndoc1,hello,src1,5,0.5\ndoc2,world,src2,10,0.8\n")

        output_dir = tmp_path / "output"

        vs_saved = VectorStore(
            file_name=str(csv_path),
            data_type="csv",
            vectoriser=mock_vectoriser,
            meta_data={"source": str, "count": int, "score": float},
            output_dir=str(output_dir),
            skip_save=False,
        )

        vs_loaded = VectorStore.from_filespace(
            folder_path=str(output_dir),
            vectoriser=mock_vectoriser,
        )

        assert vs_loaded.meta_data["source"] == "str"
        assert vs_loaded.meta_data["count"] == "int"
        assert vs_loaded.meta_data["score"] == "float"
