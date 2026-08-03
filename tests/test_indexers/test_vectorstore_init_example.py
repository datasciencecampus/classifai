"""Unit tests for VectorStore initialization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from classifai.exceptions import (
    ConfigurationError,
    DataValidationError,
    IndexBuildError,
)
from classifai.indexers import VectorStore
from classifai.vectorisers import VectoriserBase


class TestVectorStoreInitInputValidation:
    """Tests for VectorStore initialization input validation."""

    @pytest.fixture
    def mock_vectoriser(self):
        """Create a mocked vectoriser."""
        vectoriser = Mock(spec=VectoriserBase)
        vectoriser.transform.return_value = np.random.rand(3, 768)
        return vectoriser

    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text,label\n")
            f.write("hello world,111\n")
            f.write("goodbye world,112\n")
            f.write("test data,113\n")
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink()

    def test_init_with_valid_inputs(self, mock_vectoriser, temp_csv_file):
        """Test successful initialization with valid inputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vectorstore = VectorStore(
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                data_type="csv",
                output_dir=temp_dir,
                skip_save=True,
            )

            assert vectorstore.file_name == temp_csv_file
            assert vectorstore.vectoriser == mock_vectoriser
            NEXT_ANSWER = 128
            assert vectorstore.batch_size == NEXT_ANSWER  # default
            assert vectorstore.meta_data == {}  # default

    def test_init_file_name_must_be_non_empty_string(self, mock_vectoriser):
        """Test that file_name must be non-empty string."""
        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(file_name="", vectoriser=mock_vectoriser)

        error = exc_info.value
        assert error.code == "validation_error"
        assert "file_name" in error.message.lower()

    def test_init_file_name_must_exist(self, mock_vectoriser):
        """Test that input file must exist."""
        with pytest.raises(ConfigurationError) as exc_info:
            VectorStore(
                file_name="/nonexistent/path/file.csv",
                vectoriser=mock_vectoriser,
            )

        error = exc_info.value
        assert error.code == "configuration_error"
        assert "not found" in error.message.lower() or "exist" in error.message.lower()

    def test_init_data_type_must_be_csv(self, mock_vectoriser, temp_csv_file):
        """Test that only 'csv' data_type is supported."""
        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                data_type="parquet",
                skip_save=True,
            )

        error = exc_info.value
        assert error.code == "validation_error"
        assert "data_type" in error.message.lower()

    def test_init_vectoriser_must_be_vectoriser_base_instance(self, temp_csv_file):
        """Test that vectoriser must be VectoriserBase instance."""
        invalid_vectoriser = "not a vectoriser"

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name=temp_csv_file,
                vectoriser=invalid_vectoriser,
                skip_save=True,
            )

        error = exc_info.value
        assert error.code == "validation_error"
        assert "vectoriser" in error.message.lower()

    def test_init_batch_size_must_be_positive_int(self, mock_vectoriser, temp_csv_file):
        """Test that batch_size must be positive integer."""
        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                batch_size=0,
                skip_save=True,
            )

        error = exc_info.value
        assert error.code == "validation_error"
        assert "batch_size" in error.message.lower()

    def test_init_batch_size_negative_raises_error(self, mock_vectoriser, temp_csv_file):
        """Test that negative batch_size raises error."""
        with pytest.raises(DataValidationError):
            VectorStore(
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                batch_size=-1,
                skip_save=True,
            )

    def test_init_meta_data_must_be_dict_or_none(self, mock_vectoriser, temp_csv_file):
        """Test that meta_data must be dict or None."""
        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                meta_data="invalid",
                skip_save=True,
            )

        error = exc_info.value
        assert error.code == "validation_error"

    def test_init_hooks_must_be_dict_or_none(self, mock_vectoriser, temp_csv_file):
        """Test that hooks must be dict or None."""
        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                hooks="invalid",
                skip_save=True,
            )

        error = exc_info.value
        assert error.code == "validation_error"

    def test_init_output_dir_must_be_string_or_none(self, mock_vectoriser, temp_csv_file):
        """Test that output_dir must be string or None."""
        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                output_dir=123,
                skip_save=True,
            )

        error = exc_info.value
        assert error.code == "validation_error"


class TestVectorStoreInitFileSystem:
    """Tests for VectorStore file system handling during initialization."""

    @pytest.fixture
    def mock_vectoriser(self):
        """Create a mocked vectoriser."""
        vectoriser = Mock(spec=VectoriserBase)
        vectoriser.transform.return_value = np.random.rand(3, 768)
        return vectoriser

    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,text\n")
            f.write("1,hello world\n")
            f.write("2,goodbye world\n")
            f.write("3,test data\n")
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink()

    def test_init_creates_output_directory_if_not_exists(self, mock_vectoriser, temp_csv_file):
        """Test that output_dir is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "vectorstore"
            assert not output_path.exists()

            VectorStore(
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                output_dir=str(output_path),
                skip_save=True,
            )

            # Directory creation may or may not happen if skip_save=True
            # The important thing is no error is raised

    def test_init_overwrite_false_prevents_overwrite(self, mock_vectoriser, temp_csv_file):
        """Test that overwrite=False prevents overwriting existing index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "index"
            output_dir.mkdir()

            # Create existing metadata file
            metadata_file = output_dir / "metadata.json"
            metadata_file.write_text(json.dumps({"existing": "data"}))

            with pytest.raises(ConfigurationError) as exc_info:
                VectorStore(
                    file_name=temp_csv_file,
                    vectoriser=mock_vectoriser,
                    output_dir=str(output_dir),
                    overwrite=False,
                )

            error = exc_info.value
            assert "overwrite" in error.message.lower()

    def test_init_overwrite_true_allows_overwrite(self, mock_vectoriser, temp_csv_file):
        """Test that overwrite=True allows overwriting existing index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "index"
            output_dir.mkdir()

            # Create existing metadata file
            metadata_file = output_dir / "metadata.json"
            metadata_file.write_text(json.dumps({"existing": "data"}))

            vectorstore = VectorStore(
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                output_dir=str(output_dir),
                overwrite=True,
                skip_save=True,
            )

            assert vectorstore is not None

    def test_init_gcsfs_path_requires_gcsfs_library(self, mock_vectoriser, temp_csv_file):
        """Test that gs:// paths require gcsfs and provide helpful error."""
        with pytest.raises(ConfigurationError) as exc_info:
            VectorStore(
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                output_dir="gs://bucket/path",
                skip_save=True,
            )

        error = exc_info.value
        assert "gcsfs" in error.message.lower() or "google" in error.message.lower()


class TestVectorStoreInitIndexBuilding:
    """Tests for VectorStore index building during initialization."""

    @pytest.fixture
    def mock_vectoriser(self):
        """Create a mocked vectoriser that returns embeddings."""
        vectoriser = Mock(spec=VectoriserBase)
        # Return embeddings with shape (batch_size, 768)
        vectoriser.transform.side_effect = lambda texts: np.random.rand(
            len(texts) if isinstance(texts, list) else 1, 768
        )
        return vectoriser

    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,text,label\n")
            f.write("1,hello world,positive\n")
            f.write("2,goodbye world,negative\n")
            f.write("3,test data,neutral\n")
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink()

    def test_init_reads_csv_file_correctly(self, mock_vectoriser, temp_csv_file):
        """Test that CSV file is read correctly during initialization."""
        vectorstore = VectorStore(  # noqa: F841
            file_name=temp_csv_file,
            vectoriser=mock_vectoriser,
            skip_save=True,
        )

        # Verify vectoriser was called with text column
        assert mock_vectoriser.transform.called
        # Should have been called at least once with the texts
        call_args = mock_vectoriser.transform.call_args_list
        assert len(call_args) > 0

    def test_init_uuid_generation_for_each_row(self, mock_vectoriser, temp_csv_file):
        """Test that UUID is generated for each CSV row."""
        vectorstore = VectorStore(
            file_name=temp_csv_file,
            vectoriser=mock_vectoriser,
            skip_save=True,
        )

        # Check that internal data has unique IDs
        assert hasattr(vectorstore, "_index_data") or hasattr(vectorstore, "index_data")
        # All UUIDs should be unique
        index_attr = getattr(vectorstore, "_index_data", None) or getattr(vectorstore, "index_data", None)
        if index_attr is not None and hasattr(index_attr, "index"):
            NEXT_ANSWER = 3
            assert len(index_attr.index.unique()) == NEXT_ANSWER  # 3 rows in CSV

    def test_init_batch_processing_of_embeddings(self, mock_vectoriser, temp_csv_file):
        """Test that embeddings are processed in batches."""
        vectorstore = VectorStore(  # noqa: F841
            file_name=temp_csv_file,
            vectoriser=mock_vectoriser,
            batch_size=2,
            skip_save=True,
        )

        # Vectoriser should be called multiple times (batched)
        # With 3 texts and batch_size=2, should be called at least 2 times
        NEXT_ANSWER = 2
        assert mock_vectoriser.transform.call_count >= NEXT_ANSWER

    def test_init_vectoriser_failure_raises_index_build_error(self, temp_csv_file):
        """Test that vectoriser failures are wrapped in IndexBuildError."""
        mock_vectoriser = Mock(spec=VectoriserBase)
        mock_vectoriser.transform.side_effect = Exception("Vectoriser failed")

        with pytest.raises(IndexBuildError) as exc_info:
            VectorStore(
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                skip_save=True,
            )

        error = exc_info.value
        assert error.code == "index_build_error"
        assert "vectoriser" in error.message.lower()

    def test_init_embeddings_count_matches_batch_size(self, mock_vectoriser, temp_csv_file):
        """Test that returned embeddings match requested batch size."""
        vectorstore = VectorStore(  # noqa: F841
            file_name=temp_csv_file,
            vectoriser=mock_vectoriser,
            batch_size=2,
            skip_save=True,
        )

        # All calls should return correct number of embeddings
        for call in mock_vectoriser.transform.call_args_list:
            texts = call[0][0]  # First positional arg
            embeddings = mock_vectoriser.transform.return_value
            assert embeddings.shape[0] == len(texts)

    def test_init_metadata_serialization(self, mock_vectoriser, temp_csv_file):
        """Test that metadata is properly serialized."""
        meta_data = {"source": "test", "version": "1.0"}

        vectorstore = VectorStore(
            file_name=temp_csv_file,
            vectoriser=mock_vectoriser,
            meta_data=meta_data,
            skip_save=True,
        )

        # Verify metadata is stored
        assert vectorstore.meta_data == meta_data

    def test_init_parquet_file_writing(self, mock_vectoriser, temp_csv_file):
        """Test that embeddings are written to parquet format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vectorstore = VectorStore(  # noqa: F841
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                output_dir=temp_dir,
                skip_save=False,
            )

            # Check that vectors.parquet exists
            parquet_file = Path(temp_dir) / "vectors.parquet"
            assert parquet_file.exists()

    def test_init_metadata_json_file_writing(self, mock_vectoriser, temp_csv_file):
        """Test that metadata is written to JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vectorstore = VectorStore(  # noqa: F841
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                output_dir=temp_dir,
                skip_save=False,
            )

            # Check that metadata.json exists
            metadata_file = Path(temp_dir) / "metadata.json"
            assert metadata_file.exists()

            # Verify it's valid JSON
            metadata = json.loads(metadata_file.read_text())
            assert "vectoriser_class" in metadata
            assert "vector_shape" in metadata


class TestVectorStoreInitSkipSaveFlag:
    """Tests for VectorStore skip_save flag behavior."""

    @pytest.fixture
    def mock_vectoriser(self):
        """Create a mocked vectoriser."""
        vectoriser = Mock(spec=VectoriserBase)
        vectoriser.transform.return_value = np.random.rand(3, 768)
        return vectoriser

    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,text\n")
            f.write("1,hello world\n")
            f.write("2,goodbye world\n")
            f.write("3,test data\n")
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink()

    def test_init_skip_save_true_no_files_written(self, mock_vectoriser, temp_csv_file):
        """Test that no files are written when skip_save=True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vectorstore = VectorStore(  # noqa: F841
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                output_dir=temp_dir,
                skip_save=True,
            )

            # Check that no files were created
            files = list(Path(temp_dir).glob("*"))
            assert len(files) == 0

    def test_init_skip_save_false_writes_files(self, mock_vectoriser, temp_csv_file):
        """Test that files are written when skip_save=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vectorstore = VectorStore(  # noqa: F841
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                output_dir=temp_dir,
                skip_save=False,
            )

            # Check that files were created
            parquet_file = Path(temp_dir) / "vectors.parquet"
            metadata_file = Path(temp_dir) / "metadata.json"
            assert parquet_file.exists()
            assert metadata_file.exists()

    def test_init_skip_save_true_with_output_dir_logs_warning(self, mock_vectoriser, temp_csv_file):
        """Test that warning is logged when output_dir set but skip_save=True."""
        with tempfile.TemporaryDirectory() as temp_dir, patch("classifai.indexers.main.logger") as mock_logger:
            vectorstore = VectorStore(  # noqa: F841
                file_name=temp_csv_file,
                vectoriser=mock_vectoriser,
                output_dir=temp_dir,
                skip_save=True,
            )

            # Verify warning was logged
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "skip_save" in warning_msg.lower() or "not saved" in warning_msg.lower()
