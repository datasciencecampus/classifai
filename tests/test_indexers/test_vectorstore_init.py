"""Unit tests for VectorStore initialization."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import polars as pl
import pytest

from classifai._optional import OptionalDependencyError
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

    # Make transform() return embeddings matching input size
    def transform_side_effect(texts):
        """Return one embedding per text, all with shape (3,)."""
        num_texts = len(texts)
        return np.array([np.linspace(0.1, 0.3, 3) + (i * 0.3) for i in range(num_texts)])

    mock.transform.side_effect = transform_side_effect
    mock.__class__.__name__ = "MockVectoriser"
    return mock


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a real temp CSV with id, label, text columns."""
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("label,text\ndoc1,hello world\ndoc2,goodbye world\n")
    return str(csv_path)


@pytest.fixture
def temp_csv_with_metadata(tmp_path):
    """Create a real temp CSV with additional metadata columns."""
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("label,text,source\ndoc1,hello,web\ndoc2,goodbye,file\n")
    return str(csv_path)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    return str(tmp_path / "output")


# ============================================================================
# INPUT VALIDATION TESTS (DataValidationError)
# ============================================================================


class TestVectorStoreInitValidation:
    """Tests for input parameter validation in VectorStore.__init__."""

    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_file_name_empty_string_raises_error(self, mock_url_to_fs, mock_vectoriser):
        """file_name must be a non-empty string."""
        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name="",
                data_type="csv",
                vectoriser=mock_vectoriser,
            )
        assert "file_name must be a non-empty string" in str(exc_info.value)

    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_file_name_not_string_raises_error(self, mock_url_to_fs, mock_vectoriser):
        """file_name must be a string, not other types."""
        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name=123,
                data_type="csv",
                vectoriser=mock_vectoriser,
            )
        assert "file_name must be a non-empty string" in str(exc_info.value)

    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_data_type_unsupported_raises_error(self, mock_url_to_fs, mock_vectoriser):
        """data_type must be 'csv', others raise DataValidationError."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/file.parquet")

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name="file.parquet",
                data_type="parquet",
                vectoriser=mock_vectoriser,
            )
        assert "Unsupported data_type" in str(exc_info.value)

    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_vectoriser_not_base_instance_raises_error(self, mock_url_to_fs):
        """Vectoriser must be VectoriserBase instance."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/file.csv")

        with pytest.raises(ConfigurationError) as exc_info:
            VectorStore(
                file_name="file.csv",
                data_type="csv",
                vectoriser="not_a_vectoriser",  # type: ignore
            )
        assert "Vectoriser must be an instance" in str(exc_info.value)

    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_batch_size_negative_raises_error(self, mock_url_to_fs, mock_vectoriser):
        """batch_size must be >= 1."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/file.csv")

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name="file.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                batch_size=-1,
            )
        assert "batch_size must be an integer >= 1" in str(exc_info.value)

    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_batch_size_zero_raises_error(self, mock_url_to_fs, mock_vectoriser):
        """batch_size must be >= 1."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/file.csv")

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name="file.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                batch_size=0,
            )
        assert "batch_size must be an integer >= 1" in str(exc_info.value)

    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_batch_size_not_int_raises_error(self, mock_url_to_fs, mock_vectoriser):
        """batch_size must be an integer."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/file.csv")

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name="file.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                batch_size="32",  # type: ignore
            )
        assert "batch_size must be an integer >= 1" in str(exc_info.value)

    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_meta_data_not_dict_raises_error(self, mock_url_to_fs, mock_vectoriser):
        """meta_data must be dict or None."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/file.csv")

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name="file.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                meta_data="not_a_dict",  # type: ignore
            )
        assert "meta_data must be a dict or None" in str(exc_info.value)

    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_hooks_not_dict_raises_error(self, mock_url_to_fs, mock_vectoriser):
        """Hooks must be dict or None."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/file.csv")

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name="file.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                hooks="not_a_dict",  # type: ignore
            )
        assert "hooks must be a dict or None" in str(exc_info.value)

    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_output_dir_not_string_raises_error(self, mock_url_to_fs, mock_vectoriser):
        """output_dir must be string or None."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/file.csv")

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name="file.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                output_dir=123,  # type: ignore
                skip_save=True,
            )
        assert "output_dir must be a string or None" in str(exc_info.value)


# # ============================================================================
# # FILE SYSTEM HANDLING TESTS (ConfigurationError / OptionalDependencyError)
# # ============================================================================


class TestVectorStoreInitFileSystem:
    """Tests for file system handling in VectorStore.__init__."""

    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_input_file_not_exist_raises_error(self, mock_url_to_fs, mock_vectoriser):
        """Input file must exist."""
        mock_fs = Mock()
        mock_fs.exists.return_value = False
        mock_url_to_fs.return_value = (mock_fs, "/path/to/missing.csv")

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name="/path/to/missing.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
            )
        assert "Input file does not exist" in str(exc_info.value)

    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_input_fsspec_error_raises_configuration_error(self, mock_url_to_fs, mock_vectoriser):
        """Fsspec resolution failure → ConfigurationError."""
        mock_url_to_fs.side_effect = Exception("Invalid fsspec path")

        with pytest.raises(ConfigurationError) as exc_info:
            VectorStore(
                file_name="invalid://path/file.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
            )
        assert "Failed to read input directory with file loader" in str(exc_info.value)

    @patch("classifai.indexers.main.check_deps")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_gs_path_without_gcsfs_raises_helpful_error(self, mock_url_to_fs, mock_check_deps, mock_vectoriser):
        """gs:// path without gcsfs → OptionalDependencyError with helpful message."""
        mock_url_to_fs.side_effect = ImportError("gcsfs not installed")
        mock_check_deps.side_effect = OptionalDependencyError("gcsfs required")

        with pytest.raises(OptionalDependencyError) as exc_info:
            VectorStore(
                file_name="gs://bucket/file.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
            )
        assert "gcsfs" in str(exc_info.value).lower()
        assert "pip install" in str(exc_info.value).lower()


# # ============================================================================
# # OUTPUT DIRECTORY HANDLING TESTS (skip_save=False path)
# # ============================================================================


class TestVectorStoreInitOutputDirectory:
    """Tests for output directory handling when skip_save=False."""

    @patch("classifai.indexers.main.pl.read_csv")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_output_dir_derived_from_file_name(self, mock_url_to_fs, mock_read_csv, mock_vectoriser):
        """When output_dir=None, derive from file_name."""
        # Mock input file
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_fs.makedirs = Mock()

        # Mock file object that supports context manager protocol
        mock_file = MagicMock()
        mock_fs.open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_fs.open.return_value.__exit__ = Mock(return_value=None)

        mock_url_to_fs.side_effect = [
            (mock_fs, "/path/to/test.csv"),  # input
            (mock_fs, "test"),  # output (derived)
            (mock_fs, "test/metadata.json"),  # metadata save
        ]

        # Mock CSV read
        mock_read_csv.return_value = pl.DataFrame(
            {
                "label": ["doc1", "doc2"],
                "text": ["hello", "world"],
            }
        )

        # Mock the parquet write to prevent actual file I/O
        with patch("classifai.indexers.main.pl.DataFrame.write_parquet"):
            vs = VectorStore(
                file_name="/path/to/test.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                skip_save=False,
                overwrite=True,
            )

        assert vs.output_dir == "test"
        mock_fs.makedirs.assert_called()

    @patch("classifai.indexers.main.pl.read_csv")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_output_dir_exists_without_overwrite_raises_error(
        self, mock_url_to_fs, mock_read_csv, mock_vectoriser
    ):
        """Existing output_dir without overwrite=True → ConfigurationError."""
        mock_fs = Mock()
        mock_fs.exists.side_effect = [True, True]  # input exists, output exists
        mock_url_to_fs.side_effect = [
            (mock_fs, "/path/to/test.csv"),
            (mock_fs, "/path/to/output"),
        ]

        with pytest.raises(ConfigurationError) as exc_info:
            VectorStore(
                file_name="/path/to/test.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                output_dir="/path/to/output",
                overwrite=False,
                skip_save=False,
            )
        assert "already exists" in str(exc_info.value).lower()

    @patch("classifai.indexers.main.pl.read_csv")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_output_dir_exists_with_overwrite_removes_directory(
        self, mock_url_to_fs, mock_read_csv, mock_vectoriser
    ):
        """overwrite=True removes and recreates directory."""
        mock_fs = Mock()
        mock_fs.exists.side_effect = [True, True]  # input, output
        mock_fs.rm = Mock()
        mock_fs.makedirs = Mock()

        # Mock file object that supports context manager protocol
        mock_file = MagicMock()
        mock_fs.open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_fs.open.return_value.__exit__ = Mock(return_value=None)

        mock_url_to_fs.side_effect = [
            (mock_fs, "/path/to/test.csv"),
            (mock_fs, "/path/to/output"),
            (mock_fs, "/path/to/output/metadata.json"),  # ← Add 3rd call for metadata
        ]

        mock_read_csv.return_value = pl.DataFrame(
            {
                "label": ["doc1"],
                "text": ["hello"],
            }
        )

        # Mock the parquet write to prevent actual file I/O
        with patch("classifai.indexers.main.pl.DataFrame.write_parquet"):
            VectorStore(
                file_name="/path/to/test.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                output_dir="/path/to/output",
                overwrite=True,
                skip_save=False,
            )

        mock_fs.rm.assert_called_once()
        mock_fs.makedirs.assert_called()

    @patch("classifai.indexers.main.check_deps")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_output_dir_gs_path_without_gcsfs_raises_helpful_error(
        self, mock_url_to_fs, mock_check_deps, mock_vectoriser
    ):
        """gs:// output_dir without gcsfs → OptionalDependencyError."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.side_effect = [
            (mock_fs, "/path/to/test.csv"),  # input OK
            ImportError("gcsfs not installed"),  # output fails
        ]
        mock_check_deps.side_effect = OptionalDependencyError("gcsfs required")

        with pytest.raises(OptionalDependencyError) as exc_info:
            VectorStore(
                file_name="/path/to/test.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                output_dir="gs://bucket/output",
                skip_save=False,
            )
        assert "gcsfs" in str(exc_info.value).lower()

    # # ============================================================================
    # # INDEX BUILDING TESTS (IndexBuildError, _create_vector_store_index)
    # # ============================================================================

    # class TestVectorStoreInitIndexBuilding:
    #     """Tests for index building during initialization."""

    @patch("classifai.indexers.main.pl.read_csv")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_csv_reads_correctly(self, mock_url_to_fs, mock_read_csv, mock_vectoriser):
        """CSV reads, UUIDs assigned, vectoriser called."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/test.csv")

        # Mock CSV with 2 rows
        mock_read_csv.return_value = pl.DataFrame(
            {
                "label": ["doc1", "doc2"],
                "text": ["hello world", "goodbye world"],
            }
        )

        vs = VectorStore(
            file_name="/path/to/test.csv",
            data_type="csv",
            vectoriser=mock_vectoriser,
            skip_save=True,
        )

        # Verify vectoriser was called
        mock_vectoriser.transform.assert_called()
        # Verify UUIDs were assigned
        assert "uuid" in vs.vectors.columns
        assert len(vs.vectors) == 2

    @patch("classifai.indexers.main.pl.read_csv")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_csv_reads_with_metadata(self, mock_url_to_fs, mock_read_csv, mock_vectoriser):
        """CSV reads with metadata columns included."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/test.csv")

        mock_read_csv.return_value = pl.DataFrame(
            {
                "label": ["doc1", "doc2"],
                "text": ["hello", "world"],
                "source": ["web", "file"],
            }
        )

        vs = VectorStore(
            file_name="/path/to/test.csv",
            data_type="csv",
            vectoriser=mock_vectoriser,
            meta_data={"source": str},
            skip_save=True,
        )

        assert "source" in vs.vectors.columns
        assert list(vs.vectors["source"]) == ["web", "file"]

    @patch("classifai.indexers.main.pl.read_csv")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_vectoriser_failure_wrapped_appropriately(self, mock_url_to_fs, mock_read_csv, mock_vectoriser):
        """vectoriser.transform() exception → IndexBuildError with context."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/test.csv")

        mock_read_csv.return_value = pl.DataFrame(
            {
                "label": ["doc1"],
                "text": ["hello"],
            }
        )

        mock_vectoriser.transform.side_effect = RuntimeError("Vectoriser crashed")

        with pytest.raises(IndexBuildError) as exc_info:
            VectorStore(
                file_name="/path/to/test.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                skip_save=True,
            )
        assert "Vectoriser.transform failed" in str(exc_info.value)

    @patch("classifai.indexers.main.pl.read_csv")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_embeddings_count_mismatch_raises_error(self, mock_url_to_fs, mock_read_csv, mock_vectoriser):
        """Vectoriser returns wrong # embeddings → IndexBuildError."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/test.csv")

        mock_read_csv.return_value = pl.DataFrame(
            {
                "label": ["doc1", "doc2"],
                "text": ["hello", "world"],
            }
        )

        # Clear side_effect and set return_value
        mock_vectoriser.transform.side_effect = None
        mock_vectoriser.transform.return_value = np.array([[0.1, 0.2, 0.3]])

        with pytest.raises(IndexBuildError) as exc_info:
            VectorStore(
                file_name="/path/to/test.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                skip_save=True,
            )
        assert "wrong number of embeddings" in str(exc_info.value).lower()

    @patch("classifai.indexers.main.pl.read_csv")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_empty_csv_raises_error(self, mock_url_to_fs, mock_read_csv, mock_vectoriser):
        """Empty CSV (no documents) raises DataValidationError."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/test.csv")

        mock_read_csv.return_value = pl.DataFrame(
            {
                "label": [],
                "text": [],
            }
        )

        with pytest.raises(DataValidationError) as exc_info:
            VectorStore(
                file_name="/path/to/test.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                skip_save=True,
            )
        assert "no documents" in str(exc_info.value).lower()

    @patch("classifai.indexers.main.pl.read_csv")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_batch_processing_works(self, mock_url_to_fs, mock_read_csv, mock_vectoriser):
        """Batch processing embeds texts in chunks."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/test.csv")

        # 5 documents with batch_size=2 → 3 batches
        mock_read_csv.return_value = pl.DataFrame(
            {
                "label": ["doc1", "doc2", "doc3", "doc4", "doc5"],
                "text": ["a", "b", "c", "d", "e"],
            }
        )

        # Mock vectoriser to return correct number of embeddings per call
        mock_vectoriser.transform.side_effect = [
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),  # batch 1: 2 embeddings
            np.array([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]),  # batch 2: 2 embeddings
            np.array([[1.3, 1.4, 1.5]]),  # batch 3: 1 embedding
        ]

        vs = VectorStore(
            file_name="/path/to/test.csv",
            data_type="csv",
            vectoriser=mock_vectoriser,
            batch_size=2,
            skip_save=True,
        )

        # Verify vectoriser was called 3 times
        assert mock_vectoriser.transform.call_count == 3
        # Verify all embeddings were collected
        assert len(vs.vectors) == 5


# # ============================================================================
# # SAVE/METADATA TESTS (skip_save flag)
# # ============================================================================


class TestVectorStoreInitSaveHandling:
    """Tests for save/metadata handling and skip_save flag."""

    @patch("classifai.indexers.main.pl.read_csv")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_skip_save_true_no_files_written(self, mock_url_to_fs, mock_read_csv, mock_vectoriser):
        """skip_save=True → no parquet/JSON written."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/test.csv")

        mock_read_csv.return_value = pl.DataFrame(
            {
                "label": ["doc1"],
                "text": ["hello"],
            }
        )

        vs = VectorStore(  # noqa: F841
            file_name="/path/to/test.csv",
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir="/path/to/output",
            skip_save=True,
        )

        # Verify no write operations on the filesystem
        assert not mock_fs.open.called

    @patch("classifai.indexers.main.pl.read_csv")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_skip_save_false_files_written(self, mock_url_to_fs, mock_read_csv, mock_vectoriser):
        """skip_save=False → parquet + metadata.json created."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_fs.makedirs = Mock()
        mock_fs.open = MagicMock()

        # Add 3 calls: input file, output dir, metadata JSON write
        mock_url_to_fs.side_effect = [
            (mock_fs, "/path/to/test.csv"),  # input
            (mock_fs, "/path/to/output"),  # output
            (mock_fs, "/path/to/output/metadata.json"),  # metadata save
        ]

        mock_read_csv.return_value = pl.DataFrame(
            {
                "label": ["doc1"],
                "text": ["hello"],
            }
        )

        with patch("classifai.indexers.main.pl.DataFrame.write_parquet") as mock_write_parquet:
            vs = VectorStore(  # noqa: F841
                file_name="/path/to/test.csv",
                data_type="csv",
                vectoriser=mock_vectoriser,
                output_dir="/path/to/output",
                overwrite=True,
                skip_save=False,
            )

            # Verify parquet write was called
            mock_write_parquet.assert_called_once()
            # Verify JSON metadata write was attempted
            mock_fs.open.assert_called()

    # @patch("classifai.indexers.main.pl.read_csv")
    # @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    # def test_init_metadata_json_contains_required_fields(self, mock_url_to_fs, mock_read_csv, mock_vectoriser):
    #     """Saved metadata.json contains all required fields."""
    #     mock_fs = Mock()
    #     mock_fs.exists.return_value = True
    #     mock_fs.makedirs = Mock()

    #     # Capture written data
    #     written_data = {}

    #     def mock_open_func(path, mode="w", encoding=None):
    #         """Mock file that captures write calls."""
    #         class MockFile:
    #             def __enter__(self):
    #                 return self
    #             def __exit__(self, *args):
    #                 pass
    #             def write(self, data):
    #                 nonlocal written_data
    #                 written_data = data
    #         return MockFile()

    #     mock_fs.open = mock_open_func

    #     # Add 4 calls: input file, output dir, metadata JSON write (path), metadata JSON write (actual)
    #     mock_url_to_fs.side_effect = [
    #         (mock_fs, "/path/to/test.csv"),                    # input
    #         (mock_fs, "/path/to/output"),                      # output
    #         (mock_fs, "/path/to/output/metadata.json"),        # metadata save path resolve
    #         (mock_fs, "/path/to/output/metadata.json"),        # metadata save actual write
    #     ]

    #     mock_read_csv.return_value = pl.DataFrame({
    #         "label": ["doc1"],
    #         "text": ["hello"],
    #     })

    #     with patch("classifai.indexers.main.pl.DataFrame.write_parquet"):
    #         vs = VectorStore(
    #             file_name="/path/to/test.csv",
    #             data_type="csv",
    #             vectoriser=mock_vectoriser,
    #             output_dir="/path/to/output",
    #             overwrite=True,
    #             skip_save=False,
    #         )

    #         # Verify metadata was written
    #         assert written_data, "No data was written to metadata file"
    #         metadata = json.loads(written_data)
    #         assert "vectoriser_class" in metadata
    #         assert "vector_shape" in metadata
    #         assert "num_vectors" in metadata
    #         assert "batch_size" in metadata
    #         assert "created_at" in metadata
    #         assert "meta_data" in metadata
    #         assert metadata["vectoriser_class"] == "MockVectoriser"
    #         assert metadata["vector_shape"] == 3
    #         assert metadata["num_vectors"] == 1


# # ============================================================================
# # QUIET MODE TESTS
# # ============================================================================


class TestVectorStoreInitQuietMode:
    """Tests for quiet mode behavior."""

    @patch("classifai.indexers.main.pl.read_csv")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_quiet_mode_true_suppresses_progress(self, mock_url_to_fs, mock_read_csv, mock_vectoriser):
        """quiet_mode=True → progress bars suppressed."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/test.csv")

        mock_read_csv.return_value = pl.DataFrame(
            {
                "label": ["doc1", "doc2", "doc3"],
                "text": ["a", "b", "c"],
            }
        )

        vs = VectorStore(
            file_name="/path/to/test.csv",
            data_type="csv",
            vectoriser=mock_vectoriser,
            quiet_mode=True,
            skip_save=True,
        )

        # When quiet_mode=True, classifai_tqdm should be identity function (no wrapping)
        assert vs.classifai_tqdm([1, 2, 3]) == [1, 2, 3]

    @patch("classifai.indexers.main.pl.read_csv")
    @patch("classifai.indexers.main.fsspec.core.url_to_fs")
    def test_init_quiet_mode_false_shows_progress(self, mock_url_to_fs, mock_read_csv, mock_vectoriser):
        """quiet_mode=False → progress bars shown (tqdm enabled)."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "/path/to/test.csv")

        mock_read_csv.return_value = pl.DataFrame(
            {
                "label": ["doc1"],
                "text": ["hello"],
            }
        )

        vs = VectorStore(
            file_name="/path/to/test.csv",
            data_type="csv",
            vectoriser=mock_vectoriser,
            quiet_mode=False,
            skip_save=True,
        )

        # When quiet_mode=False, classifai_tqdm should be tqdm
        from tqdm.autonotebook import tqdm

        assert vs.classifai_tqdm == tqdm


# ============================================================================
# INTEGRATION TESTS (Real temp files)
# ============================================================================


class TestVectorStoreInitIntegration:
    """Integration tests using real temporary files."""

    def test_init_with_real_csv_file(self, temp_csv_file, mock_vectoriser, temp_output_dir):
        """Full initialization with real CSV file and temp output directory."""
        vs = VectorStore(
            file_name=temp_csv_file,
            data_type="csv",
            vectoriser=mock_vectoriser,
            output_dir=temp_output_dir,
            skip_save=False,
        )

        # Verify VectorStore was created successfully
        assert vs.file_name == temp_csv_file
        assert vs.vectors is not None
        assert len(vs.vectors) == 2
        assert vs.vector_shape == 3  # embeddings have 3 dimensions
        assert vs.num_vectors == 2
        assert vs.vectoriser_class == "MockVectoriser"

        # Verify files were saved
        assert os.path.exists(os.path.join(temp_output_dir, "vectors.parquet"))
        assert os.path.exists(os.path.join(temp_output_dir, "metadata.json"))

        # Verify metadata.json content
        with open(os.path.join(temp_output_dir, "metadata.json")) as f:
            metadata = json.load(f)
            assert metadata["vectoriser_class"] == "MockVectoriser"
            assert metadata["vector_shape"] == 3
            assert metadata["num_vectors"] == 2

    def test_init_with_metadata_columns(self, temp_csv_with_metadata, mock_vectoriser, temp_output_dir):
        """Initialization with metadata columns."""
        vs = VectorStore(
            file_name=temp_csv_with_metadata,
            data_type="csv",
            vectoriser=mock_vectoriser,
            meta_data={"source": str},
            output_dir=temp_output_dir,
            skip_save=False,
        )

        assert "source" in vs.vectors.columns
        assert vs.meta_data == {"source": str}

        # Verify metadata is saved correctly
        with open(os.path.join(temp_output_dir, "metadata.json")) as f:
            metadata = json.load(f)
            assert "source" in metadata["meta_data"]
