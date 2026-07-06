# pylint: disable=C0301
"""Provides functionality for creating a `VectorStore` from a CSV file.

Defines the `VectorStore` class, which is used to model and create vector
databases from CSV (text) files using a `Vectoriser` object.

This class requires a `Vectoriser` object from the vectorisers submodule to
convert the CSV's text data into vector embeddings which are then stored in the
VectorStore objects.

Key Features:
- Batch processing of input files to handle large datasets.
- Support for CSV file format
- Integration with a custom embedder for generating vector embeddings.
- Support for user-defined hooks for preprocessing and postprocessing.
- Logging for tracking progress and handling errors during processing.

VectorStore Class:

  - The `VectorStore` class is initialised with a `Vectoriser` object and a CSV
        knowledgebase.
  - Additional columns in the CSV may be specified as metadata to be included
        in the vector database.
  - Upon creation, the `VectorStore` is saved in parquet format for efficient,
        and quick reloading via the `VectorStore`'s `.from_filespace()` method.
  - A new piece of text data (or label) can be queried against the
        `VectorStore` in the following ways:
    - `.search()`: to find the most semantically similar pieces of text in the
        vector database.
    - `.reverse_search()`: to find all examples in the knowledgebase that have
            a given label.
    - `.embed()`: to generate a vector embedding for a given piece of text
        data using the vectoriser.
  - 'Hook' methods may be specified to perform pre-processing on input data
        before embedding, and post-processing on the output of the search
        methods.
"""

import json
import logging
import os
import time
import uuid

import fsspec
import numpy as np
import polars as pl
from tqdm.autonotebook import tqdm

from classifai._optional import OptionalDependencyError, check_deps
from classifai.exceptions import (
    ClassifaiError,
    ConfigurationError,
    DataValidationError,
    HookError,
    IndexBuildError,
    VectorisationError,
)

from ..vectorisers.base import VectoriserBase
from .dataclasses import (
    VectorStoreEmbedInput,
    VectorStoreEmbedOutput,
    VectorStoreReverseSearchInput,
    VectorStoreReverseSearchOutput,
    VectorStoreSearchInput,
    VectorStoreSearchOutput,
)


class VectorStore:
    """Models and creates vector databases from CSV text files.

    Converts a knowledgebase (CSV file) to a DataFrame, embeds the text column
    in batches, storing the resulting vectors for querying. Once built, the
    store supports semantic search via .search(), label-based lookup via
    .reverse_search(), and direct embedding via .embed(). The index can be
    persisted to disk and reloaded later using .from_filespace().

    Attributes:
        file_name (str): Path to the input file used to build the `vectors`
            dataframe.
        data_type (str): Format of the input file. Currently only "csv" is
            supported.
        vectoriser (VectoriserBase): Vectoriser instance used to convert text
            into vector embeddings.
        batch_size (int): The batch size to pass to the vectoriser when
            embedding.
        meta_data (dict | None): Mapping of extra CSV column names to extract
            to their Python types (e.g. {"source": str}). Values are Python
            types.
        output_dir (str | None): Directory where vectors.parquet and
            metadata.json are written. Defaults to the input file stem when
            None is passed. Ignored when skip_save=True.
        skip_save (bool): If False, saves the `VectorStore` to disk after
            creation. If True, keeps it in memory only (for testing or
            ephemeral use cases). Defaults to False.
        vectors (pl.DataFrame | None): Polars DataFrame containing the full
            knowledgebase table with columns: label, text, uuid, embeddings,
            and any columns specified in meta_data. None until the index is
            built.
        vector_shape (int): Number of dimensions in the vector embeddings.
        num_vectors (int): Total number of rows stored in the VectorStore.
        vectoriser_class (str): The type of Vectoriser used to create
            embeddings.
        hooks (dict): A dictionary of user-defined hooks for preprocessing and
            postprocessing.
        quiet_mode (bool): Whether to minimise verbose output, such as progress
            bars.
    """

    def __init__(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        file_name: str,
        data_type: str,
        vectoriser: VectoriserBase,
        batch_size: int = 128,
        meta_data: dict | None = None,
        output_dir: str | None = None,
        overwrite: bool = False,
        skip_save: bool = False,
        hooks: dict | None = None,
        quiet_mode: bool = False,
    ):
        """Generates vector embeddings from the input csv to form a `VectorStore`.

        Args:
            file_name (str): Path to the input file used to build the `vectors`
                dataframe.
            data_type (str): Format of the input file. Currently only "csv" is
                supported.
            vectoriser (VectoriserBase): Vectoriser instance used to convert
                text into vector embeddings.
            batch_size (int): The batch size to pass to the vectoriser when
                embedding. Defaults to 128.
            meta_data (dict | None): Mapping of extra CSV column names to
                extract to their Python types (e.g. {"source": str}). Values
                are Python types.
            output_dir (str | None): Directory where vectors.parquet and
                metadata.json are written. Defaults to the input file stem when
                None is passed. Ignored when skip_save=True.
            overwrite (bool): If True, allows overwriting existing folders with
                the same name. Defaults to False to prevent accidental
                overwrites. Ignored if skip_save=True.
            skip_save (bool): If False, saves the `VectorStore` to disk after
                creation. If True, keeps it in memory only (for testing or
                ephemeral use cases). Defaults to False.
            hooks (dict): A dictionary of user-defined hooks for preprocessing
                and postprocessing.
            quiet_mode (bool): Whether to minimise verbose output, such as
                progress bars.

        Raises:
            ClassifaiError: For any unexpected errors during initialisation,
                with context for debugging.
            DataValidationError: If input arguments are invalid or if there are
                issues with the input file.
            ConfigurationError: If there are configuration issues, such as
                output directory problems.
            IndexBuildError: If there are failures during index building or
                saving outputs.
            OptionalDependencyError: If the user attempts to use a gs:// path
                without having gcsfs installed.
        """
        # ---- Set verbosity (based on quiet_mode argument)

        self.quiet_mode = quiet_mode
        if self.quiet_mode:
            self.classifai_tqdm = lambda iterable, *args, **kwargs: iterable
            logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s", force=True)
        else:
            self.classifai_tqdm = tqdm
            logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", force=True)

        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

        # ---- Input validation (caller mistakes) -> DataValidationError / ConfigurationError
        if not isinstance(file_name, str) or not file_name.strip():
            raise DataValidationError("file_name must be a non-empty string.", context={"file_name": file_name})

        # use fsspec to get the filesystem and path for the input
        try:
            in_fs, in_path = fsspec.core.url_to_fs(file_name)
        except Exception as e:
            # check for cases where the user wants to use a gs:// path but doesn't have gcsfs installed, and raise a more helpful error message in this case
            if isinstance(e, ImportError) and file_name.startswith("gs://"):
                try:
                    check_deps(["gcsfs"], extra="gcp")
                except OptionalDependencyError as e:
                    raise OptionalDependencyError(
                        "Optional dependency 'gcsfs' is required to use gs:// files. Install with: pip install 'classifai[gcp]'.",
                    ) from e
            # for all other cases, raise a generic configuration error with context for debugging
            raise ConfigurationError(
                "Failed to read input directory with file loader.",
                context={
                    "file_name": file_name,
                    "cause": str(e),
                    "cause_type": type(e).__name__,
                },
            ) from e

        # check if the file exists in the filesystem
        if not in_fs.exists(in_path):
            raise DataValidationError("Input file does not exist.", context={"file_name": file_name})

        # check that the user has specified the correct datatype
        if data_type not in ["csv"]:
            raise DataValidationError(
                "Unsupported data_type. Choose from ['csv'].",
                context={"data_type": data_type},
            )

        # check that the vectoriser object is an instance of the VectoriserBase class and has a transform method
        if not isinstance(vectoriser, VectoriserBase):
            raise ConfigurationError(
                "Vectoriser must be an instance of Vectoriser(Base) with a .transform() method.",
                context={"vectoriser_type": type(vectoriser).__name__},
            )

        # check that batch_size is a positive integer
        if not isinstance(batch_size, int) or batch_size < 1:
            raise DataValidationError("batch_size must be an integer >= 1.", context={"batch_size": batch_size})

        # check that meta_data is a dict if provided
        if meta_data is not None and not isinstance(meta_data, dict):
            raise DataValidationError(
                "meta_data must be a dict or None.", context={"meta_data_type": type(meta_data).__name__}
            )

        # check that hooks is a dict if provided
        if hooks is not None and not isinstance(hooks, dict):
            raise DataValidationError("hooks must be a dict or None.", context={"hooks_type": type(hooks).__name__})

        # ---- Assign fields
        self.file_name = file_name
        self.data_type = data_type
        self.vectoriser = vectoriser
        self.batch_size = batch_size
        self.meta_data = meta_data if meta_data is not None else {}
        self.output_dir = output_dir
        self.vectors = None
        self.vector_shape = None
        self.num_vectors = None
        self.vectoriser_class = vectoriser.__class__.__name__
        self.hooks = {} if hooks is None else hooks
        self.skip_save = skip_save

        if self.output_dir is not None and self.skip_save:
            logging.warning(
                "VectorStore creation: output_dir is set to %s but skip_save is True, so the VectorStore will not be saved to disk. output_dir will be ignored.",
                self.output_dir,
            )

        if self.output_dir is not None and not isinstance(self.output_dir, str):
            raise DataValidationError(
                "output_dir must be a string or None.", context={"output_dir_type": type(self.output_dir).__name__}
            )

        if not self.skip_save:
            # ---- Output directory handling (filesystem problems) -> ConfigurationError
            try:
                if self.output_dir is None:
                    logging.info(
                        "No output directory specified, attempting to use input file name as output folder name."
                    )
                    normalized_file_name = os.path.basename(os.path.splitext(self.file_name)[0])
                    self.output_dir = os.path.join(normalized_file_name)
            except Exception as e:
                raise ConfigurationError(
                    "Failed to determine output directory from input file name.",
                    context={
                        "file_name": self.file_name,
                        "cause": str(e),
                        "cause_type": type(e).__name__,
                    },
                ) from e

            # use fsspec to get the filesystem and path for the output
            try:
                out_fs, out_path = fsspec.core.url_to_fs(self.output_dir)
            except Exception as e:
                # check for cases where the user wants to use a gs:// path but doesn't have gcsfs installed, and raise a more helpful error message in this case
                if isinstance(e, ImportError) and self.output_dir.startswith("gs://"):
                    try:
                        check_deps(["gcsfs"], extra="gcp")
                    except OptionalDependencyError as e:
                        raise OptionalDependencyError(
                            "Optional dependency 'gcsfs' is required to use gs:// files. Install with: pip install 'classifai[gcp]'.",
                        ) from e
                # for all other cases, raise a generic configuration error with context for debugging
                raise ConfigurationError(
                    "Failed to read output directory with file loader.",
                    context={
                        "output_dir": self.output_dir,
                        "cause": str(e),
                        "cause_type": type(e).__name__,
                    },
                ) from e

            try:
                # check if the output directory already exists, and handle according to overwrite flag
                if out_fs.exists(out_path):
                    if overwrite:
                        out_fs.rm(out_path, recursive=True)
                    else:
                        raise ConfigurationError(
                            "Output directory already exists. Pass overwrite=True to overwrite the folder.",
                            context={"output_dir": self.output_dir},
                        )
                out_fs.makedirs(out_path, exist_ok=True)
            except Exception as e:
                raise ConfigurationError(
                    "Failed to prepare output directory.",
                    context={
                        "output_dir": self.output_dir,
                        "cause": str(e),
                        "cause_type": type(e).__name__,
                    },
                ) from e
        else:
            logging.debug("skip_save is set to True, the VectorStore will not be saved to disk after creation.")

        # ---- Build index (wrap every unexpected failure) -> IndexBuildError
        try:
            self._create_vector_store_index()
        except ClassifaiError:
            # preserve already-classified errors (e.g. vectoriser raised DataValidationError)
            raise
        except Exception as e:
            raise IndexBuildError(
                "Failed to create vector store index.",
                context={
                    "file_name": self.file_name,
                    "data_type": self.data_type,
                    "batch_size": self.batch_size,
                    "cause_type": type(e).__name__,
                    "cause_message": str(e),
                },
            ) from e

        # ---- Save + derived metadata (IO/format problems) -> IndexBuildError
        self.vector_shape = self.vectors["embeddings"].to_numpy().shape[1]
        self.num_vectors = len(self.vectors)

        if not self.skip_save:
            try:
                logging.info("Gathering metadata and saving vector store / metadata...")
                vectors_out_path = os.path.join(self.output_dir, "vectors.parquet")
                self.vectors.write_parquet(
                    vectors_out_path
                )  # polars handles fsspec filesystems natively, so this will work with local and remote filesystems supported by fsspec

                metadata_out_path = os.path.join(self.output_dir, "metadata.json")
                self._save_metadata(metadata_out_path)

                logging.info("Vector Store created - files saved to %s", self.output_dir)
            except ClassifaiError:
                raise
            except Exception as e:
                raise IndexBuildError(
                    "Vector store was created but saving outputs failed.",
                    context={"cause_type": type(e).__name__, "cause_message": str(e)},
                ) from e
        else:
            logging.debug("skip_save is True, skipping saving VectorStore to disk.")

    def _save_metadata(self, path: str):
        """Saves metadata about the `VectorStore` to a JSON file.

        Args:
            path (str): The file path where the metadata JSON file will be
                saved.

        Raises:
            DataValidationError: If the path argument is invalid.
            ClassifaiError: If there are package-specific errors during
                serialisation or file writing.
            IndexBuildError: If there are failures during serialisation or file
                writing.
        """
        if not isinstance(path, str) or not path.strip():
            raise DataValidationError("path must be a non-empty string.", context={"path": path})

        try:
            # Convert meta_data types to strings for JSON serialisation
            serializable_column_meta_data = {
                key: value.__name__ if isinstance(value, type) else value
                for key, value in (self.meta_data or {}).items()
            }

            metadata = {
                "vectoriser_class": self.vectoriser_class,
                "vector_shape": self.vector_shape,
                "num_vectors": self.num_vectors,
                "batch_size": self.batch_size,
                "created_at": time.time(),
                "meta_data": serializable_column_meta_data,
            }

            # inside separate function use fsspec again to write the metadata file to support different filesystems
            out_fs, out_path = fsspec.core.url_to_fs(path)
            with out_fs.open(out_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

        except ClassifaiError:
            # Preserve package-specific exceptions unchanged
            raise
        except Exception as e:
            raise IndexBuildError(
                "Unexpected error while saving metadata file.",
                context={"path": path, "metadata": metadata, "cause_type": type(e).__name__, "cause_message": str(e)},
            ) from e

    def _create_vector_store_index(self):  # noqa: C901
        """Reads the input file, embeds text in batches, and populates self.vectors.

        Reads the configured input file (currently CSV only) into self.vectors
        with polars, selecting the label, text, and any specified metadata
        columns. A UUID is assigned to each row and the text column is embedded
        in batches of self.batch_size using self.vectoriser.transform(). The
        resulting embeddings are appended to self.vectors as an embeddings
        column.

        Raises:
            DataValidationError: If the text column contains no documents, or
                if an unsupported data_type is encountered.
            ClassifaiError: If there are any package-specific errors during
                the vector store index creation.
            IndexBuildError: If the input file cannot be read, if the
                vectoriser returns an incorrect number of embeddings for a
                given batch, or if any other failure occurs while building the
                embeddings table.
        """
        # ---- Reading source data (validation/format issues) -> DataValidationError / IndexBuildError
        try:
            if self.data_type == "csv":
                self.vectors = pl.read_csv(  # polars handles fsspec filesystems natively
                    self.file_name,
                    columns=["label", "text", *self.meta_data.keys()],
                    dtypes=self.meta_data | {"label": str, "text": str},
                )
                self.vectors = self.vectors.with_columns(
                    pl.Series("uuid", [str(uuid.uuid4()) for _ in range(self.vectors.height)])
                )
            else:
                raise DataValidationError(
                    "File type not supported. Choose from ['csv'].",
                    context={"data_type": self.data_type},
                )
        except ClassifaiError:
            raise
        except Exception as e:
            raise IndexBuildError(
                "Failed to read input file into a table.",
                context={"file_name": self.file_name, "data_type": self.data_type},
            ) from e

        logging.info("Processing file: %s...\n", self.file_name)

        # ---- Embedding / dataframe build (vectoriser failures and mismatches) -> IndexBuildError
        try:
            documents = self.vectors["text"].to_list()
            if not documents:
                raise DataValidationError(
                    "Input file contains no documents in column 'text'.",
                    context={"file_name": self.file_name},
                )

            embeddings: list[np.ndarray] = []
            for batch_id in self.classifai_tqdm(range(0, len(documents), self.batch_size)):
                batch = documents[batch_id : (batch_id + self.batch_size)]
                try:
                    batch_embeddings = self.vectoriser.transform(batch)
                except ClassifaiError:
                    # preserve vectoriser classification, but add context by re-wrapping
                    raise
                except Exception as e:
                    raise IndexBuildError(
                        "Vectoriser.transform failed during index build.",
                        context={
                            "file_name": self.file_name,
                            "vectoriser": self.vectoriser_class,
                            "batch_id": batch_id,
                            "batch_size": len(batch),
                        },
                    ) from e

                # Basic sanity check: batch should return same number of vectors as texts
                if len(batch_embeddings) != len(batch):
                    raise IndexBuildError(
                        "Vectoriser returned wrong number of embeddings for batch.",
                        context={
                            "file_name": self.file_name,
                            "vectoriser": self.vectoriser_class,
                            "batch_id": batch_id,
                            "expected": len(batch),
                            "got": len(batch_embeddings),
                        },
                    )

                embeddings.extend(batch_embeddings)

            self.vectors = self.vectors.with_columns(pl.Series(embeddings).alias("embeddings"))
        except ClassifaiError:
            raise
        except Exception as e:
            raise IndexBuildError(
                "Failed while creating embeddings and building vectors table.",
                context={
                    "file_name": self.file_name,
                    "vectoriser": self.vectoriser_class,
                    "cause_type": type(e).__name__,
                    "cause_message": str(e),
                },
            ) from e

    def embed(self, query: VectorStoreEmbedInput) -> VectorStoreEmbedOutput:  # noqa: C901
        """Generates vector embeddings from a `VectorStoreEmbedInput` object.

        Accepts a VectorStoreEmbedInput object and generates vector embeddings
        for its text content using the vectoriser attribute. Any preprocessing
        hooks set on the instance are applied to the input before embedding,
        and any postprocessing hooks are applied to the output before it is
        returned.

        Args:
            query (VectorStoreEmbedInput): Input object containing the text to
                be embedded and their corresponding ids.

        Returns:
            VectorStoreEmbedOutput: Output object containing the generated
                embeddings together with their corresponding ids and original
                texts.

        Raises:
            DataValidationError: If invalid arguments are passed.
            HookError: If a preprocessing or postprocessing hook raises an
                exception.
            ClassifaiError: If the embedding operation fails.
        """
        # ---- Validate arguments (caller mistakes) -> DataValidationError
        if not isinstance(query, VectorStoreEmbedInput):
            raise DataValidationError(
                "query must be a VectorStoreEmbedInput object.",
                context={"got_type": type(query).__name__},
            )

        # ---- Preprocess hook -> HookError
        if "embed_preprocess" in self.hooks:
            try:
                if not isinstance(self.hooks["embed_preprocess"], list):
                    self.hooks["embed_preprocess"] = [self.hooks["embed_preprocess"]]
                for hook in self.hooks["embed_preprocess"]:
                    query = hook(query)
                query = VectorStoreEmbedInput.validate(query)
            except Exception as e:
                raise HookError(
                    "embed_preprocess hook raised an exception.",
                    context={"hook": "embed_preprocess", "cause_type": type(e).__name__, "cause_message": str(e)},
                ) from e

        # ---- Main embed operation
        try:
            # Generate embeddings using the vectoriser
            embeddings = self.vectoriser.transform(query.text.to_list())

            # Create a DataFrame with id, text, and embedding fields
            results_df = VectorStoreEmbedOutput.from_data(
                {
                    "id": query.id,
                    "text": query.text,
                    "embedding": [embeddings[i] for i in range(len(embeddings))],
                }
            )

        except ClassifaiError:
            raise
        except Exception as e:
            raise ClassifaiError(
                "Embedding failed.",
                code="embed_failed",
                context={
                    "n_texts": len(query),
                    "vectoriser": self.vectoriser_class,
                    "cause_type": type(e).__name__,
                    "cause_message": str(e),
                },
            ) from e

        # ---- Postprocess hook -> HookError
        if "embed_postprocess" in self.hooks:
            try:
                if not isinstance(self.hooks["embed_postprocess"], list):
                    self.hooks["embed_postprocess"] = [self.hooks["embed_postprocess"]]
                for hook in self.hooks["embed_postprocess"]:
                    results_df = hook(results_df)
                results_df = VectorStoreEmbedOutput.validate(results_df)
            except Exception as e:
                raise HookError(
                    "embed_postprocess hook raised an exception.",
                    context={"hook": "embed_postprocess", "cause_type": type(e).__name__, "cause_message": str(e)},
                ) from e

        return results_df

    def reverse_search(  # noqa: C901, PLR0912
        self, query: VectorStoreReverseSearchInput, max_n_results: int = 100, partial_match: bool = False
    ) -> VectorStoreReverseSearchOutput:
        """Looks up documents in `vectors` by label.

        Performs a label-based (non-semantic) lookup against the stored
        documents using a `VectorStoreReverseSearchInput`. For each query
        entry, matching documents are found by comparing the query's doc_label
        against the label column of self.vectors. When partial_match is
        enabled, a document is considered a match if its label starts with the
        query label.

        Any preprocessing hooks set on the instance are applied to the input
        before searching, and any postprocessing hooks are applied to the output
        before it is returned.

        Args:
            query (VectorStoreReverseSearchInput): Input object containing the
                doc labels to look up in self.vectors and their corresponding
                ids.
            max_n_results (int): Maximum number of matching documents to return
                per query entry. Pass -1 to return all matches. Defaults to
                100.
            partial_match (bool): If True, matches documents whose label starts
                with the query label (prefix matching). If False, only exact
                label matches are returned. Defaults to False.

        Returns:
            VectorStoreReverseSearchOutput: Output object containing the
                matched documents, with columns for id, searched_doc_label,
                doc_label, doc_text, and any metadata columns configured on the
                `VectorStore`.

        Raises:
            DataValidationError: Raised if invalid arguments are passed.
            HookError: Raised if a preprocessing or postprocessing hook raises
                an exception.
            ClassifaiError: Raised if the reverse search operation fails.
        """
        # ---- Validate arguments (caller mistakes) -> DataValidationError
        if not isinstance(query, VectorStoreReverseSearchInput):
            raise DataValidationError(
                "query must be a VectorStoreReverseSearchInput object.",
                context={"got_type": type(query).__name__},
            )

        if not isinstance(max_n_results, int) or (max_n_results < 1 and max_n_results != -1):
            raise DataValidationError(
                "max_n_results must be an integer >= 1 or -1.", context={"max_n_results": max_n_results}
            )

        if len(query) == 0:
            raise DataValidationError("query is empty.", context={"n_queries": 0})

        # ---- Preprocess hook -> HookError
        if "reverse_search_preprocess" in self.hooks:
            try:
                if not isinstance(self.hooks["reverse_search_preprocess"], list):
                    self.hooks["reverse_search_preprocess"] = [self.hooks["reverse_search_preprocess"]]
                for hook in self.hooks["reverse_search_preprocess"]:
                    query = hook(query)
                query = VectorStoreReverseSearchInput.validate(query)
            except Exception as e:
                raise HookError(
                    "reverse_search_preprocess hook raised an exception.",
                    context={
                        "hook": "reverse_search_preprocess",
                        "cause_type": type(e).__name__,
                        "cause_message": str(e),
                    },
                ) from e

        try:
            # polars conversion
            paired_query = pl.DataFrame(
                {"id": query.id.astype(str).to_list(), "searched_doc_label": query.doc_label.astype(str).to_list()}
            )

            # rename vectors dataframe for reverse search return column names and joining
            docs = self.vectors.rename({"label": "doc_label", "text": "doc_text"}).with_columns(
                pl.col("doc_label").alias("doc_label_copy")
            )

            if partial_match:
                out = docs.join_where(paired_query, pl.col("doc_label").str.starts_with(pl.col("searched_doc_label")))
            else:
                out = paired_query.join(
                    docs,
                    left_on="searched_doc_label",
                    right_on="doc_label",
                    how="inner",
                ).rename({"doc_label_copy": "doc_label"})

            out = out.sort(by=["id", "searched_doc_label"], descending=[False, False])
            if max_n_results != -1:
                out = out.group_by("id").head(max_n_results)

            # get formatted table
            final_table = out.select(
                [
                    pl.col("id").cast(str),
                    pl.col("searched_doc_label").cast(str),
                    pl.col("doc_label").cast(str),
                    pl.col("doc_text").cast(str),
                    *[pl.col(key) for key in self.meta_data],
                ]
            )

            result_df = VectorStoreReverseSearchOutput.from_data(final_table.to_dict(as_series=False))

        except ClassifaiError:
            raise
        except Exception as e:
            raise ClassifaiError(
                "Reverse search failed.",
                code="reverse_search_failed",
                context={
                    "n_queries": len(query),
                    "max_n_results": max_n_results,
                    "cause_type": type(e).__name__,
                    "cause_message": str(e),
                },
            ) from e

        # ---- Postprocess hook -> HookError
        if "reverse_search_postprocess" in self.hooks:
            try:
                if not isinstance(self.hooks["reverse_search_postprocess"], list):
                    self.hooks["reverse_search_postprocess"] = [self.hooks["reverse_search_postprocess"]]
                for hook in self.hooks["reverse_search_postprocess"]:
                    result_df = hook(result_df)
                result_df = VectorStoreReverseSearchOutput.validate(result_df)
            except Exception as e:
                raise HookError(
                    "reverse_search_postprocess hook raised an exception.",
                    context={
                        "hook": "reverse_search_postprocess",
                        "cause_type": type(e).__name__,
                        "cause_message": str(e),
                    },
                ) from e

        return result_df

    def search(self, query: VectorStoreSearchInput, n_results=10, batch_size=None) -> VectorStoreSearchOutput:  # noqa: C901, PLR0912, PLR0915
        """Queries the `vectors` attribute for the most similar documents.

        Queries are processed in batches of batch_size, with each batch
        embedded using vectoriser.transform() and scored against all stored
        document embeddings via dot-product similarity (equivalent to cosine
        similarity when embeddings are L2-normalised). The top n_results
        documents are returned for each query, ordered by descending score.

        Any preprocessing hooks set on the instance are applied to the input
        before searching, and any postprocessing hooks are applied to the
        output before it is returned.

        Args:
            query (VectorStoreSearchInput): The input object containing the
                text query or list of queries to search for, with ids.
            n_results (int): Number of top results to return for each query.
                Defaults to 10.
            batch_size (int): The batch size for processing queries. Defaults
                to the batch_size set during initialisation.

        Returns:
            VectorStoreSearchOutput: The output object containing search
                results with columns for query_id, query_text, doc_label,
                doc_text, rank, score, and any associated metadata columns.

        Raises:
            DataValidationError: Raised if invalid arguments are passed.
            ConfigurationError: Raised if the `VectorStore` is not initialised.
            HookError: Raised if user-defined hooks fail.
            ClassifaiError: Raised if there is a package-specific error during
                the search operation.
            VectorisationError: Raised if query embedding fails.
        """
        # ---- Validate arguments (caller mistakes) -> DataValidationError
        if not isinstance(query, VectorStoreSearchInput):
            raise DataValidationError(
                "query must be a VectorStoreSearchInput object.",
                context={"got_type": type(query).__name__},
            )

        if not isinstance(n_results, int) or n_results < 1:
            raise DataValidationError("n_results must be an integer >= 1.", context={"n_results": n_results})

        query_batch_size = batch_size if batch_size is not None else self.batch_size

        if not isinstance(query_batch_size, int) or query_batch_size < 1:
            raise DataValidationError("batch_size must be an integer >= 1.", context={"batch_size": query_batch_size})

        if self.vectors is None:
            raise ConfigurationError("Vector store is not initialised (vectors is None).")

        if len(query) == 0:
            raise DataValidationError("query is empty.", context={"n_queries": 0})

        # ---- Preprocess hook -> DataValidationError if it returns invalid shape/type
        if "search_preprocess" in self.hooks:
            try:
                if not isinstance(self.hooks["search_preprocess"], list):
                    self.hooks["search_preprocess"] = [self.hooks["search_preprocess"]]
                for hook in self.hooks["search_preprocess"]:
                    query = hook(query)
                query = VectorStoreSearchInput.validate(query)
            except Exception as e:
                raise HookError(
                    "search_preprocess hook raised an exception.",
                    context={"hook": "search_preprocess", "cause_type": type(e).__name__, "cause_message": str(e)},
                ) from e

        # ---- Main search (wrap operational failures) -> SearchError / VectorisationError
        try:
            doc_embeddings = self.vectors["embeddings"].to_numpy()

            all_results: list[pl.DataFrame] = []

            for i in self.classifai_tqdm(range(0, len(query), query_batch_size), desc="Processing query batches"):
                query_text_batch = query.query.to_list()[i : i + query_batch_size]
                query_ids_batch = query.id.to_list()[i : i + query_batch_size]

                if len(query_text_batch) == 0:
                    continue

                # Embed query batch
                try:
                    query_vectors = self.vectoriser.transform(query_text_batch)
                except ClassifaiError:
                    raise
                except Exception as e:
                    raise VectorisationError(
                        "Failed to embed query batch.",
                        context={
                            "vectoriser": self.vectoriser_class,
                            "batch_start": i,
                            "batch_size": len(query_text_batch),
                            "n_results": n_results,
                        },
                    ) from e

                # Similarity + top-k
                cosine = query_vectors @ doc_embeddings.T

                idx = np.argpartition(cosine, -n_results, axis=1)[:, -n_results:]

                idx_sorted = np.zeros_like(idx)
                scores = np.zeros_like(idx, dtype=float)

                for j in range(idx.shape[0]):
                    row_scores = cosine[j, idx[j]]
                    sorted_indices = np.argsort(row_scores)[::-1]
                    idx_sorted[j] = idx[j, sorted_indices]
                    scores[j] = row_scores[sorted_indices]

                # Build batch result table
                result_df = pl.DataFrame(
                    {
                        "query_id": np.repeat(query_ids_batch, n_results),
                        "query_text": np.repeat(query_text_batch, n_results),
                        "rank": np.tile(np.arange(1, n_results + 1), len(query_text_batch)),
                        "score": scores.flatten(),
                    }
                )

                ranked_docs = self.vectors[idx_sorted.flatten().tolist()].select(
                    ["label", "text", *self.meta_data.keys()]
                )
                merged_df = result_df.hstack(ranked_docs).rename({"label": "doc_label", "text": "doc_text"})

                merged_df = merged_df.with_columns(
                    [
                        pl.col("doc_label").cast(str),
                        pl.col("doc_text").cast(str),
                        pl.col("rank").cast(int),
                        pl.col("score").cast(float),
                        pl.col("query_id").cast(str),
                        pl.col("query_text").cast(str),
                    ]
                )

                all_results.append(merged_df)

            if not all_results:
                # Shouldn't happen if len(query)>0, but keep it safe.
                empty = pl.DataFrame(
                    schema={
                        "query_id": pl.Utf8,
                        "query_text": pl.Utf8,
                        "doc_label": pl.Utf8,
                        "doc_text": pl.Utf8,
                        "rank": pl.Int64,
                        "score": pl.Float64,
                        **dict.fromkeys(self.meta_data.keys(), pl.Utf8),
                    }
                )
                return VectorStoreSearchOutput.from_data(empty.to_dict(as_series=False))

            reordered_df = pl.concat(all_results).select(
                ["query_id", "query_text", "doc_label", "doc_text", "rank", "score", *self.meta_data.keys()]
            )

            result_df = VectorStoreSearchOutput.from_data(reordered_df.to_dict(as_series=False))

        except ClassifaiError:
            raise
        except Exception as e:
            raise ClassifaiError(
                "Search failed.",
                code="search_failed",
                context={
                    "n_queries": len(query),
                    "batch_size": query_batch_size,
                    "n_results": n_results,
                    "cause_type": type(e).__name__,
                    "cause_message": str(e),
                },
            ) from e

        # ---- Postprocess hook -> DataValidationError if it returns invalid shape/type
        if "search_postprocess" in self.hooks:
            try:
                if not isinstance(self.hooks["search_postprocess"], list):
                    self.hooks["search_postprocess"] = [self.hooks["search_postprocess"]]
                for hook in self.hooks["search_postprocess"]:
                    result_df = hook(result_df)
                result_df = VectorStoreSearchOutput.validate(result_df)
            except Exception as e:
                raise HookError(
                    "search_postprocessing hook raised an exception.",
                    context={"hook": "search_postprocess", "cause_type": type(e).__name__, "cause_message": str(e)},
                ) from e

        return result_df

    @classmethod
    def from_filespace(  # noqa: C901, PLR0912, PLR0915
        cls, folder_path, vectoriser, batch_size: int | None = None, hooks: dict | None = None, quiet_mode: bool = False
    ):
        """Creates a `VectorStore` instance from a saved filespace folder.

        Reads metadata.json and vectors.parquet from folder_path using fsspec,
        so both local and remote paths (e.g. gs://) are supported. The
        vectoriser class name stored in metadata.json must match the class name
        of the supplied vectoriser object. The instance is constructed via
        object.__new__, so __init__ is never called and no embeddings are
        generated.

        Note: the returned instance does not have output_dir or skip_save
        attributes set. vector_shape and num_vectors are read directly from
        metadata.json without being cross-checked against the actual contents
        of the parquet file.

        > <div style="color:darkred">**v1.0.0 compatibility:** VectorStores built before v1.1.0 do not
            store `batch_size` in `metadata.json`. Pass `batch_size` explicitly
            when loading such stores, otherwise an `IndexBuildError` is raised.
            </div>

        Args:
            folder_path (str): Path to the folder containing metadata.json and
                vectors.parquet. Supports any fsspec-compatible path
                (local, gs://, etc.).
            batch_size (int | None): Overrides the batch_size stored in
                metadata. Defaults to None, which uses the value from
                metadata.json. Must be set explicitly when loading a store
                built before v1.1.0, as older metadata files do not include
                this field.
            vectoriser: An object with a callable .transform(texts) method. Its
                class name must match the vectoriser_class value stored in
                metadata.json.
            hooks (dict | None): A dictionary of user-defined hooks for preprocessing
                and postprocessing. Defaults to None.
            quiet_mode (bool): Whether to minimise verbose output, such as
                progress bars. Defaults to False.

        Returns:
            VectorStore: A `VectorStore` instance with vectors populated
                from the parquet file. file_name, data_type, and
                batch_size are all set to None.

        Raises:
            DataValidationError: If folder_path is not a non-empty string,
                does not point to an existing directory, if metadata.json
                is missing or malformed, or if vectors.parquet is missing,
                empty, or does not contain the required columns.
            OptionalDependencyError: If the user attempts to use a gs:// path
                without having gcsfs installed.
            ConfigurationError: If vectoriser does not have a callable
                .transform() method, if the fsspec path cannot be resolved,
                or if the vectoriser class name does not match the one stored in
                metadata.json.
            IndexBuildError: If metadata.json or vectors.parquet cannot
                be read or parsed, or if the instance cannot be constructed.
        """
        # ---- Validate arguments (caller mistakes) -> DataValidationError / ConfigurationError
        if not isinstance(folder_path, str) or not folder_path.strip():
            raise DataValidationError("folder_path must be a non-empty string.", context={"folder_path": folder_path})

        # use fsspec to get the filesystem and path for the input
        try:
            in_fs, in_path = fsspec.core.url_to_fs(folder_path)
        except Exception as e:
            # check for cases where the user wants to use a gs:// path but doesn't have gcsfs installed, and raise a more helpful error message in this case
            if isinstance(e, ImportError) and folder_path.startswith("gs://"):
                try:
                    check_deps(["gcsfs"], extra="gcp")
                except OptionalDependencyError as e:
                    raise OptionalDependencyError(
                        "Optional dependency 'gcsfs' is required to use gs:// filespaces. Install with: pip install 'classifai[gcp]'.",
                    ) from e
            # for all other cases, raise a generic configuration error with context for debugging
            raise ConfigurationError(
                "Failed to read input directory with file loader.",
                context={
                    "folder_path": folder_path,
                    "cause": str(e),
                    "cause_type": type(e).__name__,
                },
            ) from e

        # check if the folder exists in the filesystem
        if not in_fs.isdir(in_path):
            raise DataValidationError(
                "folder_path must be an existing directory.", context={"folder_path": folder_path}
            )

        if not hasattr(vectoriser, "transform") or not callable(getattr(vectoriser, "transform", None)):
            raise ConfigurationError(
                "vectoriser must provide a callable .transform(texts) method.",
                context={"vectoriser_type": type(vectoriser).__name__},
            )

        if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
            raise DataValidationError("batch_size must be an integer >= 1 or None.", context={"batch_size": batch_size})

        if hooks is not None and not isinstance(hooks, dict):
            raise DataValidationError("hooks must be a dict or None.", context={"hooks_type": type(hooks).__name__})

        # ---- Load metadata -> IndexBuildError
        metadata_in_path = os.path.join(in_path, "metadata.json")
        if not in_fs.exists(metadata_in_path):
            raise DataValidationError(
                "Metadata file not found in folder_path.",
                context={"folder_path": folder_path, "metadata_path": metadata_in_path},
            )

        try:
            with in_fs.open(metadata_in_path, encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            raise IndexBuildError(
                "Failed to read metadata.json.",
                context={"metadata_path": metadata_in_path, "cause_type": type(e).__name__, "cause_message": str(e)},
            ) from e

        # ---- Validate metadata content -> DataValidationError
        if not isinstance(metadata, dict):
            raise DataValidationError(
                "metadata.json did not contain a JSON object.",
                context={"metadata_path": metadata_in_path, "metadata_type": type(metadata).__name__},
            )

        required_keys = ["vectoriser_class", "vector_shape", "num_vectors", "batch_size", "created_at", "meta_data"]
        missing = [k for k in required_keys if k not in metadata]
        if missing:
            raise DataValidationError(
                "Metadata file is missing required keys.",
                context={"metadata_path": metadata_in_path, "missing_keys": missing},
            )

        if not isinstance(metadata["meta_data"], dict):
            raise DataValidationError(
                "metadata.meta_data must be an object/dict.",
                context={"metadata_path": metadata_in_path, "meta_data_type": type(metadata["meta_data"]).__name__},
            )

        # ---- Deserialize meta_data types safely -> DataValidationError
        try:
            # get the column metadata and convert types to built-in types
            deserialized_column_meta_data = {
                key: getattr(__builtins__, value, value)  # Use built-in types or keep as-is
                for key, value in metadata["meta_data"].items()
            }
        except Exception as e:
            raise DataValidationError(
                "Unable to deserialize metadata column types from metadata in metadata file.",
                context={
                    "metadata_path": metadata_in_path,
                    "meta_data": metadata["meta_data"],
                    "cause_type": type(e).__name__,
                    "cause_message": str(e),
                },
            ) from e

        # ---- Load parquet -> IndexBuildError / DataValidationError
        vectors_in_path = os.path.join(folder_path, "vectors.parquet")
        if not in_fs.exists(vectors_in_path):
            raise DataValidationError(
                "Vectors Parquet file not found in folder_path.",
                context={"folder_path": folder_path, "vectors_path": vectors_in_path},
            )

        required_columns = ["label", "text", "embeddings", "uuid", *deserialized_column_meta_data.keys()]

        try:
            df = pl.read_parquet(vectors_in_path, columns=required_columns)  # polars handles fsspec path natively
        except Exception as e:
            raise IndexBuildError(
                "Failed to read vectors.parquet.",
                context={
                    "vectors_path": vectors_in_path,
                    "cause_type": type(e).__name__,
                    "cause_message": str(e),
                },
            ) from e

        if df.is_empty():
            raise DataValidationError(
                "Vectors Parquet file is empty.",
                context={"vectors_path": vectors_in_path},
            )

        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            raise DataValidationError(
                "Vectors Parquet file is missing required columns.",
                context={"vectors_path": vectors_in_path, "missing_columns": missing_cols},
            )

        # ---- Validate vectoriser class match -> ConfigurationError
        if metadata["vectoriser_class"] != vectoriser.__class__.__name__:
            raise ConfigurationError(
                "Vectoriser class in metadata does not match provided vectoriser.",
                context={
                    "metadata_vectoriser_class": metadata["vectoriser_class"],
                    "provided_vectoriser_class": vectoriser.__class__.__name__,
                },
            )

        # ---- Construct instance without __init__ and assign fields
        try:
            vector_store = object.__new__(cls)
            vector_store.file_name = None
            vector_store.data_type = None
            vector_store.vectoriser = vectoriser
            vector_store.batch_size = batch_size if batch_size is not None else metadata["batch_size"]
            vector_store.meta_data = deserialized_column_meta_data
            vector_store.vectors = df
            vector_store.vector_shape = metadata["vector_shape"]
            vector_store.num_vectors = metadata["num_vectors"]
            vector_store.vectoriser_class = metadata["vectoriser_class"]
            vector_store.hooks = {} if hooks is None else hooks
            vector_store.quiet_mode = quiet_mode
            if vector_store.quiet_mode:
                vector_store.classifai_tqdm = lambda iterable, *args, **kwargs: iterable
                logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s", force=True)
            else:
                vector_store.classifai_tqdm = tqdm
                logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", force=True)
            logging.getLogger("httpcore").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

        except Exception as e:
            raise IndexBuildError(
                "Failed to initialise VectorStore instance from filespace.",
                context={
                    "folder_path": folder_path,
                    "metadata_path": metadata_in_path,
                    "vectors_path": vectors_in_path,
                    "cause_type": type(e).__name__,
                    "cause_message": str(e),
                },
            ) from e

        return vector_store
