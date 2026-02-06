# pylint: disable=C0301
"""This module provides functionality for creating a vector index from a text file.
It defines the `VectorStore` class, which is used to model and create vector databases
from CSV text files using a vectoriser object.

This class interacts with the Vectoriser class from the vectorisers submodule,
expecting that any vector model used to generate embeddings used in the
VectorStore objects is an instance of one of these classes, most notably
that each vectoriser object should have a transform method.

Key Features:
- Batch processing of input files to handle large datasets.
- Support for CSV file format (additional formats may be added in future updates).
- Integration with a custom embedder for generating vector embeddings.
- Logging for tracking progress and handling errors during processing.

Dependencies:
- polars: For handling data in tabular format and saving it as a Parquet file.
- tqdm: For displaying progress bars during batch processing.
- numpy: for vector cosine similarity calculations
- A custom file iterator (`iter_csv`) for reading input files in batches.

Usage:
This module is intended to be used with the Vectoriers mdodule and the
the servers module from ClassifAI, to created scalable, modular, searchable
vector databases from your own text data.
"""

import json
import logging
import os
import shutil
import time
import uuid

import numpy as np
import polars as pl
from tqdm.autonotebook import tqdm

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

# Configure logging for your application
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


class VectorStore:
    """A class to model and create 'VectorStore' objects for building and searching vector databases from CSV text files.

    Attributes:
        file_name (str): the original file with the knowledgebase to build the vector store
        data_type (str): the data type of the original file (curently only csv supported)
        vectoriser (object): A Vectoriser object from the corresponding ClassifAI Pacakge module
        batch_size (int): the batch size to pass to the vectoriser when embedding
        meta_data (dict[str:type]): key-value pairs of metadata to extract from the input file and their correpsonding types
        output_dir (str): the path to the output directory where the VectorStore will be saved
        vectors (np.array): a numpy array of vectors for the vector DB
        vector_shape (int): the dimension of the vectors
        num_vectors (int): how many vectors are in the vector store
        vectoriser_class (str): the type of vectoriser used to create embeddings
        hooks (dict): A dictionary of user-defined hooks for preprocessing and postprocessing.
    """

    def __init__(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        file_name,
        data_type,
        vectoriser,
        batch_size=8,
        meta_data=None,
        output_dir=None,
        overwrite=False,
        hooks=None,
    ):
        """Initializes the VectorStore object by processing the input CSV file and generating
        vector embeddings.

        Args:
            file_name (str): The name of the input CSV file.
            data_type (str): The type of input data (currently supports only "csv").
            vectoriser (object): The vectoriser object used to transform text into
                                vector embeddings.
            batch_size (int, optional): The batch size for processing the input file and batching to
            vectoriser. Defaults to 8.
            meta_data (dict, optional): key,value pair metadata column names to extract from the input file and their types.
                                Defaults to None.
            output_dir (str, optional): The directory where the vector store will be saved.
                                Defaults to None, where input file name will be used.
            overwrite (bool, optional): If True, allows overwriting existing folders with the same name. Defaults to false to prevent accidental overwrites.
            hooks (dict, optional): A dictionary of user-defined hooks for preprocessing and postprocessing. Defaults to None.


        Raises:
            DataValidationError: If input arguments are invalid or if there are issues with the input file.
            ConfigurationError: If there are configuration issues, such as output directory problems.
            IndexBuildError: If there are failures during index building or saving outputs.
        """
        # ---- Input validation (caller mistakes) -> DataValidationError / ConfigurationError
        if not isinstance(file_name, str) or not file_name.strip():
            raise DataValidationError("file_name must be a non-empty string.", context={"file_name": file_name})

        if not os.path.exists(file_name):
            raise DataValidationError("Input file does not exist.", context={"file_name": file_name})

        if data_type not in ["csv"]:
            raise DataValidationError(
                "Unsupported data_type. Choose from ['csv'].",
                context={"data_type": data_type},
            )

        if not isinstance(vectoriser, VectoriserBase):
            raise ConfigurationError(
                "vectoriser must be an instance of Vectoriser(Base) with a .transform(texts) method.",
                context={"vectoriser_type": type(vectoriser).__name__},
            )

        if not isinstance(batch_size, int) or batch_size < 1:
            raise DataValidationError("batch_size must be an integer >= 1.", context={"batch_size": batch_size})

        if meta_data is not None and not isinstance(meta_data, dict):
            raise DataValidationError(
                "meta_data must be a dict or None.", context={"meta_data_type": type(meta_data).__name__}
            )

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

        # ---- Output directory handling (filesystem problems) -> ConfigurationError
        try:
            if self.output_dir is None:
                logging.info("No output directory specified, attempting to use input file name as output folder name.")
                normalized_file_name = os.path.basename(os.path.splitext(self.file_name)[0])
                self.output_dir = os.path.join(normalized_file_name)

            if os.path.isdir(self.output_dir):
                if overwrite:
                    shutil.rmtree(self.output_dir)
                else:
                    raise ConfigurationError(
                        "Output directory already exists. Pass overwrite=True to overwrite the folder.",
                        context={"output_dir": self.output_dir},
                    )
            os.makedirs(self.output_dir, exist_ok=True)
        except ClassifaiError:
            raise
        except Exception as e:
            raise ConfigurationError(
                "Failed to prepare output directory.",
                context={"output_dir": self.output_dir},
            ) from e

        # ---- Build index (wrap every unexpected failure) -> IndexBuildError
        try:
            self._create_vector_store_index()
        except ClassifaiError:
            # preserve already-classified errors (e.g. vectoriser raised DataValidationError)
            raise
        except Exception as e:
            raise IndexBuildError(
                "Failed to create vector store index.",
                context={"file_name": self.file_name, "data_type": self.data_type, "batch_size": self.batch_size},
            ) from e

        # ---- Save + derived metadata (IO/format problems) -> IndexBuildError
        try:
            logging.info("Gathering metadata and saving vector store / metadata...")

            self.vector_shape = self.vectors["embeddings"].to_numpy().shape[1]
            self.num_vectors = len(self.vectors)

            self.vectors.write_parquet(os.path.join(self.output_dir, "vectors.parquet"))
            self._save_metadata(os.path.join(self.output_dir, "metadata.json"))

            logging.info("Vector Store created - files saved to %s", self.output_dir)
        except ClassifaiError:
            raise
        except Exception as e:
            raise IndexBuildError(
                "Vector store was created but saving outputs failed.",
                context={"output_dir": self.output_dir},
            ) from e

    def _save_metadata(self, path):
        """Saves metadata about the vector store to a JSON file.

        Args:
            path (str): The file path where the metadata JSON file will be saved.

        Raises:
            DataValidationError: If the path argument is invalid.
            IndexBuildError: If there are failures during serialization or file writing.
        """
        if not isinstance(path, str) or not path.strip():
            raise DataValidationError("path must be a non-empty string.", context={"path": path})

        try:
            # Convert meta_data types to strings for JSON serialization
            serializable_column_meta_data = {
                key: value.__name__ if isinstance(value, type) else value
                for key, value in (self.meta_data or {}).items()
            }

            metadata = {
                "vectoriser_class": self.vectoriser_class,
                "vector_shape": self.vector_shape,
                "num_vectors": self.num_vectors,
                "created_at": time.time(),
                "meta_data": serializable_column_meta_data,
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

        except ClassifaiError:
            # Preserve package-specific exceptions unchanged
            raise
        except (TypeError, ValueError) as e:
            # Usually means something in `meta_data` isn't JSON-serializable
            raise IndexBuildError(
                "Failed to serialize vector store metadata to JSON.",
                context={"path": path},
            ) from e
        except OSError as e:
            # Permission denied, invalid path, disk full, etc.
            raise IndexBuildError(
                "Failed to write metadata file.",
                context={"path": path},
            ) from e
        except Exception as e:
            raise IndexBuildError(
                "Unexpected error while saving metadata file.",
                context={"path": path},
            ) from e

    def _create_vector_store_index(self):  # noqa: C901
        """Processes text strings in batches, generates vector embeddings, and creates the
        vector store.
        Called from the constructor once other metadata has been set.
        Iterates over data in batches, stores batch data and generated embeddings.
        Creates a Polars DataFrame with the captured data and embeddings, and saves it as
        a Parquet file in the output_dir attribute, and stores in the vectors attribute.

        Raises:
            DataValidationError: If there are issues reading or validating the input file.
            IndexBuildError: If there are failures during embedding or building the vectors table.
        """
        # ---- Reading source data (validation/format issues) -> DataValidationError / IndexBuildError
        try:
            if self.data_type == "csv":
                self.vectors = pl.read_csv(
                    self.file_name,
                    columns=["id", "text", *self.meta_data.keys()],
                    dtypes=self.meta_data | {"id": str, "text": str},
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
            for batch_id in tqdm(range(0, len(documents), self.batch_size)):
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
                context={"file_name": self.file_name, "vectoriser": self.vectoriser_class},
            ) from e

    def embed(self, query: VectorStoreEmbedInput) -> VectorStoreEmbedOutput:
        """Converts text into vector embeddings using the vectoriser and returns a VectorStoreEmbedOutput dataframe with columns 'id', 'text', and 'embedding'.

        Args:
            query (VectorStoreEmbedInput): The VectorStoreEmbedInput object containing the strings to be embedded and their ids.

        Returns:
            VectorStoreEmbedOutput: The output object containing the embeddings along with their corresponding ids and texts.

        Raises:
            DataValidationError: Raised if invalid arguments are passed.
            HookError: Raised if user-defined hooks fail.
            ClassifaiError: Raised if embedding operation fails.
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
                modified_query = self.hooks["embed_preprocess"](query)
                query = VectorStoreEmbedInput.validate(modified_query)
            except Exception as e:
                raise HookError(
                    "embed_preprocess hook raised an exception.",
                    context={"hook": "embed_preprocess"},
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
                context={"n_texts": len(query), "vectoriser": self.vectoriser_class},
            ) from e

        # ---- Postprocess hook -> HookError
        if "embed_postprocess" in self.hooks:
            try:
                modified_results_df = self.hooks["embed_postprocess"](results_df)
                results_df = VectorStoreEmbedOutput.validate(modified_results_df)
            except Exception as e:
                raise HookError(
                    "embed_postprocess hook raised an exception.",
                    context={"hook": "embed_postprocess"},
                ) from e

        return results_df

    def reverse_search(self, query: VectorStoreReverseSearchInput, n_results=100) -> VectorStoreReverseSearchOutput:
        """Reverse searches the vector store using a VectorStoreReverseSearchInput object and returns
        matched results in VectorStoreReverseSearchOutput object. In batches, converts users text queries into vector embeddings,
        computes cosine similarity with stored document vectors, and retrieves the top results.

        Args:
            query (VectorStoreReverseSearchInput): A VectorStoreReverseSearchInput object containing the text query or list of queries to search for with ids.
            n_results (int, optional): Number of top results to return for each query. Default 100.

        Returns:
            result_df (VectorStoreReverseSearchOutput): A VectorStoreReverseSearchOutput object containing reverse search results with columns for query ID, query text,
                          document ID, document text and any associated metadata columns.

        Raises:
            DataValidationError: Raised if invalid arguments are passed.
            HookError: Raised if user-defined hooks fail.
            ClassifaiError: Raised if reverse search operation fails.
        """
        # ---- Validate arguments (caller mistakes) -> DataValidationError
        if not isinstance(query, VectorStoreReverseSearchInput):
            raise DataValidationError(
                "query must be a VectorStoreReverseSearchInput object.",
                context={"got_type": type(query).__name__},
            )

        if not isinstance(n_results, int) or n_results < 1:
            raise DataValidationError("n_results must be an integer >= 1.", context={"n_results": n_results})

        if len(query) == 0:
            raise DataValidationError("query is empty.", context={"n_queries": 0})

        # ---- Preprocess hook -> HookError
        if "reverse_search_preprocess" in self.hooks:
            try:
                modified_query = self.hooks["reverse_search_preprocess"](query)
                query = VectorStoreReverseSearchInput.validate(modified_query)
            except Exception as e:
                raise HookError(
                    "reverse_search_preprocess hook raised an exception.",
                    context={"hook": "reverse_search_preprocess"},
                ) from e

        try:
            # polars conversion
            paired_query = pl.DataFrame(
                {"id": query.id.astype(str).to_list(), "doc_id": query.doc_id.astype(str).to_list()}
            )

            # join query with vdb to get matches
            joined_table = paired_query.join(self.vectors.rename({"id": "doc_id"}), on="doc_id", how="inner")

            # get formatted table
            final_table = joined_table.select(
                [
                    pl.col("id").cast(str),
                    pl.col("doc_id").cast(str),
                    pl.col("text").cast(str).alias("doc_text"),
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
                context={"n_queries": len(query), "n_results": n_results},
            ) from e

        # ---- Postprocess hook -> HookError
        if "reverse_search_postprocess" in self.hooks:
            try:
                modified_result_df = self.hooks["reverse_search_postprocess"](result_df)
                result_df = VectorStoreReverseSearchOutput.validate(modified_result_df)
            except Exception as e:
                raise HookError(
                    "reverse_search_postprocess hook raised an exception.",
                    context={"hook": "reverse_search_postprocess"},
                ) from e

        return result_df

    def search(self, query: VectorStoreSearchInput, n_results=10, batch_size=8) -> VectorStoreSearchOutput:  # noqa: C901, PLR0912, PLR0915
        """Searches the vector store using queries from a VectorStoreSearchInput object and returns
        ranked results in VectorStoreSearchOutput object. In batches, converts users text queries into vector embeddings,
        computes cosine similarity with stored document vectors, and retrieves the top results.

        Args:
            query (VectorStoreSearchInput): A VectoreStoreSearchInput object containing the text query or list of queries to search for with ids.
            n_results (int, optional): Number of top results to return for each query. Default 10.
            batch_size (int, optional): The batch size for processing queries. Default 8.

        Returns:
            result_df (VectorStoreSearchOutput): A VectorStoreSearchOutput object containing search results with columns for query ID, query text,
                          document ID, document text, rank, score, and any associated metadata columns.

        Raises:
            DataValidationError: Raised if invalid arguments are passed.
            ConfigurationError: Raised if the vector store is not initialized.
            HookError: Raised if user-defined hooks fail.
            VectorisationError: Raised if embedding queries fails.
        """
        # ---- Validate arguments (caller mistakes) -> DataValidationError
        if not isinstance(query, VectorStoreSearchInput):
            raise DataValidationError(
                "query must be a VectorStoreSearchInput object.",
                context={"got_type": type(query).__name__},
            )

        if not isinstance(n_results, int) or n_results < 1:
            raise DataValidationError("n_results must be an integer >= 1.", context={"n_results": n_results})

        if not isinstance(batch_size, int) or batch_size < 1:
            raise DataValidationError("batch_size must be an integer >= 1.", context={"batch_size": batch_size})

        if self.vectors is None:
            raise ConfigurationError("Vector store is not initialized (vectors is None).")

        if len(query) == 0:
            raise DataValidationError("query is empty.", context={"n_queries": 0})

        # ---- Preprocess hook -> DataValidationError if it returns invalid shape/type
        if "search_preprocess" in self.hooks:
            try:
                modified_query = self.hooks["search_preprocess"](query)
                query = VectorStoreSearchInput.validate(modified_query)
            except Exception as e:
                raise HookError(
                    "search_preprocess hook raised an exception.",
                    context={"hook": "search_preprocess"},
                ) from e

        # ---- Main search (wrap operational failures) -> SearchError / VectorisationError
        try:
            doc_embeddings = self.vectors["embeddings"].to_numpy()

            all_results: list[pl.DataFrame] = []

            for i in tqdm(range(0, len(query), batch_size), desc="Processing query batches"):
                query_text_batch = query.query.to_list()[i : i + batch_size]
                query_ids_batch = query.id.to_list()[i : i + batch_size]

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
                        "rank": np.tile(np.arange(n_results), len(query_text_batch)),
                        "score": scores.flatten(),
                    }
                )

                ranked_docs = self.vectors[idx_sorted.flatten().tolist()].select(["id", "text", *self.meta_data.keys()])
                merged_df = result_df.hstack(ranked_docs).rename({"id": "doc_id", "text": "doc_text"})

                merged_df = merged_df.with_columns(
                    [
                        pl.col("doc_id").cast(str),
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
                        "doc_id": pl.Utf8,
                        "doc_text": pl.Utf8,
                        "rank": pl.Int64,
                        "score": pl.Float64,
                        **dict.fromkeys(self.meta_data.keys(), pl.Utf8),
                    }
                )
                return VectorStoreSearchOutput.from_data(empty.to_pandas())

            reordered_df = pl.concat(all_results).select(
                ["query_id", "query_text", "doc_id", "doc_text", "rank", "score", *self.meta_data.keys()]
            )

            result_df = VectorStoreSearchOutput.from_data(reordered_df.to_pandas())

        except ClassifaiError:
            raise
        except Exception as e:
            raise ClassifaiError(
                "Search failed.",
                code="search_failed",
                context={"n_queries": len(query), "batch_size": batch_size, "n_results": n_results},
            ) from e

        # ---- Postprocess hook -> DataValidationError if it returns invalid shape/type
        if "search_postprocess" in self.hooks:
            try:
                modified_result_df = self.hooks["search_postprocess"](result_df)
                result_df = VectorStoreSearchOutput.validate(modified_result_df)
            except Exception as e:
                raise HookError(
                    "search_postprocessing hook raised an exception.",
                    context={"hook": "search_postprocess"},
                ) from e

        return result_df

    @classmethod
    def from_filespace(cls, folder_path, vectoriser, hooks: dict | None = None):  # noqa: C901, PLR0912, PLR0915
        """Creates a `VectorStore` instance from stored metadata and Parquet files.
        This method reads the metadata and vectors from the specified folder,
        validates the contents, and initializes a `VectorStore` object with the
        loaded data. It checks that the metadata contains the required keys,
        that the Parquet file exists and is not empty, and that the vectoriser class
        matches the one used to create the vectors. If any checks fail, it raises
        a `ValueError` with an appropriate message.
        This method is useful for loading previously created vector stores without
        needing to reprocess the original text data.

        Args:
            folder_path (str): The folder path containing the metadata and Parquet files.
            vectoriser (object): The vectoriser object used to transform text into vector embeddings.
            hooks (dict, optional): A dictionary of user-defined hooks for preprocessing and postprocessing. Defaults to None.

        Returns:
            VectorStore: An instance of the `VectorStore` class.

        Raises:
            DataValidationError: If input arguments are invalid or if there are issues with the metadata or Parquet files.
            ConfigurationError: If there are configuration issues, such as vectoriser mismatches.
            IndexBuildError: If there are failures during loading or parsing the files.
        """
        # ---- Validate arguments (caller mistakes) -> DataValidationError / ConfigurationError
        if not isinstance(folder_path, str) or not folder_path.strip():
            raise DataValidationError("folder_path must be a non-empty string.", context={"folder_path": folder_path})

        if not os.path.isdir(folder_path):
            raise DataValidationError(
                "folder_path must be an existing directory.", context={"folder_path": folder_path}
            )

        if not hasattr(vectoriser, "transform") or not callable(getattr(vectoriser, "transform", None)):
            raise ConfigurationError(
                "vectoriser must provide a callable .transform(texts) method.",
                context={"vectoriser_type": type(vectoriser).__name__},
            )

        if hooks is not None and not isinstance(hooks, dict):
            raise DataValidationError("hooks must be a dict or None.", context={"hooks_type": type(hooks).__name__})

        # ---- Load metadata -> IndexBuildError
        metadata_path = os.path.join(folder_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise DataValidationError(
                "Metadata file not found in folder_path.",
                context={"folder_path": folder_path, "metadata_path": metadata_path},
            )

        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            raise IndexBuildError(
                "Failed to read metadata.json.",
                context={"metadata_path": metadata_path},
            ) from e

        # ---- Validate metadata content -> DataValidationError
        if not isinstance(metadata, dict):
            raise DataValidationError(
                "metadata.json did not contain a JSON object.",
                context={"metadata_path": metadata_path, "metadata_type": type(metadata).__name__},
            )

        required_keys = ["vectoriser_class", "vector_shape", "num_vectors", "created_at", "meta_data"]
        missing = [k for k in required_keys if k not in metadata]
        if missing:
            raise DataValidationError(
                "Metadata file is missing required keys.",
                context={"metadata_path": metadata_path, "missing_keys": missing},
            )

        if not isinstance(metadata["meta_data"], dict):
            raise DataValidationError(
                "metadata.meta_data must be an object/dict.",
                context={"metadata_path": metadata_path, "meta_data_type": type(metadata["meta_data"]).__name__},
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
                context={"metadata_path": metadata_path, "meta_data": metadata["meta_data"]},
            ) from e

        # ---- Load parquet -> IndexBuildError / DataValidationError
        vectors_path = os.path.join(folder_path, "vectors.parquet")
        if not os.path.exists(vectors_path):
            raise DataValidationError(
                "Vectors Parquet file not found in folder_path.",
                context={"folder_path": folder_path, "vectors_path": vectors_path},
            )

        required_columns = ["id", "text", "embeddings", "uuid", *deserialized_column_meta_data.keys()]

        try:
            df = pl.read_parquet(vectors_path, columns=required_columns)
        except Exception as e:
            raise IndexBuildError(
                "Failed to read vectors.parquet.",
                context={"vectors_path": vectors_path},
            ) from e

        if df.is_empty():
            raise DataValidationError(
                "Vectors Parquet file is empty.",
                context={"vectors_path": vectors_path},
            )

        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            raise DataValidationError(
                "Vectors Parquet file is missing required columns.",
                context={"vectors_path": vectors_path, "missing_columns": missing_cols},
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
        vector_store = object.__new__(cls)
        vector_store.file_name = None
        vector_store.data_type = None
        vector_store.vectoriser = vectoriser
        vector_store.batch_size = None
        vector_store.meta_data = deserialized_column_meta_data
        vector_store.vectors = df
        vector_store.vector_shape = metadata["vector_shape"]
        vector_store.num_vectors = metadata["num_vectors"]
        vector_store.vectoriser_class = metadata["vectoriser_class"]
        vector_store.hooks = {} if hooks is None else hooks

        return vector_store
