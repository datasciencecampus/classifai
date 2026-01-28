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

    def __init__(  # noqa: PLR0913
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
            ValueError: If the data type is not supported or if the folder name conflicts with an existing folder.
        """
        # Run the Pydantic validator first which will raise errors if the inputs are invalid

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

        if self.data_type not in ["csv"]:
            raise ValueError(f"Data type '{self.data_type}' not supported. Choose from ['csv'].")

        if self.output_dir is None:
            logging.info("No output directory specified, attempting to use input file name as output folder name.")

            # Normalize the file name to ensure it doesn't include relative paths or extensions
            normalized_file_name = os.path.basename(os.path.splitext(self.file_name)[0])
            # Check if the folder exists in the specified subdirectory
            self.output_dir = os.path.join(normalized_file_name)
            if os.path.isdir(self.output_dir):
                if overwrite:
                    shutil.rmtree(self.output_dir)
                else:
                    raise ValueError(
                        f"The name '{self.output_dir}' is already used as a folder in the subdirectory. Pass overwrite=True to overwrite the folder."
                    )
            os.makedirs(self.output_dir, exist_ok=True)

        else:
            if os.path.isdir(self.output_dir):
                if overwrite:
                    shutil.rmtree(self.output_dir)
                else:
                    raise ValueError(
                        f"The name '{self.output_dir}' is already used as a folder in the subdirectory. Pass overwrite=True to overwrite the folder."
                    )
            os.makedirs(self.output_dir, exist_ok=True)

        self._create_vector_store_index()

        logging.info("Gathering metadata and saving vector store / metadata...")

        self.vector_shape = self.vectors["embeddings"].to_numpy().shape[1]
        self.num_vectors = len(self.vectors)

        ## save everything to the folder etc: metadata, parquet and vectoriser
        self.vectors.write_parquet(os.path.join(self.output_dir, "vectors.parquet"))
        self._save_metadata(os.path.join(self.output_dir, "metadata.json"))

        logging.info("Vector Store created - files saved to %s", self.output_dir)

    def _save_metadata(self, path):
        """Saves metadata about the vector store to a JSON file.

        Args:
            path (str): The file path where the metadata JSON file will be saved.

        Raises:
            Exception: If an error occurs while saving the metadata file.
        """
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
        except Exception:
            logging.error("Something went wrong trying to save the metadata file")
            raise

    def _create_vector_store_index(self):
        """Processes text strings in batches, generates vector embeddings, and creates the
        vector store.
        Called from the constructor once other metadata has been set.
        Iterates over data in batches, stores batch data and generated embeddings.
        Creates a Polars DataFrame with the captured data and embeddings, and saves it as
        a Parquet file in the output_dir attribute, and stores in the vectors attribute.

        Raises:
            Exception: If an error occurs during file processing or vector generation.
        """
        # NOTE: read_excel schema_overrides only allows polars datatypes, not python built-in types
        #       Excel support disabled until we decide how to handle this.
        #
        # if self.data_type == "excel":
        #     self.vectors = pl.read_excel(
        #         self.file_name,
        #         has_header=True,
        #         columns=["id", "text", *self.meta_data.keys()],
        #         schema_overrides={"id": pl.String, "text": pl.String} | self.meta_data,
        #     )
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
            raise ValueError("File type not supported: {self.data_type}. Choose from ['csv'].")

        logging.info("Processing file: %s...\n", self.file_name)
        try:
            documents = self.vectors["text"].to_list()
            embeddings = []
            for batch_id in tqdm(range(0, len(documents), self.batch_size)):
                batch = documents[batch_id : (batch_id + self.batch_size)]
                embeddings.extend(self.vectoriser.transform(batch))
            self.vectors = self.vectors.with_columns(pl.Series(embeddings).alias("embeddings"))
        except Exception as e:
            logging.error("Error creating Polars DataFrame")
            raise e

    def validate(self):
        """Validates the vector store by checking if the loaded vectoriser matches the one used to create the vectors
        and testing the search functionality.
        """
        # This method is a placeholder for future validation logic.
        # Currently, it does not perform any validation.

    def embed(self, query: VectorStoreEmbedInput) -> VectorStoreEmbedOutput:
        """Converts text into vector embeddings using the vectoriser and returns a VectorStoreEmbedOutput dataframe with columns 'id', 'text', and 'embedding'.

        Args:
            query (VectorStoreEmbedInput): The VectorStoreEmbedInput object containing the strings to be embedded and their ids.

        Returns:
            VectorStoreEmbedOutput: The output object containing the embeddings along with their corresponding ids and texts.
        """
        # Validate the input object
        if not isinstance(query, VectorStoreEmbedInput):
            raise ValueError("Query must be a VectorStoreEmbedInput object.")

        # Check if there is a user defined preprocess hook for the VectorStore embed method
        if "embed_preprocess" in self.hooks:
            modified_query = self.hooks["embed_preprocess"](query)
            try:
                query = VectorStoreEmbedInput.validate(modified_query)
            except Exception as e:
                raise ValueError(
                    f"Preprocessing hook returned an invalid VectorStoreEmbedInput object. Error: {e}"
                ) from e

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

        # Check if there is a user defined postprocess hook for the VectorStore embed method
        if "embed_postprocess" in self.hooks:
            modified_results_df = self.hooks["embed_postprocess"](results_df)
            try:
                results_df = VectorStoreEmbedOutput.validate(modified_results_df)
            except Exception as e:
                raise ValueError(
                    f"Postprocessing hook returned an invalid VectorStoreEmbedOutput object. Error: {e}"
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
            ValueError: Raised if invalid arguments are passed.
        """
        # Validate the input object
        if not isinstance(query, VectorStoreReverseSearchInput):
            raise ValueError("Query must be a VectorStoreReverseSearchInput object.")

        # Check if there is a user defined preprocess hook for the VectorStore reverse search method
        if "reverse_search_preprocess" in self.hooks:
            modified_query = self.hooks["reverse_search_preprocess"](query)
            try:
                query = VectorStoreReverseSearchInput.validate(modified_query)
            except Exception as e:
                raise ValueError(
                    f"Preprocessing hook returned an invalid VectorStoreReverseSearchInput object. Error: {e}"
                ) from e

        # polars conversion
        paired_query = pl.DataFrame({"id": query.id, "doc_id": query.doc_id})

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

        result_df = VectorStoreReverseSearchOutput.from_data(final_table.to_pandas())

        # Check if there is a user defined postprocess hook for the VectorStore reverse search method
        if "reverse_search_postprocess" in self.hooks:
            modified_result_df = self.hooks["reverse_search_postprocess"](result_df)
            try:
                result_df = VectorStoreReverseSearchOutput.validate(modified_result_df)
            except Exception as e:
                raise ValueError(
                    f"Preprocessing hook returned an invalid VectorStoreReverseSearchOutput object. Error: {e}"
                ) from e

        return result_df

    def search(self, query: VectorStoreSearchInput, n_results=10, batch_size=8) -> VectorStoreSearchOutput:
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
            ValueError: Raised if invalid arguments are passed.
        """
        # Validate the input object
        if not isinstance(query, VectorStoreSearchInput):
            raise ValueError("Query must be a VectorStoreSearchInput object.")

        # Check if there is a user defined preprocess hook for the VectorStore search method
        if "search_preprocess" in self.hooks:
            modified_query = self.hooks["search_preprocess"](query)
            try:
                query = VectorStoreSearchInput.validate(modified_query)
            except Exception as e:
                raise ValueError(
                    f"Preprocessing hook returned an invalid VectorStoreSearchInput object. Error: {e}"
                ) from e

        # Initialize an empty list to store results from each batch
        all_results = []

        # Process the queries in batches
        for i in tqdm(range(0, len(query), batch_size), desc="Processing query batches"):
            # Get the current batch of queries
            query_text_batch = query.query.to_list()[i : i + batch_size]
            query_ids_batch = query.id.to_list()[i : i + batch_size]

            # Convert the current batch of queries to vectors
            query_vectors = self.vectoriser.transform(query_text_batch)

            # Compute cosine similarity between the query batch and document vectors
            cosine = query_vectors @ self.vectors["embeddings"].to_numpy().T

            # Get the top n_results indices for each query in the batch
            idx = np.argpartition(cosine, -n_results, axis=1)[:, -n_results:]

            # Sort top n_results indices by their scores in descending order
            idx_sorted = np.zeros_like(idx)
            scores = np.zeros_like(idx, dtype=float)

            for j in range(idx.shape[0]):
                row_scores = cosine[j, idx[j]]
                sorted_indices = np.argsort(row_scores)[::-1]
                idx_sorted[j] = idx[j, sorted_indices]
                scores[j] = row_scores[sorted_indices]

            # Build a DataFrame for the current batch results
            result_df = pl.DataFrame(
                {
                    "query_id": np.repeat(query_ids_batch, n_results),
                    "query_text": np.repeat(query_text_batch, n_results),
                    "rank": np.tile(np.arange(n_results), len(query_text_batch)),
                    "score": scores.flatten(),
                }
            )

            # Get the vector store results for the current batch
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
            # Append the current batch results to the list
            all_results.append(merged_df)

        # Concatenate all batch results into a single DataFrame
        reordered_df = pl.concat(all_results).select(
            [
                "query_id",
                "query_text",
                "doc_id",
                "doc_text",
                "rank",
                "score",
                *self.meta_data.keys(),
            ]
        )

        # Now that polars has been used for processing convert back to pandas for user familiarity
        result_df = VectorStoreSearchOutput.from_data(reordered_df.to_pandas())

        # Check if there is a user defined postprocess hook for the VectorStore search method
        if "search_postprocess" in self.hooks:
            modified_result_df = self.hooks["search_postprocess"](result_df)
            try:
                result_df = VectorStoreSearchOutput.validate(modified_result_df)
            except Exception as e:
                raise ValueError(
                    f"Preprocessing hook returned an invalid VectorStoreSearchOutput object. Error: {e}"
                ) from e

        return result_df

    @classmethod
    def from_filespace(cls, folder_path, vectoriser, hooks: dict | None = None):
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
            ValueError: If required files or metadata keys are missing, or if the vectoriser class does not match.
        """
        # check that the metadata, vectoiser info and parquet exist

        # load the metadata file
        metadata_path = os.path.join(folder_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise ValueError(f"Metadata file not found in {folder_path}")
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        # check that the correct keys exist in metadata
        required_keys = [
            "vectoriser_class",
            "vector_shape",
            "num_vectors",
            "created_at",
            "meta_data",
        ]
        for key in required_keys:
            if key not in metadata:
                raise ValueError(f"Metadata file is missing required key: {key}")

        # get the column metadata and convert types to built-in types
        deserialized_column_meta_data = {
            key: getattr(__builtins__, value, value)  # Use built-in types or keep as-is
            for key, value in metadata["meta_data"].items()
        }

        # check that the vector shape and num vectors are correct
        # load the parquet file
        vectors_path = os.path.join(folder_path, "vectors.parquet")
        if not os.path.exists(vectors_path):
            raise ValueError(f"Vectors Parquet file not found in {folder_path}")

        df = pl.read_parquet(
            vectors_path,
            columns=["id", "text", "embeddings", "uuid", *deserialized_column_meta_data.keys()],
        )
        if df.is_empty():
            raise ValueError(f"Vectors Parquet file is empty in {folder_path}")
        # check parquet file has the correct columns
        required_columns = [
            "id",
            "text",
            "embeddings",
            "uuid",
            *deserialized_column_meta_data.keys(),
        ]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Vectors Parquet file is missing required column: {col}")

        # check that the vectoriser class matches the one provided
        if metadata["vectoriser_class"] != vectoriser.__class__.__name__:
            raise ValueError(
                f"Vectoriser class in metadata ({metadata['vectoriser_class']}) does not match provided vectoriser ({vectoriser.__class__.__name__})"
            )

        # create the VectorStore instance and add the new data to the fields
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
