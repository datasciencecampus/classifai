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
- pandas: For handling data in tabular format and saving it as a Parquet file.
- tqdm: For displaying progress bars during batch processing.
- numpy: for vector cosine similarity calculations
- A custom file iterator (`iter_csv`) for reading input files in batches.

Usage:
This module is intended to be used as with the Vectoriers mdodule and the
the servers module from ClassifAI package, to created scalable, modular, searchable
vector databases from your own text data.
"""

import json
import logging
import os
import time

import numpy as np
import polars as pl
import tqdm

from .helpers.file_iters import iter_csv

# Configure logging for your application
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


class VectorStore:
    """A class to model and create 'VectorStore' objects for building and searching vector databases from CSV text files.

    Attributes:
        file_name (str): the original CSV file associated with the vector store
        data_type (str): the data type of the original file (curently only csv)
        vectoriser (object): A Vectoriser object from the corresponding ClassifAI Pacakge module
        batch_size (int): the batch size to pass to the vectoriser when embedding
        meta_data (list[str]): list of metadata stored in the vector DB
        vectors: (np.array): a numpy array of vectors for the vector DB
        vector_shape (int): the dimension of the vectors
        num_vectors (int): how many vectors are in the vector store
        vectoriser_class (str): the type of vectoriser used to create embeddings
    """

    def __init__(self, file_name, data_type, vectoriser, batch_size=8, meta_data=None):
        """Initializes the VectorStore object by processing the input CSV file and generating
        vector embeddings.

        Args:
            file_name (str): The name of the input CSV file.
            data_type (str): The type of input data (currently supports only "csv").
            vectoriser (object): The vectoriser object used to transform text into
                                vector embeddings.
            batch_size (int, optional): The batch size for processing the input file and batching to
            vectoriser. Defaults to 8.
            meta_data (list, optional): List of metadata column names to extract from the input file.
                                Defaults to None.

        Raises:
            ValueError: If the data type is not supported or if the folder name conflicts with an existing folder.
        """
        self.file_name = file_name
        self.data_type = data_type
        self.vectoriser = vectoriser
        self.batch_size = batch_size
        self.meta_data = meta_data if meta_data is not None else []
        self.vectors = None
        self.vector_shape = None
        self.num_vectors = None
        self.vectoriser_class = vectoriser.__class__.__name__

        if self.data_type not in ["csv"]:
            raise ValueError(
                "Data type must be one of ['csv'] (more file types added in later update!)"
            )

        os.makedirs("classifai_vector_stores", exist_ok=True)

        # Normalize the file name to ensure it doesn't include relative paths or extensions
        normalized_file_name = os.path.basename(os.path.splitext(self.file_name)[0])
        # Check if the folder exists in the specified subdirectory
        subdir_path = os.path.join("classifai_vector_stores", normalized_file_name)
        if os.path.isdir(subdir_path):
            raise ValueError(
                f"The name '{subdir_path}' is already used as a folder in the subdirectory."
            )
        os.makedirs(subdir_path, exist_ok=True)

        self._create_vector_store_index()

        logging.info("Gathering metadata and saving vector store / metadata...")

        self.vector_shape = self.vectors["embeddings"].to_numpy().shape[1]
        self.num_vectors = len(self.vectors)

        ## save everything to the folder etc: metadata, parquet and vectoriser
        self.vectors.write_parquet(os.path.join(subdir_path, "vectors.parquet"))
        self._save_metadata(os.path.join(subdir_path, "metadata.json"))

        logging.info("Vector Store created - files saved to %s", subdir_path)

    def _save_metadata(self, path):
        """Saves metadata about the vector store to a JSON file.

        Args:
            path (str): The file path where the metadata JSON file will be saved.

        Raises:
            Exception: If an error occurs while saving the metadata file.
        """
        try:
            metadata = {
                "vectoriser_class": self.vectoriser_class,
                "vector_shape": self.vector_shape,
                "num_vectors": self.num_vectors,
                "created_at": time.time(),
                "meta_data": self.meta_data,
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
        except Exception:
            logging.error("Something went wrong trying to save the metadata file")
            raise

    def _create_vector_store_index(self):
        """Processes the input file in batches, generates vector embeddings, and creates the vector store.

        Raises:
            Exception: If an error occurs during file processing or vector generation.
        """
        # set up the file indexer
        try:
            # if self.data_type == "csv":
            file_loader = iter_csv

        except Exception:
            logging.error("Error setting up file loader")
            raise

        # set up the captured data structure that will store the data and generated embeddings
        captured_data = {x: [] for x in ["id", "text", *self.meta_data]}
        captured_embeddings = []

        logging.info(
            "Processing file: %s in batches of size %d...\n",
            self.file_name,
            self.batch_size,
        )

        # Process the file in batches by iterating over the appropriate file loader
        for batch_no, batch in enumerate(
            tqdm.tqdm(
                file_loader(
                    file_name=self.file_name,
                    meta_data=self.meta_data,
                    batch_size=self.batch_size,
                ),
                desc="Processing batches",
            )
        ):
            # try to process each batch provided by file iter
            try:

                # get batch text and id and meta-data columns, store in corresponding captured_data
                for k in captured_data.keys():
                    captured_data[k].extend([entry[k] for entry in batch])

                # generate embeddings for the text in the batch and store them
                batch_vectors = self.vectoriser.transform(
                    [entry["text"] for entry in batch]
                )
                captured_embeddings.extend(batch_vectors)

            # if any error occurs while processing a batch, log the error and continue to next batch
            except (KeyError, ValueError, TypeError) as e:
                logging.error("Error processing batch %d: %s", batch_no, e)
                continue

        logging.info(
            "\nFinished creating vectors, attempting to create vector store object..."
        )

        # now that all batches are processed and text vectorised, save it
        try:
            self.vectors = pl.DataFrame({x: captured_data[x] for x in captured_data})
            self.vectors = self.vectors.with_columns(
                pl.Series("embeddings", captured_embeddings)
            )

        except Exception:
            logging.error("Error creating Polars DataFrame or saving to Parquet file")
            raise

    def validate(self):
        """Validates the vector store by checking if the loaded vectoriser matches the one used to create the vectors
        and testing the search functionality.
        """
        # This method is a placeholder for future validation logic.
        # Currently, it does not perform any validation.

    def embed(self, text):
        """Converts text into vector embeddings using the vectoriser.

        Args:
            text (str or list): The text or list of text to be converted into vector embeddings.

        Returns:
            np.ndarray: The vector embeddings generated by the vectoriser.
        """
        return self.vectoriser.transform(text)

    def search(self, query, ids=None, n_results=10):
        """Searches the vector store using a text query or list of queries and returns ranked results.

        Args:
            query (str or list): The text query or list of queries to search for.
            ids (list, optional): List of query IDs. Defaults to None.
            n_results (int, optional): The number of top results to return for each query. Defaults to 10.

        Returns:
            pl.DataFrame: A DataFrame containing the search results with columns for query ID, query text,
                          document ID, document text, rank, score, and metadata.
        """
        # if the query is a string, convert it to a list
        if isinstance(query, str):
            query = [query]

        # convert the query/queries to vectors using the embedder
        query_vectors = self.vectoriser.transform(query)
        document_vectors = self.vectors["embeddings"].to_numpy()
        if not ids:
            ids = list(range(len(query)))

        # Compute cosine similarity between quries and each document
        cosine = query_vectors @ document_vectors.T

        # Get the top n_results indices for each query
        idx = np.argpartition(cosine, -n_results, axis=1)[:, -n_results:]

        # Sort top k indices by their scores in descending order
        idx_sorted = np.zeros_like(idx)
        scores = np.zeros_like(idx, dtype=float)

        for i in range(idx.shape[0]):
            row_scores = cosine[i, idx[i]]
            sorted_indices = np.argsort(row_scores)[::-1]
            idx_sorted[i] = idx[i, sorted_indices]
            scores[i] = row_scores[sorted_indices]

        # build polars dataframe for reults where query and ids and broadcasted, and rank is tiled
        result_df = pl.DataFrame(
            {
                "query_id": np.repeat(ids, n_results),
                "query_text": np.repeat(query, n_results),
                "rank": np.tile(np.arange(n_results), len(ids)),
                "score": scores.flatten(),
            }
        )

        # get the vector store results ids, texts and metadata based on sorted idx and merge with result_df
        ranked_docs = self.vectors[idx_sorted.flatten().tolist()].select(
            ["id", "text", *self.meta_data]
        )
        merged_df = result_df.hstack(ranked_docs).rename(
            {"id": "doc_id", "text": "doc_text"}
        )

        # reorder the df into presentable format
        reordered_df = merged_df.select(
            [
                "query_id",
                "query_text",
                "doc_id",
                "doc_text",
                "rank",
                "score",
                *self.meta_data,
            ]
        )

        return reordered_df

    @classmethod
    def from_filespace(cls, folder_path, vectoriser):
        """Creates a `VectorStore` instance from stored metadata and Parquet files.

        Args:
            folder_path (str): The folder path containing the metadata and Parquet files.
            vectoriser (object): The vectoriser object used to transform text into vector embeddings.

        Returns:
            VectorStore: An instance of the `VectorStore` class.

        Raises:
            ValueError: If required files or metadata keys are missing, or if the vectoriser class does not match.
        """
        # check that the metadataq, vectoiser info and parquet exist
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

        # load the parquet file
        vectors_path = os.path.join(folder_path, "vectors.parquet")
        if not os.path.exists(vectors_path):
            raise ValueError(f"Vectors Parquet file not found in {folder_path}")

        df = pl.read_parquet(vectors_path)
        if df.is_empty():
            raise ValueError(f"Vectors Parquet file is empty in {folder_path}")
        # check parquet file has the correct columns
        required_columns = ["id", "text", "embeddings", *metadata["meta_data"]]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(
                    f"Vectors Parquet file is missing required column: {col}"
                )

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
        vector_store.meta_data = metadata["meta_data"]
        vector_store.vectors = df
        vector_store.vector_shape = metadata["vector_shape"]
        vector_store.num_vectors = metadata["num_vectors"]
        vector_store.vectoriser_class = metadata["vectoriser_class"]

        return vector_store
