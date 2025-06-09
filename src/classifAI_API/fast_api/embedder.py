"""Define embedding handler."""

import sys

sys.path.append("src/")

import os
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from google import genai
from google.cloud import storage

from .google_configurations.config import get_secret

def embed_as_array(documents: list[str], api_key: str, model_name: str="text-embedding-004", model_task: str="CLASSIFICATION") -> np.ndarray:
    """
    Embeds a list of text documents into vector representations using Google's GenAI text embedding model.

    Parameters
    ----------
    documents : list[str]
        A list of strings, each representing a document to be embedded.
    api_key : str
        The Google API key for authentication with the GenAI service.
    model_name : str
        The name of the GenAI embeddings model to use.
        (default - text-embedding-004)
    model_task : str
        The task to assign the GenAI embeddings model.
        (default - CLASSIFICATION)

    Returns
    -------
    np.ndarray
        A numpy ndarray of shape (n_documents, embedding_dimension) where each row
        corresponds to the embedding vector for a document in the input list.

    Notes
    -----
    This function requires the 'google.generativeai' package and numpy to be installed.
    The embeddings are generated using the model_name model with the
    model_task task type.

    Examples
    --------
    >>> documents = ["This is the first document.", "This is the second document."]
    >>> api_key = "your_google_api_key" # pragma: allowlist secret
    >>> embeddings = embed_as_array(documents, api_key, mdel_name, model_task)
    >>> embeddings.shape
    (2, 768)  # Assuming the embedding dimension is 768
    """
    if len(documents) == 0:
        return np.empty((0, 0))

    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=model_name,
        contents=documents,
        config=genai.types.EmbedContentConfig(task_type=model_task),
    )
    queries_array = np.array([res.values for res in result.embeddings])
    return queries_array


class ParquetNumpyVectorStore:
    """Wrapper for parquet vector dataset."""

    def __init__(
        self,
        knowledgebase: pl.DataFrame,
        embedding_column: str = "embeddings",
        sample_size: int = 20,
        tolerance: float = 1e-4,
    ):
        """
        Initialize the vector store with a Polars DataFrame.

        Runs a quick check that the vectors are normalised.

        Args:
            knowledgebase: Polars DataFrame containing embedding vectors
            embedding_column: Name of the column containing embedding vectors
            sample_size: Number of vectors to sample for norm checking
            tolerance: Tolerance for norm deviation from 1.0

        Raises
        ------
            ValueError: If vectors in the sample don't have unit norm (within tolerance)
        """
        self.knowledgebase = knowledgebase

        # Validate that embedding column exists
        if embedding_column not in knowledgebase.columns:
            raise ValueError(
                f"Column '{embedding_column}' not found in the provided DataFrame"
            )

        # Take a sample of the vectors to check for unit norm
        sample_size = min(sample_size, len(knowledgebase))
        if sample_size > 0:
            sample = knowledgebase.sample(sample_size, seed=3141)
            embeddings_sample = sample[embedding_column].to_numpy()

            # Calculate norms of the sampled vectors
            norms = np.array(
                [np.linalg.norm(vec) for vec in embeddings_sample]
            )

            # Check if all norms are approximately 1.0
            is_unit_norm = np.all(np.abs(norms - 1.0) < tolerance)

            if not is_unit_norm:
                non_unit_indices = np.where(np.abs(norms - 1.0) >= tolerance)[
                    0
                ]
                error_msg = (
                    f"Detected {len(non_unit_indices)} vectors in the sample that don't have unit norm. "
                    f"Example norms: {norms[non_unit_indices[:5]]}"
                )
                raise ValueError(error_msg)
        if sample_size <= 0:
            raise ValueError("The knowledgebase must have at least one row.")

    @classmethod
    def from_gcs_bucket(
        cls,
        client: storage.Client,
        bucket_name: str | None = None,
        local_dir: str = "/tmp/",
        prefix: str = "db/",
        parquet_filename: str = "classifai.parquet",
        force_refresh: bool = False,
    ):
        """Inititialise VectorStore from GCS bucket.

        Parameters
        ----------
        client : storage.Client
            GCS client object
        bucket_name : str | None
            Name of GCS bucket. If None, fetched from secrets
            Default: None
        local_dir : str
            Location of local/instance temporary directory
            Default: '/tmp/'
        prefix : str
            GCS bucket folder
            Default: 'db/'
        parquet_filename: str
            name of the .parquet knowledgebase
            Default "classifai.parquet"
        force_refresh : bool
            Whether to delete and re-fetch if database exists
            Default: False

        Examples
        --------
        >>> client = storage.Client()
        >>> pull_vdb_to_local(client, force_refresh=False)
        """

        local_path = Path(local_dir)
        target_dir = local_path / prefix
        path_to_knowledgebase = str(
            Path(local_dir) / prefix / parquet_filename
        )

        # If local data folder structure is in place, either remove it
        # if force_refresh is set, or try to use an existing parquet file if
        # it is not set. If none is found, continue on to pull one from
        # the Google Bucket
        if target_dir.exists(): 
            if force_refresh:
                shutil.rmtree(target_dir)
            else:
                try:
                    knowledgebase = pl.read_parquet(path_to_knowledgebase).rename(
                        {"documents": "description"}
                    )
                    new_vector_store = cls(knowledgebase=knowledgebase)
                    return new_vector_store
                except FileNotFoundError:
                    pass 
            
        target_dir.mkdir(parents=True, exist_ok=True)
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            relative_path = Path(blob.name).relative_to(prefix)
            local_file = target_dir / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_file))

        knowledgebase = pl.read_parquet(path_to_knowledgebase).rename(
            {"documents": "description"}
        )

        new_vector_store = cls(knowledgebase=knowledgebase)
        return new_vector_store

    @classmethod
    def from_local_storage(
        cls,
        parquet_filepath_local: str = ""
    ):
        """Inititialise VectorStore from local parquet file.

        Parameters
        ----------
        parquet_filepath_local: str
            path to the .parquet knowledgebase
        """

        if not Path(parquet_filepath_local).exists():
            raise FileNotFoundError(f"could not find knowledgebase at\n{parquet_filepath_local}")

        try:
            knowledgebase = pl.read_parquet(parquet_filepath_local).rename(
                {"documents": "description"}
            )
            new_vector_store = cls(knowledgebase=knowledgebase)
            return new_vector_store
        except pl.exceptions.ComputeError as e:
            print(f"The file {parquet_filepath_local} does not appear to be a valid .parquet file", e) 
            raise e
        except Exception as e:
            raise e

    def query(
        self,
        query_embeddings: np.ndarray,
        ids: list[str],
        k: int = 3,
        scores_as_distance: bool = True,
    ):
        """
        Perform vectorized nearest neighbors search.

        This method finds the k-nearest neighbors in the knowledge base for each query embedding
        using Cosine similarity (or squared l2 distance if scores_as_distance=True).

        Parameters
        ----------
        query_embeddings : np.ndarray
            Query embeddings, either a 1D array of shape (embedding_dim,) for a single query,
            or a 2D array of shape (n_queries, embedding_dim) for multiple queries.
        ids : list[str]
            List of identifiers corresponding to the stored embeddings in the knowledge base.
        k : int, default=3
            Number of nearest neighbors to retrieve for each query.
        scores_as_distance : bool, default=True
            If True, converts similarity scores to distances (1 - similarity).
            If False, returns raw similarity scores (higher is better).

        Returns
        -------
        dict
            A dictionary containing:
            - 'idx' : np.ndarray of shape (n_queries, k)
                Indices of the k nearest neighbors for each query.
            - 'scores' : np.ndarray of shape (n_queries, k)
                Similarity scores (or distances if scores_as_distance=True) for each neighbor.
            - 'ids' : list
                The original list of identifiers.

        Note
        ----
        The Cosine calculation assumes that embeddings in the knowledgebase are normalized.
        """
        # Extract the embeddings matrix from the dataframe
        document_embeddings = self.knowledgebase["embeddings"].to_numpy()

        # If query_embeddings is 1D, convert to 2D for consistent processing
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # Calculate the norms of the query_embeddings
        # (should be 1 but just being defensive)
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)

        # Compute cosine - result will be (n_queries, n_document_embeddings_rows)
        cosine = (query_embeddings @ document_embeddings.T) / query_norms

        # Get top k indices for each query_embeddings in one operation
        idx = np.argpartition(cosine, -k, axis=1)[:, -k:]

        # Sort top k indices by their scores in descending order
        # For each row, get the scores at the top k indices, then sort them
        idx_sorted = np.zeros_like(idx)
        scores = np.zeros_like(idx, dtype=float)

        # Vectorized approach for sorting the top k indices
        for i in range(idx.shape[0]):
            row_scores = cosine[i, idx[i]]
            sorted_indices = np.argsort(row_scores)[::-1]
            idx_sorted[i] = idx[i, sorted_indices]
            scores[i] = row_scores[sorted_indices]

        if scores_as_distance:
            # This gives the squared l2 norm.
            # For unit vectors where ||a|| = ||b|| = 1:
            # L2²(a,b) = 1 + 1 - 2(a·b) = 2(1 - a·b) = 2(1 - cos(θ))
            scores = 2 * (1 - scores)

        return dict(idx=idx_sorted, scores=scores, ids=ids)

    def create_json_array_response(
        self, query: dict, include_bridge: bool = False
    ):
        """
        Process new search results into expected format.

        This method takes query results and formats them into a standardized JSON array response.

        Parameters
        ----------
        query : dict
            Dictionary containing search results with the following keys:
            - scores : ndarray
                Array of similarity scores
            - idx : ndarray
                Array of indices of matching knowledge base entries
            - ids : list
                List of input identifiers
        include_bridge : bool, default=False
            Whether to include bridge information in the response

        Returns
        -------
        list
            List of dictionaries, each containing:
            - input_id : object
                Identifier for the input
            - response : list
                List of matching results with label, description, distance, rank,
                and bridge (if include_bridge=True)

        Notes
        -----
        The distance values are rounded to 3 decimal places, and rank is calculated
        based on these distances in ascending order.
        """
        scores = query["scores"]
        idx = query["idx"]
        ids = query["ids"]

        if include_bridge:
            vars_to_include = ["label", "description", "bridge"]
        else:
            vars_to_include = ["label", "description"]

        data_list = []
        for i, row in enumerate(idx):
            distance_row = scores[i, :]
            responses = (
                self.knowledgebase[row, vars_to_include]
                .with_columns(distance=np.round(distance_row, 3))
                .with_columns(
                    pl.col("distance")
                    .rank(descending=False, method="ordinal")
                    .alias("rank")
                )
            )
            if not include_bridge:
                responses = responses.with_columns(bridge=pl.lit(""))
            responses = responses.to_dicts()
            data_list.append({"input_id": ids[i], "response": responses})

        return data_list


    def create_embeddings_json_array_response(
        self, 
        embeddings: list[np.ndarray],
        descriptions: list[str],
        ids: list[str],
        description_labels: list[str]
        ):
        """
        Process new search results into expected format.

        This method takes query results and formats them into a standardized JSON array response.

        Parameters
        ----------
        embeddings : list[np.ndarray]
            list of embeddings arrays for each search term 
        descriptions : list[str]
            list of the search terms
        ids : list
            list of search term identifiers
        description_labels : list
            list of embedding descriptions

        Returns
        -------
        dictionary keyed by ids, each entry containing a dictionary with the corresponding 
        description and embedding values (as a list)
        """

        embeddings = [[float(e) for e in embedding] for embedding in embeddings]
        output_obj =  [{"id": str(id),
                        "description": description,
                        "embedding": embedding} \
                    for id, description, embedding in \
                    zip(ids, descriptions, embeddings)]
        return {'category_labels': description_labels,
                'data': output_obj}