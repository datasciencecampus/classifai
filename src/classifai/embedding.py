"""Class to create an embedding database and query it."""

import math
import os
import uuid

import chromadb
import dotenv
import pandas as pd
from chromadb.utils.embedding_functions import (
    GoogleGenerativeAiEmbeddingFunction,
    HuggingFaceEmbeddingFunction,
)

from .doc_utils import clean_text
from .utils import get_secret


class EmbeddingHandler:
    """Class to create a Chroma embedding database and query it."""

    def __init__(
        self,
        embedding_model_name: str = "models/text-embedding-004",  # "sentence-transformers/all-MiniLM-L6-v2"
        db_dir: str = "/tmp/db",
        k_matches: int = 3,
        task_type: str = "CLASSIFICATION",
        distance_metric: str = "l2",
        vdb_name: str = "default_collection",
    ):
        """Initialise EmbeddingHandler.

        Parameters
        ----------
        embedding_model_name : str, optional
            The model to use for embeddings, by default use the Hugging Face model
            "sentence-transformers/all-MiniLM-L6-v2".
        db_dir : str, optional
            The path to the where the database is stored, by default "data/soc-index/db".
        k_matches : int, optional
            The number of nearest matches to retrieve, by default 3.
        task_type: str, optional
            The task type if using Google embeddings, by default "CLASSIFICATION".
        distance_metric : str, optional
            The distance metric used for the embedding search, by default "l2". Must be one of: {"l2", "ip", "cosine"}.
        vdb_name : str, optional
            Name given to ChromaDB collection.
        """

        dotenv.load_dotenv(dotenv.find_dotenv())

        self.embedding_model_name = embedding_model_name
        self.task_type = task_type
        self.db_dir = db_dir
        self.k_matches = k_matches
        self.distance_metric = distance_metric
        self.api_key = get_secret("GOOGLE_API_KEY")
        self.vdb_name = vdb_name

        if self.embedding_model_name.startswith("models"):
            self.embedding_function = GoogleGenerativeAiEmbeddingFunction(
                api_key=self.api_key,
                model_name=self.embedding_model_name,
                task_type=task_type,
            )

        else:
            self.embedding_function = HuggingFaceEmbeddingFunction(
                api_key=os.getenv("HUGGINGFACE_API_KEY"),
                model_name=self.embedding_model_name,
            )

        self._create_vector_store()

    def _prime_vector_store(self):
        """Initialise Chroma VectorDB on known DB dir."""
        self.vector_store = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.vector_store.get_collection(
            name="classifai-collection",
            embedding_function=self.embedding_function,
        )

    def _create_vector_store(self):
        """Initialise Chroma VectorDB on known DB dir."""

        self.vector_store = chromadb.PersistentClient(path=self.db_dir)

        self.collection = self.vector_store.get_or_create_collection(
            name=self.vdb_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": self.distance_metric},
        )

    def embed_index(self, file_name: str = None):
        """Read a simple index file into chunks. Each row = one index entry.

        Parameters
        ----------
        file_name : str, optional
            The name of the index file to read. If provided, the
            file will be read and the entities will be embedded, by default None.
        """

        self.vector_store.delete_collection(name=self.vdb_name)
        self.collection = self.vector_store.create_collection(
            name=self.vdb_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": self.distance_metric},
        )

        docs = []
        label = []
        ids = []
        if file_name is not None:
            with open(file_name) as file:
                for line in file:
                    if line:
                        bits = line.split(":", 1)
                        docs.append(bits[1].replace("\n", "").strip())
                        label.append(dict(label=bits[0]))
                        ids.append(str(uuid.uuid3(uuid.NAMESPACE_URL, line)))

        self.collection.add(documents=docs, metadatas=label, ids=ids)

    def embed_master_index(self, file_name):
        """Embed ASHE_SOC2020_master_index.

        Parameters
        ----------
        file_name : str, optional
            The name of the index file to read. If provided, the
            file will be read and the entities will be embedded, by default None.
        """

        self.vector_store.delete_collection(name=self.vdb_name)
        self.collection = self.vector_store.create_collection(
            name=self.vdb_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": self.distance_metric},
        )

        docs = []
        labels = []
        ids = []
        if file_name is not None:
            with open(file_name, encoding="latin-1") as file:
                for line in file:
                    if line:
                        bits = line.split("  ", 1)
                        docs.append(clean_text(bits[0]).title())
                        labels.append(
                            dict(label=bits[1].replace("\n", "").strip())
                        )
                        ids.append(str(uuid.uuid3(uuid.NAMESPACE_URL, line)))

        for i in range(math.ceil(len(docs) / 500)):
            start_index = i * 500
            end_index = (i + 1) * 500

            self.collection.add(
                documents=docs[start_index:end_index],
                metadatas=labels[start_index:end_index],
                ids=ids[start_index:end_index],
            )

    def embed_index_csv(
        self,
        file: str | pd.DataFrame,
        label_column: str,
        embedding_columns: list[str],
    ):
        """Read a CSV file and embed it. Each row = one index entry.

        Parameters
        ----------
        file : str
            The name of the index file to read.
        label_column : str
            The name of the column containing the document label.
        embedding_columns : list
            List of columns in the CSV to concatenate and embed.
        """

        self.vector_store.delete_collection(name=self.vdb_name)
        self.collection = self.vector_store.get_or_create_collection(
            name=self.vdb_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": self.distance_metric},
        )

        if isinstance(file, str):
            df = pd.read_csv(file)
        else:
            df = file.copy()

        df["embed_column"] = df[embedding_columns].agg(" ".join, axis=1)

        docs = df["embed_column"].to_list()

        labels = []
        for i in df[label_column]:
            labels.append(dict(label=i))
        ids = [
            str(uuid.uuid3(uuid.NAMESPACE_URL, str(i)))
            for i in range(len(labels))
        ]

        for i in range(math.ceil(len(docs) / 100)):
            start_index = i * 100
            end_index = (i + 1) * 100

            self.collection.add(
                documents=docs[start_index:end_index],
                metadatas=labels[start_index:end_index],
                ids=ids[start_index:end_index],
            )

    def search_index(
        self,
        input_data: list[dict],
        embedded_fields=None,
    ) -> dict:
        """Return k document chunks with the highest relevance to the query.

        Parameters
        ----------
        input_data : list[dict]
            List of dictionaries with each dictionary representing an document to classify.
        id_field : str, optional
            The name of the unique id field for each entry, by default "uid".
        embedded_fields : list[str], optional
            The list of fields to embed and search against the database, by default None.

        Returns
        -------
        query_result : dict
            The raw result from the embedding search.
        """

        if embedded_fields is None:
            embedded_fields = []

        # create the query text list
        query_texts = self._create_query_texts(input_data, embedded_fields)

        # query the database
        query_result = self.collection.query(
            query_texts=query_texts,
            n_results=self.k_matches,
            include=["documents", "metadatas", "distances"],
        )

        return query_result

    @staticmethod
    def _create_query_texts(
        input_data: list[dict], embedded_fields: list[str]
    ) -> list[str]:
        """Create a list of strings to embed and query.

        Parameters
        ----------
        input_data : list[dict]
            List of dictionaries with each dictionary representing an document to classify.
        embedded_fields : list[str]
            The fields within the input data to embed.

        Returns
        -------
        query_texts: list[str]
            List of strings to embed and query against the database.
        """

        query_texts = []

        for entry in input_data:
            query_text = []
            for field, value in entry.items():
                if field in embedded_fields:
                    query_text.append(value)
            query_texts.append(" ".join(query_text))

        return query_texts

    @staticmethod
    def process_result_for_rag(result: dict):
        """
        Process the result of the embedding search for rag.

        Parameters
        ----------
        result (dict): The raw result of the embedding search.

        Returns
        -------
        list: The processed result as a list of strings.
        """
        soc_candidate_list_all_results = []
        for soc_candidates in result["metadatas"]:
            soc_candidate_list = []
            for soc_candidate in soc_candidates:
                soc_candidate_list.append(
                    f"{soc_candidate['label']}:{soc_candidate['description']}"
                )
            soc_candidate_list_all_results.append(
                "\n".join(soc_candidate_list)
            )
        return soc_candidate_list_all_results
