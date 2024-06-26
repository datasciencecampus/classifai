"""Class to create an embedding database and query it."""

import os
import uuid

import chromadb
import dotenv
from chromadb.utils.embedding_functions import (
    GoogleGenerativeAiEmbeddingFunction,
    HuggingFaceEmbeddingFunction,
)


class EmbeddingHandler:
    """Class to create a Chroma embedding database and query it."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",  # models/text-embedding-004
        db_dir: str = "data/soc-index/db",
        k_matches: int = 3,
        task_type: str = "CLASSIFICATION",
    ):
        """Initialise EmbeddingHandler.

        Parameters
        ----------
        embedding_model_name : str, optional
            The model to use for embeddings, by default use the Hugging Face model
            "sentence-transformers/all-MiniLM-L6-v2".
        db_dir: str, optional
            The path to the where the database is stored, by default "data/soc-index/db".
        k_matches : int, optional
            The number of nearest matches to retrieve, by default 3.
        task_type : str, optional
            The task type if using Google embeddings, by default "CLASSIFICATION".
        """

        dotenv.load_dotenv(dotenv.find_dotenv())

        if embedding_model_name.startswith("models"):
            self.embedding_function = GoogleGenerativeAiEmbeddingFunction(
                api_key=os.getenv("GOOGLE_API_KEY"),
                model_name=embedding_model_name,
                task_type=task_type,
            )

        else:
            self.embedding_function = HuggingFaceEmbeddingFunction(
                api_key=os.getenv("HUGGINGFACE_API_KEY"),
                model_name=embedding_model_name,
            )

        self.db_dir = db_dir
        self.k_matches = k_matches
        self._create_vector_store()

    def _create_vector_store(self):
        """Initialise Chroma VectorDB on known DB dir."""

        self.vector_store = chromadb.PersistentClient(path=self.db_dir)

        self.collection = self.vector_store.get_or_create_collection(
            name="my_collection", embedding_function=self.embedding_function
        )

    def embed_index(self, file_name: str = None):
        """Read a simple index file into chunks. Each row = one index entry.

        Parameters
        ----------
        file_name : str, optional
            The name of the index file to read. If provided, the
            file will be read and the entities will be embedded, by default None.
        """

        self.vector_store.delete_collection(name="my_collection")
        self.collection = self.vector_store.create_collection(
            name="my_collection", embedding_function=self.embedding_function
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
        embedded_fields : list, optional
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
        input_data: list[dict], embedded_fields: list
    ) -> list:
        """Create a list of strings to embed and query.

        Parameters
        ----------
        input_data : list[dict]
            List of dictionaries with each dictionary representing an document to classify.
        embedded_fields : list
            The fields within the input data to embed.

        Returns
        -------
        query_texts: list
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
