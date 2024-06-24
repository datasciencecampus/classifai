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
    ):
        """Initialise EmbeddingHandler.

        Parameters
        ----------
        embedding_model_name : str, optional
            The model to use for embeddings, by default use the Hugging Face model
            "sentence-transformers/all-MiniLM-L6-v2".
        k_matches : int, optional
            The number of nearest matches to retrieve, by default 3.
        """

        dotenv.load_dotenv(dotenv.find_dotenv())

        if embedding_model_name.startswith("models"):
            self.embedding_function = GoogleGenerativeAiEmbeddingFunction(
                api_key=os.getenv("GOOGLE_API_KEY"),
                model_name=embedding_model_name,
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
        id_field="uid",
        embedded_fields=None,
        process_output=True,
    ) -> dict:
        """Return k document chunks with the highest relevance to the query.

        Parameters
        ----------
        input_data : list[dict]
            List of dictionaries with each dictionary representing an document to classify.
        id_field : str, optional
            The name of the unique id field for each entry, by default "uid".
        embedded_fields : _type_, optional
            _description_, by default None.
        process_output : bool, optional
            Whether to process the output of the embedding search or leave it in its raw
            format, by default True.

        Returns
        -------
        dict
            The unprocessed or processed result from the embedding search.
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

        # process query results
        if process_output is True:
            return self._process_output(query_result, input_data, id_field)
        else:
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

    @staticmethod
    def _process_output(
        query_result: dict, input_data: list[dict], id_field: str
    ) -> dict:
        """Process the results of the query into a dictionary format.

        Parameters
        ----------
        query_result : dict
            The results of the query to process.
        input_data : list[dict]
            List of dictionaries with each dictionary representing an document to classify.
        id_field : str
            The name of the unique ID field.

        Returns
        -------
        output_dict: dict
            The results of the embedding search in JSON format.
        """

        output_dict = dict()
        for label_list, description_list, distance_list, input_dict in zip(
            query_result["metadatas"],
            query_result["documents"],
            query_result["distances"],
            input_data,
        ):
            for label, description, distance in zip(
                label_list, description_list, distance_list
            ):
                label.update({"description": description})
                label.update({"distance": distance})
            output_dict[input_dict[id_field]] = label_list
        return output_dict
