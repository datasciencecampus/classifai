"""Class to embed records."""

import os
import uuid

from langchain.docstore.document import Document
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    VertexAIEmbeddings,
)
from langchain_community.vectorstores import Chroma
from pyprojroot import here


class EmbeddingHandler:
    """Class to embed data."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",  # all-mpnet-base-v2
        db_dir: str = os.path.join(here(), "data/sic-index/db"),
        k_matches: int = 20,
    ):
        """Initialise EmbeddingHandler.

        Args:
            vector_store (Optional[Chroma], optional): _description_.
            Defaults to None.
            embedding_function (Optional[HuggingFaceEmbeddings], optional):
            _description_. Defaults to None.
        """

        if embedding_model_name.startswith("textembedding-"):
            self.embeddings = VertexAIEmbeddings(
                model_name=embedding_model_name
            )

        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name
            )

        self.db_dir = db_dir
        self.vector_store = self._create_vector_store()
        self.k_matches = k_matches

    def _create_vector_store(self) -> Chroma:
        """Initialise Chroma VectorDB on known DB dir in data.

        Returns
        -------
            Chroma: the langchain vectorstore object for Chroma
        """
        return Chroma(
            embedding_function=self.embeddings, persist_directory=self.db_dir
        )

    def embed_index(
        self,
        from_empty: bool = True,
        file_name: str = None,
    ):
        """Read a simple index file into chunks. Each row = one index entry.

        Args:
            from_empty (bool): whether to drop current vector db content
        """
        if from_empty:
            self.vector_store._client.delete_collection("langchain")
            self.vector_store = self._create_vector_store()

        docs = []
        ids = []
        if file_name is not None:
            with open(file_name) as file:
                for line in file:
                    if line:
                        bits = line.split(",", 1)
                        docs.append(
                            Document(
                                page_content=bits[1],
                                metadata={"code": bits[0]},
                            )
                        )
                        ids.append(str(uuid.uuid3(uuid.NAMESPACE_URL, line)))

        self.vector_store.add_documents(docs, ids=ids)

    def search_index(
        self, query: str, return_dicts: bool = True
    ) -> list[dict]:
        """Return k document chunks with the highest relevance to the query.

        Args:
            query (str): Question for which most relevant index entries will
            be returned
            return_dicts: if True, data returned as dictionary, key = rank

        Returns
        -------
            List[dict]: List of top k aindex entries by relevance
        """
        top_matches = self.vector_store.similarity_search_with_score(
            query=query, k=self.k_matches
        )

        if return_dicts:
            return [
                {"distance": float(doc[1])}
                | {"title": doc[0].page_content}
                | doc[0].metadata
                for doc in top_matches
            ]
        return top_matches
