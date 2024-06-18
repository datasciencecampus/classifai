"""Class to embed records."""

import os

from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    VertexAIEmbeddings,
)
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
