"""A module for embedding text using a locally-running Ollama server."""

import numpy as np

from classifai._optional import check_deps
from classifai.errors import ExternalServiceError, VectorisationError

from .base import VectoriserBase


class OllamaVectoriser(VectoriserBase):
    """A wrapper class allowing a locally-running ollama server to generate text embeddings.

    Attributes:
        model_name (str): The name of the local model to use.
    """

    def __init__(self, model_name: str):
        """Initializes the OllamaVectoriser with the specified model name and device.

        Args:
            model_name (str): The name of the local model to use.

        Notes:
            requires an ollama server to be running locally (`ollama serve`)
        """
        check_deps(["ollama"], extra="ollama")

        self.model_name = model_name

    def transform(self, texts):
        """Transforms input text(s) into embeddings using the Huggingface model.

        Args:
            texts (str or list of str): The input text(s) to embed. Can be a single string or a list of strings.

        Returns:
            numpy.ndarray: A 2D array of embeddings, where each row corresponds to an input text.

        Raises:
            ExternalServiceError: If the Ollama service fails to generate embeddings.
            VectorisationError: If embedding extraction from the Ollama response fails.
        """
        import ollama  # type: ignore

        # If a single string is passed as arg to texts, convert to list
        if isinstance(texts, str):
            texts = [texts]

        try:
            response = ollama.embed(model=self.model_name, input=texts)
        except Exception as e:
            raise ExternalServiceError(
                "Failed to generate embeddings using Ollama.",
                context={"vectoriser": "ollama", "model": self.model_name},
            ) from e

        try:
            return np.array(response.embeddings)
        except Exception as e:
            raise VectorisationError(
                "Failed to extract embeddings from Ollama response.",
                context={"vectoriser": "ollama", "model": self.model_name},
            ) from e
