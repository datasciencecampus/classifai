"""A module for embedding text using a locally-running Ollama server."""

import numpy as np
from pydantic import ValidationError

from classifai._optional import check_deps

from .base import VectoriserBase
from .boundaries import TransformInput, TransformOutput


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
            TypeError: If the input is not a string or a list of strings.
        """
        import ollama  # type: ignore

        try:
            # Validate and normalize input using Pydantic
            validated_input = TransformInput(texts=texts)
            texts = validated_input.texts
        except ValidationError as e:
            raise ValueError(f"Invalid input: {e}") from e

        response = ollama.embed(model=self.model_name, input=texts)

        try:
            validated_output = TransformOutput.from_ndarray(np.array(response.embeddings))
        except ValidationError as e:
            raise ValueError(f"Invalid output: {e}") from e

        return validated_output
