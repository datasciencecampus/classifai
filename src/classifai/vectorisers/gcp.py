"""A module for embedding text using Google Cloud Platform's GenAI API."""

from __future__ import annotations

import logging

import numpy as np

from classifai._optional import check_deps

from .base import VectoriserBase

logging.getLogger("google.auth").setLevel(logging.WARNING)
logging.getLogger("google.cloud").setLevel(logging.WARNING)
logging.getLogger("google.api_core").setLevel(logging.WARNING)


class GcpVectoriser(VectoriserBase):
    """A class for embedding text using Google Cloud Platform's GenAI API.

    Attributes:
        model_name (str): The name of the embedding model to use.
        vectoriser (genai.Client): The GenAI client instance for embedding text.
    """

    def __init__(
        self,
        project_id,
        api_key=None,
        location="europe-west2",
        model_name="text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT",
    ):
        """Initializes the GcpVectoriser with the specified project ID, location, and model name.

        Args:
            project_id (str): The Google Cloud project ID.
            api_key (str, optional): The API key for authenticating with the GenAI API. Defaults to None.
            location (str, optional): The location of the GenAI API. Defaults to 'europe-west2'.
            model_name (str, optional): The name of the embedding model. Defaults to "text-embedding-004".
            task_type (str, optional): The embedding task. Defaults to "CLASSIFICATION".
                                       See https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types
                                       for other options.

        Raises:
            RuntimeError: If the GenAI client fails to initialize.
        """
        check_deps(["google-genai"], extra="gcp")
        from google import genai  # type: ignore

        self.model_name = model_name
        self.model_config = genai.types.EmbedContentConfig(task_type=task_type)

        try:
            self.vectoriser = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                api_key=api_key,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GCP Vectoriser through ganai.Client API: {e}") from e

    def transform(self, texts):
        """Transforms input text(s) into embeddings using the GenAI API.

        Args:
            texts (str or list of str): The input text(s) to embed. Can be a single string or a list of strings.

        Returns:
            numpy.ndarray: A 2D array of embeddings, where each row corresponds to an input text.

        Raises:
            TypeError: If the input is not a string or a list of strings.
        """
        # If a single string is passed as arg to texts, convert to list
        if isinstance(texts, str):
            texts = [texts]

        # The Vertex AI call to  embed content
        embeddings = self.vectoriser.models.embed_content(
            model=self.model_name, contents=texts, config=self.model_config
        )

        # Extract embeddings from the response object
        # embeddings = [embedding[0] for embedding in embeddings]
        result = np.array([res.values for res in embeddings.embeddings])

        return result
