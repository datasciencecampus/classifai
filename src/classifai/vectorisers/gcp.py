"""A module for embedding text using Google Cloud Platform's GenAI API."""

from __future__ import annotations

import logging

import numpy as np

from classifai._optional import check_deps
from classifai.exceptions import ConfigurationError, ExternalServiceError, VectorisationError

from .base import VectoriserBase

logging.getLogger("google.auth").setLevel(logging.WARNING)
logging.getLogger("google.cloud").setLevel(logging.WARNING)
logging.getLogger("google.api_core").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class GcpVectoriser(VectoriserBase):
    """A class for embedding text using Google Cloud Platform's GenAI API.

    This class provides functionality to embed text using Google's GenAI API.
    It supports two authentication methods for setting up the client:

    1. Using `project_id` and `location`: This method requires specifying the
        Google Cloud project ID and the location of the GenAI API. It is
        suitable for users who have a Google Cloud project and want to
        authenticate using project-based credentials. It will require local
        authentication through the Google Cloud SDK.

    2. Using `api_key`: This method requires providing an API key for
        authentication. It is suitable for users who want to authenticate using
        an API key without specifying a project ID and location. This approach
        does not require local authentication through the Google Cloud SDK.

    Attributes:
        model_name (str): The name of the embedding model to use.
        vectoriser (genai.Client): The GenAI client instance for embedding
            text.
        task_type (str): The embedding task type (e.g., "CLASSIFICATION").
        transform_kwargs (dict): Additional keyword arguments for the embed_content method.
    """

    def __init__(  # noqa: PLR0913
        self,
        project_id=None,
        api_key=None,
        location="europe-west2",
        model_name="text-embedding-004",
        task_type="CLASSIFICATION",
        client_kwargs=None,
        transform_kwargs=None,
    ):
        """Initialises the GcpVectoriser with the specified project ID, location, and model name.

        Args:
            project_id (str): [optional] The Google Cloud project ID. Defaults
                to None.
            api_key (str): [optional] The API key for authenticating with the
                GenAI API. Defaults to None.
            location (str): [optional] The location of the GenAI API. Defaults
                to "europe-west2".
            model_name (str): [optional] The name of the embedding model.
                Defaults to "text-embedding-004".
            task_type (str): [optional] The embedding task. Defaults to
                "CLASSIFICATION". See
                https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types
                for other options.
            client_kwargs (dict): [optional] Additional keyword arguments to pass to
                the GenAI client. To invoke via the Agent Platform API
                (formerly VertexAI) API rather than the GenAI API, pass the
                additional kwarg `vertexai=True`. Defaults to None.
            transform_kwargs (dict): [optional] Additional keyword arguments to pass to
                the embed_content method. Defaults to None.

        Raises:
            `ConfigurationError`: If the authentication arguments are invalid,
                or if the GenAI client fails to initialise.
        """
        check_deps(["google-genai"], extra="gcp")
        from google import genai  # type: ignore

        self.model_name = model_name
        self.task_type = task_type
        self.transform_kwargs = transform_kwargs or {}

        # Build client kwargs
        _client_kwargs = client_kwargs.copy() if client_kwargs else {}

        if project_id and not api_key:
            _client_kwargs.setdefault("project", project_id)
            _client_kwargs.setdefault("location", location)
        elif api_key and not project_id:
            _client_kwargs.setdefault("api_key", api_key)
        else:
            raise ConfigurationError(
                "Provide either 'project_id' and 'location' together, or 'api_key' alone for GCP Vectoriser.",
                context={"vectoriser": "gcp"},
            )

        try:
            self.vectoriser = genai.Client(
                **_client_kwargs,
            )
        except Exception as e:
            raise ConfigurationError(
                "Failed to initialise GCP GenAI client.",
                context={"vectoriser": "gcp", "cause": str(e), "cause_type": type(e).__name__},
            ) from e

    def transform(self, texts: str | list[str], **kwargs) -> np.ndarray:
        """Transforms input text(s) into embeddings using the GenAI API.

        Args:
            texts (str | list[str]): The input text(s) to embed. Can be a
                single string or a list of strings.
            **kwargs: Additional parameters to override or supplement stored
                transform_kwargs for the `embed_content` method.

        Returns:
            numpy.ndarray: A 2D array of embeddings, where each row
                corresponds to an input text.

        Raises:
            `ExternalServiceError`: If the GenAI API request fails.
            `VectorisationError`: If the response format from the GenAI API is
                unexpected.
        """
        from google import genai  # type: ignore

        if isinstance(texts, str):
            texts = [texts]

        # Merge stored transform_kwargs with runtime overrides
        config_kwargs = {**self.transform_kwargs, **kwargs}
        config_kwargs.setdefault("task_type", self.task_type)

        config = genai.types.EmbedContentConfig(**config_kwargs)

        try:
            embeddings = self.vectoriser.models.embed_content(model=self.model_name, contents=texts, config=config)
        except Exception as e:
            raise ExternalServiceError(
                "GCP embedding request failed.",
                context={
                    "vectoriser": "gcp",
                    "model": self.model_name,
                    "n_texts": len(texts),
                    "cause": str(e),
                    "cause_type": type(e).__name__,
                },
            ) from e

        try:
            result = np.array([res.values for res in embeddings.embeddings])
        except Exception as e:
            raise VectorisationError(
                "Unexpected embedding response format from GCP.",
                context={
                    "vectoriser": "gcp",
                    "model": self.model_name,
                    "cause": str(e),
                    "cause_type": type(e).__name__,
                },
            ) from e

        return result
