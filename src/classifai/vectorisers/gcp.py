"""A module for embedding text using Google Cloud Platform's GenAI API."""

from __future__ import annotations

import logging

import numpy as np

from classifai._optional import check_deps

from .base import VectoriserBase
from .boundaries import GcpVectoriserInput, TransformInput, TransformOutput

logging.getLogger("google.auth").setLevel(logging.WARNING)
logging.getLogger("google.cloud").setLevel(logging.WARNING)
logging.getLogger("google.api_core").setLevel(logging.WARNING)


class GcpVectoriser(VectoriserBase):
    """A class for embedding text using Google Cloud Platform's GenAI API.

    Attributes:
        model_name (str): The name of the embedding model to use.
        vectoriser (genai.Client): The GenAI client instance for embedding text.
        hooks (dict): A dictionary of user-defined hooks for preprocessing and postprocessing.
    """

    def __init__(
        self,
        project_id,
        location="europe-west2",
        model_name="text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT",
        hooks=None,
    ):
        """Initializes the GcpVectoriser with the specified project ID, location, and model name.

        Args:
            project_id (str): The Google Cloud project ID.
            location (str, optional): The location of the GenAI API. Defaults to 'europe-west2'.
            model_name (str, optional): The name of the embedding model. Defaults to "text-embedding-004".
            task_type (str, optional): The embedding task. Defaults to "CLASSIFICATION".
                                       See https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types
                                       for other options.
            hooks (dict, optional): A dictionary of user-defined hooks for preprocessing and postprocessing. Defaults to None.

        Raises:
            RuntimeError: If the GenAI client fails to initialize.
        """
        check_deps(["google-genai"], extra="gcp")
        from google import genai  # type: ignore

        # Run the Pydantic validator first which will raise errors if the inputs are invalid
        validated_inputs = GcpVectoriserInput(
            project_id=project_id,
            location=location,
            model_name=model_name,
            task_type=task_type,
        )

        self.model_name = validated_inputs.model_name
        self.model_config = genai.types.EmbedContentConfig(task_type=validated_inputs.task_type)

        try:
            self.vectoriser = genai.Client(
                vertexai=True,
                project=validated_inputs.project_id,
                location=validated_inputs.location,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GCP Vectoriser through ganai.Client API: {e}") from e

        self.hooks = {} if hooks is None else hooks

    def transform(self, texts):
        """Transforms input text(s) into embeddings using the GenAI API.

        Args:
            texts (str or list of str): The input text(s) to embed. Can be a single string or a list of strings.

        Returns:
            numpy.ndarray: A 2D array of embeddings, where each row corresponds to an input text.

        Raises:
            TypeError: If the input is not a string or a list of strings.
        """
        # Run the Pydantic validator first which will raise errors if the inputs are invalid
        validated_input = TransformInput(texts=texts)
        if self.hooks["transform_preprocess"]:
            # pass the validated_input to the user defined function
            hook_output = self.hooks["transform_postprocess"](validated_input)
            # revalidate the output of the user defined function
            validated_input = TransformInput(hook_output)

        # The Vertex AI call to  embed content
        embeddings = self.vectoriser.models.embed_content(
            model=self.model_name, contents=validated_input.texts, config=self.model_config
        )

        # Extract embeddings from the response object
        # embeddings = [embedding[0] for embedding in embeddings]
        result = np.array([res.values for res in embeddings.embeddings])

        # Validate the output before returning which will raise errors if the outputs are invalid
        validated_output = TransformOutput(embeddings=result)
        if self.hooks["transform_postprocess"]:
            # pass the validated_output to the user defined function
            hook_output = self.hooks["transform_postprocess"](validated_output)
            # revalidate the output of the user defined function
            validated_output = TransformOutput(hook_output)
        return validated_output.embeddings
