"""A module for embedding text using a locally-running Ollama server."""

import numpy as np

from classifai._optional import check_deps

from .base import VectoriserBase
from .boundaries import OllamaVectoriserInput, TransformInput, TransformOutput


class OllamaVectoriser(VectoriserBase):
    """A wrapper class allowing a locally-running ollama server to generate text embeddings.

    Attributes:
        model_name (str): The name of the local model to use.
        hooks (dict): A dictionary of user-defined hooks for preprocessing and postprocessing.
    """

    def __init__(self, model_name: str, hooks=None):
        """Initializes the OllamaVectoriser with the specified model name and device.

        Args:
            model_name (str): The name of the local model to use.
            hooks (dict, optional): A dictionary of user-defined hooks for preprocessing and postprocessing. Defaults to None.

        Notes:
            requires an ollama server to be running locally (`ollama serve`)
        """
        check_deps(["ollama"], extra="ollama")

        # Run the Pydantic validator first which will raise errors if the inputs are invalid
        validated_inputs = OllamaVectoriserInput(model_name=model_name)

        self.model_name = validated_inputs.model_name

        self.hooks = hooks

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

        validated_input = TransformInput(texts=texts)
        if self.hooks["transform_preprocess"]:
            # pass the validated_outputs to the user defined function
            hook_output = self.subroutes["transform_postprocess"](validated_input)
            # revalidate the output of the user defined function
            validated_input = TransformInput(hook_output)

        response = ollama.embed(model=self.model_name, input=validated_input.texts)

        # Validate the output before returning which will raise errors if the outputs are invalid
        validated_output = TransformOutput(embeddings=np.array(response.embeddings))
        if self.hooks["transform_postprocess"]:
            # pass the validated_outputs to the user defined function
            hook_output = self.hooks["transform_postprocess"](validated_output)
            # revalidate the output of the user defined function
            validated_output = TransformOutput(hook_output)

        return validated_output.embeddings
