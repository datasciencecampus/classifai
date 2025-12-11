"""A module for embedding text using a locally-running Ollama server."""

from classifai._optional import check_deps

from .base import VectoriserBase


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

        self.model_name = model_name

        self.hooks = {} if hooks is None else hooks

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

        if type(texts) is str:
            texts = [texts]

        if type(texts) is not list:
            raise TypeError("Input texts must be a string or a list of strings.")

        # Check if there is a user defined preprocess hook for the OllamaVectoriser transform method
        if "transform_preprocess" in self.hooks:
            # pass the args to the preprocessing function as a dictionary
            hook_output = self.hooks["transform_preprocess"]({"texts": texts})

            # Unpack the dictionary back into the argument variables
            texts = hook_output.get("texts", texts)

        response = ollama.embed(model=self.model_name, input=texts)

        # Check if there is a user defined postprocess hook for the OllamaVectoriser transform method
        if "transform_postprocess" in self.hooks:
            # pass args to the postprocessing function as a dictionary
            hook_output = self.hooks["transform_postprocess"]({"embeddings": response.embeddings})

            # Unpack the dictionary back into the argument variables
            embeddings = hook_output.get("embeddings", response)

        return embeddings
