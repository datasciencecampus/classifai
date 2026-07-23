"""A module that provides a wrapper for Huggingface Transformers models to generate text embeddings."""

import numpy as np
from numpy import ndarray as np_ndarray

from classifai._optional import check_deps
from classifai.exceptions import ConfigurationError, VectorisationError

from .base import VectoriserBase


class HuggingFaceVectoriser_CPU_ONNX(VectoriserBase):
    """A general wrapper class for Huggingface Transformers models to generate text embeddings.

    The `HuggingFaceVectoriser` accepts most encoder-based models from the
    Huggingface Transformers library, and provides a simple interface to
    generate embeddings from text data. Additional configuration options, such
    as `trust_remote_code` or a HuggingFaceAPI `token` can be passed via the
    `tokenizer_kwargs` and `model_kwargs` parameters.

    Attributes:
        model_name (str): The name of the Huggingface model to use.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the
            specified model.
        model (transformers.PreTrainedModel): The Huggingface model instance.
    """

    def __init__(
        self,
        model_name,
        model_revision="main",
        tokenizer_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
    ):
        """Initialises the HuggingfaceVectoriser with the specified model name and device.

        Args:
            model_name (str): The name of the Huggingface model to use.
            model_revision (str): [optional] The specific model revision to
                use. Defaults to "main".
            tokenizer_kwargs (dict): [optional] Additional keyword arguments to
                pass to the tokenizer. Defaults to None.
            model_kwargs (dict): [optional] Additional keyword arguments to
                pass to the model. Defaults to None.

        Raises:
            `ExternalServiceError`: If the model or tokenizer cannot be loaded.
            `ConfigurationError`: If the model cannot be initialised on the
                specified device.
        """
        # check_deps(["transformers", "torch"], extra="huggingface")
        check_deps(["fastembed"], extra="huggingface_simple")
        # import torch  # type: ignore
        # from transformers import AutoModel, AutoTokenizer  # type: ignore
        from fastembed import TextEmbedding

        self.model_name = model_name

        try:
            self.model = TextEmbedding(model_name=self.model_name)
        except Exception as e:
            raise ConfigurationError(
                "Failed to initialise model.",
                context={
                    "vectoriser": "huggingface-cpu-onnx",
                    "model": model_name,
                    "cause": str(e),
                    "cause_type": type(e).__name__,
                },
            ) from e

    def transform(self, texts: str | list[str]) -> np_ndarray:
        """Transforms input text(s) into embeddings using the Huggingface model.

        Args:
            texts (str | list[str]): The input text(s) to embed. Can be a
                single string or a list of strings.

        Returns:
            numpy.ndarray: A 2D array of embeddings, where each row
                corresponds to an input text.

        Raises:
            `VectorisationError`: If tokenization, model inference, or
                embedding extraction fails.
        """
        # import torch  # type: ignore

        # If a single string is passed as arg to texts, convert to list
        if isinstance(texts, str):
            texts = [texts]

        # Tokenization / tensor move can fail (e.g., device issues, weird tokenizer config)
        try:
            embeddings = self.model.embed(texts)
        except Exception as e:
            raise VectorisationError(
                "Embedding failed.",
                context={
                    "vectoriser": "huggingface-cpu-onnx",
                    "model": self.model_name,
                    "n_texts": len(texts),
                    "cause": str(e),
                    "cause_type": type(e).__name__,
                },
            ) from e

        return np.array(list(embeddings))
