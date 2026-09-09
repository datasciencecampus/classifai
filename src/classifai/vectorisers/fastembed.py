"""A module that provides a wrapper for FastEmbed models to generate text embeddings."""

import numpy as np

from classifai._optional import check_deps
from classifai.exceptions import ExternalServiceError, VectorisationError

from .base import VectoriserBase


class FastEmbedVectoriser(VectoriserBase):
    """A lightweight wrapper class for generating embeddings with FastEmbed.

    The `FastEmbedVectoriser` uses FastEmbed's ONNX backend to generate
    embeddings from FastEmbed-compatible sentence embedding models without
    requiring `torch` or `transformers` as runtime dependencies. The
    `model_name` must be a name recognised by FastEmbed. To see all
    supported models, you can run:
    `FastEmbedVectoriser.list_supported_models()`

    Attributes:
        model_name (str): The official FastEmbed name of the embedding model.
        model (fastembed.TextEmbedding): The FastEmbed model instance.
        specific_model_path (str | None): The path of the local FastEmbed model.
    """

    def __init__(
        self,
        model_name: str,
        specific_model_path: str | None = None,
        model_kwargs: dict | None = None,
    ):
        """Initialises the FastEmbedVectoriser with the specified model name.

        Args:
            model_name (str): The official name of the embedding model for FastEmbed
                (e.g., "sentence-transformers/all-MiniLM-L6-v2").
            specific_model_path (str): [optional] The local directory path
                containing a pre-downloaded ONNX model. Used for offline
                deployments. Defaults to None.
            model_kwargs (dict): [optional] Additional keyword arguments to
                pass to the model (e.g., `cache_dir`). Defaults to None.

        Raises:
            `ExternalServiceError`: If the FastEmbed model cannot be loaded.
        """
        check_deps(["fastembed"], extra="fastembed")
        from fastembed import TextEmbedding  # type: ignore

        self.model_name = model_name
        self.specific_model_path = specific_model_path
        model_kwargs = dict(model_kwargs or {})

        if self.specific_model_path is not None:
            model_kwargs["specific_model_path"] = str(self.specific_model_path)

        try:
            self.model = TextEmbedding(model_name=self.model_name, **model_kwargs)
        except Exception as e:
            raise ExternalServiceError(
                "Failed to load FastEmbed model.",
                context={
                    "vectoriser": "fastembed",
                    "model": self.model_name,
                    "cause": str(e),
                    "cause_type": type(e).__name__,
                },
            ) from e

    def transform(self, texts: str | list[str]) -> np.ndarray:
        """Transforms input text(s) into embeddings using FastEmbed.

        Args:
            texts (str | list[str]): The input text(s) to embed. Can be a
                single string or a list of strings.

        Returns:
            numpy.ndarray: A 2D array of embeddings, where each row
                corresponds to an input text.

        Raises:
            `VectorisationError`: If FastEmbed fails to generate or parse
                embeddings.
        """
        # If a single string is passed as arg to texts, convert to list
        if isinstance(texts, str):
            texts = [texts]

        try:
            raw_embeddings = list(self.model.embed(texts))
        except Exception as e:
            raise VectorisationError(
                "Failed to generate embeddings using FastEmbed.",
                context={
                    "vectoriser": "fastembed",
                    "model": self.model_name,
                    "n_texts": len(texts),
                    "cause": str(e),
                    "cause_type": type(e).__name__,
                },
            ) from e

        try:
            embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        except Exception as e:
            raise VectorisationError(
                "Failed to convert FastEmbed embeddings to a numpy array.",
                context={
                    "vectoriser": "fastembed",
                    "model": self.model_name,
                    "n_texts": len(texts),
                    "cause": str(e),
                    "cause_type": type(e).__name__,
                },
            ) from e

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.ndim != 2:  # noqa: PLR2004
            raise VectorisationError(
                "FastEmbed returned embeddings with an unexpected shape.",
                context={
                    "vectoriser": "fastembed",
                    "model": self.model_name,
                    "n_texts": len(texts),
                    "shape": list(embeddings.shape),
                },
            )

        return embeddings

    @staticmethod
    def list_supported_models() -> list[dict[str, any]]:
        """Wrapper to list the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.
        """
        check_deps(["fastembed"], extra="fastembed")
        from fastembed import TextEmbedding  # type: ignore

        return TextEmbedding.list_supported_models()
