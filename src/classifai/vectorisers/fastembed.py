"""A module that provides a wrapper for FastEmbed models to generate text embeddings."""

from typing import TYPE_CHECKING

import numpy as np

from classifai._optional import check_deps
from classifai.exceptions import ExternalServiceError, VectorisationError

from .base import VectoriserBase

if TYPE_CHECKING:
    from fastembed import TextEmbedding  # type: ignore[import-not-found]


EXPECTED_EMBEDDING_DIMS = 2


class FastEmbedVectoriser(VectoriserBase):
    """A lightweight wrapper class for generating embeddings with FastEmbed.

    The `FastEmbedVectoriser` uses FastEmbed's ONNX backend to generate
    embeddings from HuggingFace-compatible sentence embedding models without
    requiring `torch` or `transformers` as runtime dependencies.

    Attributes:
        model_name (str): The name or path of the FastEmbed-compatible model.
        model (fastembed.TextEmbedding): The FastEmbed model instance.
    """

    def __init__(self, model_name: str):
        """Initialises the FastEmbedVectoriser with the specified model name.

        Args:
            model_name (str): The name or local path of the embedding model.

        Raises:
            `ExternalServiceError`: If the FastEmbed model cannot be loaded.
        """
        self.model_name = model_name
        self.model = _load_fastembed_model(model_name)

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
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

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

        if embeddings.ndim != EXPECTED_EMBEDDING_DIMS:
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


def _load_fastembed_model(model_name: str) -> "TextEmbedding":
    """Load a FastEmbed embedding model."""
    check_deps(["fastembed"], extra="fastembed")
    from fastembed import TextEmbedding  # type: ignore

    try:
        return TextEmbedding(model_name=model_name)
    except Exception as e:
        raise ExternalServiceError(
            "Failed to load FastEmbed model.",
            context={
                "vectoriser": "fastembed",
                "model": model_name,
                "cause": str(e),
                "cause_type": type(e).__name__,
            },
        ) from e
