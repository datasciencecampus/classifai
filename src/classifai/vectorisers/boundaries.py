import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

##
# The following is the Pydantic Base Model for all pre-set vectorisers input and output data which must be adhered to:
##


class TransformInput(BaseModel):
    texts: str | list[str] = Field(..., description="The input text(s) to embed.")

    @field_validator("texts")
    @classmethod
    def validate_and_normalize_texts(cls, v):
        if isinstance(v, str):
            return [v]  # Normalize single string to a list
        if isinstance(v, list):
            if not all(isinstance(item, str) and item.strip() for item in v):
                raise ValueError("All items in the list must be non-empty strings.")
            return v
        raise ValueError("Input must be a string or a list of strings.")


EXPECTED_EMBEDDING_DIMENSION = 2


class TransformOutput(BaseModel):
    embeddings: NDArray[np.float32] = Field(..., description="A 2D NumPy array of embeddings.")
    n_texts: int = Field(..., ge=1, description="The number of input texts.")
    embedding_dim: int = Field(..., ge=1, description="The dimensionality of the embeddings.")

    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError("Embeddings must be a NumPy array.")
        if v.ndim != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError("Embeddings must be a 2D NumPy array.")
        return v

    @classmethod
    def from_ndarray(cls, arr: NDArray[np.float32]) -> "TransformOutput":
        """Create a TransformOutput instance from a NumPy array."""
        if arr.ndim != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError("Expected a 2D NumPy array.")
        n_texts, embedding_dim = arr.shape
        return cls(embeddings=arr, n_texts=n_texts, embedding_dim=embedding_dim)
