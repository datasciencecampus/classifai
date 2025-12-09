import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

####
# Input and output models for transform method that is common between the different vectorisers
####


class TransformInput(BaseModel):
    texts: str | list[str] = Field(..., description="The input text(s) to embed.")

    @field_validator("texts", mode="after")
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
    embeddings: NDArray | list[list[float]] = Field(..., description="A 2D array of embeddings.")

    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, v):
        if isinstance(v, np.ndarray):
            if v.ndim != EXPECTED_EMBEDDING_DIMENSION:
                raise ValueError("Embeddings must be a 2D NumPy array.")
            return v
        elif isinstance(v, list):
            arr = np.array(v)
            if arr.ndim != EXPECTED_EMBEDDING_DIMENSION:
                raise ValueError("Embeddings must be a 2D array.")
            return arr
        else:
            raise ValueError("Embeddings must be a NumPy array or a 2D list.")

    class Config:
        arbitrary_types_allowed = True


####
# Class initialization input models for each vectoriser
####


class GcpVectoriserInput(BaseModel):
    project_id: str = Field(description="The Google Cloud project ID.")
    location: str = Field(
        description="The location of the GenAI API.",
    )
    model_name: str = Field(
        description="The name of the embedding model.",
    )
    task_type: str = Field(
        description="The embedding model task type.",
    )


class HuggingFaceVectoriserInput(BaseModel):
    model_name: str = Field(
        description="The name of the Hugging Face embedding model.",
    )
    device: str | None = Field(
        default=None,
        description="The device to use for computation ('cpu', 'cuda', or None).",
    )
    model_revision: str = Field(
        description="The specific model revision to use.",
    )


class OllamaVectoriserInput(BaseModel):
    model_name: str = Field(
        description="The name of the local Ollama model to use.",
    )
