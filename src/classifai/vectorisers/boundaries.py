import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

####
# Input and output models for transform method that is common between the different vectorisers
####


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

    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError("Embeddings must be a NumPy array.")
        if v.ndim != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError("Embeddings must be a 2D NumPy array.")
        return v


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
    task_type: str = Field(
        model_revision="Essentially, the tag version of model as uploaded to Hugging Face.",
    )


class OllamaVectoriserInput(BaseModel):
    model_name: str = Field(
        description="The name of the local Ollama model to use.",
    )
