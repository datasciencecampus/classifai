from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

from ..indexers import VectorStore


class StartApiInput(BaseModel):
    vector_stores: list[VectorStore] = Field("The list of vectorstores to create endpoints for.")
    endpoint_names: list[str] = Field("The list of endpoint names for the vectorstores.")
    port: int = Field(8000, description="The port to run the API server on.", gt=1023, lt=49152)

    class Config:
        arbitrary_types_allowed = True

    @field_validator("vector_stores", mode="after")
    def validate_vector_stores(cls, value):
        if not isinstance(value, list):
            raise TypeError("vector_stores must be a list")
        for v in value:
            if not isinstance(v, VectorStore):
                raise TypeError(f"All elements of vector_stores must be instances of VectorStore, got {type(v)}")
        return value

    @field_validator("endpoint_names", mode="after")
    def validate_endpoint_names(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("endpoint_names must be unique")
        return value

    @model_validator(mode="after")
    def validate_length_matching(self) -> Self:
        if len(self.vector_stores) != len(self.endpoint_names):
            raise ValueError("The length of vector_stores must match the length of endpoint_names")
        return self
