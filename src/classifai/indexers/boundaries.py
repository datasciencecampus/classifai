from pathlib import Path
from typing import Literal

import pandera.pandas as pa
from pydantic import BaseModel, DirectoryPath, Field, FilePath, field_validator, model_validator
from typing_extensions import Self

from ..vectorisers.base import VectoriserBase

# Pydantic and Pandera models for input and output validation
#######


class VectorStoreInput(BaseModel):
    file_name: FilePath = Field(
        description="Path to the data file to be indexed.",
    )
    data_type: Literal["csv"] = Field(description="The type of data file. Currently only 'csv' is supported.")
    vectoriser: VectoriserBase = Field(..., description="An instance of a class inheriting from VectoriserBase.")
    batch_size: int = Field(
        gt=0,
        description="The batch size for processing data. Must be greater than 0.",
    )
    meta_data: list[str] | None = Field(
        None,
        description="List of metadata fields to include in the index, aligning with column names of the input file.",
    )
    output_dir: Path | None = Field(
        None, description="Directory to save the vector store, if not set will attempt to reåuse file_name path."
    )
    overwrite: bool = Field(False, description="Whether to overwrite existing vector store in the output directory.")

    class Config:
        arbitrary_types_allowed = True

    @field_validator("meta_data", mode="after")
    def validate_metadata(cls, value):
        if value is None:
            return {}
        return value


class FromFileSpaceInput(BaseModel):
    folder_path: DirectoryPath = Field(
        description="The folder path containing the metadata and vector parquet files to be loaded."
    )  # DirectoryPath ensures the path is a valid directory on the filesystem compared to Path used in the VectorStoreInput above which doesn't need to exist
    vectoriser: VectoriserBase = Field(description="An instance of a class inheriting from VectoriserBase.")

    class Config:
        arbitrary_types_allowed = True


class SearchInput(BaseModel):
    query: str | list[str] = Field(..., description="The text query or list of queries to search for.")
    ids: list[str | int] | None = Field(
        None,
        description="List of query IDs. Must be unique and match the length of the query list.",
    )
    n_results: int = Field(
        gt=0,
        description="Number of top results to return for each query. Must be greater than 0.",
    )
    batch_size: int = Field(
        gt=0,
        description="The batch size for processing queries. Must be greater than 0.",
    )

    # custom validator to check that query ids are unique, converts query to list if string, and assigns default ids if none provided, and checks list ids length matches list query length
    @model_validator(mode="after")
    def validate_ids_and_query(self) -> Self:
        if isinstance(self.query, str):
            self.query = [self.query]

        if self.ids is None:
            self.ids = list(range(len(self.query)))
            self.ids = [str(i) for i in self.ids]

        elif len(set(self.ids)) != len(self.ids):
            raise ValueError("'ids' must contain unique values.")

        if len(self.ids) != len(self.query):
            raise ValueError("'ids' must have the same length as 'query'.")

        return self


class SearchOutputSchema(pa.DataFrameModel):
    query_id: str  # Ensure query_id is a string
    query_text: str  # Ensure query_text is a string
    doc_id: str  # Ensure doc_id is a string
    doc_text: str  # Ensure doc_text is a string
    rank: int = pa.Field(ge=0)  # Ensure rank is a non-negative integer
    score: float = pa.Field(ge=0.0)  # Ensure score is between 0 and 1

    # Add metadata columns dynamically if needed
    class Config:
        strict = False  # Allow additional columns (e.g., metadata)


class ReverseSearchInput(BaseModel):
    query: str | list[str] = Field(description="The text query or list of queries to search for.")
    ids: list[str | int] | None = Field(
        None,
        description="List of query IDs. Must be unique and match the length of the query list.",
    )
    n_results: int = Field(
        100,
        gt=0,
        description="Number of top results to return for each query. Must be greater than 0.",
    )

    # custom validator to check that query ids are unique, converts query to list if string, and assigns default ids if none provided, and checks list ids length matches list query length
    @model_validator(mode="after")
    def validate_ids_and_query(self) -> Self:
        if isinstance(self.query, str):
            self.query = [self.query]

        if self.ids is None:
            self.ids = list(range(len(self.query)))
            self.ids = [str(i) for i in self.ids]

        elif len(set(self.ids)) != len(self.ids):
            raise ValueError("'ids' must contain unique values.")

        if len(self.ids) != len(self.query):
            raise ValueError("'ids' must have the same length as 'query'.")

        return self


class ReverseSearchOutputSchema(pa.DataFrameModel):
    query_id: str  # Ensure query_id is a string
    doc_id: str  # Ensure doc_id is a string
    doc_text: str  # Ensure doc_text is a string

    # Add metadata columns dynamically if needed
    class Config:
        strict = False  # Allow additional columns (e.g., metadata)
