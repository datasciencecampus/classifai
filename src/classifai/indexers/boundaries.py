import pandera as pa
from pydantic import BaseModel, Field, root_validator

# Pydantic and Pandera models for input and output validation
#######


class initInput:
    # TODO: define init input parameters
    pass


class SearchInput(BaseModel):
    query: str | list[str] = Field(..., description="The text query or list of queries to search for.")
    ids: list[str | int] = Field(
        None,
        description="List of query IDs. Must be unique and match the length of the query list.",
    )
    n_results: int = Field(
        10,
        gt=0,
        description="Number of top results to return for each query. Must be greater than 0.",
    )
    batch_size: int = Field(
        8,
        gt=0,
        description="The batch size for processing queries. Must be greater than 0.",
    )

    @root_validator
    def validate_ids_and_query(cls, values):
        query = values.get("query")
        ids = values.get("ids")

        # Ensure `query` is a list
        if isinstance(query, str):
            query = [query]
            values["query"] = query

        # Validate `ids` if provided
        if ids:
            if len(ids) != len(query):
                raise ValueError("'ids' must have the same length as 'query'.")
            if len(set(ids)) != len(ids):
                raise ValueError("'ids' must contain unique values.")

        return values


class SearchOutputSchema(pa.SchemaModel):
    query_id: pa.typing.Series[str]  # Ensure query_id is a string
    query_text: pa.typing.Series[str]  # Ensure query_text is a string
    doc_id: pa.typing.Series[str]  # Ensure doc_id is a string
    doc_text: pa.typing.Series[str]  # Ensure doc_text is a string
    rank: pa.typing.Series[int] = pa.Field(ge=0)  # Ensure rank is a non-negative integer
    score: pa.typing.Series[float] = pa.Field(ge=0.0, le=1.0)  # Ensure score is between 0 and 1

    # Add metadata columns dynamically if needed
    class Config:
        strict = False  # Allow additional columns (e.g., metadata)


class ReverseSearchInput(BaseModel):
    query: str | list[str] = Field(..., description="The text query or list of queries to search for.")
    ids: list[str | int] = Field(
        None,
        description="List of query IDs. Must be unique and match the length of the query list.",
    )
    n_results: int = Field(
        100,
        gt=0,
        description="Number of top results to return for each query. Must be greater than 0.",
    )

    @root_validator
    def validate_ids_and_query(cls, values):
        query = values.get("query")
        ids = values.get("ids")

        # Ensure `query` is a list
        if isinstance(query, str):
            query = [query]
            values["query"] = query

        # Validate `ids` if provided
        if ids:
            if len(ids) != len(query):
                raise ValueError("'ids' must have the same length as 'query'.")
            if len(set(ids)) != len(ids):
                raise ValueError("'ids' must contain unique values.")

        return values


class ReverseSearchOutputSchema(pa.SchemaModel):
    query_id: pa.typing.Series[str]  # Ensure query_id is a string
    doc_id: pa.typing.Series[str]  # Ensure doc_id is a string
    doc_text: pa.typing.Series[str]  # Ensure doc_text is a string
    rank: pa.typing.Series[int] = pa.Field(ge=0)  # Ensure rank is a non-negative integer
    score: pa.typing.Series[float] = pa.Field(ge=0.0, le=1.0)  # Ensure score is between 0 and 1

    # Add metadata columns dynamically if needed
    class Config:
        strict = False  # Allow additional columns (e.g., metadata)
