# pylint: disable=C0301
"""Pydantic Classes to model request and response data for ClassifAI FastAPI RESTful API."""

import pandas as pd
from pydantic import BaseModel, Extra, Field


class ClassifaiEntry(BaseModel):
    """Atomic model for a single row of input data (i.e. a single query input) , includes 'id' and
    'description' which are expected as str type.
    """

    id: str = Field(examples=["1"])
    description: str = Field(
        description="User string describing inforation need/query",
        examples=["How to ice skate?"],
    )


class ClassifaiData(BaseModel):
    """Model for a list of many ClassifaiEntry pydantic models, i.e. several queries to be searched in the VectorStore."""

    entries: list[ClassifaiEntry] = Field(description="array of search queries to be searched in the VectorStore")


class ResultEntry(BaseModel):
    """Atomic model for a single row of vector store result data (i.e. a single vectorstore entry) , includes 'label', 'description', 'score' and 'rank' which are expected as str, str, float and int types respectively."""

    label: str
    description: str
    score: float
    rank: int

    class Config:  # pylint: disable=R0903
        """Sub-class to permit additional extra metadata (e.g., metadata columns from vectorstore construction)."""

        extra = Extra.allow


class ResultsList(BaseModel):
    """Model for a list of many ResultEntry pydantic models, representing a ranked list of vector store search results."""

    input_id: str
    response: list[ResultEntry]


class ResultsResponseBody(BaseModel):
    """Model for set of ranked lists, corresponding to multiple input queries and their own ranked ResultsLists."""

    data: list[ResultsList]


class RevClassifaiEntry(BaseModel):
    """Atomic model for a single row of reverse search data includes 'id' and 'code' which are expected as str type."""

    id: str = Field(examples=["1"])
    code: str = Field(
        examples=["0001"], description="VectorStore row entry 'ID' to be looked up, searched in the 'id' column."
    )


class RevClassifaiData(BaseModel):
    """Model for a list of many RevClassifaiEntry pydantic models, i.e. several vectorstore row entry codes to be looked up in the VectorStore."""

    entries: list[RevClassifaiEntry] = Field(description="array of VectorStore row entry IDs to be retrieved")


class RevResultEntry(BaseModel):
    """Atomic model for single reverse search result entry, includes 'label' and 'description' which are expected as str types."""

    label: str
    description: str

    class Config:
        extra = Extra.allow  # Allow extra keys (e.g., metadata columns)


class RevResultsList(BaseModel):
    """Model for a list of many RevResultEntry pydnatic models, representing a list of vector store entries found matching an input RevClassifaiEntry 'id'."""

    input_id: str
    response: list[RevResultEntry]


class RevResultsResponseBody(BaseModel):
    """Model for set of reverse ranked lists, corresponding to multiple input RevClassifaiEntry and their own RevResultsList."""

    data: list[RevResultsList]


class EmbeddingsList(BaseModel):
    """model for set of embeddings lists, for all row entries submmitted."""

    idx: str
    description: str
    embedding: list


class EmbeddingsResponseBody(BaseModel):
    """model for set of list of embeddings, for all row entries submmitted."""

    data: list[EmbeddingsList]


def convert_dataframe_to_reverse_search_pydantic_response(df: pd.DataFrame, meta_data: dict) -> RevResultsResponseBody:
    """Convert a Pandas DataFrame into a JSON object conforming to the RevResultsResponseBody Pydantic model.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing reverse search results.
        meta_data (dict): dictionary of metadata column names mapping to their types.

    Returns:
        RevResultsResponseBody: Pydantic model containing the structured response.
    """
    results_list = []

    # Group rows by `id`
    grouped = df.groupby("id")

    for input_id, group_df in grouped:
        # Convert group_df to a list of dictionaries
        rows_as_dicts = group_df.to_dict(orient="records")

        # Build the list of RevResultEntry objects for the current group
        response_entries = []
        for row in rows_as_dicts:
            # Extract metadata columns dynamically
            metadata_values = {meta: row[meta] for meta in meta_data if meta in row}

            # Create a RevResultEntry object
            response_entries.append(
                RevResultEntry(
                    label=row["doc_id"],
                    description=row["doc_text"],
                    **metadata_values,  # Add metadata dynamically
                )
            )

        # Create a RevResultsList object for the current `id`
        results_list.append(
            RevResultsList(
                input_id=input_id,
                response=response_entries,
            )
        )

    # Create the RevResultsResponseBody object
    response_body = RevResultsResponseBody(data=results_list)

    return response_body


def convert_dataframe_to_pydantic_response(df: pd.DataFrame, meta_data: dict) -> ResultsResponseBody:
    """Convert a Pandas DataFrame into a JSON object conforming to the ResultsResponseBody Pydantic model.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing query results.
        meta_data (dict): dictionary of metadata column names mapping to their types.

    Returns:
        ResultsResponseBody: Pydantic model containing the structured response.
    """
    # Group rows by `query_id`
    grouped = df.groupby("query_id")

    results_list = []
    for query_id, group_df in grouped:
        # Convert group_df to a list of dictionaries
        rows_as_dicts = group_df.to_dict(orient="records")

        # Build the list of ResultEntry objects for the current group
        response_entries = []
        for row in rows_as_dicts:
            # Extract metadata columns dynamically
            metadata_values = {meta: row[meta] for meta in meta_data}

            # Create a ResultEntry object
            response_entries.append(
                ResultEntry(
                    label=row["doc_id"],
                    description=row["doc_text"],
                    score=row["score"],  # Assuming `score` is a column in the DataFrame
                    rank=row["rank"],  # Assuming `rank` is a column in the DataFrame
                    **metadata_values,  # Add metadata dynamically
                )
            )

        # Create a ResultsList object for the current query_id
        results_list.append(
            ResultsList(
                input_id=query_id,  # type: ignore[arg-type]
                response=response_entries,
            )
        )

    # Create the ResultsResponseBody object
    response_body = ResultsResponseBody(data=results_list)

    return response_body
