# pylint: disable=C0301
"""Pydantic Classes to model request and response data for FastAPI RESTful API."""

import pandas as pd
from pydantic import BaseModel, Extra, Field


class ClassifaiEntry(BaseModel):
    """Model for a single row of data (SOC or SIC row etc), includes 'id' and
    'description' which are expected as str type.
    """

    id: str = Field(examples=["1"])
    description: str = Field(
        description="User string describing occupation or industry",
        examples=["A butcher's shop"],
    )


class ClassifaiData(BaseModel):
    """Pydantic object which contains list of many SOC/SIC Classifai Entry pydantic models."""

    entries: list[ClassifaiEntry] = Field(description="array of SOC/SIC Entries to be classified")


class ResultEntry(BaseModel):
    """Model for single vdb entry."""

    label: str
    description: str
    score: float
    rank: int

    class Config:  # pylint: disable=R0903
        """Sub-class to permit additional extra keys (e.g., metadata columns)."""

        extra = Extra.allow


class ResultsList(BaseModel):
    """model for ranked list of VDB entries for a single row input."""

    input_id: str
    response: list[ResultEntry]


class ResultsResponseBody(BaseModel):
    """model for set of ranked lists, for all row entries submmitted."""

    data: list[ResultsList]


class RevClassifaiEntry(BaseModel):
    """Model for a single row of reverse search data (SOC or SIC row etc), includes 'id' and 'code' which are expected as str type."""

    id: str = Field(examples=["1"])
    code: str = Field(examples=["0001"], description="Input code to query vdb for")


class RevClassifaiData(BaseModel):
    """Pydantic object which contains list of many SOC/SIC Reverse Search Entry pydantic models."""

    entries: list[RevClassifaiEntry] = Field(description="array of Rev SOC/SIC Entries to be classified")


class RevResultEntry(BaseModel):
    """Model for single reverse query vdb entry."""

    label: str
    description: str

    class Config:
        extra = Extra.allow  # Allow extra keys (e.g., metadata columns)


class RevResultsList(BaseModel):
    """Model for set of matching entries for reverse search."""

    input_id: str
    response: list[RevResultEntry]


class RevResultsResponseBody(BaseModel):
    """Model for reverse search response."""

    data: list[RevResultsList]


class EmbeddingsList(BaseModel):
    """model for set of embeddings lists, for all row entries submmitted."""

    idx: str
    description: str
    embedding: list


class EmbeddingsResponseBody(BaseModel):
    """model for set of list of embeddings, for all row entries submmitted."""

    data: list[EmbeddingsList]


def convert_dataframe_to_reverse_search_pydantic_response(
    df: pd.DataFrame, meta_data: dict, ids: list[str]
) -> RevResultsResponseBody:
    """Convert a Pandas DataFrame into a JSON object conforming to the RevResultsResponseBody Pydantic model.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing query results.
        meta_data (dict): dictionary of metadata column names mapping to their types.
        ids (list): list of ids (str) to be reverse-searched.

    Returns:
        RevResultsResponseBody: Pydantic model containing the structured response.
    """
    results_list = []

    # Group rows by `query_id`
    for query_id in ids:
        group_df = df[df["query_id"] == query_id]
        if group_df.empty:
            results_list.append(
                RevResultsList(
                    input_id=query_id,
                    response=[],
                )
            )
            continue

        # Convert group_df to a list of dictionaries
        rows_as_dicts = group_df.to_dict(orient="records")

        # Build the list of ResultEntry objects for the current group
        response_entries = []
        for row in rows_as_dicts:
            # Extract metadata columns dynamically
            metadata_values = {meta: row[meta] for meta in meta_data}

            # Create a ResultEntry object
            response_entries.append(
                RevResultEntry(
                    label=row["doc_id"],
                    description=row["doc_text"],
                    **metadata_values,  # Add metadata dynamically
                )
            )

        # Create a ResultsList object for the current query_id
        results_list.append(
            RevResultsList(
                input_id=query_id,
                response=response_entries,
            )
        )

    # Create the ResultsResponseBody object
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
